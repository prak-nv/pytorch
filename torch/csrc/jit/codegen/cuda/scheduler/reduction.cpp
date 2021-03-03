#include <torch/csrc/jit/codegen/cuda/scheduler/reduction.h>

#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <ATen/cuda/CUDAContext.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
// Largest Power of 2 less-than n
constexpr int64_t lastPow2(int64_t n) {
  TORCH_INTERNAL_ASSERT(n >= 0);
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  n |= (n >> 16); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  n |= (n >> 32); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  return std::max((int64_t)1, n - (n >> 1));
}
} // namespace

ReductionParams reductionHeuristic(
    int64_t num_elems_in_reduction,
    int64_t num_outputs_for_reduction,
    bool fastest_dim_reduction,
    size_t n_tensor_inputs,
    size_t max_input_size) {
  ReductionParams rparams;

  rparams.fastest_dim = fastest_dim_reduction;

  int64_t gdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t gdimy = LaunchParams::UNINITIALIZED_VAL;
  int64_t bdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t bdimy = LaunchParams::UNINITIALIZED_VAL;

  // 1. Basic balancing of unroll factor

  // Set unroll to 128b, don't unroll if we have many inputs
  rparams.loop_unroll = 16 / (int64_t)max_input_size;

  rparams.loop_unroll = ceilDiv(
      rparams.loop_unroll,
      std::max((lastPow2((int64_t)n_tensor_inputs) >> 1), (int64_t)1));

  // Evaluate Dimensions of Reduction TensorView
  TORCH_INTERNAL_ASSERT(
      num_elems_in_reduction > 0 && num_outputs_for_reduction > 0);

  // 2. Initial Definition of Block Dimensions
  // bdimx is our inner dimension
  // bdimy is our outer dimension
  if (rparams.fastest_dim) {
    if (num_elems_in_reduction > 32) {
      rparams.loop_unroll =
          std::min(num_elems_in_reduction / 32, (int64_t)rparams.loop_unroll);
    } else {
      rparams.loop_unroll = 1;
    }

    bdimx = ceilDiv(num_elems_in_reduction, rparams.loop_unroll);
    bdimy = num_outputs_for_reduction;
  } else {
    bdimx = num_outputs_for_reduction;
    bdimy = num_elems_in_reduction;
  }

  // 3. Applying Power of 2 Blocking on thread dimensions
  constexpr int kMaxNumThreads = 512;

  if (bdimx < kMaxNumThreads) {
    bdimx = lastPow2(bdimx);
  } else {
    bdimx = kMaxNumThreads;
  }

  if (bdimy < kMaxNumThreads) {
    bdimy = lastPow2(bdimy);
  } else {
    bdimy = kMaxNumThreads;
  }

  // Don't use more than a warp on inner most dimension
  // Try to enforce bidx * bidy = 512
  // With bidx = min(bidx, warp size)
  // If bidy is small, drive threads back into bidx
  int64_t bdimx_prev = bdimx;
  int64_t device_warp_size = at::cuda::warp_size();
  bdimx = std::min(bdimx, device_warp_size);
  bdimy = std::min(bdimy, kMaxNumThreads / bdimx);
  bdimx = std::min(bdimx_prev, kMaxNumThreads / bdimy);

  // 4. Distributing work across a block
  int64_t red_elems_per_thread = num_elems_in_reduction;
  int64_t outputs_produced_per_block_iter = 1;
  // Set work done by bdimx
  if (rparams.fastest_dim) {
    // Reduction is performed across warp threads (cross-thread reduction)
    red_elems_per_thread = ceilDiv(num_elems_in_reduction, bdimx);
  } else {
    // Warp threads are applied across the output
    outputs_produced_per_block_iter = bdimx;
  }

  // Decide if we need should use bdimy for reduction as well
  constexpr int kMinValuesPerThread = 16;
  constexpr int kMaxValuesPerThread = 256;
  if (!rparams.fastest_dim ||
      // If we used bdimy and still had 16 values per thread, then use bdimy for
      // reduction (as well as bdimy if fastest_dim)
      red_elems_per_thread >= (bdimy * kMinValuesPerThread) ||
      // If we have at least 256 reduction elements, do this anyways.
      red_elems_per_thread >= kMaxValuesPerThread) {
    // Use y dim of the block for the reduce if there are many reduction
    // elements

    red_elems_per_thread = ceilDiv(red_elems_per_thread, bdimy);
    rparams.cross_block = true;
    rparams.multiple_reds_per_blk = false;
  } else {
    // Otherwise do multiple reductions across bdimy
    if (!rparams.fastest_dim) {
      // If fastest dim reduction we always do cross block on bdimx so leave as
      // true
      rparams.cross_block = false;
    }
    rparams.multiple_reds_per_blk = true;
    outputs_produced_per_block_iter *= bdimy;
  }

  // 5. Distributing work across blocks

  // WARNING: Current device for codegen may not be the target device
  int device_max_threads_per_multiprocessor =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor;
  int device_multiprocessor_count =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  // Assuming not limited by register use or shared memory use
  int blocks_per_sm = device_max_threads_per_multiprocessor / (bdimx * bdimy);
  int target_grid_size = device_multiprocessor_count * blocks_per_sm;

  // Setting the number of blocks based on the number of outputs
  gdimx = ceilDiv(num_outputs_for_reduction, outputs_produced_per_block_iter);

  if (rparams.cross_block
      // If we have more reduction elements per thread than we should
      && red_elems_per_thread >= kMaxValuesPerThread
      // And our grid size is small
      && gdimx <= target_grid_size) {
    // Try to do a cross grid reduction
    int blks_per_out_1 = ceilDiv(target_grid_size, gdimx);
    int blks_per_out_2 = ceilDiv(red_elems_per_thread, kMinValuesPerThread);
    int blks_per_out_3 = ceilDiv(red_elems_per_thread, kMaxValuesPerThread);
    int blks_per_output =
        std::max(std::min(blks_per_out_1, blks_per_out_2), blks_per_out_3);

    // If a cross-grid reduction was generated
    if (blks_per_output > 1) {
      rparams.cross_grid = true;
      gdimy = std::max(1, blks_per_output);
    }
  }

  if (rparams.fastest_dim && rparams.multiple_reds_per_blk &&
      gdimx < device_multiprocessor_count) {
    // Not enough parallelization, keep a full warp but push parallelization
    // back into grid from block
    rparams.multiple_reds_per_blk = false;
    auto threads = bdimy * bdimx;
    if (threads > 32) {
      auto grid_desired_factor = device_multiprocessor_count / gdimx;
      auto threads_max_factor = ceilDiv(threads, 32);
      threads_max_factor = std::min(bdimy, threads_max_factor);
      auto factor = threads_max_factor > grid_desired_factor
          ? grid_desired_factor
          : threads_max_factor;

      bdimy = ceilDiv(bdimy, factor);
      gdimx *= factor;
    }
  }

  // Try to set up some unrolling for slow dim case
  if (!rparams.fastest_dim) {
    auto max_unroll = rparams.loop_unroll;
    // Default to no unrolling
    rparams.loop_unroll = 1;
    if (rparams.cross_grid) {
      // We can't unroll the inner dimension of this case as we can't run
      // multiple grid reductions. So try to unroll reduction dimension.

      if (red_elems_per_thread > kMinValuesPerThread) {
        auto available_unroll_red_elems =
            ceilDiv(red_elems_per_thread, kMinValuesPerThread);
        // Don't unroll if we're under a full wave.
        auto available_unroll_grid =
            std::max((gdimx * gdimy) / device_multiprocessor_count, (int64_t)1);
        rparams.loop_unroll = std::min(
            std::min(max_unroll, available_unroll_grid),
            available_unroll_red_elems);
        // When we have very large cases like we do here, we don't want blocks
        // going to far across dram
        red_elems_per_thread =
            ceilDiv(red_elems_per_thread, rparams.loop_unroll);

        gdimy = std::min(red_elems_per_thread, gdimy);
        // Let's bring in the max elems per thread a bit on outer reductions
        if (red_elems_per_thread > 32) {
          gdimy = std::min(ceilDiv(red_elems_per_thread, 32), (int64_t)65535);
        }
      }
    } else {
      if (outputs_produced_per_block_iter > 1) {
        int64_t available_unroll =
            std::max((int64_t)1, outputs_produced_per_block_iter / 2);
        rparams.loop_unroll = std::min(max_unroll, available_unroll);
      }
    }
  }

  const char* debug_env = getenv("PYTORCH_NVFUSER_RED_SCHED_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cout << "\n===== Reduction Parameters ========" << std::endl
              << "Inputs:" << std::endl
              << "\tRed Elems: " << num_elems_in_reduction
              << " Red Outputs: " << num_outputs_for_reduction
              << " Red On Fastest Dim? " << fastest_dim_reduction << std::endl
              << "Reduction Characteristics:" << std::endl
              << "\tMultiple Reds Per Block? " << rparams.multiple_reds_per_blk
              << " Cross Block? " << rparams.cross_block << " Cross Grid? "
              << rparams.cross_grid << std::endl
              << "Recommended Blocking:" << std::endl
              << "\tGridX: " << gdimx << " GridY: " << gdimy
              << " BlckX: " << bdimx << " BlckY: " << bdimy << std::endl
              << " Loop unroll: " << rparams.loop_unroll << std::endl
              << "====================================" << std::endl;
  }

  rparams.lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,
      gdimy,
      LaunchParams::UNINITIALIZED_VAL,
      bdimx,
      bdimy,
      LaunchParams::UNINITIALIZED_VAL);
  return rparams;
}

TORCH_CUDA_CU_API c10::optional<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& fusion_inputs,
    TensorView* red_tv) {
  FUSER_PERF_SCOPE("getReductionHeuristics");

  auto evaluator = executor_utils::bindFusionInputs(fusion_inputs, fusion);

  return getReductionHeuristics(fusion, evaluator, red_tv);
}

TORCH_CUDA_CU_API c10::optional<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    ExpressionEvaluator& evaluator,
    TensorView* red_tv) {
  FUSER_PERF_SCOPE("getReductionHeuristics");

  FusionGuard fg(fusion);

  auto red_root_dom = red_tv->getRootDomain();
  bool fastest_dim_reduction = true;
  for (size_t i = red_root_dom.size(); i > 0; i--) {
    if (red_root_dom[i - 1]->isBroadcast()) {
      continue;
    } else if (red_root_dom[i - 1]->isReduction()) {
      fastest_dim_reduction = true;
      break;
    } else {
      fastest_dim_reduction = false;
      break;
    }
  }

  TORCH_INTERNAL_ASSERT(
      red_tv != nullptr, "Reduction TensorView wasn't found.");

  TORCH_INTERNAL_ASSERT(
      red_tv->hasReduction(), "TensorView doesn't have a reduction.");
  const auto red_expr = red_tv->definition();

  TORCH_INTERNAL_ASSERT(
      red_expr->getExprType() != c10::nullopt &&
          (red_expr->getExprType().value() == ExprType::ReductionOp ||
           red_expr->getExprType().value() == ExprType::WelfordOp),
      "TensorView doesn't have a reduction.");

  int64_t num_outputs_for_reduction = 1;
  int64_t red_elements = 1;

  for (auto id : red_tv->getRootDomain()) {
    auto inferred_val = evaluator.evaluate(id->rawExtent());
    TORCH_INTERNAL_ASSERT(
        inferred_val.has_value(), "Error inferring reduction size.");
    if (id->isReduction()) {
      red_elements *= inferred_val.value();
    } else {
      num_outputs_for_reduction *= inferred_val.value();
    }
  }

  size_t max_dtype_size = 1;
  size_t n_tensor_inputs = 0;
  for (auto inp : fusion->inputs()) {
    if (inp->isA<TensorView>()) {
      max_dtype_size =
          std::max(max_dtype_size, dataTypeSize(inp->getDataType().value()));
      n_tensor_inputs++;
    }
  }

  TORCH_INTERNAL_ASSERT(
      n_tensor_inputs > 0,
      "Tried to schedule a fusion with no tensor inputs, currently not supported.");

  return reductionHeuristic(
      red_elements,
      num_outputs_for_reduction,
      fastest_dim_reduction,
      n_tensor_inputs,
      max_dtype_size);
}

// Reduction schedule types

// Outer Dim, cross block
//      [outputs, |rF-Leftover, rf-Unroll, X-Block]
// Idx:     0     |   1(-3)       2(-2)     3(-1)
//                               |- Reordered -|
//                               V             V
//      [outputs, |rF-Leftover, X-Block, rF-Unroll]
// Idx:     0     |   1(-3)       2(-2)     3(-1)
//      [Out-Leftover, Out-PerBlock, |rF-Leftover, X-Block, rF-Unroll]
// Idx:     0              1         |   2(-3)       3(-2)     4(-1)
//         bidx           tidx                       tidy      unroll

// Outer Dim
//      [Out-Leftover, Out-PerBlock, |rF-Leftover, X-Block]
// Idx:     0               1        |   1(-2)     2(-1)
//         bidx            tidx

// fusion is the input IR that will be modified by this function
void scheduleReduction(
    Fusion* fusion,
    const ReductionParams& rparams,
    TensorView* red_tv,
    std::vector<TensorView*> outs_of_red) {
  FUSER_PERF_SCOPE("scheduleReduction");
  FusionGuard fg(fusion);
  constexpr int kLoopUnrollSplit = 4;

  // If either of these are nullptr at the end of this function don't do
  // anything. Otherwise Transform and parallize entire fusion based on
  // reference_tv and compute at most inlined from reduction_tv to inputs and
  // outputs.
  TensorView* reference_tv = nullptr;
  TensorView* reduction_tv = nullptr;

  // We coalesce all reduction axes to the right;
  scheduler_utils::mergeReduction(red_tv);

  // Merge all iteration dimensions
  if (red_tv->domain()->domain().size() > 1) {
    scheduler_utils::mergeNonReduction(red_tv);
    for (auto iter_tv : outs_of_red) {
      scheduler_utils::mergeNonReduction(iter_tv);
    }
  }

  // Evaluate Dimensions of Reduction TensorView
  auto red_ids = red_tv->domain()->domain();

  TORCH_INTERNAL_ASSERT(
      red_ids.size() == 1 || red_ids.size() == 2,
      "We coalesced all dimensions into 1 or 2 previously.");

  if (red_ids.size() == 1) {
    TORCH_INTERNAL_ASSERT(
        rparams.fastest_dim,
        "If all dims are reduction, so should the fastest dim.");
  }

  // Scheduling the Reduction
  if (rparams.fastest_dim) {
    const bool has_iter_axis = red_ids.size() == 2;
    const int iter_axis = 0;
    const int reduce_axis = red_ids.size() == 2 ? 1 : 0;

    // Do multiple reductions per block
    if (rparams.multiple_reds_per_blk) {
      // Fastest dim, multiple reductions per block
      // Output Dimensions
      // [Out-BIDx, Out-TIDy
      //  0         1
      //
      //  Reduction Dimensions
      //  rF-Remain, rf-Unswitch, rf-Unroll, X-TIDx]
      //  2 (-4)     3 (-3)       4 (-2)     5 (-1)

      red_tv->split(
          reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
      red_tv->split(reduce_axis, rparams.loop_unroll);
      red_tv->split(reduce_axis, 1);

      auto red_tv_rf = scheduler_utils::rfactorHelper(red_tv, {-4, -3, -2});

      red_tv_rf->axis(-1)->parallelize(ParallelType::TIDx);
      red_tv_rf->axis(-3)->parallelize(ParallelType::Unswitch);

      if (has_iter_axis) {
        red_tv_rf->split(
            iter_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
        red_tv_rf->axis(1)->parallelize(ParallelType::TIDy);
        red_tv_rf->axis(0)->parallelize(ParallelType::BIDx);
      }
      if (rparams.loop_unroll == 1) {
        reference_tv = red_tv_rf;
        reduction_tv = red_tv;
      } else {
        // Perform careful unrolling of inputs
        std::vector<TensorView*> cached_inputs;
        {
          auto in_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
          for (auto tv : in_tvs) {
            auto cached_tv = tv->cache_after();
            cached_inputs.emplace_back(cached_tv);
          }
        }

        TransformPropagator::from(red_tv_rf);

        // Inline rfactor into reduction
        red_tv_rf->computeAt(red_tv, -1, ComputeAtMode::MostInlined);

        // Find unswitch position
        int unswitch_axis = -1;
        for (int i = 0; i < red_tv_rf->nDims(); i++) {
          if (red_tv_rf->axis(i)->getParallelType() == ParallelType::Unswitch) {
            unswitch_axis = i;
          }
        }
        if (unswitch_axis != -1) {
          unswitch_axis++;
        }

        // Input to cahced_input we want outside unswitched position
        // Cached input to rfactor we want inlined
        for (auto cached_input : cached_inputs) {
          auto consumers_of_input_cache =
              scheduler_utils::consumerTvsOf(cached_input);
          for (auto consumer : consumers_of_input_cache) {
            if (consumer != red_tv_rf) {
              consumer->computeAt(red_tv_rf, -1, ComputeAtMode::MostInlined);
            }
            cached_input->computeAt(consumer, unswitch_axis);
          }
        }
        scheduler_utils::computeWithOutputs(
            red_tv, -1, ComputeAtMode::MostInlined);

        scheduler_utils::parallelizeAllLike(
            red_tv_rf, scheduler_utils::allTvs(fusion));
      }
      // Do a cross-warp reduction per block
    } else {
      if (rparams.cross_grid) {
        // Fastest dim, cross grid, cross block
        //      [outputs,
        // Idx:     0
        //   | rf-Remain, r-BIDy, r-TIDy, rf-Unswitch, rf-Unroll, r-TIDx]
        //     1(-6)      2(-5)   3(-4)   4(-3)       5(-2)      6(-1)|
        //                Reduction Dimensions
        red_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
        red_tv->split(reduce_axis, rparams.loop_unroll);
        red_tv->split(reduce_axis, 1);
        red_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
        red_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::BIDy));

        auto red_tv_rf = scheduler_utils::rfactorHelper(red_tv, {-6, -3, -2});

        red_tv_rf->axis(-1)->parallelize(ParallelType::TIDx);
        red_tv_rf->axis(-3)->parallelize(ParallelType::Unswitch);
        red_tv_rf->axis(-4)->parallelize(ParallelType::TIDy);
        red_tv_rf->axis(-5)->parallelize(ParallelType::BIDy);

        if (has_iter_axis) {
          red_tv_rf->axis(iter_axis)->parallelize(ParallelType::BIDx);
        }

        if (rparams.loop_unroll == 1) {
          reference_tv = red_tv_rf;
          reduction_tv = red_tv;
        } else {
          // Perform careful unrolling of inputs
          std::vector<TensorView*> cached_inputs;
          {
            auto in_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
            for (auto tv : in_tvs) {
              auto cached_tv = tv->cache_after();
              cached_inputs.emplace_back(cached_tv);
            }
          }

          TransformPropagator::from(red_tv_rf);

          // Inline rfactor into reduction
          red_tv_rf->computeAt(red_tv, -1, ComputeAtMode::MostInlined);

          // Find unswitch position
          int unswitch_axis = -1;
          for (int i = 0; i < red_tv_rf->nDims(); i++) {
            if (red_tv_rf->axis(i)->getParallelType() ==
                ParallelType::Unswitch) {
              unswitch_axis = i;
            }
          }
          if (unswitch_axis != -1) {
            unswitch_axis++;
          }

          // Input to cahced_input we want outside unswitched position
          // Cached input to rfactor we want inlined
          for (auto cached_input : cached_inputs) {
            auto consumers_of_input_cache =
                scheduler_utils::consumerTvsOf(cached_input);
            for (auto consumer : consumers_of_input_cache) {
              if (consumer != red_tv_rf) {
                consumer->computeAt(red_tv_rf, -1, ComputeAtMode::MostInlined);
              }
              cached_input->computeAt(consumer, unswitch_axis);
            }
          }
          scheduler_utils::computeWithOutputs(
              red_tv, -1, ComputeAtMode::MostInlined);

          scheduler_utils::parallelizeAllLike(
              red_tv_rf, scheduler_utils::allTvs(fusion));
        }

      } else {
        // Reduction Splits
        // Output Dimensions
        // [BIDx
        //  0
        //
        // Reduction Dimensions
        // rF-Remain, rf-Unswitch, rf-Unroll, r-TIDy, r-TIDx]
        // 1(-5)      2(-4)        3(-3)      4(-2)   5(-1)

        red_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
        red_tv->split(reduce_axis, rparams.loop_unroll);
        red_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
        red_tv->split(reduce_axis, 1);

        auto red_tv_rf = scheduler_utils::rfactorHelper(red_tv, {-5, -4, -2});

        red_tv_rf->axis(-1)->parallelize(ParallelType::TIDx);
        red_tv_rf->axis(-3)->parallelize(ParallelType::TIDy);
        red_tv_rf->axis(-4)->parallelize(ParallelType::Unswitch);

        if (has_iter_axis) {
          red_tv_rf->axis(iter_axis)->parallelize(ParallelType::BIDx);
        }

        if (rparams.loop_unroll == 1) {
          reference_tv = red_tv_rf;
          reduction_tv = red_tv;
        } else {
          // Perform careful unrolling of inputs
          std::vector<TensorView*> cached_inputs;
          {
            auto in_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
            for (auto tv : in_tvs) {
              auto cached_tv = tv->cache_after();
              cached_inputs.emplace_back(cached_tv);
            }
          }

          TransformPropagator::from(red_tv_rf);

          // Inline rfactor into reduction
          red_tv_rf->computeAt(red_tv, -1, ComputeAtMode::MostInlined);

          // Find unswitch position
          int unswitch_axis = -1;
          for (int i = 0; i < red_tv_rf->nDims(); i++) {
            if (red_tv_rf->axis(i)->getParallelType() ==
                ParallelType::Unswitch) {
              unswitch_axis = i;
            }
          }
          if (unswitch_axis != -1) {
            unswitch_axis++;
          }

          // Input to cahced_input we want outside unswitched position
          // Cached input to rfactor we want inlined
          for (auto cached_input : cached_inputs) {
            auto consumers_of_input_cache =
                scheduler_utils::consumerTvsOf(cached_input);
            for (auto consumer : consumers_of_input_cache) {
              if (consumer != red_tv_rf) {
                consumer->computeAt(red_tv_rf, -1, ComputeAtMode::MostInlined);
              }
              cached_input->computeAt(consumer, unswitch_axis);
            }
          }
          scheduler_utils::computeWithOutputs(
              red_tv, -1, ComputeAtMode::MostInlined);

          scheduler_utils::parallelizeAllLike(
              red_tv_rf, scheduler_utils::allTvs(fusion));
        }
      }
    }
  } else {
    // Slow dim reductions
    if (rparams.cross_block) {
      if (rparams.cross_grid) {
        // Unrolling in this case could apply to the reduction dimension,
        // however not enabled at this moment
        //
        // Reduction Splits
        // Output Dimensions
        // [x-BIDx, x-TIDx,
        //  0         1
        //
        // Reduction Dimensions
        // rF-Leftover, r-BIDy, r-TIDy, rf-Unswitch, rf-Unroll]
        // 2(-5)        3(-4)   4(-3)   5(-2)        6(-1)
        red_tv->split(1, rparams.loop_unroll);
        red_tv->split(1, 1);
        red_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDy));
        red_tv->split(1, NamedScalar::getParallelDim(ParallelType::BIDy));

        red_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));

        auto red_tv_rf = scheduler_utils::rfactorHelper(
            red_tv,
            {-5, -2, -1}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

        red_tv_rf->axis(-2)->parallelize(ParallelType::Unswitch);
        red_tv_rf->axis(-3)->parallelize(ParallelType::TIDy);
        red_tv_rf->axis(-4)->parallelize(ParallelType::BIDy);
        red_tv_rf->axis(1)->parallelize(ParallelType::TIDx);
        red_tv_rf->axis(0)->parallelize(ParallelType::BIDx);
        if (rparams.loop_unroll == 1) {
          reference_tv = red_tv_rf;
          reduction_tv = red_tv;
        } else {
          // Perform careful unrolling of inputs
          std::vector<TensorView*> cached_inputs;
          {
            auto in_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
            for (auto tv : in_tvs) {
              auto cached_tv = tv->cache_after();
              cached_inputs.emplace_back(cached_tv);
            }
          }

          TransformPropagator::from(red_tv_rf);

          // Inline rfactor into reduction
          red_tv_rf->computeAt(red_tv, -1, ComputeAtMode::MostInlined);

          // Find unswitch position
          int unswitch_axis = -1;
          for (int i = 0; i < red_tv_rf->nDims(); i++) {
            if (red_tv_rf->axis(i)->getParallelType() ==
                ParallelType::Unswitch) {
              unswitch_axis = i;
            }
          }
          if (unswitch_axis != -1) {
            unswitch_axis++;
          }

          // Input to cahced_input we want outside unswitched position
          // Cached input to rfactor we want inlined
          for (auto cached_input : cached_inputs) {
            auto consumers_of_input_cache =
                scheduler_utils::consumerTvsOf(cached_input);
            for (auto consumer : consumers_of_input_cache) {
              if (consumer != red_tv_rf) {
                consumer->computeAt(red_tv_rf, -1, ComputeAtMode::MostInlined);
              }
              cached_input->computeAt(consumer, unswitch_axis);
            }
          }
          scheduler_utils::computeWithOutputs(
              red_tv, -1, ComputeAtMode::MostInlined);

          scheduler_utils::parallelizeAllLike(
              red_tv_rf, scheduler_utils::allTvs(fusion));
        }

      } else {
        // Reduction Splits
        // Output Dimensions
        // [x-BIDx, x-Unswitch, x-Unroll, x-TIDx
        //  0       1           2         3
        //
        // Reduction Dimensions
        // rF-Leftover, r-TIDy]
        // 4(-2)        5(-1)
        red_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDy));
        red_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));
        red_tv->split(0, rparams.loop_unroll);
        red_tv->split(0, 1);

        auto red_tv_rf = scheduler_utils::rfactorHelper(
            red_tv, {-2}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

        red_tv_rf->axis(-1)->parallelize(ParallelType::TIDy);
        red_tv_rf->axis(3)->parallelize(ParallelType::TIDx);
        red_tv_rf->axis(1)->parallelize(ParallelType::Unswitch);
        red_tv_rf->axis(0)->parallelize(ParallelType::BIDx);

        red_tv_rf->reorder({{-2, 0}});

        if (rparams.loop_unroll == 1) {
          reference_tv = red_tv_rf;
          reduction_tv = red_tv;
        } else {
          // Perform careful unrolling of inputs
          std::vector<TensorView*> cached_inputs;
          {
            auto in_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
            for (auto tv : in_tvs) {
              auto cached_tv = tv->cache_after();
              cached_inputs.emplace_back(cached_tv);
            }
          }

          TransformPropagator::from(red_tv_rf);
          // Inline rfactor into reduction
          // red_tv_rf->computeAt(red_tv, -1, ComputeAtMode::MostInlined);

          // Find unswitch position
          int unswitch_axis = -1;
          for (int i = 0; i < red_tv_rf->nDims(); i++) {
            if (red_tv_rf->axis(i)->getParallelType() ==
                ParallelType::Unswitch) {
              unswitch_axis = i;
            }
          }
          if (unswitch_axis != -1) {
            unswitch_axis++;
          }

          // Input to cahced_input we want outside unswitched position
          // Cached input to rfactor we want inlined
          for (auto cached_input : cached_inputs) {
            auto consumers_of_input_cache =
                scheduler_utils::consumerTvsOf(cached_input);
            for (auto consumer : consumers_of_input_cache) {
              if (consumer != red_tv_rf) {
                consumer->computeAt(red_tv_rf, -1, ComputeAtMode::MostInlined);
              }
              // TODO: Is this unsafe given broadcasted inputs? Should we get
              // local unswitch positions?
              cached_input->computeAt(consumer, unswitch_axis);
            }
          }

          unswitch_axis = -1;
          for (int i = 0; i < red_tv->nDims(); i++) {
            if (red_tv->axis(i)->getParallelType() == ParallelType::Unswitch) {
              unswitch_axis = i;
            }
          }

          scheduler_utils::computeWithOutputs(
              red_tv, unswitch_axis, ComputeAtMode::MostInlined);

          scheduler_utils::parallelizeAllLike(
              red_tv_rf, scheduler_utils::allTvs(fusion));

          fusion->printKernel();
        }
      }
    } else {
      // Reduction Splits
      // Output Dimensions
      // [x-BIDx, x-TIDx,
      //  0       1
      //
      // Reduction Dimensions
      // r-Leftover]
      // 2(-1)
      red_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));

      red_tv->axis(0)->parallelize(ParallelType::BIDx);
      red_tv->axis(1)->parallelize(ParallelType::TIDx);

      reference_tv = red_tv;
      reduction_tv = red_tv;
    }
  }
  // Propagate strategy

  if (reference_tv != nullptr) {
    TransformPropagator::from(reference_tv);
  }

  if (reduction_tv != nullptr) {
    // Want to inline, especially backwards based on reduction_tv, otherwise
    // rfactor tv may not be inlined correctly
    scheduler_utils::computeAtInputs(
        reduction_tv, -1, ComputeAtMode::MostInlined);
    scheduler_utils::computeWithOutputs(
        reduction_tv, -1, ComputeAtMode::MostInlined);
  }

  if (reference_tv != nullptr) {
    scheduler_utils::parallelizeAllLike(
        reference_tv, scheduler_utils::allTvs(fusion));
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
