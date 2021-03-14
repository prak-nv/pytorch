#include <torch/csrc/jit/codegen/cuda/scheduler/normalization.h>

#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
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

ReductionParams multipleReductionHeuristic(
    int64_t reduction_dim_size,
    int64_t outer_dim_size,
    int64_t inner_dim_size,
    bool fastest_dim_reduction) {
  if (fastest_dim_reduction) {
    TORCH_INTERNAL_ASSERT(reduction_dim_size > 0);
  } else {
    TORCH_INTERNAL_ASSERT(
        reduction_dim_size > 0 && (outer_dim_size > 0 || inner_dim_size > 0));
  }

  int64_t gdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t gdimy = LaunchParams::UNINITIALIZED_VAL;
  int64_t bdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t bdimy = LaunchParams::UNINITIALIZED_VAL;

  ReductionParams rparams;
  rparams.fastest_dim = fastest_dim_reduction;
  rparams.multiple_reds_per_blk = true;
  rparams.cross_block = false;
  rparams.cross_grid = false;

  // Is fastest dimension a reduction dimension?
  if (rparams.fastest_dim) {
    const int64_t kMaxThreadsPerCTA =
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;

    const int64_t kBlockThresholdFastestDim = 1024;
    if (reduction_dim_size <= kMaxThreadsPerCTA) {
      rparams.persistent_kernel = true;

      if (reduction_dim_size <= kBlockThresholdFastestDim) {
        // const int log2_elements = log2_ceil(reduction_dim_size);
        // const int next_power_of_two = 1 << log2_elements;
        // const int kBatchesPerWarp = (next_power_of_two <= 128) ? 2 : 1;
        // rparams.num_warps = 4;

        // TODO: multiple batches per warp causes layer-norm errors
        const int kBatchesPerWarp = 1;
        rparams.batches_per_block = rparams.num_warps * kBatchesPerWarp;
        gdimx = std::max(
            ceilDiv(outer_dim_size, rparams.batches_per_block), (int64_t)1);
        bdimx = at::cuda::warp_size();
      } else {
        // rparams.num_warps = 1;
        // rparams.batches_per_block = 1;
        gdimx = std::max(outer_dim_size, (int64_t)1);
        bdimx = std::min(reduction_dim_size, kMaxThreadsPerCTA);
      }
      // bdimy is the number of warps per block
      bdimy = rparams.num_warps;
      rparams.loop_unroll = ceilDiv(reduction_dim_size, bdimx);
    } else {
      // ILP = sizeof(float4) / sizeof(float)
      const int64_t ILP = 4;
      rparams.loop_unroll = ILP;
      int64_t max_block_size =
          std::min(reduction_dim_size / ILP, kMaxThreadsPerCTA);

      // Combine vectorization while maximizing GPU utilisation
      if (ILP > 1) {
        max_block_size /= 2;
      }

      bdimx = 1;
      while (bdimx < max_block_size) {
        bdimx *= 2;
      }

      // Launch at least a single warp - the kernel assumes that.
      bdimx = std::max(bdimx, (int64_t)at::cuda::warp_size());
      gdimx = std::max(outer_dim_size, (int64_t)1);
    }
  } else {
    rparams.persistent_kernel = false;

    // Warning: Reduce Maximum Threads Per CTA for FP16
    // Register usage exceeds maximum registers per CTA
    // Ampere - 896
    // Volta - 768
    const int64_t kMaxThreadsPerCTA = 512;
    const int64_t kBlockThresholdNotFastestDim = 64;

    // Setup Block Size
    bdimy = std::min(inner_dim_size, kMaxThreadsPerCTA);
    bdimx = 1;
    if (bdimy <= kBlockThresholdNotFastestDim &&
        reduction_dim_size >= kBlockThresholdNotFastestDim) {
      while (bdimy * bdimx <= kMaxThreadsPerCTA &&
             bdimx <= reduction_dim_size) {
        bdimx *= 2;
      }
      bdimx /= 2;
    }
    bdimx = std::max(bdimx, (int64_t)1);

    // Setup Grid Size
    // Estimate maximum number of active blocks
    const int64_t kMaxThreadsPerSM =
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor;
    const int64_t kSMCount =
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int64_t kNumThreads = bdimx * bdimy;
    const int64_t kActiveBlocks = kMaxThreadsPerSM / kNumThreads;
    const int64_t kMaxActiveBlocks = kActiveBlocks * kSMCount;

    // First, tile blocks over the y-axis
    gdimy = std::min(ceilDiv(inner_dim_size, bdimy), kMaxActiveBlocks);
    // Then, fill the x-axis with remaining blocks
    gdimx = std::min(ceilDiv(kMaxActiveBlocks, gdimy), outer_dim_size);
    gdimx = std::max(gdimx, (int64_t)1);
  }

  const char* debug_env = getenv("PYTORCH_NVFUSER_RED_SCHED_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cout << "\n===== Multiple Reduction Parameters ========" << std::endl
              << "Inputs:" << std::endl
              << "\tRed Elems: " << reduction_dim_size
              << " Red Outer: " << outer_dim_size
              << " Red Inner: " << inner_dim_size << " Red On Fastest Dim? "
              << fastest_dim_reduction << std::endl
              << "Reduction Characteristics:" << std::endl
              << "\tMultiple Reds Per Block? " << rparams.multiple_reds_per_blk
              << " Cross Block? " << rparams.cross_block << " Cross Grid? "
              << rparams.cross_grid << std::endl
              << "Recommended Blocking:" << std::endl
              << "\tGridX: " << gdimx << " GridY: " << gdimy << std::endl
              << "\tBlckX: " << bdimx << " BlckY: " << bdimy << std::endl
              << "====================================" << std::endl;
  }

  // Infer BDIMx to avoid conflicts with computeLaunchParams for fastest
  // dimension reduction
  rparams.lparams = LaunchParams(
      gdimx,
      gdimy,
      LaunchParams::UNINITIALIZED_VAL,
      (rparams.fastest_dim && rparams.persistent_kernel)
          ? LaunchParams::UNINITIALIZED_VAL
          : bdimx,
      bdimy,
      LaunchParams::UNINITIALIZED_VAL);
  return rparams;
}

TORCH_CUDA_CU_API c10::optional<ReductionParams> getNormalizationHeuristics(
    Fusion* fusion,
    ExpressionEvaluator& evaluator,
    const std::vector<TensorView*>& reduction_tv) {
  FusionGuard fg(fusion);
  if (!fusion->hasReduction()) {
    return c10::nullopt;
  }

  TORCH_INTERNAL_ASSERT(
      !reduction_tv.empty(),
      "Must at least pass one reduction to normalization scheduler.");

  // Check Reduction Invariants
  for (auto tv : reduction_tv) {
    TORCH_INTERNAL_ASSERT(tv != nullptr, "Reduction TensorView wasn't found.");
    TORCH_INTERNAL_ASSERT(
        tv->hasReduction(), "TensorView doesn't have a reduction.");
    TORCH_INTERNAL_ASSERT(
        tv->definition()->getExprType() != c10::nullopt &&
            tv->definition()->getExprType().value() == ExprType::ReductionOp,
        "TensorView doesn't have a reduction.");
  }

  bool requires_persistence = false;
  bool fits_register_persistence = true;

  auto persistent_buffers = scheduler_utils::persistentBuffers(fusion);

  requires_persistence = !persistent_buffers.buffers.empty();
  if(requires_persistence){
    int64_t persistent_buffer_size = 0;
    for(auto tv : persistent_buffers.buffers){

      int64_t tv_size = 0;
      for(auto id : tv->getMaybeRFactorDomain()){
        if(id->isReduction()){
          continue;
        }
        // Can parallelize these dimensions
        if(persistent_buffers.unmappable_dims.count(id)){
          continue;
        }

        auto id_size = evaluator.evaluate(id->rawExtent());
        TORCH_INTERNAL_ASSERT(
            id_size.has_value(),
            "Cannot generate heuristics if we don't have input information.");

        if (tv_size == 0) {
          tv_size = id_size.value();
        } else {
          tv_size *= id_size.value();
        }
      }
      persistent_buffer_size += tv_size * dataTypeSize(tv->getDataType().value());
    }
    
    constexpr int64_t register_file_size = 256*1024;
    // Don't use more than 75% of register file for persistent buffers
    if(persistent_buffer_size * 4 > register_file_size * 3){
      fits_register_persistence = false;      
    }
  }

  TORCH_INTERNAL_ASSERT(
      (requires_persistence && fits_register_persistence) ||
          !requires_persistence,
      "If requires persistence, must fit persitent.");

  auto first_red_tv = reduction_tv[0];
  auto properties =
      scheduler_utils::getProperties(fusion, evaluator, first_red_tv);

  return multipleReductionHeuristic(
      properties.reduction_numel,
      properties.iter_outside_red,
      properties.iter_inside_red,
      properties.fastest_dim_reduction);
}

TORCH_CUDA_CU_API c10::optional<ReductionParams> getNormalizationHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& fusion_inputs,
    const std::vector<TensorView*>& reduction_tv) {
  FUSER_PERF_SCOPE("scheduleNormalization");

  auto evaluator = executor_utils::bindFusionInputs(fusion_inputs, fusion);

  return getNormalizationHeuristics(fusion, evaluator, reduction_tv);
}

void scheduleNormalization(
    Fusion* fusion,
    const ReductionParams& rparams,
    const std::vector<TensorView*>& reduction_tvs,
    //TODO: Remove!
    std::vector<TensorView*>& other_tvs) {
  FusionGuard fg(fusion);

  auto first_reduction_tv = reduction_tvs.front();
  const size_t kReductionRootDims = first_reduction_tv->getRootDomain().size();

  const auto& in_tv = ir_utils::filterByType<TensorView>(fusion->inputs());
  const auto& out_tv = ir_utils::filterByType<TensorView>(fusion->outputs());

  std::vector<TensorView*> cached_inputs;
  // If we're going to unroll or make a persistent kernel, make a cache of the
  // inputs
  if (rparams.loop_unroll > 1 || rparams.persistent_kernel) {
    auto in_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    for (auto tv : in_tvs) {
      auto cached_tv = tv->cache_after();
      cached_inputs.emplace_back(cached_tv);
    }
  }

  // For intermediate outputs, apply cache_fork
  for (const auto output : fusion->outputs()) {
    if (!output->uses().empty()) {
      if (output->getValType().value() == ValType::TensorView) {
        other_tvs.push_back(output->as<TensorView>()->cache_fork());
      }
    }
  }

  TORCH_INTERNAL_ASSERT(
      !reduction_tvs.empty(),
      "Error in schedule normalization, no reduction tvs provided.");

  TensorView* reference_red_tv = reduction_tvs[0];
  TensorView* reference_rf_tv = nullptr;

  // We coalesce all reduction axes to the right;
  scheduler_utils::mergeReduction(reference_red_tv);

  // Merge all iteration dimensions
  if (reference_red_tv->domain()->domain().size() > 1) {
    scheduler_utils::mergeNonReduction(reference_red_tv);
  }

  // Scheduling the Reduction
  if (rparams.fastest_dim) {
    const bool kHasOuterAxis = reduction_tvs.front()->nDims() > 1;
    if (rparams.persistent_kernel) {
      // Fastest dim persistent
      if (kHasOuterAxis && rparams.batches_per_block > 1 &&
          rparams.num_warps > 1) {
        // Output Splits
        //      [Out-Lft, Out-PerBlock?, Out-NumWarps>|, <Reduction Dims>]
        // Idx: |     0             1             2   |
        //      ---------------------------------------
        //       Output Dimensions
        reference_red_tv->split(0, rparams.batches_per_block);
        reference_red_tv->split(1, rparams.num_warps);
      }

      // Reduction Split
      //      [outer,   |rf-Unroll, rF-Leftover|]
      // Idx:     0     |   (-2)       (-1)    |
      //                ----------------------
      //                Reduction Dimensions
      reference_red_tv->split(-1, rparams.loop_unroll, false);

      TransformPropagator::from(reference_red_tv);

      std::vector<TensorView*> rfactor_tvs;
      for (auto reduction_tv : reduction_tvs) {
        rfactor_tvs.push_back(reduction_tv->rFactor({-2}));
        if (reference_rf_tv == nullptr) {
          reference_rf_tv = rfactor_tvs[0];
        }
      }

      if (kHasOuterAxis) {
        // 4) ComputeAt Structure
        const int kComputeAtAxis = 1;
        for (auto rf_tv : rfactor_tvs) {
          scheduler_utils::computeWithOutputs(rf_tv, kComputeAtAxis);
          scheduler_utils::computeAtInputs(rf_tv, kComputeAtAxis);
        }
      }

      // 6) Parallel Binding
      //      [Out-Lft, Out-PerBlock?, Out-NumWarps>|, rf-Unroll,  rF-Lft]
      // Idx: [   0        1              2         |      3         4   ]
      //      [  BIDx      1             TIDy       |      3        TIDx ]
      //      |-------------------------------------|--------------------]
      //                    Outer                         Reduction
      if (kHasOuterAxis) {
        reference_rf_tv->axis(0)->parallelize(ParallelType::BIDx);
        if (rparams.num_warps > 1) {
          reference_rf_tv->axis(2)->parallelize(ParallelType::TIDy);
        }
      }
      reference_rf_tv->axis(-1)->parallelize(ParallelType::TIDx);
      scheduler_utils::parallelizeAllLike(
          reference_rf_tv, scheduler_utils::allTvs(fusion));
    } else {
      
      // Need to rework non-persistent version
      TORCH_INTERNAL_ASSERT("Not implemented yet.");
      // Fastest dim non persistent
      // Reduction Splits
      //      [ Outer  |, rF-Leftover, rf-Unroll, rf-TDX|]
      // Idx:     0    |     1             2         3  |
      //               ----------------------------------
      //                       Reduction Dimensions
      reference_red_tv->split(-1, rparams.lparams.bdimx());
      reference_red_tv->split(-2, rparams.loop_unroll);

      TransformPropagator::from(reference_red_tv);

      std::vector<TensorView*> rfactor_tvs;
      for (auto reduction_tv : reduction_tvs) {
        auto reduction_tv_rf = reduction_tv->rFactor({-3, -2});
        rfactor_tvs.push_back(reduction_tv_rf);
        if (reference_rf_tv == nullptr) {
          reference_rf_tv = rfactor_tvs[0];
        }
      }

        if (kHasOuterAxis) {
          // 4) ComputeAt Structure
          const int kComputeAtAxis = 1;
          for (auto rf_tv : rfactor_tvs) {
            scheduler_utils::computeWithOutputs(rf_tv, kComputeAtAxis);
            scheduler_utils::computeAtInputs(rf_tv, kComputeAtAxis);
          }
        }

        if (kHasOuterAxis) {
          auto duplicate_tv =
              scheduler_utils::findTensorViewsToDuplicate(fusion, other_tvs);

          // Any TVs with multiple uses and dependencies with same IterDomain
          // Order of Duplication is necessary for correctness
          for (auto tensor : duplicate_tv) {
            auto result = tensor->duplicate();
            other_tvs.insert(other_tvs.end(), result.begin(), result.end());
          }

          auto compute_inline_tv =
              scheduler_utils::findTensorViewsToComputeAtInline(
                  fusion, other_tvs);
          for (auto tensor : compute_inline_tv) {
            auto uses = tensor->uses();
            TORCH_INTERNAL_ASSERT(
                uses.size() == 1,
                "This inline-computeAt TensorView ",
                tensor->name(),
                " is used multiple times.")
            Expr* expr = *uses.begin();
            TensorView* consumer = expr->output(0)->as<TensorView>();
            tensor->computeAt(consumer, -1);
          }
        }

        //      [ outer |, rF-Leftover, rf-Unroll, rf-TDX]
        // Idx: [  BIDx |     1           2         TIDx ]
        //      |-------|--------------------------------]
        //        Outer             Reduction
        // For all TensorViews
        for (auto tv : other_tvs) {
          if (tv->getRootDomain().size() == kReductionRootDims) {
            if (kHasOuterAxis) {
              tv->axis(0)->parallelize(ParallelType::BIDx);
            }
            tv->axis(-1)->parallelize(ParallelType::TIDx);
          }
        }

        // Reduction TensorViews
        for (auto tv : reduction_tvs) {
          if (kHasOuterAxis) {
            tv->axis(0)->parallelize(ParallelType::BIDx);
          }
          tv->axis(-1)->parallelize(ParallelType::TIDx);
        }

        // rFactor TensorViews
        for (auto tv : rfactor_tvs) {
          if (kHasOuterAxis) {
            tv->axis(0)->parallelize(ParallelType::BIDx);
          }
          tv->axis(-1)->parallelize(ParallelType::TIDx);
        }
      } // end non-persistent
      // end fastest_dim logic
    } else {
      // Need to rework into persistent vs non-persistent
      TORCH_INTERNAL_ASSERT("Not implemented yet.");
      const bool outer_axis_exists = reduction_tvs.front()->nDims() > 2;
      const int reduction_axis =
          reduction_tvs.front()->domain()->getReductionAxis().value();
      const int inner_axis = reduction_axis - 1;
      TORCH_INTERNAL_ASSERT(!outer_axis_exists || (inner_axis != 0));

      std::vector<TensorView*> rfactor_tv;
      for (auto tv : reduction_tvs) {
        bool rfactor_axis = false;

        // Reduction Splits - [outer, inner, reduction-Leftover, TDX?]
        if (rparams.lparams.bdimx() > 1) {
          // Reduction Split
          //      [outer, inner, | rF-Leftover, rf-TIDx  ]
          // Idx:     0     1    |   (-2)       (-1)     |
          //                     -------------------------
          //                        Reduction Dimensions
          rfactor_axis = true;
          tv->split(
              reduction_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
        }

        // Inner Splits
        //      [Outer, |Inner-Lft, Inner-BIDy, Inner-TIDy|, <Reduction Dims>]
        // Idx:         |     0        1             2    |
        //              ---------------------------------------
        //                          Inner Dimensions
        tv->split(inner_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
        tv->split(inner_axis, NamedScalar::getParallelDim(ParallelType::BIDy));

        // Outer Splits
        //      [Outer-Leftover, Outer-BIDx |, Inner, <Reduction Dims>]
        // Idx: |     0             1       |
        //      -----------------------------
        //             Outer Dimensions
        if (outer_axis_exists && rparams.lparams.gdimx() > 1) {
          tv->split(0, NamedScalar::getParallelDim(ParallelType::BIDx));
        }

        if (rfactor_axis) {
          auto reduction_tv_rf = tv->rFactor({-2});
          rfactor_tv.push_back(reduction_tv_rf);
        }
      }

      // 2) Other Tensor Splits
      for (auto tv : other_tvs) {
        if (tv->getRootDomain().size() == kReductionRootDims) {
          // Reduction Splits - [outer, inner, reduction-Leftover, TDX?]
          if (rparams.lparams.bdimx() > 1) {
            tv->split(
                reduction_axis,
                NamedScalar::getParallelDim(ParallelType::TIDx));
          }

          // Inner Splits - [outer, inner-Leftover, BDY, TDY, reduction]
          tv->split(
              inner_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
          tv->split(
              inner_axis, NamedScalar::getParallelDim(ParallelType::BIDy));

          // Outer Splits
          // [outer-Leftover, BDX?, inner-Leftover, BDY, TDY, reduction]
          if (outer_axis_exists && rparams.lparams.gdimx() > 1) {
            tv->split(0, NamedScalar::getParallelDim(ParallelType::BIDx));
          }
        }
      }

      int kBIDyAxis = -1;
      if (outer_axis_exists) {
        if (rparams.lparams.gdimx() > 1) {
          kBIDyAxis = 3;
        } else {
          kBIDyAxis = 2;
        }
      } else {
        kBIDyAxis = 1;
      }
      TORCH_INTERNAL_ASSERT(kBIDyAxis > 0);
      const int kTIDyAxis = kBIDyAxis + 1;

      // 3) ComputeAt structure
      // [outer-lft, BDX?, inner-lft, BDY, TDY, reduction-lft, TDX?]
      const size_t kComputeAtAxis = kTIDyAxis + 1;
      for (auto output : out_tv) {
        auto inputs_for_output = fusion->inputsOf(output);
        for (auto input : in_tv) {
          if (inputs_for_output.find(input) != inputs_for_output.end()) {
            input->computeAt(output, kComputeAtAxis);
          }
        }
      }

      // 4) Find TensorViews to duplicate and computeAt inline
      auto duplicate_tv =
          scheduler_utils::findTensorViewsToDuplicate(fusion, other_tvs);

      // Any TVs with multiple uses and dependencies with same IterDomain
      // Order of Duplication is necessary for correctness
      for (auto tensor : duplicate_tv) {
        auto result = tensor->duplicate();
        // Add duplicated TVs to Other TVs
        other_tvs.insert(other_tvs.end(), result.begin(), result.end());
      }

      // 5) Handle Inline-ComputeAt
      auto compute_inline_tv =
          scheduler_utils::findTensorViewsToComputeAtInline(fusion, other_tvs);
      for (auto tensor : compute_inline_tv) {
        auto uses = tensor->uses();
        TORCH_INTERNAL_ASSERT(
            uses.size() == 1,
            "This inline-computeAt TensorView ",
            tensor->name(),
            " is used multiple times.")
        Expr* expr = *uses.begin();
        TensorView* consumer = expr->output(0)->as<TensorView>();
        tensor->computeAt(consumer, -1);
      }

      // 6) Parallel Bindings
      for (auto tv : other_tvs) {
        if (tv->getRootDomain().size() == kReductionRootDims) {
          if (outer_axis_exists && rparams.lparams.gdimx() > 1) {
            tv->axis(1)->parallelize(ParallelType::BIDx);
          }

          tv->axis(kBIDyAxis)->parallelize(ParallelType::BIDy);
          tv->axis(kTIDyAxis)->parallelize(ParallelType::TIDy);

          if (tv->nDims() > kComputeAtAxis && rparams.lparams.bdimx() > 1) {
            tv->axis(-1)->parallelize(ParallelType::TIDx);
          }
        }
      }

      for (auto tv : reduction_tvs) {
        if (outer_axis_exists && rparams.lparams.gdimx() > 1) {
          tv->axis(1)->parallelize(ParallelType::BIDx);
        }

        tv->axis(kBIDyAxis)->parallelize(ParallelType::BIDy);
        tv->axis(kTIDyAxis)->parallelize(ParallelType::TIDy);

        if (tv->nDims() > kComputeAtAxis && rparams.lparams.bdimx() > 1) {
          tv->axis(-1)->parallelize(ParallelType::TIDx);
        }
      }

      for (auto tv : rfactor_tv) {
        if (outer_axis_exists && rparams.lparams.gdimx() > 1) {
          tv->axis(1)->parallelize(ParallelType::BIDx);
        }

        tv->axis(kBIDyAxis)->parallelize(ParallelType::BIDy);
        tv->axis(kTIDyAxis)->parallelize(ParallelType::TIDy);

        if (tv->nDims() > kComputeAtAxis && rparams.lparams.bdimx() > 1) {
          tv->axis(-1)->parallelize(ParallelType::TIDx);
        }
      }
    } // end non_fastest_dim logic

    // If castOp then Broadcast, inline computeAt castOp with BroadcastOp
    for (const auto input : in_tv) {
      if (input->getRootDomain().size() != kReductionRootDims) {
        scheduler_utils::handleCastBroadcastInput(fusion, input);
      }
    }
  }

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
