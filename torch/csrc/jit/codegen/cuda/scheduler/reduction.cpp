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

ReductionParams innerReductionHeuristic(
    const int64_t num_elems_in_reduction,
    const int64_t num_outputs_for_reduction,
    const size_t n_tensor_inputs,
    const size_t max_input_size) {
  // Set some targets for parallelization

  const int64_t n_elems = num_elems_in_reduction * num_outputs_for_reduction;
  const int64_t l2_cache_size =
      at::cuda::getCurrentDeviceProperties()->l2CacheSize * 4;

  const int64_t warp_size =
      n_elems * max_input_size * n_tensor_inputs < l2_cache_size
      ? (int64_t)32 / max_input_size
      : 32;

  // WARNING: Current device for codegen may not be the target device
  const int64_t device_max_threads_per_multiprocessor =
      (int64_t)at::cuda::getCurrentDeviceProperties()
          ->maxThreadsPerMultiProcessor;

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  auto const max_unroll = ceilDiv(
      // Available unrolling based on size of data type
      (int64_t)16 / (int64_t)max_input_size,
      // Reduce unrolling if we have many inputs, start reduction at 2 inputs
      std::max((lastPow2((int64_t)n_tensor_inputs) >> 1), (int64_t)1));

  int64_t target_blocks = 1;
  int64_t target_unroll = 1;
  int64_t target_threads = warp_size;

  // If we have one warp per block, how many blocks would that be?
  target_blocks = ceilDiv(n_elems, warp_size);

  // If we have more than a wave, put parallelism into unrolling
  if (target_blocks > device_multiprocessor_count) {
    target_unroll = std::min(
        max_unroll, ceilDiv(target_blocks, device_multiprocessor_count));
    target_blocks = ceilDiv(n_elems, warp_size * target_unroll);
  }

  // Cap target blocks to 4 waves
  target_blocks = std::min(target_blocks, device_multiprocessor_count * 4);

  if (target_blocks * target_unroll * target_threads < n_elems) {
    // targetting 4 waves, so try to use a quarter of available threads
    target_threads = std::min(
        ceilDiv(n_elems, target_blocks * target_unroll),
        ceilDiv(device_max_threads_per_multiprocessor, (int64_t)4));
  }

  // To get to target threads:
  // Prioritize
  // (1) x dim in reduction
  // (2) unrolling in reduction
  // (3) y in output
  // To get target blocks:
  // Prioritize
  // (1) x dim in multiple outputs
  // (2) y dim in multiple reductions

  // TODO: Flip block y and x
  // Blocks for reductions
  int64_t gdimy = 1;
  // Blocks for outputs
  int64_t gdimx = 1;

  // Threads for outputs
  int64_t bdimy = 1;
  // Threads for reduction
  int64_t bdimx = 1;

  // Should we unroll from reduction axis, or outs axis
  bool unroll_reduction = true;

  bool multiple_reductions_per_block = false;

  // Unroll amount
  int64_t unroll_factor = 1;

  // Grab what we can out of reduction domain, but don't go over a warp size yet
  bdimx = std::min(num_elems_in_reduction, (int64_t)warp_size);
  // Put everything else in bdimy for now
  bdimy = target_threads / bdimx > 0 ? target_threads / bdimx : 1;

  int64_t remainder_in_reduction = ceilDiv(num_elems_in_reduction, bdimx);
  int64_t remainder_in_output = ceilDiv(num_outputs_for_reduction, bdimy);

  // Adjust blocking and setup unrolling
  if (remainder_in_reduction == 1) {
    // Small number of reduction elements, don't try to unroll the reduction dim
    unroll_reduction = false;
    // Try unrolling output dimension
    unroll_factor = std::min(target_unroll, remainder_in_output);
    remainder_in_output =
        ceilDiv(num_outputs_for_reduction, unroll_factor * bdimy);
    gdimx = std::min(remainder_in_output, target_blocks);
    remainder_in_output =
        ceilDiv(num_outputs_for_reduction, unroll_factor * bdimy * gdimx);
  } else {
    // If we have reduction elements left, re-adjust the block dims
    bdimx = std::min(num_elems_in_reduction, target_threads);
    remainder_in_reduction = ceilDiv(num_elems_in_reduction, bdimx);

    bdimy = target_threads / bdimx > 1 ? target_threads / bdimx : 1;
    remainder_in_output = ceilDiv(num_outputs_for_reduction, bdimy);

    unroll_factor = std::min(remainder_in_reduction, target_unroll);
    if (unroll_factor == 1) {
      // If we can't unroll reduction dim, unroll output dim
      unroll_reduction = false;
      unroll_factor = std::min(remainder_in_output, target_unroll);
      remainder_in_output =
          ceilDiv(num_outputs_for_reduction, bdimy * unroll_factor);
    } else {
      remainder_in_reduction =
          ceilDiv(num_elems_in_reduction, bdimx * unroll_factor);
    }
    gdimx = remainder_in_output;
  }

  // Cross grid reduction if we haven't hit our target blocks, and we have many
  // reduction elements.
  if (gdimx < target_blocks && remainder_in_reduction > 4) {
    gdimy =
        std::min(ceilDiv(remainder_in_reduction, (int64_t)4), (int64_t)65535);
    remainder_in_reduction = ceilDiv(
        num_elems_in_reduction,
        bdimx * (unroll_reduction ? unroll_factor : 1) * gdimy);
  } else if (remainder_in_reduction > 32) {
    gdimy =
        std::min(ceilDiv(remainder_in_reduction, (int64_t)32), (int64_t)65535);
    remainder_in_reduction = ceilDiv(remainder_in_reduction, gdimy);
  }

  // Try to do some cleanup of ragged waves on device
  // gdimx is a remainder of a split, so can only control bdimy
  if (
      // If we have less than 8 waves of blocks
      gdimy * gdimx < device_multiprocessor_count * 8 &&
      // And we don't have an even divisible number of blocks
      (gdimy * gdimx) % device_multiprocessor_count != 0 &&
      // And we have more than one wave
      gdimy * gdimx > device_multiprocessor_count) {
    // round waves down
    auto waves =
        std::max((gdimx * gdimy) / device_multiprocessor_count, (int64_t)1);
    auto new_gdimy =
        std::max((waves * device_multiprocessor_count) / gdimx, (int64_t)1);
    if (
        // If difference is less than 25% of the original gdimy
        (new_gdimy - gdimy) * 4 < gdimy &&
        // and difference is less than 25% of the original number of blocks
        ((new_gdimy * gdimx) - (gdimy * gdimx)) * 4 < gdimy * gdimx) {
      gdimy = new_gdimy;
    }
  }

  ReductionParams rparams;
  rparams.fastest_dim = true;
  rparams.cross_block = true;
  rparams.cross_grid = gdimy > 1;
  rparams.multiple_reds_per_blk = bdimy > 1;
  rparams.loop_unroll = unroll_factor;
  rparams.reduction_unroll = unroll_reduction;

  rparams.lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,
      gdimy,
      LaunchParams::UNINITIALIZED_VAL,
      bdimx,
      bdimy,
      LaunchParams::UNINITIALIZED_VAL);

  const char* debug_env = getenv("PYTORCH_NVFUSER_RED_SCHED_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cout << "\n===== Reduction Parameters ========" << std::endl
              << "Inputs:" << std::endl
              << "\tRed Elems: " << num_elems_in_reduction
              << " Red Outputs: " << num_outputs_for_reduction
              << " Red On Fastest Dim " << std::endl
              << "Reduction Characteristics:" << std::endl
              << "\tMultiple Reds Per Block? " << rparams.multiple_reds_per_blk
              << " Cross Block? " << rparams.cross_block << " Cross Grid? "
              << rparams.cross_grid << std::endl
              << "Recommended Blocking:" << std::endl
              << "\tGridX: " << gdimx << " GridY: " << gdimy
              << " BlckX: " << bdimx << " BlckY: " << bdimy << std::endl
              << " Loop unroll: " << rparams.loop_unroll << std::endl
              << " Unrol reduction dim: " << rparams.reduction_unroll
              << std::endl
              << "====================================" << std::endl;
  }

  return rparams;
}

ReductionParams OuterReductionHeuristic(
    const int64_t num_elems_in_reduction,
    const int64_t num_outputs_for_reduction,
    const size_t n_tensor_inputs,
    const size_t max_input_size) {
  // Set some targets for parallelization

  const int64_t n_elems = num_elems_in_reduction * num_outputs_for_reduction;
  const int64_t l2_cache_size =
      at::cuda::getCurrentDeviceProperties()->l2CacheSize * 4;

  const int64_t warp_size =
      n_elems * max_input_size * n_tensor_inputs < l2_cache_size
      ? (int64_t)32 / max_input_size
      : 32;

  int64_t target_blocks = 1;
  int64_t target_unroll = 1;
  int64_t target_threads = warp_size;

  // WARNING: Current device for codegen may not be the target device
  const int64_t device_max_threads_per_multiprocessor =
      (int64_t)at::cuda::getCurrentDeviceProperties()
          ->maxThreadsPerMultiProcessor;

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  auto const max_unroll = ceilDiv(
      // Available unrolling based on size of data type
      (int64_t)16 / (int64_t)max_input_size,
      // Reduce unrolling if we have many inputs, start reduction at 2 inputs
      std::max((lastPow2((int64_t)n_tensor_inputs) >> 1), (int64_t)1));

  // If we have one warp per block, how many blocks would that be?
  target_blocks = ceilDiv(n_elems, (int64_t)warp_size);

  // If we have more than a wave, put parallelism into unrolling
  if (target_blocks > device_multiprocessor_count) {
    target_unroll = std::min(
        max_unroll, ceilDiv(target_blocks, device_multiprocessor_count));
    target_blocks = ceilDiv(target_blocks, target_unroll);
  }

  // Cap target blocks to 4 waves
  target_blocks = std::min(target_blocks, device_multiprocessor_count * 4);

  if (target_blocks * target_unroll * target_threads < n_elems) {
    // targetting 4 waves, so try to use a quarter of available threads
    target_threads = std::min(
        ceilDiv(n_elems, target_blocks * target_unroll),
        ceilDiv(device_max_threads_per_multiprocessor, (int64_t)4));
  }

  // To get to target threads:
  // Prioritize
  // (1) x dim in iter domain
  // (2) unrolling in iter domain
  // (3) y in reduction domain
  // To get target blocks:
  // Prioritize
  // (1) x dim in multiple outputs
  // (2) y dim in multiple reductions - need to flip unrolling to reduction
  // domain for this

  // Blocks for reductions
  int64_t gdimy = 1;
  // Blocks for outputs
  int64_t gdimx = 1;

  // Threads for reduction
  int64_t bdimy = 1;
  // Threads for output
  int64_t bdimx = 1;

  // Should we unroll from reduction axis, or outs axis
  bool unroll_reduction = false;

  bool multiple_reductions_per_block = true;

  // Unroll amount
  int64_t unroll_factor = 1;

  int64_t remainder_in_reduction = num_elems_in_reduction;
  int64_t remainder_in_output = num_outputs_for_reduction;

  if (ceilDiv(num_outputs_for_reduction, warp_size) <
      device_multiprocessor_count) {
    // If we can't hit a full wave, leave bdimx as warp_size, and prioritize
    // bdimy
    bdimx = std::min(lastPow2(num_outputs_for_reduction), (int64_t)warp_size);
  } else {
    bdimx = std::min(
        target_threads, ceilDiv(num_outputs_for_reduction, target_blocks));
    bdimx = std::max(bdimx, (int64_t)warp_size);
  }
  bdimy = ceilDiv(target_threads, bdimx);

  remainder_in_output = ceilDiv(num_outputs_for_reduction, bdimx);
  remainder_in_reduction = ceilDiv(remainder_in_reduction, bdimy);

  if (num_outputs_for_reduction >=
      device_multiprocessor_count * target_threads) {
    // If we easily saturate the GPU, don't use block dim y and unroll output
    // dimension, this could be a more gentle transition starting earlier
    bdimx = target_threads;
    remainder_in_output = ceilDiv(num_outputs_for_reduction, bdimx);

    bdimy = 1;
    remainder_in_reduction = num_elems_in_reduction;

    // Assume unroll in output, switch to remainder if cross grid
    // Don't unroll if we don't have 2 full waves
    unroll_factor = std::min(
        ceilDiv(remainder_in_output, device_multiprocessor_count * 2),
        target_unroll);
    if (unroll_factor == 1) {
      unroll_reduction = true;
      unroll_factor = 1;
      // TODO: Add to inner case
      if (remainder_in_reduction > 1) {
        unroll_factor = std::min(remainder_in_reduction, unroll_factor);
        remainder_in_reduction = ceilDiv(remainder_in_reduction, unroll_factor);
      }
    } else {
      remainder_in_output = ceilDiv(remainder_in_output, unroll_factor);
      unroll_reduction = false;
    }
  }

  gdimx = remainder_in_output;

  // Expand in bdimy, unroll in the reduction dimension, and potentially reduce
  // cross grid in gdimy.
  if ((gdimx < device_multiprocessor_count && remainder_in_reduction > 1) ||
      // If we haven't hit a full wave with gdimx, and we have reduction
      // elements left
      remainder_in_reduction > 256
      // Or we just have a lot of reduction elements
  ) {
    // Reset unrolling
    unroll_factor = 1;
    unroll_reduction = true;
    // Reset remainder in reduction (since unroll may have changed)
    remainder_in_reduction = ceilDiv(num_elems_in_reduction, bdimy);
    // Reset remainder in out (since unroll may have changed)
    remainder_in_output = ceilDiv(num_outputs_for_reduction, bdimx);
    gdimx = remainder_in_output;
    remainder_in_output = ceilDiv(remainder_in_output, gdimx);

    // Expand the bdimy dimension before we consider going cross grid upto a
    // quarter of the SM capacity times max unroll
    auto max_threads =
        ceilDiv(device_max_threads_per_multiprocessor, (int64_t)4);
    if (bdimy * bdimx < max_threads) {
      bdimy = std::max(max_threads / bdimx, (int64_t)1);
      remainder_in_reduction = ceilDiv(num_elems_in_reduction, bdimy);
      if (remainder_in_reduction > 1) {
        unroll_factor = std::min(max_unroll, remainder_in_reduction);
        remainder_in_reduction =
            ceilDiv(num_elems_in_reduction, bdimy * unroll_factor);
      }
    }

    gdimy = remainder_in_reduction;
    if (gdimy > 32) {
      // Don't do too many cross grid reductions if we have many of them
      // available
      gdimy = std::min(
          ceilDiv(remainder_in_reduction, (int64_t)32), (int64_t)65535);
      // We don't want to have to do too many reductions after we go cross grid
      gdimy = std::min(gdimy, bdimx * bdimy * 2);
    }
  }

  // bdimx should never be > max threads, but do this anyways for safety
  bdimx = std::min(device_max_threads_per_multiprocessor, bdimx);
  if (bdimy * bdimx > device_max_threads_per_multiprocessor) {
    bdimy = std::max(device_max_threads_per_multiprocessor / bdimx, (int64_t)1);
  }

  // Try to do some cleanup of ragged waves on device
  if (
      // If we have less than 8 waves of blocks
      gdimy * gdimx < device_multiprocessor_count * 8 &&
      // And we don't have an even divisible number of blocks
      (gdimy * gdimx) % device_multiprocessor_count != 0 &&
      // And we have more than one wave
      gdimy * gdimx > device_multiprocessor_count) {
    // round waves down
    auto waves =
        std::max((gdimx * gdimy) / device_multiprocessor_count, (int64_t)1);
    auto new_gdimy =
        std::max((waves * device_multiprocessor_count) / gdimx, (int64_t)1);
    if (
        // If difference is less than 25% of the original gdimy
        (new_gdimy - gdimy) * 4 < gdimy &&
        // and difference is less than 25% of the original number of blocks
        ((new_gdimy * gdimx) - (gdimy * gdimx)) * 4 < gdimy * gdimx) {
      gdimy = new_gdimy;
    }
  }

  ReductionParams rparams;
  rparams.fastest_dim = false;
  // cross grid implies cross block
  rparams.cross_block = bdimy > 1 || gdimy > 1;
  rparams.cross_grid = gdimy > 1;
  rparams.multiple_reds_per_blk = bdimx > 1;
  rparams.loop_unroll = unroll_factor;
  rparams.reduction_unroll = unroll_reduction;

  const char* debug_env = getenv("PYTORCH_NVFUSER_RED_SCHED_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cout << "\n===== Reduction Parameters ========" << std::endl
              << "Inputs:" << std::endl
              << "\tRed Elems: " << num_elems_in_reduction
              << " Red Outputs: " << num_outputs_for_reduction
              << " Red On Outer Dim " << std::endl
              << "Reduction Characteristics:" << std::endl
              << "\tMultiple Reds Per Block? " << rparams.multiple_reds_per_blk
              << " Cross Block? " << rparams.cross_block << " Cross Grid? "
              << rparams.cross_grid << std::endl
              << "Recommended Blocking:" << std::endl
              << "\tGridX: " << gdimx << " GridY: " << gdimy
              << " BlckX: " << bdimx << " BlckY: " << bdimy << std::endl
              << " Loop unroll: " << rparams.loop_unroll << std::endl
              << " Unrol reduction dim: " << rparams.reduction_unroll
              << std::endl
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

} // namespace

ReductionParams reductionHeuristic(
    int64_t num_elems_in_reduction,
    int64_t num_outputs_for_reduction,
    bool fastest_dim_reduction,
    size_t n_tensor_inputs,
    size_t max_input_size) {
  if (fastest_dim_reduction) {
    return innerReductionHeuristic(
        num_elems_in_reduction,
        num_outputs_for_reduction,
        n_tensor_inputs,
        max_input_size);
  } else {
    return OuterReductionHeuristic(
        num_elems_in_reduction,
        num_outputs_for_reduction,
        n_tensor_inputs,
        max_input_size);
  }
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
        //   | rf-Remain, r-BIDy, r-TIDy, r-Unswitch, rf-Unroll, r-TIDx]
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
        // TODO: Evaluate
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
    if (rparams.cross_block) {
      if (rparams.cross_grid) {
        // Unrolling in this case can only be applied to the reduction dimension
        // since currently, grid reductions cannot be called multiple times
        //
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
        if (rparams.reduction_unroll || rparams.loop_unroll == 1) {
          // Reduction Splits
          // Output Dimensions
          // [x-BIDx, x-TIDx
          //  0       1
          //
          // Reduction Dimensions
          // rF-Leftover, r-TIDy, rf-Unswitch, rf-Unroll]
          // 2(-4)        3(-3)   4(-2)       5(-1)
          red_tv->split(1, rparams.loop_unroll);
          red_tv->split(1, 1);
          red_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDy));
          red_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));

          auto red_tv_rf = scheduler_utils::rfactorHelper(
              red_tv,
              {-4, -2, -1}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

          red_tv_rf->axis(-2)->parallelize(ParallelType::Unswitch);
          red_tv_rf->axis(-3)->parallelize(ParallelType::TIDy);
          red_tv_rf->axis(1)->parallelize(ParallelType::TIDx);
          red_tv_rf->axis(0)->parallelize(ParallelType::BIDx);

          if (rparams.loop_unroll == 1) {
            reference_tv = red_tv_rf;
            reduction_tv = red_tv;
          } else {
            // Perform careful unrolling of inputs
            std::vector<TensorView*> cached_inputs;
            {
              auto in_tvs =
                  ir_utils::filterByType<TensorView>(fusion->inputs());
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
                  consumer->computeAt(
                      red_tv_rf, -1, ComputeAtMode::MostInlined);
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

          {
            // If we leave unswitch on we get a predicate around block reduce
            // which produces incorrect values.
            auto vals_post_reduction = DependencyCheck::getAllUseChains(red_tv);
            for (auto chain : vals_post_reduction) {
              auto tvs_post_reduction =
                  ir_utils::filterByType<TensorView>(chain);
              for (auto tv : tvs_post_reduction) {
                for (auto id : tv->domain()->domain()) {
                  if (id->getParallelType() == ParallelType::Unswitch) {
                    id->parallelize(ParallelType::Serial);
                  }
                }
              }
            }
          }
          // Finished scheduling with iter domain unroll
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
