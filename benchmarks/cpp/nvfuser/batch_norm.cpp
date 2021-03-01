
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include "utils.h"

using namespace torch::jit::fuser::cuda;

static TensorView* setupBatchNorm(
    Fusion* fusion,
    TensorView* input,
    TensorView* weight,
    TensorView* bias,
    const int kNumberOfDims) {
  FusionGuard fg(fusion);

  const float kEps = 1e-5;
  std::vector<int> reduction_axes;
  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  torch::jit::fuser::cuda::Val* num_features = new Double(1);
  for (size_t axis = 0; axis < kNumberOfDims; ++axis) {
    if (axis != 1) {
      reduction_axes.push_back(axis);
      broadcast_mask[axis] = true;
      num_features =
          mul(num_features, input->domain()->domain()[axis]->extent());
    }
  }

  auto x_sum = sum(input, reduction_axes);
  auto x_sum_bcast = broadcast(x_sum, broadcast_mask);
  auto x_mean = div(x_sum_bcast, num_features);

  auto x_mean_sub = sub(input, x_mean);
  auto x_mean_sub_pow = mul(x_mean_sub, x_mean_sub);
  auto var_sum = sum(x_mean_sub_pow, reduction_axes);
  auto var_sum_bcast = broadcast(var_sum, broadcast_mask);
  auto var = div(var_sum_bcast, num_features);

  auto var_eps = add(var, new Double(kEps));
  auto rvar = unaryOp(UnaryOpType::Rsqrt, var_eps);
  auto norm = mul(x_mean_sub, rvar);

  auto weight_bcast = broadcast(weight, broadcast_mask);
  auto bias_bcast = broadcast(bias, broadcast_mask);
  auto norm_gamma = mul(norm, weight_bcast);
  auto norm_gamma_bias = add(norm_gamma, bias_bcast);
  return norm_gamma_bias;
}

bool canDuplicate(const Expr* expr) {
  return expr->outputs().size() == 1 &&
      expr->output(0)->getValType().value() == ValType::TensorView &&
      (expr->getExprType().value() == ExprType::BinaryOp ||
       expr->getExprType().value() == ExprType::UnaryOp ||
       expr->getExprType().value() == ExprType::TernaryOp ||
       expr->getExprType().value() == ExprType::BroadcastOp);
}

bool isConstantAllocation(const TensorView* tv) {
  if (!tv->hasComputeAt()) {
    // We cannot determine allocation size without computeAt structure.
    // Assume Non-Constant Allocation
    return false;
  }

  bool constant_allocation = true;
  auto domain = tv->domain()->domain();
  for (size_t axis = tv->getComputeAtPosition(); axis < domain.size(); ++axis) {
    if (!domain[axis]->isBroadcast() && !domain[axis]->isReduction() &&
        !domain[axis]->isParallelized()) {
      constant_allocation &= domain[axis]->extent()->isConstScalar();
    }
  }
  return constant_allocation;
}

//! Find all TensorViews that require duplication to avoid recompute
//! computeAt error when applying inline ComputeAt
std::vector<TensorView*> findTensorViewsToDuplicate(Fusion& fusion) {
  std::vector<TensorView*> duplicate_tv;
  for (auto expr : fusion.exprs()) {
    if (expr->getExprType().value() != ExprType::ReductionOp) {
      auto out_tvs = ir_utils::filterByType<TensorView>(expr->outputs());
      for (auto out_tv : out_tvs) {
        if (!out_tv->hasReduction() && out_tv->uses().size() > 1 && !fusion.hasOutput(out_tv)) {
          duplicate_tv.push_back(out_tv);
        }
      }
    }
  }
  return duplicate_tv;
}

static void MagicScheduler_BatchNorm_Reduction(benchmark::State& benchmark_state) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{
      32,
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1)};

  // setup fusion
  auto input = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();
  fusion.addInput(input);
  auto weight = TensorViewBuilder().ndims(1).dtype(DataType::Float).build();
  auto bias = TensorViewBuilder().ndims(1).dtype(DataType::Float).build();
  fusion.addInput(weight);
  fusion.addInput(bias);

  const int kNumberOfDims = input_shape.size();
  std::vector<int> reduction_axes;
  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  torch::jit::fuser::cuda::Val* num_features = new Double(1);
  for (size_t axis = 0; axis < kNumberOfDims; ++axis) {
    if (axis != 1) {
      reduction_axes.push_back(axis);
      broadcast_mask[axis] = true;
      num_features =
          mul(num_features, input->domain()->domain()[axis]->extent());
    }
  }

  auto x_sum = sum(input, reduction_axes);
  auto x_sum_bcast = broadcast(x_sum, broadcast_mask);
  auto x_mean = div(x_sum_bcast, num_features);

  auto x_mean_sub = sub(input, x_mean);
  auto x_mean_sub_pow = mul(x_mean_sub, x_mean_sub);
  auto var_sum = sum(x_mean_sub_pow, reduction_axes);
  auto var_sum_bcast = broadcast(var_sum, broadcast_mask);
  auto var = div(var_sum_bcast, num_features);

  const float kEps = 1e-5;
  auto var_eps = add(var, new Double(kEps));
  auto rvar = unaryOp(UnaryOpType::Rsqrt, var_eps);
  auto norm = mul(x_mean_sub, rvar);

  auto weight_bcast = broadcast(weight, broadcast_mask);
  auto bias_bcast = broadcast(bias, broadcast_mask);
  auto norm_gamma = mul(norm, weight_bcast);
  auto norm_gamma_bias = add(norm_gamma, bias_bcast);
  fusion.addOutput(norm_gamma_bias);

  std::vector<TensorView*> reduction_tensors;
  for (auto expr : fusion.exprs()) {
    if (expr->getExprType().value() == ExprType::ReductionOp) {
      auto out_tvs = ir_utils::filterByType<TensorView>(expr->outputs());
      for (auto out_tv : out_tvs) {
        if (out_tv->hasReduction()) {
          reduction_tensors.push_back(out_tv);
        }
      }
    }
  }

  const int kInnerReduction = benchmark_state.range(1) * benchmark_state.range(1);
  const int kUpperThreshold = 4096;
  const int kLowerThreshold = 1024;
  const int kSmallVecSize = (kInnerReduction >= kLowerThreshold) ? 2 : 1;
  const int kVecSize = (kInnerReduction >= kUpperThreshold) ? 4 : kSmallVecSize;
  const int kNumThreads = std::min(kInnerReduction / kVecSize, 1024);

  for (auto expr : fusion.exprs()) {
    auto out_tvs = ir_utils::filterByType<TensorView>(expr->outputs());
    for (auto out_tv : out_tvs) {
      auto outer_reorder = out_tv->reorder({{0, 1}, {1, 0}});

      // merge inner reductions together
      auto inner_merge = outer_reorder->merge(-2);

      // [N, C, H*W] => [N, C, H*W / (TDX * V), TDX * V]
      auto inner_split = inner_merge->split(-1, kNumThreads * kVecSize);

      // [N, C, H*W] => [N, C, H*W / (TDX * V), TDX, V]
      auto vec_split = inner_split->split(-1, kVecSize);
    }
  }

  std::vector<TensorView*> first_rfactor_tvs;
  std::vector<TensorView*> second_rfactor_tvs;
  for (auto reduction_tv : reduction_tensors) {
    // thread reduction first
    first_rfactor_tvs.push_back(reduction_tv->rFactor({-3, -1}));

    // block reduction
    second_rfactor_tvs.push_back(reduction_tv->rFactor({-1}));
  }

  // Initial ComputeAt
  input->computeAt(norm_gamma_bias, 1);
  first_rfactor_tvs[0]->computeAt(reduction_tensors[0], 2);

  auto to_duplicate_tvs = findTensorViewsToDuplicate(fusion);
  auto duplicated_tvs = to_duplicate_tvs[0]->duplicate();

  to_duplicate_tvs[0]->computeAt(norm_gamma_bias, -1);
  to_duplicate_tvs[0]->computeWith(norm_gamma_bias, -1);

  duplicated_tvs[0]->computeAt(var_sum, -1);
  duplicated_tvs[0]->computeWith(var_sum, -1);

  // Create cache-after for vectorization
  auto c1 = input->cache_after(first_rfactor_tvs[0]->definition(), -2);
  auto c2 = input->cache_after(duplicated_tvs[0]->definition(), -2);
  auto c3 = input->cache_after(to_duplicate_tvs[0]->definition(), -2);

  c1->computeAt(first_rfactor_tvs[0], -2);
  c2->computeAt(duplicated_tvs[0], -2);
  c3->computeAt(x_mean_sub, -2);

  auto ca_loop_map_ = ComputeAtMap(ComputeAtMap::MappingMode::LOOP);
  ca_loop_map_.build();

  for (auto rf : first_rfactor_tvs) {
    ca_loop_map_.getConcreteMappedID(rf->axis(0))
        ->parallelize(ParallelType::BIDx);

    ca_loop_map_.getConcreteMappedID(rf->axis(-2))
        ->parallelize(ParallelType::TIDx);
  }

  auto used_vals = DependencyCheck::getAllValsBetween(
      {fusion.inputs().begin(), fusion.inputs().end()}, fusion.outputs());

  auto used_tvs = ir_utils::filterByType<TensorView>(used_vals);

  for (auto tv : used_tvs) {
    for (auto id : tv->domain()->domain()) {
      id->parallelize(ca_loop_map_.getConcreteMappedID(id)->getParallelType());
    }
  }

  // Apply vectorization
  c1->axis(-1)->parallelize(ParallelType::Vectorize);
  c2->axis(-1)->parallelize(ParallelType::Vectorize);
  c3->axis(-1)->parallelize(ParallelType::Vectorize);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones({input_shape[1]}, options);
  at::Tensor at_bias = at::zeros({input_shape[1]}, options);
  std::vector<c10::IValue> inputs({at_x, at_weight, at_bias});

  // outputs
  std::vector<at::Tensor> outputs;

  FusionExecutor executor;
  executor.setMeasureKernelTimeFlag(true);
  executor.compileFusion(&fusion);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    outputs = executor.runFusion(
        c10::ArrayRef<c10::IValue>(inputs));
    benchmark_state.SetIterationTime(executor.kernelTimeMs() / 1000.0);
    cudaDeviceSynchronize();
  }
}

/*
static TensorView* setupBatchNormReduction(
    Fusion* fusion,
    TensorView* input,
    const int kNumberOfDims) {
  FusionGuard fg(fusion);

  std::vector<int> reduction_axes;
  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  torch::jit::fuser::cuda::Val* num_features = new Double(1);
  for (size_t axis = 0; axis < kNumberOfDims; ++axis) {
    if (axis != 1) {
      reduction_axes.push_back(axis);
      broadcast_mask[axis] = true;
      num_features =
          mul(num_features, input->domain()->domain()[axis]->extent());
    }
  }

  auto x_sum = sum(input, reduction_axes);
  auto x_sum_bcast = broadcast(x_sum, broadcast_mask);
  auto x_mean = div(x_sum_bcast, num_features);
  return x_mean;
}

static void MagicScheduler_BatchNorm_Reduction(benchmark::State& benchmark_state) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{
      32,
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1)};

  // setup fusion
  auto input = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();
  fusion.addInput(input);

  const int kNumberOfDims = input_shape.size();
  std::vector<int> reduction_axes;
  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  torch::jit::fuser::cuda::Val* num_features = new Double(1);
  for (size_t axis = 0; axis < kNumberOfDims; ++axis) {
    if (axis != 1) {
      reduction_axes.push_back(axis);
      broadcast_mask[axis] = true;
      num_features =
          mul(num_features, input->domain()->domain()[axis]->extent());
    }
  }

  auto x_sum = sum(input, reduction_axes);
  //auto output =
      //setupBatchNormReduction(&fusion, input, input_shape.size());
  fusion.addOutput(x_sum);

  std::vector<TensorView*> reduction_tensors;
  std::vector<TensorView*> other_tensors;
  analyzeFusion(&fusion, reduction_tensors, other_tensors);

  const int kNumThreads = 1024;

  auto first_reduction_tv = reduction_tensors[0];

  // merge inner reductions together
  auto inner_merge = first_reduction_tv->merge(-2);

  // each thread contributes to the reduction
  auto inner_split = inner_merge->split(-1, kNumThreads);

  // thread reduction first
  auto inner_rfactor1 = inner_split->rFactor({-2});

  // block reduction next
  auto inner_rfactor2 = inner_split->rFactor({-1});

  input->computeAt(x_sum, 2);

  inner_rfactor1->axis(-1)->parallelize(ParallelType::TIDx);
  inner_rfactor2->axis(-1)->parallelize(ParallelType::TIDx);
  x_sum->axis(-1)->parallelize(ParallelType::TIDx);
  x_sum->axis(1)->parallelize(ParallelType::BIDx);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  std::vector<c10::IValue> inputs({at_x});

  // outputs
  std::vector<at::Tensor> outputs;

  FusionExecutor executor;
  executor.setMeasureKernelTimeFlag(true);
  executor.compileFusion(&fusion);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    outputs = executor.runFusion(
        c10::ArrayRef<c10::IValue>(inputs));
    benchmark_state.SetIterationTime(executor.kernelTimeMs() / 1000.0);
    cudaDeviceSynchronize();
  }
}
*/

//------------------------------------------------------------------------------

static void MagicScheduler_BatchNorm(benchmark::State& benchmark_state) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{
      32,
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1)};

  // setup fusion
  auto input = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();
  auto weight = TensorViewBuilder().ndims(1).dtype(DataType::Float).build();
  auto bias = TensorViewBuilder().ndims(1).dtype(DataType::Float).build();
  fusion.addInput(input);
  fusion.addInput(weight);
  fusion.addInput(bias);

  auto output =
      setupBatchNorm(&fusion, input, weight, bias, input_shape.size());
  fusion.addOutput(output);

  std::vector<TensorView*> reduction_tensors;
  std::vector<TensorView*> other_tensors;
  analyzeFusion(&fusion, reduction_tensors, other_tensors);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones({input_shape[1]}, options);
  at::Tensor at_bias = at::zeros({input_shape[1]}, options);
  std::vector<c10::IValue> inputs({at_x, at_weight, at_bias});

  // outputs
  std::vector<at::Tensor> outputs;

  auto reduction_params =
      getNormalizationHeuristics(&fusion, inputs, reduction_tensors);
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");

  scheduleNormalization(
      &fusion, reduction_params.value(), reduction_tensors, other_tensors);

  FusionExecutor executor;
  executor.setMeasureKernelTimeFlag(true);
  executor.compileFusion(&fusion);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    outputs = executor.runFusion(
        c10::ArrayRef<c10::IValue>(inputs), reduction_params.value().lparams);
    benchmark_state.SetIterationTime(executor.kernelTimeMs() / 1000.0);
    cudaDeviceSynchronize();
  }
}

static void MagicScheduler_BatchNorm_Baseline(
    benchmark::State& benchmark_state) {
  const float kMomentum = 0.1;
  const float kEps = 1e-5;
  std::vector<int64_t> input_shape{
      32,
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1)};

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones({input_shape[1]}, options);
  at::Tensor at_bias = at::zeros({input_shape[1]}, options);
  at::Tensor at_mean = at::zeros({input_shape[1]}, options);
  at::Tensor at_var = at::ones({input_shape[1]}, options);

  auto ato_weight = c10::optional<at::Tensor>(at_weight);
  auto ato_bias = c10::optional<at::Tensor>(at_bias);
  auto ato_running_mean = c10::optional<at::Tensor>(at_mean);
  auto ato_running_var = c10::optional<at::Tensor>(at_var);

  cudaDeviceSynchronize();

  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto output = at::batch_norm(
        at_x,
        ato_weight,
        ato_bias,
        ato_running_mean,
        ato_running_var,
        true,
        kMomentum,
        kEps,
        false);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(MagicScheduler_BatchNorm_Reduction)
    ->RangeMultiplier(2)
    ->Ranges({{64, 64}, {8, 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(MagicScheduler_BatchNorm)
    ->RangeMultiplier(2)
    ->Ranges({{64, 64}, {8, 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(MagicScheduler_BatchNorm_Baseline)
    ->RangeMultiplier(2)
    ->Ranges({{64, 64}, {8, 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

/*
BENCHMARK(MagicScheduler_BatchNorm)
    ->RangeMultiplier(2)
    ->Ranges({{64, 512}, {8, 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(MagicScheduler_BatchNorm_Baseline)
    ->RangeMultiplier(2)
    ->Ranges({{64, 512}, {8, 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
*/
