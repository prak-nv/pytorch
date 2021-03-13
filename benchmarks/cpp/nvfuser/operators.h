#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <vector>

using namespace torch::jit::fuser::cuda;

static TensorView* setupSoftmax(
    Fusion* fusion,
    TensorView* input,
    const int kNumberOfDims,
    const int kReductionAxis) {
  FusionGuard fg(fusion);

  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  broadcast_mask[kReductionAxis] = true;

  auto max_val = max(input, {kReductionAxis});
  auto bcast_max = broadcast(max_val, broadcast_mask);
  auto x_max_sub = sub(input, bcast_max);
  auto exp = unaryOp(UnaryOpType::Exp, x_max_sub);
  auto sum_exp = sum(exp, {kReductionAxis});
  auto bcast_sum = broadcast(sum_exp, broadcast_mask);
  auto output = div(exp, bcast_sum);
  return output;
}

// Returns mask and output
static std::pair<TensorView*, TensorView*> setupDropout(
    Fusion* fusion,
    TensorView* input,
    double prob) {
  FusionGuard fg(fusion);
  auto rand_vals = unaryOp(UnaryOpType::RandLike, input);
  auto prob_scalar = new Double(prob);
  auto mask = lt(rand_vals, prob_scalar);
  auto apply_mask = mul(input, mask);
  auto scale = div(new Double(1), sub(new Double(1), prob_scalar));
  auto out = mul(apply_mask, scale);
  return std::make_pair(mask, out);
}