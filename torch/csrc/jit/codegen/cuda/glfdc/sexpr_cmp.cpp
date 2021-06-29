#include <torch/csrc/jit/codegen/cuda/glfdc/sexpr_cmp.h>

std::size_t torch::jit::fuser::cuda::glfdc::SExprHash::operator()(
    const SExprNode& e) const noexcept {
  using namespace detail;

  std::size_t seed = 0;

  hash_combine(seed, e.op_);

  hash_combine(seed, svalue(e.lhs_).first);
  hash_combine(seed, svalue(e.lhs_).second);

  hash_combine(seed, svalue(e.rhs_).first);
  hash_combine(seed, svalue(e.rhs_).second);

  return seed;
}
