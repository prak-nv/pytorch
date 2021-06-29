#include <torch/csrc/jit/codegen/cuda/glfdc/expr.h>

#include <c10/util/Exception.h>

using namespace torch::jit::fuser::cuda::glfdc;

void ExprDAG::markSExprReuse(SExprRef ref) {
  TORCH_INTERNAL_ASSERT(ref.index() < reused_subexpressions_.size());

  bool was_reused = reused_subexpressions_[ref.index()];

  // If subexpressions already marked we can quit recursion, since children
  // are already marked too
  if (was_reused) {
    return;
  }

  ++reuse_count_;

  reused_subexpressions_[ref.index()] = true;
  auto e = this->fetch(ref);

  if (!isValue(e.lhs_))
    markSExprReuse(c10::get<SExprRef>(e.lhs_));

  if (!isValue(e.rhs_))
    markSExprReuse(c10::get<SExprRef>(e.rhs_));
}

SExprRef ExprDAG::addSubExprNode(SExprNode expr) {
  const auto new_idx = abs_offset_t(subexpressions_.size());
  subexpressions_.push_back(expr);
  reused_subexpressions_.push_back(false);
  TORCH_INTERNAL_ASSERT(
      reused_subexpressions_.size() == subexpressions_.size());
  return SExprRef{new_idx};
}

c10::optional<SExprNode> Expr::rootNode() const noexcept {
  const auto* ref = c10::get_if<SExprRef>(&subexpr_);
  if (ref == nullptr) {
    return c10::nullopt;
  }
  return dag_->fetch(*ref);
}

c10::optional<SExprRef> Expr::rootRef() const noexcept {
  const auto* ref = c10::get_if<SExprRef>(&subexpr_);
  if (ref == nullptr) {
    return c10::nullopt;
  }
  return *ref;
}

Operand Expr::operand() const noexcept {
  return subexpr_;
}

// Returns value of
c10::optional<Value> Expr::value() const noexcept {
  auto* val = c10::get_if<Value>(&subexpr_);
  if (val == nullptr)
    return c10::nullopt;
  return *val;
}