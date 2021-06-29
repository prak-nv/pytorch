#include <torch/csrc/jit/codegen/cuda/glfdc/expr_builder.h>

#include <torch/csrc/jit/codegen/cuda/glfdc/cfold.h>

#include <cstring>
#include <tuple>
#include <type_traits>

using namespace torch::jit::fuser::cuda::glfdc;

namespace {

std::pair<Operand, Operand> reorderCommutative(Operand l, Operand r) {
  using torch::jit::fuser::cuda::glfdc::detail::svalue;
  // We reorder svalues (deterministic value for subexpression node)
  // to ensure that in commutative expressions like "A + B" and "B + A"
  // we always obtain the same form. This is to slightly aid Common
  // Subexpression Evaluation.
  if (svalue(l) < svalue(r))
    std::swap(l, r);

  return std::make_pair(l, r);
}

} // namespace

ExpressionBuilder::ExpressionBuilder() : dag_(new ExprDAG) {}

Value ExpressionBuilder::getBinding(uintptr_t unbound) {
  TORCH_INTERNAL_ASSERT(dag_ != nullptr);
  auto it = dag_->binding_lookup_.find(unbound);

  if (it != dag_->binding_lookup_.end()) {
    TORCH_INTERNAL_ASSERT(it->second < dag_->binding_keys_.size());
    return SymbolicValue{it->second};
  }
  // Unless already found create binding
  return createNewBinding(unbound);
}

Value ExpressionBuilder::createNewBinding(uintptr_t unbound) {
  TORCH_INTERNAL_ASSERT(dag_ != nullptr);
  TORCH_INTERNAL_ASSERT(
      dag_->binding_lookup_.find(unbound) == dag_->binding_lookup_.end());

  const auto next_slot = abs_offset_t(dag_->binding_keys_.size());
  dag_->binding_lookup_.insert(std::make_pair(unbound, next_slot));
  dag_->binding_keys_.push_back(unbound);

  return SymbolicValue{next_slot};
}

Operand ExpressionBuilder::createSExpr_(OperatorKind op, Operand l, Operand r) {
  TORCH_INTERNAL_ASSERT(dag_ != nullptr);

  auto is_commutative = [](OperatorKind op) -> bool {
    return (op == OperatorKind::add) || (op == OperatorKind::mul);
  };

  if (is_commutative(op))
    std::tie(l, r) = reorderCommutative(l, r);

  // Possibly constant-fold
  auto opt_cfold = cfold(op, l, r);

  if (opt_cfold.has_value())
    return opt_cfold.value();

  SExprNode e = {l, r, op};

  auto it = seen_exprs_.find(e);

  if (it != seen_exprs_.end()) {
    // We've seen it already
    dag_->markSExprReuse(it->second);
    return it->second;
  }

  // Expr unseen - allocate new
  auto ref = dag_->addSubExprNode(e);

  if (!isValue(l))
    dag_->markSExprReuse(c10::get<SExprRef>(l));
  if (!isValue(r))
    dag_->markSExprReuse(c10::get<SExprRef>(r));

  seen_exprs_.insert(std::make_pair(e, ref));

  return ref;
}
