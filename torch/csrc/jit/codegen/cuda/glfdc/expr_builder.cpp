#include "expr_builder.h"

#include "cfold.h"

#include <cstring>
#include <tuple>
#include <type_traits>

using namespace glfdc;

namespace {

std::pair<Operand, Operand> reorder_commutative(Operand l, Operand r) {
  if (detail::svalue(l) > detail::svalue(r))
    std::swap(l, r);

  return std::make_pair(l, r);
}

} // namespace

ExpressionBuilder::ExpressionBuilder() : dag_(new ExprDAG) {}

Value ExpressionBuilder::getBinding(uintptr_t unbound) {
  assert(dag_ != nullptr);
  auto it = dag_->unbound_lookup_.find(unbound);

  if (it != dag_->unbound_lookup_.end()) {
    assert(it->second < dag_->unbound_values_.size());
    return UnboundValue{it->second};
  }

  return create_new_binding(unbound);
}

Value ExpressionBuilder::create_new_binding(uintptr_t unbound) {
  assert(dag_ != nullptr);
  assert(dag_->unbound_lookup_.find(unbound) == dag_->unbound_lookup_.end());

  const size_t slot = dag_->unbound_values_.size();
  dag_->unbound_lookup_.insert(std::make_pair(unbound, slot));
  dag_->unbound_values_.push_back(unbound);

  return UnboundValue{slot};
}

Operand ExpressionBuilder::create_sexpr_(
    OperatorKind op,
    Operand l,
    Operand r) {
  assert(dag_ != nullptr);

  const bool is_commutative =
      (op == OperatorKind::add) || (op == OperatorKind::mul);

  if (is_commutative)
    std::tie(l, r) = reorder_commutative(l, r);

  auto opt_cfold = cfold(op, l, r);

  if (opt_cfold.has_value())
    return opt_cfold.value();

  SExpr e = {l, r, op};
  auto it = seen_exprs_.find(e);

  if (it != seen_exprs_.end()) {
    // We've seen it already
    dag_->markSExprReuse(it->second);
    return it->second;
  }

  // Expr unseen - allocate new
  auto& reuses = dag_->reused_subexpressions_;
  auto ref = dag_->addSubExpr(e);

  if (!is_value(l))
    dag_->markSExprReuse(c10::get<SExprRef>(l));
  if (!is_value(r))
    dag_->markSExprReuse(c10::get<SExprRef>(r));

  seen_exprs_.insert(std::make_pair(e, ref));

  return ref;
}
