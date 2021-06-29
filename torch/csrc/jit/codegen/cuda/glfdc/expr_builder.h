#include "bitvector.h"
#include "expr.h"
#include "sexpr_cmp.h"

#include <memory>
#include <unordered_map>

#include "c10/util/Optional.h"

namespace glfdc {

// ExpressionBuilder - creates expressions and subexpressions
struct ExpressionBuilder {
  ExpressionBuilder();

  // Returns symbolic value for unknown that can be used in subexpression
  Value getBinding(uintptr_t unbound); // O(log2(n))

  Operand create_sexpr(OperatorKind op, Operand l, Operand r) {
    return create_sexpr_(op, l, r);
  }
  Operand create_sexpr(OperatorKind op, Operand l, Value v) {
    return create_sexpr_(op, l, Operand(v));
  }
  Operand create_sexpr(OperatorKind op, Value v, Operand r) {
    return create_sexpr_(op, Operand(v), r);
  }
  Operand create_sexpr(OperatorKind op, Value v1, Value v2) {
    return create_sexpr_(op, Operand(v1), Operand(v2));
  }

  Expr create_expr(Operand op) const noexcept {
    return Expr{dag_.get(), op};
  }

  const ExprDAG& dag() const noexcept {
    return *dag_;
  }

  auto takeDAG() noexcept {
    return std::move(dag_);
  }

  // Returns bitmap of reused subexpressions
  const bitvector_t& reuses() const {
    return dag_->reused_subexpressions_;
  }

 private:
  Value create_new_binding(uintptr_t);
  Operand create_sexpr_(OperatorKind op, Operand l, Operand r);

 private:
  using sexpr_lookup_t =
      std::unordered_map<SExpr, SExprRef, sexpr_hash, sexpr_eq>;
  sexpr_lookup_t seen_exprs_;

  std::unique_ptr<ExprDAG> dag_;
};

} // namespace glfdc
