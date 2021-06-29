#include "bitvector.hh"
#include "expr.hh"
#include "sexpr_cmp.hh"

#include <memory>
#include <unordered_map>

#include "c10/util/Optional.h"
#include "c10/util/flat_hash_map.h"

namespace glfdc {

// ExpressionBuilder - creates expressions and subexpressions
struct ExpressionBuilder
{
  ExpressionBuilder();

  // Returns symbolic value for unknown that can be used in subexpression
  Value get_binding(uintptr_t unbound); // O(log2(n))

  // Adds equivalent value for other symbolic value
  Value add_binding_equivalence(uintptr_t unbound, uintptr_t equivalent); // O(log2(n))

  Operand create_sexpr(OperatorKind op, Operand l, Operand r) { return create_sexpr_(op, l, r); }
  Operand create_sexpr(OperatorKind op, Operand l, Value v) { return create_sexpr_(op, l, Operand(v)); }
  Operand create_sexpr(OperatorKind op, Value v, Operand r){ return create_sexpr_(op, Operand(v), r); }
  Operand create_sexpr(OperatorKind op, Value v1, Value v2) { return create_sexpr_(op, Operand(v1), Operand(v2)); }

  Expr create_expr(Operand op) const noexcept
  {
    return Expr{dag_.get(), op};
  }

  const ExprDAG& dag() const noexcept {
    return *dag_;
  }

  auto takeDAG() noexcept
  {
    return std::move(dag_);
  }

  // Returns bitmap of reused subexpressions
  const auto& reuses() const
  {
    return reused_subexpressions_;
  }

  std::size_t reuse_count() const noexcept
  {
    return reuse_count_;
  }

private:
  Value create_new_binding(uintptr_t);
  Operand create_sexpr_(OperatorKind op, Operand l, Operand r);

  void mark_reuse(SExprRef ref);

private:
  using sexpr_lookup_t = std::unordered_map<SExpr, SExprRef, sexpr_hash, sexpr_eq>;
  //using sexpr_lookup_t = ska::flat_hash_map<SExpr, SExprRef, sexpr_hash, sexpr_eq>;
  // TODO: something faster here:
  sexpr_lookup_t seen_exprs_;

  // only needed for lazy_eval construction
  bitvector_t reused_subexpressions_;

  std::unique_ptr<ExprDAG> dag_;
  std::size_t reuse_count_ = 0;
};

} // namespace glfdc

