#pragma once

#include "sexpr.hh"

#include <cassert>
#include <map>
#include <vector>

namespace glfdc {

struct ExprDAG;

// Expr - points at root of expression reduction tree
// NB: Expr should not outlive ExprDAG, since stores index to;
struct Expr
{
  const ExprDAG *dag_;
  Operand subexpr_;
};

// Keeps DAG of all expressions reduction graphs
struct ExprDAG
{
  std::map<uintptr_t, std::size_t> unbound_lookup_; // XXX: any mapping - rb-tree for now
  std::vector<uintptr_t> unbound_values_;

  std::vector<SExpr> subexpressions_; // List of subexpressions

public:
  SExprRef add_subexpr(SExpr expr)
  {
    size_t new_idx = subexpressions_.size();
    subexpressions_.push_back(expr);

    return SExprRef{new_idx};
  }

  // Returns expression for subexpression reference
  SExpr fetch(SExprRef e) const noexcept
  {
    assert(e.index_ < subexpressions_.size());
    return subexpressions_[e.index_];
  }

  uintptr_t get_binding(UnboundValue ubv) const noexcept // O(1)
  {
    assert(ubv.index_ < unbound_values_.size());
    return unbound_values_[ubv.index_];
  }
};

} // namespace glfdc

