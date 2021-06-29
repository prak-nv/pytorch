#include "expr.h"

using namespace glfdc;

void ExprDAG::markSExprReuse(SExprRef ref) {
  assert(ref.index() < reused_subexpressions_.size());

  bool was_reused = reused_subexpressions_[ref.index()];
  reuse_count_ += std::size_t(was_reused ^ true);

  if (was_reused)
    return;

  reused_subexpressions_[ref.index()] = true;
  auto e = this->fetch(ref);

  if (!is_value(e.lhs_))
    markSExprReuse(c10::get<SExprRef>(e.lhs_));

  if (!is_value(e.rhs_))
    markSExprReuse(c10::get<SExprRef>(e.rhs_));
}