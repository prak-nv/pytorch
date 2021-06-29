#include "eval.h"

#include "cfold.h"
#include "expr.h"

using namespace glfdc;

namespace {

using skiptree_t = std::vector<std::pair<SExprRef, std::size_t>>;

// Recursive preorder binary tree traversal.
// Its also counting nodes and values of each subtree;
template <typename Tree_, typename LeftChildFn_, typename RightChildFn_>
size_t preorder_dfs_fn(Tree_ t, skiptree_t& list, LeftChildFn_ left_fn, RightChildFn_ right_fn)
{
  list.emplace_back(t, 0);
  size_t index = list.size() - 1;

  auto left = left_fn(t);
  size_t lchild_count = 0, rchild_count = 0;

  if (left.has_value())
    lchild_count = preorder_dfs_fn(left.value(), list, left_fn, right_fn);

  auto right = right_fn(t);

  if (right.has_value())
    rchild_count = preorder_dfs_fn(right.value(), list, left_fn, right_fn);

  list[index].second = lchild_count + rchild_count;

  return 1 + lchild_count + rchild_count;
}

template <typename Tree_, typename Fn_, typename LeftChildFn_, typename RightChildFn_>
auto make_preorder_dfs_traversal(Fn_ visit, LeftChildFn_ left_fn, RightChildFn_ right_fn)
{
  return [visit, left_fn, right_fn] (Tree_ t) -> void {

    // XXX: This is already our operation list constructed
    // we're still missing stack operands information
    // TODO: rewrite as an iterative traversal or simply return this list?
    skiptree_t pre_list;

    preorder_dfs_fn(t, pre_list, left_fn, right_fn);

    for (auto p : pre_list)
      c10::guts::apply(visit, p);
  };
}

auto make_sexpr_child_getters(const ExprDAG& dag) noexcept {

  auto left_child_f =
    [&d = dag](SExprRef ref) -> c10::optional<SExprRef> {
    SExpr e = d.fetch(ref);

    if (is_value(e.lhs_))
      return c10::nullopt;

    return c10::get<SExprRef>(e.lhs_);
  };

  auto right_child_f =
    [&d = dag](SExprRef ref) -> c10::optional<SExprRef> {
    SExpr e = d.fetch(ref);

    if (is_value(e.rhs_))
      return c10::nullopt;

    return c10::get<SExprRef>(e.rhs_);
  };

  return std::make_pair(left_child_f, right_child_f);
}

} // namespace anonymous

const Expr& ExprEvaluator::expr() const
{
  return expr_;
}

const ExprDAG& ExprEvaluator::dag() const
{
  return *expr_.dag_;
}

scalar_type ExprEvaluator::scalar_operand_value(Operand op) noexcept
{
  assert(is_value(op));

  if (is_unbound_value(op))
    // Dummy value to hold a place on stack
    return stack_t::GAP_VALUE;

  const auto val = c10::get<Value>(op);
  return c10::get<scalar_type>(val);
}

void ExprEvaluator::prepare_eval_operand(Operand op)
{
   if (is_value(op))
   {
     initial_stack_.push(scalar_operand_value(op));

     if (is_unbound_value(op))
     {
       auto &unbound = c10::get<UnboundValue>(c10::get<Value>(op));
       // Memoize we need to update stack value
       binding_gaps_.emplace_back(initial_stack_.size() - 1, dag().get_binding(unbound));
     }
   }
   // Not a value - child operations will handle everything
}

void ExprEvaluator::calculate_subops_operands() // O(n) - n is number of operations
{
  assert(!operations_.empty());
  auto& first_op = operations_.front();

  // First operation should proceed all following operations
  assert(first_op.nsubops == operations_.size() - 1);

  // NB: This does DFS on preorder binary tree node list - we're able to recover tree structure since:
  // - iff there is left subtree following node is its root node
  // - we memoize how many nodes subtree has in its root
  //
  // To explain this tree layout:
  //
  //                                            root node child count
  //                                            right subtree root child count
  //                                                |
  //                left subtree root child count   |
  //                           |                    |
  //                           |                    |
  //                           V                    V
  // [root][left subtree nodes][right subtree nodes]
  //
  // We also use such traversal during lazy evaluation.
  calculate_operands_range(0, operations_.size());
}

size_t ExprEvaluator::calculate_operands_range(size_t start, size_t end)
{
  assert(start < end && "Visiting empty subtree");
  assert(end <= operations_.size());

  assert(operations_[start].noperands == Operation::INVALID_POP && "Already calculated");

  auto e = dag().fetch(operations_[start].ref);
  unsigned operand_count = 0;

  // next operation is one to perform to evaluate left subtree if there is any,
  size_t lroot_idx = start + 1;
  // otherwise right subtree root index (if any) 
  size_t rroot_idx = start + 1;
  // otherwise start+1 is equal to end

  if (is_value(e.lhs_))
  {
     operand_count += 1;
  }
  else
  {
    assert(lroot_idx < end);

    auto l_subop = operations_[lroot_idx];

    // Coerce right subtree root index
    rroot_idx = lroot_idx + 1 + l_subop.nsubops;

    operand_count += calculate_operands_range(lroot_idx, rroot_idx);
  }

  if (is_value(e.rhs_))
  {
    assert(rroot_idx == end && "Right subtree should be empty");
    operand_count += 1;
  }
  else
  {
    assert(rroot_idx < end && "Right subtree shouldn't be empty");

    auto r_subop = operations_[rroot_idx];
    assert(rroot_idx + 1 + r_subop.nsubops == end);

    operand_count += calculate_operands_range(rroot_idx, end);
  }

  operations_[start].noperands = operand_count;

  return operand_count;
}

void ExprEvaluator::prepare_eval() // O(n)
{
  // If expression is value or runtime value, we don't need to create anything.
  // This value can be evaluated without creating any stack.  
  if (is_value(expr_.subexpr_))
    return;

  // Create list of operations to perform
  // Normaly we'd like to evaluate them postorder (of mirrored tree ie. right child first),
  // however for lazy evaluation those need to be checked preorder.

  auto visit = [this](SExprRef ref, std::size_t child_count) -> void {
     const SExpr expr = dag().fetch(ref);

     prepare_eval_operand(expr.lhs_);
     prepare_eval_operand(expr.rhs_);

     assert(child_count < std::numeric_limits<unsigned>::max());

     operations_.push_back(Operation{ref, unsigned(child_count), Operation::INVALID_POP});
  };

  auto [lchld, rchld] = make_sexpr_child_getters(dag());
  auto traversal = make_preorder_dfs_traversal<SExprRef>(visit, lchld, rchld);

  traversal(c10::get<SExprRef>(expr_.subexpr_));

  calculate_subops_operands();
}

// TODO: iterative version
void ExprEvaluator::evaluate_subexpr(size_t op_index, stack_t& eval_stack, EvalState& es) const
{
  assert(op_index < operations_.size());

  const Operation &op = operations_[op_index];
  SExprRef ref = op.ref;

  auto* slot = es.load(ref);
  const bool is_reused_subexpr = (slot != nullptr);

  // If already evaluated
  if (is_reused_subexpr && *slot != c10::nullopt)
  {
    eval_stack.drop(op.noperands);
    eval_stack.push(slot->value());
    return;
  }

  // Evaluate expr
  SExpr root_expr = dag().fetch(ref);
  std::size_t lhs_op = op_index + 1;
  std::size_t rhs_op = !is_value(root_expr.lhs_)? lhs_op + 1 + operations_[lhs_op].nsubops : lhs_op;

  // NB: we evaluate right subexpr first, because of order of operands on stack
  // This is still DFS but of mirrored tree (or right child first)
  //
  // Right needs evaluation?
  if (!is_value(root_expr.rhs_))
    evaluate_subexpr(rhs_op, eval_stack, es);

  // Result or value of right subtree on the top
  scalar_type rval = eval_stack.pop_top();
  
  // Left needs evaluation?
  if (!is_value(root_expr.lhs_))
    evaluate_subexpr(lhs_op, eval_stack, es);
  
  scalar_type lval = eval_stack.pop_top();

  scalar_type result = cfold(root_expr.op_, lval, rval);
  eval_stack.push(result);

  // Memoize if reused
  if (is_reused_subexpr)
    *slot = result;
}

ExprEvaluator::opt_scalar_t ExprEvaluator::evaluate(EvalState& es, binding_fn_t binding_fn) const
{
  if (is_value(expr_.subexpr_))
    return evaluate_value(c10::get<Value>(expr_.subexpr_), binding_fn);

  constexpr std::size_t root_idx = 0;

  stack_t eval_stack = initial_stack_;
  if (!eval_stack.fill_gaps(binding_gaps_, std::move(binding_fn)))
    return c10::nullopt;

  evaluate_subexpr(root_idx, eval_stack, es);

  return eval_stack.top();
}

ExprEvaluator::opt_scalar_t ExprEvaluator::evaluate_value(Value val, binding_fn_t binding_fn) const
{
  if (is_scalar(val))
    return c10::get<scalar_type>(val);

  UnboundValue unbound = c10::get<UnboundValue>(val);

  return binding_fn(dag().get_binding(unbound));
}

ExprEvaluator::ExprEvaluator(const Expr &e) : expr_(e)
{
  prepare_eval(); // O(n)
}
