#include <torch/csrc/jit/codegen/cuda/glfdc/eval.h>

#include <torch/csrc/jit/codegen/cuda/glfdc/cfold.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/eval_optimizer.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/sexpr_cmp.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/symbolic_expr_visitor.h>

using namespace torch::jit::fuser::cuda::glfdc;

EvalState::EvalState(ReusedExprMapping mapping)
    : memo_(mapping.size()), mapping_(std::move(mapping)) {}

// Returns pointer to slot for reused subexpression value or nullptr if
// subexpression is not reused
EvalState::ValueSlotPtr EvalState::load(SExprRef sexpr) {
  auto opt_map = mapping_.slot(sexpr); // O(1)

  if (!opt_map.has_value())
    return nullptr;

  return memo_.data() + opt_map.value();
}

EvalState::ValueSlot EvalState::recall(abs_offset_t slot) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      size_t(slot) < slotCount(), "Slot index out of bounds");
  return memo_[size_t(slot)];
}

// Stores subexpression value
void EvalState::store(ValueSlot& slot, scalar_type val) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!slot.has_value(), "Already evaluated");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      uintptr_t(&slot) >= uintptr_t(memo_.data()), "Invalid slot pointer");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      uintptr_t(&slot) < uintptr_t(memo_.data() + memo_.size()),
      "Invalid slot pointer");
  slot = val;
}

void EvalState::remember(abs_offset_t slot, scalar_type value) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      size_t(slot) < slotCount(), "Slot index out of bounds");
  memo_[size_t(slot)] = value;
}

// Creates an EvalState with no slots for memoized values
EvalState EvalState::createEmpty() {
  return EvalState(ReusedExprMapping::createForEagerEval());
}

// Clears all memoized values from EvalState
void EvalState::forgetAllMemoized() {
  std::fill(memo_.begin(), memo_.end(), c10::nullopt);
}

std::size_t EvalState::slotCount() const noexcept {
  return memo_.size();
}

ReusedExprMapping ReusedExprMapping::createForMemoizedEval(const ExprDAG& dag) {
  const BitVector& reused_sexprs = dag.reuseBitmap();
  sparse_map mapping(reused_sexprs.size());

  std::size_t slot_counter = 0;

  // Assign slot for each reused expression
  for (std::size_t i = 0; i < reused_sexprs.size(); ++i) {
    // if subexpression is reused assign mapping
    if (reused_sexprs[i])
      mapping.insert(i, slot_counter++);
  }

  return {mapping};
}

const Expr& SymbolicExpr::expr() const {
  return expr_;
}

const ExprDAG& SymbolicExpr::dag() const {
  return expr_.dag();
}

SymbolicExpr::SymbolicExpr(const Expr& e)
    : expr_(e), templ_(SymbolicExprVisitor::createTemplate(e, stack_order_)) {}

void SymbolicExpr::evaluateSubExpr(
    size_t op_index,
    stack_t& eval_stack,
    EvalState& es) const {
  TORCH_INTERNAL_ASSERT(op_index < templ_.operations.size());

  const Operation& op = templ_.operations[op_index];
  SExprRef ref = op.ref;

  auto* slot = es.load(ref);
  const bool is_reused_subexpr = (slot != nullptr);

  // If already evaluated
  if (is_reused_subexpr && *slot != c10::nullopt) {
    eval_stack.drop(op.noperands);
    eval_stack.push(slot->value());
    return;
  }

  // Evaluate expression
  SExprNode root_expr = dag().fetch(ref);
  std::size_t lhs_op = op_index + 1;
  std::size_t rhs_op = !isValue(root_expr.lhs_)
      ? lhs_op + 1 + templ_.operations[lhs_op].nsubops
      : lhs_op;

  scalar_type lval, rval;
  // NB: we evaluate subexpr depending on order of operands on stack
  if (stack_order_ == OperandStackOrder::RL) {
    // This is still DFS but of mirrored tree (or right child first)
    //
    // Right needs evaluation?
    if (isSExpr(root_expr.rhs_))
      evaluateSubExpr(rhs_op, eval_stack, es);

    // Result or value of right subtree on the top
    rval = eval_stack.pop_top();

    // Left needs evaluation?
    if (isSExpr(root_expr.lhs_))
      evaluateSubExpr(lhs_op, eval_stack, es);

    // Pop left operand from stack
    lval = eval_stack.pop_top();
  } else {
    // This is still DFS of tree (left child first)
    // Left needs evaluation?
    if (isSExpr(root_expr.lhs_))
      evaluateSubExpr(lhs_op, eval_stack, es);

    // Pop left operand from stack
    lval = eval_stack.pop_top();

    if (isSExpr(root_expr.rhs_))
      evaluateSubExpr(rhs_op, eval_stack, es);

    // Result or value of right subtree on the top
    rval = eval_stack.pop_top();
  }

  scalar_type result = cfold(root_expr.op_, lval, rval);
  // Push the result on top of the stack
  eval_stack.push(result);

  // Memoize if reused
  if (is_reused_subexpr)
    *slot = result;
}

opt_scalar_t SymbolicExpr::evaluateExpr(EvalState& es, binding_fn_t provider)
    const {
  if (program_.isValid()) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(program_.isCurrentVersion(expr_));
    // We generate program that direcly refers to slots for memoized values
    // If that it is required we can generate version for eager evaluation
    // as an alternative we can support only stack machine evaluation for such
    // cases.
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        dag().reuseCount() == es.slotCount(), "Eager evaluation not supported");
    auto result = evaluateOptimized(es, provider);
    // Note we are using an empty EvalState, so we wont return previosly
    // calculated wrong value
    EvalState empty = EvalState::createEmpty();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        result == evaluateTemplate(empty, provider));
    // Used only in debug build
    (void)empty;

    return result;
  } else {
    return evaluateTemplate(es, provider);
  }
}

opt_scalar_t SymbolicExpr::evaluateOptimized(EvalState& es, binding_fn_t fn)
    const {
  return evaluateOptimizedExpr(es, program_, templ_.initial_stack, fn);
}

opt_scalar_t SymbolicExpr::evaluateTemplate(
    EvalState& es,
    binding_fn_t provider) const {
  // Prepare evaluation stack, as copy of initial stack
  stack_t eval_stack = templ_.initial_stack;

  // Drop cookie arguments for optimized form
  eval_stack.drop(templ_.binding_gaps.size());

  // Fill all placeholder values with runtime values
  if (!eval_stack.fillSymbolicValues(templ_.binding_gaps, provider))
    return c10::nullopt;

  // If operand is not a value evaluate subexpression
  if (isSExpr(expr_.operand())) {
    constexpr std::size_t root_idx = 0;
    evaluateSubExpr(root_idx, eval_stack, es);
  }

  // Return final evaluation result for top of the stack
  return eval_stack.top();
}

void SymbolicExpr::optimize() {
  if (!templ_.operations.empty()) {
    program_ = optimizeExpr(expr_, templ_, stack_order_);
  }
}

std::unique_ptr<EvalState> torch::jit::fuser::cuda::glfdc::
    createEvalStateForDAG(const ExprDAG& dag) {
  auto mapping = ReusedExprMapping::createForMemoizedEval(dag);
  return std::make_unique<EvalState>(mapping);
}
