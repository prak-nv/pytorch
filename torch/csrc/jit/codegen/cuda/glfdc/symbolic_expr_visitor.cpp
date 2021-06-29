#include <torch/csrc/jit/codegen/cuda/glfdc/sexpr_cmp.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/symbolic_expr_visitor.h>

using namespace torch::jit::fuser::cuda::glfdc;

DAGAccess::DAGAccess(const ExprDAG& dag) noexcept : dag_(&dag) {}
// Retrives left child reference
c10::optional<SExprRef> DAGAccess::getLeft(SExprRef ref) const noexcept {
  SExprNode e = dag_->fetch(ref);

  if (isValue(e.lhs_)) {
    return c10::nullopt;
  }

  return c10::get<SExprRef>(e.lhs_);
}

// Retrives right child reference
c10::optional<SExprRef> DAGAccess::getRight(SExprRef ref) const noexcept {
  SExprNode e = dag_->fetch(ref);

  if (isValue(e.rhs_)) {
    return c10::nullopt;
  }

  return c10::get<SExprRef>(e.rhs_);
}

SExprNode DAGAccess::getNode(SExprRef ref) const noexcept {
  return dag_->fetch(ref);
}

const ExprDAG& DAGAccess::dag() const noexcept {
  return *dag_;
}

SymbolicExpr::Template SymbolicExprVisitor::createTemplate(
    Expr expr,
    OperandStackOrder ord) {
  DAGAccess getters(expr.dag());

  SymbolicExprVisitor visitor(getters);
  auto e = expr.rootRef();

  SymbolicExpr::Template ret;

  // If expression is value or runtime value, we don't need to create
  // anything. This value can be evaluated without creating any operations.
  if (e == c10::nullopt) {
    visitor.addStackArgument(expr.operand());
  } else {
    // Othwise traverse subexpression tree to prepare operation list and
    // operands
    skiptree_t list;

    visitor.traverseSubtreeDFS(*e, list, ord);

    for (auto p : list) {
      auto ref = p.first;
      auto child_count = p.second;
      TORCH_INTERNAL_ASSERT(child_count < std::numeric_limits<unsigned>::max());
      visitor.operations_.push_back(
          Operation{ref, unsigned(child_count), Operation::INVALID_POP});
    }
    // Calculate how many operands operations require
    // (information used when we are retrieve already calculated value)
    visitor.calculateSubopsOperands();
  }

  // Append cookies for symbolic values, those are used to fill gaps
  // in optimized form by fill instruction
  visitor.prepareStackCookies();

  ret.operations = std::move(visitor.operations_);
  ret.initial_stack = std::move(visitor.initial_stack_);
  ret.binding_gaps = std::move(visitor.binding_gaps_);

  return ret;
}

SymbolicExprVisitor::SymbolicExprVisitor(DAGAccess getters)
    : dag_access_(getters) {}

const ExprDAG& SymbolicExprVisitor::dag() const noexcept {
  return dag_access_.dag();
}

// Creates operations list for symbolic evaluation
void SymbolicExprVisitor::prepareOperations(const skiptree_t& list) {
  for (auto p : list) {
    auto ref = p.first;
    auto child_count = p.second;
    TORCH_INTERNAL_ASSERT(child_count < std::numeric_limits<unsigned>::max());
    operations_.push_back(
        Operation{ref, unsigned(child_count), Operation::INVALID_POP});
  }
  // Calculate how many operands operations require
  // (information used when we are retrieve already calculated value)
  calculateSubopsOperands();
}

// Place cookies onto the stack
void SymbolicExprVisitor::prepareStackCookies() {
  // Note reverse order.
  for (auto it = binding_gaps_.rbegin(); it != binding_gaps_.rend(); ++it) {
    initial_stack_.push(detail::bitcast<scalar_type>(it->second));
  }
}

// Adds value of the operand to the initial stack.
void SymbolicExprVisitor::addStackArgument(Operand op) {
  if (isValue(op)) {
    // Place value or placeholder on stack
    auto val = c10::get<Value>(op);
    scalar_type s = stack_t::GAP_VALUE;

    // Memoize we need to update stack value at evaluation time
    if (isSymbolicValue(val)) {
      auto symbol = c10::get<SymbolicValue>(val);
      binding_gaps_.emplace_back(
          initial_stack_.size(), dag().getBinding(symbol));

    } else {
      s = c10::get<scalar_type>(val);
    }
    initial_stack_.push(s);
  }
  // Not a value - child operations will handle everything
}

// Adds stack operands for left side of an expression
void SymbolicExprVisitor::prepareLHSOperand(SExprNode node) {
  addStackArgument(node.lhs_);
}

// Adds stack operands for right side of an expression
void SymbolicExprVisitor::prepareRHSOperand(SExprNode node) {
  addStackArgument(node.rhs_);
}

// Pre-order visit
void SymbolicExprVisitor::onVisitPre(SExprNode node, OperandStackOrder ord) {
  if (ord == OperandStackOrder::RL) {
    // Lhs on bottom
    prepareLHSOperand(node);
  } else {
    prepareRHSOperand(node);
  }
}

// Post-order visit
void SymbolicExprVisitor::onVisitPost(SExprNode node, OperandStackOrder ord) {
  if (ord == OperandStackOrder::RL) {
    // Rhs on top
    prepareRHSOperand(node);
  } else {
    prepareLHSOperand(node);
  }
}

// Traveses node of subexpression tree and prepares initial operand stack
// of an expression
std::size_t SymbolicExprVisitor::traverseSubtreeDFS(
    SExprRef ref,
    skiptree_t& list,
    OperandStackOrder ord) {
  // Add this node to operation list
  list.emplace_back(ref, 0);

  size_t index = list.size() - 1;

  auto left = dag_access_.getLeft(ref);
  // Count of children in left and right subtree
  size_t lchild_count = 0, rchild_count = 0;

  // Fetch root node of the subtree
  auto node = dag_access_.getNode(ref);

  // Pre-order visit - adds operands for first subtree
  onVisitPre(node, ord);

  if (left.has_value()) {
    lchild_count = traverseSubtreeDFS(left.value(), list, ord);
  }

  auto right = dag_access_.getRight(ref);

  if (right.has_value()) {
    rchild_count = traverseSubtreeDFS(right.value(), list, ord);
  }

  // Post-order visit - adds operands for second subtree
  onVisitPost(node, ord);

  // Update total number of number of children
  list[index].second = lchild_count + rchild_count;

  // Return number of nodes in current subtree
  return 1 + lchild_count + rchild_count;
}

void SymbolicExprVisitor::calculateSubopsOperands() {
  TORCH_INTERNAL_ASSERT(!operations_.empty());
  auto& first_op = operations_.front();

  // First operation should proceed all following operations
  TORCH_INTERNAL_ASSERT(first_op.nsubops == operations_.size() - 1);

  calculateOperandsRange(0, operations_.size());
}

std::size_t SymbolicExprVisitor::calculateOperandsRange(
    size_t start,
    size_t end) {
  TORCH_INTERNAL_ASSERT(start < end && "Visiting empty subtree");
  TORCH_INTERNAL_ASSERT(end <= operations_.size());

  TORCH_INTERNAL_ASSERT(
      operations_[start].noperands == Operation::INVALID_POP &&
      "Already calculated");

  auto e = dag_access_.getNode(operations_[start].ref);
  unsigned operand_count = 0;

  // next operation is one to perform to evaluate left subtree if there is
  // any,
  size_t lroot_idx = start + 1;
  // otherwise right subtree root index (if any)
  size_t rroot_idx = start + 1;
  // otherwise start+1 is equal to end

  // Calculate number of operands for left subtree
  if (isValue(e.lhs_)) {
    operand_count += 1;
  } else {
    TORCH_INTERNAL_ASSERT(lroot_idx < end);

    auto l_subop = operations_[lroot_idx];

    // Coerce right subtree root index
    rroot_idx = lroot_idx + 1 + l_subop.nsubops;

    operand_count += calculateOperandsRange(lroot_idx, rroot_idx);
  }

  // Calculate number of operands for right subtree
  if (isValue(e.rhs_)) {
    TORCH_INTERNAL_ASSERT(rroot_idx == end, "Right subtree should be empty");
    operand_count += 1;
  } else {
    TORCH_INTERNAL_ASSERT(rroot_idx < end, "Right subtree shouldn't be empty");

    auto r_subop = operations_[rroot_idx];
    TORCH_INTERNAL_ASSERT(rroot_idx + 1 + r_subop.nsubops == end);

    operand_count += calculateOperandsRange(rroot_idx, end);
  }

  // Update operand count for an operation
  operations_[start].noperands = operand_count;

  return operand_count;
}