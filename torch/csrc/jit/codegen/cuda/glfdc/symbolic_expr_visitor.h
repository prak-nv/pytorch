#pragma once

#include <torch/csrc/jit/codegen/cuda/glfdc/eval.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/eval_stack.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace glfdc {

// Helper for accessing child tree nodes during traversal of ExprDAG in
// linerized form
class DAGAccess {
 public:
  explicit DAGAccess(const ExprDAG& dag) noexcept;
  // Retrives left child reference
  c10::optional<SExprRef> getLeft(SExprRef ref) const noexcept;

  // Retrives right child reference
  c10::optional<SExprRef> getRight(SExprRef ref) const noexcept;

  // Returns node pointed by ref
  SExprNode getNode(SExprRef ref) const noexcept;

  // Returns DAG
  const ExprDAG& dag() const noexcept;

 private:
  // Expression DAG
  const ExprDAG* dag_ = nullptr;
};

// Prepares symbolic expression Template for evalution
class SymbolicExprVisitor {
  using initial_stack_t = InitialStack<scalar_type>;
  using stack_t = EvalStack<scalar_type>;

 public:
  static SymbolicExpr::Template createTemplate(Expr expr, OperandStackOrder);

 private:
  SymbolicExprVisitor(DAGAccess getters);

  const ExprDAG& dag() const noexcept;

  // Creates operations list for symbolic evaluation
  void prepareOperations(const skiptree_t& list);
  // Place cookies onto the stack
  void prepareStackCookies();

  // Adds value of the operand to the initial stack.
  void addStackArgument(Operand op);

  // Adds stack operands for left side of an expression
  void prepareLHSOperand(SExprNode node);

  // Adds stack operands for right side of an expression
  void prepareRHSOperand(SExprNode node);

  // Pre-order visit
  void onVisitPre(SExprNode node, OperandStackOrder);

  // Post-order visit
  void onVisitPost(SExprNode node, OperandStackOrder);

  // Traveses node of subexpression tree and prepares initial operand stack
  // of an expression
  std::size_t traverseSubtreeDFS(
      SExprRef ref,
      skiptree_t& list,
      OperandStackOrder);

  // Calulates number of operands used by operation
  void calculateSubopsOperands();

  // Calulates number of operands used by operation for subtree in range of
  // start to end
  std::size_t calculateOperandsRange(size_t start, size_t end);

 private:
  // Provides tree node access to DAG in linearized form
  DAGAccess dag_access_;
  // List of operations to perform to evaluate an expression stored in
  // linearized tree form
  std::vector<Operation> operations_;
  // Initial stack of known operands with gaps for runtime values
  initial_stack_t initial_stack_;
  // List of gap indices on initial_stack for runtime values
  // Those are filled in on copy of the stack during evaluation
  std::vector<BindingGapType> binding_gaps_;
};

} // namespace glfdc
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch