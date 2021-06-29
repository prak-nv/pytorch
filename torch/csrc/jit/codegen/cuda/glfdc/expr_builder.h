#include <torch/csrc/jit/codegen/cuda/glfdc/bitvector.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/expr.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/sexpr_cmp.h>

#include <memory>
#include <unordered_map>

#include <c10/util/Optional.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace glfdc {

// ExpressionBuilder - creates expressions and subexpressions
class TORCH_CUDA_CU_API ExpressionBuilder {
 public:
  ExpressionBuilder();

  // Returns symbolic value for unknown that can be used in subexpression
  Value getBinding(uintptr_t unbound); // O(log2(n))

  // Creates operand of subexpression.
  Operand createOperand(OperatorKind op, Operand l, Operand r) {
    return createSExpr_(op, l, r);
  }
  Operand createOperand(OperatorKind op, Operand l, Value v) {
    return createSExpr_(op, l, Operand(v));
  }
  Operand createOperand(OperatorKind op, Value v, Operand r) {
    return createSExpr_(op, Operand(v), r);
  }
  Operand createOperand(OperatorKind op, Value v1, Value v2) {
    return createSExpr_(op, Operand(v1), Operand(v2));
  }

  // Creates expression for value or subexpression given by operand
  Expr createExpr(Operand op) const noexcept {
    return Expr{op, *dag_};
  }

  const ExprDAG& dag() const noexcept {
    return *dag_;
  }

  // Takes ownership of ExprDAG disowning ExpressionBuilder
  auto takeDAG() noexcept {
    return std::move(dag_);
  }

  // Returns bitmap of reused subexpressions for CSE
  const BitVector& reuses() const {
    return dag_->reused_subexpressions_;
  }

 private:
  Value createNewBinding(uintptr_t);
  Operand createSExpr_(OperatorKind op, Operand l, Operand r);

 private:
  using sexpr_lookup_t =
      std::unordered_map<SExprNode, SExprRef, SExprHash, SExprEq>;
  sexpr_lookup_t seen_exprs_;

  std::unique_ptr<ExprDAG> dag_;
};

} // namespace glfdc
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
