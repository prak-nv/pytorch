#pragma once

#include <torch/csrc/jit/codegen/cuda/glfdc/bitvector.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/sexpr.h>

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <map>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace glfdc {

class ExprDAG;
class ExpressionBuilder;

// Expr - points subexpression and its owner expression DAG (ExprDAG)
// This is convinence class which wraps subexpression and also stores its DAG
//
// NB: Expr should not outlive ExprDAG, since stores index to;
class TORCH_CUDA_CU_API Expr {
 public:
  Expr(Value v, const ExprDAG& dag) noexcept : dag_(&dag), subexpr_(v) {}
  Expr(SExprRef ref, const ExprDAG& dag) noexcept : dag_(&dag), subexpr_(ref) {}
  Expr(Operand op, const ExprDAG& dag) noexcept : dag_(&dag), subexpr_(op) {}

  // Return root node of expression
  c10::optional<SExprNode> rootNode() const noexcept;

  // Returns reference to root expression of expression
  c10::optional<SExprRef> rootRef() const noexcept;

  // Returns operand of expression
  Operand operand() const noexcept;

  // Returns value of expression when it is just single value
  c10::optional<Value> value() const noexcept;

  const ExprDAG& dag() const noexcept {
    return *dag_;
  }

 private:
  // Owner DAG of an expression
  const ExprDAG* dag_;
  // Subexpression or value refered to
  Operand subexpr_;
};

// Keeps DAG of all expressions reduction graphs
class TORCH_CUDA_CU_API ExprDAG {
  friend class ExpressionBuilder;

 public:
  // Adds single subexpression to DAG performing simple CSE
  SExprRef addSubExprNode(SExprNode expr);

  // Returns expression for subexpression reference
  TORCH_CUDA_CU_API SExprNode fetch(SExprRef e) const noexcept {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(e.index() < subexpressions_.size());
    return subexpressions_[e.index()];
  }

  // Returns binding cookie for give symbolic value
  TORCH_CUDA_CU_API uintptr_t getBinding(SymbolicValue ubv) const noexcept {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ubv.index() < binding_keys_.size());
    return binding_keys_[ubv.index()];
  }

  // Returns bitmap of subexpression reuses
  const BitVector& reuseBitmap() const noexcept {
    return reused_subexpressions_;
  }

  // Returns true if an subexpression has any reuses and was made common part of
  // other expression  by CSE (Common Subexpression Elimination)
  bool isReusedSubExpr(SExprRef e) const noexcept {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(e.index() < reused_subexpressions_.size());
    return reused_subexpressions_[e.index()];
  }

  // Returns number of reused subexpression inside DAG
  TORCH_CUDA_CU_API std::size_t reuseCount() const noexcept {
    return reuse_count_;
  }

  // Marks subexpression as reused for CSE
  TORCH_CUDA_CU_API void markSExprReuse(SExprRef ref);

 private:
  // Maps unbound values (symbolic) cookies to index into binding_keys_
  std::map<uintptr_t, abs_offset_t> binding_lookup_;

  // Stores all unbound values cookies
  std::vector<uintptr_t> binding_keys_;

  // List of subexpressions nodes
  std::vector<SExprNode> subexpressions_;

  // One bit reference count of subexpressions reuse for CSE
  BitVector reused_subexpressions_;

  // Number of reused subexpressions
  std::size_t reuse_count_ = 0;
};

} // namespace glfdc
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
