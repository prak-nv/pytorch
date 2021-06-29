#pragma once

// Basic data structures for expressions
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <c10/util/Optional.h>
#include <c10/util/variant.h>

#include <cstddef>
#include <cstdint>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace glfdc {

using scalar_type = std::int64_t;
using opt_scalar_t = c10::optional<scalar_type>;

enum abs_offset_t : std::size_t {};

// Symbolic operand of subexpression
struct TORCH_CUDA_CU_API SymbolicValue {
 public:
  SymbolicValue() noexcept = default;
  SymbolicValue(abs_offset_t index) : index_(index) {}

  SymbolicValue(const SymbolicValue&) noexcept = default;
  SymbolicValue(SymbolicValue&&) noexcept = default;

  SymbolicValue& operator=(const SymbolicValue&) noexcept = default;
  SymbolicValue& operator=(SymbolicValue&&) noexcept = default;

  bool operator!=(SymbolicValue other) const noexcept {
    return index_ != other.index_;
  }

  bool operator==(SymbolicValue other) const noexcept {
    return index_ == other.index_;
  }

  // Returns index of symbolic value inside ExprDAG
  abs_offset_t index() const noexcept {
    return index_;
  }

  TORCH_CUDA_CU_API friend inline void swap(
      SymbolicValue& l,
      SymbolicValue& r) noexcept {
    std::swap(l.index_, r.index_);
  }

 private:
  // Index into array of all bindings values (cookies) of owner ExprDAG
  abs_offset_t index_;
};

// Known or unknown value of subexpression operand
using Value = c10::variant<scalar_type, SymbolicValue>;

static_assert(std::is_copy_constructible<Value>::value, "");
static_assert(std::is_copy_assignable<Value>::value, "");

// Reference to subexpression (index to subexpression inside ExprDAG)
//
// This holds an index to specific subexpression
class TORCH_CUDA_CU_API SExprRef {
 public:
  SExprRef() noexcept = default;
  explicit SExprRef(abs_offset_t index) noexcept : index_(index) {}

  SExprRef(const SExprRef&) noexcept = default;
  SExprRef(SExprRef&&) noexcept = default;

  SExprRef& operator=(const SExprRef&) noexcept = default;
  SExprRef& operator=(SExprRef&&) noexcept = default;

  bool operator!=(SExprRef other) const noexcept {
    return index_ != other.index_;
  }

  bool operator==(SExprRef other) const noexcept {
    return index_ == other.index_;
  }

  // Returns index of subexpression into subexpression list of owner DAG
  abs_offset_t index() const noexcept {
    return index_;
  };

  TORCH_CUDA_CU_API friend inline void swap(SExprRef& l, SExprRef& r) {
    std::swap(l.index_, r.index_);
  }

 private:
  // Index of subexpression in owner ExprDAG
  abs_offset_t index_;
};

// Operand of subexpression
using Operand = c10::variant<c10::monostate, Value, SExprRef>;

// Returns true IFF operand is know or unknown (symbolic) scalar
TORCH_CUDA_CU_API inline bool isValue(Operand op) noexcept {
  return c10::holds_alternative<Value>(op);
}

// Returns true IFF operand is subexpression
TORCH_CUDA_CU_API inline bool isSExpr(Operand op) noexcept {
  return c10::holds_alternative<SExprRef>(op);
}

// Returns true IFF Value is SymbolicValue
TORCH_CUDA_CU_API inline bool isSymbolicValue(Value val) noexcept {
  return c10::holds_alternative<SymbolicValue>(val);
}

// Returns true IFF Operand is UnboundValue
TORCH_CUDA_CU_API inline bool isSymbolicValue(Operand op) noexcept {
  return isValue(op) && isSymbolicValue(c10::get<Value>(op));
}

// Returns true IFF Value is scalar constant
TORCH_CUDA_CU_API inline bool isScalar(Value val) noexcept {
  return c10::holds_alternative<scalar_type>(val);
}

// Returns true IFF Operand is scalar constant
TORCH_CUDA_CU_API inline bool isScalar(Operand op) noexcept {
  return isValue(op) && isScalar(c10::get<Value>(op));
}

// Operator of subexpression
enum class OperatorKind : char {
  add = '+',
  sub = '-',
  mul = '*',
  div = '/',
  mod = '%',
  ceildiv = 'c',
  log_and = '&',
};

// Subexpression of expression reduction DAG
// This holds binary operation to perform on other subexpression or values (be
// it symbolic or constant)
// Please note that subexpression is constant size which is welcome property
// This means we can use it as constant size key for CSE (Common Subexpression
// Elimination)
struct TORCH_CUDA_CU_API SExprNode {
  Operand lhs_; // reference to left hand operand of subexpression
  Operand rhs_; // reference to right hand operand of subexpression

  OperatorKind op_; // operator of binary subexpression

  SExprNode() noexcept = default;
  SExprNode(const SExprNode&) noexcept = default;
  SExprNode& operator=(const SExprNode&) noexcept = default;
};

} // namespace glfdc
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
