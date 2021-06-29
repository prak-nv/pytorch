#pragma once

// Basic data structures for expressions

#include <cstddef>
#include <cstdint>
#include "c10/util/variant.h"

namespace glfdc {

using scalar_type = std::int64_t;

// Symbolic operand of subexpression
struct UnboundValue {
  std::size_t
      index_; // Index into array of all UnboundValues inside owner ExprDAG

  UnboundValue() noexcept = default;

  UnboundValue(const UnboundValue&) noexcept = default;
  UnboundValue(UnboundValue&&) noexcept = default;

  UnboundValue& operator=(const UnboundValue&) noexcept = default;
  UnboundValue& operator=(UnboundValue&&) noexcept = default;

  bool operator!=(UnboundValue other) const noexcept {
    return index_ != other.index_;
  }

  bool operator==(UnboundValue other) const noexcept {
    return index_ == other.index_;
  }
};

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

// Known or unknown value of subexpression operand
using Value = c10::variant<scalar_type, UnboundValue>;

static_assert(std::is_copy_constructible<Value>::value, "");
static_assert(std::is_copy_assignable<Value>::value, "");

// Reference to subexpression (index to subexpression DAG)
struct SExprRef {
  std::size_t index_; // Index of subexpression in owner ExprDAG

  SExprRef() noexcept = default;

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

  size_t index() const noexcept {
    return index_;
  };
};

// Operand of subexpression
using Operand = c10::variant<c10::monostate, Value, SExprRef>;

// Returns true IFF operand is know or unknown (symbolic) scalar
inline bool is_value(Operand op) noexcept {
  return op.index() == 1;
}

inline bool is_unbound_value(Value val) noexcept {
  return val.index() == 1;
}

inline bool is_unbound_value(Operand op) noexcept {
  return is_value(op) && is_unbound_value(c10::get<Value>(op));
}

inline bool is_scalar(Value val) noexcept {
  return val.index() == 0;
}

inline bool is_scalar(Operand op) noexcept {
  return is_value(op) && is_scalar(c10::get<Value>(op));
}

inline bool is_sexpr(Operand op) noexcept {
  return op.index() == 2;
}

// Subexpression of expression reduction DAG
// This holds binary operation to perform on other subexpression or values (be
// it symbolic or constant)
struct SExpr {
  Operand lhs_; // reference to left hand operand of subexpression
  Operand rhs_; // reference to right hand operand of subexpression

  OperatorKind op_; // operator

  SExpr() noexcept = default;
  SExpr(const SExpr&) noexcept = default;
  SExpr& operator=(const SExpr&) noexcept = default;

  bool is_unbound() const noexcept {
    return is_unbound_value(lhs_) || is_unbound_value(rhs_);
  }
};

inline void swap(SExprRef& l, SExprRef& r) {
  std::swap(l.index_, r.index_);
}

inline void swap(UnboundValue& l, UnboundValue& r) noexcept {
  std::swap(l.index_, r.index_);
}

} // namespace glfdc