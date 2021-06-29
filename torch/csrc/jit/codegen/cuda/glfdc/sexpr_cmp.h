#pragma once

#include <torch/csrc/jit/codegen/cuda/glfdc/sexpr.h>

#include <cstring>
#include <functional>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace glfdc {
namespace detail {

// Boost hash_combine();
template <typename Ty_>
inline void hash_combine(std::size_t& seed, const Ty_& v) noexcept {
  std::hash<Ty_> h;
  seed ^= h(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Operand kind for value calculation (see explaination in svalue)
inline unsigned operandKind(Operand op) noexcept {
  enum : unsigned {
    val_scalar,
    val_unbound,
    val_expr,
  };

  if (isValue(op))
    return unsigned(isSymbolicValue(op) ? val_unbound : val_scalar);

  return unsigned(val_expr);
}

template <typename ToTy_, typename FromTy_>
ToTy_ bitcast(FromTy_ val) noexcept {
  static_assert(sizeof(FromTy_) <= sizeof(ToTy_), "Size must match");
  static_assert(
      std::is_trivially_copyable<FromTy_>::value, "Must be memcpy-able");
  static_assert(
      std::is_trivially_copyable<ToTy_>::value, "Must be memcpy-able");

  ToTy_ ret{};
  std::memcpy(&ret, &val, sizeof(val));

  return ret;
}

// Value of scalar
inline std::size_t scalarValue(Value v) noexcept {
  return bitcast<std::size_t>(c10::get<scalar_type>(v));
}

inline std::size_t operandValue(Operand op) noexcept {
  if (isValue(op)) {
    auto v = c10::get<Value>(op);
    return isSymbolicValue(op) ? c10::get<SymbolicValue>(v).index()
                               : scalarValue(v);
  }

  return c10::get<SExprRef>(op).index();
}

// Value of operand to ensure cannonical form of associative operations
inline std::pair<unsigned, std::size_t> svalue(Operand op) noexcept {
  return std::make_pair(operandKind(op), operandValue(op));
}

} // namespace detail
} // namespace glfdc

namespace glfdc {

// Subexpression equality functor
struct SExprEq {
  bool operator()(const SExprNode& e1, const SExprNode& e2) const {
    return (e1.op_ == e2.op_) && (e1.lhs_ == e2.lhs_) && (e1.rhs_ == e2.rhs_);
  }
};

// Hashing function for subexpressions
struct SExprHash {
  std::size_t operator()(const SExprNode& e) const noexcept;
};

} // namespace glfdc
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
