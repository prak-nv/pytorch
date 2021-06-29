#pragma once

// 

#include "sexpr.hh"

#include <cstring>
#include <functional>

namespace glfdc { namespace detail {

// Boost hash_combine();
template <typename Ty_>
inline void hash_combine(std::size_t & seed, const Ty_ & v) noexcept
{
  std::hash<Ty_> h;
  seed ^= h(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Operand kind for value calculation (see explaination in svalue)
inline unsigned operand_kind(Operand op) noexcept
{
  enum : unsigned
  {
    val_scalar,
    val_unbound,
    val_expr,
  };

  if (is_value(op))
    return unsigned(is_unbound_value(op)? val_unbound : val_scalar);
  
  return unsigned(val_expr);
}

template <typename ToTy_, typename FromTy_>
ToTy_ bitcast(FromTy_ val) noexcept
{
  static_assert(sizeof(FromTy_) <= sizeof(ToTy_), "Size must match");
  static_assert(std::is_trivially_copyable<FromTy_>::value, "Must be memcpy-able");
  static_assert(std::is_trivially_copyable<ToTy_>::value, "Must be memcpy-able");

  ToTy_ ret{};
  std::memcpy(&ret, &val, sizeof(val));

  return ret;
}

// Value of scalar
inline std::size_t scalar_v(Value v) noexcept
{
  return bitcast<std::size_t>(c10::get<scalar_type>(v));
}

inline std::size_t operand_value(Operand op) noexcept
{
  if (is_value(op))
  {
    auto v = c10::get<Value>(op);
    return is_unbound_value(op)? c10::get<UnboundValue>(v).index_ : scalar_v(v);
  }

  return c10::get<SExprRef>(op).index(); 
}

// Value of operand to ensure cannonical form of associative operations
inline std::pair<unsigned, std::size_t> svalue(Operand op) noexcept
{
  return std::make_pair(operand_kind(op), operand_value(op));
}

}} // namespace glfdc::detail

namespace glfdc {

// Subexpression equality functor
struct sexpr_eq
{
  bool operator() (const SExpr& e1, const SExpr& e2) const
  {
    return (e1.op_ == e2.op_) && (e1.lhs_ == e2.lhs_) && (e1.rhs_ == e2.rhs_);
  }
};

// Hashing function for subexpressions
struct sexpr_hash
{
  inline std::size_t operator() (const SExpr& e) const noexcept;
};

inline std::size_t sexpr_hash::operator() (const SExpr& e) const noexcept
{
  using namespace detail;

  std::size_t seed = 0;

  hash_combine(seed, e.op_);

  hash_combine(seed, svalue(e.lhs_).first);
  hash_combine(seed, svalue(e.lhs_).second);

  hash_combine(seed, svalue(e.rhs_).first);
  hash_combine(seed, svalue(e.rhs_).second);

  return seed;
}


} // namespace glfdc
