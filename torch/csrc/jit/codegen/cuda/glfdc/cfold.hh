#pragma once

#include "sexpr.hh"

#include <cassert>

#include "c10/util/Optional.h"

namespace glfdc {

// Evaluates constant expression
inline scalar_type cfold(OperatorKind op, scalar_type l, scalar_type r) noexcept
{
  switch(op)
  {
  case OperatorKind::log_and:
    return l && r;
  case OperatorKind::add:
    return l + r; 
  case OperatorKind::sub:
    return l - r;
  case OperatorKind::mul:
    return l * r;
  case OperatorKind::div:
  {
    if (r == 0) return 0;
    return l / r;
  }
  case OperatorKind::ceildiv:
  {
     if (r == 0) return 0;
     return (l + r - 1) / r;
  }
  case OperatorKind::mod:
  {
    if (r == 0) return 0;
    return l % r;
  }
  }

  assert(false && "Unreachable");
}

// Preforms constant folding if possible to do so
// NB: it is not recursive, assumes operands and v1 and v2 are already constant folded
inline c10::optional<scalar_type> cfold(OperatorKind op, Operand v1, Operand v2)
{
  auto as_scalar = [] (Operand op) -> scalar_type {
    return c10::get<scalar_type>(c10::get<Value>(op));
  };

  if (!is_value(v1) || is_unbound_value(v1))
    return c10::nullopt;

  if (!is_value(v2) || is_unbound_value(v2))
    return c10::nullopt;

  return cfold(op, as_scalar(v1), as_scalar(v2));
}

} // namespace glfdc

