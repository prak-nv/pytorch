#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/sexpr.h>

#include <c10/util/Optional.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace glfdc {

// Evaluates constant expression
TORCH_CUDA_CU_API inline scalar_type cfold(
    OperatorKind op,
    scalar_type l,
    scalar_type r) noexcept {
  switch (op) {
    case OperatorKind::log_and:
      return l && r;
    case OperatorKind::add:
      return l + r;
    case OperatorKind::sub:
      return l - r;
    case OperatorKind::mul:
      return l * r;
    case OperatorKind::div: {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(r != 0);
      return l / r;
    }
    case OperatorKind::ceildiv: {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(r != 0);
      return (l + r - 1) / r;
    }
    case OperatorKind::mod: {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(r != 0);
      return l % r;
    }
  }

  TORCH_INTERNAL_ASSERT(false, "Unreachable");
}

// Preforms constant folding if possible to do so
// NB: it is not recursive, assumes operands and v1 and v2 are already constant
// folded
TORCH_CUDA_CU_API inline c10::optional<scalar_type> cfold(
    OperatorKind op,
    Operand v1,
    Operand v2) {
  auto as_scalar = [](Operand op) -> scalar_type {
    return c10::get<scalar_type>(c10::get<Value>(op));
  };

  if (!isValue(v1) || isSymbolicValue(v1))
    return c10::nullopt;

  if (!isValue(v2) || isSymbolicValue(v2))
    return c10::nullopt;

  return cfold(op, as_scalar(v1), as_scalar(v2));
}

} // namespace glfdc
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
