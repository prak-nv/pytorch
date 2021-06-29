#pragma once

#include <torch/csrc/jit/codegen/cuda/glfdc/eval.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace glfdc {

// Prepares symbolic expression in optimized form
UProgram optimizeExpr(
    Expr e,
    const SymbolicExpr::Template& templ,
    OperandStackOrder);

// Evaluates symbolic expression in optimized form
c10::optional<scalar_type> evaluateOptimizedExpr(
    EvalState& es,
    const UProgram& program,
    const InitialStack<scalar_type>& init,
    SymbolicExpr::binding_fn_t fn) noexcept;

} // namespace glfdc
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch