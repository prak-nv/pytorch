#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Apply double-buffering transformation
std::vector<kir::Expr*> applyDoubleBuffering(
    const std::vector<kir::Expr*>& indexed_loops);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
