#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

TORCH_CUDA_API std::vector<Expr*> reorderExprsForLoopNestGeneration();

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
