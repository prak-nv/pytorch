#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/lower_compute_at_map.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

TORCH_CUDA_API std::vector<Expr*> reorderExprsTest();

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch