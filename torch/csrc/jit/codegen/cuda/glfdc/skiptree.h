#pragma once

#include <torch/csrc/jit/codegen/cuda/glfdc/sexpr.h>

#include <utility>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace glfdc {

// NB: This does DFS on binary tree node list - we're able to recover
// tree structure since:
// - iff there is left subtree following node is its root node
// - we memoize how many nodes subtree has in its root
//
// To explain this tree layout:
//
//                                            root node child count
//                                            right subtree root child count
//                                                |
//                left subtree root child count   |
//                           |                    |
//                           |                    |
//                           V                    V
// [root][left subtree nodes][right subtree nodes]
//
// We also use such traversal during lazy evaluation.
using skiptree_t = std::vector<std::pair<SExprRef, std::size_t>>;

} // namespace glfdc
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch