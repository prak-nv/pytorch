#pragma once

#include <vector>

namespace torch {
namespace jit {
namespace fuser {

// template<std::size_t Sz_>
// using MaskVector = SmallVector<Sz_>

using MaskVector = std::vector<bool>;

// using ContiguityMask = MaskVector<32>;
using ContiguityMask = MaskVector;

} // namespace fuser
} // namespace jit
} // namespace torch
