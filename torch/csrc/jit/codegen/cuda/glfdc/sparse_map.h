#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include <c10/util/Optional.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace glfdc {

// The twist on sparse-set
// Paper: “An Efficient Representation for Sparse Sets” 1993 Preston Briggs ,
// Linda Torczon http://www.cs.rice.edu/MSCP/papers/loplas.sets.ps.gz Also well
// described https://research.swtch.com/sparse
//
// Uses same trick but stores both k,v pair in dense array
class sparse_map {
  using map_entry_t = std::pair<std::size_t, std::size_t>;

 public:
  explicit sparse_map() {}

  explicit sparse_map(std::size_t size) : sparse_(size), dense_() {
    sparse_.shrink_to_fit();
  }

  sparse_map(const sparse_map&) = default;
  sparse_map& operator=(const sparse_map&) = default;

  // has - queries if map contains element with key k
  bool has(std::size_t k) const noexcept {
    if (k >= sparse_.size())
      return false;

    if (sparse_[k] >= dense_.size())
      return false;

    return dense_[sparse_[k]].first == k;
  }

  // find - finds element with key k
  c10::optional<std::size_t> find(std::size_t k) const {
    if (!has(k))
      return c10::nullopt;

    return dense_[sparse_[k]].second;
  }

  // NB: Insert operation semantics is different from traditional STL semantics
  // Precondition of this function is that there is no existing mapping for key
  // k.
  void insert(std::size_t k, std::size_t v) noexcept {
    TORCH_INTERNAL_ASSERT(k < sparse_.size());
    TORCH_INTERNAL_ASSERT(!has(k));

    sparse_[k] = dense_.size();
    dense_.emplace_back(k, v);
  }

  void shrink_to_fit() noexcept {
    dense_.shrink_to_fit();
  }

  std::size_t size() const noexcept {
    return dense_.size();
  }

  bool empty() const noexcept {
    return dense_.empty();
  }

 private:
  std::vector<std::size_t> sparse_;
  std::vector<map_entry_t> dense_;
};

} // namespace glfdc
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
