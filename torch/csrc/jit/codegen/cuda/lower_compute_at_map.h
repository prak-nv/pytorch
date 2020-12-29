#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <unordered_map>
#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class ComputeAtMap {
 public:
  ComputeAtMap() = default;

  void build();

  //! The standard form of compute at defines this compute at and relative
  //! compute at, where relative is compute at position in consumer. However the
  //! mechanism we rely on here for ordering is only dependent on local
  //! information which is "produce at", what position in an output TV domain
  //! should the output TV be produced at. Since we don't have this information
  //! correct for outputs, we're going to transform this information into
  //! produce_at.
  //
  // TODO: Move away from compute at in favor of produce at.
  const std::unordered_map<TensorView*, int>& produce_at_map() const {
    return produce_at_map_;
  }

  //! Disjoint sets of iter domains, only defined if iter domain is within
  //! compute at of a tensor view. Maps these iter domains to a set containing
  //! all other iter domains in the fusion that map to the same loop nest.
  const std::unordered_map<
      IterDomain*,
      std::shared_ptr<std::unordered_set<IterDomain*>>>&
  disjoint_iter_sets() const {
    return disjoint_iter_sets_;
  }

 private:
  std::unordered_map<TensorView*, int> produce_at_map_;

  std::unordered_map<
      IterDomain*,
      std::shared_ptr<std::unordered_set<IterDomain*>>>
      disjoint_iter_sets_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch