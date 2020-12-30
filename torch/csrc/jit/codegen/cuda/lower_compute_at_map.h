#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

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

  // Returns the position in tv->domain() that the buffer should be computed at
  // / stored at
  int producedAt(TensorView* tv) const {
    auto produce_at_it = produce_at_map_.find(tv);
    TORCH_INTERNAL_ASSERT(
        produce_at_it != produce_at_map_.end(),
        "Could not find a produced at entry for ",
        tv);
    return produce_at_it->second;
  }

  // Disjoint sets of iter domains, only defined if iter domain is within
  // compute at of a tensor view. Maps these iter domains to a set containing
  // all other iter domains in the fusion that map to the same loop nest.
  // const std::unordered_map<
  //     IterDomain*,
  //     std::shared_ptr<std::unordered_set<IterDomain*>>>&
  // disjoint_iter_sets() const {
  //   return disjoint_iter_sets_;
  // }

  //! Returns if id0 and id1 are mapped to eachother, meaning they represent the
  //! same loop nest in the lowered code
  bool areMapped(IterDomain* id0, IterDomain* id1) const;

  bool areMapped(kir::IterDomain* id0, kir::IterDomain* id1) const;

  //! Returns an iter domain that is parallelized that the provided iter domain
  //! is mapped to. If no parallelized iter domain exists, returns provided iter
  //! domain.
  IterDomain* getParallelizedMappedID(IterDomain* id) const;

  kir::IterDomain* getParallelizedMappedID(kir::IterDomain* id) const;

  //! Returns an iter domain that is the maximum expanded size of all iter
  //! domains the one provided maps to. Useful for opening loops to the correct
  //! iteration size. Not guarenteed to return the same ID every call, but is
  //! guarenteed to return iter domains in the same disjoint set.
  IterDomain* getConcreteMappedID(IterDomain* id) const;

  kir::IterDomain* getConcreteMappedID(kir::IterDomain* id) const;

 private:
  std::unordered_map<TensorView*, int> produce_at_map_;

  std::unordered_map<
      IterDomain*,
      std::shared_ptr<std::unordered_set<IterDomain*>>>
      disjoint_iter_sets_;

  std::unordered_map<
      kir::IterDomain*,
      std::shared_ptr<std::unordered_set<kir::IterDomain*>>>
      kir_disjoint_iter_sets_;

  // Tracks if there's a parallel iter domain associated a disjoint iter domain
  // set
  // TODO: Use shared_pointer instead of pointer to the unordered maps
  std::unordered_map<
      std::shared_ptr<std::unordered_set<IterDomain*>>,
      IterDomain*>
      parallel_type_map_;

  std::unordered_map<
      std::shared_ptr<std::unordered_set<kir::IterDomain*>>,
      kir::IterDomain*>
      kir_parallel_type_map_;

  // For each IterDomain set we will track how many concrete root domains were
  // used to generate the IterDomain
  std::unordered_map<IterDomain*, int> n_concrete_ids_;

  std::unordered_map<kir::IterDomain*, int> kir_n_concrete_ids_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch