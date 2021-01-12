#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <deque>
#include <unordered_map>

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

  // Returns the position in tv->domain() that the buffer should be computed at
  // / stored at
  int producedAt(kir::TensorView* tv) const {
    return producedAt(tv->fuserTv());
  }

  //! Returns if id0 and id1 are mapped to eachother, meaning they represent the
  //! same loop nest in the lowered code
  bool areMapped(IterDomain* id0, IterDomain* id1) const;

  bool areMapped(kir::IterDomain* id0, kir::IterDomain* id1) const;

  //! Returns an iter domain that is parallelized that the provided iter domain
  //! is mapped to. If no parallelized iter domain exists, returns provided iter
  //! domain.
  ParallelType getMappedParallelType(IterDomain* id) const;

  ParallelType getMappedParallelType(kir::IterDomain* id) const;

  // // Using concrete mapped IDs map the IterDomains in the provided vectors
  // std::unordered_map<IterDomain*, IterDomain*> mapFromTo(
  //     const std::vector<IterDomain*>& from,
  //     const std::vector<IterDomain*>& to) const;

  // std::unordered_map<kir::IterDomain*, kir::IterDomain*> mapFromTo(
  //     const std::vector<kir::IterDomain*>& from,
  //     const std::vector<kir::IterDomain*>& to) const;

  // TODO: This is terrible, but we have nice functionality in iter_visitor that
  // isn't moved over. Use of this is limited to indexing and this should
  // definitely be removed by building out kernel ir to have better parity with
  // fusion ir.
  IterDomain* toFusion(kir::IterDomain* kir) const;

  //! Returns an iter domain that is the maximum expanded size of all iter
  //! domains the one provided maps to. Useful for opening loops to the correct
  //! iteration size. Not guarenteed to return the same ID every call, but is
  //! guarenteed to return iter domains in the same disjoint set.
  IterDomain* getConcreteMappedID(IterDomain* id) const;

  kir::IterDomain* getConcreteMappedID(kir::IterDomain* id) const;

  // Prints mapping information via Fusion IR
  std::string toString();

 private:
  void map_ids(IterDomain* id0, IterDomain* id1);

 private:
  std::unordered_map<TensorView*, int> produce_at_map_;

  // Disjoint sets of iter domains, only defined if iter domain is within
  // compute at of a tensor view. Maps these iter domains to a set containing
  // all other iter domains in the fusion that map to the same loop nest.
  std::unordered_map<IterDomain*, std::shared_ptr<std::deque<IterDomain*>>>
      disjoint_iter_set_maps_;

  std::unordered_map<
      kir::IterDomain*,
      std::shared_ptr<std::deque<kir::IterDomain*>>>
      kir_disjoint_iter_set_maps_;

  // Keep a list of disjoint_iter_sets that's deterministic to iterate over
  std::deque<std::shared_ptr<std::deque<IterDomain*>>> disjoint_iter_sets_;

  // Tracks if there's a parallel iter domain associated a disjoint iter domain
  // set
  // TODO: Use shared_pointer instead of pointer to the unordered maps
  std::unordered_map<std::shared_ptr<std::deque<IterDomain*>>, ParallelType>
      parallel_type_map_;

  std::
      unordered_map<std::shared_ptr<std::deque<kir::IterDomain*>>, ParallelType>
          kir_parallel_type_map_;

  // For each IterDomain set we will track how many concrete root domains were
  // used to generate the IterDomain
  std::unordered_map<IterDomain*, IterDomain*> concrete_id_map_;

  std::unordered_map<kir::IterDomain*, kir::IterDomain*> kir_concrete_id_map_;

  // Map kir::IterDomain* back to the fusion IR IterDomain*.
  // TODO: This should be removed!
  std::unordered_map<kir::IterDomain*, IterDomain*> kir_2_fusion;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch