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
  // There's three modes of these iter domain mappings. For indexing, for loop
  // nest mapping/generation, and to figure out the parallelization strategy.
  //
  // For index/loop mode consider:
  //
  // consumer[i0, b1] = producer[i0]
  // consumer->merge(0) (consumer will now be [i0 * b1])
  // When producer is replayed as consumer (the direction we use for mapping)
  // with BestEffortReplay forward_bcast_mismatch = True the producer to
  // consumer map will have both a mapping of consumer(i0) to producer(i0) as
  // well as consumer(i0*b1) to producer(i0). This latter mapping is important
  // for loop nest mappings as the consumer will generate a loop based on i0*b1
  // and the producer may be computeAt inside this loop nest. However, for
  // indexing we do not want these two maps as producer may be indexed as i0*i1
  // depending on the loop nest structure and how it was built. Therefore we
  // really need to carry two sets of maps around for lowering.
  //
  // TODO: Document parallel mode example.
  enum class MappingMode { PARALLEL, LOOP, INDEX };

  ComputeAtMap() = default;
  ComputeAtMap(MappingMode mapping_mode) : mapping_mode_(mapping_mode) {}

  void build();

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

  //! Returns an iter domain that is the maximum expanded size of all iter
  //! domains the one provided maps to. Useful for opening loops to the correct
  //! iteration size. Not guarenteed to return the same ID every call, but is
  //! guarenteed to return iter domains in the same disjoint set.
  IterDomain* getConcreteMappedID(IterDomain* id) const;

  kir::IterDomain* getConcreteMappedID(kir::IterDomain* id) const;

  // TODO: Would be great if we didn't need this, but we have nice functionality
  // in iter_visitor that isn't moved over. Use of this is limited to indexing
  // and this should definitely be removed by building out kernel ir to have
  // better parity with fusion ir.
  IterDomain* toFusion(kir::IterDomain* kir) const;

  // Prints mapping information via Fusion IR
  std::string toString();

 private:
  void map_ids(IterDomain* id0, IterDomain* id1);

  MappingMode mapping_mode_ = MappingMode::LOOP;

 private:
  // This is actually only used when mapping mode == LOOP. Only used in expr
  // sorting, it's actually maximum position where a loop is shared across any
  // neighbor.
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
  std::unordered_map<std::shared_ptr<std::deque<IterDomain*>>, ParallelType>
      parallel_type_map_;

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