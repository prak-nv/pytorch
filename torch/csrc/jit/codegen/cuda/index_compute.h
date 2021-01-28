#pragma once

#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

/*
 * Index compute takes in a list of indices typically generated from the
 * surrounding for loop nest. The number of indicies are intended to match the
 * number of dimensions of the incomming TensorView which may have less or more
 * dimensions than its root due to split/merge operations.
 * Split/merge operations are then replayed backwards produce resulting
 * indices (based on input indices) that match the root dimension.
 *
 * For example with GLOBAL tensor:
 * TV[I, J]
 * TV[Io, Ii{4}, J] = TV.split(I, factor=4)
 * ALLOC: NONE
 * INDEX: indexCompute {i, j, k} -> {i * 4 + j, k}
 * FLATTENED_INDEX: {i * 4 + j, k} -> {i * 4 + j * J + k}
 * PREDICATE: {i * 4 + j, k} -> i * 4 + j < I
 *
 *
 * For example with SHARED tensor:
 *
 * global_TV[I, J]
 * global_TV[Io, Ii{4}, J] = global_TV.split(I, factor=4)
 * smem_TV.compute_at(global_TV, 1)
 * global_TV.parallelize(1, threadIDx.x)
 *
 * ALLOC: alloc(smem_TV, 4 x J)
 * INDEX: indexCompute(smem_TV, {threadIdx.x, k}) -> {threadIdx.x, k}
 * FLATTENED_INDEX: {threadIdx.x * 4 + j, k} -> {threadIdx.x * 4 + j * J + k}
 * PREDICATE: {threadIdx.x * 4 + j, k} -> threadIdx.x * 4 + j < I // Same as if
 * global
 *
 *
 * For example with LOCAL tensor:
 * global_TV[I, J, K]
 * global_TV[Io, Ii{4}, J] = global_TV.split(I, factor=4)
 * reg_TV.compute_at(global_TV, 1)
 * global_TV.parallelize(1, threadIDx.x)
 * global_TV{i, j, k, l} -> { i * 4 + j, k, l }
 * global_TV{ i * 4 + j, k, l } -> { i * 4 + j * J * K  +  k * K  +  l}
 *
 * ALLOC: alloc(reg_TV, J x K)
 * INDEX: {k, l} -> {k, l}
 * FLATTENED_INDEX: {k, l} -> {k * J + l}
 * PREDICATE: i * 4 + j < I && k < J && l < K ->  // Same as if global
 *
 * These indices can then be flattened later based on strides.
 */

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class IndexCompute : public BackwardVisitor {
 protected:
  using BackwardVisitor::handle;

  void handle(Split*) override;
  void handle(Merge*) override;
  void handle(Expr*) override;

  // return extent_map_[id] if exists, else return id->extent()
  kir::Val* getExtent(kir::IterDomain* id);

  bool hasZeroMerged(kir::IterDomain* id);

  // Tensor domain we're mapping back to root
  const TensorDomain* td_;

  // Map we update as we propagate backward, containing all IDs in the
  // propagation. Initial indices are mapped with this map at tv->domain()
  // and are back propagated to tv->rootDomain(). This index_map_ keeps the
  // indices at intermediate IterDomain's in that back propagation.
  std::unordered_map<kir::IterDomain*, kir::Val*> index_map_;

  // Map from IterDomain to their broadcasted extent. If a TV has I0*I1 but its
  // producer has B0*I1 this map will contain a mapping from the ID{B0*I1} to
  // the extent I0*I1. Also contains updated extents if we merge in a 0 index.
  // See zero_merged_in_.
  std::unordered_map<kir::IterDomain*, kir::Val*> extent_map_;

  // This set keeps track of IterDomain's that have had a zero index merged into
  // them. This happens if we do something like tv->axis(0)->split(4) then
  // tv->computeAt(1, ...) if this tensor is in smem or lmem the backward
  // indexing would be (0, i) then when we do the backward computation that zero
  // and i would attempt to be merged together. We handle indices like these
  // specially.
  std::unordered_set<kir::IterDomain*> zero_merged_in_;

  // IDs that are a result of contiguous merges
  std::unordered_set<kir::IterDomain*> contig_ids;

  // Mentions if we should propagate an index down a particular IterDomain path
  // if there's an option
  std::unordered_set<kir::IterDomain*> preferred_paths_;

 public:
  const std::unordered_map<kir::IterDomain*, kir::Val*>& indexMap() const {
    return index_map_;
  }

  const std::unordered_map<kir::IterDomain*, kir::Val*>& extentMap() const {
    return extent_map_;
  }

  const std::unordered_set<kir::IterDomain*>& zeroMergedIn() const {
    return zero_merged_in_;
  }

  // Propagate back from _td using initial_index_map
  IndexCompute(
      const TensorDomain* _td,
      std::unordered_map<kir::IterDomain*, kir::Val*> initial_index_map,
      std::unordered_map<kir::IterDomain*, kir::Val*> _extent_map,
      std::unordered_set<kir::IterDomain*> _zero_merged_in,
      const std::vector<bool>& _root_contiguity,
      std::unordered_set<kir::IterDomain*> preferred_paths = {});

  // Updates index_map, extent_map, and zero_merged_in based on id_map and
  // returns a new IndexCompute ready to be used.
  IndexCompute updateIndexCompute(
      const TensorDomain* new_td,
      const std::unordered_map<IterDomain*, IterDomain*>& id_map,
      const std::vector<bool>& _root_contiguity);

  virtual void run();

  // Map producer contiguity information to consumer, if entries don't match
  // mark as false
  static std::vector<bool> contiguityPasC(
      kir::TensorView* producer,
      kir::TensorView* consumer);

  static std::vector<bool> contiguityAnd(
      const std::vector<bool>& contig1,
      const std::vector<bool>& contig2);
};

//! Apply swizzle and update root indices accordingly
class IndexSwizzle : public IndexCompute {
 public:
  IndexSwizzle(
      const TensorView* tv,
      std::unordered_map<kir::IterDomain*, kir::Val*> initial_index_map,
      std::unordered_map<kir::IterDomain*, kir::Val*> extent_map,
      std::unordered_set<kir::IterDomain*> zero_merged_in);

  void run() override;

 protected:
  using IndexCompute::handle;

  void handle(Expr* e) override;

 private:
  const TensorView* tv_ = nullptr;
  SwizzleType swizzle_type_ = SwizzleType::NoSwizzle;
  std::vector<IterDomain*> ids_to_swizzle_;
  std::unordered_set<IterDomain*> swizzled_ids_;
};

// Simple interface for IndexCompute
// If getComputeAtAxis and more generally TensorView const model is fixed, we
// can make the below tensorviews const.
class Index {
 private:
  // Producer indexing if it's in shared or local memory
  static kir::TensorIndex* getProducerIndex_impl(
      TensorView* producer,
      const TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);

  // Consumer indexing if it's in shared or local memory
  static kir::TensorIndex* getConsumerIndex_impl(
      const TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);

  // Producer if it's in global memory
  static kir::TensorIndex* getGlobalProducerIndex(
      TensorView* producer,
      const TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);

  // Consumer indexing if it's in global memory
  static kir::TensorIndex* getGlobalConsumerIndex(
      const TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);

 public:
  // Indexing functions
  // Consumer = Producer
  // i.e. T0 = T1... -> T0 is the consumer, T1 is the producer
  // Producer indexing dispatch
  static kir::TensorIndex* getProducerIndex(
      TensorView* producer,
      const TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);

  // Consumer index dispatch
  static kir::TensorIndex* getConsumerIndex(
      const TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);

  // Consumer indices for predicates, keep all indices matching in root domain.
  // Even those not used for physical addressing. Returns pair <root indices, if
  // indices are mapped to rfactor dom>
  static std::pair<std::vector<kir::Val*>, bool> getConsumerRootPredIndices(
      const kir::TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops,
      const std::vector<bool>& root_contiguity,
      bool unswitch = false);

  static std::vector<kir::Val*> getProducerRootVectIndices(
      const kir::TensorView* producer_tv,
      const kir::TensorView* consumer_tv,
      const std::vector<kir::ForLoop*>& loops,
      const std::vector<bool>& root_contiguity,
      const ComputeAtRootDomainMap& ca_root_map);

  static std::vector<kir::Val*> getConsumerRootVectIndices(
      const kir::TensorView* consumer_tv,
      const std::vector<kir::ForLoop*>& loops,
      const std::vector<bool>& root_contiguity,
      const ComputeAtRootDomainMap& ca_root_map);
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
