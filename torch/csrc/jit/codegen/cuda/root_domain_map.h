#pragma once

#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Represents an iteration domain of a TensorDomain. Only used for
//! root domain mapping.
//!
//! Note that an IterDomain object may be reused
//! across multiple TensorDomains, but an IterDomain in a
//! TensorDomain may not be necessarily mappable to the same
//! IterDomain used in a different TensorDomain. Thus, for the purpose
//! of root domain mapping, an iteration domain needs to be identified
//! with an IterDomain and its TensorDomain.
class DomainKey {
 public:
  DomainKey() = default;
  DomainKey(
      const TensorDomain* td,
      const IterDomain* id,
      const IterDomain* concrete_id = nullptr)
      : td_(td), id_(id), concrete_id_(concrete_id) {}
  const TensorDomain* td() const {
    return td_;
  }
  const IterDomain* id() const {
    return id_;
  }
  const IterDomain* concreteId() const {
    return concrete_id_;
  }
  bool operator==(const DomainKey& other) const {
    return td() == other.td() && id() == other.id() &&
        concreteId() == other.concreteId();
  }
  std::ostream& print(std::ostream& os) const;

 private:
  const TensorDomain* td_ = nullptr;
  const IterDomain* id_ = nullptr;
  const IterDomain* concrete_id_ = nullptr;
};

inline std::ostream& operator<<(std::ostream& os, const DomainKey& key) {
  return key.print(os);
}

struct DomainKeyHash {
  std::size_t operator()(const DomainKey& key) const {
    return std::hash<const TensorDomain*>{}(key.td()) ^
        std::hash<const IterDomain*>{}(key.id());
  }
};

using DomainKeySet = std::unordered_set<DomainKey, DomainKeyHash>;

template <typename Mapped>
using DomainKeyMap = std::unordered_map<DomainKey, Mapped, DomainKeyHash>;

class RootDomainMap;

//! A helper class to find all DomainKeys that are consumers of
//! reduction outputs. Such consumer IterDomains may not be mapped to
//! the producer reduction domain since the corresponding reduction
//! loop must be closed before any of the consumers can appear.
class TORCH_CUDA_API UnmappableReductionDomains : private IterVisitor {
 public:
  UnmappableReductionDomains();
  virtual ~UnmappableReductionDomains() = default;

  //! Returns true when mapping consumer domains would cause a
  //! reduction output domain to be mapped with a consumer domain of
  //! the redution. It needs to be avoided as computing consumers of
  //! reduction outputs within the corresponding reduction loop is not
  //! possible. This routine is used to build root domain mappings.
  bool isReductionOutputMapped(
      const std::vector<DomainKey>& consumer_domains,
      const RootDomainMap& root_map) const;
  // const DisjointSet<DomainKey, DomainKeyHash>& eq_set) const;

 private:
  using IterVisitor::handle;
  void handle(ReductionOp* op) override;

 private:
  //! Map from Reduction output DomainKeys to consumer DomainKeys
  DomainKeyMap<DomainKeySet> reduction_domains_;
};

//! Models root-domain mappings for computeAt
//!
//! Two iteration domains are mapped when computeAt of one iteration
//! domain is possible at another iteration domain. Consider a simple
//! example:
//!    T2 [i0,i1] = T1[i2,i3] + T0[i4,i5]
//! This will create mappings between i0, i2 and i4.
class TORCH_CUDA_API RootDomainMap : private BackwardVisitor {
 public:
  //! Create a DisjointSet of root IterDomains by traversing the
  //! current fusion entirely. IterDomains that can be mapped each
  //! other with computeAt are grouped into the same subset in the
  //! DisjointSet.
  explicit RootDomainMap();

  //! Check if two iterdomains can be mapped to each other
  //!
  //! \param td_a A TensorDomain
  //! \param id_a An IterDomain in td_a
  //! \param td_b Another TensorDomain
  //! \param id_b An IterDomain in td_b
  //! \returns Boolean representing if they are mapped
  bool canMap(
      const TensorDomain* td_a,
      const IterDomain* id_a,
      const TensorDomain* td_b,
      const IterDomain* id_b) const;

  //! Return a map from a producer TensorDomain to a consumer
  //! TensorDomain
  //!
  //! \param producer A producer TensorDomain
  //! \param consumer A consumer TensorDomain
  //! \param root_dims_to_map Maps only producer root domains in this set
  std::unordered_map<IterDomain*, IterDomain*> mapProducerToConsumer(
      const TensorDomain* producer,
      const TensorDomain* consumer,
      const std::unordered_set<const IterDomain*>& root_dims_to_map) const;

  //! Return a map from a consumer TensorDomain to a producer
  //! TensorDomain
  //!
  //! \param consumer A consumer TensorDomain
  //! \param producer A producer TensorDomain
  //! \param root_dims_to_map Maps only consumer root domains in this set
  std::unordered_map<IterDomain*, IterDomain*> mapConsumerToProducer(
      const TensorDomain* consumer,
      const TensorDomain* producer,
      const std::unordered_set<const IterDomain*>& root_dims_to_map) const;

  //! Print out mappings
  std::ostream& print(std::ostream& os) const;

 private:
  //! Set a pair of producer-consumer domains as mappable
  void setMapped(const DomainKey& producer, const DomainKey& consumer) {
    eq_set_.join(producer, consumer);
  }

  //! Track a pair of producer-consumer domains as potentially mappable.
  void setMaybeMapped(
      const TensorDomain* producer_td,
      const IterDomain* producer_id,
      const TensorDomain* consumer_td,
      const IterDomain* consumer_id);

  void addToPendingList(const DomainKey& producer, const DomainKey& consumer);

  //! Map pointwise IterDomains from inputs of expressions to outputs.
  //! Do not map reduction IterDomains in inputs.
  void mapPointwiseOrReductionOp(Expr* e);

  using BackwardVisitor::handle;

  void handle(Expr* e) override;

  void handle(UnaryOp* uop) override {
    mapPointwiseOrReductionOp(uop);
  }

  void handle(BinaryOp* bop) override {
    mapPointwiseOrReductionOp(bop);
  }

  void handle(TernaryOp* top) override {
    mapPointwiseOrReductionOp(top);
  }

  void handle(ReductionOp* op) override {
    mapPointwiseOrReductionOp(op);
  }

  void handle(BroadcastOp* op) override;

  //! Maps all consumers with a producer.
  //! This is called for each of TensorViews in a backward traversal,
  //! recursively building mappings from the output tensors to the
  //! input tensors.
  bool mapAllConsumers(const DomainKey& producer_key);

  void handle(TensorView* tv) override;

  bool hasMatchingDomains(const std::vector<DomainKey>& unique_domains);

  bool safeToMap(const DomainKeySet& domains);

  //! Check if two iterdomains can be mapped to each other
  //!
  //! \param key_a A DomainKey
  //! \param td_b Another TensorDomain
  //! \param id_b An IterDomain in td_b
  //! \returns Boolean representing if they are mapped
  bool canMap(
      const DomainKey& key_a,
      const TensorDomain* td_b,
      const IterDomain* id_b) const;

  bool canMap(const DomainKey& key_a, const DomainKey& key_b) const;

  std::vector<DomainKey> getConcretizedKeys(
      const TensorDomain* td,
      const IterDomain* id) const;
  std::unordered_set<const IterDomain*>& getConcretizedDomains(
      const TensorDomain* td,
      const IterDomain* id);
  bool hasConcretizedDomains(const TensorDomain* td, const IterDomain* id)
      const;
  static DomainKey getDefaultConcretizedDomain(
      const TensorDomain* td,
      const IterDomain* id);

  //! Return a map between root IterDomains of a producer-consumer
  //! pair.
  //!
  //! \param producer A producer TensorDomain
  //! \param consumer A consumer TensorDomain
  //! \param root_dims_to_map Maps only from IterDomains in this set
  //! \param producer_to_consumer Maps from producer to consumer if true
  std::unordered_map<IterDomain*, IterDomain*> map(
      const TensorDomain* producer,
      const TensorDomain* consumer,
      const std::unordered_set<const IterDomain*>& root_dims_to_map,
      bool producer_to_consumer) const;

 private:
  DisjointSet<DomainKey, DomainKeyHash> eq_set_;
  //! Keep track of what we want to try and map. Set in attemptToProveId.
  DomainKeyMap<DomainKeySet> pending_map_;
  std::unordered_set<Expr*> visited_;
  UnmappableReductionDomains incompatible_domains_;
  DomainKeyMap<std::unordered_set<const IterDomain*>> bcast_map_;
};

inline std::ostream& operator<<(std::ostream& os, const RootDomainMap& map) {
  return map.print(os);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
