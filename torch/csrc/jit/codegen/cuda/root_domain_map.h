#pragma once

#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class DomainKey {
 public:
  DomainKey() = default;
  DomainKey(const TensorDomain* td, const IterDomain* id) : td_(td), id_(id) {}
  const TensorDomain* td() const {
    return td_;
  }
  const IterDomain* id() const {
    return id_;
  }
  bool operator==(const DomainKey& other) const {
    return td() == other.td() && id() == other.id();
  }
  std::ostream& print(std::ostream& os) const;

 private:
  const TensorDomain* td_ = nullptr;
  const IterDomain* id_ = nullptr;
};

inline std::ostream& operator<<(std::ostream& os, const DomainKey& key) {
  return key.print(os);
}

struct DomainKeyHash {
  std::size_t operator()(const DomainKey& key) const {
    return std::hash<const TensorDomain*>{}(key.td()) ^ std::hash<const IterDomain*>{}(key.id());
  }
};

class TORCH_CUDA_API FindIncompatibleDomains : private IterVisitor {
 public:
  FindIncompatibleDomains();
  virtual ~FindIncompatibleDomains() = default;

  bool isReductionOutputMerged(const std::vector<DomainKey>& consumer_domains,
                               const DisjointSet<DomainKey, DomainKeyHash>& eq_set) const;

 private:
  using IterVisitor::handle;
  void handle(ReductionOp* op) override;

 private:
  std::unordered_map<DomainKey, std::unordered_set<DomainKey, DomainKeyHash>, DomainKeyHash> inconsistent_domains_;
};

//! Models equivalence provable by the graph
//!
//! This traversal processes root domains only,
//! equalities , e.g. :
//!    T2 [i0,i1] = T1[i2,i3] + T0[i4,i5]
//! will prove that i2 and i4 are equal in the sense that
//!    i2.start = i4.start, i2.extent = i4.extent
//! Depends on ConcretizeDomain, and equalities involving
//! broadcast domains are defined based on the concretized version
class TORCH_CUDA_API RootDomainMap : private BackwardVisitor {
 public:
  explicit RootDomainMap();

// API call to check if two IterDomains are equal
// checks start and extent, contains both scalar check and graph traversal
// broadcast domains are concretized before comparing

  //! Checks if two iterdomains are equal
  //!
  //! Equality defined as equal start and equal extent
  //! true means a and b are equal
  //! false only means that they cannot be proven equal based
  //! on scalar check and graph traversal
  //!
  //! \param a An iterdomain
  //! \param b Another iterdomain from the same fusion
  //! \returns Boolean representing if they are proven to be
  //!          equivalent in the sense that they have equal
  //!          start and extent
  bool canMap(const TensorDomain* td_a, const IterDomain* id_a,
              const TensorDomain* td_b, const IterDomain* id_b) const;

  std::unordered_map<IterDomain*, IterDomain*> mapProducerToConsumer(
      const TensorDomain* producer, const TensorDomain* consumer,
      const std::unordered_set<const IterDomain*>& root_dims_to_map) const;

  std::unordered_map<IterDomain*, IterDomain*> mapConsumerToProducer(
      const TensorDomain* consumer, const TensorDomain* producer,
      const std::unordered_set<const IterDomain*>& root_dims_to_map) const;

  std::ostream& print(std::ostream& os) const;

 private:
  // Utility class to record new equality found
  void proveId(const DomainKey& producer,
               const DomainKey& consumer) {
    eq_set_.join(producer, consumer);
  }

  void attemptToProveId(const TensorDomain* producer_td, const IterDomain* producer_id,
                        const TensorDomain* consumer_td, const IterDomain* consumer_id);

  // Inspect a pointwise or reduction op and record the identified equality
  void provePointwiseOrReductionOp(Expr* e);

  using BackwardVisitor::handle;

  void handle(Expr* e) override;

  void handle(UnaryOp* uop) override {
    provePointwiseOrReductionOp(uop);
  }

  void handle(BinaryOp* bop) override {
    provePointwiseOrReductionOp(bop);
  }

  void handle(TernaryOp* top) override {
    provePointwiseOrReductionOp(top);
  }

  void handle(ReductionOp* op) override {
    provePointwiseOrReductionOp(op);
  }

  void handle(BroadcastOp* op) override;

  bool mapAllConsumers(const DomainKey& producer_key);

  void handle(TensorView* tv) override;

  std::unordered_map<IterDomain*, IterDomain*> map(
      const TensorDomain* producer, const TensorDomain* consumer,
      const std::unordered_set<const IterDomain*>& root_dims_to_map,
      bool producer_to_consumer) const;

 private:
  DisjointSet<DomainKey, DomainKeyHash> eq_set_;
  std::unordered_map<DomainKey, std::unordered_set<DomainKey, DomainKeyHash>, DomainKeyHash> pending_map_;
  std::unordered_set<Expr*> visited_;
  FindIncompatibleDomains incompatible_domains_;
};

inline std::ostream& operator<<(std::ostream& os, const RootDomainMap& map) {
  return map.print(os);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
