#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

bool hasMatchingDomains(const std::vector<DomainKey>& domains,
                        const DisjointSet<DomainKey, DomainKeyHash>& eq_set) {
  for (const auto& key: domains) {
    for (const auto& other_key: domains) {
      if (key == other_key) continue;
      const auto& root = other_key.td()->getRootDomain();
      if (std::any_of(root.begin(), root.end(),
                      [&](const IterDomain* id) {
                        return eq_set.areEquivalent(key, DomainKey(other_key.td(), id));
                      })) {
        return true;
      }
    }
  }
  return false;
}

bool safeToJoin(const std::unordered_set<DomainKey, DomainKeyHash>& domains,
                const DisjointSet<DomainKey, DomainKeyHash>& eq_set,
                const FindIncompatibleDomains& inconsistent_domains) {
  //std::cerr << "safeTojoin?: " << domains << std::endl;
  if (domains.size() <= 1) {
    return true;
  }
  // Filter out equivalent domains
  std::vector<DomainKey> unique_domains;
  for (const auto& domain: domains) {
    if (std::none_of(unique_domains.begin(), unique_domains.end(),
                     [&](const auto& unique_dom) {
                       return eq_set.areEquivalent(domain, unique_dom);
                     })) {
      unique_domains.push_back(domain);
    }
  }
  //std::cerr << "Consumer domains: " << unique_domains << std::endl;
  if (hasMatchingDomains(unique_domains, eq_set)) {
    //std::cerr << "Has matching domains" << std::endl;
    return false;
  }
  if (inconsistent_domains.isReductionOutputMerged(unique_domains, eq_set)) {
    //std::cerr << "Prevented by reduction" << std::endl;
    return false;
  }
  return true;
}

} // namespace

std::ostream& DomainKey::print(std::ostream& os) const {
  std::stringstream ss;
  ss << "{";
  if (td_) {
    ss << td_ << " (root: " << td_->getRootDomain()
       << ", maybe rfactor: " << td_->getMaybeRFactorDomain() << ")";
  } else {
    ss << "null";
  }
  ss << ", ";
  if (id_) {
    ss << id_;
  } else {
    ss << "null";
  }
  ss << "}";
  return os << ss.str();
}

FindIncompatibleDomains::FindIncompatibleDomains() {
  Fusion* fusion = FusionGuard::getCurFusion();
  traverse(fusion);
}

void FindIncompatibleDomains::handle(ReductionOp* op) {
  TensorView* out_tv = op->out()->as<TensorView>();
  std::vector<DomainKey> reduction_keys;
  for (const auto id: out_tv->getMaybeRFactorDomain()) {
    if (id->isReduction()) {
      DomainKey key(out_tv->domain(), id);
      reduction_keys.push_back(key);
      inconsistent_domains_.insert({key, {}});
    }
  }
  auto use_chains = DependencyCheck::getAllUseChains(out_tv);
  for (const auto& chain: use_chains) {
    for (const auto& tv: ir_utils::filterByType<TensorView>(chain)) {
      const auto& root_domain = tv->getRootDomain();
      for (const auto& id: root_domain) {
        DomainKey consumer_key(tv->domain(), id);
        for (const auto& reduction_key: reduction_keys) {
          inconsistent_domains_.at(reduction_key).insert(consumer_key);
        }
      }
    }
  }
}

bool FindIncompatibleDomains::isReductionOutputMerged(
    const std::vector<DomainKey>& consumer_domains,
    const DisjointSet<DomainKey, DomainKeyHash>& eq_set) const {
  for (const auto& kv: inconsistent_domains_) {
    const auto& reducion_domain = kv.first;
    const auto& incompatible_domains = kv.second;
    //std::cerr << "Inconsistent reduction: " << reducion_domain << std::endl;
    DomainKey consumer_domain_with_reduction;
    bool reduction_found = false;
    for (const auto& consumer_domain: consumer_domains) {
      if (eq_set.areEquivalent(reducion_domain, consumer_domain)) {
        consumer_domain_with_reduction = consumer_domain;
        //std::cerr << "reduction found in: " << consumer_domain << std::endl;
        reduction_found = true;
        break;
      }
    }
    if (!reduction_found) {
      //std::cerr << "reduction not found in consumers\n";
      continue;
    }
    // Make sure no incompatible domains will be merged with the reduction domain.
    for (const auto& consumer_domain: consumer_domains) {
      if (consumer_domain == consumer_domain_with_reduction) {
        continue;
      }
      if (std::any_of(incompatible_domains.begin(), incompatible_domains.end(),
                      [&](const DomainKey& incompatible_domain) {
                        return eq_set.areEquivalent(consumer_domain, incompatible_domain);
                      })) {
        // Merging them will result in inconsistency
        //std::cerr << "incompatible consumer found: " << consumer_domain << std::endl;
        return true;
      }
    }
  }
  return false;
}

RootDomainMap::RootDomainMap() {
  Fusion* fusion = FusionGuard::getCurFusion();
  TORCH_INTERNAL_ASSERT(fusion != nullptr);
  // std::cerr << "Traversing for root domain map\n";
  std::vector<Val*> leaves;
  for (Val* val : fusion->deterministic_vals()) {
    if (!fusion->used(val)) {
      leaves.push_back(val);
    }
    // Register all domain keys. This simplifies the matching analysis.
    if (val->isA<TensorView>()) {
      auto td = val->as<TensorView>()->domain();
      for (const auto& id: td->getRootDomain()) {
        eq_set_.insert(DomainKey(td, id));
      }
      if (td->hasRFactor()) {
        for (const auto& id: td->getRFactorDomain()) {
          eq_set_.insert(DomainKey(td, id));
        }
      }
    }
  }
  traverseFrom(fusion, leaves, false);
  //std::cerr << "Traversal done\n";
  //print(std::cerr);
  if (!pending_map_.empty()) {
    std::stringstream ss;
    ss << "pending map:\n";
    for (auto& kv: pending_map_) {
      ss << "\t" << kv.first << "\n";
      for (auto& dk: kv.second) {
        ss << "\t\t" << dk << "\n";
      }
    }
    std::cerr << ss.str();
  }
  TORCH_INTERNAL_ASSERT(pending_map_.empty());
}

bool RootDomainMap::canMap(const TensorDomain* td_a, const IterDomain* id_a,
                           const TensorDomain* td_b, const IterDomain* id_b) const {
  TORCH_INTERNAL_ASSERT(id_a->getOrigin() == nullptr || id_a->isRFactorProduct(),
                        "Non-root domain is not supproted: ", id_a);
  TORCH_INTERNAL_ASSERT(id_b->getOrigin() == nullptr || id_b->isRFactorProduct(),
                        "Non-root domain is not supproted: ", id_b);
  DomainKey key_a(td_a, id_a);
  DomainKey key_b(td_b, id_b);
  return eq_set_.areEquivalent(key_a, key_b);
}

void RootDomainMap::attemptToProveId(const TensorDomain* producer_td, const IterDomain* producer_id,
                                     const TensorDomain* consumer_td, const IterDomain* consumer_id) {
  DomainKey producer_key(producer_td, producer_id);
  DomainKey consumer_key(consumer_td, consumer_id);
  auto it = pending_map_.find(producer_key);
  if (it == pending_map_.end()) {
    //std::cerr << "Adding a new pending map set for " << producer_key << std::endl;
    it = pending_map_.insert({producer_key, {}}).first;
  }
  auto& consumer_set = it->second;
  consumer_set.insert(consumer_key);
}

void RootDomainMap::handle(Expr* e) {
  //std::cerr << "handle: " << e << std::endl;
  // Expr can be visited multiple times.
  if (visited_.find(e) != visited_.end()) {
    return;
  }
  BackwardVisitor::handle(e);
  visited_.insert(e);
}

void RootDomainMap::provePointwiseOrReductionOp(Expr* e) {
  if (e->output(0)->getValType() != ValType::TensorView) {
    return;
  }

  // std::cerr << "Visiting Expr: " << e << std::endl;

  // Broadcast is handled separately, so e should never be BroadcastOp.
  TORCH_INTERNAL_ASSERT(e->getExprType() != ExprType::BroadcastOp);

  TORCH_INTERNAL_ASSERT(e->outputs().size() == 1);
  const TensorView* out_tv = e->output(0)->as<TensorView>();
  const TensorDomain* out_td = out_tv->domain();
  const auto& out_root = out_td->getRootDomain();

  // Record equalities from output to all the inputs
  // ignores un-concretizable broadcasts
  for (auto* i : ir_utils::filterByType<TensorView>(e->inputs())) {
    const TensorDomain* in_td = i->domain();
    std::vector<IterDomain*> in_root =
        TensorDomain::noReductions(i->getMaybeRFactorDomain());
    TORCH_INTERNAL_ASSERT(in_root.size() == out_root.size());
    for (size_t it = 0; it < in_root.size(); it++) {
      attemptToProveId(in_td, in_root[it], out_td, out_root[it]);
    }
  }
}

void RootDomainMap::handle(BroadcastOp* op) {
  //std::cerr << "BroadcastOp: " << op << std::endl;
  const TensorDomain* in_td = op->in()->as<TensorView>()->domain();
  const TensorDomain* out_td = op->out()->as<TensorView>()->domain();
  const auto in_root = TensorDomain::noReductions(in_td->getRootDomain());
  const auto& out_root = out_td->getRootDomain();
  const auto& bcast_dim_flags = op->getBroadcastDimFlags();
  TORCH_INTERNAL_ASSERT(out_root.size() == bcast_dim_flags.size(),
                        "dim flags: ", bcast_dim_flags,
                        ", out root: ", out_root);
  auto in_it = in_root.begin();
  auto out_it = out_root.begin();
  while (in_it != in_root.end() && out_it != out_root.end()) {
    if (bcast_dim_flags.at(std::distance(out_root.begin(), out_it))) {
      // new broadcast dim. No matching dimension in the input
      // tensor.
      ++out_it;
      continue;
    }
    attemptToProveId(in_td, *in_it, out_td, *out_it);
    ++in_it;
    ++out_it;
  }
  // At this point, the input domain should have been scanned
  // entirely.
  TORCH_INTERNAL_ASSERT(in_it == in_root.end(),
                        "Unmatched domain detected: ", *in_it, " of ", in_td);
  // On the other hand, the output may still have some domains left,
  // and they must be new broadcast domains.
  for (; out_it != out_root.end(); ++out_it) {
    TORCH_INTERNAL_ASSERT(bcast_dim_flags.at(std::distance(out_root.begin(), out_it)),
                          "Unmatched domain detected: ", *out_it, " of ", out_td);
  }
}

bool RootDomainMap::mapAllConsumers(const DomainKey& producer_key) {
  //std::cerr << "mapAllConsumers for : " << producer_key << std::endl;
  auto it = pending_map_.find(producer_key);
  TORCH_INTERNAL_ASSERT(it != pending_map_.end());
  const auto& consumer_set = it->second;
  // All entries in key_set must be equivalent with each other.
  TORCH_INTERNAL_ASSERT(consumer_set.size() > 0);
  bool consistent = safeToJoin(consumer_set, eq_set_, incompatible_domains_);
  if (consistent) {
    for (const auto pending_consumer: consumer_set) {
#if 0
      std::cerr << "Equivalent Ids found: " << producer_key << " == " << pending_consumer
                << std::endl;
#endif
      proveId(producer_key, pending_consumer);
    }
  }
  // This entry should never be used again, so remove it.
  pending_map_.erase(it);
  return consistent;
}

void RootDomainMap::handle(TensorView* tv) {
  if (false) {
    std::stringstream ss;
    ss << tv;
    std::cerr << "Visiting TensorView: " << ss.str() << std::endl;
  }
  const TensorDomain* td = tv->domain();
  const auto root = TensorDomain::noReductions(td->getMaybeRFactorDomain());
  std::stringstream tvstr;
  tvstr << tv;
  for (auto id : root) {
    const DomainKey key(td, id);
    auto it = pending_map_.find(key);
    if (it == pending_map_.end()) {
      // No consumer of this ID found. Just need to add this to
      // eq_set_ but nothing equivalent with it yet
      //std::cerr << "No consumer found: " << key << " of " <<
      //tvstr.str() << std::endl;
      continue;
    }
    if (mapAllConsumers(key)) {
      //std::cerr << "Mapping of " << key << " (" << tvstr.str() << ") succeeded\n";
    }
  }
}

std::unordered_map<IterDomain*, IterDomain*> RootDomainMap::map(
    const TensorDomain* producer, const TensorDomain* consumer,
    const std::unordered_set<const IterDomain*>& root_dims_to_map,
    bool producer_to_consumer) const {
  const auto& producer_root = producer->getMaybeRFactorDomain();
  const auto& consumer_root = consumer->getRootDomain();
  const TensorDomain* src_td = producer_to_consumer ? producer : consumer;
  const TensorDomain* dst_td = producer_to_consumer ? consumer : producer;
  const auto& src_ids = producer_to_consumer ? producer_root : consumer_root;
  const auto& dst_ids = producer_to_consumer ? consumer_root : producer_root;
  std::unordered_map<IterDomain*, IterDomain*> id_map;
  for (const auto& src_id: src_ids) {
    if (root_dims_to_map.find(src_id) == root_dims_to_map.end()) {
      //std::cerr << "Not found in root_dims_to_map: " << src_id << std::endl;
      continue;
    }
    for (const auto& dst_id : dst_ids) {
      //std::cerr << "Examining " << src_id << " == " << dst_id << std::endl;
      if (canMap(src_td, src_id, dst_td, dst_id)) {
        TORCH_INTERNAL_ASSERT(id_map.insert({src_id, dst_id}).second,
                              "Multiple matching ID detected for ", src_id);
      }
    }
  }
  return id_map;
}

std::unordered_map<IterDomain*, IterDomain*> RootDomainMap::mapProducerToConsumer(
    const TensorDomain* producer, const TensorDomain* consumer,
    const std::unordered_set<const IterDomain*>& root_dims_to_map) const {
  return map(producer, consumer, root_dims_to_map, true);
}

std::unordered_map<IterDomain*, IterDomain*> RootDomainMap::mapConsumerToProducer(
    const TensorDomain* consumer, const TensorDomain* producer,
    const std::unordered_set<const IterDomain*>& root_dims_to_map) const {
  return map(producer, consumer, root_dims_to_map, false);
}

std::ostream& RootDomainMap::print(std::ostream& os) const {
  return eq_set_.print(os);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
