#include <torch/csrc/jit/codegen/cuda/lower_compute_at_map.h>

#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace {

// Class to figure out how many non-broadcast axes were used to produce an iter
// domain. This is important for figuring out what the correct broadcasted
// extent is of an iteration domain
class ConcreteInputCounter : public IterVisitor {
 public:
  // Returns number of non-braodcast non-reduction iteration domains used to
  // generate the iteration domains in provided target domain.
  static std::unordered_map<IterDomain*, int> produceCounts(
      const std::vector<IterDomain*>& domain) {
    std::unordered_map<IterDomain*, int> count_map;
    if (domain.empty()) {
      return count_map;
    }
    ConcreteInputCounter counter(domain);
    std::transform(
        counter.concrete_domain_set_.begin(),
        counter.concrete_domain_set_.end(),
        std::inserter(count_map, count_map.begin()),
        [](std::pair<IterDomain*, std::unordered_set<IterDomain*>> entry) {
          return std::make_pair(entry.first, entry.second.size());
        });
    // Inputs may be root domains which wouldn't have any entries if no exprs
    // were traversed, so manually insert their count
    for (auto id : domain) {
      if (count_map.find(id) == count_map.end()) {
        count_map[id] = id->isBroadcast() ? 0 : 1;
      }
    }
    return count_map;
  }

  ConcreteInputCounter(const std::vector<IterDomain*>& domain_) {
    traverseFrom(
        domain_[0]->fusion(),
        std::vector<Val*>(domain_.begin(), domain_.end()));
  }

 private:
  std::unordered_set<IterDomain*>& getEntry(IterDomain* id) {
    auto concrete_set_it = concrete_domain_set_.find(id);
    if (concrete_set_it == concrete_domain_set_.end()) {
      concrete_set_it =
          concrete_domain_set_
              .emplace(std::make_pair(id, std::unordered_set<IterDomain*>()))
              .first;
      if (!id->isBroadcast()) {
        concrete_set_it->second.emplace(id);
      }
    }

    return concrete_set_it->second;
  }

  void handle(Expr* expr) override {
    // If we end up moving swizzle to an Expr it would be identity here, instead
    // of outputs being a function of all inputs
    switch (expr->getExprType().value()) {
      case (ExprType::Split):
      case (ExprType::Merge):
        break;
      default:
        TORCH_INTERNAL_ASSERT(
            false, "Invalid expr type found in transform traversal.");
    }

    std::unordered_set<IterDomain*> resulting_set;
    for (auto input_id : ir_utils::filterByType<IterDomain>(expr->inputs())) {
      auto input_entry = getEntry(input_id);
      resulting_set.insert(input_entry.begin(), input_entry.end());
    }
    for (auto output_id : ir_utils::filterByType<IterDomain>(expr->outputs())) {
      concrete_domain_set_.emplace(std::make_pair(output_id, resulting_set));
    }
  }

  std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>>
      concrete_domain_set_;
};

template <class T>
std::deque<T*> deduplicateDeque(std::deque<T*>& deque) {
  std::unordered_set<T*> used;
  std::deque<T*> deduped;
  for (auto entry : deque) {
    if (used.find(entry) == used.end()) {
      deduped.push_back(entry);
      used.emplace(entry);
    }
  }
  return deduped;
}

} // namespace

void ComputeAtMap::map_ids(IterDomain* id0, IterDomain* id1) {
  auto set_it_0 = disjoint_iter_set_maps_.find(id0);
  auto set_it_1 = disjoint_iter_set_maps_.find(id1);
  if (set_it_0 == disjoint_iter_set_maps_.end() &&
      set_it_1 == disjoint_iter_set_maps_.end()) {
    // Neither iter domain has been mapped, so make a new disjoint set
    auto new_set = std::make_shared<std::deque<IterDomain*>>();
    new_set.get()->push_back(id0);
    new_set.get()->push_back(id1);
    disjoint_iter_set_maps_.emplace(std::make_pair(id0, new_set));
    disjoint_iter_set_maps_.emplace(std::make_pair(id1, new_set));
    disjoint_iter_sets_.push_back(new_set);

    // Update parallel type map
    if (id0->isParallelized() && id1->isParallelized()) {
      TORCH_INTERNAL_ASSERT(id0->getParallelType() == id1->getParallelType());
      parallel_type_map_[new_set] = id0->getParallelType();
    } else if (id0->isParallelized() || id1->isParallelized()) {
      parallel_type_map_[new_set] = id0->isParallelized()
          ? id0->getParallelType()
          : id1->getParallelType();
    }
  } else if (
      set_it_0 != disjoint_iter_set_maps_.end() &&
      set_it_1 != disjoint_iter_set_maps_.end()) {
    // Both iter domains have been mapped, so join their sets together
    auto set0_ptr = set_it_0->second;
    auto set1_ptr = set_it_1->second;

    // If the sets are already the same, do nothing
    if (set0_ptr == set1_ptr) {
      return;
    }

    // Place everything in set1 into set0 and remap all ID's in set1 to set0
    auto& set1 = *set1_ptr;
    for (auto id : set1) {
      set0_ptr->push_back(id);
      disjoint_iter_set_maps_[id] = set0_ptr;
    }

    // If both sets had a parallel type associated with them, make sure they
    // are the same
    auto parallel_type_0_it = parallel_type_map_.find(set0_ptr);
    auto parallel_type_1_it = parallel_type_map_.find(set1_ptr);
    if (parallel_type_0_it != parallel_type_map_.end() &&
        parallel_type_1_it != parallel_type_map_.end()) {
      TORCH_INTERNAL_ASSERT(
          parallel_type_0_it->second == parallel_type_1_it->second);
    }

    // Remove set1 from the parallel type map as it shouldn't exist anymore
    parallel_type_map_.erase(set1_ptr);

  } else if (set_it_0 != disjoint_iter_set_maps_.end()) {
    // set0 already exists but set1 does not, use set0
    auto set0 = set_it_0->second;
    set0->push_back(id1);
    disjoint_iter_set_maps_[id1] = set0;

    auto parallel_type_0_it = parallel_type_map_.find(set0);
    if (parallel_type_0_it != parallel_type_map_.end() &&
        id1->isParallelized()) {
      // If set0 had a parallel type and id1 has a parallel type make surue they
      // match
      TORCH_INTERNAL_ASSERT(
          parallel_type_0_it->second == id1->getParallelType());
    } else if (
        parallel_type_0_it == parallel_type_map_.end() &&
        id1->isParallelized()) {
      // Set parallel type of set0 as the newly added id1 if id1 is parallel
      parallel_type_map_[set0] = id1->getParallelType();
    }

  } else {
    // set1 already exists but set0 does not, use set1
    auto set1 = set_it_1->second;
    set1->push_back(id0);
    disjoint_iter_set_maps_[id0] = set1;

    auto parallel_type_1_it = parallel_type_map_.find(set1);
    if (parallel_type_1_it != parallel_type_map_.end() &&
        id0->isParallelized()) {
      // If set1 had a parallel type and id0 has a parallel type make surue they
      // match
      TORCH_INTERNAL_ASSERT(
          parallel_type_1_it->second == id0->getParallelType());
    } else if (
        parallel_type_1_it == parallel_type_map_.end() &&
        id0->isParallelized()) {
      // Set parallel type of set1 as the newly added id1 if id1 is parallel
      parallel_type_map_[set1] = id0->getParallelType();
    }
  }
}

void ComputeAtMap::build() {
  Fusion* fusion = FusionGuard::getCurFusion();
  TORCH_INTERNAL_ASSERT(fusion != nullptr);

  // Consumers can only show up once in an expression, keep track of all of them
  std::vector<TensorView*> consumer_tvs;

  for (auto expr : fusion->exprs()) {
    if (!expr->outputs()[0]->isA<TensorView>()) {
      continue;
    }

    // TODO: Do we need to map all output tensors if more than one, or just the
    // first? For indexing we may need all of them mapped. Do we expect all
    // other outputs to be replayed as the first or should we do it?
    auto c_tv = expr->outputs()[0]->as<TensorView>();
    consumer_tvs.push_back(c_tv);
    int c_max_ca_pos = 0;
    bool terminating_output = c_tv->isFusionOutput() && c_tv->uses().empty();

    auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());

    for (auto p_tv : tv_inputs) {
      // if this is a producer tv, (i.e. not a terminating output tv), then
      // produce at is the same as this compute at position
      produce_at_map_[p_tv] = p_tv->getThisComputeAtAxis();

      auto p2c_root_map =
          PairwiseRootDomainMap(p_tv, c_tv)
              .mapProducerToConsumer(p_tv->domain(), c_tv->domain());

      // Look for matching ID transformations in producer and consumer...
      // TODO: Map all of BestEffortReplay, will be needed for indexing
      BestEffortReplay replay(
          c_tv->domain()->domain(), p_tv->domain()->domain(), p2c_root_map);

      auto p2c_map = replay.getReplay();

      // Map the entire replay map:
      for (auto entry : p2c_map) {
        auto p_id = entry.first;
        auto c_id = entry.second;
        // Map the id's together
        map_ids(p_id, c_id);
      }

      for (size_t p_id_i = 0; p_id_i < p_tv->getThisComputeAtAxis(); p_id_i++) {
        auto p_id = p_tv->axis(p_id_i);

        auto c_id_it = p2c_map.find(p_id);

        if (c_id_it != p2c_map.end()) {
          auto c_id = c_id_it->second;

          if (terminating_output) {
            int ca_pos = (int)std::distance(
                             c_tv->domain()->domain().begin(),
                             std::find(
                                 c_tv->domain()->domain().begin(),
                                 c_tv->domain()->domain().end(),
                                 c_id))
                // Add one since this is CA position, not the axis position.
                + 1;
            c_max_ca_pos = std::max(c_max_ca_pos, ca_pos);
          }
        }
      }
    }
    if (terminating_output) {
      auto produce_at_it = produce_at_map_.find(c_tv);
      if (produce_at_it == produce_at_map_.end()) {
        produce_at_map_[c_tv] = c_max_ca_pos;
      }
    }
  }

  std::unordered_set<std::shared_ptr<std::deque<IterDomain*>>> active_sets;
  // Populate disjoint_iter_sets_ as they are all computed now
  for (auto iter_set_map : disjoint_iter_set_maps_) {
    active_sets.emplace(iter_set_map.second);
  }

  auto disjoint_iter_set_end = std::remove_if(
      disjoint_iter_sets_.begin(),
      disjoint_iter_sets_.end(),
      [&](std::shared_ptr<std::deque<IterDomain*>>& set) {
        return active_sets.find(set) == active_sets.end();
      });

  disjoint_iter_sets_ = std::deque<std::shared_ptr<std::deque<IterDomain*>>>(
      disjoint_iter_sets_.begin(), disjoint_iter_set_end);

  // deduplicate iter domain entries in each set
  for (auto iter_set : disjoint_iter_sets_) {
    *iter_set = deduplicateDeque(*iter_set);
  }

  // For each IterDomain set we will track how many concrete root domains were
  // used to generate the IterDomain. Used to populate conrete_id_map
  std::unordered_map<IterDomain*, int> n_concrete_ids_;

  for (auto c_tv : consumer_tvs) {
    auto counts = ConcreteInputCounter::produceCounts(c_tv->domain()->domain());
    n_concrete_ids_.insert(counts.begin(), counts.end());
  }

  for (auto inp_tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    auto counts =
        ConcreteInputCounter::produceCounts(inp_tv->domain()->domain());
    n_concrete_ids_.insert(counts.begin(), counts.end());
  }

  // Populate concrete id map
  for (auto set : disjoint_iter_sets_) {
    int max_pos = -1;
    IterDomain* concrete_id = nullptr;
    for (auto id : *set) {
      int pos = n_concrete_ids_.at(id);
      if (pos > max_pos) {
        max_pos = pos;
        concrete_id = id;
      }
    }
    TORCH_INTERNAL_ASSERT(
        concrete_id != nullptr, "Could not concretize an IterDomain set.");

    for (auto id : *set) {
      concrete_id_map_[id] = concrete_id;
    }
    concrete_id->parallelize(getMappedParallelType(concrete_id));
  }

  // ===== CONVERSION TO KERNEL IR =========

  // Convert everything to lowered structures (kernel ir), as we will use
  // this class frequently during lowering.

  auto gpu_lower = GpuLower::current();

  std::unordered_map<
      std::shared_ptr<std::deque<IterDomain*>>,
      std::shared_ptr<std::deque<kir::IterDomain*>>>
      disjoint_set_2_kir;

  for (auto disjoint_iter_set : disjoint_iter_set_maps_) {
    auto fusion_set = disjoint_iter_set.second;
    auto kir_set_it = disjoint_set_2_kir.find(fusion_set);
    if (kir_set_it == disjoint_set_2_kir.end()) {
      auto kir_set = std::make_shared<std::deque<kir::IterDomain*>>();
      std::transform(
          fusion_set->begin(),
          fusion_set->end(),
          std::inserter(*kir_set, kir_set->begin()),
          [&gpu_lower](IterDomain* id) {
            return gpu_lower->lowerValue(id)->as<kir::IterDomain>();
          });
      disjoint_set_2_kir.emplace(std::make_pair(fusion_set, kir_set));
      kir_disjoint_iter_set_maps_.emplace(std::make_pair(
          gpu_lower->lowerValue(disjoint_iter_set.first)->as<kir::IterDomain>(),
          kir_set));
    } else {
      kir_disjoint_iter_set_maps_.emplace(std::make_pair(
          gpu_lower->lowerValue(disjoint_iter_set.first)->as<kir::IterDomain>(),
          kir_set_it->second));
    }
  }

  for (auto entry : parallel_type_map_) {
    auto fusion_set = entry.first;
    auto kir_set = disjoint_set_2_kir.at(fusion_set);
    kir_parallel_type_map_.emplace(std::make_pair(kir_set, entry.second));
  }

  for (auto entry : concrete_id_map_) {
    kir_concrete_id_map_.emplace(std::make_pair(
        gpu_lower->lowerValue(entry.first)->as<kir::IterDomain>(),
        gpu_lower->lowerValue(entry.second)->as<kir::IterDomain>()));
  }

  for (auto entry : disjoint_iter_set_maps_) {
    kir_2_fusion[gpu_lower->lowerValue(entry.first)->as<kir::IterDomain>()] =
        entry.first;
  }
}

bool ComputeAtMap::areMapped(IterDomain* id0, IterDomain* id1) const {
  auto set0_it = disjoint_iter_set_maps_.find(id0);
  auto set1_it = disjoint_iter_set_maps_.find(id1);
  if (set0_it == disjoint_iter_set_maps_.end() ||
      set1_it == disjoint_iter_set_maps_.end()) {
    return false;
  }
  return (set0_it->second.get() == set1_it->second.get());
}

bool ComputeAtMap::areMapped(kir::IterDomain* id0, kir::IterDomain* id1) const {
  auto set0_it = kir_disjoint_iter_set_maps_.find(id0);
  auto set1_it = kir_disjoint_iter_set_maps_.find(id1);
  if (set0_it == kir_disjoint_iter_set_maps_.end() ||
      set1_it == kir_disjoint_iter_set_maps_.end()) {
    return false;
  }
  return (set0_it->second.get() == set1_it->second.get());
}

IterDomain* ComputeAtMap::getConcreteMappedID(IterDomain* id) const {
  auto it = concrete_id_map_.find(id);
  if (it != concrete_id_map_.end()) {
    return it->second;
  }
  return id;
}

kir::IterDomain* ComputeAtMap::getConcreteMappedID(kir::IterDomain* id) const {
  auto it = kir_concrete_id_map_.find(id);
  if (it != kir_concrete_id_map_.end()) {
    return it->second;
  }
  return id;
}

ParallelType ComputeAtMap::getMappedParallelType(IterDomain* id) const {
  auto disjoint_set_it = disjoint_iter_set_maps_.find(id);
  if (disjoint_set_it == disjoint_iter_set_maps_.end()) {
    return id->getParallelType();
  }
  auto parallel_type_it = parallel_type_map_.find(disjoint_set_it->second);
  if (parallel_type_it == parallel_type_map_.end()) {
    return id->getParallelType();
  }
  return parallel_type_it->second;
}

ParallelType ComputeAtMap::getMappedParallelType(kir::IterDomain* id) const {
  auto disjoint_set_it = kir_disjoint_iter_set_maps_.find(id);
  if (disjoint_set_it == kir_disjoint_iter_set_maps_.end()) {
    return id->parallelType();
  }
  auto parallel_type_it = kir_parallel_type_map_.find(disjoint_set_it->second);
  if (parallel_type_it == kir_parallel_type_map_.end()) {
    return id->parallelType();
  }
  return parallel_type_it->second;
}

std::string ComputeAtMap::toString() {
  std::stringstream ss;

  ss << "produce_at_map_{\n";
  for (const auto& entry : produce_at_map_) {
    ss << "  " << entry.first << " -> " << entry.second << "\n";
  }
  ss << "} end produce_at_map_\n";

  // TODO: Fix this iteration to loop over our deque
  std::unordered_set<std::shared_ptr<std::deque<IterDomain*>>> disjoint_sets;

  for (auto entry : disjoint_iter_set_maps_) {
    disjoint_sets.emplace(entry.second);
  }

  for (const auto& disjoint_set : disjoint_sets) {
    ss << "  disjoint_set{ ";
    for (auto it = disjoint_set->begin(); it != disjoint_set->end(); it++) {
      if (it != disjoint_set->begin()) {
        ss << ", ";
      }
      ss << (*it);
    }
    ss << " }";
    if (parallel_type_map_.find(disjoint_set) != parallel_type_map_.end()) {
      ss << "  -> " << parallel_type_map_.at(disjoint_set) << "\n";
    } else {
      ss << "  -> " << ParallelType::Serial << "\n";
    }
  }
  return ss.str();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch