#include <torch/csrc/jit/codegen/cuda/lower_compute_at_map.h>

#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
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
    if (mapping_mode_ == MappingMode::PARALLEL) {
      if (id0->isParallelized() && id1->isParallelized()) {
        // Both are parallelized, make sure they're the same, set entry for
        // parallel map
        TORCH_INTERNAL_ASSERT(id0->getParallelType() == id1->getParallelType());
        parallel_type_map_[new_set] = id0->getParallelType();
      } else if (id0->isParallelized() || id1->isParallelized()) {
        // Only one is parallelized, set entry for parallel map
        parallel_type_map_[new_set] = id0->isParallelized()
            ? id0->getParallelType()
            : id1->getParallelType();
      }
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

    // Update parallel type map
    if (mapping_mode_ == MappingMode::PARALLEL) {
      auto parallel_type_0_it = parallel_type_map_.find(set0_ptr);
      auto parallel_type_1_it = parallel_type_map_.find(set1_ptr);
      if (parallel_type_0_it != parallel_type_map_.end() &&
          parallel_type_1_it != parallel_type_map_.end()) {
        // If both sets had a parallel type associated with them, make sure they
        // are the same
        TORCH_INTERNAL_ASSERT(
            parallel_type_0_it->second == parallel_type_1_it->second);
      } else if (parallel_type_1_it != parallel_type_map_.end()) {
        // Set 1 has a parallel type, set 0 does not, set parallel entry
        parallel_type_map_[set0_ptr] = parallel_type_1_it->second;
      } // Else set 0 already has the right parallel type set in the map, if at
        // all

      // Remove set1 from the parallel type map as it shouldn't exist anymore
      parallel_type_map_.erase(set1_ptr);
    }

  } else if (set_it_0 != disjoint_iter_set_maps_.end()) {
    // set0 already exists but set1 does not, use set0
    auto set0 = set_it_0->second;
    set0->push_back(id1);
    disjoint_iter_set_maps_[id1] = set0;

    // Update parallel type map
    if (mapping_mode_ == MappingMode::PARALLEL) {
      auto parallel_type_0_it = parallel_type_map_.find(set0);
      if (parallel_type_0_it != parallel_type_map_.end() &&
          id1->isParallelized()) {
        // set0 has a parallel type already and id1 has a parallel type, make
        // sure they match. No need to update map
        TORCH_INTERNAL_ASSERT(
            parallel_type_0_it->second == id1->getParallelType());
      } else if (
          parallel_type_0_it == parallel_type_map_.end() &&
          id1->isParallelized()) {
        // Set parallel type of set0 as the newly added id1 is parallel
        parallel_type_map_[set0] = id1->getParallelType();
      }
    }

  } else {
    // set1 already exists but set0 does not, use set1
    auto set1 = set_it_1->second;
    set1->push_back(id0);
    disjoint_iter_set_maps_[id0] = set1;

    // Update parallel type map
    if (mapping_mode_ == MappingMode::PARALLEL) {
      auto parallel_type_1_it = parallel_type_map_.find(set1);
      if (parallel_type_1_it != parallel_type_map_.end() &&
          id0->isParallelized()) {
        // Set1 already has a parallel type and id0 has a parallel type make
        // sure they match
        TORCH_INTERNAL_ASSERT(
            parallel_type_1_it->second == id0->getParallelType());
      } else if (
          parallel_type_1_it == parallel_type_map_.end() &&
          id0->isParallelized()) {
        // Set1 doesn't have a parallel type but the newly added id0 has a
        // parallel type
        parallel_type_map_[set1] = id0->getParallelType();
      }
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
    // Iteration domains that mapped from producers into the consumer that were
    // to the left of respective producer->getThisComputeAtPos in the producers
    std::unordered_set<IterDomain*> mapped_c_ids_left_of_ca;

    auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());

    for (auto p_tv : tv_inputs) {
      // If outside computeAt axis, we don't want to directly map
      // consumer/producer as their thread mappings could change as long as it's
      // across shared/global memory.
      // TODO: Make better consistency checks allowing this when not acros
      // shared/global memory and looking for consistency.

      // Mark axes outside compute at point for parallel type tracking
      std::unordered_set<IterDomain*> right_of_ca_point;
      if (mapping_mode_ == MappingMode::PARALLEL &&
          p_tv->getThisComputeAtAxis() < p_tv->nDims()) {
        right_of_ca_point.insert(
            p_tv->domain()->domain().begin() + p_tv->getThisComputeAtAxis(),
            p_tv->domain()->domain().end());
      }
      // if this is a producer tv, (i.e. not a terminating output tv), then
      // produce at is the same as this compute at position. Loop mode does
      // its own thing, see below in this function.
      if (mapping_mode_ != MappingMode::LOOP) {
        produce_at_map_[p_tv] = p_tv->getThisComputeAtAxis();
      }

      auto c2p_root_map =
          PairwiseRootDomainMap(p_tv, c_tv)
              .mapConsumerToProducer(c_tv->domain(), p_tv->domain());

      // Look for matching ID transformations in producer and consumer, replay
      // producer as consumer. We want to play producer as consumer instead of
      // the other way around since consumer may have some broadcasted axes
      // producer doesn't have merged into loops producer may use. If we did
      // consumer as producer we wouldn't have this information in the mapping.
      // If we're using this map for indexing, we do not want to propagate
      // broadcast mismatches. If we're using it to identify loop nests, we do
      // want to propagate mismatches.
      BestEffortReplay replay_PasC(
          p_tv->domain()->domain(),
          c_tv->domain()->domain(),
          c2p_root_map,
          mapping_mode_ == MappingMode::LOOP);

      auto c2p_map = replay_PasC.getReplay();

      // Map the entire replay map
      // Also reverse the map, as we use p2c_map to find this computeAt position
      // in consumer. This could be removed if we changed computeAt of
      // TensorViews
      std::unordered_map<IterDomain*, IterDomain*> p2c_map;
      for (auto entry : c2p_map) {
        auto c_id = entry.first;
        auto p_id = entry.second;
        // If outside CA point and we're creating parallel map, do not map the
        // axis
        if (mapping_mode_ == MappingMode::PARALLEL &&
            right_of_ca_point.find(p_id) != right_of_ca_point.end()) {
          continue;
        }
        // Map the id's together
        map_ids(p_id, c_id);
        p2c_map[p_id] = c_id;
      }

      // Track which id's in the consumer are mapped to from within the producer
      // compute at position
      for (size_t p_id_i = 0; p_id_i < p_tv->getThisComputeAtAxis(); p_id_i++) {
        auto p_id = p_tv->axis(p_id_i);
        auto c_id_it = p2c_map.find(p_id);
        if (c_id_it != p2c_map.end()) {
          auto c_id = c_id_it->second;
          mapped_c_ids_left_of_ca.emplace(c_id);
        }
      }
    }

    // For expression sorting we want to know the maximum iteration domain that
    // we might have to map with producers. Consider a simple consumer with this
    // compute at position as 1, but a producer who's compute at position maps
    // to the consumers position 2, we need to exprSort starting with both
    // positions in the consumer available to map to neighbors. We produce this
    // special produce_at_map in loop mode. Pos is like compute at position, one
    // above last thing that mapped.
    int max_mapped_id_pos = 0;
    bool terminating_output = c_tv->isFusionOutput() && c_tv->uses().empty();
    if (terminating_output || mapping_mode_ == MappingMode::LOOP) {
      for (size_t c_i = 0; c_i < c_tv->nDims(); c_i++) {
        if (mapped_c_ids_left_of_ca.find(c_tv->axis(c_i)) !=
            mapped_c_ids_left_of_ca.end()) {
          max_mapped_id_pos = c_i + 1;
        }
      }
      produce_at_map_[c_tv] =
          std::max(max_mapped_id_pos, (int)c_tv->getThisComputeAtAxis());
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
      // Uncertain if the following is needed, Maybe it makes sense to not
      // create loop nests based on rfactor axes if we can avoid it
      // // Don't use rfactor iter domains if not required.
      // if(id->isRFactorProduct() && id->definition() == nullptr){
      //   continue;
      // }
      int pos = n_concrete_ids_.at(id);
      if (pos > max_pos) {
        max_pos = pos;
        concrete_id = id;
      }
    }
    // Uncertain if the following is needed, Maybe it makes sense to not
    // create loop nests based on rfactor axes if we can avoid it
    // if(concrete_id == nullptr){
    //   // Same thing as above, but consider non-input rfactor iter domains
    //   for (auto id : *set) {
    //     int pos = n_concrete_ids_.at(id);
    //     if (pos > max_pos) {
    //       max_pos = pos;
    //       concrete_id = id;
    //     }
    //   }
    // }
    TORCH_INTERNAL_ASSERT(
        concrete_id != nullptr, "Could not concretize an IterDomain set.");

    // If parallel mode, parallelize the the concrete id
    // TODO: Would be good to simply keep a parallelization map and make lookups
    // to it through lowering.
    if (mapping_mode_ == MappingMode::PARALLEL) {
      auto parallel_map_it = parallel_type_map_.find(set);
      if (parallel_map_it != parallel_type_map_.end()) {
        concrete_id->parallelize(parallel_map_it->second);
      }
    }

    for (auto id : *set) {
      concrete_id_map_[id] = concrete_id;
    }
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

  // Make sure we have all IterDomains that could be used to generate a ForLoop
  for (auto expr : fusion->exprs()) {
    if (!expr->outputs()[0]->isA<TensorView>()) {
      continue;
    }

    auto tv_outputs = ir_utils::filterByType<TensorView>(expr->outputs());

    for (auto out : tv_outputs) {
      for (auto entry : out->domain()->domain()) {
        kir_2_fusion[gpu_lower->lowerValue(entry)->as<kir::IterDomain>()] =
            entry;
      }
    }
  }
}

bool ComputeAtMap::areMapped(IterDomain* id0, IterDomain* id1) const {
  if (id0 == id1)
    return true;
  auto set0_it = disjoint_iter_set_maps_.find(id0);
  auto set1_it = disjoint_iter_set_maps_.find(id1);
  if (set0_it == disjoint_iter_set_maps_.end() ||
      set1_it == disjoint_iter_set_maps_.end()) {
    return false;
  }
  return (set0_it->second.get() == set1_it->second.get());
}

bool ComputeAtMap::areMapped(kir::IterDomain* id0, kir::IterDomain* id1) const {
  if (id0 == id1)
    return true;
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
  TORCH_INTERNAL_ASSERT(
      mapping_mode_ == MappingMode::PARALLEL,
      "Need to restrict mode to parallel mode to use this function.");
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
  TORCH_INTERNAL_ASSERT(
      mapping_mode_ == MappingMode::PARALLEL,
      "Need to restrict mode to parallel mode to use this function.");
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

// std::unordered_map<IterDomain*, IterDomain*> ComputeAtMap::mapFromTo(
//     const std::vector<IterDomain*>& from,
//     const std::vector<IterDomain*>& to) const {
//   std::unordered_map<IterDomain*, IterDomain*> concrete_to_from;
//   for (auto from_id : from) {
//     auto concrete_id = getConcreteMappedID(from_id);
//     concrete_to_from[concrete_id] = from_id;
//   }

//   std::unordered_map<IterDomain*, IterDomain*> from_to_to;
//   for(auto to_id : to){
//     auto concrete_id = getConcreteMappedID(to_id);

//     auto from_it = concrete_to_from.find(concrete_id);
//     if(from_it == concrete_to_from.end()){
//       continue;
//     }

//     from_to_to[from_it->second] = to_id;
//   }

//   return from_to_to;
// }

// std::unordered_map<kir::IterDomain*, kir::IterDomain*>
// ComputeAtMap::mapFromTo(
//     const std::vector<kir::IterDomain*>& from,
//     const std::vector<kir::IterDomain*>& to) const {
//   std::unordered_map<kir::IterDomain*, kir::IterDomain*> concrete_to_from;
//   for (auto from_id : from) {
//     auto concrete_id = getConcreteMappedID(from_id);
//     concrete_to_from[concrete_id] = from_id;
//   }

//   std::unordered_map<kir::IterDomain*, kir::IterDomain*> from_to_to;
//   for (auto to_id : to) {
//     auto concrete_id = getConcreteMappedID(to_id);

//     auto from_it = concrete_to_from.find(concrete_id);
//     if (from_it == concrete_to_from.end()) {
//       continue;
//     }

//     from_to_to[from_it->second] = to_id;
//   }

//   return from_to_to;
// }

IterDomain* ComputeAtMap::toFusion(kir::IterDomain* kir) const {
  auto kir_2_fusion_it = kir_2_fusion.find(kir);
  TORCH_INTERNAL_ASSERT(
      kir_2_fusion_it != kir_2_fusion.end(),
      "Kernel ir is not guarneteed to be reversible into fusion ir, could not find fusion entry.");
  return kir_2_fusion_it->second;
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
    if (mapping_mode_ == MappingMode::PARALLEL) {
      if (parallel_type_map_.find(disjoint_set) != parallel_type_map_.end()) {
        ss << "  -> " << parallel_type_map_.at(disjoint_set);
      } else {
        ss << "  -> " << ParallelType::Serial;
      }
    }
    ss << "\n";
  }
  return ss.str();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch