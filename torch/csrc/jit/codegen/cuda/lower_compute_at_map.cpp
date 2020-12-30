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
  static std::vector<int> getCount(const std::vector<IterDomain*>& domain) {
    if (domain.empty()) {
      return std::vector<int>();
    }
    std::vector<int> concrete_count(domain.size(), 0);
    ConcreteInputCounter counter(domain);
    for (size_t i = 0; i < domain.size(); i++) {
      auto concrete_input_it = counter.concrete_domain_set_.find(domain[i]);
      if (concrete_input_it != counter.concrete_domain_set_.end()) {
        concrete_count[i] = concrete_input_it->second.size();
      } else {
        // If no entry is found, then the ID is a root domain
        concrete_count[i] = domain[i]->isBroadcast() ? 0 : 1;
      }
    }
    return concrete_count;
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

} // namespace
void ComputeAtMap::build() {
  Fusion* fusion = FusionGuard::getCurFusion();
  TORCH_INTERNAL_ASSERT(fusion != nullptr);

  // Consumers can only show up once in an expression, keep track of all of them
  std::vector<TensorView*> consumer_tvs;

  for (auto expr : fusion->exprs()) {
    if (!expr->outputs()[0]->isA<TensorView>()) {
      continue;
    }

    // TODO: Do we need to map all output tensors, or just the first? For
    // indexing we may need all of them mapped. Do we expect all other outputs
    // to be replayed as the first or should we do it?
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
      BestEffortReplay replay(
          c_tv->domain()->domain(), p_tv->domain()->domain(), p2c_root_map);

      auto p2c_map = replay.getReplay();

      for (size_t p_id_i = 0; p_id_i < p_tv->getThisComputeAtAxis(); p_id_i++) {
        auto p_id = p_tv->axis(p_id_i);
        auto p_id_it = disjoint_iter_sets_.find(p_id);

        if (p_id_it == disjoint_iter_sets_.end()) {
          auto new_set = std::make_shared<std::unordered_set<IterDomain*>>();
          new_set.get()->emplace(p_id);
          p_id_it =
              disjoint_iter_sets_.emplace(std::make_pair(p_id, new_set)).first;
        }

        auto c_id_it = p2c_map.find(p_id);

        if (c_id_it != p2c_map.end()) {
          auto c_id = c_id_it->second;
          auto disjoint_set = p_id_it->second;
          disjoint_set->emplace(c_id);
          disjoint_iter_sets_.emplace(std::make_pair(c_id, disjoint_set));

          if (c_id->isParallelized()) {
            auto parallel_entry_it = parallel_type_map_.find(disjoint_set);
            if (parallel_entry_it != parallel_type_map_.end()) {
              TORCH_INTERNAL_ASSERT(
                  parallel_entry_it->second->getParallelType() ==
                      c_id->getParallelType(),
                  "Compute at iteration domain ",
                  c_id,
                  " in tensor ",
                  c_tv,
                  " maps to another tensor's iter domain ",
                  parallel_entry_it->second,
                  " however parallelization strategies do not match. ",
                  "Only one of these parallel strategies should be set.");
            } else {
              parallel_type_map_[disjoint_set] = c_id;
            }
          }

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

  for (auto c_tv : consumer_tvs) {
    auto counts = ConcreteInputCounter::getCount(c_tv->domain()->domain());
    for (size_t i = 0; i < counts.size(); i++) {
      n_concrete_ids_[c_tv->axis(i)] = counts[i];
    }
  }

  // Convert everything to lowered structures, as we will use this class
  // frequently during lowering.

  auto gpu_lower = GpuLower::current();

  std::unordered_map<
      std::shared_ptr<std::unordered_set<IterDomain*>>,
      std::shared_ptr<std::unordered_set<kir::IterDomain*>>>
      disjoint_set_2_kir;

  for (auto entry : disjoint_iter_sets_) {
    auto fusion_set = entry.second;
    auto kir_set_it = disjoint_set_2_kir.find(fusion_set);
    if (kir_set_it == disjoint_set_2_kir.end()) {
      auto kir_set = std::make_shared<std::unordered_set<kir::IterDomain*>>();
      std::transform(
          fusion_set->begin(),
          fusion_set->end(),
          std::inserter(*kir_set, kir_set->begin()),
          [&gpu_lower](IterDomain* id) {
            return gpu_lower->lowerValue(id)->as<kir::IterDomain>();
          });
      disjoint_set_2_kir.emplace(std::make_pair(fusion_set, kir_set));
      kir_disjoint_iter_sets_.emplace(std::make_pair(
          gpu_lower->lowerValue(entry.first)->as<kir::IterDomain>(), kir_set));
    } else {
      kir_disjoint_iter_sets_.emplace(std::make_pair(
          gpu_lower->lowerValue(entry.first)->as<kir::IterDomain>(),
          kir_set_it->second));
    }
  }

  for (auto entry : parallel_type_map_) {
    auto fusion_set = entry.first;
    auto kir_set = disjoint_set_2_kir.at(fusion_set);
    kir_parallel_type_map_.emplace(std::make_pair(
        kir_set, gpu_lower->lowerValue(entry.second)->as<kir::IterDomain>()));
  }

  for (auto entry : n_concrete_ids_) {
    kir_n_concrete_ids_.emplace(std::make_pair(
        gpu_lower->lowerValue(entry.first)->as<kir::IterDomain>(),
        entry.second));
  }
}

bool ComputeAtMap::areMapped(IterDomain* id0, IterDomain* id1) const {
  auto set0_it = disjoint_iter_sets_.find(id0);
  auto set1_it = disjoint_iter_sets_.find(id1);
  if (set0_it == disjoint_iter_sets_.end() ||
      set1_it == disjoint_iter_sets_.end()) {
    return false;
  }
  return (set0_it->second.get() == set1_it->second.get());
}

bool ComputeAtMap::areMapped(kir::IterDomain* id0, kir::IterDomain* id1) const {
  auto set0_it = kir_disjoint_iter_sets_.find(id0);
  auto set1_it = kir_disjoint_iter_sets_.find(id1);
  if (set0_it == kir_disjoint_iter_sets_.end() ||
      set1_it == kir_disjoint_iter_sets_.end()) {
    return false;
  }
  return (set0_it->second.get() == set1_it->second.get());
}

IterDomain* ComputeAtMap::getConcreteMappedID(IterDomain* id) const {
  auto disjoint_set = disjoint_iter_sets_.find(id);
  if (disjoint_set == disjoint_iter_sets_.end()) {
    return id;
  }

  int max_concrete_root_doms = 0;
  IterDomain* concrete_dom = id;
  for (auto id : (*disjoint_set->second)) {
    auto id_size_it = n_concrete_ids_.find(id);
    TORCH_INTERNAL_ASSERT(
        id_size_it != n_concrete_ids_.end(),
        "Never computed how many concrete iter domains are associated with ",
        id);
    if (id_size_it->second > max_concrete_root_doms) {
      max_concrete_root_doms = id_size_it->second;
      concrete_dom = id;
    }
  }
  return concrete_dom;
}

kir::IterDomain* ComputeAtMap::getConcreteMappedID(kir::IterDomain* id) const {
  auto disjoint_set = kir_disjoint_iter_sets_.find(id);
  if (disjoint_set == kir_disjoint_iter_sets_.end()) {
    return id;
  }

  int max_concrete_root_doms = 0;
  kir::IterDomain* concrete_dom = id;
  for (auto id : (*disjoint_set->second)) {
    auto id_size_it = kir_n_concrete_ids_.find(id);
    TORCH_INTERNAL_ASSERT(
        id_size_it != kir_n_concrete_ids_.end(),
        "Never computed how many concrete iter domains are associated with ",
        id);
    if (id_size_it->second > max_concrete_root_doms) {
      max_concrete_root_doms = id_size_it->second;
      concrete_dom = id;
    }
  }
  return concrete_dom;
}

IterDomain* ComputeAtMap::getParallelizedMappedID(IterDomain* id) const {
  auto disjoint_set_it = disjoint_iter_sets_.find(id);
  if (disjoint_set_it == disjoint_iter_sets_.end()) {
    return id;
  }
  auto parallelized_id_it = parallel_type_map_.find(disjoint_set_it->second);
  if (parallelized_id_it == parallel_type_map_.end()) {
    return id;
  }
  return parallelized_id_it->second;
}

kir::IterDomain* ComputeAtMap::getParallelizedMappedID(
    kir::IterDomain* id) const {
  auto disjoint_set_it = kir_disjoint_iter_sets_.find(id);
  if (disjoint_set_it == kir_disjoint_iter_sets_.end()) {
    return id;
  }
  auto parallelized_id_it =
      kir_parallel_type_map_.find(disjoint_set_it->second);
  if (parallelized_id_it == kir_parallel_type_map_.end()) {
    return id;
  }
  return parallelized_id_it->second;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch