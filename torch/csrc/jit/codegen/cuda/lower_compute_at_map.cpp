#include <torch/csrc/jit/codegen/cuda/lower_compute_at_map.h>

#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

void ComputeAtMap::build() {
  Fusion* fusion = FusionGuard::getCurFusion();
  TORCH_INTERNAL_ASSERT(fusion != nullptr);

  for (auto expr : fusion->exprs()) {
    if (!expr->outputs()[0]->isA<TensorView>()) {
      continue;
    }

    // TODO: Do we need to map all output tensors, or just the first? For
    // indexing we may need all of them mapped. Do we expect all other outputs
    // to be replayed as the first or should we do it?
    auto c_tv = expr->outputs()[0]->as<TensorView>();

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
            auto parallel_entry_it =
                parallel_type_map_.find(disjoint_set.get());
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
              parallel_type_map_[disjoint_set.get()] = c_id;
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

IterDomain* ComputeAtMap::getParallelizedMappedID(IterDomain* id) const {
  auto disjoint_set_it = disjoint_iter_sets_.find(id);
  if (disjoint_set_it == disjoint_iter_sets_.end()) {
    return id;
  }
  auto parallelized_id_it =
      parallel_type_map_.find(disjoint_set_it->second.get());
  if (parallelized_id_it == parallel_type_map_.end()) {
    return id;
  }
  return parallelized_id_it->second;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch