#include <torch/csrc/jit/codegen/cuda/test_index_compute.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// We're going to replay this split operation on the corresponding ID
void TestReplay::handle(Split* s) {
  auto in = s->in();

  auto concrete_in = ca_maps_.getConcreteMappedID(in);
  auto mapped_in_it = concrete_to_id.find(concrete_in);
  if (mapped_in_it == concrete_to_id.end()) {
    return;
  }

  auto mapped_in = mapped_in_it->second;

  if (leaf_ids.find(mapped_in) == leaf_ids.end()) {
    return;
  }

  auto replayed_outs =
      IterDomain::split(mapped_in, s->factor(), s->innerSplit());

  auto concrete_outer = ca_maps_.getConcreteMappedID(s->outer());
  auto concrete_inner = ca_maps_.getConcreteMappedID(s->inner());

  concrete_to_id[concrete_outer] = replayed_outs.first;
  concrete_to_id[concrete_inner] = replayed_outs.second;

  leaf_ids.erase(mapped_in);
  leaf_ids.emplace(replayed_outs.first);
  leaf_ids.emplace(replayed_outs.second);
}

// We're going to replay this merge operation on the corresponding IDs
void TestReplay::handle(Merge* m) {
  auto in_outer = m->outer();
  auto in_inner = m->inner();

  auto concrete_in_outer = ca_maps_.getConcreteMappedID(in_outer);
  auto concrete_in_inner = ca_maps_.getConcreteMappedID(in_inner);

  auto mapped_in_outer_it = concrete_to_id.find(concrete_in_outer);
  auto mapped_in_inner_it = concrete_to_id.find(concrete_in_inner);

  if (mapped_in_outer_it == concrete_to_id.end() ||
      mapped_in_inner_it == concrete_to_id.end()) {
    return;
  }

  auto mapped_in_outer = mapped_in_outer_it->second;
  auto mapped_in_inner = mapped_in_inner_it->second;

  if (leaf_ids.find(mapped_in_outer) == leaf_ids.end() &&
      leaf_ids.find(mapped_in_inner) == leaf_ids.end()) {
    return;
  }

  auto replayed = IterDomain::merge(mapped_in_outer, mapped_in_inner);

  auto concrete_replayed = ca_maps_.getConcreteMappedID(m->out());

  leaf_ids.erase(mapped_in_outer);
  leaf_ids.erase(mapped_in_inner);

  leaf_ids.emplace(replayed);

  concrete_to_id[concrete_replayed] = replayed;
}

TensorDomain* TestReplay::computeReplay() {
  // Extract iter domain's from the loop structure
  std::vector<IterDomain*> fusion_loop_structure;

  std::transform(
      loop_structure_.begin(),
      loop_structure_.end(),
      std::back_inserter(fusion_loop_structure),
      [&](kir::ForLoop* fl) { return ca_maps_.toFusion(fl->iter_domain()); });

  // Get all inputs that generated that loop structure, some root inputs can be
  // mapped to eachother
  auto all_inputs = InputsOf::outputs(
      FusionGuard::getCurFusion(),
      std::vector<Val*>(
          fusion_loop_structure.begin(), fusion_loop_structure.end()));

  auto all_iter_inputs = ir_utils::filterByType<IterDomain>(all_inputs);

  // Reduce those inputs to a single set of axes to remove the iter domains that
  // map to eachother
  std::unordered_set<IterDomain*> root_axes;
  for (auto root_id : all_iter_inputs) {
    auto concrete_id = ca_maps_.getConcreteMappedID(root_id);
    if (concrete_to_id.find(concrete_id) != concrete_to_id.end()) {
      continue;
    }
    root_axes.emplace(root_id);
    concrete_to_id[concrete_id] = root_id;
    leaf_ids.emplace(root_id);
  }

  auto replay_exprs = ExprSort::getExprs(
      FusionGuard::getCurFusion(),
      {fusion_loop_structure.begin(), fusion_loop_structure.end()});

  // Run the replay
  for (auto expr : replay_exprs) {
    OptInDispatch::handle(expr);
  }

  // Representation of a tensor replayed as the loop structure.
  std::vector<IterDomain*> loops_replayed_domain;

  // std::cout<<"Mapping:"<<std::endl;
  // Lookup is based on concrete mapped ID because that's what we used to mark
  // them during replay. Loop_id's though should already be concrete so lookup
  // may be redundant.
  std::transform(
      fusion_loop_structure.begin(),
      fusion_loop_structure.end(),
      std::back_inserter(loops_replayed_domain),
      [&](IterDomain* loop_id) {
        auto concrete_to_id_it =
            concrete_to_id.find(ca_maps_.getConcreteMappedID(loop_id));
        TORCH_INTERNAL_ASSERT(
            concrete_to_id_it != concrete_to_id.end(),
            "Could not find required iter domain in reference replay.");
        auto replayed_id = concrete_to_id_it->second;
        leaf_ids.erase(replayed_id);
        return replayed_id;
      });

  // Add any remaining leaf iter domains
  for (auto entry : leaf_ids) {
    loops_replayed_domain.push_back(entry);
  }
  if (replay_exprs.empty()) {
    auto domain = new TensorDomain(
        // Order for root axis does matter in this case
        loops_replayed_domain);
    return domain;
  } else {
    auto domain = new TensorDomain(
        // Order doesn't matter for root axis
        std::vector<IterDomain*>(root_axes.begin(), root_axes.end()),
        loops_replayed_domain);
    return domain;
  }
}

IndexCompute getReferenceIndexing(
    const std::vector<kir::ForLoop*>& loop_structure,
    const ComputeAtMap& ca_maps,
    TensorDomain* reference_tensor) {
  auto gpu_lower = GpuLower::current();

  std::unordered_map<kir::IterDomain*, kir::Val*> initial_index_map;

  for (size_t loop_i = 0; loop_i < loop_structure.size(); loop_i++) {
    auto lowered_id = gpu_lower->lowerValue(reference_tensor->axis(loop_i))
                          ->as<kir::IterDomain>();
    initial_index_map[lowered_id] = loop_structure[loop_i]->index();
  }
  return getReferenceIndexing(
      loop_structure, ca_maps, reference_tensor, initial_index_map, {});
}

IndexCompute getReferenceIndexing(
    const std::vector<kir::ForLoop*>& loop_structure,
    const ComputeAtMap& ca_maps,
    TensorDomain* reference_tensor,
    std::unordered_map<kir::IterDomain*, kir::Val*> index_map,
    std::unordered_set<IterDomain*> preferred_paths) {
  auto gpu_lower = GpuLower::current();

  std::unordered_map<kir::IterDomain*, kir::Val*> reference_extent_map;
  for (auto loop : loop_structure) {
    // If there's a broadcast merged in the for loop ID we want to track its
    // extent
    auto inputs = InputsOf::outputs(
        FusionGuard::getCurFusion(), {ca_maps.toFusion(loop->iter_domain())});
    auto iter_inputs = ir_utils::filterByType<IterDomain>(inputs);

    // If any of the inputs are a broadcast, explicitly mark the loop id's
    // extent
    if (std::any_of(iter_inputs.begin(), iter_inputs.end(), [](IterDomain* id) {
          return id->isBroadcast();
        })) {
      reference_extent_map[loop->iter_domain()] = loop->iter_domain()->extent();
    }
  }

  // Convert to preferred_path to kir::IterDomain for IndexCompute
  std::unordered_set<kir::IterDomain*> kir_preferred_path;
  std::transform(
      preferred_paths.begin(),
      preferred_paths.end(),
      std::inserter(kir_preferred_path, kir_preferred_path.begin()),
      [&gpu_lower](IterDomain* id) {
        return gpu_lower->lowerValue(id)->as<kir::IterDomain>();
      });

  IndexCompute compute(
      reference_tensor,
      index_map,
      reference_extent_map,
      std::unordered_set<kir::IterDomain*>(),
      reference_tensor->contiguity(),
      kir_preferred_path);

  compute.run();

  return compute;
}

namespace {

class PreferredPathCompute : public IterVisitor {
 private:
  void handle(Expr* e) override {
    auto all_iter_inputs = ir_utils::filterByType<IterDomain>(e->inputs());
    if (std::any_of(
            all_iter_inputs.begin(),
            all_iter_inputs.end(),
            [&](IterDomain* inp_id) {
              return this->preferred_path.find(inp_id) !=
                  this->preferred_path.end();
            })) {
      auto all_iter_outputs = ir_utils::filterByType<IterDomain>(e->outputs());
      preferred_path.insert(all_iter_outputs.begin(), all_iter_outputs.end());
    }
  }

 private:
  std::unordered_set<IterDomain*> preferred_path;

 public:
  static std::unordered_set<IterDomain*> compute(
      TensorDomain* reference_domain,
      const std::unordered_set<IterDomain*>& preferred_roots) {
    std::unordered_set<IterDomain*> reference_root(
        reference_domain->getRootDomain().begin(),
        reference_domain->getRootDomain().end());

    // This assert doesn't work because preferred_roots may be on index tensors
    // rfactor domain, which could differ from references root domain. Would be
    // nice to make sure that preferred_roots are somewhere in the history of
    // reference, but chose not to do that due to complexity. May make sense to
    // do in a debug mode.
    //
    // TORCH_INTERNAL_ASSERT(
    //     std::all_of(
    //         preferred_roots.begin(),
    //         preferred_roots.end(),
    //         [&reference_root](IterDomain* preferred_root) {
    //           return reference_root.find(preferred_root) !=
    //               reference_root.end();
    //         }),
    //     "Preferred path compute recieved root tensors to prefer that are not
    //     in reference.");

    std::vector<Val*> val_domain(
        reference_domain->domain().begin(), reference_domain->domain().end());

    PreferredPathCompute compute;
    compute.preferred_path = preferred_roots;
    compute.traverseFrom(FusionGuard::getCurFusion(), val_domain);
    return compute.preferred_path;
  }
};
} // namespace

std::unordered_set<IterDomain*> buildPreferredPaths(
    TensorDomain* reference_tensor,
    std::unordered_set<IterDomain*> preferred_roots) {
  return PreferredPathCompute::compute(reference_tensor, preferred_roots);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch