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
  auto replayed_outs =
      IterDomain::split(mapped_in, s->factor(), s->innerSplit());
  auto concrete_outer = ca_maps_.getConcreteMappedID(s->outer());
  auto concrete_inner = ca_maps_.getConcreteMappedID(s->inner());
  concrete_to_id[concrete_outer] = replayed_outs.first;
  concrete_to_id[concrete_inner] = replayed_outs.second;
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
  auto mapped_in_outer = mapped_in_outer_it->first;
  auto mapped_in_inner = mapped_in_inner_it->second;

  auto replayed = IterDomain::merge(mapped_in_outer, mapped_in_inner);
  auto concrete_replayed = ca_maps_.getConcreteMappedID(m->out());
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

  // Reduce those inputs to a single set of concrete axes to remove the iter
  // domains that map to eachother
  std::unordered_set<IterDomain*> concrete_root_axes;
  std::transform(
      all_iter_inputs.begin(),
      all_iter_inputs.end(),
      std::inserter(concrete_root_axes, concrete_root_axes.begin()),
      [&](IterDomain* id) { return ca_maps_.getConcreteMappedID(id); });

  // Create a map from the concrete_id's to actual id's. This is really just the
  // replay map to track inputs being used to produce outputs
  for (auto id : concrete_root_axes) {
    concrete_to_id[id] = id;
  }

  // Vector of val's to traverse on with IterVisitor
  std::vector<Val*> val_loop_structure;

  std::transform(
      fusion_loop_structure.begin(),
      fusion_loop_structure.end(),
      std::back_inserter(val_loop_structure),
      [](IterDomain* id) { return id; });

  // Replay the transformations
  traverseFrom(fusion_loop_structure[0]->fusion(), val_loop_structure);

  // representation of a tensor replayed as the loop structure.
  std::vector<IterDomain*> loops_replayed_domain;
  // Lookup is based on concrete mapped ID because that's what we used to mark
  // them during replay. Loop_id's though should already be concrete so lookup
  // may be redundant.
  std::transform(
      fusion_loop_structure.begin(),
      fusion_loop_structure.end(),
      std::back_inserter(loops_replayed_domain),
      [&](IterDomain* loop_id) {
        return concrete_to_id.at(ca_maps_.getConcreteMappedID(loop_id));
      });

  // Create tensor domain with concrete root domains as the root, and the
  // replayed loop structure as the domain
  return new TensorDomain(
      // Order doesn't matter for root axis
      std::vector<IterDomain*>(
          concrete_root_axes.begin(), concrete_root_axes.end()),
      loops_replayed_domain);
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

    TORCH_INTERNAL_ASSERT(
        std::all_of(
            preferred_roots.begin(),
            preferred_roots.end(),
            [&reference_root](IterDomain* preferred_root) {
              return reference_root.find(preferred_root) !=
                  reference_root.end();
            }),
        "Preferred path compute recieved root tensors to prefer that are not in reference.");

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

TestIndexing::TestIndexing()
    : gpu_lower(GpuLower::current()),
      ir_builder(gpu_lower->kernel()),
      ca_maps_(GpuLower::current()->caMaps()) {}

void TestIndexing::visit(kir::ForLoop* fl) {
  for_loops.push_back(fl);
  // Modifying in place, make a copy of the vector
  const std::vector<kir::Expr*> exprs = fl->body().exprs();
  for (auto expr : exprs) {
    handle(expr);
  }
  for_loops.pop_back();
}

void TestIndexing::visit(kir::IfThenElse* ite) {
  for (auto expr : ite->thenBody().exprs()) {
    handle(expr);
  }
}

void TestIndexing::handle(kir::Expr* expr) {
  if (expr->isA<kir::ForLoop>() || expr->isA<kir::IfThenElse>()) {
    expr->accept(this);
    return;
  }
  if (!ir_utils::isTVOp(expr)) {
    expr->accept(this);
    return;
  }

  // TODO: This should all be done in kir, not fusion ir.
  TORCH_INTERNAL_ASSERT(expr->outputs()[0]->isA<kir::TensorView>());

  auto reference_tensor = TestReplay::getReference(for_loops, ca_maps_);

  auto ref_compute =
      getReferenceIndexing(for_loops, ca_maps_, reference_tensor);

  auto out_fuser_tv = expr->outputs()[0]->as<kir::TensorView>()->fuserTv();

  std::unordered_map<IterDomain*, IterDomain*> root_ref_to_index_tv;
  // Root of reference tensor is already all concrete ids, so for replay which
  // generates map we can simply map to concrete ids
  for (auto index_root : out_fuser_tv->getRootDomain()) {
    auto concrete_root = ca_maps_.getConcreteMappedID(index_root);
    root_ref_to_index_tv.emplace(std::make_pair(concrete_root, index_root));
  }

  BestEffortReplay replay_out_as_ref(
      out_fuser_tv->domain()->domain(),
      reference_tensor->domain(),
      root_ref_to_index_tv,
      true);

  auto ref_2_out = replay_out_as_ref.getReplay();

  auto output = ref_compute.updateIndexCompute(
      out_fuser_tv->domain(),
      ref_2_out,
      {},
      out_fuser_tv->domain()->contiguity());

  TORCH_INTERNAL_ASSERT(false);
}

void TestIndexing::generate(std::vector<kir::Expr*>& exprs) {
  for (auto expr : exprs) {
    handle(expr);
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch