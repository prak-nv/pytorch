#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/lower_trivial_reductions.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

bool isTrivialReduction(TensorView* tv, IterDomain* id);

bool traverseToRFactorTensor(TensorView* tv, IterDomain* root_id) {
  TORCH_INTERNAL_ASSERT(
      root_id->definition() == nullptr, "Not root IterDomain: ", root_id);

  if (tv->definition() == nullptr) {
    return false;
  }

  const auto& inputs = tv->definition()->inputs();

  if (inputs.size() != 1 || !inputs[0]->isA<TensorView>()) {
    return false;
  }

  auto producer = inputs[0]->as<TensorView>();

  if (!producer->hasRFactor()) {
    return false;
  }

  auto c2p = PairwiseRootDomainMap(producer, tv)
                 .mapConsumerToProducer(tv->domain(), producer->domain());

  auto producer_id_it = c2p.find(root_id);
  if (producer_id_it == c2p.end()) {
    // No matching producer is found. Stop traversing.
    return false;
  }

  auto producer_root_id = producer_id_it->second;

  return isTrivialReduction(producer, producer_root_id);
}

bool isTrivialReduction(TensorView* tv, IterDomain* id) {
  auto id_inputs = InputsOf::output(id->fusion(), id);
  for (auto root_id : ir_utils::filterByType<IterDomain>(id_inputs)) {
    if (root_id->isReduction() && root_id->rawExtent()->isOneInt()) {
      continue;
    }
    if (!traverseToRFactorTensor(tv, root_id)) {
      return false;
    }
  }
  return true;
}

} // namespace

std::unordered_set<IterDomain*> detectTrivialReductions(Fusion* fusion) {
  auto used_vals = DependencyCheck::getAllValsBetween(
      {fusion->inputs().begin(), fusion->inputs().end()}, fusion->outputs());

  std::unordered_set<IterDomain*> trivial_reductions;

  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    for (auto id : tv->domain()->domain()) {
      if (isTrivialReduction(tv, id)) {
        trivial_reductions.insert(id);
      }
    }
  }

  return trivial_reductions;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
