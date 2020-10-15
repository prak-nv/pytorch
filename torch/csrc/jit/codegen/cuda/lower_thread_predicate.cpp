
#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

kir::Val* getPredicatePerParallelType(
    ParallelType pt,
    const ThreadPredicateMap::SourceMapType& source_map) {
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  if (pt == ParallelType::BIDx || pt == ParallelType::BIDy ||
      pt == ParallelType::BIDz) {
    auto source = source_map.at(pt);
    TORCH_INTERNAL_ASSERT(!source.empty(), "No predicate source found");
    TORCH_INTERNAL_ASSERT(source.size() == 1, "Multiple sources detected");
    auto src = *source.begin();
    auto flag_name = kir::GridReduction::getPredicateFlagName(src);
    return ir_builder.create<kir::NamedScalar>(flag_name, DataType::Bool);
  } else {
    return ir_builder.eqExpr(
        kir::NamedScalar::getParallelIndex(pt), ir_builder.create<kir::Int>(0));
  }
}

kir::Bool* getPredicate(
    const ir_utils::ParallelTypeBitmap& bits,
    const ThreadPredicateMap::SourceMapType& source_map) {
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  if (bits.none()) {
    return ir_builder.create<kir::Bool>(true);
  }

  kir::Val* pred = nullptr;

  for (const auto& pt_bool : bits.getMap()) {
    if (pt_bool.second) {
      const auto tp = getPredicatePerParallelType(pt_bool.first, source_map);
      pred = (pred == nullptr) ? tp : ir_builder.andExpr(pred, tp);
    }
  }

  TORCH_INTERNAL_ASSERT(pred != nullptr);
  TORCH_INTERNAL_ASSERT(pred->dtype() == DataType::Bool);

  return pred->as<kir::Bool>();
}

void mergeSourceMap(
    ThreadPredicateMap::SourceMapType& dst,
    const ThreadPredicateMap::SourceMapType& src) {
  for (const auto& kv : src) {
    const auto& src_key = kv.first;
    const auto& src_value = kv.second;
    auto& dst_set = dst[src_key];
    for (const auto& src_tensor : src_value) {
      dst_set.insert(src_tensor);
    }
  }
}

void addToSouceMap(
    ThreadPredicateMap::SourceMapType& dst,
    const TensorView* tv,
    const ir_utils::ParallelTypeBitmap& reducton_pred) {
  for (const auto& kv : reducton_pred.getMap()) {
    if (kv.second) {
      ParallelType ptype = kv.first;
      dst[ptype].insert(tv);
    }
  }
}

void maskSouceMap(
    ThreadPredicateMap::SourceMapType& src_map,
    const ir_utils::ParallelTypeBitmap& mask) {
  for (const auto& kv : mask.getMap()) {
    if (!kv.second) {
      ParallelType ptype = kv.first;
      src_map[ptype].clear();
    }
  }
}

// A bit of a hack for now for GEMM tiling so we don't fetch tiles multiple
// times. It's safe to do, there may simply be a better place to do it.
ir_utils::ParallelTypeBitmap avoidRedundantWritesToSmem(
    const TensorView* out_tv,
    const ir_utils::ParallelTypeBitmap& pred) {
  auto new_pred = pred;
  if (out_tv->getMemoryType() == MemoryType::Shared) {
    for (size_t i = 0; i < out_tv->nDims(); i++) {
      auto id = out_tv->getComputeAtAxis(i).first;
      if (out_tv->axis(i)->isBroadcast() && id->isThreadDim()) {
        new_pred.set(id->getParallelType(), true);
      }
    }
  }
  return new_pred;
}

} // namespace

// Update the reduction_deps bitset based on provided Expr
void ThreadPredicateMap::updateBitSet(const Expr* expr) {
  FUSER_PERF_SCOPE("ThreadPredicateMap::updateBitSet");

  // Which predicates were set for the inputs
  ir_utils::ParallelTypeBitmap input_preds;

  // Which dims are reductions in inputs
  ir_utils::ParallelTypeBitmap input_reductions;

  // Which dims are bcast in inputs
  ir_utils::ParallelTypeBitmap input_bcasts;

  SourceMapType src_map;

  // Run through inputs and update bitsets
  for (const auto* inp : expr->inputs()) {
    if (!ir_utils::isTV(inp))
      continue;

    auto tv_inp = inp->as<TensorView>();
    TORCH_INTERNAL_ASSERT(
        thread_predicates_.find(tv_inp) != thread_predicates_.end(),
        "Thread predicate map was not initialized, couldn't find ",
        inp);

    input_preds |= at(tv_inp).first;

    mergeSourceMap(src_map, at(tv_inp).second);

    ir_utils::ParallelTypeBitmap id_reductions;
    ir_utils::ParallelTypeBitmap id_bcasts;
    ir_utils::ParallelTypeBitmap id_ptypes;

    for (auto id : tv_inp->domain()->domain()) {
      if (id->isThread()) {
        id_ptypes.set(id->getParallelType(), true);
        if (id->isReduction())
          id_reductions.set(id->getParallelType(), true);
        if (id->isBroadcast())
          id_bcasts.set(id->getParallelType(), true);
      }
    }

    // Validate the combination of ptypes, reductions, bcasts
    for (size_t i = 0; i < ir_utils::ParallelTypeBitmap::num_p_type; i++) {
      if (input_reductions[i]) {
        if (id_ptypes[i]) {
          TORCH_INTERNAL_ASSERT(
              id_reductions[i],
              "Mismatched parallelized reductions found on inputs of epxr: ",
              expr);
          TORCH_CHECK(
              !id_bcasts[i],
              "Invalid broadcast and reduction combination, tried to parallelize both with the same thread dim: ",
              inp);
        }
      }
    }

    // Accumulate
    input_reductions |= id_reductions;
    input_bcasts |= id_bcasts;

    if (id_reductions.any()) {
      // add tv_inp as a source
      addToSouceMap(src_map, tv_inp, id_reductions);
    }
  }

  // Update map for this tv, before accumulating to other inputs
  // Add any reductions this id has to any input predicates
  auto output_preds = input_preds | input_reductions;

  // Figure out which dims bcast wants to reset
  const auto bcast_reset_mask = ~(output_preds & input_bcasts);

  // Get rid of any reductions which are bcasted
  output_preds &= bcast_reset_mask;
  
  // Similarly, drop non-relevant source tensors
  maskSouceMap(src_map, bcast_reset_mask);

  // Run through outputs and set bitset predicates
  for (auto* out : expr->outputs()) {
    if (auto tv = dynamic_cast<const TensorView*>(out)) {
      TORCH_INTERNAL_ASSERT(find(tv) == end());
      insert(tv, avoidRedundantWritesToSmem(tv, output_preds), src_map);
    }
  }
}

// TODO(kir): revisit this - can we build it from the kernel IR?
ThreadPredicateMap::ThreadPredicateMap(Fusion* _fusion) : fusion_(_fusion) {
  FUSER_PERF_SCOPE("ThreadPredicateMap");

  // Initialize mapping for input tensors
  for (auto inp : fusion_->inputs()) {
    if (auto tv = dynamic_cast<const TensorView*>(inp)) {
      insert(tv, ir_utils::ParallelTypeBitmap(), SourceMapType());
    }
  }
  for (auto expr : fusion_->exprs(true)) {
    updateBitSet(expr);
  }
}

ThreadPredicateMap::const_iterator ThreadPredicateMap::find(
    const TensorView* tv) const {
  return thread_predicates_.find(tv);
}

ThreadPredicateMap::const_iterator ThreadPredicateMap::end() const {
  return thread_predicates_.end();
}

const ThreadPredicateMap::MapType::mapped_type& ThreadPredicateMap::at(
    const TensorView* tv) const {
  return thread_predicates_.at(tv);
}

ThreadPredicateMap::MapType::mapped_type& ThreadPredicateMap::at(
    const TensorView* tv) {
  return thread_predicates_.at(tv);
}

void ThreadPredicateMap::insert(
    const TensorView* tv,
    const ir_utils::ParallelTypeBitmap& pred,
    const SourceMapType& src_map) {
  insert(tv, std::make_pair(pred, src_map));
}

void ThreadPredicateMap::insert(
    const TensorView* tv,
    const std::pair<ir_utils::ParallelTypeBitmap, SourceMapType>&
        pred_and_src) {
  thread_predicates_.insert(std::make_pair(tv, pred_and_src));
}

kir::Bool* ThreadPredicateMap::getExpr(const TensorView* out_tv) const {
  TORCH_INTERNAL_ASSERT(find(out_tv) != end(), "Couldn't find ", out_tv);
  return getPredicate(at(out_tv).first, at(out_tv).second);
}

void ThreadPredicateMap::print() const {
  std::cout << "\nThreadPredicateMap\n";
  std::cout << "--------------------------------\n";
  for (const auto& kv : thread_predicates_) {
    std::cout << "T" << kv.first->name() << " {";
    // ir_utils::ParallelTypeBitmap
    for (auto ptkv : kv.second.first.getMap()) {
      if (ptkv.second) {
        std::cout << " " <<  ptkv.first;
      }
    }
    std::cout << " }\n";
    // SourceMapType
    for (const auto& pkv : kv.second.second) {
      std::cout << "    " << pkv.first << ".{";
      for (auto tv : pkv.second) {
        std::cout << " " << tv;
      }
      std::cout << " }\n";
    }
  }
  std::cout << "--------------------------------\n\n";
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
