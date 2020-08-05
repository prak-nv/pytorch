#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

namespace torch {
namespace jit {
namespace fuser {

bool PredicateCompute::hasPredicates(const kir::TensorIndex* ti) {
  std::vector<Bool*> preds;
  for (auto ind : ti->indices())
    if (FusionGuard::getCurFusion()->origin(ind) != nullptr)
      return true;
  return false;
}

std::vector<kir::Bool*> PredicateCompute::computePredicates(
    const kir::TensorIndex* ti) {
  const TensorView* tv = ti->view();
  const std::vector<IterDomain*>& root = tv->getRootDomain();

  std::vector<kir::Bool*> preds(root.size(), new kir::Bool(true));

  bool no_pred_needed = true;
  for (auto id : tv->domain()->domain())
    if (id->getOrigin() != nullptr)
      no_pred_needed = false;

  if (no_pred_needed) {
    return preds;
  }

  TORCH_INTERNAL_ASSERT(
      root.size() == ti->nDims(),
      "Predicate compute received mismatched TensorView and TensorIndex.");

  Val* extent = nullptr;

  for (size_t i = 0; i < ti->nDims(); i++) {
    bool zero_ind = ti->index(i)->isZeroInt();
    bool simple_ind = ti->index(i)->getOrigin() == nullptr;

    if (root[i]->isBroadcast()) {
      continue;
    } else if (simple_ind && !zero_ind) {
      continue;
    } else if (zero_ind) {
      if (root[i]->extent()->isOneInt())
        continue;
      if (extent == nullptr) {
        extent = root[i]->extent();
      } else {
        extent = mul(extent, root[i]->extent());
      }
    } else {
      auto local_extent = root[i]->extent();
      if (extent != nullptr) {
        local_extent = mul(extent, local_extent);
      }
      auto pred = kir::ltExpr(ti->index(i), local_extent);
      extent = nullptr;
      TORCH_INTERNAL_ASSERT(
          pred->getValType().value() == ValType::KirScalar &&
          pred->getDataType().value() == DataType::Bool);
      preds[i] = pred->as<kir::Bool>();
    }
  }
  return preds;
}

kir::Bool* PredicateCompute::getInlinePredicate(
    Expr* expr,
    const std::vector<kir::ForLoop*>& loops,
    kir::Bool* thread_pred) {
  if (loops.empty())
    return new kir::Bool(true);

  TORCH_INTERNAL_ASSERT(
      ir_utils::isTVOp(expr),
      "Cannot generate predicate based on operation without a TensorView.");

  auto out_tv = ir_utils::getTVOutput(expr);

  auto pred_contiguity = out_tv->domain()->contiguity();

  for (auto inp : expr->inputs()) {
    if (!ir_utils::isTV(inp)) {
      continue;
    }
    auto inp_tv = inp->as<TensorView>();
    if (inp_tv->domain()->hasRFactor()) {
      continue;
    } else if (
        inp_tv->getMemoryType() == MemoryType::Shared ||
        inp_tv->getMemoryType() == MemoryType::Local) {
      continue;
    } else {
      pred_contiguity = IndexCompute::contiguityAnd(
          pred_contiguity,
          IndexCompute::contiguityPasC(inp_tv->domain(), out_tv->domain()));
    }
  }

  auto domain_indices = loop_utils::getIndicesForTV(out_tv, loops, true);
  auto root_indices = IndexCompute::get(
      out_tv->domain(), domain_indices, pred_contiguity, true);
  auto pred_ti = new kir::TensorIndex(out_tv, root_indices);
  auto all_preds = PredicateCompute::computePredicates(pred_ti);

  // If we have thread predicates, add those
  if (thread_pred != nullptr) {
    all_preds.push_back(thread_pred);
  }

  std::vector<kir::Bool*> preds;

  for (auto pred : all_preds)
    if (!(pred->isConst()) || !(pred->isConst() && pred->value().value()))
      preds.push_back(pred);

  if (preds.empty()) {
    return new kir::Bool(true);
  }

  Val* cond = preds[0];

  for (decltype(preds.size()) i{1}; i < preds.size(); i++) {
    cond = kir::andExpr(cond, preds[i]);
  }

  TORCH_INTERNAL_ASSERT(
      cond->getValType().value() == ValType::KirScalar &&
          cond->getDataType().value() == DataType::Bool,
      "Error computing predicate, should be returning a Bool, but returning ",
      cond->getDataType().value());

  return cond->as<kir::Bool>();
}

namespace {

class TORCH_CUDA_API ExtractTVExprs {
  // Get all exprs within loop and containing loops
 public:
  static std::vector<Expr*> get(kir::ForLoop* loop) {
    ExtractTVExprs ee;
    ee.extract(loop);
    return ee.exprs;
  }

 private:
  // Open the for loop.
  void extract(kir::ForLoop* fl) {
    for (auto expr : fl->body().exprs()) {
      if (ir_utils::isTVOp(expr)) {
        exprs.push_back(expr);
      } else if (expr->getExprType().value() == ExprType::ForLoop) {
        extract(expr->as<kir::ForLoop>());
      }
    }
  };

  std::vector<Expr*> exprs;
};

IterDomain* getTermIDInMap(
    IterDomain* id,
    std::unordered_map<IterDomain*, IterDomain*> map) {
  auto entry = id;
  while (map.find(entry) != map.end()) {
    entry = map.at(entry);
  }
  return entry;
}

} // namespace

kir::Bool* UnrollPredicate::get(
    const std::vector<kir::ForLoop*>& outer_loops,
    kir::ForLoop* unrolled_loop) {
  UnrollPredicate up(outer_loops, unrolled_loop);

  std::unordered_set<kir::Bool*> pred_set;
  for (auto entry : up.predicates) {
    pred_set.emplace(entry.second);
  }

  if (up.predicates.empty()) {
    return new kir::Bool(true);
  }

  Val* unroll_pred = nullptr;
  for (auto pred : pred_set) {
    if (unroll_pred == nullptr) {
      unroll_pred = pred;
    } else {
      unroll_pred = kir::andExpr(unroll_pred, pred);
    }
  }
  TORCH_INTERNAL_ASSERT(
      unroll_pred->getValType().value() == ValType::KirScalar &&
      unroll_pred->getDataType().value() == DataType::Bool);
  return unroll_pred->as<kir::Bool>();
}

void UnrollPredicate::predicateOn(Expr* tv_expr) {
  if (for_loops.empty())
    return;

  auto out_tv = ir_utils::getTVOutput(tv_expr);

  bool pred_reductions = loop_utils::loopsHasReductions(out_tv, for_loops);

  auto pred_contiguity = out_tv->domain()->contiguity();

  for (auto inp : tv_expr->inputs()) {
    if (!ir_utils::isTV(inp)) {
      continue;
    }
    auto inp_tv = inp->as<TensorView>();
    if (inp_tv->domain()->hasRFactor()) {
      continue;
    } else if (
        inp_tv->getMemoryType() == MemoryType::Shared ||
        inp_tv->getMemoryType() == MemoryType::Local) {
      continue;
    } else {
      pred_contiguity = IndexCompute::contiguityAnd(
          pred_contiguity,
          IndexCompute::contiguityPasC(inp_tv->domain(), out_tv->domain()));
    }
  }

  auto domain_indices =
      loop_utils::getUnrollPredIndicesForTV(out_tv, for_loops);
  auto root_indices = IndexCompute::get(
      out_tv->domain(), domain_indices, pred_contiguity, true);
  auto pred_ti = new kir::TensorIndex(out_tv, root_indices);
  auto all_preds = PredicateCompute::computePredicates(pred_ti);

  TORCH_INTERNAL_ASSERT(
      all_preds.size() == out_tv->getRootDomain().size(),
      "Predicates should be produced for every dimension, even if it's simply set as true.");

  for (size_t i = 0; i < all_preds.size(); i++) {
    if (all_preds[i]->isConst() && all_preds[i]->value().value()) {
      continue;
    }
    auto term_id = getTermIDInMap(out_tv->getRootDomain()[i], forward_root_map);
    predicates[term_id] = all_preds[i];
  }
}

void UnrollPredicate::openLoop(kir::ForLoop* fl) {
  for_loops.push_back(fl);

  for (auto expr : fl->body().exprs()) {
    if (ir_utils::isTVOp(expr)) {
      predicateOn(expr);
    } else if (expr->getExprType().value() == ExprType::ForLoop) {
      openLoop(expr->as<kir::ForLoop>());
    }
  }

  for_loops.pop_back();
}

UnrollPredicate::UnrollPredicate(
    const std::vector<kir::ForLoop*>& outer_loops,
    kir::ForLoop* unrolled_loop)
    : for_loops(outer_loops) {
  auto exprs = ExtractTVExprs::get(unrolled_loop);

  for (auto expr : exprs) {
    auto out_tv = ir_utils::getTVOutput(expr);
    for (auto inp : expr->inputs()) {
      if (inp->getValType().value() != ValType::TensorView) {
        continue;
      }

      auto root_p2c = TensorDomain::mapRootPtoC(
          inp->as<TensorView>()->domain(), out_tv->domain());
      for (auto entry : root_p2c) {
        auto p_id = entry.first;
        auto c_id = entry.second;
        if (p_id != c_id) {
          forward_root_map[p_id] = c_id;
        }
      }
    }
  }

  openLoop(unrolled_loop);
}

} // namespace fuser
} // namespace jit
} // namespace torch
