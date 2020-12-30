#include <torch/csrc/jit/codegen/cuda/lower_loops.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <algorithm>
#include <deque>
#include <numeric>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

LoopNestGenerator::LoopNestGenerator(
    Fusion* fusion,
    const std::vector<Expr*>& exprs,
    const ComputeAtMap& ca_maps)
    : fusion_(fusion),
      ir_builder_(GpuLower::current()->kernel()),
      ca_maps_(ca_maps) {
  generate(exprs);
}

namespace {

// TODO(kir): revisit and try to simplify this
kir::ForLoop* openForHelper(kir::ForLoop* scope, IterDomain* id) {
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());
  const auto kir_id = gpu_lower->lowerValue(id)->as<kir::IterDomain>();
  kir::ForLoop* new_scope = nullptr;
  if (id->isThread()) {
    std::stringstream ss;
    ss << id->getParallelType();
    new_scope = ir_builder.create<kir::ForLoop>(
        ir_builder.create<kir::NamedScalar>(ss.str(), DataType::Int),
        kir_id,
        scope);
  } else {
    new_scope = ir_builder.create<kir::ForLoop>(
        ir_builder.create<kir::Int>(c10::nullopt), kir_id, scope);
  }
  if (scope != nullptr) {
    scope->body().push_back(new_scope);
  }
  return new_scope;
}

} // namespace

void LoopNestGenerator::openFor(IterDomain* iter_domain) {
  if (for_loops_.size() > 0) {
    const auto new_scope = openForHelper(for_loops_.back(), iter_domain);
    for_loops_.push_back(new_scope);
  } else {
    for_loops_.push_back(openForHelper(nullptr, iter_domain));
    lowered_exprs_.push_back(for_loops_.back());
  }
}

void LoopNestGenerator::closeFor() {
  TORCH_INTERNAL_ASSERT(!for_loops_.empty());
  for_loops_.pop_back();
}

void LoopNestGenerator::pushBack(kir::Expr* expr) {
  if (for_loops_.size() == 0) {
    lowered_exprs_.push_back(expr);
  } else {
    for_loops_.back()->body().push_back(expr);
  }
}

void LoopNestGenerator::handle(const Expr* expr) {
  const auto gpu_lower = GpuLower::current();

  // Check if it's a tensor view expression we need to place in the loop nest
  // structure
  if (!ir_utils::isTVOp(expr)) {
    for (auto out : expr->outputs()) {
      TORCH_INTERNAL_ASSERT(
          out->getValType().value() == ValType::Scalar,
          "Unrecognized output type found in expr ",
          expr,
          " cannot lower ",
          out->getValType().value());

      pushBack(ir_builder_.create<kir::Allocate>(
          gpu_lower->lowerValue(out),
          MemoryType::Local,
          ir_builder_.create<kir::Int>(1)));
    }
    pushBack(gpu_lower->lowerExpr(expr));
    return;
  }

  TensorView* out = expr->output(0)->as<TensorView>();

  // Figure out what the entire loop structure should look like.
  std::deque<IterDomain*> loop_structure;

  // As we go through iteration domains track the previous view
  const TensorView* last_ca_view = nullptr;
  // Check where in the previous view our last axis was in that view
  int64_t last_ca_view_ind = 0;

  // Look at each axis individually in out's domain
  for (int64_t out_i = 0; out_i < (int64_t)ca_maps_.producedAt(out); out_i++) {
    // Grab the axis information
    auto ca_point = out->getComputeAtAxis(out_i);
    auto ca_view = ca_point.second;
    auto ca_id = ca_point.first;

    // Figure out if there are axes in the compute at tensor view that aren't
    // in out, make sure to also open them. Check where to start looking for
    // them in the compute at view.
    size_t start = 0;
    if (last_ca_view == nullptr) {
      // Start at the begining, we haven't processed any axes yet.
      start = 0;
    } else if (last_ca_view == ca_view) {
      // This view is the same as the last axis, so start where we left off.
      start = last_ca_view_ind + 1;
    } else {
      // This is a new view, figure out where we are in it, and start from there
      for (start = 0; start < ca_view->nDims(); start++) {
        if (loop_structure.back() == ca_view->getComputeAtAxis(start).first) {
          break;
        }
      }
      start++;
    }

    // Go from start, and open all loops in the computeAt view until we hit the
    // one associated with out->getComputeAtAxis(out_i)
    for (size_t ca_i = start; ca_i < ca_view->nDims(); ca_i++) {
      // Note that ca_view->getComputeAtAxis(ca_i) is equivalent to
      // std::pair(ca_view->axis(ca_i), ca_view)
      loop_structure.push_back(ca_view->getComputeAtAxis(ca_i).first);

      // Update the last view processed
      last_ca_view_ind = ca_i;
      last_ca_view = ca_view;
      if (ca_view->getComputeAtAxis(ca_i).first == ca_id) {
        break;
      }
    }

    // Shouldn't ever hit this, but make sure we hit the break above, meaning we
    // added all necessary axes from the compute at view.
    TORCH_INTERNAL_ASSERT(
        ca_view->getComputeAtAxis(last_ca_view_ind).first == ca_id);
  }

  // We're up to the compute at point in loop_structure, grab the remaining
  // axes.
  for (int64_t out_i = (int64_t)out->getThisComputeAtAxis();
       out_i < (int64_t)out->nDims();
       out_i++) {
    // It's actually local, but getComputeAtAxis returns a std::pair, axis
    // doesn't
    loop_structure.push_back(out->getComputeAtAxis(out_i).first);
  }

  // At this point loop_structure contains our overal target loop nest structure
  // Lets get a copy of the loop structure, and figure out which loops we need
  // to open.
  auto loops_to_open = loop_structure;

  // Pop out loops already opened
  for (const auto& existing_loop : for_loops_) {
    if (loops_to_open.empty()) {
      // Nothing to open
      break;
    }
    if (gpu_lower->lowerValue(loops_to_open.front())->as<kir::IterDomain>() ==
        existing_loop->iter_domain()) {
      loops_to_open.pop_front();
    }
  }

  // At this point for_loops_ + loops_to_open contains our overal target loop
  // nest structure. Open loops in "loops_to_open".
  while (!loops_to_open.empty()) {
    openFor(loops_to_open.front());
    loops_to_open.pop_front();
  }

  kir::Expr* alloc_expr = nullptr;

  //  Place the expression
  pushBack(gpu_lower->lowerExpr(expr));

  // Reduce the loop nest structure back to computeAt
  if (out->getThisComputeAtAxis() == 0) {
    while (!for_loops_.empty()) {
      closeFor();
    }
  } else {
    const auto ca_axis = out->getThisComputeAtAxis() - 1;
    const auto target_domain =
        gpu_lower->lowerValue(out->getComputeAtAxis(ca_axis).first)
            ->as<kir::IterDomain>();
    while (!for_loops_.empty() &&
           for_loops_.back()->iter_domain() != target_domain) {
      closeFor();
    }
  }
}

// Generate the loop nest structure and place it in lowered_exprs_
void LoopNestGenerator::generate(const std::vector<Expr*>& exprs) {
  FusionGuard fg(fusion_);

  TORCH_INTERNAL_ASSERT(lowered_exprs_.empty());

  // Process the carefully ordered expressions
  for (const auto* expr : exprs) {
    handle(expr);
  }
}
// ======================================================
LoopNestGenerator2::LoopNestGenerator2(
    Fusion* fusion,
    const std::vector<Expr*>& exprs,
    const ComputeAtMap& ca_maps)
    : fusion_(fusion),
      ir_builder_(GpuLower::current()->kernel()),
      ca_maps_(ca_maps) {
  generate(exprs);
}

namespace {

// TODO(kir): revisit and try to simplify this
kir::ForLoop* openForHelper2(kir::ForLoop* scope, IterDomain* id) {
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());
  const auto kir_id = gpu_lower->lowerValue(id)->as<kir::IterDomain>();
  kir::ForLoop* new_scope = nullptr;
  if (id->isThread()) {
    std::stringstream ss;
    ss << id->getParallelType();
    new_scope = ir_builder.create<kir::ForLoop>(
        ir_builder.create<kir::NamedScalar>(ss.str(), DataType::Int),
        kir_id,
        scope);
  } else {
    new_scope = ir_builder.create<kir::ForLoop>(
        ir_builder.create<kir::Int>(c10::nullopt), kir_id, scope);
  }
  if (scope != nullptr) {
    scope->body().insert(0, new_scope);
  }
  return new_scope;
}

} // namespace

void LoopNestGenerator2::openFor(IterDomain* iter_domain) {
  if (for_loops_.size() > 0) {
    const auto new_scope = openForHelper2(for_loops_.back(), iter_domain);
    // for_loop_allocations_.insert({new_scope, 0});
    for_loops_.push_back(new_scope);
  } else {
    for_loops_.push_back(openForHelper2(nullptr, iter_domain));
    lowered_exprs_.insert(lowered_exprs_.begin(), for_loops_.back());
  }
}

void LoopNestGenerator2::closeFor() {
  TORCH_INTERNAL_ASSERT(!for_loops_.empty());
  for_loops_.pop_back();
}

void LoopNestGenerator2::pushFront(kir::Expr* expr) {
  if (for_loops_.size() == 0) {
    lowered_exprs_.insert(lowered_exprs_.begin(), expr);
  } else {
    for_loops_.back()->body().insert(0, expr);
  }
}

void LoopNestGenerator2::handle(const Expr* expr) {
  const auto gpu_lower = GpuLower::current();

  // Check if it's a tensor view expression we need to place in the loop nest
  // structure
  if (!ir_utils::isTVOp(expr)) {
    for (auto out : expr->outputs()) {
      TORCH_INTERNAL_ASSERT(
          out->getValType().value() == ValType::Scalar,
          "Unrecognized output type found in expr ",
          expr,
          " cannot lower ",
          out->getValType().value());

      pushFront(ir_builder_.create<kir::Allocate>(
          gpu_lower->lowerValue(out),
          MemoryType::Local,
          ir_builder_.create<kir::Int>(1)));
    }
    pushFront(gpu_lower->lowerExpr(expr));
    return;
  }

  TensorView* out_tv = expr->output(0)->as<TensorView>();

  // Figure out what the entire loop structure should look like.
  std::deque<IterDomain*> loop_structure;
  // Look at each axis individually in out's domain
  for (int64_t out_i = 0; out_i < (int64_t)ca_maps_.producedAt(out_tv);
       out_i++) {
    auto concrete_id = ca_maps_.getConcreteMappedID(out_tv->axis(out_i));
    auto parallel_id = ca_maps_.getParallelizedMappedID(out_tv->axis(out_i));
    concrete_id->parallelize(parallel_id->getParallelType());
    loop_structure.push_back(concrete_id);
  }

  for (int64_t out_i = (int64_t)ca_maps_.producedAt(out_tv);
       out_i < out_tv->nDims();
       out_i++) {
    loop_structure.push_back(out_tv->axis(out_i));
  }

  auto out_id_it = loop_structure.begin();
  auto for_loop_it = for_loops_.begin();
  auto last_for_loop_matched = for_loops_.begin();

  while (out_id_it != loop_structure.end() && for_loop_it != for_loops_.end()) {
    auto lowered_out_id =
        gpu_lower->lowerValue(*out_id_it)->as<kir::IterDomain>();
    if (ca_maps_.areMapped(lowered_out_id, (*for_loop_it)->iter_domain())) {
      out_id_it++;
      last_for_loop_matched = ++for_loop_it;
    } else {
      ++for_loop_it;
    }
  }

  auto n_loops_to_close =
      std::distance(last_for_loop_matched, for_loops_.end());
  for (size_t i = 0; i < n_loops_to_close; i++) {
    closeFor();
  }

  for (; out_id_it != loop_structure.end(); ++out_id_it) {
    openFor(*out_id_it);
  }

  pushFront(gpu_lower->lowerExpr(expr));
}

// Generate the loop nest structure and place it in lowered_exprs_
void LoopNestGenerator2::generate(const std::vector<Expr*>& exprs) {
  FusionGuard fg(fusion_);

  TORCH_INTERNAL_ASSERT(lowered_exprs_.empty());

  // Process the carefully ordered expressions
  for (auto it = exprs.rbegin(); it != exprs.rend(); ++it) {
    handle(*it);
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
