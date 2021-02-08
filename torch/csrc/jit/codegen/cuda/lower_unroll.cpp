#include <torch/csrc/jit/codegen/cuda/lower_unroll.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Provide a new for loop matching the one provided, sets parent_scope as
// parent_scope, but does not insert into parent scope.
kir::ForLoop* cloneLoopNest(
    const kir::ForLoop* for_loop,
    kir::Expr* parent_scope) {
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());
  const auto new_loop = ir_builder.create<kir::ForLoop>(
      for_loop->index(),
      for_loop->start(),
      for_loop->extent(),
      for_loop->iter_domain(),
      parent_scope);
  for (auto expr : for_loop->body().exprs()) {
    if (auto nested_for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      expr = cloneLoopNest(nested_for_loop, new_loop);
    }
    new_loop->body().push_back(expr);
  }
  return new_loop;
}

// Provide a new for loop matching the one provided, sets parent_scope as
// parent_scope, but does not insert into parent scope.
// Replace Set operations with VectorizeSet
kir::ForLoop* cloneLoopNestVectorize(
    const kir::ForLoop* for_loop,
    kir::Expr* parent_scope) {
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());
  const auto new_loop = ir_builder.create<kir::ForLoop>(
      for_loop->index(),
      for_loop->start(),
      for_loop->extent(),
      for_loop->iter_domain(),
      parent_scope);
  for (auto expr : for_loop->body().exprs()) {
    if (auto nested_for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      expr = cloneLoopNest(nested_for_loop, new_loop);
    } else if (
        expr->isA<kir::UnaryOp>() &&
        expr->as<kir::UnaryOp>()->operation() == UnaryOpType::Set) {
      auto unaryOp = expr->as<kir::UnaryOp>();
      auto input = unaryOp->in()->as<kir::TensorView>();
      auto output = unaryOp->out()->as<kir::TensorView>();
      expr = ir_builder.create<kir::UnaryOp>(
          UnaryOpType::VectorizeSet, output, input);
    }
    new_loop->body().push_back(expr);
  }
  return new_loop;
}

kir::Val* generateIndex(
    kir::Expr* expr,
    std::vector<kir::ForLoop*> for_loops) {
  TORCH_INTERNAL_ASSERT(expr != nullptr && expr->isA<kir::UnaryOp>() &&
      expr->as<kir::UnaryOp>()->operation() == UnaryOpType::Set);

  auto unaryOp = expr->as<kir::UnaryOp>();
  auto input = unaryOp->in()->as<kir::TensorView>();
  auto output = unaryOp->out()->as<kir::TensorView>();

  bool isVectorizedRead = output->memoryType() == MemoryType::Local &&
      input->memoryType() == MemoryType::Global;

  auto tensor_index = (isVectorizedRead)
      ? Index::getProducerIndex(
            input->fuserTv(), output->fuserTv(), for_loops)
      : Index::getConsumerIndex(input->fuserTv(), for_loops);

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());
  kir::Val* index = nullptr;
  for (auto idx : tensor_index->indices()) {
    index = (index == nullptr) ? idx : ir_builder.addExpr(index, idx);
  }
  return index;
}

void cloneLoopNestVectorizeMisaligned(
    kir::ForLoop* vec_loop,
    std::vector<kir::ForLoop*> nested_loops,
    kir::IfThenElse* vectorized_ite) {
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  // Original:
  // for (extent / vector_size)
  //    for (vector_size)
  //        // read
  //        // computation
  //    end for
  // end for

  auto outer_loop = nested_loops.back();
  for(auto expr : outer_loop->body().exprs())
    expr->print();

  TORCH_CHECK(outer_loop->body().exprs().size() == 1);
  auto outer_expr = outer_loop->body().exprs().front();
  TORCH_CHECK(
      outer_expr != nullptr && outer_expr->isA<kir::ForLoop>());

  auto front_expr = vec_loop->body().exprs().front();
  auto back_expr = vec_loop->body().exprs().front();

  bool valid_read = front_expr != nullptr && front_expr->isA<kir::UnaryOp>() &&
      front_expr->as<kir::UnaryOp>()->operation() == UnaryOpType::Set;
  bool valid_write = back_expr != nullptr && back_expr->isA<kir::UnaryOp>() &&
      back_expr->as<kir::UnaryOp>()->operation() == UnaryOpType::Set;
  TORCH_CHECK(valid_read || valid_write);

  // max_iter = F1.extent()
  // vector_size = F2.extent()
  // extent = max_iter * vector_size

  // base_address = generateIndex()
  // shift = base_address % vector_size
  // outer_shift = (shift > 0)
  // remainder = (extent - shift) % vector_size
  // vec_iter = (extent - shift) / vector_size

  const auto zero = ir_builder.create<kir::Int>(0);
  const auto one = ir_builder.create<kir::Int>(1);

  const auto max_iter = outer_loop->extent();
  const auto vector_size = vec_loop->extent();
  const auto extent = ir_builder.mulExpr(max_iter, vector_size);

  nested_loops.push_back(vec_loop);
  auto base_index = generateIndex((valid_read) ? front_expr : back_expr, nested_loops);

  auto shift = ir_builder.modExpr(base_index, vector_size);
  auto outer_shift = ir_builder.ltExpr(shift, zero);
  auto remaining_extent = ir_builder.subExpr(extent, shift);
  auto remainder = ir_builder.modExpr(remaining_extent, vector_size);
  auto vec_iter = ir_builder.divExpr(remaining_extent, vector_size);

  // Old: index, iter_domain, parent_scope
  // New: index, iter_domain, initial, extent, parent_scope

  // Pre:
  // for (0; start_shift)
  //    for (shift)
  //        // read
  //        // computation
  //    end for
  // end for
  const auto pre_loop = ir_builder.create<kir::ForLoop>(
      outer_loop->index(),
      outer_loop->start(),
      outer_shift,
      outer_loop->iter_domain(),
      vectorized_ite);
  vectorized_ite->thenBody().push_back(pre_loop);

  // clone vec_loop operations
  const auto pre_vec_loop = ir_builder.create<kir::ForLoop>(
    vec_loop->index(),
    vec_loop->start(),
    shift,
    vec_loop->iter_domain(),
    pre_loop);
  for (auto expr : vec_loop->body().exprs()) {
    pre_vec_loop->body().push_back(expr);
  }
  pre_loop->body().push_back(pre_vec_loop);

  // Vectorize:
  // for (0; vec_iter)
  //    for (vector_size - vectorized)
  //        // read
  //    end for
  //    for (vector_size)
  //        // computation
  //    end for
  // end for 
  const auto vectorized_loop = ir_builder.create<kir::ForLoop>(
      outer_loop->index(),
      zero,
      vec_iter,
      outer_loop->iter_domain(),
      vectorized_ite);
  vectorized_ite->thenBody().push_back(vectorized_loop);

  if (valid_read) {
    const auto vectorized_read_loop = ir_builder.create<kir::ForLoop>(
        outer_loop->index(),
        zero,
        one,
        outer_loop->iter_domain(),
        vectorized_loop);

    auto unaryOp = front_expr->as<kir::UnaryOp>();
    auto input = unaryOp->in()->as<kir::TensorView>();
    auto output = unaryOp->out()->as<kir::TensorView>();

    auto vec_read =
      ir_builder.create<kir::UnaryOp>(UnaryOpType::VectorizeSet, output, input);
    vectorized_loop->body().push_back(vectorized_read_loop);
  }

  // clone computation ops in vec_loop
  // ignore set operations
  const auto computation_loop = ir_builder.create<kir::ForLoop>(
    vec_loop->index(),
    vec_loop->start(),
    vec_loop->extent(),
    vec_loop->iter_domain(),
    vectorized_loop);
  for (auto expr : vec_loop->body().exprs()) {
    bool isSetOp = expr->isA<kir::UnaryOp>() &&
      expr->as<kir::UnaryOp>()->operation() == UnaryOpType::Set;
    if (!isSetOp) {
      computation_loop->body().push_back(expr);
    }
  }
  vectorized_loop->body().push_back(computation_loop);

  if (valid_write) {
    const auto vectorized_write_loop = ir_builder.create<kir::ForLoop>(
        outer_loop->index(),
        outer_loop->start(),
        one,
        outer_loop->iter_domain(),
        vectorized_loop);

    auto unaryOp = back_expr->as<kir::UnaryOp>();
    auto input = unaryOp->in()->as<kir::TensorView>();
    auto output = unaryOp->out()->as<kir::TensorView>();

    auto vec_write =
      ir_builder.create<kir::UnaryOp>(UnaryOpType::VectorizeSet, output, input);
    vectorized_loop->body().push_back(vectorized_write_loop);
  }

  // Post:
  // for (vec_iter; max_iter)
  //    for (remainder)
  //        // read
  //        // computation
  //    end for
  // end for
  const auto post_loop = ir_builder.create<kir::ForLoop>(
      outer_loop->index(),
      vec_iter,
      max_iter,
      outer_loop->iter_domain(),
      vectorized_ite);
  vectorized_ite->thenBody().push_back(post_loop);

  // clone vec_loop with remainder extent
  const auto post_vec_loop = ir_builder.create<kir::ForLoop>(
    vec_loop->index(),
    zero,
    remainder,
    vec_loop->iter_domain(),
    post_loop);
  for (auto expr : vec_loop->body().exprs()) {
    post_vec_loop->body().push_back(expr);
  }
  post_loop->body().push_back(post_vec_loop);
}

// Returns true if expr is an expression that initializes a reduction
// buffer.
bool isReductionInitExpr(const kir::Expr* expr) {
  // False if its output isn't a TensorView
  if (!ir_utils::isTVOp(expr)) {
    return false;
  }
  // False if it doesn't have any reduction axis
  const auto out_tv = expr->outputs()[0]->as<kir::TensorView>();
  if (!out_tv->domain()->hasReduction()) {
    return false;
  }
  // False if it has have TensorView inputs as initialization should
  // never use TensorViews
  const auto tv_filter_inp_view =
      ir_utils::filterByType<kir::TensorView>(expr->inputs());
  if (tv_filter_inp_view.begin() != tv_filter_inp_view.end()) {
    return false;
  }
  return true;
}

} // namespace

kir::Bool* UnrollPass::getThreadPredicate(const kir::TensorView* tv) {
  // No thread predicate is needed predicate when tv is output of a
  // parallel broadcast expression.
  if (auto bop = dynamic_cast<kir::BroadcastOp*>(tv->definition())) {
    TORCH_INTERNAL_ASSERT(bop->out()->isA<kir::TensorView>());
    const auto out = bop->out()->as<kir::TensorView>()->fuserTv();
    if (ir_utils::getParallelBroadcastDomains(out, thread_predicates_).any()) {
      return nullptr;
    }
  }
  return thread_predicates_.getExpr(tv->fuserTv());
}

void UnrollPass::handle(kir::Expr* expr) {
  if (ir_utils::isTVOp(expr)) {
    // If tv op, predicate it
    const auto out_tv = expr->outputs()[0]->as<kir::TensorView>();
    const bool should_predicate = !for_loops_.empty() ||
        out_tv->memoryType() == MemoryType::Global ||
        out_tv->memoryType() == MemoryType::Shared;
    if (!should_predicate) {
      return;
    }
    kir::IrBuilder ir_builder(GpuLower::current()->kernel());
    const auto thread_pred = isReductionInitExpr(expr)
        ? ir_builder.create<kir::Bool>(true)
        : getThreadPredicate(out_tv);
    const auto pred =
        PredicateCompute::getInlinePredicate(expr, for_loops_, thread_pred);

    // If we need a predicate, put expr inside an if then else
    if (!pred->isConst() || !(pred->isConst() && pred->value().value())) {
      non_trivial_pred_found_ = true;
      kir::ForLoop* insert_scope =
          for_loops_.empty() ? nullptr : for_loops_.back();
      kir::IfThenElse* inline_ite =
          ir_builder.create<kir::IfThenElse>(pred, insert_scope);
      inline_ite->thenBody().push_back(expr);
      if (for_loops_.empty()) {
        // Special handling for top level output expressions that still
        // need predicates. One motivating example is a reduction op that
        // reduces to a scalar (issue #491)
        loop_replacement_map_.insert({expr, inline_ite});
      } else {
        for_loops_.back()->body().insert_before(expr, inline_ite);
        for_loops_.back()->body().erase(expr);
      }
    }
  } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
    handle(for_loop);
  }
}

// We should factor our actual predicate generation from unrolling but insering
// IR nodes "unroll_pred" or "inline_pred", then generate those later.
void UnrollPass::handle(kir::ForLoop* fl) {
  // Setup for loop scoping
  const bool is_unroll =
      fl->iter_domain()->parallelType() == ParallelType::Unroll ||
      fl->iter_domain()->parallelType() == ParallelType::Unswitch ||
      fl->iter_domain()->parallelType() == ParallelType::Vectorize ||
      fl->iter_domain()->parallelType() == ParallelType::VectorizeMisaligned;

  // If we're not looking for an unroll loop, or didn't find one, process as
  // normal.
  if (!is_unroll || !look_for_unroll_) {
    for_loops_.push_back(fl);

    // Make copy of exprs because we replace them inplace in fl
    const auto exprs_copy = fl->body().exprs();
    for (auto expr : exprs_copy) {
      handle(expr);
    }

    for_loops_.pop_back();
    return;
  }

  auto unroll_pred = UnswitchPredicate::get(for_loops_, fl, p2c_root_map_);

  kir::ForLoop* parent_scope = for_loops_.empty() ? nullptr : for_loops_.back();

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());
  kir::IfThenElse* unroll_ite =
      ir_builder.create<kir::IfThenElse>(unroll_pred, parent_scope);

  // Get the loop nest for the unrolled path
  if (fl->iter_domain()->parallelType() == ParallelType::Vectorize) {
    kir::ForLoop* unrolled_loop_nest = cloneLoopNestVectorize(fl, unroll_ite);
    unroll_ite->thenBody().push_back(unrolled_loop_nest);
    loop_replacement_map_.insert({fl, unroll_ite});
    return;
  } else if (fl->iter_domain()->parallelType() == ParallelType::VectorizeMisaligned) {
    cloneLoopNestVectorizeMisaligned(fl, for_loops_, unroll_ite);
    loop_replacement_map_.insert({fl, unroll_ite});
    return;
  } else {
    kir::ForLoop* unrolled_loop_nest = cloneLoopNest(fl, unroll_ite);
    unroll_ite->thenBody().push_back(unrolled_loop_nest);
  }

  // Loop nest for inlined path
  kir::ForLoop* inlined_loop = cloneLoopNest(fl, unroll_ite);

  // Add inline predicates for inlined loop nest
  look_for_unroll_ = false;
  non_trivial_pred_found_ = false;
  handle(inlined_loop);
  look_for_unroll_ = true;
  if (!non_trivial_pred_found_) {
    inlined_loop->setParentScope(parent_scope);
    loop_replacement_map_.insert({fl, inlined_loop});
  } else {
    unroll_ite->elseBody().push_back(inlined_loop);
    loop_replacement_map_.insert({fl, unroll_ite});
  }
}

// Generate the loop nest structure and place it in lowered_exprs
void UnrollPass::computeMap(const std::vector<kir::Expr*>& exprs) {
  FUSER_PERF_SCOPE("UnrollPass::computeMap");

  // Run through loop nests and further lower the expressions
  for (auto* expr : exprs) {
    handle(expr);
  }
}

// TODO(kir): incorporate this into a new Scope interface
kir::Expr* UnrollPass::applyReplacements(kir::Expr* expr) const {
  auto handle_scope = [this](kir::Scope& scope) {
    for (size_t i = 0; i < scope.size(); ++i) {
      scope[i] = applyReplacements(scope[i]);
    }
  };

  const auto it = loop_replacement_map_.find(expr);
  if (it != loop_replacement_map_.end()) {
    return it->second;
  } else {
    if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      handle_scope(for_loop->body());
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      handle_scope(ite->thenBody());
      handle_scope(ite->elseBody());
    }
    return expr;
  }
}

std::vector<kir::Expr*> UnrollPass::runPass(
    Fusion* fusion,
    const std::vector<kir::Expr*>& exprs,
    const ThreadPredicateMap& thread_predicates) {
  FUSER_PERF_SCOPE("UnrollPass::runPass");

  UnrollPass unroll_pass(fusion, thread_predicates);
  unroll_pass.computeMap(exprs);

  std::vector<kir::Expr*> mutated_exprs;
  mutated_exprs.reserve(exprs.size());
  for (auto expr : exprs) {
    mutated_exprs.push_back(unroll_pass.applyReplacements(expr));
  }

  return mutated_exprs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
