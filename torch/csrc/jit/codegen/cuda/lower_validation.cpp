#include <torch/csrc/jit/codegen/cuda/lower_validation.h>

#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

//! A parallel type validation pass to make sure all the outputs of
//!   welford ops are parallelized the same way. Will infer and modify serial
//!   parallel types if other output/s are parallelized, so that
//!   user wouldn't have to specify the same parallelization
//!   3 times. Will throw if conflicts are detected, i.e.
//!   TIDx vs BIDx etc.
class ValidateParallelType : public IterVisitor {
 public:
  static void validate(Fusion* fusion) {
    ValidateParallelType VPT;
    VPT.traverse(fusion);
  }

 private:
  using IterVisitor::handle;
  void convertIterDomain(IterDomain* id0, IterDomain* id1) {
    const auto ptype0 = id0->getParallelType();
    const auto ptype1 = id1->getParallelType();

    if (ptype0 != ptype1) {
      TORCH_CHECK(
          ptype0 == ParallelType::Serial || ptype1 == ParallelType::Serial,
          "Error promoting parallel types");
      if (ptype0 == ParallelType::Serial) {
        id0->parallelize(ptype1);
      }
      if (ptype1 == ParallelType::Serial) {
        id1->parallelize(ptype0);
      }
    }
  }

  void handle(WelfordOp* wop) override {
    auto out_var = wop->outVar()->as<TensorView>();
    auto out_avg = wop->outAvg()->as<TensorView>();
    auto out_n = wop->outN()->as<TensorView>();
    TORCH_INTERNAL_ASSERT(out_var->nDims() == out_avg->nDims());
    TORCH_INTERNAL_ASSERT(out_var->nDims() == out_n->nDims());
    for (size_t i = 0; i < out_var->nDims(); i++) {
      // TODO: can be cleaner.
      convertIterDomain(out_var->axis(i), out_avg->axis(i));
      convertIterDomain(out_avg->axis(i), out_n->axis(i));
      convertIterDomain(out_n->axis(i), out_var->axis(i));
    }
  }
};

} // namespace

void validateIr(Fusion* fusion) {
  FUSER_PERF_SCOPE("validateIr");

  FusionGuard fg(fusion);

  fusion->validateInputs();

  // Convert all output broadcast iterdomains to strided
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    for (auto id : tv->getMaybeRFactorDomain()) {
      if (id->isBroadcast()) {
        id->toStridedBroadcast();
      }
    }
  }

  // Validate Parallelization
  ValidateParallelType::validate(fusion);
}

namespace {

class VectorizeValidator : public OptInDispatch {
 private:
  VectorizeValidator(IterDomain* vectorized_id)
      : vectorized_id_(vectorized_id) {}

  using OptInDispatch::handle;

  void handle(Split* s) final {
    if (s->outer() == vectorized_id_) {
      is_valid = false;
    } else if (s->inner() == vectorized_id_) {
      vectorized_id_ = s->in();
    }
  }

  void handle(Merge* m) final {
    if (m->inner()->isBroadcast() && !m->outer()->isBroadcast()) {
      vectorized_id_ = m->outer();
    } else {
      vectorized_id_ = m->inner();
    }
  }

 private:
  IterDomain* vectorized_id_ = nullptr;
  bool is_valid = true;

 public:
  static void validate(TensorView* tv) {
    // Make sure there's only one vectorized ID
    IterDomain* v_id = nullptr;
    for (auto id : tv->domain()->domain()) {
      if (id->getParallelType() == ParallelType::Vectorize) {
        TORCH_INTERNAL_ASSERT(
            v_id == nullptr,
            "Found two vectorized domains in ",
            tv,
            " only one is allowed.");
        v_id = id;
      }
    }

    // If no vectorized id's found simply return;
    if (v_id == nullptr) {
      return;
    }

    auto fusion = FusionGuard::getCurFusion();

    TORCH_CHECK(
        v_id->rawExtent()->isConstScalar(),
        "Vectorizing a domain requires a constant size.");

    ExpressionEvaluator const_expr_eval(fusion);

    auto vector_size_optional = const_expr_eval.evaluate(v_id->rawExtent());

    TORCH_CHECK(
        vector_size_optional.has_value(),
        "Could not evalualte constant value bound to vectorized dim.");

    auto vector_size = ((int64_t)dataTypeSize(tv->getDataType().value())) *
        vector_size_optional.value();

    // Allow half2, float2, float4 and same sized vtypes.
    std::array<int64_t, 3> allowed_vector_sizes = {4, 8, 16}; // NOLINT

    TORCH_CHECK(
        std::find(
            allowed_vector_sizes.begin(),
            allowed_vector_sizes.end(),
            vector_size) != allowed_vector_sizes.end(),
        "Tried to vectorize a dim resulting in a word size of ",
        vector_size,
        " however, vector sizes only upto and including 16 bytes are supported.");

    auto replay_exprs = ExprSort::getExprs(fusion, {v_id});

    VectorizeValidator validator(v_id);

    for (auto expr_it = replay_exprs.rbegin(); expr_it != replay_exprs.rend();
         ++expr_it) {
      auto expr = *expr_it;
      validator.handle(expr);
    }

    TORCH_CHECK(
        validator.is_valid,
        "Invalid vectorized pattern found, vectorization iter domains must be descendants of inner most dimension.",
        "Issue found in, ",
        tv);

    TORCH_INTERNAL_ASSERT(validator.vectorized_id_ != nullptr);

    // TODO: Contiguity is based on root domain not rfactor. Seems this
    // generally doesn't cause problems, though contiguity should be on rfactor
    // domain as that's the domain we index on.
    IterDomain* last_root_dim = nullptr;
    int last_root_dim_pos = -1;
    for (size_t i = tv->getRootDomain().size(); i > 0; i--) {
      auto r_id = tv->getRootDomain()[i - 1];
      if (r_id->isReduction() || r_id->isBroadcast()) {
        continue;
      }
      last_root_dim = r_id;
      last_root_dim_pos = (int)i - 1;
      break;
    }

    if (last_root_dim == nullptr) {
      // Should never get here, but that would mean there are no concrete dims,
      // so we should be fine.
      return;
    }

    TORCH_CHECK(
        last_root_dim == validator.vectorized_id_ &&
            tv->domain()->contiguity()[last_root_dim_pos],
        "Vectorized dim has to be from a contiguous inner most position.");
  }
};

} // namespace

void validateVectorize(Fusion* fusion) {
  FUSER_PERF_SCOPE("validateVectorize");
  FusionGuard fg(fusion);

  auto used_vals = DependencyCheck::getAllValsBetween(
      {fusion->inputs().begin(), fusion->inputs().end()}, fusion->outputs());

  std::unordered_set<TensorView*> used_tvs;

  for (auto val : used_vals) {
    if (ir_utils::isTV(val)) {
      used_tvs.emplace(val->as<TensorView>());
    }
  }

  for (auto tv : used_tvs) {
    bool has_vectorize_dim = false;

    for (size_t i = 0; i < tv->nDims(); i++) {
      IterDomain* id = tv->axis(i);
      IterDomain* concrete_id =
          GpuLower::current()->caParallelMap().getConcreteMappedID(id);

      if (concrete_id->getParallelType() == ParallelType::Vectorize) {
        // If we want to do this check up front we would have to do 2 things:
        // (1) Check that the tensor view with vectorize being set on it is
        // getting it set outside the local compute at position
        // (2) Check any producers of the tensor view with vectorize being set
        // on it to make sure their compute at position isn't to the right of
        // the vectorize dim.
        TORCH_INTERNAL_ASSERT(
            i >= tv->getComputeAtPosition(),
            "IterDomains to the left of the compute at point cannot be vectorized.");
        has_vectorize_dim = true;
      }
    }
    if (has_vectorize_dim) {
      TORCH_INTERNAL_ASSERT(
          tv->definition() == nullptr ||
              (tv->definition()->isA<UnaryOp>() &&
               tv->definition()->as<UnaryOp>()->getUnaryOpType() ==
                   UnaryOpType::Set),
          "Vectorized accesses cannot be inline with computation, they are only supported with a Set operation.");
      VectorizeValidator::validate(tv);
    }
  }
}

void validateParallelize(Fusion* fusion) {
  FUSER_PERF_SCOPE("validateParallelize");
  FusionGuard fg(fusion);

  const auto& par_map = GpuLower::current()->caParallelMap();
  const auto& loop_map = GpuLower::current()->caLoopMap();

  auto exprs = ExprSort::getExprs(fusion);

  for (auto expr : exprs) {
    if (!ir_utils::isTVOp(expr)) {
      continue;
    }
    for (auto producer : ir_utils::filterByType<TensorView>(expr->inputs())) {
      // Parallelization on input tensors have no effect.
      if (producer->isFusionInput()) {
        continue;
      }
      for (size_t i = 0; i < producer->nDims(); ++i) {
        // If a producer axis is threaded, either with threadIdx or
        // blockIdx, there must be a mapped consumer axis with the
        // same ParallelType. An exception is when the producer is
        // allocated on shared memory and its parallelized with
        // threadIdx. In that case, there is no parallelization
        // constraint on the consumer as syncthreads will be inserted
        // when necessary.
        auto producer_axis = producer->axis(i);
        auto producer_ptype =
            par_map.getConcreteMappedID(producer_axis)->getParallelType();
        if (!isParallelTypeThread(producer_ptype)) {
          continue;
        }
        // No constraint on the consumer tensor when the producer
        // axis is parallelized with threadIdx and allocates on
        // shared memory
        if (isParallelTypeThreadDim(producer_ptype) &&
            producer->getMemoryType() == MemoryType::Shared) {
          continue;
        }
        // There should be also nothing to validate when the producer
        // axis is reduction.
        if (producer_axis->isReduction()) {
          continue;
        }
        // There must be a mappable consumer axis that has the same
        // parallel type.
        for (auto consumer :
             ir_utils::filterByType<TensorView>(expr->outputs())) {
          auto it = std::find_if(
              consumer->domain()->domain().begin(),
              consumer->domain()->domain().end(),
              [&](IterDomain* consumer_axis) {
                return loop_map.areMapped(producer_axis, consumer_axis);
              });
          TORCH_INTERNAL_ASSERT(
              it != consumer->domain()->domain().end(),
              "Inconsistent parallelization found between TV",
              producer->name(),
              " (",
              producer,
              ") and TV",
              consumer->name(),
              "(",
              consumer,
              "). ",
              "TV",
              consumer->name(),
              " does not have a matching axis for parallelized producer axis, ",
              producer_axis,
              ". CA Map: ",
              loop_map.toString());
          auto consumer_axis = *it;
          auto consumer_ptype =
              par_map.getConcreteMappedID(consumer_axis)->getParallelType();
          TORCH_INTERNAL_ASSERT(
              producer_ptype == consumer_ptype,
              "Inconsistent parallelization found between TV",
              producer->name(),
              " (",
              producer,
              ") and TV",
              consumer->name(),
              "(",
              consumer,
              "). "
              "Producer axis, ",
              producer_axis,
              " is parallelized with ",
              stringifyThread(producer_ptype),
              ", but the parallel type of its matching consumer axis, ",
              consumer_axis,
              " is ",
              stringifyThread(consumer_ptype),
              ".");
        }
      }
    }
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
