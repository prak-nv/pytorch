#include <torch/csrc/jit/codegen/cuda/lower_validation.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace{
  class ValidateParallelType : public IterVisitor {

 public:
  static void validate(Fusion* fusion) {
    ValidateParallelType VPT;
    VPT.traverse(fusion);
  }

 private:
  void convertIterDomain(IterDomain* id0, IterDomain* id1){
    const auto ptype0 = id0->getParallelType();
    const auto ptype1 = id1->getParallelType();

    if( ptype0 != ptype1){
      TORCH_CHECK(ptype0 == ParallelType::Serial || ptype1==ParallelType::Serial,"Error promoting parallel types");
      if (ptype0 == ParallelType::Serial){
        id0->parallelize(ptype1);
      }
      if (ptype1 == ParallelType::Serial){
        id1->parallelize(ptype0);
      }
    }
  }

  void handle(WelfordOp* wop) override {
    auto out_var = wop->outVar()->as<TensorView>();
    auto out_avg = wop->outAvg()->as<TensorView>();
    auto out_n = wop->outN()->as<TensorView>();
    TORCH_INTERNAL_ASSERT(out_var->nDims()==out_avg->nDims());
    TORCH_INTERNAL_ASSERT(out_var->nDims()==out_n->nDims());
    for(size_t i =0;i<out_var->nDims();i++){
      // TODO: can be cleaner.
      convertIterDomain(out_var->axis(i),out_avg->axis(i));
      convertIterDomain(out_avg->axis(i),out_n->axis(i));
      convertIterDomain(out_n->axis(i),out_var->axis(i));
    }
  }
};

}//namespace

void validateIr(Fusion* fusion) {
  FUSER_PERF_SCOPE("validateIr");

  FusionGuard fg(fusion);

  auto used_vals = DependencyCheck::getAllValsBetween(
      {fusion->outputs().begin(), fusion->outputs().end()}, fusion->inputs());

  std::unordered_set<TensorView*> used_tvs;

  for (auto val : used_vals) {
    if (ir_utils::isTV(val)) {
      used_tvs.emplace(val->as<TensorView>());
    }
  }

  fusion->validateInputs();

  for (auto tv : used_tvs) {
    for (decltype(tv->nDims()) i{0}; i < tv->nDims(); i++) {
      IterDomain* id = tv->getComputeAtAxis(i).first;

      if (id->isBlockDim()) {
        TORCH_CHECK(
            !id->isBroadcast(),
            "Parallelization across blocks on broadcast axes is not supported, but found on, ",
            tv,
            ".");
      }
      if (tv->hasBroadcast() && tv->getMemoryType() != MemoryType::Global) {
        auto td = tv->domain()->domain();
        auto ca_inputs = ir_utils::iterDomainInputsOf(
            {td.begin(), td.begin() + tv->getThisComputeAtAxis()});
        auto non_ca_inputs = ir_utils::iterDomainInputsOf(
            {td.begin() + tv->getThisComputeAtAxis(), td.end()});

        std::unordered_set<IterDomain*> ca_inputs_set(
            ca_inputs.begin(), ca_inputs.end());
        std::unordered_set<IterDomain*> non_ca_inputs_set(
            non_ca_inputs.begin(), non_ca_inputs.end());

        for (auto id : tv->getRootDomain()) {
          if (id->isBroadcast()) {
            // If a broadcast dimension is an input to both an axis within the
            // computeAt point and outside the compute at point we would have to
            // look at consumers to figure out what that axis will be
            // broadcasted to, because we would have to generate everything the
            // consumer could need on that axis. This could be supported but is
            // not at this point.
            TORCH_INTERNAL_ASSERT(
                !(ca_inputs_set.find(id) != ca_inputs_set.end() &&
                  non_ca_inputs_set.find(id) != non_ca_inputs_set.end()),
                "Cannot generate a kernel where a root broadcast dimension is input to both IterDomains outside and within the computeAt point.");
          }
        }
      }
    }
  }

  // Convert all output broadcast iterdomains to strided
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    for (auto id : tv->getMaybeRFactorDomain()) {
      if (id->isBroadcast()) {
        id->toStridedBroadcast();
      }
    }
  }

  //Validate Parallelization
  ValidateParallelType::validate(fusion);

}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
