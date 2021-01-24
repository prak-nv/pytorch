#pragma once

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TORCH_CUDA_API SchedulerEntry {
 public:
  static std::unique_ptr<SchedulerEntry> makeEntry(
      ScheduleHeuristic sh,
      Fusion* fusion,
      ExpressionEvaluator& ee);
  static c10::optional<ScheduleHeuristic> proposeHeuristics(Fusion* fusion);

  bool operator==(const SchedulerEntry& other);
  virtual void schedule(Fusion* fusion) = 0;

  ScheduleHeuristic heuristc() {
    return heuristc_;
  }

  bool hasParam() {
    return has_param_;
  }

  ReductionParams params() {
    return rparams_;
  }

 protected:
  explicit SchedulerEntry(ScheduleHeuristic heuristic, bool has_param)
      : heuristc_(heuristic), has_param_(has_param) {}
  const ScheduleHeuristic heuristc_;
  const bool has_param_;
  // TODO: will need a generic params type at some point
  ReductionParams rparams_;
};

c10::optional<SchedulerEntry> getScheduler(Fusion* fusion);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
