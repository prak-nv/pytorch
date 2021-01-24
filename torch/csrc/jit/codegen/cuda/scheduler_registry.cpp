#include <torch/csrc/jit/codegen/cuda/scheduler_registry.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

bool SchedulerEntry::operator==(const SchedulerEntry& other) {
  if (has_param_ != other.has_param_) {
    return false;
  }
  if (has_param_) {
    return rparams_ == other.rparams_;
  }
  return true;
}

namespace {
inline bool isTrivialReduction(ReductionOp* red) {
  auto o_tv = red->out()->as<TensorView>();
  // Assuming graph unscheduled at this point.
  for (auto id : o_tv->getRootDomain()) {
    if (id->isReduction() && !id->rawExtent()->isOneInt()) {
      return false;
    }
  }
  return true;
}
} // namespace

namespace {

class SingleReductionScheduler : public SchedulerEntry {
 public:
  explicit SingleReductionScheduler(Fusion* fusion, ExpressionEvaluator& ee)
      : SchedulerEntry(ScheduleHeuristic::Reduction, true) {
    getHeuristics(fusion, ee);
  }

  static bool canSchedule(Fusion* fusion) {
    int reduction_count = 0;
    for (auto expr : fusion->exprs()) {
      if (auto red = expr->as<ReductionOp>()) {
        // Check trivial
        if (!isTrivialReduction(red)) {
          reduction_count++;
          if (reduction_count > 1) {
            return false;
          }
        }
      }
    }
    return (reduction_count == 1);
  }

  void schedule(Fusion* fusion) override {
    // TODO find outputs of tv: what would we need to fill in?
    auto red_tv = findReductionTV(fusion);
    auto output_tv = findOutputsOfRed(fusion);
    scheduleReduction(fusion, rparams_, red_tv, output_tv);
  }

 private:
  void getHeuristics(Fusion* fusion, ExpressionEvaluator& ee) {
    auto red_tv = findReductionTV(fusion);
    auto param = getReductionHeuristics(fusion, ee, red_tv);
    TORCH_INTERNAL_ASSERT(param.has_value());
    rparams_ = param.value();
  }

  TensorView* findReductionTV(Fusion* fusion) {
    for (auto expr : fusion->exprs()) {
      if (auto red = expr->as<ReductionOp>()) {
        if (!isTrivialReduction(red)) {
          return red->out()->as<TensorView>();
        }
      }
    }
    TORCH_INTERNAL_ASSERT(false, "unreachable");
    return nullptr;
  }

  std::vector<TensorView*> findOutputsOfRed(Fusion* fusion) {
    return {};
  }
};

class PointWiseScheduler : public SchedulerEntry {
 public:
  explicit PointWiseScheduler(Fusion* fusion)
      : SchedulerEntry(ScheduleHeuristic::PointWise, false) {}

  static bool canSchedule(Fusion* fusion) {
    int reduction_count = 0;
    for (auto expr : fusion->exprs()) {
      if (auto red = expr->as<ReductionOp>()) {
        // Check trivial
        if (!isTrivialReduction(red)) {
          return false;
        }
      }
    }
    return (reduction_count == 1);
  }

  void schedule(Fusion* fusion) override {
    scheduleFusion(fusion);
  }
};

class NormalizationScheduler : public SchedulerEntry {
 public:
  explicit NormalizationScheduler(Fusion* fusion, ExpressionEvaluator& ee)
      : SchedulerEntry(ScheduleHeuristic::Normalization, true) {}

  void schedule(Fusion* fusion) override {
    return;
  }
};

// Schedule Table
const static std::vector<ScheduleHeuristic>& all_heuristics() {
  static const std::vector<ScheduleHeuristic> hlist = {
      ScheduleHeuristic::Reduction,
      ScheduleHeuristic::PointWise,
      ScheduleHeuristic::Normalization};
  return hlist;
}

// Simple dispatcher interface
bool canSchedule(ScheduleHeuristic sh, Fusion* fusion) {
  switch (sh) {
    case ScheduleHeuristic::PointWise:
      return false;
    case ScheduleHeuristic::Reduction:
      return false;
    case ScheduleHeuristic::Normalization:
      return false;
    default:
      TORCH_INTERNAL_ASSERT(false, "unreachable");
      return false;
  }
  return false;
}
} // namespace

std::unique_ptr<SchedulerEntry> SchedulerEntry::makeEntry(
    ScheduleHeuristic sh,
    Fusion* fusion,
    ExpressionEvaluator& ee) {
  switch (sh) {
    case ScheduleHeuristic::PointWise:
      return std::make_unique<PointWiseScheduler>(fusion);
    case ScheduleHeuristic::Reduction:
      return std::make_unique<SingleReductionScheduler>(fusion, ee);
    case ScheduleHeuristic::Normalization:
      return std::make_unique<NormalizationScheduler>(fusion, ee);
    default:
      TORCH_INTERNAL_ASSERT(false, "unreachable");
      return std::make_unique<PointWiseScheduler>(fusion);
  }
  return nullptr;
}

// Simply loop through the list as baseline strategy
c10::optional<ScheduleHeuristic> proposeHeuristics(Fusion* fusion) {
  for (auto sh : all_heuristics()) {
    if (canSchedule(sh, fusion)) {
      return sh;
    }
  }
  return c10::nullopt;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
