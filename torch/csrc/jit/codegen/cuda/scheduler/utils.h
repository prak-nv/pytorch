#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/reduction_heuristic.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class ExpressionEvaluator;

namespace scheduler_utils {

// Return positions of reduction axes in provided tv
std::vector<int> reductionAxes(TensorView* tv);

// Merge all reduction to the right side and returns total number of***
// reduction axes
size_t mergeReduction(TensorView* tv);

// merge all non-reduction axes to the left side and returns total number of
// iteration axes
size_t mergeNonReduction(TensorView* tv);

int log2_ceil(int value);

void scheduleReductionComputeAt(
    TensorView* red_tv,
    TensorView* red_tv_rf,
    const std::vector<TensorView*>& outs_of_red);

// Makes rfactor generic with reduction ops and Welford
TensorView* rfactorHelper(TensorView* red_tv, const std::vector<int>& axes);

bool canDuplicate(const Expr* expr);

bool isConstantAllocation(const TensorView* tv);

//! Find all TensorViews that require duplication to avoid recompute
//! computeAt error when applying inline ComputeAt
std::vector<TensorView*> findTensorViewsToDuplicate(
    Fusion* fusion,
    const std::vector<TensorView*>& other_tv);

bool canComputeAtInline(TensorView* tv);

//! Find all TensorViews that require inline ComputeAt
//! to avoid non-static allocation error
std::vector<TensorView*> findTensorViewsToComputeAtInline(
    Fusion* fusion,
    const std::vector<TensorView*>& tensors);

//! Place all cache TensorViews in Shared Memory
//! All point-wise TensorViews inherit shared memory from their parents
void setupSharedMemory(
    Fusion* fusion,
    const std::vector<TensorView*>& cache_tv);

// TODO: Review this. Seems we should be using a root map here, or we should
// simply be replaying all tensors as a reduction tv.
void organizeAxes(
    const std::vector<TensorView*>& reduction_tv,
    const std::vector<TensorView*>& all_tv);

// If tv is broadcasted (used in a broadcast op) return that op, otherwise
// return nullptr
Expr* isBroadcasted(TensorView* tv);

// If tv is casted (used in a cast op) return that op, otherwise return nullptr
Expr* isCasted(TensorView* tv);

void handleCastBroadcastInput(Fusion* fusion, TensorView* input);

// TODO: Remove
void cacheInputs(
    Fusion* fusion,
    const ReductionParams& rparams,
    const std::vector<TensorView*>& reduction_tv,
    std::vector<TensorView*>& other_tv);

// TODO: Is there a use for this?
std::vector<TensorView*> producerTvsOf(TensorView* tv);

// TODO: Is there a use for this?
std::vector<TensorView*> consumerTvsOf(TensorView* tv);

// TODO: Is there a use for this?
std::vector<TensorView*> producerTvsOf(std::vector<TensorView*> tvs);

// TODO: Is there a use for this?
std::vector<TensorView*> consumerTvsOf(std::vector<TensorView*> tvs);

TORCH_CUDA_CU_API std::vector<TensorView*> allTvs(Fusion* fusion);

void parallelizeAllLike(
    TensorView* reference_tv,
    const std::vector<TensorView*>& all_tvs);

void computeAtInputs(
    TensorView* consumer,
    int pos,
    ComputeAtMode mode = ComputeAtMode::Standard);

void computeWithOutputs(
    TensorView* producer,
    int pos,
    ComputeAtMode mode = ComputeAtMode::Standard);

// returns all tensor views in fusion that are used between outputs and inputs.
// Order is non-deterministic and non-repeating.
// TODO: This would be good to have determinsitic and to put outside scheduling
// as it's generally useful
std::vector<TensorView*> allTvs(Fusion* fusion);

struct PersistentBufferInfo {
  std::vector<TensorView*> buffers;
  std::unordered_set<IterDomain*> unmappable_dims;
};

// Buffers whos roots can't map to all producer roots based on compute at. These
// are the buffers we would make persistent in a persistent kerenl or would have
// to recompute if we can't make a persistent kernel.
PersistentBufferInfo persistentBuffers(Fusion* fusion);

struct TvProperties {
  // How many elements in tensor view are there to reduce
  int64_t reduction_numel = 1;
  // How many reductions do we need to perform, i.e. how many iter dimension
  // elements are there
  int64_t iteration_numel = 1;
  // Do we reduce the fastest dimension, if no reduction mark true
  bool fastest_dim_reduction = true;
  // What's the iter numel to the left of the reduction (if there is one)
  int64_t iter_outside_red = 1;
  // What's the iter numel to the right of the reduction (if this is or isn't
  // one)
  int64_t iter_inside_red = 1;
};

// Fill TvProperties structure about tv
TvProperties getProperties(
    Fusion* fusion,
    ExpressionEvaluator& evaluator,
    TensorView* tv);

} // namespace scheduler_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
