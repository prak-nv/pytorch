#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_allocation.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

class AllocationInserter : public kir::MutableIrVisitor {
 private:
  struct AllocationInformation {
    // The for loop that the allocation must be placed in, nullptr if not within
    // a loop
    kir::ForLoop* for_loop = nullptr;

    // The expression that this allocation must be placed before
    kir::Expr* place_before = nullptr;

    // The allocation position relative to buffer
    size_t alloc_pos = 0;

    // The buffer this allocation is for
    kir::TensorView* buffer = nullptr;

    // The allocation expression
    kir::Allocate* alloc_expr = nullptr;

    // Initialization
    kir::Expr* init_expr = nullptr;
  };

  // Find allocation point
  void findAllocationPosition(AllocationInformation& info, kir::Expr* expr) {
    size_t alloc_pos = 0;
    kir::ForLoop* for_loop = nullptr;
    auto fuser_tv = info.buffer->fuserTv();
    size_t fl_idx_next = 0;

    for (auto fl : for_loops) {
      if (alloc_pos == fuser_tv->getThisComputeAtAxis()) {
        break;
      }

      if (fuser_tv->axis(alloc_pos)->isReduction()) {
        const auto outputs =
            FusionGuard::getCurFusion()->getTerminatingOutputs();
        TORCH_INTERNAL_ASSERT(
            std::find(outputs.begin(), outputs.end(), fuser_tv) !=
                outputs.end(),
            "Invalid computeAt of T",
            fuser_tv->name(),
            ". A reducation axis is detected within computeAt axes even though it is not an output tensor.");
        break;
      }

      auto fl_id = fl->iter_domain();

      if (fl_id->parallelType() == ParallelType::Unroll) {
        break;
      }

      auto local_id = gpu_lower->lowerValue(fuser_tv->axis(alloc_pos))
                          ->as<kir::IterDomain>();

      if (gpu_lower->caLoopMap().areMapped(local_id, fl_id)) {
        alloc_pos++;
      }

      for_loop = fl;
      ++fl_idx_next;
    }

    info.alloc_pos = alloc_pos;
    info.for_loop = for_loop;

    if (info.for_loop == nullptr) {
      info.place_before = for_loops.size() > 0 ? for_loops[0] : expr;
    } else {
      if (info.for_loop == for_loops.back()) {
        // Inline allocation, place before expr
        info.place_before = expr;
      } else {
        // Place allocation after the last computeAt axis
        // TODO: may be more efficient to place before the first non-computeAt
        // axis
        info.place_before = for_loops.at(fl_idx_next);
      }
    }
  }

  // Create initialization expression if init_val is non-null.
  void createInitExpr(AllocationInformation& info, kir::Val* init_val) {
    if (init_val == nullptr) {
      info.init_expr = nullptr;
      return;
    }

    auto fuser_tv = info.buffer->fuserTv();

    std::vector<kir::IterDomain*> init_dims;
    for (size_t axis_i = info.alloc_pos; axis_i < fuser_tv->nDims(); axis_i++) {
      if (info.buffer->fuserTv()->axis(axis_i)->isReduction() ||
          info.buffer->fuserTv()->axis(axis_i)->isBroadcast()) {
        continue;
      }
      auto concrete_id =
          gpu_lower
              ->lowerValue(gpu_lower->caParallelMap().getConcreteMappedID(
                  fuser_tv->axis(axis_i)))
              ->as<kir::IterDomain>();
      init_dims.push_back(concrete_id);
    }
    kir::Expr* init_expr = ir_builder.create<kir::UnaryOp>(
        UnaryOpType::Set, info.buffer, init_val);
    for (auto init_loop_it = init_dims.rbegin();
         init_loop_it != init_dims.rend();
         ++init_loop_it) {
      auto id = *init_loop_it;
      kir::ForLoop* new_loop = nullptr;
      if (isParallelTypeThread((*init_loop_it)->parallelType())) {
        std::stringstream ss;
        ss << id->parallelType();
        new_loop = ir_builder.create<kir::ForLoop>(
            ir_builder.create<kir::NamedScalar>(ss.str(), DataType::Int),
            id,
            nullptr);
      } else {
        new_loop = ir_builder.create<kir::ForLoop>(
            ir_builder.create<kir::Int>(c10::nullopt), id, nullptr);
      }
      init_expr->setParentScope(new_loop);
      new_loop->body().push_back(init_expr);
      init_expr = new_loop;
    }
    info.init_expr = init_expr;
  }

  void createAllocExpr(AllocationInformation& info, bool is_output) {
    if (is_output) {
      info.alloc_expr = nullptr;
      return;
    }

    auto fuser_tv = info.buffer->fuserTv();

    std::vector<kir::Val*> alloc_dims;
    const MemoryType memory_type = info.buffer->memoryType();
    for (size_t axis_i = 0; axis_i < fuser_tv->nDims(); axis_i++) {
      const auto local_id =
          gpu_lower->lowerValue(fuser_tv->axis(axis_i))->as<kir::IterDomain>();

      if (
          // If we're reducing this dimension, don't use it in the allocation
          // computation
          local_id->isReduction() ||
          // If this is a broadcast dimension, don't use it in the allocation
          // computation
          local_id->isBroadcast()) {
        continue;
      }

      auto concrete_id =
          gpu_lower
              ->lowerValue(gpu_lower->caParallelMap().getConcreteMappedID(
                  fuser_tv->axis(axis_i)))
              ->as<kir::IterDomain>();
      const bool is_block_dim =
          isParallelTypeBlockDim(concrete_id->parallelType());
      const bool is_thread_dim =
          isParallelTypeThreadDim(concrete_id->parallelType());
      const bool is_thread = isParallelTypeThread(concrete_id->parallelType());

      if (axis_i < info.alloc_pos) {
        // Even when the axis is outside the allocation position, if the
        // tensor is shared with respect to the axis, the buffer size
        // needs to be expanded for the axis. Sharing occurs in two
        // cases: 1) the tensor is on shared memory with the axis
        // parallelized by TIDs, and 2) the tensor is on global memory
        // with the axis parallelized by TIDs or BIDs.
        if (!((memory_type == MemoryType::Shared && is_thread_dim) ||
              (memory_type == MemoryType::Global && is_thread))) {
          continue;
        }
      } else {
        if (
            // If shared memory, don't use any IDs bound to a grid dimension
            (memory_type == MemoryType::Shared && is_block_dim) ||
            // If local memory, don't use any IDs bound to a grid or block
            // dimension
            (memory_type == MemoryType::Local && is_thread)) {
          continue;
        }
      }
      alloc_dims.push_back(concrete_id->rawExtent());
    }

    // Multiply all the dimensions we're going to use for the allocation
    // together to get the total size
    kir::Val* size = nullptr;
    if (alloc_dims.size() == 0) {
      size = ir_builder.create<kir::Int>(1);
    } else {
      size = alloc_dims[0];
      for (size_t i = 1; i < alloc_dims.size(); i++) {
        size = ir_builder.mulExpr(size, alloc_dims[i]);
      }
    }

    // Create the allocation node
    info.alloc_expr = ir_builder.create<kir::Allocate>(
        info.buffer, info.buffer->memoryType(), size);
  }

  void handle(kir::Expr* expr) {
    if (!ir_utils::isTVOp(expr) || expr->isA<kir::Allocate>()) {
      expr->accept(this);
      return;
    }

    // // Found where the allocation needs to be inserted

    for (auto out : expr->outputs()) {
      if (!out->isA<kir::TensorView>()) {
        continue;
      }

      auto out_tv = out->as<kir::TensorView>();

      kir::Val* init = nullptr;
      if (expr->isA<kir::ReductionOp>() && out_tv->fuserTv()->hasReduction()) {
        init = expr->as<kir::ReductionOp>()->init();
      }

      const bool is_output = std::find(
                                 gpu_lower->kernel()->outputs().begin(),
                                 gpu_lower->kernel()->outputs().end(),
                                 out) != gpu_lower->kernel()->outputs().end();

      // Don't need to alloc outputs, and if we don't need to initialize we're
      // done.
      if (is_output && init == nullptr) {
        continue;
      }

      AllocationInformation allocation;
      allocation.buffer = out_tv;
      findAllocationPosition(allocation, expr);
      createAllocExpr(allocation, is_output);
      createInitExpr(allocation, init);

      allocs.push_back(allocation);
    }
  }

  void visit(kir::ForLoop* fl) final {
    for_loops.push_back(fl);
    // Modifying in place, make a copy of the vector
    const std::vector<kir::Expr*> exprs = fl->body().exprs();
    for (auto expr : exprs) {
      handle(expr);
    }
    for_loops.pop_back();
  }

  void visit(kir::IfThenElse*) final {
    TORCH_INTERNAL_ASSERT(
        false,
        "Pass does not support conditional statements, ",
        "this pass should be run before any conditionals are placed in code.");
  }

  AllocationInserter(std::vector<kir::Expr*> _loop_nests)
      : loop_nests_(std::move(_loop_nests)),
        gpu_lower(GpuLower::current()),
        ir_builder(gpu_lower->kernel()) {
    // Compute all allocations
    const std::vector<kir::Expr*> exprs = loop_nests_;
    for (auto expr : exprs) {
      handle(expr);
    }

    // First, place allocations of dynamic smem tensors at the very
    // beginning of the expr list. Traverse backward as they should be
    // placed in topological order.
    for (auto it = allocs.rbegin(); it != allocs.rend(); ++it) {
      const auto& alloc = *it;
      if (alloc.alloc_expr == nullptr) {
        continue;
      }
      // Dynamic smem exprs need to be at the begining of the kernel outside for
      // loops
      if (alloc.buffer->memoryType() == MemoryType::Shared &&
          !kir::ExpressionEvaluator::isConst(alloc.alloc_expr->size())) {
        loop_nests_.insert(loop_nests_.begin(), alloc.alloc_expr);
      }
    }

    // Place the remaining allocations.
    for (const auto& alloc : allocs) {
      if (alloc.alloc_expr == nullptr) {
        continue;
      }
      if (alloc.buffer->memoryType() == MemoryType::Shared &&
          !kir::ExpressionEvaluator::isConst(alloc.alloc_expr->size())) {
        continue;
      }
      if (alloc.for_loop == nullptr) {
        auto place_before_it = std::find(
            loop_nests_.begin(), loop_nests_.end(), alloc.place_before);
        TORCH_INTERNAL_ASSERT(
            place_before_it != loop_nests_.end(),
            "Could not figure out where to place allocation. ",
            "Use of the buffer, ",
            toString(alloc.buffer),
            ", could not be found.",
            toString(alloc.place_before));
        loop_nests_.insert(place_before_it, alloc.alloc_expr);
      } else {
        alloc.for_loop->body().insert_before(
            alloc.place_before, alloc.alloc_expr);
      }
    }

    // Now that allocations are in place, place the initializations
    for (const auto& alloc : allocs) {
      if (alloc.init_expr == nullptr) {
        continue;
      }
      if (alloc.for_loop == nullptr) {
        auto place_before_it = std::find(
            loop_nests_.begin(), loop_nests_.end(), alloc.place_before);
        // Don't need a check here as if the allocation placement succeeded
        // this will too
        loop_nests_.insert(place_before_it, alloc.init_expr);
      } else {
        alloc.for_loop->body().insert_before(
            alloc.place_before, alloc.init_expr);
        alloc.init_expr->setParentScope(alloc.for_loop);
      }
    }
  }

 private:
  std::deque<AllocationInformation> allocs;

  std::vector<kir::ForLoop*> for_loops;

  std::vector<kir::Expr*> loop_nests_;

  GpuLower* gpu_lower;

  kir::IrBuilder ir_builder;

 public:
  static std::vector<kir::Expr*> insert(
      const std::vector<kir::Expr*>& loop_nests) {
    AllocationInserter inserter(loop_nests);
    return inserter.loop_nests_;
  }
};

} // namespace

std::vector<kir::Expr*> insertAllocations(
    const std::vector<kir::Expr*>& exprs) {
  FUSER_PERF_SCOPE("insertAllocations");
  return AllocationInserter::insert(exprs);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
