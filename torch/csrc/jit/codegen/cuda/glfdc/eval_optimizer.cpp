#include <torch/csrc/jit/codegen/cuda/glfdc/eval_optimizer.h>

#include <torch/csrc/jit/codegen/cuda/glfdc/eval_uinstr.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/eval_vm.h>

#include <iostream>

using namespace torch::jit::fuser::cuda::glfdc;

namespace {

using OperandsStack = ExternalStack<scalar_type, StackDir::GrowsUp>;
using ScratchSpace = ExternalStack<scalar_type, StackDir::GrowsDown>;

static UOP createMathUOP(mop_kind op, mop_src_kind src, mop_dst_kind dst) {
  unsigned ret = unsigned(op) + unsigned(src) + unsigned(dst);
  TORCH_INTERNAL_ASSERT(
      ret < DST_KINDS_COUNT * SRC_KINDS_COUNT * MATH_OPS_COUNT,
      "Out of range mathematic");
  return UOP(ret);
}

template <typename Ty_>
static void setToAtleast(Ty_& rhs, Ty_ value) noexcept {
  rhs = std::max<Ty_>(rhs, value);
}

// Returns number of operands on operand stack required by math operation
std::size_t uopOperandCount(mop_src_kind kind) noexcept {
  switch (kind) {
    case mop_src_kind::roo:
      return 2;
    case mop_src_kind::loo:
      return 2;
    case mop_src_kind::ob:
      return 1;
    case mop_src_kind::ao:
      return 1;
    default:
      break;
  }
  return 0;
}

// Optimizes expression by creating program for interpretation.
class ExprOptimizer {
 public:
  static UProgram optimize(
      OperandStackOrder ORD,
      Expr e,
      const std::vector<Operation>& ops,
      const std::vector<BindingGapType>& bindings);

 private:
  ExprOptimizer(
      Expr e,
      const std::vector<Operation>& ops,
      const std::vector<BindingGapType>& bindings)
      : program_(),
        ops_(ops),
        gaps_(bindings),
        dag_(&e.dag()),
        reuses_(ReusedExprMapping::createForMemoizedEval(e.dag())) {}

  // Create SymbolicExpr optimized form
  template <OperandStackOrder ORD>
  UProgram optimizeExpr();

  // Create SymbolicExpr
  template <OperandStackOrder ORD>
  void optimizeSubExpr(std::size_t op_index, mop_dst_kind parent_need);

  void optimizeSubExprRL(std::size_t op_index, mop_dst_kind parent_need) {
    optimizeSubExpr<OperandStackOrder::RL>(op_index, parent_need);
  }

  void optimizeSubExprLR(size_t op_index, mop_dst_kind parent_need) {
    optimizeSubExpr<OperandStackOrder::LR>(op_index, parent_need);
  }

  // Helper for picking recall/remember instruction opcodes pair
  static std::pair<UOP, UOP> pickRecallRem(mop_dst_kind dst) {
    switch (dst) {
      case mop_dst_kind::a:
        return std::make_pair(UOP::recall_a, UOP::rem_a);
      case mop_dst_kind::b:
        return std::make_pair(UOP::recall_b, UOP::rem_b);
      case mop_dst_kind::t:
        return std::make_pair(UOP::recall_t, UOP::rem_t);
    }
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false, "Unhandled destination");
  }

  // Helper for creating mathematical operation opcode
  static mop_kind pickMathUOP(OperatorKind k) noexcept {
    switch (k) {
      case OperatorKind::add:
        return mop_kind::add;
      case OperatorKind::sub:
        return mop_kind::sub;
      case OperatorKind::mul:
        return mop_kind::mul;
      case OperatorKind::div:
        return mop_kind::div;
      case OperatorKind::mod:
        return mop_kind::mod;
      case OperatorKind::ceildiv:
        return mop_kind::ceildiv;
      default:
        break;
    }
    TORCH_INTERNAL_ASSERT(false, "Unhandled operator");
  }

  // Returns true if operation requires use of scratch space
  static bool needsArgInScratch(bool eval_second) noexcept {
    // When operand of evaluation as is not an expression and is evaluated as
    // second then this operand has to be temporarily stored for time of
    // evaluation first subexpression.
    return eval_second;
  }

  // Checks if scalar operand can fit into UInstr argument as immediate
  static bool scalarFitsUInstArg(Operand op) noexcept {
    TORCH_INTERNAL_ASSERT(isScalar(op), "Not a scalar operand");
    scalar_type v = c10::get<scalar_type>(c10::get<Value>(op));
    return v >= 0 && v <= scalar_type(UInst::MaxArg);
  }

  // Checks if scratch space usage can be elided.
  template <OperandStackOrder ORD>
  static bool canElideScratchUsage(const SExprNode& e) noexcept {
    Operand op = ORD == OperandStackOrder::LR ? e.lhs_ : e.rhs_;
    return isScalar(op) && scalarFitsUInstArg(op);
  }

  // Elides scratch space usage by placing argument as instruction immediate
  // (for LR)
  static void elideScratchSpaceUseLR(
      const SExprNode& e,
      unsigned& imm,
      mop_src_kind& src) noexcept {
    TORCH_INTERNAL_ASSERT(isSExpr(e.rhs_));
    TORCH_INTERNAL_ASSERT(canElideScratchUsage<OperandStackOrder::LR>(e));
    // Obtain value
    scalar_type v = c10::get<scalar_type>(c10::get<Value>(e.lhs_));
    // And place it in instruction immediate
    imm = v;
    TORCH_INTERNAL_ASSERT(src == mop_src_kind::tb, "Invalid src");
    // Update instruction source so it uses immediate
    src = mop_src_kind::ib;
  }

  // Elides scratch space usage but placing argument as instruction immediate
  // (for RL)
  static void elideScratchSpaceUseRL(
      const SExprNode& e,
      unsigned& imm,
      mop_src_kind& src) noexcept {
    TORCH_INTERNAL_ASSERT(src == mop_src_kind::at);
    TORCH_INTERNAL_ASSERT(isSExpr(e.lhs_));
    TORCH_INTERNAL_ASSERT(canElideScratchUsage<OperandStackOrder::RL>(e));
    scalar_type v = c10::get<scalar_type>(c10::get<Value>(e.rhs_));
    imm = v;
    // Update instruction source so it uses immediate
    src = mop_src_kind::ai;
  }

  template <OperandStackOrder ORD>
  static void elideScratchSpaceUse(
      const SExprNode& e,
      unsigned& imm,
      mop_src_kind& src) {
    if (ORD == OperandStackOrder::LR) {
      elideScratchSpaceUseLR(e, imm, src);
    } else {
      elideScratchSpaceUseRL(e, imm, src);
    }
  }

  // Picks where from operands of an subexpression come from (stack order LR).
  static mop_src_kind pickUOPSourceLR(
      bool eval_first,
      bool eval_second) noexcept {
    std::array<mop_src_kind, 2> potential_srcs;

    if (needsArgInScratch(eval_second)) {
      // Need to save for later the "a" operand from top of stack
      potential_srcs[0] = mop_src_kind::tb;
      potential_srcs[1] = mop_src_kind::tb;
    } else {
      potential_srcs[0] = mop_src_kind::ao;
      potential_srcs[1] = mop_src_kind::loo;
    }

    mop_src_kind src = (eval_first) ? potential_srcs[0] : potential_srcs[1];
    return src;
  }

  // Picks where from operands of an subexpression come from (stack order RL).
  static mop_src_kind pickUOPSourceRL(
      bool eval_first,
      bool eval_second) noexcept {
    std::array<mop_src_kind, 2> potential_srcs;

    if (needsArgInScratch(eval_second)) {
      potential_srcs[0] = mop_src_kind::at;
      potential_srcs[1] = mop_src_kind::at;
    } else {
      potential_srcs[0] = mop_src_kind::ob;
      potential_srcs[1] = mop_src_kind::roo;
    }

    mop_src_kind src = (eval_first) ? potential_srcs[0] : potential_srcs[1];
    return src;
  }

  // Picks where from operands of an subexpression come from.
  template <OperandStackOrder ORD>
  static mop_src_kind pickUOPSource(
      bool eval_first,
      bool eval_second) noexcept {
    return (ORD == OperandStackOrder::LR)
        ? pickUOPSourceLR(eval_first, eval_second)
        : pickUOPSourceRL(eval_first, eval_second);
  }

  // Calculates size of current reservation of temporary space
  void reserveScratch() {
    ++scratch_space_top_;
    // Update scratch space highwater to figure out required allocation
    setToAtleast(scratch_space_highwater_, scratch_space_top_);

    if (scratch_space_top_ > operands_stack_free_) {
      // Needs more space than available on operand stack
      int required = scratch_space_top_ - operands_stack_free_;
      setToAtleast(scratch_extra_allocation_, required);
    }
  }

  // Retains space reserved scratch space
  void retainScratch() {
    --scratch_space_top_;
    TORCH_INTERNAL_ASSERT(
        scratch_space_top_ >= 0, "Underflow of scratch space area");
  }

  void consumeOperands(std::size_t operands_count) {
    operands_stack_free_ += operands_count;
  }

  // Inserts instructions needed for recall and returns matching remember
  //
  // We obtain sequence like:
  // RECALL, RDROP_N, RSKIP_N, <value calculation>, REM
  // RECALL tries to obtain already calculated value.
  // - If that succeeds RDROP_N drops required operands from operand stack and
  // RSKIP_N advances to next instruction after REM instruction.
  // - If that fails RDROP_N and RSKIP_N do nothing value is calculated and
  // REM stores the calculated value.
  UInst beginRecallRemInstrSeq(
      const Operation& op,
      mop_dst_kind parent_request) {
    UOP recall, rem;
    // Choses where to place potential recall and where from remember should
    // be
    std::tie(recall, rem) = pickRecallRem(parent_request);

    // We may have obtain the value of that subexpression since it has reuses
    // We insert recall operation that can succeed and give us value
    TORCH_INTERNAL_ASSERT(
        op.ref.index() <= UInst::MaxArg, "Index is too big to encode");

    // Find slot index for subexpression
    auto slot = reuses_.getSlotIndex(op.ref);
    TORCH_INTERNAL_ASSERT(slot.has_value());

    program_.push_back(UInst{recall, unsigned(*slot)});
    // if a recall operation succeeds we should drop number operands from
    // stack
    program_.push_back(UInst{UOP::rdrop_n, op.noperands});

    constexpr unsigned INVALID_SKIP = 0;
    // If a recall operation succeeds we should drop number operands from
    // stack NOTE: The argument to skip operation is left invalid where do we
    // need to skip. It should be adjusted afterwards
    program_.push_back(UInst{UOP::rskip_n, INVALID_SKIP});

    return UInst{rem, unsigned(*slot)};
  }

  // Finishes RECALL/REM instruction sequence. See beginRecallRemInstSeq()
  // above.
  void finishRecallRemInstrSeq(c10::optional<UInst> rem, size_t rskip_n_idx) {
    TORCH_INTERNAL_ASSERT(rem.has_value());
    // We should remember value of an expression now that we calculated it
    // first time ever
    program_.push_back(*rem);
    // Now we can adjust skip. we want to skip to program operation right
    // after remember operation (-1 since next eval loop iteration will
    // increase IP)
    unsigned skip_arg = program_.size() - rskip_n_idx - 1;
    TORCH_INTERNAL_ASSERT(program_[rskip_n_idx].op == UOP::rskip_n);
    program_[rskip_n_idx].arg = skip_arg;
  }

  template <OperandStackOrder ORD>
  void insertMathInstr(
      const SExprNode& e,
      mop_dst_kind parent_request,
      bool eval_first,
      bool eval_second) {
    // Create Uop for destination
    mop_kind mop = pickMathUOP(e.op_);
    mop_src_kind src = pickUOPSource<ORD>(eval_first, eval_second);

    // Immediate value of operation, to be updated elision of use of scratch
    // space occurs.
    unsigned imm = 0;

    bool uses_scratch = needsArgInScratch(eval_second);
    const bool elide_scratch_use = canElideScratchUsage<ORD>(e);

    if (uses_scratch && elide_scratch_use) {
      // Adjust src and immediate for scratch space use elision
      elideScratchSpaceUse<ORD>(e, imm, src);
      // Don't need to use scratch space
      uses_scratch = false;
    }

    // Calculate space on operand stack drained by this allocation
    std::size_t consumed_operands = uopOperandCount(src);

    // Update free space on operand stack after this instruction
    consumeOperands(consumed_operands);

    // Create operation for expression
    auto uop = createMathUOP(mop, src, parent_request);
    program_.push_back(UInst{uop, imm});

    if (uses_scratch) {
      // Operation consumed top of scratch space
      retainScratch();
    }

    if (parent_request == mop_dst_kind::t) {
      // Allocate one slot in scratch space for result
      reserveScratch();
    }
  }

 private:
  // Instructions of optimized program
  std::vector<UInst> program_;
  // All operations to optimize
  const std::vector<Operation>& ops_;
  // Destinations of symbolic values
  const std::vector<BindingGapType>& gaps_;
  // Owner DAG of an expression
  const ExprDAG* dag_ = nullptr;
  // Highest number of values in scratch space
  int scratch_space_highwater_ = 0;
  // Current top of scratch space
  int scratch_space_top_ = 0;
  // Extra allocation required for scratch space
  int scratch_extra_allocation_ = 0;
  // Current number of values removed from operand stack
  int operands_stack_free_ = 0;
  // Expression reuses
  ReusedExprMapping reuses_;
};

template <OperandStackOrder ORD>
void ExprOptimizer::optimizeSubExpr(
    size_t op_index,
    mop_dst_kind parent_request) {
  TORCH_INTERNAL_ASSERT(op_index < ops_.size());

  // Current operation
  const Operation& op = ops_[op_index];
  // Reference to the root of an expression
  SExprRef ref = op.ref;
  // Evaluate expression
  SExprNode root_expr = dag_->fetch(ref);

  // Do we need to add recall/remember instruction pair?
  const bool is_reused_subexpr = dag_->isReusedSubExpr(ref);
  std::size_t rskip_n_idx = 0;

  // Remember instruction
  c10::optional<UInst> rem;

  if (is_reused_subexpr) {
    rem = beginRecallRemInstrSeq(op, parent_request);
    TORCH_INTERNAL_ASSERT(program_.back().op == UOP::rskip_n);
    // Remember index of rskip_n, so we can adjust it later
    rskip_n_idx = program_.size() - 1;
  }

  // Which sides of an expression need evaluation
  bool eval_rhs = isSExpr(root_expr.rhs_);
  bool eval_lhs = isSExpr(root_expr.lhs_);

  // Calculate indices of left and right operation:
  const std::size_t lhs_op = op_index + 1;
  const std::size_t rhs_op =
      eval_lhs ? lhs_op + 1 + ops_[lhs_op].nsubops : lhs_op;

  // Values depended on operand stack order (LR, RL)
  // Those determine order of recursion for instruction generation.
  //
  // Indices of operations (for LR)
  std::size_t first_eval_op = lhs_op;
  std::size_t second_eval_op = rhs_op;
  // Children subexpressions needing evaluation (for LR)
  bool eval_first = eval_lhs;
  bool eval_second = eval_rhs;
  // Destination of instruction required (for LR)
  mop_dst_kind first_child_request = mop_dst_kind::a;
  mop_dst_kind second_child_request = mop_dst_kind::b;

  if (ORD == OperandStackOrder::RL) {
    // Adjust values for right hand side evaluation
    std::swap(first_eval_op, second_eval_op);
    std::swap(eval_first, eval_second);
    std::swap(first_child_request, second_child_request);
  }

  // First evaluated child subexpression or value on top of operand stack
  // will be used later. Since
  const bool uses_scratch = needsArgInScratch(eval_second);

  // Check if we can avoid using scratch space.
  // We can do so if it is known scalar_value that fits instruction immediate
  const bool elide_scratch_use = canElideScratchUsage<ORD>(root_expr);

  if (uses_scratch && !elide_scratch_use) {
    // We need to store second hand side to temporary memory, since it
    // will be needed after we evaluate first side of an expression.
    if (eval_first) {
      // First evaluated child subexpression is not operand
      // We need to change destination to scratch space.
      first_child_request = mop_dst_kind::t;
    } else {
      // First value is on top of operand stack
      program_.push_back(UInst{UOP::store_o2t, 0});
      consumeOperands(1);
    }
    // Reserve place in scratch space
    reserveScratch();
  }

  // Generate code for first child of an expression
  if (eval_first) {
    // For LR order traverse lhs with parent need
    // mop_dst_kind::a/mop_dst_kind::t For RL order traverse rhs with parent
    // need mop_dst_kind::b/mop_dst_kind::t
    optimizeSubExpr<ORD>(first_eval_op, first_child_request);
  } else if (uses_scratch && elide_scratch_use) {
    // Operand will be provided as immediate of an instruction
    program_.push_back(UInst{UOP::drop_n, 1});
  }

  // Generate code for second child of expression
  if (eval_second) {
    // For LR order traverse rhs with need mop_dst_kind::b
    // For RL order traverse lhs with parent need mop_dst_kind::a
    optimizeSubExpr<ORD>(second_eval_op, second_child_request);
  }

  // Insert instriction for calculation of root of this subexpression
  insertMathInstr<ORD>(root_expr, parent_request, eval_first, eval_second);

  if (is_reused_subexpr) {
    finishRecallRemInstrSeq(rem, rskip_n_idx);
  }
}

template <OperandStackOrder ORD>
UProgram ExprOptimizer::optimizeExpr() {
  // Clear all occupation counters
  scratch_space_highwater_ = 0;
  scratch_space_top_ = 0;
  scratch_extra_allocation_ = 0;
  operands_stack_free_ = 0;

  // Generate instructions to update SymbolicValues referenced by this
  // expression (if any) on operands stack
  for (auto gap : gaps_) {
    TORCH_INTERNAL_ASSERT(
        gap.first <= UInst::MaxArg, "Index on stack doesn't fit immediate");
    program_.push_back(UInst{UOP::fill_sym, gap.first});
    consumeOperands(1);
  }

  TORCH_INTERNAL_ASSERT(!ops_.empty(), "Requires atleast one operation");
  const auto& root_operand = ops_[0].ref;
  // Operand stack space required
  if (isValue(root_operand)) {
    // We get single value, thus it is sufficient to move it from OperandsStack
    // to scratch space
    consumeOperands(1);
    // Pop from operand
    program_.push_back(UInst{UOP::store_o2t});
    reserveScratch();
  } else {
    optimizeSubExpr<ORD>(0, mop_dst_kind::t);
  }

  // UOP:::end instruction returns result and terminates program.
  program_.push_back(UInst{UOP::end, 0});
  // End operation consumes value form scratch space
  retainScratch();

  const std::size_t num_operands = ops_[0].noperands;
  const std::size_t num_symbolic = gaps_.size();

  // Binding cookies follow symbolic values thus InitialStack will have size of
  // those
  const std::size_t operands_stack_size = num_operands + num_symbolic;

  // Calculate required allocation size for stack and scratch space
  const std::size_t capacity_required =
      operands_stack_size + scratch_extra_allocation_;

  // Verify that program consumes all operands
  TORCH_INTERNAL_ASSERT(
      operands_stack_size >= size_t(operands_stack_free_),
      "Should not underflow the operand stack");

  // Verify that scratch space highwater is smaller or equal than total required
  // capacity
  TORCH_INTERNAL_ASSERT(size_t(scratch_space_highwater_) <= capacity_required);

  UProgram result;
  result.required_alloc = capacity_required;
  result.instructions = program_;
  // We treat number of reused subexpressions as version number optimized.
  // The reason is:
  // If any expression was added to ExprDAG and its subexpression was reused
  // (and previously was not) the order of slot indices in EvalState have
  // changed. If this expression refers to any other subexpression In all other
  // cases the optimized program for an expression is still valid.
  result.version = dag_->reuseCount();

  return result;
}

UProgram ExprOptimizer::optimize(
    OperandStackOrder order,
    Expr e,
    const std::vector<Operation>& ops,
    const std::vector<BindingGapType>& bindings) {
  ExprOptimizer self(e, ops, bindings);
  if (order == OperandStackOrder::LR) {
    return self.optimizeExpr<OperandStackOrder::LR>();
  } else {
    return self.optimizeExpr<OperandStackOrder::RL>();
  }
}

} // namespace

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace glfdc {

UProgram optimizeExpr(
    Expr e,
    const SymbolicExpr::Template& templ,
    OperandStackOrder ord) {
  return ExprOptimizer::optimize(ord, e, templ.operations, templ.binding_gaps);
}

c10::optional<scalar_type> evaluateOptimizedExpr(
    EvalState& es,
    const UProgram& program,
    const InitialStack<scalar_type>& init,
    SymbolicExpr::binding_fn_t fn) noexcept {
  // Allocate memory required memory on the stack
  void* memory = alloca(program.required_alloc * sizeof(scalar_type));

  MemoryBlock block(program.required_alloc, memory);
  return evaluateLoop(program.instructions, init, block, fn, es);
}

} // namespace glfdc
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch