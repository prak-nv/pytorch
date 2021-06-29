#pragma once

// Symbolic expression evaluator

#include <torch/csrc/jit/codegen/cuda/glfdc/bitvector.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/expr.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/initial_stack.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/sexpr.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/skiptree.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/sparse_map.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/value_binding.h>

#include <algorithm>
#include <functional>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace glfdc {

template <typename Ty_>
class EvalStack;

// Operation for evaluator stack machine
//
// Used for SymbolicExpression::Template list for operations
// to evaluate given expression or prepare optimized (UProgram) form afterwards
struct Operation {
  // Reference to subexpression
  const SExprRef ref;

  // Number of following operations to skip if lazy evaluated
  // (makes possible to recover tree structure)
  // Please refer to skiptree.h for description
  unsigned nsubops;
  // total number of operands on stack to calculate subexpression value
  unsigned noperands;

  static constexpr unsigned INVALID_POP = unsigned(-1);
};

// ReusedExprMapping - Holds mapping of reused subexpressions
//
// Subexpression of an expression appear in 2 or more different expressions is
// called reused
//
// Ie. for set of expression eg. "(x+1)*3" and "(x+1)*y"
// Subexpression "(x+1)" is reused subexpression
//
// All reused subexpression are numbered monotonicaly by order of creation.
// This number is used as index into EvalState slot list of memoized values.
class TORCH_CUDA_CU_API ReusedExprMapping {
  using opt_index_t = c10::optional<std::size_t>;

 public:
  ReusedExprMapping() noexcept = default;

  // Given subexpressions returns its slot number or nullopt
  opt_index_t slot(SExprRef ref) const noexcept {
    return slot_mapping_.find(ref.index());
  }

  // Returns number of reused expressions
  std::size_t size() const noexcept {
    return slot_mapping_.size();
  }

  // Returns true if there is no other mapping
  bool empty() const noexcept {
    return slot_mapping_.empty();
  }

  // Creates empty mapping for eager evaluation of expression
  static ReusedExprMapping createForEagerEval() {
    return {sparse_map{}};
  }

  // Returns index of the slot for reused subexpression if it exists
  c10::optional<abs_offset_t> getSlotIndex(SExprRef ref) const noexcept {
    if (!slot_mapping_.has(ref.index())) {
      return c10::nullopt;
    }
    return abs_offset_t(*slot_mapping_.find(ref.index()));
  }

  // create_lazy_mapping - creates mapping for reused subexpression and its
  // value slot indices
  static ReusedExprMapping createForMemoizedEval(const ExprDAG& dag);

 private:
  ReusedExprMapping(sparse_map map) : slot_mapping_(map) {}

  // Mapping of memoized values to slot indices
  const sparse_map slot_mapping_;
};

// EvalState - holds memoized values for reused subexpressions
// Each subexpression that is reused obtains slot to store value
// that can be reused later on.
class TORCH_CUDA_CU_API EvalState {
 public:
  using ValueSlot = c10::optional<scalar_type>;
  using ValueSlotPtr = ValueSlot*;

  // Creates an EvalState with no slots for memoized values
  static EvalState createEmpty();

  explicit EvalState(ReusedExprMapping mapping);

  EvalState(const EvalState&) = default;

  // Returns pointer to slot for reused subexpression value or nullptr if
  // subexpression is not reused
  ValueSlotPtr load(SExprRef sexpr);

  // Returns memoized value given slot number
  ValueSlot recall(abs_offset_t slot);

  // Stores subexpression value
  void store(ValueSlot& slot, scalar_type val);

  // Stores memoized value given slot number
  void remember(abs_offset_t slot, scalar_type value);

  // Clears all memoized values from EvalState
  void forgetAllMemoized();

  // Returns number of slots for memoized value
  std::size_t slotCount() const noexcept;

 private:
  // List of slots for memoized values in order determined by ReuseExprMapping
  std::vector<ValueSlot> memo_;
  // Mapping of all reused subexpression references to index in memo_
  ReusedExprMapping mapping_;
};

// Creates EvalState for given ExprDAG, that holds memoized value for subsequent
// evaluation reuse.
TORCH_CUDA_CU_API std::unique_ptr<EvalState> createEvalStateForDAG(
    const ExprDAG& dag);

// OperandStackOrder - determines order of operand on stack, ie. evaluation
// order left/right side first
enum class OperandStackOrder {
  // Operands for left side on top of the operand stack
  LR,
  // Operands for right side on top of the operand stack
  RL,
};

// Declaration of an instruction opcode for symbolic expression (SymbolicExpr)
// in optimized form
//
// See eval_uinstr.h for details
enum class UOP : uint8_t;

// UInst - single instruction SymbolicExpression optimized form
//
// Instruction consists and optional immediate 24b argument
struct UInst {
  // Opcode of an instruction
  UOP op;
  // Immediate or argument used by some instructions
  unsigned arg : 24;
  static constexpr size_t MaxArg = (1 << 24) - 1;
};

// Program of optimized form of SymbolicExpr
struct UProgram {
  // List of instructions to perform to evaluate an expression
  std::vector<UInst> instructions;
  // Required stack allocation for operands and temporaries
  std::size_t required_alloc;
  // Version of a program to ensure it was translated for correct ExprDAG state
  std::size_t version = 0;

  // Returns if optimized form exists
  bool isValid() const noexcept {
    return !instructions.empty();
  }

  // Returns true if optimized form was created for same version of DAG
  bool isCurrentVersion(const Expr& e) const noexcept {
    return e.dag().reuseCount() == version;
  }
};

// SymbolicExpr - symbolic expression
//
// This is stack machine that is holding expression in form of expression
// reduction tree. All operations to calculate expression value are kept in
// linear tree form to allow lazy evaluation.
//
// On creation of expression evaluator initial stack of known (scalar values is
// prepared and memoized) During evaluation runtime (unknown) values are being
// queried and inserted to the stack copy Afterwards expression is calculated by
// DFS right first tree reduction using stack algorithm
//
// Symbolic expression can be later on translated to "optimized" form for faster
// evaluation. This is possible once all expressions belonging to ExprDag are
// already created. This translation requires knowledge of all subexpressions
// reuses, thus no additional subexpression can be added (result of such
// calculation would not be memoized).
class TORCH_CUDA_CU_API SymbolicExpr {
  // stores where to update on stack runtime value
  using initial_stack_t = InitialStack<scalar_type>;
  using stack_t = EvalStack<scalar_type>;

 public:
  // User provided function that returns concrete value for symbolic value
  // Argument to this function is same unique value for which the binding was
  // created. See ExpressionBuilder::getBinding();
  using binding_fn_t = const BoundValueProvider&;

  // Template of symbolic expression
  struct Template {
    // List of operations to perform to evaluate an expression stored in
    // linearized tree form
    std::vector<Operation> operations;
    // initial stack of known operands with gaps for runtime values
    initial_stack_t initial_stack;
    // List of gap indices on initial_stack for runtime values
    // Those are filled in on copy of the stack during evaluation
    std::vector<BindingGapType> binding_gaps;
  };

  explicit SymbolicExpr(const Expr& e);

  // Evaluates a symbolic expression.
  template <typename Ty_>
  opt_scalar_t evaluate(
      EvalState& es,
      const SymbolicValueProviderBase<Ty_>& symbols) const {
    return evaluateExpr(es, symbols.getProvider());
  }

  // Returns reference to owner DAG
  const ExprDAG& dag() const;

  // Returns reference to expression in ExprDAG
  const Expr& expr() const;

  // Translates expression to optimized form.
  //
  // NOTE: This should be done only after all expressions has been added to
  // ExprDAG
  void optimize();

  // Returns true if expression wasn't translated to optimized form
  bool isOptimized() const noexcept {
    return program_.isValid();
  }

 private:
  opt_scalar_t evaluateExpr(EvalState& e, binding_fn_t binding_fn) const;

  void evaluateSubExpr(size_t op_index, stack_t& stack, EvalState& es) const;

  // Evaluates expression in optimized form
  opt_scalar_t evaluateOptimized(EvalState& es, binding_fn_t binding_fn) const;
  // Evaluates expression from template (fallback)
  opt_scalar_t evaluateTemplate(EvalState& es, binding_fn_t binding_fn) const;

 private:
  // Reference to an expression in owner ExprDAG
  Expr expr_;
  // Template of an expression
  const Template templ_;

  // Optimized form of an expression, consist of one or more UInst instruction
  // if expression has been optimized.
  UProgram program_;
  // Order of operands on the initial_stack
  static constexpr OperandStackOrder stack_order_ = OperandStackOrder::RL;
};

} // namespace glfdc
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
