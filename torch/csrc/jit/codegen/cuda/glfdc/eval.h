#pragma once

// Expression evaluator

#include "bitvector.h"
#include "expr.h"
#include "sexpr.h"
#include "sparse_map.h"
#include "stack.h"

#include <algorithm>
#include <functional>

#include "c10/util/Optional.h"

namespace glfdc {

struct Expr;
struct ExprDAG;

struct EvalState;

// Operation for evaluator stack machine
struct Operation {
  const SExprRef ref;

  // TODO: bitfields/ushort? 12b/16b should be plenty and we could pack it more
  unsigned nsubops; // number of following operations to skip if lazy evaluated
                    // (makes possible to recover tree structure)
  unsigned noperands; // total number of operands on stack to calculate
                      // subexpression value

  static constexpr unsigned INVALID_POP = unsigned(-1);
};

using opt_index_t = c10::optional<std::size_t>;

// Holds mapping of reused subexpressions
struct ReusedExprMapping {
  // Mapping of memoized values to slot values
  const sparse_map slot_mapping;

  // Given subexpressions returns its slot number or nullopt
  opt_index_t slot(SExprRef ref) const noexcept {
    return slot_mapping.find(ref.index_);
  }

  std::size_t size() const noexcept {
    return slot_mapping.size();
  }

  bool empty() const noexcept {
    return slot_mapping.empty();
  }

  // Creates empty mapping for eager evaluation of expression
  static ReusedExprMapping create_eager_mapping() {
    return {sparse_map{}};
  }

  // create_lazy_mapping - Allocates mapping for reused subexpression and its
  // value slot index
  static ReusedExprMapping create_lazy_mapping(
      const bitvector_t& reused_sexprs) // O(n)
  {
    sparse_map mapping(reused_sexprs.size());

    std::size_t slot_counter = 0;

    for (std::size_t i = 0; i < reused_sexprs.size(); ++i) {
      // if subexpression is reused assign mapping
      if (reused_sexprs[i])
        mapping.insert(i, slot_counter++);
    }

    return {mapping};
  }
};

// EvalState - holds memoized values for reused subexpressions
struct EvalState {
  using LazyScalar = c10::optional<scalar_type>;
  using LazyScalarSlot = LazyScalar*;

  explicit EvalState(ReusedExprMapping mapping)
      : memo_(mapping.size()), mapping_(std::move(mapping)) {}

  EvalState(const EvalState&) = default;

  // Returns pointer to slot for reused subexpression value or nullptr if
  // subexpression is not reused
  LazyScalarSlot load(SExprRef sexpr) {
    auto opt_map = mapping_.slot(sexpr); // O(1)

    if (!opt_map.has_value())
      return nullptr;

    return memo_.data() + opt_map.value();
  }

  // Stores subexpression value
  void store(LazyScalar& slot, scalar_type val) {
    assert(!slot.has_value() && "Already evaluated");
    assert(
        uintptr_t(&slot) >= uintptr_t(memo_.data()) && "Invalid slot pointer");
    assert(
        uintptr_t(&slot) < uintptr_t(memo_.data() + memo_.size()) &&
        "Invalid slot pointer");

    slot = val;
  }

  void clear() {
    std::fill(memo_.begin(), memo_.end(), c10::nullopt);
  }

 private:
  std::vector<LazyScalar> memo_;
  ReusedExprMapping mapping_;
};

// ExprEvaluator - evaluates expressions
//
// This is stack machine keeping that is prepared for expression reduction tree.
// All operations to calculate expression value are kept in linear tree form to
// allow lazy evaluation
//
// On creation of expression evaluator initial stack of known (scalar values is
// prepared and memoized) During evaluation runtime (unknow) values are being
// queried and inserted to the stack copy Afterwards expression is calculated by
// DFS right first tree reduction using stack algorithm
struct ExprEvaluator {
  // stores where to update on stack runtime value
  // TODO: should we also handle stack updates lazily?
  using stack_t = EvalStack<scalar_type>;
  using opt_scalar_t = c10::optional<scalar_type>;
  using binding_fn_t = std::function<opt_scalar_t(uintptr_t)>;

  explicit ExprEvaluator(const Expr& e);

  opt_scalar_t evaluate(EvalState& e, binding_fn_t binding_fn) const;

  const ExprDAG& dag() const;
  const Expr& expr() const;

 private:
  static scalar_type scalar_operand_value(Operand op) noexcept;

  void calculate_subops_operands(); // O(n) - n is number of operations
  size_t calculate_operands_range(size_t start, size_t end);

  void prepare_eval();
  void prepare_eval_operand(Operand op);

  void evaluate_subexpr(size_t op_index, stack_t& eval_stack, EvalState& es)
      const;
  opt_scalar_t evaluate_value(Value val, binding_fn_t binding_fn) const;

 private:
  std::vector<Operation> operations_; // Inorder Depth First list of operations
  stack_t initial_stack_; // initial stack of known operands with gaps for
                          // runtime values
  std::vector<binding_gap_t>
      binding_gaps_; // list of gap indices on initial stack

  Expr expr_;
};

} // namespace glfdc
