#pragma once

#include <torch/csrc/jit/codegen/cuda/glfdc/cfold.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/eval.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/eval_stack.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/sexpr_cmp.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace glfdc {

using OperandsStack = ExternalStack<scalar_type, StackDir::GrowsUp>;
using ScratchSpace = ExternalStack<scalar_type, StackDir::GrowsDown>;

// Opcodes of an instruction
enum class UOP : std::uint8_t {

  // Mathematical Instruction variations.
  // sources: o o (stack order RL)
  // destination: t
  add_roo2t,
  sub_roo2t,
  mul_roo2t,
  div_roo2t,
  mod_roo2t,
  ceildiv_roo2t,

  // destination: a
  add_roo2a,
  sub_roo2a,
  mul_roo2a,
  div_roo2a,
  mod_roo2a,
  ceildiv_roo2a,

  // destination: b
  add_roo2b,
  sub_roo2b,
  mul_roo2b,
  div_roo2b,
  mod_roo2b,
  ceildiv_roo2b,

  // sources: o o (stack order LR)
  // destination: t
  add_loo2t,
  sub_loo2t,
  mul_loo2t,
  div_loo2t,
  mod_loo2t,
  ceildiv_loo2t,

  // destination: a
  add_loo2a,
  sub_loo2a,
  mul_loo2a,
  div_loo2a,
  mod_loo2a,
  ceildiv_loo2a,

  // destination: b
  add_loo2b,
  sub_loo2b,
  mul_loo2b,
  div_loo2b,
  mod_loo2b,
  ceildiv_loo2b,

  // sources: a t
  // destination: t
  add_at2t,
  sub_at2t,
  mul_at2t,
  div_at2t,
  mod_at2t,
  ceildiv_at2t,

  // destination: a
  add_at2a,
  sub_at2a,
  mul_at2a,
  div_at2a,
  mod_at2a,
  ceildiv_at2a,

  // destination: b
  add_at2b,
  sub_at2b,
  mul_at2b,
  div_at2b,
  mod_at2b,
  ceildiv_at2b,

  // sources: o b
  // destination: t
  add_ob2t,
  sub_ob2t,
  mul_ob2t,
  div_ob2t,
  mod_ob2t,
  ceildiv_ob2t,

  // destination: a
  add_ob2a,
  sub_ob2a,
  mul_ob2a,
  div_ob2a,
  mod_ob2a,
  ceildiv_ob2a,

  // destination: b
  add_ob2b,
  sub_ob2b,
  mul_ob2b,
  div_ob2b,
  mod_ob2b,
  ceildiv_ob2b,

  // sources: t b
  // destination: t
  add_tb2t,
  sub_tb2t,
  mul_tb2t,
  div_tb2t,
  mod_tb2t,
  ceildiv_tb2t,

  // destination: a
  add_tb2a,
  sub_tb2a,
  mul_tb2a,
  div_tb2a,
  mod_tb2a,
  ceildiv_tb2a,

  // destination: b
  add_tb2b,
  sub_tb2b,
  mul_tb2b,
  div_tb2b,
  mod_tb2b,
  ceildiv_tb2b,

  // sources: ao
  // destination: t
  add_ao2t,
  sub_ao2t,
  mul_ao2t,
  div_ao2t,
  mod_ao2t,
  ceildiv_ao2t,

  // destination: a
  add_ao2a,
  sub_ao2a,
  mul_ao2a,
  div_ao2a,
  mod_ao2a,
  ceildiv_ao2a,

  // destination: b
  add_ao2b,
  sub_ao2b,
  mul_ao2b,
  div_ao2b,
  mod_ao2b,
  ceildiv_ao2b,

  // sources: ai
  // destination: t
  add_ai2t,
  sub_ai2t,
  mul_ai2t,
  div_ai2t,
  mod_ai2t,
  ceildiv_ai2t,

  // destination: a
  add_ai2a,
  sub_ai2a,
  mul_ai2a,
  div_ai2a,
  mod_ai2a,
  ceildiv_ai2a,

  // destination: b
  add_ai2b,
  sub_ai2b,
  mul_ai2b,
  div_ai2b,
  mod_ai2b,
  ceildiv_ai2b,

  // sources: ib
  // destination: t
  add_ib2t,
  sub_ib2t,
  mul_ib2t,
  div_ib2t,
  mod_ib2t,
  ceildiv_ib2t,

  // destination: a
  add_ib2a,
  sub_ib2a,
  mul_ib2a,
  div_ib2a,
  mod_ib2a,
  ceildiv_ib2a,

  // destination: b
  add_ib2b,
  sub_ib2b,
  mul_ib2b,
  div_ib2b,
  mod_ib2b,
  ceildiv_ib2b,

  // Recall memoized into source
  recall_a,
  recall_b,
  recall_t,

  // Remember value from source
  rem_a,
  rem_b,
  rem_t,

  // Pickup operand and place as temporary
  store_o2t,

  // Skip n operations if last recall succeed
  rskip_n,
  // Drop n operations if last recall succeed
  rdrop_n,
  // Fill in symbolic value on operands stack
  fill_sym,
  // Unconditionally drop n operands
  drop_n,
  // End of code
  end,
};

constexpr unsigned MATH_OPS_COUNT = 6;
constexpr unsigned DST_KINDS_COUNT = 3;
constexpr unsigned OP_SOURCE_MULT = DST_KINDS_COUNT * MATH_OPS_COUNT;

// MathOp - Mathematical operation execution helper class templates

// In spirit each operation consists of combination of 3 phases:
// - get operands repesented
// - calculate
// - retire result for next operation

// There are 8 possible cases from where we can get operands.
// Those are represented by mop_src_kind enum
// mop_src_kind:
enum class mop_src_kind {
  // both operands come from operand stack (stack order RL)
  roo = 0 * OP_SOURCE_MULT,
  // both operands come from operand stack (stack order LR)
  loo = 1 * OP_SOURCE_MULT,
  // first operand is in a register second on temperaries stack
  at = 2 * OP_SOURCE_MULT,
  // first operand is in operand stack second is in b register
  ob = 3 * OP_SOURCE_MULT,
  // first operand comes from temporaries stack, second from b register
  tb = 4 * OP_SOURCE_MULT,
  // first operand comes from a register second comes from operand stack
  ao = 5 * OP_SOURCE_MULT,
  // first operand comes from a register second comes from instruction immediate
  ai = 6 * OP_SOURCE_MULT,
  // first operand comes from instruction immediate and second comes from b
  // register
  ib = 7 * OP_SOURCE_MULT,
};

// Number of source kinds (keep in sync with src_kinds)
constexpr std::size_t SRC_KINDS_COUNT = 8;

// Those cases are possible transitions to minimize memory access when
// calculating operands Without those we would perform uneccesery stack.push()
// stack.pop() pairs

template <mop_src_kind SK>
class MathOpFetchPhase {
  // Specialized variants below:
  //
  // static void fetch(
  //    OperandsStack& operands,
  //    ScratchSpace& tmps,
  //    unsigned imm,
  //    scalar_type& a,
  //    scalar_type& b
  //    );
};

// MathOpFetchPhase<mop_src_kind::roo> - Specialization for case "OO" peformed
// for expression where both operands are on operand stack Example is evalution
// of expression tree like (1 + 3). Version for right side first evaluations
template <>
class MathOpFetchPhase<mop_src_kind::roo> {
 protected:
  static void fetch(
      OperandsStack& operands,
      ScratchSpace& tmps,
      unsigned imm,
      scalar_type& a,
      scalar_type& b) {
    b = operands.pop_top();
    a = operands.pop_top();
  }
};

// As above but for left side first evaluation.
template <>
class MathOpFetchPhase<mop_src_kind::loo> {
 protected:
  static void fetch(
      OperandsStack& operands,
      ScratchSpace& tmps,
      unsigned imm,
      scalar_type& a,
      scalar_type& b) {
    a = operands.pop_top();
    b = operands.pop_top();
  }
};

// MatOpFetchPhase<mop_src_kind::at> - Specialization for case "AT"  performed
// where expression left operand is an expression and right operand was result
// of an expression Example (e1 + e2)
template <>
class MathOpFetchPhase<mop_src_kind::at> {
 protected:
  static void fetch(
      OperandsStack& operands,
      ScratchSpace& tmps,
      unsigned imm,
      scalar_type& a,
      scalar_type& b) noexcept {
    // A directly calculated result of an expression
    // a = a;
    b = tmps.pop_top();
  }
};

// MatOpFetchPhase<mop_src_kind::ob> - Specialization for case "OT" performed
// for expressions where left operand is on operand stack and right operand is
// on temporary stack Example would (1 + e)
template <>
class MathOpFetchPhase<mop_src_kind::ob> {
 protected:
  static void fetch(
      OperandsStack& operands,
      ScratchSpace& tmps,
      unsigned imm,
      scalar_type& a,
      scalar_type& b) noexcept {
    a = operands.pop_top();
    // Result already in b after calculating result of right subexpression
    // b = b
  }
};

// MatOpFetchPhase<mop_src_kind::tb> - Specialization for case "TB"  performed
// where expression left operand is a value and right operand was result of an
// expression Example (1 + e)
template <>
class MathOpFetchPhase<mop_src_kind::tb> {
 protected:
  static void fetch(
      OperandsStack& operands,
      ScratchSpace& tmps,
      unsigned imm,
      scalar_type& a,
      scalar_type& b) noexcept {
    a = tmps.pop_top();
    // Result already in b after calculating result of right subexpression
    // b = b
  }
};

// MatOpFetchPhase<mop_src_kind::ao> - Specialization for case "AO"  performed
// where expression left operand  was result of an expression and right operand
// is a value on top of operand stack Example (e + 1)
template <>
class MathOpFetchPhase<mop_src_kind::ao> {
 protected:
  static void fetch(
      OperandsStack& operands,
      ScratchSpace& tmps,
      unsigned imm,
      scalar_type& a,
      scalar_type& b) noexcept {
    // Result already in a after calculating result of right subexpression
    // a = a
    b = operands.pop_top();
  }
};

// MatOpFetchPhase<src_kind::ai> - Specialization for case "AI"  performed where
// expression left operand  was result of an  expression and right operand is a
// value that is not symbolic. Example (e + 1)
// NOTE this is special case when we would have to use AT case to temporarily
// store second operand but can encode it as immediate of an instruction.
template <>
class MathOpFetchPhase<mop_src_kind::ai> {
 protected:
  static void fetch(
      OperandsStack& operands,
      ScratchSpace& tmps,
      unsigned imm,
      scalar_type& a,
      scalar_type& b) noexcept {
    // Result already in a after calculating result of right subexpression
    // a = a
    b = imm;
  }
};

// MatOpFetchPhase<src_kind::ib> - Specialization for case "IB"  performed where
// expression right operand  was result of an  expression and right operand is a
// value that is not symbolic. Example (1 + e)
// NOTE this is special case when we would have to use TB case to temporarily
// store second operand, but we can encode it as immediate of an instruction
// instead.
template <>
class MathOpFetchPhase<mop_src_kind::ib> {
 protected:
  static void fetch(
      OperandsStack& operands,
      ScratchSpace& tmps,
      unsigned imm,
      scalar_type& a,
      scalar_type& b) noexcept {
    a = imm;
    // Result already in a after calculating result of right subexpression
    // b = b;
  }
};

// Kind of operation to perform in calculation phase
enum class mop_kind {
  // Addition
  add = 0,
  // Subtraction
  sub = 1,
  // Multiplication
  mul = 2,
  // Division
  div = 3,
  // Modulo
  mod = 4,
  // division with ceiling
  ceildiv = 5,
};

// MathOpCalcPhase - calculates result of an expression
template <mop_kind MO>
class MathOpCalcPhase {
  // Specialized:
  //
  // static void calculate(scalar_type& r, scalar_type a, scalar_type b)
  // noexcept;
  //
};

// Addition
template <>
class MathOpCalcPhase<mop_kind::add> {
 protected:
  static void calculate(scalar_type& r, scalar_type a, scalar_type b) noexcept {
    r = cfold(OperatorKind::add, a, b);
  }
};

// Subtraction
template <>
class MathOpCalcPhase<mop_kind::sub> {
 protected:
  static void calculate(scalar_type& r, scalar_type a, scalar_type b) noexcept {
    r = cfold(OperatorKind::sub, a, b);
  }
};

// Multiplication
template <>
class MathOpCalcPhase<mop_kind::mul> {
 protected:
  static void calculate(scalar_type& r, scalar_type a, scalar_type b) noexcept {
    r = cfold(OperatorKind::mul, a, b);
  }
};

// Division
template <>
class MathOpCalcPhase<mop_kind::div> {
 protected:
  static void calculate(scalar_type& r, scalar_type a, scalar_type b) noexcept {
    r = cfold(OperatorKind::div, a, b);
  }
};

// Modulo
template <>
class MathOpCalcPhase<mop_kind::mod> {
 protected:
  static void calculate(scalar_type& r, scalar_type a, scalar_type b) noexcept {
    r = cfold(OperatorKind::mod, a, b);
  }
};

// Division with celing
template <>
class MathOpCalcPhase<mop_kind::ceildiv> {
 protected:
  static void calculate(scalar_type& r, scalar_type a, scalar_type b) noexcept {
    r = cfold(OperatorKind::ceildiv, a, b);
  }
};

// Instruction destinations of calculated result
enum class mop_dst_kind {
  // Scratch space
  t = 0 * MATH_OPS_COUNT,
  // Register A
  a = 1 * MATH_OPS_COUNT,
  // Register B
  b = 2 * MATH_OPS_COUNT,
};

// MathOpRetire<> - retire result of an expression
template <mop_dst_kind DK>
class MathOpRetirePhase {
  // Specialized for mop_dst_kind's below
  // static void retire(scalar_type r, scalar_type &a, scalar_type &b,
  // scalar_type &t, ScratchSpace &tmps);
};

// Speciazation placing result of an expression in temporary stack
template <>
class MathOpRetirePhase<mop_dst_kind::t> {
 protected:
  static void retire(
      scalar_type r,
      scalar_type& a,
      scalar_type& b,
      scalar_type& t,
      ScratchSpace& tmps) noexcept {
    t = r;
    tmps.push(t);
  }
};

template <>
class MathOpRetirePhase<mop_dst_kind::a> {
 protected:
  static void retire(
      scalar_type r,
      scalar_type& a,
      scalar_type& b,
      scalar_type& t,
      ScratchSpace& tmps) noexcept {
    a = r;
  }
};

template <>
class MathOpRetirePhase<mop_dst_kind::b> {
 protected:
  static void retire(
      scalar_type r,
      scalar_type& a,
      scalar_type& b,
      scalar_type& t,
      ScratchSpace& tmps) noexcept {
    b = r;
  }
};

template <mop_kind MO, mop_src_kind SRC, mop_dst_kind DST>
class MathOp : private MathOpFetchPhase<SRC>,
               private MathOpCalcPhase<MO>,
               private MathOpRetirePhase<DST> {
 public:
  static void execute(
      scalar_type& a,
      scalar_type& b,
      scalar_type& t,
      unsigned imm,
      OperandsStack& operands,
      ScratchSpace& tmps) noexcept {
    // Prepare a and b depending on source
    MathOpFetchPhase<SRC>::fetch(operands, tmps, imm, a, b);
    scalar_type r;
    // Assign value of an (a "op" b) expression to r
    MathOpCalcPhase<MO>::calculate(r, a, b);
    // assign r to destination
    MathOpRetirePhase<DST>::retire(r, a, b, t, tmps);
  }
};

constexpr unsigned ALL_MATH_UOPS_COUNT = OP_SOURCE_MULT * SRC_KINDS_COUNT;

namespace detail {

// Returns mop_src_kind of MathOp opcode
template <UOP Opcode_>
inline constexpr mop_src_kind mathUOPSrc() noexcept {
  static_assert(
      size_t(Opcode_) < ALL_MATH_UOPS_COUNT, "UOP is not MathOP Opcode");
  // MathOp opcodes are arranged by blocks of OP_SOURCE_MULT values
  // Integer division returns number of such block, that multiplied by
  // OP_SOURCE_MULT returns value of mop_src_kind enum. See mop_src_kind
  // definition.
  return mop_src_kind((size_t(Opcode_) / OP_SOURCE_MULT) * OP_SOURCE_MULT);
}

inline constexpr mop_src_kind mathUOPSrc(UOP opcode) noexcept {
  TORCH_INTERNAL_ASSERT(
      size_t(opcode) < ALL_MATH_UOPS_COUNT, "UOP is not MathOP Opcode");
  // MathOp opcodes are arranged by blocks of OP_SOURCE_MULT values
  // Integer division returns number of such block, that multiplied by
  // OP_SOURCE_MULT returns value of mop_src_kind enum. See mop_src_kind
  // definition.
  return mop_src_kind((size_t(opcode) / OP_SOURCE_MULT) * OP_SOURCE_MULT);
}

// Returns mop_dst_kind of MathOp opcode
template <UOP Opcode_>
inline constexpr mop_dst_kind mathUOPDest() noexcept {
  static_assert(
      size_t(Opcode_) < ALL_MATH_UOPS_COUNT, "UOP is not MathOP Opcode");
  // Blocks of MATH_OPS_COUNT instruction, number of such block, multiplied by
  // MATH_OPS_COUNT
  return mop_dst_kind(
      ((size_t(Opcode_) / MATH_OPS_COUNT) % DST_KINDS_COUNT) * MATH_OPS_COUNT);
}

template <UOP Opcode_>
inline constexpr mop_kind mathUOPKind() noexcept {
  static_assert(
      size_t(Opcode_) < ALL_MATH_UOPS_COUNT, "UOP is not MathOP Opcode");
  // Blocks of MATH_OPS_COUNT instructions
  return mop_kind(size_t(Opcode_) % MATH_OPS_COUNT);
}

// Determines type of corresponding MathOp based on code opcode
template <UOP Opcode_>
struct MathOpTypeHelper {
  // static_assert(size_t(Opcode_)< ALL_MATH_UOPS_COUNT, "UOP is not MathOP
  // Opcode");

  static constexpr mop_kind Kind = mathUOPKind<Opcode_>();
  static constexpr mop_src_kind Src = mathUOPSrc<Opcode_>();
  static constexpr mop_dst_kind Dst = mathUOPDest<Opcode_>();

  using type = MathOp<Kind, Src, Dst>;
};

template <UOP Opcode_>
using math_op_type = typename MathOpTypeHelper<Opcode_>::type;

} // namespace detail

// Recall operation retire phase
template <mop_dst_kind DST>
class RecallOpRetire {
 public:
  static void retire(
      scalar_type r,
      scalar_type& a,
      scalar_type& b,
      ScratchSpace& tmps) {
    switch (DST) {
      case mop_dst_kind::t:
        tmps.push(r);
        return;
      case mop_dst_kind::a:
        a = r;
        return;
      case mop_dst_kind::b:
        b = r;
        return;
    }
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false, "Unhandled recall destination");
  }
};

// RecallOp - implements opcodes UOP::recall_{a, b, t}
//
// Tries to load value from EvalState
// Sets recalled_flag depending on if operation succeeded or not
// Value is set to destination register or stored in scratch space
template <mop_dst_kind DST>
class RecallOp : private RecallOpRetire<DST> {
 public:
  static void execute(
      EvalState& es,
      bool& recalled_flag,
      scalar_type& a,
      scalar_type& b,
      unsigned arg,
      ScratchSpace& tmps) {
    // Obtain slot index from immediate
    auto index = abs_offset_t(arg);
    auto slot = es.recall(index);
    // Value already calculated?
    recalled_flag = slot.has_value();
    if (recalled_flag) {
      scalar_type r = *slot;
      RecallOpRetire<DST>::retire(r, a, b, tmps);
    }
  }
};

// RemOpFetch - Fetch phase of UOP::rem_{a,b,t} (remember operation)
//
// Picks the source of the value depending on where it was stored by previous
// operation
template <mop_dst_kind SRC>
class RemOpFetch {
 public:
  static scalar_type fetch(
      scalar_type a,
      scalar_type b,
      const ScratchSpace& tmps) {
    switch (SRC) {
      case mop_dst_kind::t:
        return tmps.top();
      case mop_dst_kind::a:
        return a;
      case mop_dst_kind::b:
        return b;
    }
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        false, "Unhandled calculation destination for rem");
  }
};

// RemOp implements opcodes UOP::rem_{a,b, t} semantics
//
// Stores calculated value in EvalState for reuse next time
template <mop_dst_kind DST>
class RemOp : RemOpFetch<DST> {
 public:
  static void execute(
      EvalState& es,
      scalar_type a,
      scalar_type b,
      unsigned arg,
      ScratchSpace& tmps) {
    // Note that tmps.top() is not popped
    scalar_type r = RemOpFetch<DST>::fetch(a, b, tmps);
    auto slot = abs_offset_t(arg);
    es.remember(slot, r);
  }
};
// StoreToScrachOp - implements opcode UOP::store_o2t semantics
//
// Moves value from operand stack to tmps.
class StoreToScrachOp {
 public:
  static void execute(OperandsStack& operands, ScratchSpace& tmps) {
    tmps.push(operands.pop_top());
  }
};

// RSkipNOp - implements opcode UOP::rskip_n semantics
//
// If recall succeeds advances execution
class RSkipNOp {
 public:
  static void execute(unsigned& ip, unsigned arg, bool recalled_flag) noexcept {
    if (recalled_flag) {
      // Advance instruction pointer n operations
      ip += arg;
    }
  }
};

// RDropNOp - implements opcode UOP::rdrop_n semantics
//
// If last recall instruction succeeds drop operands from operands stack
class RDropNOp {
 public:
  static void execute(
      OperandsStack& operands,
      unsigned arg,
      bool recalled_flag) noexcept {
    if (recalled_flag) {
      // Drop n operands from stack
      operands.drop(arg);
    }
  }
};

// FillSymOp - implements opcode UOP::fill_sym semantics
//
// Updates operand stack with symbolic value
class FillSymOp {
 private:
  using opt_scalar_t = c10::optional<scalar_type>;
  using provider_t = const BoundValueProvider&;

 public:
  static bool execute(
      OperandsStack& operands,
      unsigned arg,
      provider_t provider) noexcept {
    uintptr_t cookie = detail::bitcast<uintptr_t>(operands.pop_top());
    auto opt_value = provider.get(cookie);
    if (opt_value) {
      operands.fill_at(arg, *opt_value);
      return true;
    }
    return false;
  }
};

// DropNOp - implements opcode UOP::drop_n semantics
//
// Unconditionally drop N operands from operands stack
class DropNOp {
 public:
  static void execute(
      OperandsStack& operands,
      unsigned arg,
      bool recalled_flag) noexcept {
    // Drop n operands from operands stack
    operands.drop(arg);
  }
};

// EndOp
class EndOp {
 private:
  using opt_scalar_t = c10::optional<scalar_type>;

 public:
  // Returns result of the program
  static opt_scalar_t execute(ScratchSpace& tmps) {
    return tmps.top();
  }
};

} // namespace glfdc
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch