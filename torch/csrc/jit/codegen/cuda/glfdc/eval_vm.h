#pragma once

#include <torch/csrc/jit/codegen/cuda/glfdc/eval_uinstr.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace glfdc {

using MemoryBlock = DEVector<scalar_type>;

template <typename BindingFn_>
c10::optional<scalar_type> evaluateLoop(
    const std::vector<UInst>& instructions,
    const InitialStack<scalar_type>& init,
    MemoryBlock& memory,
    BindingFn_& fn,
    EvalState& es) noexcept {
  // Instruction pointer
  unsigned ip = 0;

  // Registers
  scalar_type a, b, t;

  // Stack of operands
  OperandsStack operands(memory);
  // Copy values to operand stack
  init.copy_to(operands);

  ScratchSpace tmps(memory);
  bool recalled_flag = false;

  // Zero the registers
  a = 0;
  b = 0;
  t = 0;

  for (; ip < instructions.size(); ++ip) {
    auto instr = instructions[ip];
    unsigned imm = instr.arg;
    switch (instr.op) {
        // Mathematical Instruction variations.
        // sources: o o (stack order RL)
        // destination: t
      case UOP::add_roo2t:
        detail::math_op_type<UOP::add_roo2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_roo2t:
        detail::math_op_type<UOP::sub_roo2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_roo2t:
        detail::math_op_type<UOP::mul_roo2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_roo2t:
        detail::math_op_type<UOP::div_roo2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_roo2t:
        detail::math_op_type<UOP::mod_roo2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_roo2t:
        detail::math_op_type<UOP::ceildiv_roo2t>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // destination: a
      case UOP::add_roo2a:
        detail::math_op_type<UOP::add_roo2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_roo2a:
        detail::math_op_type<UOP::sub_roo2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_roo2a:
        detail::math_op_type<UOP::mul_roo2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_roo2a:
        detail::math_op_type<UOP::div_roo2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_roo2a:
        detail::math_op_type<UOP::mod_roo2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_roo2a:
        detail::math_op_type<UOP::ceildiv_roo2a>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // destination: b
      case UOP::add_roo2b:
        detail::math_op_type<UOP::add_roo2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_roo2b:
        detail::math_op_type<UOP::sub_roo2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_roo2b:
        detail::math_op_type<UOP::mul_roo2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_roo2b:
        detail::math_op_type<UOP::div_roo2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_roo2b:
        detail::math_op_type<UOP::mod_roo2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_roo2b:
        detail::math_op_type<UOP::ceildiv_roo2b>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // sources: o o (stack order LR)
        // destination: t
      case UOP::add_loo2t:
        detail::math_op_type<UOP::add_loo2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_loo2t:
        detail::math_op_type<UOP::sub_loo2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_loo2t:
        detail::math_op_type<UOP::mul_loo2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_loo2t:
        detail::math_op_type<UOP::div_loo2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_loo2t:
        detail::math_op_type<UOP::mod_loo2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_loo2t:
        detail::math_op_type<UOP::ceildiv_loo2t>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // destination: a
      case UOP::add_loo2a:
        detail::math_op_type<UOP::add_loo2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_loo2a:
        detail::math_op_type<UOP::sub_loo2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_loo2a:
        detail::math_op_type<UOP::mul_loo2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_loo2a:
        detail::math_op_type<UOP::div_loo2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_loo2a:
        detail::math_op_type<UOP::mod_loo2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_loo2a:
        detail::math_op_type<UOP::ceildiv_loo2a>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // destination: b
      case UOP::add_loo2b:
        detail::math_op_type<UOP::add_loo2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_loo2b:
        detail::math_op_type<UOP::sub_loo2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_loo2b:
        detail::math_op_type<UOP::mul_loo2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_loo2b:
        detail::math_op_type<UOP::div_loo2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_loo2b:
        detail::math_op_type<UOP::mod_loo2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_loo2b:
        detail::math_op_type<UOP::ceildiv_loo2b>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // sources: a t
        // destination: t
      case UOP::add_at2t:
        detail::math_op_type<UOP::add_at2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_at2t:
        detail::math_op_type<UOP::sub_at2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_at2t:
        detail::math_op_type<UOP::mul_at2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_at2t:
        detail::math_op_type<UOP::div_at2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_at2t:
        detail::math_op_type<UOP::mod_at2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_at2t:
        detail::math_op_type<UOP::ceildiv_at2t>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // destination: a
      case UOP::add_at2a:
        detail::math_op_type<UOP::add_at2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_at2a:
        detail::math_op_type<UOP::sub_at2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_at2a:
        detail::math_op_type<UOP::mul_at2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_at2a:
        detail::math_op_type<UOP::div_at2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_at2a:
        detail::math_op_type<UOP::mod_at2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_at2a:
        detail::math_op_type<UOP::ceildiv_at2a>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // destination: b
      case UOP::add_at2b:
        detail::math_op_type<UOP::add_at2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_at2b:
        detail::math_op_type<UOP::sub_at2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_at2b:
        detail::math_op_type<UOP::mul_at2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_at2b:
        detail::math_op_type<UOP::div_at2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_at2b:
        detail::math_op_type<UOP::mod_at2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_at2b:
        detail::math_op_type<UOP::ceildiv_at2b>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // sources: o b
        // destination: t
      case UOP::add_ob2t:
        detail::math_op_type<UOP::add_ob2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_ob2t:
        detail::math_op_type<UOP::sub_ob2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_ob2t:
        detail::math_op_type<UOP::mul_ob2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_ob2t:
        detail::math_op_type<UOP::div_ob2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_ob2t:
        detail::math_op_type<UOP::mod_ob2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_ob2t:
        detail::math_op_type<UOP::ceildiv_ob2t>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // destination: a
      case UOP::add_ob2a:
        detail::math_op_type<UOP::add_ob2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_ob2a:
        detail::math_op_type<UOP::sub_ob2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_ob2a:
        detail::math_op_type<UOP::mul_ob2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_ob2a:
        detail::math_op_type<UOP::div_ob2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_ob2a:
        detail::math_op_type<UOP::mod_ob2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_ob2a:
        detail::math_op_type<UOP::ceildiv_ob2a>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // destination: b
      case UOP::add_ob2b:
        detail::math_op_type<UOP::add_ob2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_ob2b:
        detail::math_op_type<UOP::sub_ob2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_ob2b:
        detail::math_op_type<UOP::mul_ob2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_ob2b:
        detail::math_op_type<UOP::div_ob2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_ob2b:
        detail::math_op_type<UOP::mod_ob2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_ob2b:
        detail::math_op_type<UOP::ceildiv_ob2b>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // sources: t b
        // destination: t
      case UOP::add_tb2t:
        detail::math_op_type<UOP::add_tb2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_tb2t:
        detail::math_op_type<UOP::sub_tb2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_tb2t:
        detail::math_op_type<UOP::mul_tb2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_tb2t:
        detail::math_op_type<UOP::div_tb2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_tb2t:
        detail::math_op_type<UOP::mod_tb2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_tb2t:
        detail::math_op_type<UOP::ceildiv_tb2t>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // destination: a
      case UOP::add_tb2a:
        detail::math_op_type<UOP::add_tb2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_tb2a:
        detail::math_op_type<UOP::sub_tb2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_tb2a:
        detail::math_op_type<UOP::mul_tb2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_tb2a:
        detail::math_op_type<UOP::div_tb2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_tb2a:
        detail::math_op_type<UOP::mod_tb2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_tb2a:
        detail::math_op_type<UOP::ceildiv_tb2a>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // destination: b
      case UOP::add_tb2b:
        detail::math_op_type<UOP::add_tb2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_tb2b:
        detail::math_op_type<UOP::sub_tb2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_tb2b:
        detail::math_op_type<UOP::mul_tb2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_tb2b:
        detail::math_op_type<UOP::div_tb2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_tb2b:
        detail::math_op_type<UOP::mod_tb2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_tb2b:
        detail::math_op_type<UOP::ceildiv_tb2b>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // sources: ao
        // destination: t
      case UOP::add_ao2t:
        detail::math_op_type<UOP::add_ao2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_ao2t:
        detail::math_op_type<UOP::sub_ao2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_ao2t:
        detail::math_op_type<UOP::mul_ao2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_ao2t:
        detail::math_op_type<UOP::div_ao2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_ao2t:
        detail::math_op_type<UOP::mod_ao2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_ao2t:
        detail::math_op_type<UOP::ceildiv_ao2t>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // destination: a
      case UOP::add_ao2a:
        detail::math_op_type<UOP::add_ao2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_ao2a:
        detail::math_op_type<UOP::sub_ao2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_ao2a:
        detail::math_op_type<UOP::mul_ao2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_ao2a:
        detail::math_op_type<UOP::div_ao2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_ao2a:
        detail::math_op_type<UOP::mod_ao2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_ao2a:
        detail::math_op_type<UOP::ceildiv_ao2a>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // destination: b
      case UOP::add_ao2b:
        detail::math_op_type<UOP::add_ao2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_ao2b:
        detail::math_op_type<UOP::sub_ao2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_ao2b:
        detail::math_op_type<UOP::mul_ao2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_ao2b:
        detail::math_op_type<UOP::div_ao2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_ao2b:
        detail::math_op_type<UOP::mod_ao2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_ao2b:
        detail::math_op_type<UOP::ceildiv_ao2b>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // sources: ai
        // destination: t
      case UOP::add_ai2t:
        detail::math_op_type<UOP::add_ai2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_ai2t:
        detail::math_op_type<UOP::sub_ai2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_ai2t:
        detail::math_op_type<UOP::mul_ai2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_ai2t:
        detail::math_op_type<UOP::div_ai2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_ai2t:
        detail::math_op_type<UOP::mod_ai2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_ai2t:
        detail::math_op_type<UOP::ceildiv_ai2t>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // destination: a
      case UOP::add_ai2a:
        detail::math_op_type<UOP::add_ai2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_ai2a:
        detail::math_op_type<UOP::sub_ai2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_ai2a:
        detail::math_op_type<UOP::mul_ai2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_ai2a:
        detail::math_op_type<UOP::div_ai2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_ai2a:
        detail::math_op_type<UOP::mod_ai2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_ai2a:
        detail::math_op_type<UOP::ceildiv_ai2a>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // destination: b
      case UOP::add_ai2b:
        detail::math_op_type<UOP::add_ai2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_ai2b:
        detail::math_op_type<UOP::sub_ai2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_ai2b:
        detail::math_op_type<UOP::mul_ai2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_ai2b:
        detail::math_op_type<UOP::div_ai2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_ai2b:
        detail::math_op_type<UOP::mod_ai2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_ai2b:
        detail::math_op_type<UOP::ceildiv_ai2b>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // sources: ib
        // destination: t
      case UOP::add_ib2t:
        detail::math_op_type<UOP::add_ib2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_ib2t:
        detail::math_op_type<UOP::sub_ib2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_ib2t:
        detail::math_op_type<UOP::mul_ib2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_ib2t:
        detail::math_op_type<UOP::div_ib2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_ib2t:
        detail::math_op_type<UOP::mod_ib2t>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_ib2t:
        detail::math_op_type<UOP::ceildiv_ib2t>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // destination: a
      case UOP::add_ib2a:
        detail::math_op_type<UOP::add_ib2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_ib2a:
        detail::math_op_type<UOP::sub_ib2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_ib2a:
        detail::math_op_type<UOP::mul_ib2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_ib2a:
        detail::math_op_type<UOP::div_ib2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_ib2a:
        detail::math_op_type<UOP::mod_ib2a>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_ib2a:
        detail::math_op_type<UOP::ceildiv_ib2a>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // destination: b
      case UOP::add_ib2b:
        detail::math_op_type<UOP::add_ib2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::sub_ib2b:
        detail::math_op_type<UOP::sub_ib2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mul_ib2b:
        detail::math_op_type<UOP::mul_ib2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::div_ib2b:
        detail::math_op_type<UOP::div_ib2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::mod_ib2b:
        detail::math_op_type<UOP::mod_ib2b>::execute(
            a, b, t, imm, operands, tmps);
        break;
      case UOP::ceildiv_ib2b:
        detail::math_op_type<UOP::ceildiv_ib2b>::execute(
            a, b, t, imm, operands, tmps);
        break;

        // Recall memoized into source
      case UOP::recall_t:
        RecallOp<mop_dst_kind::t>::execute(es, recalled_flag, a, b, imm, tmps);
        break;
      case UOP::recall_a:
        RecallOp<mop_dst_kind::a>::execute(es, recalled_flag, a, b, imm, tmps);
        break;
      case UOP::recall_b:
        RecallOp<mop_dst_kind::b>::execute(es, recalled_flag, a, b, imm, tmps);
        break;

        // Remember value from source
      case UOP::rem_t:
        RemOp<mop_dst_kind::t>::execute(es, a, b, imm, tmps);
        break;
      case UOP::rem_a:
        RemOp<mop_dst_kind::a>::execute(es, a, b, imm, tmps);
        break;
      case UOP::rem_b:
        RemOp<mop_dst_kind::b>::execute(es, a, b, imm, tmps);
        break;

        // Pickup operand and place as temporary
      case UOP::store_o2t:
        StoreToScrachOp::execute(operands, tmps);
        break;

        // Skip n operations if last recall succeed
      case UOP::rskip_n:
        RSkipNOp::execute(ip, imm, recalled_flag);
        break;
        // Drop n operations if last recall succeed
      case UOP::rdrop_n:
        RDropNOp::execute(operands, imm, recalled_flag);
        break;
        // Fill in symbolic value on operands stack
      case UOP::fill_sym:
        if (!FillSymOp::execute(operands, imm, fn))
          return c10::nullopt;
        break;
        // Unconditionally drop n operands
      case UOP::drop_n:
        DropNOp::execute(operands, imm, recalled_flag);
        break;
        // End of code
      case UOP::end:
        return EndOp::execute(tmps);
        break;
    }
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false, "Unterminated program");
  return c10::nullopt;
}

} // namespace glfdc
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
