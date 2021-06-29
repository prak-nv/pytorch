#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include "glfdc/expr_builder.h"
#include "glfdc/eval.h"

#include <unordered_map>

namespace glfdc
{
struct ExpressionBuilder;
struct EvalState;
}


namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

class ExprBuilderVisitor : IrVisitor
{
public:
  explicit ExprBuilderVisitor(glfdc::ExpressionBuilder& bld, const Kernel& kernel);

  c10::optional<glfdc::Expr> build(const Val* value) {
    auto ref = build_operand(value);
    return builder_->create_expr(ref);
  }

  glfdc::Operand build_operand(const Val* value) {
    TORCH_CHECK(value->isScalar());
    TORCH_CHECK(value->dtype() == DataType::Int);

    if (value->isA<Int>())
    {
      if (value->isConst())
      {
         auto opt_val = value->as<Int>()->value();
         if (opt_val.has_value())
           return glfdc::Value(opt_val.value());
         else
           return builder_->get_binding(reinterpret_cast<uintptr_t>(value));
      }
    }

    current_operand_ = c10::nullopt;
    value->accept(this);
    TORCH_CHECK(current_operand_.has_value());
    auto ret = current_operand_.value();
    current_operand_ = c10::nullopt;
    return ret;
  }

private:
  void unhandled(const void*) final;
  void visit(const Int* value) final;
  void visit(const NamedScalar* named_scalar) final;
  void visit(const UnaryOp* unary_op) final;
  void visit(const BinaryOp* binary_op) final;

  glfdc::ExpressionBuilder *builder_;
  c10::optional<glfdc::Operand> current_operand_ = c10::nullopt;
  const Kernel* kernel_;
};

inline ExprBuilderVisitor::ExprBuilderVisitor(glfdc::ExpressionBuilder& bld, const Kernel& kernel) : builder_(&bld), kernel_(&kernel)
{
}

inline void ExprBuilderVisitor::unhandled(const void*) {
  TORCH_INTERNAL_ASSERT(
      false, "Kernel IR expression evaluation reached an unsupported node");
}

inline void ExprBuilderVisitor::visit(const Int* value) {
  TORCH_INTERNAL_ASSERT(!value->isConst());
  if (auto def = value->definition()) {
    def->accept(this);
  } else if (kernel_->isInput(value)) {
    current_operand_ = builder_->get_binding(reinterpret_cast<uintptr_t>(value));
  }
}

inline void ExprBuilderVisitor::visit(const UnaryOp* unary_op) {

  glfdc::Operand oper = build_operand(unary_op->in());
  switch (unary_op->operation())
  {
  case UnaryOpType::Neg:
    current_operand_ = builder_->create_sexpr(glfdc::OperatorKind::sub, glfdc::Value(0), oper);
    break;
  case UnaryOpType::Cast:
    current_operand_ = oper;
  default:
    break;
  }

  TORCH_INTERNAL_ASSERT(current_operand_.has_value() && "Unexpected operator type");
}

inline void ExprBuilderVisitor::visit(const BinaryOp* binary_op) {
  auto lhs = build_operand(binary_op->lhs());
  auto rhs = build_operand(binary_op->rhs());

  c10::optional<glfdc::OperatorKind> oper;

  switch (binary_op->operation())
  {
    case BinaryOpType::Add:
      oper = glfdc::OperatorKind::add;
      break;
    case BinaryOpType::Sub:
      oper = glfdc::OperatorKind::sub;
      break;
    case BinaryOpType::Mul:
      oper = glfdc::OperatorKind::mul;
      break;
    case BinaryOpType::Div:
      oper = glfdc::OperatorKind::div;
      break;
    case BinaryOpType::Mod:
      oper = glfdc::OperatorKind::mod;
      break;
    case BinaryOpType::CeilDiv:
      oper = glfdc::OperatorKind::ceildiv;
      break;
  }

  TORCH_INTERNAL_ASSERT(oper.has_value() && "Unexpected operator type");

  current_operand_ = builder_->create_sexpr(oper.value(), lhs, rhs);
}

inline void ExprBuilderVisitor::visit(const NamedScalar* scalar) {
  current_operand_ = builder_->get_binding(reinterpret_cast<uintptr_t>(scalar));
  // It's a legal expresison node so we must handle it
}

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
