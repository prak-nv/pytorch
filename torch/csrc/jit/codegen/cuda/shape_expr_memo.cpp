#include <torch/csrc/jit/codegen/cuda/shape_expr_memo.h>

#include <torch/csrc/jit/codegen/cuda/glfdc/expr_builder.h>

using namespace torch::jit::fuser::cuda;
using kir::ExprBuilderVisitor;

ExprBuilderVisitor::ExprBuilderVisitor(
    glfdc::ExpressionBuilder& bld,
    const Kernel& kernel)
    : builder_(&bld), kernel_(&kernel) {}

glfdc::SymbolicExpr ExprBuilderVisitor::build(const kir::Val* value) {
  auto ref = buildOperand(value);
  return glfdc::SymbolicExpr{builder_->createExpr(ref)};
}

glfdc::SymbolicExpr ExprBuilderVisitor::build(int64_t value) {
  return glfdc::SymbolicExpr(builder_->createExpr(glfdc::Value(value)));
}

glfdc::SymbolicExpr ExprBuilderVisitor::buildPaddedWarp(
    const glfdc::SymbolicExpr& e) {
  TORCH_INTERNAL_ASSERT(
      &e.dag() == &builder_->dag(), "Expression not in current DAG");
  // Round up symbolic value e to C10_WARP_SIZE
  // ceiled =  (e + C10_WARP_SIZE - 1) / C10_WARP_SIZE
  auto ceiling = builder_->createOperand(
      glfdc::OperatorKind::ceildiv,
      e.expr().operand(),
      glfdc::Value(C10_WARP_SIZE));
  // ceiled * C10_WARP_SIZE
  auto rounded_up = builder_->createOperand(
      glfdc::OperatorKind::mul,
      glfdc::Operand(ceiling),
      glfdc::Value(C10_WARP_SIZE));

  return glfdc::SymbolicExpr(builder_->createExpr(rounded_up));
}

std::unique_ptr<glfdc::ExprDAG> ExprBuilderVisitor::takeDAG() noexcept {
  return builder_->takeDAG();
}

// Creates an operand for expression symbolic or exact value or subexpression
glfdc::Operand ExprBuilderVisitor::buildOperand(const kir::Val* value) {
  TORCH_INTERNAL_ASSERT(value->isScalar(), "Operand should be scalar");
  TORCH_INTERNAL_ASSERT(
      value->dtype() == DataType::Int, "Operand should be integer");

  if (value->isA<kir::Int>()) {
    if (value->isConst()) {
      auto opt_val = value->as<kir::Int>()->value();
      if (opt_val.has_value()) {
        return glfdc::Value(opt_val.value());
      } else {
        return builder_->getBinding(reinterpret_cast<uintptr_t>(value));
      }
    }
  }

  current_operand_ = c10::nullopt;
  value->accept(this);

  TORCH_INTERNAL_ASSERT(current_operand_.has_value());
  auto ret = current_operand_.value();
  current_operand_ = c10::nullopt;

  return ret;
}

void ExprBuilderVisitor::unhandled(const void*) {
  TORCH_INTERNAL_ASSERT(
      false, "Kernel IR expression evaluation reached an unsupported node");
}

void ExprBuilderVisitor::visit(const kir::Int* value) {
  TORCH_INTERNAL_ASSERT(!value->isConst());
  if (auto def = value->definition()) {
    def->accept(this);
  } else if (kernel_->isInput(value)) {
    current_operand_ = builder_->getBinding(reinterpret_cast<uintptr_t>(value));
  }
}

void ExprBuilderVisitor::visit(const kir::UnaryOp* unary_op) {
  glfdc::Operand oper = buildOperand(unary_op->in());
  switch (unary_op->operation()) {
    case UnaryOpType::Neg:
      // Negation is not directly supported, thus we form (0 - x) expression
      current_operand_ = builder_->createOperand(
          glfdc::OperatorKind::sub, glfdc::Value(0), oper);
      break;
    case UnaryOpType::Cast:
      current_operand_ = oper;
    default:
      break;
  }

  TORCH_INTERNAL_ASSERT(
      current_operand_.has_value() && "Unexpected operator type");
}

void ExprBuilderVisitor::visit(const kir::BinaryOp* binary_op) {
  // Traverse left of subexpression
  auto lhs = buildOperand(binary_op->lhs());

  // Traverse right of subexpression
  auto rhs = buildOperand(binary_op->rhs());

  c10::optional<glfdc::OperatorKind> oper;

  switch (binary_op->operation()) {
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

  // Create partial subexpression for binary operation.
  current_operand_ = builder_->createOperand(oper.value(), lhs, rhs);
}

void ExprBuilderVisitor::visit(const kir::NamedScalar* scalar) {
  // Create runtime binding for symbolic value.
  current_operand_ = builder_->getBinding(reinterpret_cast<uintptr_t>(scalar));
}
