#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/glfdc/eval.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

namespace glfdc {
struct ExpressionBuilder;
} // namespace glfdc

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

// IR visitor creating memoized expressions
class ExprBuilderVisitor : IrVisitor {
 public:
  explicit ExprBuilderVisitor(
      glfdc::ExpressionBuilder& bld,
      const Kernel& kernel);

  glfdc::ExprEvaluator build(const Val* value);

  std::unique_ptr<glfdc::ExprDAG> takeDAG() noexcept;

 private:
  glfdc::Operand buildOperand(const Val* value);
  void unhandled(const void*) final;
  void visit(const Int* value) final;
  void visit(const NamedScalar* named_scalar) final;
  void visit(const UnaryOp* unary_op) final;
  void visit(const BinaryOp* binary_op) final;

  glfdc::ExpressionBuilder* builder_ = nullptr;
  c10::optional<glfdc::Operand> current_operand_ = c10::nullopt;
  const Kernel* kernel_ = nullptr;
};

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
