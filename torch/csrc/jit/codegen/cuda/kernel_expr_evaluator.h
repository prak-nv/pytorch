
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <c10/util/Optional.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

//! $$$
class TORCH_CUDA_API ExpressionEvaluator : private IrVisitor {
 public:
  //! $$$
  void bind(const Val* value, Int::ScalarType concrete_value);

  //! $$$
  c10::optional<Int::ScalarType> evaluate(const Val* value);

  //! $$$
  static bool isConst(const Val* value);

  //! Debugging helper, prints all the currently known values
  void print() const;

 private:
  void unhandled(const void*) final;
  void visit(const Int* value) final;
  void visit(const NamedScalar* named_scalar) final;
  void visit(const UnaryOp* unary_op) final;
  void visit(const BinaryOp* binary_op) final;

 private:
  std::unordered_map<const Val*, Int::ScalarType> known_values_;
};

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
