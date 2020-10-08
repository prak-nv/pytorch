
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
  void bind(const Val* value, Int::ScalarType concrete_value) {
    TORCH_CHECK(value->isScalar());
    TORCH_CHECK(value->dtype() == DataType::Int);
    TORCH_CHECK(!value->isConst(), "Tried to bind to a constant value");
    TORCH_CHECK(
        value->definition() == nullptr,
        "Tried to bind to a value that is computed in the kernel IR");
    known_values_[value] = concrete_value;
  }

  //! $$$
  c10::optional<Int::ScalarType> evaluate(const Val* value) {
    TORCH_CHECK(value->isScalar());
    TORCH_CHECK(value->dtype() == DataType::Int);

    //$$$ PERF_SCOPE

    // Const scalar?
    if (value->isScalar() && value->isConst()) {
      return value->as<Int>()->value();
    }

    // Is the value known (either explicit binding or memoized)?
    const auto it = known_values_.find(value);
    if (it != known_values_.end()) {
      return it->second;
    }

    value->accept(this);
    return known_values_[value];
  }

  //! Debugging helper, prints all the currently known values
  void print() const;

 private:
  void unhandled(const void*) final {
    TORCH_INTERNAL_ASSERT(
        false, "Kernel IR expression evaluation reached an unsupported node");
  }

  void visit(const Int* value) final {
    TORCH_INTERNAL_ASSERT(!value->isConst());
    if (auto def = value->definition()) {
      def->accept(this);
    }
  }

  void visit(const NamedScalar* named_scalar) final {
    TORCH_INTERNAL_ASSERT(
        false, "Attempting to evaluate an unbound named scalar");
  }

  void visit(const UnaryOp* unary_op) final {
    const auto in = evaluate(unary_op->in());
    if (in.has_value()) {
      switch (unary_op->operation()) {
        case UnaryOpType::Neg:
          known_values_[unary_op->out()] = -*in;
          break;
        case UnaryOpType::Cast:
          known_values_[unary_op->out()] = *in;
          break;
        default:
          TORCH_CHECK(!"Unexpected operator type");
      }
    }
  }

  void visit(const BinaryOp* binary_op) final {
    const auto lhs = evaluate(binary_op->lhs());
    const auto rhs = evaluate(binary_op->rhs());
    if (lhs.has_value() && rhs.has_value()) {
      switch (binary_op->operation()) {
        case BinaryOpType::Add:
          known_values_[binary_op->out()] = *lhs + *rhs;
          break;
        case BinaryOpType::Sub:
          known_values_[binary_op->out()] = *lhs - *rhs;
          break;
        case BinaryOpType::Mul:
          known_values_[binary_op->out()] = *lhs * *rhs;
          break;
        case BinaryOpType::Div:
          TORCH_CHECK(*rhs != 0);
          known_values_[binary_op->out()] = *lhs / *rhs;
          break;
        case BinaryOpType::Mod:
          TORCH_CHECK(*rhs != 0);
          known_values_[binary_op->out()] = *lhs % *rhs;
          break;
        case BinaryOpType::CeilDiv:
          TORCH_CHECK(*rhs != 0);
          known_values_[binary_op->out()] = (*lhs + *rhs - 1) / *rhs;
          break;
        case BinaryOpType::And:
          known_values_[binary_op->out()] = Int::ScalarType(*lhs && *rhs);
          break;
        default:
          TORCH_CHECK(!"Unexpected operator type");
      }
    }
  }

 private:
  std::unordered_map<const Val*, Int::ScalarType> known_values_;
};

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
