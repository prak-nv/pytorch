#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>

#include <c10/util/Optional.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Calculate Fusion IR expressions
class TORCH_CUDA_CU_API ExpressionEvaluator : private OptOutDispatch {
 public:
  explicit ExpressionEvaluator(Fusion* fusion) : fusion_(fusion) {}

  //! Returns the associated fusion object
  Fusion* fusion() const {
    return fusion_;
  }

  //! Bind a concrete value to an IR variable
  void bind(Val* value, Int::ScalarType concrete_value);

  //! Try to evaluate a Fusion IR value
  c10::optional<Int::ScalarType> evaluate(Val* value);

  //! Debugging helper, prints all the currently known values
  void print() const;

 protected:
  c10::optional<Int::ScalarType> getValue(Val* value);

  using OptOutDispatch::handle;
  void handle(UnaryOp*) override;
  void handle(BinaryOp*) override;

 private:
  std::unordered_map<const Val*, Int::ScalarType> known_values_;
  Fusion* fusion_ = nullptr;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
