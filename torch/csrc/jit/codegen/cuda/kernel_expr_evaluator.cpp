
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

void ExpressionEvaluator::print() const {
  std::cout << "\nEvaluation context\n";
  std::cout << "--------------------\n";
  /* TODO $$$
  for (const auto& kv : known_values_) {
    std::cout << kv.first << " = " << kv.second;
    if (kv.first->isConstScalar()) {
      std::cout << " ; original value = "
                << kv.first->as<Int>()->value().value();
    }
    std::cout << " ; " << *kv.first->getValType() << "\n";
  }
  */
  std::cout << "--------------------\n\n";
}

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
