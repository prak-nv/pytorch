
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

namespace torch {
namespace jit {
namespace fuser {

class Kernel;

namespace kir {

//! Kernel IR builder interface
//!
//! TODO $$$
//!
class IrBuilder {
 public:
  explicit IrBuilder(Kernel* kernel) : kernel_(kernel) {}

  // Allocate a new IR node
  template <class T, class... Args>
  T* create(Args&&... args) {
    // TODO $$$
    return new T(std::forward<Args>(args)...);
  }

  // Binary expressions
  Val* andExpr(Val* lhs, Val* rhs);
  Val* eqExpr(Val* lhs, Val* rhs);
  Val* ltExpr(Val* lhs, Val* rhs);
  Val* addExpr(Val* lhs, Val* rhs);
  Val* subExpr(Val* lhs, Val* rhs);
  Val* mulExpr(Val* lhs, Val* rhs);
  Val* divExpr(Val* lhs, Val* rhs);
  Val* ceilDivExpr(Val* lhs, Val* rhs);
  Val* modExpr(Val* lhs, Val* rhs);

 private:
  Val* newResult(const Val* lhs, const Val* rhs);
  Val* newArithmeticExpr(BinaryOpType op_type, Val* lhs, Val* rhs);
  Val* newLogicExpr(BinaryOpType op_type, Val* lhs, Val* rhs);

 private:
  // Non-owning pointer to the kernel to be modified
  Kernel* kernel_ = nullptr;
};

} // namespace kir
} // namespace fuser
} // namespace jit
} // namespace torch
