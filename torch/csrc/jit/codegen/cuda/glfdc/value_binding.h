#pragma once

#include <torch/csrc/jit/codegen/cuda/glfdc/sexpr.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace glfdc {

// Interface for obtaining symbolic values
class BoundValueProvider {
 public:
  virtual opt_scalar_t get(uintptr_t) const noexcept = 0;
};

// Base for symbolic value provider
template <typename Ty_>
class SymbolicValueProviderBase : private BoundValueProvider {
 public:
  virtual opt_scalar_t getSymbolValue(const Ty_*) const noexcept = 0;

  // Returns interface for symbolic value provider
  virtual const BoundValueProvider& getProvider() const noexcept {
    return *this;
  }

 private:
  opt_scalar_t get(uintptr_t cookie) const noexcept override {
    return getSymbolValue(reinterpret_cast<const Ty_*>(cookie));
  }
};

} // namespace glfdc
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch