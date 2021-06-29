#pragma once

#include <stack>
#include <tuple>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace glfdc {

// Forward declaration
enum class StackDir : unsigned;

// Index and cookie of value to replace
using BindingGapType = std::pair<std::size_t, std::uintptr_t>;
using BindCookieType = typename std::tuple_element<1, BindingGapType>::type;

template <typename Ty_, StackDir>
class ExternalStack;

// InitialStack - simple std::stack wrapper to allow bulk copy of elements to
// other stack.
template <typename Ty_>
class InitialStack : private std::stack<Ty_, std::vector<Ty_>> {
  using base_t = std::stack<Ty_, std::vector<Ty_>>;

 public:
  InitialStack() = default;
  InitialStack(const InitialStack&) = default;

  using base_t::emplace;
  using base_t::empty;
  using base_t::pop;
  using base_t::push;
  using base_t::size;
  using base_t::swap;
  using base_t::top;

  // Copies all elements of this stack to ExternalStack
  //
  // Defined in eval_stack.h
  template <StackDir Dir_>
  void copy_to(ExternalStack<Ty_, Dir_>& dst) const noexcept;

 protected:
  using base_t::c;
};

} // namespace glfdc
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch