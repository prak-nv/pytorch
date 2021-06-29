#pragma once

#include <cassert>
#include <limits>
#include <stack>
#include <tuple>
#include <utility>
#include <vector>

namespace glfdc {

using binding_gap_t = std::pair<std::size_t, uintptr_t>;
using bind_cookie_t = typename std::tuple_element<1, binding_gap_t>::type;

template <typename Ty_, typename C_ = std::vector<Ty_>>
class EvalStack : std::stack<Ty_, C_> {
  using base_t = std::stack<Ty_, C_>;

 public:
  static constexpr Ty_ GAP_VALUE = std::numeric_limits<Ty_>::max();

  EvalStack() = default;

  EvalStack(const EvalStack&) = default;
  EvalStack(EvalStack&&) = default;

  EvalStack& operator=(const EvalStack&) = default;
  EvalStack& operator=(EvalStack&&) = default;

  using base_t::emplace;
  using base_t::pop;
  using base_t::push;

  Ty_ pop_top() {
    assert(!this->c.empty());
    Ty_ ret = this->c.back();
    pop();
    return ret;
  }

  using base_t::swap;

  void drop(std::size_t n) noexcept {
    assert(n <= this->c.size());
    this->c.resize(this->c.size() - n);
  }

  template <typename BndFn_>
  bool fill_gaps(
      const std::vector<binding_gap_t>& binding_gaps,
      BndFn_ fn) noexcept {
    for (auto [idx, bind] : binding_gaps) {
      assert(idx < this->c.size());
      assert(this->c[idx] == GAP_VALUE);

      auto val = fn(bind);
      if (!val.has_value())
        return false;

      this->c[idx] = *val;
    }
    return true;
  }

  using base_t::empty;
  using base_t::size;
  using base_t::top;
};

} // namespace glfdc
