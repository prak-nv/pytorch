#pragma once

#include <torch/csrc/jit/codegen/cuda/glfdc/initial_stack.h>

#include <limits>
#include <tuple>
#include <utility>

#include <c10/util/Exception.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace glfdc {

// Determines memory layout of the stack.
enum class StackDir : unsigned { GrowsUp, GrowsDown };

// DEVector - Double ended vector
//
// Two vectors in single external memory block allocation.
// Allows insert and remove elements from two ends ie.
// at front and back of memory block.
template <typename Ty_>
class DEVector {
  static_assert(
      std::is_trivially_destructible<Ty_>::value,
      "DEVector doesn't call destructors");

 public:
  // From which end of memory block should be accessed
  enum class Access { FromFront, FromBack };

  DEVector(size_t max_capacity, void* memory) noexcept
      : data_(static_cast<Ty_*>(memory)), max_capacity_(max_capacity) {}

  DEVector(const DEVector&) noexcept = default;
  DEVector(DEVector&&) noexcept = delete;

  DEVector& operator=(const DEVector&) = default;
  DEVector& operator=(DEVector&&) = delete;

  template <Access E>
  void push_back(Ty_ v) noexcept {
    if (E == Access::FromFront) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          (front_size_) <= (max_capacity_ - back_size_));
      data_[front_size_++] = v;
    } else {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          (back_size_) <= (max_capacity_ - front_size_));
      data_[max_capacity_ - 1 - back_size_] = v;
      ++back_size_;
    }
  }

  template <Access E>
  void pop_back() noexcept {
    if (E == Access::FromFront) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(front_size_ > 0);
      --front_size_;
    } else {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(back_size_ > 0);
      --back_size_;
    }
  }

  // Returns copy of element at back of the vector
  template <Access E>
  Ty_ back() const noexcept {
    if (E == Access::FromFront) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(front_size_ > 0);
      return data_[front_size_ - 1];
    } else {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(back_size_ > 0);
      return data_[max_capacity_ - back_size_];
    }
  }

  template <Access E>
  Ty_& at(size_t i) noexcept {
    if (E == Access::FromFront) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(i < (max_capacity_ - back_size_));
      return data_[i];
    } else {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(i < (max_capacity_ - front_size_));
      return data_[max_capacity_ - 1 - i];
    }
  }

  template <Access E>
  bool empty() const noexcept {
    if (E == Access::FromFront) {
      return front_size_ == 0;
    }
    return back_size_ == 0;
  }

  template <Access E>
  std::size_t size() const noexcept {
    if (E == Access::FromFront) {
      return front_size_;
    }
    return back_size_;
  }

  template <Access E>
  void set_size(std::size_t sz) noexcept {
    if (E == Access::FromFront) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          sz <= (max_capacity_ - back_size_),
          "Front size bigger than capacity");
      front_size_ = sz;
    } else {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          sz <= (max_capacity_ - front_size_),
          "Back size bigger than capacity");
      back_size_ = sz;
    }
  }

  template <Access E>
  std::size_t capacity() const noexcept {
    if (E == Access::FromFront) {
      return max_capacity_ - back_size_;
    } else {
      return max_capacity_ - front_size_;
    }
  }

 private:
  Ty_* data_ = 0;
  std::size_t front_size_ = 0;
  std::size_t back_size_ = 0;
  std::size_t max_capacity_;
};

// ExternalStack - stack implementation using preallocated memory block, that
// can grow up/down
template <typename Ty_, StackDir Dir_>
class ExternalStack {
  static_assert(
      std::is_trivially_move_constructible<Ty_>::value,
      "Unsupported type");
  static_assert(
      std::is_trivially_copy_constructible<Ty_>::value,
      "Unsupported type");
  static_assert(
      std::is_trivially_move_assignable<Ty_>::value,
      "Unsupported type");
  static_assert(
      std::is_trivially_copy_assignable<Ty_>::value,
      "Unsupported type");

  // Double ended vector type used as storage
  using Container = DEVector<Ty_>;
  using Access = typename Container::Access;

  // Determine from which side we should use double ended vector
  static constexpr Access Dir =
      (Dir_ == StackDir::GrowsUp) ? Access::FromFront : Access::FromBack;

  // Friendship for InitialStack::copy_to
  friend class InitialStack<Ty_>;

 public:
  explicit ExternalStack(Container& mem_block) noexcept : de_vec_(&mem_block) {}
  ExternalStack() = delete;
  ExternalStack(const ExternalStack&) = delete;
  ExternalStack(ExternalStack&&) = delete;

  ExternalStack& operator=(const ExternalStack&) = delete;
  ExternalStack& operator=(ExternalStack&&) = delete;

  Ty_ pop_top() noexcept {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(de_vec_ != nullptr);
    Ty_ result = de_vec_->template back<Dir>();
    de_vec_->template pop_back<Dir>();
    return result;
  }

  Ty_ top() const noexcept {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(de_vec_ != nullptr);
    Ty_ result = de_vec_->template back<Dir>();
    return result;
  }

  void drop(std::size_t n) noexcept {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(de_vec_ != nullptr);
    size_t new_size = de_vec_->template size<Dir>() - n;
    de_vec_->template set_size<Dir>(new_size);
  }

  void fill_at(std::size_t index, Ty_ v) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(de_vec_ != nullptr);
    de_vec_->template at<Dir>(index) = v;
  }

  void push(Ty_ v) noexcept {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(de_vec_ != nullptr);
    de_vec_->template push_back<Dir>(v);
  }

  std::size_t size() const noexcept {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(de_vec_ != nullptr);
    return de_vec_->template size<Dir>();
  }

  bool empty() const noexcept {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(de_vec_ != nullptr);
    return de_vec_->template empty();
  }

  std::size_t capacity() const noexcept {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(de_vec_ != nullptr);
    return de_vec_->template capacity<Dir>();
  }

 private:
  void set_size(std::size_t sz) noexcept {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(de_vec_ != nullptr);
    de_vec_->template set_size<Dir>(sz);
  }

 private:
  Container* de_vec_ = nullptr;
};

// EvalStack - Simple std::stack extension for use in stack machine evaluation.
template <typename Ty_>
class EvalStack : InitialStack<Ty_> {
  using base_t = InitialStack<Ty_>;

 public:
  static constexpr Ty_ GAP_VALUE = std::numeric_limits<Ty_>::max();

  EvalStack() = default;

  EvalStack(const base_t& stack) : base_t(stack) {}

  EvalStack(const EvalStack&) = default;
  EvalStack(EvalStack&&) = default;

  EvalStack& operator=(const EvalStack&) = default;
  EvalStack& operator=(EvalStack&&) = default;

  using base_t::emplace;
  using base_t::pop;
  using base_t::push;

  // pop_top - Shorthand for stack.top(); stack.pop();
  Ty_ pop_top() {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!this->c.empty());
    Ty_ ret = this->c.back();
    pop();
    return ret;
  }

  using base_t::swap;

  // drop - drops n operands from stack
  void drop(std::size_t n) noexcept {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(n <= this->c.size());
    this->c.resize(this->c.size() - n);
  }

  // Replaces placeholder values based on indices from binding_gaps vector
  // using value returned by fn
  template <typename BndFn_>
  bool fillSymbolicValues(
      const std::vector<BindingGapType>& binding_gaps,
      const BndFn_& binding) noexcept {
    for (auto binding_idx : binding_gaps) {
      auto idx = binding_idx.first;
      auto cookie = binding_idx.second;

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(idx < this->c.size());
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(this->c[idx] == GAP_VALUE);

      auto val = binding.get(cookie);

      if (!val.has_value()) {
        return false;
      }
      this->c[idx] = *val;
    }
    return true;
  }

  using base_t::empty;
  using base_t::size;
  using base_t::top;
};

template <typename Ty_>
template <StackDir Dir_>
inline void InitialStack<Ty_>::copy_to(
    ExternalStack<Ty_, Dir_>& copy) const noexcept {
  // Set size of destination stack
  const std::size_t sz = this->size();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      sz <= copy.capacity(), "Capacity of External block is too small");

  copy.set_size(sz);

  // Copy all elements one by one
  for (std::size_t i = 0; i < sz; ++i) {
    copy.fill_at(i, this->c[i]);
  }
}

} // namespace glfdc
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
