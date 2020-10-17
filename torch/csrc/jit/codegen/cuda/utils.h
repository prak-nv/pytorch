#pragma once

#include <c10/util/Exception.h>

#include <unordered_map>
#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Common Functions
constexpr int64_t ceilDiv(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

// Simple mixin for suppressing copy & move operations, ex:
//
//  class Foo : public NonCopyable {
//   ...
//  };
//
class NonCopyable {
 public:
  NonCopyable() = default;

  // No copy/move semantics
  NonCopyable(const NonCopyable&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
};

// A generic root for a hierarchy of polymorphic classes:
// - It ensures virtual destructors
// - Provides the base->as<Derived>() and node->isA<T>() notation
class PolymorphicBase {
 public:
  virtual ~PolymorphicBase() = default;

  // Replacement for static_cast<T*>(ptr): ptr->as<T>()
  // (checked in DEBUG builds)
  template <class T>
  T* as() {
#ifdef NDEBUG
    auto downcast_ptr = static_cast<T*>(this);
#else
    auto downcast_ptr = dynamic_cast<T*>(this);
    TORCH_INTERNAL_ASSERT(downcast_ptr != nullptr);
#endif
    return downcast_ptr;
  }

  template <class T>
  const T* as() const {
#ifdef NDEBUG
    auto downcast_ptr = static_cast<const T*>(this);
#else
    auto downcast_ptr = dynamic_cast<const T*>(this);
    TORCH_INTERNAL_ASSERT(downcast_ptr != nullptr);
#endif
    return downcast_ptr;
  }

  // Check if the runtime time is T (or derived from T)
  //
  // NOTE: Don't use this for conditional casts. Use:
  //
  //  if (auto t = dynamic_cast<T>(p)) { ... }
  //
  // instead of:
  //
  //  if (p->isA<T>()) { auto t = p->as<T>(); ... }
  //
  template <class T>
  bool isA() const {
    return dynamic_cast<const T*>(this) != nullptr;
  }
};

//! Container class DisjointSet models equivalence relationships
//!
//! Each instance of this class keeps a set of equivalent classes
//! DisjointSet::join(a,b) makes the full class of a and b equivalent
//! DisjointSet::areEqual(a,b) checks if a and b belong same class
//!
//! \note The template type T is assumed to be hashable
template <typename T, typename Hash=std::hash<T>>
class DisjointSet {
 public:
  DisjointSet() = default;

  //! Joins the equivalent class that a and b belong to
  //! areEqual(a',b') will be true for each a'=a and b'=b
  //!
  //! \param a An element from a equivalent class
  //!          will create a new equivalent class if a does
  //!          not belong to any
  //! \param b An element from another equivalent class
  //!          will create a new equivalent class if b does
  //!          not belong to any
  void join(T a, T b) {
    // cases where either of the quiv class doesn't exist
    if (!entry_map.count(a) && !entry_map.count(b)) {
      createPoint(a);
      entry_map[b] = fixedPoint(a);
    } else if (!entry_map.count(a)) {
      entry_map[a] = fixedPoint(b);
    } else if (!entry_map.count(b)) {
      entry_map[b] = fixedPoint(a);
    } else {
      // case where both equiv classes exist and need to join
      const int i0 = fixedPoint(a);
      const int i1 = fixedPoint(b);
      int new_parent = 0;
      int new_child = 0;

      // Either order here is correct but joining larger class to smaller class
      // tend to be faster
      std::tie(new_parent, new_child) = (weights[i0] < weights[i1])
          ? std::make_pair(i0, i1)
          : std::make_pair(i1, i0);
      weights[new_parent] += weights[new_child];
      set_map[new_child] = new_parent;
    }
  }

  //! Checks if a and b belong to the same equivalent class
  //!
  //! \param a An element from a equivalent class
  //! \param b An element from another equivalent class
  //! \returns Boolean value representing if a and b are
  //!          recorded to be in the same equivalent class
  //!          will return false if any of a or b doesn't
  //!          have an equivalent class recorded
  bool areEquivalent(T a, T b) const {
    if (!entry_map.count(a) || !entry_map.count(b)) {
      return false;
    }
    return fixedPoint(a) == fixedPoint(b);
  }

  bool contains(T a) const {
    return entry_map.count(a) > 0;
  }

  bool insert(T a) {
    if (!contains(a)) {
      createPoint(a);
      return true;
    } else {
      return false;
    }
  }

  std::ostream& print(std::ostream& os) const {
    std::unordered_map<int, std::unordered_set<T, Hash>> fixedPointMap;
    for (const auto& kv: entry_map) {
      int fixed_point = fixedPoint(kv.first);
      auto it = fixedPointMap.find(fixed_point);
      if (it == fixedPointMap.end()) {
        it = fixedPointMap.insert({fixed_point, {}}).first;
      }
      it->second.insert(kv.first);
    }
    os << "{\n";
    for (const auto& kv: fixedPointMap) {
      os << "\t{ ";
      for (const auto& val: kv.second) {
        os << val << " ";
      }
      os << "}\n";
    }
    os << "}\n";
    return os;
  }

 private:
  // Internal fixed point implementation:
  //  Returns the equivalent class that e belongs to
  int fixedPoint(int e) const {
    TORCH_INTERNAL_ASSERT(static_cast<int>(set_map.size()) > e);
    while (set_map[e] != e) {
      // Chasing to fixed point
      e = set_map[e];
    }
    return e;
  }

  //! Utility to check the class i belongs to:
  //!
  //! Will create a new class if no match seen
  //! \param e element e to find the equiv class for
  //! \returns the equivalent class that e belongs to
  //!
  int fixedPoint(T e) const {
    // Handles case when i doesn't have an equivalence class
    TORCH_INTERNAL_ASSERT(entry_map.count(e));

    // Use fixed point as a representation for the equiv class
    return fixedPoint(entry_map.at(e));
  }

  //! Utility to create a new equiv class for i
  //
  //! \param i Element i to create the equiv class for
  void createPoint(T i) {
    entry_map[i] = next_index_;
    set_map.push_back(next_index_++);
    weights.push_back(1);
  }

 private:
  // Internal representation of the equivalence class as integers
  // set_map implements the "parent" relationship
  std::vector<int> set_map;
  // Weights is used for preliminary perf optimization
  std::vector<int> weights;

  // Map the input of type T to its equivalence class
  std::unordered_map<T, int, Hash> entry_map;

  // Running counter for generating new index when
  // Creating new equiv classes
  int next_index_ = 0;
};

template <typename T, typename Hash>
inline std::ostream& operator<<(std::ostream& os, DisjointSet<T, Hash>& s) {
  return s.print(os);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
