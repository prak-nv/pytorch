#pragma once

#include <c10/util/Exception.h>

#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Container class DisjointSet models equivalence relationships
//!
//! Each instance of this class keeps a set of equivalent classes
//! DisjointSet::join(a,b) makes the full class of a and b equivalent
//! DisjointSet::areEqual(a,b) checks if a and b belong same class
//!
//! \note The template type T is assumed to be hashable
template <typename T>
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
      entry_map[b] = getOrCreateFixedPoint(a);
    } else if (!entry_map.count(a)) {
      entry_map[a] = getOrCreateFixedPoint(b);
    } else if (!entry_map.count(b)) {
      entry_map[b] = getOrCreateFixedPoint(a);
    } else {
      // case where both equiv classes exist and need to join
      const int i0 = getOrCreateFixedPoint(a);
      const int i1 = getOrCreateFixedPoint(b);
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
    return getOrCreateFixedPoint(a) == getOrCreateFixedPoint(b);
  }

 private:
  // Internal fixed point implementation:
  //  Returns the equivalent class that e belongs to
  int getFixedPoint(int e) const {
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
  int getOrCreateFixedPoint(T e) const {
    // Handles case when i doesn't have an equivalence class
    TORCH_INTERNAL_ASSERT(entry_map.count(e));

    // Use fixed point as a representation for the equiv class
    return getFixedPoint(entry_map.at(e));
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
  std::unordered_map<T, int> entry_map;

  // Running counter for generating new index when
  // Creating new equiv classes
  int next_index_ = 0;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
