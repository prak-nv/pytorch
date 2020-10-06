
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace kir {

class Kernel;

//! Maps TensorViews to std::pair<ir_utils::ParallelTypeBitmap, SourceMapType>>
//!
//! Map from TensorView to bit set represnting <BIDx, BIDy, BIDz, TIDx, TIDy,
//! TIDz> If any dependency of TV had a parallelized reduction, we will track
//! it here. This will be used for predicate generation to prevent
//! parallelization on that axis. This is important if we have a reduction on
//! for example TIDx, as the reduced value is only valid on threadIdx.x == 0
//! therefore if we use that value later in the kernel we have that predicate.
//! If we follow a reduction parallelized on TIDx with a broadcast on TIDx we
//! no longer need the predicate and can reset the bit accordingly
//!
class TORCH_CUDA_API ThreadPredicateMap {
 public:
  using SourceMapType = std::unordered_map<
      ParallelType,
      std::unordered_set<const kir::TensorView*>,
      TypeHash>;

  // TODO(kir): replace std::pair<> with struct
  using MapType = std::unordered_map<
      const kir::TensorView*,
      std::pair<ir_utils::ParallelTypeBitmap, SourceMapType>>;

  using const_iterator = MapType::const_iterator;

 public:
  explicit ThreadPredicateMap(const kir::Kernel* kernel);

  // TODO(kir): these methods are only used by getParallelBroadcastDomains()
  const_iterator find(const kir::TensorView* tv) const;
  const_iterator end() const;
  const MapType::mapped_type& at(const kir::TensorView* tv) const;
  MapType::mapped_type& at(const kir::TensorView* tv);

  // Returns a Bool predicate expression for a given output TensorView.
  kir::Bool* getExpr(const kir::TensorView* out_tv) const;

 private:
  // Update the thread_predicates bitset based on provided Expr
  void updateBitSet(kir::Expr*);

  void insert(
      const kir::TensorView* tv,
      const ir_utils::ParallelTypeBitmap& pred,
      const SourceMapType& src_map);

  void insert(
      const kir::TensorView* tv,
      const MapType::mapped_type& pred_and_src);

 private:
  MapType thread_predicates_;
};

} // namespace kir
} // namespace fuser
} // namespace jit
} // namespace torch
