#pragma once

#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

#include <c10/util/Exception.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <deque>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TensorDomain;
class TensorView;

class ComputeAt {
 public:
  // Runs the compute at pass making producer look like consumer, computing
  // producer relative to consumer
  static void runAt(
      TensorView* producer,
      TensorView* consumer,
      unsigned int consumer_position,
      ComputeAtMode mode = ComputeAtMode::Standard);

  // Runs the compute with pass making consumer look like producer, computing
  // producer relative to consumer
  static void runWith(
      TensorView* producer,
      TensorView* consumer,
      unsigned int producer_position,
      ComputeAtMode mode = ComputeAtMode::Standard);

 private:
  TensorView* producer_;
  TensorView* consumer_;
  TensorView* reference_;
  unsigned int reference_position_;
  unsigned int producer_position_ = 0;
  ComputeAtRootDomainMap root_map_;

  ComputeAtMode mode_ = ComputeAtMode::Standard;

  // Runs replayPasC and sets producer computeAt settings. Returns
  // producer_compute_at_pos.
  unsigned int backwardComputeAt_impl(
      TensorView* producer,
      TensorView* consumer,
      unsigned int consumer_compute_at_pos);

  // Runs replayCasP and sets producer computeAt settings. Returns
  // consumer_compute_at_pos.
  unsigned int forwardComputeAt_impl(
      TensorView* producer,
      TensorView* consumer,
      unsigned int producer_compute_at_pos);

  // Look through all the use chains of producer. Check if there's a single
  // consumer for all chains at or after the consumer specified in the computeAt
  // call.
  void setCommonConsumer();

  // Propagate backward from consumer to producer, check if it increase
  // computeAt position on tensors, if so take it!
  void traverseBackward();

  // Traverse from producer to common_consumer if it exists or through all uses
  // of producer
  void traverseForward();

  // Run the computeAt pass
  void runPass();

  // Common consumer if it exists
  TensorView* common_consumer_ = nullptr;

  // Producer use chains set in, used in a few spots.
  std::deque<std::deque<TensorView*>> producer_use_chains_;

  ComputeAt(
      TensorView* _producer,
      TensorView* _consumer,
      TensorView* _reference,
      unsigned int _reference_position,
      ComputeAtMode _mode);

  ComputeAt() = delete;
  ~ComputeAt() = default;
  ComputeAt(ComputeAt&) = delete;
  ComputeAt& operator=(const ComputeAt& other) = delete;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
