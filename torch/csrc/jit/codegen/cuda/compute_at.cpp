#include <torch/csrc/jit/codegen/cuda/compute_at.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

ComputeAtData::ComputeAtData(TensorView* tv)
    : tv_ref_(tv),
      original_has_compute_at_(tv->hasComputeAt()),
      original_compute_at_position(tv->getThisComputeAtAxis()),
      original_domain_(tv->domain()),
      new_compute_at_domain_(tv->domain()) {}

// Clear pass based data
void ComputeAtData::clearPass() {
  // If the last pass set a position, update the new_compute_at_position if
  // latest position would be greater than previously set.
  if (current_traversal_position_set &&
      current_traversal_position > new_compute_at_position) {
    new_compute_at_position = current_traversal_position;
  }

  current_traversal_position_set = false;
  current_traversal_position = 0;
}

void ComputeAtData::setPassPosition(unsigned int pos) {
  if (current_traversal_position_set) {
    // A single traversal cannot try to enforce more than one position on a
    // TensorView as it would produce in incorrect code. If this is hit, then
    // the given tensor and its production should be duplicated.
    TORCH_CHECK(
        pos == current_traversal_position,
        "Error during computeAt. ComputeAt pass wanted to set position of TensorView: ",
        tv_ref_->name(),
        " at position ",
        pos,
        " but was already set to position ",
        current_traversal_position,
        ". This tensor would have to be recomputed to satsify the selected computeAt position.");
  }

  if (pos > original_compute_at_position) {
    current_traversal_position = pos;
    touched_ = true;
    current_traversal_position_set = true;
  }
}

unsigned int ComputeAtData::getNewPosition() const {
  // If the last pass set a position, return the latest position if
  // it would be greater than previously set.
  if (current_traversal_position_set &&
      current_traversal_position > new_compute_at_position) {
    return current_traversal_position;
  } else {
    return new_compute_at_position;
  }
}

void ComputeAtData::validateNewComputeAt() const {
  FUSER_PERF_SCOPE("validateNewComputeAt");

  TORCH_INTERNAL_ASSERT(
      !touched() || getNewPosition() >= original_compute_at_position,
      "Invalid computeAt detected. This computeAt would invalidate the set computeAt on ",
      tv_ref_,
      " as the new computeAt position was found to be ",
      getNewPosition(),
      ".");
  auto mismatch = BestEffortReplay::findFirstMismatchedID(
      tv_ref_->domain(), original_domain_);
  TORCH_CHECK(
      mismatch >= (int)original_compute_at_position,
      "Invalid computeAt detected. This computeAt call would invalidate the set computeAt on ",
      tv_ref_,
      " as the previous set computeAt was on the domain ",
      original_domain_,
      " with a computeAt position of ",
      original_compute_at_position,
      ".");
}

void ComputeAtData::setComputeAtDomain(TensorDomain* td) {
  if (new_compute_at_domain_ != original_domain_) {
    size_t mismatch =
        BestEffortReplay::findFirstMismatchedID(new_compute_at_domain_, td);
    TORCH_INTERNAL_ASSERT(
        mismatch == new_compute_at_domain_->nDims(),
        "TensorDomain, ",
        td,
        ", does not match with the previously set domain of ",
        tv_ref_,
        ", which is ",
        new_compute_at_domain_);
  }
  new_compute_at_domain_ = td;
}

namespace {

// Wrapper around set_intersection
template <typename T>
std::set<T> set_intersection(const std::set<T>& set1, const std::set<T>& set2) {
  std::set<T> intersection;
  std::set_intersection(
      set1.begin(),
      set1.end(),
      set2.begin(),
      set2.end(),
      std::inserter(intersection, intersection.begin()));
  return intersection;
}

// convert an iterable of Val* to be an iterable of TensorView*
template <typename T1, typename T2>
T1 tvIterable(const T2& val_iterable) {
  T1 tv_iterable = T1();
  std::transform(
      val_iterable.begin(),
      val_iterable.end(),
      std::back_inserter(tv_iterable),
      [](Val* v) {
        TORCH_INTERNAL_ASSERT(
            v->getValType().value() == ValType::TensorView,
            "When following the computeAt dependency chain, a non TensorView value was found.");
        return v->as<TensorView>();
      });
  return tv_iterable;
}

std::deque<std::deque<TensorView*>> tvChains(
    std::deque<std::deque<Val*>> val_chains) {
  std::deque<std::deque<TensorView*>> tv_chains(val_chains.size());
  for (size_t i = 0; i < val_chains.size(); i++) {
    tv_chains[i] = tvIterable<std::deque<TensorView*>>(val_chains[i]);
  }
  return tv_chains;
}

unsigned int getReplayablePosPasC(
    TensorView* producer,
    TensorView* consumer,
    const ComputeAtRootDomainMap& root_map_) {
  auto mappable_roots =
      root_map_.getMappableDims(producer->domain(), consumer->domain(), true);

  std::cout << "PasC Mappable " << consumer << " && " << producer << std::endl
            << "  ";

  for (auto entry : mappable_roots) {
    std::cout << entry << ", ";
  }
  std::cout << std::endl;

  for (size_t consumer_pos = consumer->nDims(); consumer_pos > 0;
       consumer_pos--) {
    auto root_dim_vals = IterVisitor::getInputsTo(
        {consumer->domain()->domain().begin(),
         consumer->domain()->domain().begin() + consumer_pos});
    auto root_dim = ir_utils::filterByType<IterDomain>(root_dim_vals);
    if (std::any_of(
            root_dim.begin(),
            root_dim.end(),
            [&mappable_roots](IterDomain* root_id) {
              bool not_mappable =
                  mappable_roots.find(root_id) == mappable_roots.end();
              if (not_mappable) {
                std::cout << root_id << " not mappable" << std::endl;
              }
              return not_mappable;
            })) {
      continue;
    }
    return consumer_pos;
  }
  return 0;
}

unsigned int getReplayablePosCasP(
    TensorView* consumer,
    TensorView* producer,
    const ComputeAtRootDomainMap& root_map_) {
  std::cout << "CasP Mappable " << consumer << " && " << producer << std::endl
            << "  ";
  auto mappable_roots =
      root_map_.getMappableDims(producer->domain(), consumer->domain(), false);
  for (auto entry : mappable_roots) {
    std::cout << entry << ", ";
  }
  std::cout << std::endl;

  for (size_t producer_pos = producer->nDims(); producer_pos > 0;
       producer_pos--) {
    std::cout << producer_pos << std::endl;
    auto all_vals = DependencyCheck::getAllValsBetween(
        {producer->getMaybeRFactorDomain().begin(),
         producer->getMaybeRFactorDomain().end()},
        {producer->domain()->domain().begin(),
         producer->domain()->domain().begin() + producer_pos});

    for (auto root_dim : producer->getMaybeRFactorDomain()) {
      if (all_vals.find(root_dim) != all_vals.end()) {
        if (mappable_roots.find(root_dim) == mappable_roots.end()) {
          std::cout << root_dim << " not mappable" << std::endl;
          continue;
        }
      }
    }

    return producer_pos;
  }
  return 0;
}

} // namespace

void ComputeAt::runAt(
    TensorView* producer,
    TensorView* consumer,
    unsigned int consumer_position,
    ComputeAtMode mode) {
  FUSER_PERF_SCOPE("ComputeAt::run");

  // Make sure the correct fusion is setup between this and consumer.
  TORCH_CHECK(
      producer->fusion() == consumer->fusion(),
      producer,
      " and ",
      consumer,
      " are not in the same fusion.");

  // Make sure Fusion Guard is set appropriately
  FusionGuard fg(producer->fusion());
  ComputeAt ca(producer, consumer, consumer, consumer_position, mode);
  ca.runPass();
}

void ComputeAt::runWith(
    TensorView* producer,
    TensorView* consumer,
    unsigned int producer_position,
    ComputeAtMode mode) {
  FUSER_PERF_SCOPE("ComputeAt::runWith");

  // Make sure the correct fusion is setup between this and consumer.
  TORCH_CHECK(
      producer->fusion() == consumer->fusion(),
      producer,
      " and ",
      consumer,
      " are not in the same fusion.");

  // Make sure Fusion Guard is set appropriately
  FusionGuard fg(producer->fusion());

  ComputeAt ca(producer, consumer, producer, producer_position, mode);
  ca.runPass();
}

// Actually applies transformation
unsigned int ComputeAt::backwardComputeAt_impl(
    TensorView* producer,
    TensorView* consumer,
    unsigned int consumer_compute_at_pos) {
  FUSER_PERF_SCOPE("backwardComputeAt_impl");

  auto& producer_entry = tv_data.at(producer);

  std::cout << "\n"
            << consumer << "\n -> " << producer << "\n at "
            << consumer_compute_at_pos << std::endl;
  if (mode_ == ComputeAtMode::BestEffort) {
    consumer_compute_at_pos = std::min(
        consumer_compute_at_pos,
        getReplayablePosPasC(producer, consumer, root_map_));
  } else if (mode_ == ComputeAtMode::MostInlined) {
    consumer_compute_at_pos =
        getReplayablePosPasC(producer, consumer, root_map_);
  }
  std::cout << " Really at " << consumer_compute_at_pos << std::endl;

  auto replay = TransformReplay::replayPasC(
      producer->domain(),
      consumer->domain(),
      (int)consumer_compute_at_pos,
      root_map_);

  producer_entry.setPassPosition(replay.second);

  if (producer_entry.shouldSetComputeAt(replay.second)) {
    const TensorDomain* current_domain = producer->domain();
    TensorDomain* new_domain = replay.first;
    producer->setDomain(new_domain);
    root_map_.setAlias(current_domain, new_domain);
    if (!producer->isFusionInput()) {
      producer->setComputeAt(replay.second);
    }
    producer_entry.setComputeAtDomain(producer->domain());
  }

  return replay.second;
}

// Actually applies transformation, replay consumer based on producer, set
// compute at of producer, set pass position of consumer, return position
// relative to consumer
unsigned int ComputeAt::forwardComputeAt_impl(
    TensorView* producer,
    TensorView* consumer,
    unsigned int producer_compute_at_pos) {
  FUSER_PERF_SCOPE("forwardComputeAt_impl");

  auto& consumer_entry = tv_data.at(consumer);
  const auto& producer_entry = tv_data.at(producer);

  std::cout << consumer << "\n <- " << producer << "\n at "
            << producer_compute_at_pos << std::endl;
  if (mode_ == ComputeAtMode::BestEffort) {
    producer_compute_at_pos = std::min(
        producer_compute_at_pos,
        getReplayablePosCasP(producer, consumer, root_map_));
  } else if (mode_ == ComputeAtMode::MostInlined) {
    producer_compute_at_pos =
        getReplayablePosCasP(producer, consumer, root_map_);
  }
  std::cout << "Really at " << producer_compute_at_pos << std::endl;

  auto replay = TransformReplay::replayCasP(
      consumer->domain(),
      producer->domain(),
      (int)producer_compute_at_pos,
      root_map_);

  if (producer_entry.shouldSetComputeAt(producer_compute_at_pos)) {
    int producer_rel_pos = replay.second;
    int producer_this_pos = (int)producer_compute_at_pos;
    // When the producer CA axes have reductions, they are not used to
    // replay the consumer.
    if (producer_this_pos > producer_rel_pos) {
      producer_this_pos = producer_rel_pos;
    }
    if (!producer->isFusionInput()) {
      producer->setComputeAt(producer_this_pos);
    }
  }

  consumer_entry.setPassPosition(replay.second);
  if (consumer_entry.shouldSetComputeAt(replay.second) &&
      !(consumer == consumer_ && reference_ == consumer_)) {
    const TensorDomain* current_domain = consumer->domain();
    TensorDomain* new_domain = replay.first;
    consumer->setDomain(new_domain);
    root_map_.setAlias(current_domain, new_domain);
    consumer_entry.setComputeAtDomain(consumer->domain());
  }

  return replay.second;
}

void ComputeAt::setCommonConsumer() {
  FUSER_PERF_SCOPE("ComputeAt::setCommonConsumer");

  // Convert the first chain to a set.
  std::set<TensorView*> common_consumers(
      producer_use_chains_.front().begin(), producer_use_chains_.front().end());

  // Run through all use chains of producer, and intersect them to find common
  // TVs
  for (auto tv_chain : producer_use_chains_) {
    common_consumers = set_intersection(
        common_consumers,
        std::set<TensorView*>(tv_chain.begin(), tv_chain.end()));
  }

  auto all_chains =
      tvChains(DependencyCheck::getAllDependencyChains(producer_, consumer_));

  // Right now we only support compute at if at some point in the graph consumer
  // is dependent on producer.
  TORCH_CHECK(
      !all_chains.empty(),
      "Compute At expects ",
      producer_->name(),
      " is a dependency of ",
      consumer_->name(),
      ", however it is not.");

  // Remove all TVs from producer to consumer as common consumer must be at or
  // after consumer
  for (const auto& tv_chain : all_chains) {
    for (auto tv : tv_chain) {
      if (tv != consumer_)
        common_consumers.erase(tv);
    }
  }

  // If there is a common consumer, grab the first one at or after consumer
  common_consumer_ = nullptr;
  if (!common_consumers.empty()) {
    for (auto tv : producer_use_chains_.front()) {
      if (common_consumers.find(tv) != common_consumers.end()) {
        common_consumer_ = tv;
        break;
      }
    }
    TORCH_INTERNAL_ASSERT(
        common_consumer_ != nullptr,
        "Hit a logical inconsistency in the computeAt pass.");
  }
}

// Similar to backward traversal in traverseAllKnown but we should only apply
// computeAt if it will increase computeAt positions.
void ComputeAt::traverseBackward() {
  FUSER_PERF_SCOPE("ComputeAt::traverseBackward");
  if (reference_ == producer_) {
    // Forward compute at don't need to run backward traversal
    return;
  }

  // propagate *backward* through all *producer* use_chains or from *producer*
  // to common_consumer if common_consumer exists. Only apply transform if
  // increases computeAt position.
  auto chains =
      tvChains(DependencyCheck::getAllDependencyChains(producer_, consumer_));

  for (auto tv_chain : chains) {
    TensorView* running_producer = tv_chain.back();
    TensorView* running_consumer = nullptr;
    unsigned int running_consumer_pos = reference_position_;
    tv_chain.pop_back();

    TORCH_INTERNAL_ASSERT(running_producer == consumer_);

    while (!tv_chain.empty()) {
      running_consumer = running_producer;
      running_producer = tv_chain.back();
      tv_chain.pop_back();

      running_consumer_pos = backwardComputeAt_impl(
          running_producer, running_consumer, running_consumer_pos);
    }
  }
}

void ComputeAt::traverseForward() {
  FUSER_PERF_SCOPE("ComputeAt::traverseForward");

  // propagate forward through all *producer* use_chains or from *producer* to
  // common_consumer if common_consumer exists.
  auto chains = producer_use_chains_;
  if (common_consumer_ != nullptr) {
    chains = tvChains(
        DependencyCheck::getAllDependencyChains(producer_, common_consumer_));
  }

  unsigned int producer_pos = reference_ == producer_
      ? reference_position_
      : tv_data.at(producer_).getNewPosition();

  // propagate forward through all chains
  for (auto tv_dep_chain : chains) {
    TensorView* running_producer = nullptr;
    TensorView* running_consumer = tv_dep_chain.front();
    tv_dep_chain.pop_front();
    unsigned int running_producer_pos = producer_pos;

    TORCH_INTERNAL_ASSERT(running_consumer == producer_);

    while (!tv_dep_chain.empty()) {
      running_producer = running_consumer;
      running_consumer = tv_dep_chain.front();
      tv_dep_chain.pop_front();
      running_producer_pos = forwardComputeAt_impl(
          running_producer, running_consumer, running_producer_pos);
    }
  }
}

void ComputeAt::runPass() {
  FUSER_PERF_SCOPE("ComputeAt::runPass");

  // Initialize tv_data for all TensorViews we may modify
  auto chains = producer_use_chains_;
  if (common_consumer_ != nullptr) {
    chains = tvChains(
        DependencyCheck::getAllDependencyChains(producer_, common_consumer_));
  }

  for (const auto& tv_chain : chains) {
    for (auto tv : tv_chain) {
      if (tv_data.find(tv) == tv_data.end()) {
        tv_data[tv] = ComputeAtData(tv);
      }
    }
  }

  // Traverse backward through all dep chains from producer to consumer
  traverseBackward();

  // Clear data from backward traversal:
  for (auto& entry : tv_data) {
    entry.second.clearPass();
  }

  // Start at producer and traverse forward through all chains
  traverseForward();

  setupOutputs();

  for (const auto& entry : tv_data) {
    entry.second.validateNewComputeAt();
  }

  if (reference_ == consumer_) {
    TORCH_INTERNAL_ASSERT(
        BestEffortReplay::findFirstMismatchedID(
            consumer_->domain(), tv_data.at(consumer_).getOriginalDomain()) ==
            (int)consumer_->domain()->nDims(),
        "ComputeAt logic changed the consumer domain which should not happen. Domain was ",
        tv_data.at(consumer_).getOriginalDomain(),
        " but is now: ",
        consumer_->domain());
  }
}

void ComputeAt::setupOutputs() {
  FUSER_PERF_SCOPE("ComputeAt::setupOutputs");

  if (common_consumer_ != nullptr)
    return;

  std::vector<TensorView*> touched_output_order;
  const auto& terminating_outputs =
      FusionGuard::getCurFusion()->getTerminatingOutputs();

  for (auto out : ir_utils::filterByType<TensorView>(
           FusionGuard::getCurFusion()->outputs())) {
    if (tv_data.find(out) != tv_data.end()) {
      if (tv_data[out].touched()) {
        // No need to adjust computeAt when an output is not
        // a terminating output.
        if (std::find(
                terminating_outputs.begin(), terminating_outputs.end(), out) !=
            terminating_outputs.end()) {
          touched_output_order.push_back(out);
        }
      }
    }
  }

  if (touched_output_order.size() > 0) {
    for (size_t i = 0; i < touched_output_order.size() - 1; i++) {
      touched_output_order[i]->setComputeAt(
          (int)tv_data.at(touched_output_order[i]).getNewPosition());
    }
  }
}

ComputeAt::ComputeAt(
    TensorView* _producer,
    TensorView* _consumer,
    TensorView* _reference,
    unsigned int _reference_position,
    ComputeAtMode _mode)
    : producer_(_producer),
      consumer_(_consumer),
      reference_(_reference),
      reference_position_(_reference_position),
      mode_(_mode) {
  TORCH_INTERNAL_ASSERT(
      reference_ == producer_ || reference_ == consumer_,
      "For compute at reference must be producer or consumer, it's neither.",
      " reference: ",
      reference_,
      " consumer: ",
      consumer_,
      " producer: ",
      producer_);
  TORCH_INTERNAL_ASSERT(
      reference_position_ >= 0 && reference_position_ <= reference_->nDims(),
      "Invalid computeAt axis, received ",
      reference_position_,
      " but should be > -",
      reference_->nDims(),
      " and <= ",
      reference_->nDims(),
      ".");

  producer_use_chains_ = tvChains(DependencyCheck::getAllUseChains(producer_));

  // Look through all the use chains of producer. Check if there's a single
  // consumer for all chains at or after the consumer specified in the computeAt
  // call.
  setCommonConsumer();

  root_map_.build();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
