#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

std::deque<TensorView*> deduplicate(const std::deque<TensorView*>& tv_deuqe) {
  std::deque<TensorView*> deduplicated;
  std::unordered_set<TensorView*> inserted;
  for (auto tv_entry : tv_deuqe) {
    if (inserted.find(tv_entry) == inserted.end()) {
      deduplicated.emplace_back(tv_entry);
      inserted.emplace(tv_entry);
    }
  }
  return deduplicated;
}
}; // namespace

class TransformPropagator {
  std::deque<TensorView*> tvInputs(Expr* expr) {
    auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    return std::deque<TensorView*>(tv_inputs.begin(), tv_inputs.end());
  }

  std::deque<TensorView*> tvOutputs(Expr* expr) {
    auto tv_outputs = ir_utils::filterByType<TensorView>(expr->outputs());
    return std::deque<TensorView*>(tv_outputs.begin(), tv_outputs.end());
  }

  std::deque<TensorView*> consumersOf(TensorView* tv) {
    std::deque<TensorView*> consumer_tvs;
    for (auto def : tv->uses()) {
      auto outs = tvOutputs(def);
      consumer_tvs.insert(consumer_tvs.end(), outs.begin(), outs.end());
    }
    return deduplicate(consumer_tvs);
  }

  std::deque<TensorView*> producersFor(TensorView* tv) {
    auto def = tv->definition();
    if (def == nullptr) {
      return {};
    }

    return deduplicate(tvInputs(def));
  }

  bool replayPasC(TensorView* producer_tv, TensorView* consumer_tv = nullptr) {
    if (producer_tv == starting_tv) {
      return false;
    }
    auto pairwiseMap = PairwiseRootDomainMap(producer_tv, consumer_tv);
    auto producerAsC = TransformReplay::replayPasC(
        producer_tv->domain(), consumer_tv->domain(), -1, pairwiseMap);

    if (replayed_pos.find(producer_tv) != replayed_pos.end()) {
      if (producerAsC.second <= replayed_pos.at(producer_tv)) {
        return false;
      }
    }

    // std::cout<<"Replayed: "<<producer_tv<<" as "<<consumer_tv<<std::endl;
    producer_tv->setDomain(producerAsC.first);
    replayed_pos[producer_tv] = producerAsC.second;
    // std::cout<<"  "<<producer_tv<<std::endl;

    return true;
  }

  bool replayCasP(TensorView* consumer_tv, TensorView* producer_tv = nullptr) {
    if (consumer_tv == starting_tv) {
      return false;
    }
    auto pairwiseMap = PairwiseRootDomainMap(producer_tv, consumer_tv);
    auto consumerAsP = TransformReplay::replayCasP(
        consumer_tv->domain(), producer_tv->domain(), -1, pairwiseMap);

    if (replayed_pos.find(consumer_tv) != replayed_pos.end()) {
      if (consumerAsP.second <= replayed_pos.at(consumer_tv)) {
        return false;
      }
    }

    // std::cout<<"Replayed: "<<consumer_tv<<" as "<<producer_tv<<std::endl;
    consumer_tv->setDomain(consumerAsP.first);
    replayed_pos[consumer_tv] = consumerAsP.second;
    // std::cout<<"  "<<consumer_tv<<std::endl;

    return true;
  }

  // These should only exist in the scope of from
  std::unordered_map<TensorView*, unsigned int> replayed_pos;
  TensorView* starting_tv = nullptr;

 public:
  void from(TensorView* tv) {
    starting_tv = tv;

    // Tensors we should try to propagate in the consumer direction
    std::deque<TensorView*> consumer_propagation{tv};

    // Tensors we should try to propagate in the producer direction
    std::deque<TensorView*> producer_propagation{tv};

    // While tensor views are being replayed, if they're modified, make sure we
    // propagate back to all producers as well as consumers. This is definitely
    // not the most efficient implementation as what we do is any time a tv is
    // changed we propagate both forward and backward. If a forward pass touches
    // every node, the backward pass will try to replay every node, potentially
    // multiple times.
    while (!consumer_propagation.empty() || !producer_propagation.empty()) {
      while (!consumer_propagation.empty()) {
        // Tensor view we will replay onto consumers
        auto tv = consumer_propagation.front();
        consumer_propagation.pop_front();

        // Replay tv forward to its consumers.
        for (auto consumer_tv : consumersOf(tv)) {
          auto replayed = replayCasP(consumer_tv, tv);
          // If consumer has changed, mark we should propagate its consumers
          if (replayed) {
            consumer_propagation.emplace_back(consumer_tv);
            producer_propagation.emplace_back(consumer_tv);
          }
        }
      }

      while (!producer_propagation.empty()) {
        // Tensor view we will replay onto producers
        auto tv = producer_propagation.front();
        producer_propagation.pop_front();

        // Replay tv backward to its producers
        for (auto producer_tv : producersFor(tv)) {
          auto replayed = replayPasC(producer_tv, tv);
          if (replayed) {
            producer_propagation.emplace_back(producer_tv);
            consumer_propagation.emplace_back(producer_tv);
          }
        }
      }
    }
  }
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch