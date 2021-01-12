#include <torch/csrc/jit/codegen/cuda/segment.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

std::vector<SegmentedGroup*> SegmentedGroup::getNeighbors() {
  std::vector<SegmentedGroup*> neighbors;
  for (auto inp : producer_edges) {
    neighbors.push_back(inp->from_);
  }
  for (auto out : consumer_edges) {
    neighbors.push_back(out->to_);
  }
  return neighbors;
}

std::vector<SegmentedGroup*> SegmentedGroup::getMergeCandidates() {
  std::vector<SegmentedGroup*> neighbors = getNeighbors();

  // Don't look for candidates if already merged
  if (merged) {
    return {};
  }

  // Can this node be merged with another? Check if neighbors are merged, if
  // so and merged neighbor is within 1 level or node merged with neighbor is
  // within 1 level, can't merge this node with anything else.
  bool can_merge_this = true;
  for (size_t i = 0; i < neighbors.size(); i++) {
    if (!neighbors[i]->merged) {
      continue;
    }
    if (std::abs(neighbors[i]->level - level) <= 1) {
      can_merge_this = false;
    }
    if (std::abs(neighbors[i]->merge_with->level - level) <= 1) {
      can_merge_this = false;
    }
  }
  if (!can_merge_this) {
    return {};
  }

  std::vector<bool> can_merge(true, neighbors.size());

  // Find neighbors with a level that is only 1 differant than this groups level
  for (size_t i = 0; i < neighbors.size(); i++) {
    if (std::abs(neighbors[i]->level - level) > 1) {
      can_merge[i] = false;
    }
  }

  // Check neighbor of neighbors we're considering, if any of them are merged
  // with another node, make sure the resulting edge wouldn't have a level
  // difference of 1
  for (size_t i = 0; i < neighbors.size(); i++) {
    if (!can_merge[i]) {
      continue;
    }

    for (auto neighbor_neighbor : neighbors[i]->getNeighbors()) {
      // Don't check self
      if (neighbor_neighbor == neighbors[i]) {
        continue;
      }
      if (neighbor_neighbor->merged) {
        // check neighbor_neighbor level
        if (std::abs(neighbor_neighbor->level - level) <= 1) {
          can_merge[i] = false;
        }
        if (std::abs(neighbor_neighbor->level - neighbors[i]->level) <= 1) {
          can_merge[i] = false;
        }

        // check neighbor_neighber->merged->level
        if (std::abs(neighbor_neighbor->merge_with->level - level) <= 1) {
          can_merge[i] = false;
        }
        if (std::abs(
                neighbor_neighbor->merge_with->level - neighbors[i]->level) <=
            1) {
          can_merge[i] = false;
        }
      }
    }
  }

  std::vector<SegmentedGroup*> merge_candidates;
  for (size_t i = 0; i < neighbors.size(); i++) {
    if (can_merge[i]) {
      merge_candidates.push_back(neighbors[i]);
    }
  }
  return merge_candidates;
}

void SegmentedGroup::clearTraversalInfo() {
  level = -1;
  visited = false;
  merge_with = nullptr;
  merged = false;
}

std::ostream& operator<<(std::ostream& os, const SegmentedGroup* group) {
  os << "g{";
  for (size_t i = 0; i < group->exprs_.size(); i++) {
    os << group->exprs_[i]->name();
    if (i + 1 != group->exprs_.size())
      os << ", ";
  }
  os << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const SegmentedEdge* edge) {
  os << "e{ " << edge->from_ << " -> " << edge->to_ << " }" << std::endl;
  return os;
}

void SegmentCandidateFinder::resetTraversal() {
  for (auto& group : groups) {
    // Start traversal at input groups
    if (group.producer_edges.empty()) {
      to_visit.push_back(&group);
    }
    group.visited = false;
    group.level = 0;
  }
}

void SegmentCandidateFinder::resetLevels() {
  while (!to_visit.empty()) {
    auto visit = to_visit.front();
    to_visit.pop_front();

    // All inputs processed?
    bool ready = true;
    if (!visit->producer_edges.empty()) {
      ready = std::all_of(
          visit->producer_edges.begin(),
          visit->producer_edges.end(),
          [&](SegmentedEdge* dep) { return dep->from_->visited; });
    }

    if (!ready) {
      // In case traversal doesn't complete because there's an error in the
      // DAG topology.
      next_to_visit.push_back(visit);
      continue;
    }

    visit->visited = true;

    to_visit.insert(to_visit.end(), next_to_visit.begin(), next_to_visit.end());
    next_to_visit.clear();

    for (auto out : visit->consumer_edges) {
      to_visit.push_back(out->to_);
    }

    visit->level = 0;
    for (auto inp : visit->producer_edges) {
      visit->level = std::max(visit->level, inp->from_->level + 1);
    }
  }
  TORCH_INTERNAL_ASSERT(next_to_visit.empty(), "Error in graph, is not a DAG.");
}

std::string SegmentCandidateFinder::toString(int verbosity) const {
  std::stringstream ss;
  for (auto group : groups) {
  }
  for (auto group : groups) {
    ss << &group << "\n";

    if (verbosity > 1) {
      if (group.producer_edges.size() > 0) {
        ss << "  produced by groups: { \n";
        for (auto producer_edge : group.producer_edges) {
          ss << "    " << producer_edge->from_ << " via " << producer_edge->val_
             << "\n";
        }
        ss << "  }"
           << "\n";
      }
    }
    if (verbosity > 0) {
      if (group.consumer_edges.size() > 0) {
        ss << "  Consumed by groups: { \n";
        for (auto consumer_edge : group.consumer_edges) {
          ss << "    " << consumer_edge->to_ << "\n";
        }
        ss << "  }"
           << "\n";
      }
    }
  }
  return ss.str();
}

namespace {

std::vector<Val*> uniqueValConcat(
    const std::vector<std::vector<Val*>>& val_vecs) {
  std::vector<Val*> unique_vals;
  std::unordered_set<Val*> added;
  for (auto vec : val_vecs) {
    for (auto val : vec) {
      if (added.find(val) == added.end()) {
        unique_vals.push_back(val);
        added.emplace(val);
      }
    }
  }
  return unique_vals;
}

// Concat's producer edges of sg1 and sg2, but removes any edges from/to sg1/sg2
std::vector<SegmentedEdge*> getMergedProducerEdges(
    const SegmentedGroup* sg1,
    const SegmentedGroup* sg2) {
  TORCH_INTERNAL_ASSERT(
      sg1 != nullptr && sg2 != nullptr,
      "This function doesn't handle trivial.");

  auto producer_edges = sg1->producer_edges;
  producer_edges.insert(
      producer_edges.end(),
      sg2->producer_edges.begin(),
      sg2->producer_edges.end());

  producer_edges.erase(
      std::remove_if(
          producer_edges.begin(),
          producer_edges.end(),
          [&sg1, &sg2](SegmentedEdge* se) {
            return (se->to_ == sg1 && se->from_ == sg2) ||
                (se->to_ == sg2 && se->from_ == sg1);
          }),
      producer_edges.end());

  return producer_edges;
}

// Concat's consumer edges of sg1 and sg2, but removes any edges from/to sg1/sg2
std::vector<SegmentedEdge*> getMergedConsumerEdges(
    const SegmentedGroup* sg1,
    const SegmentedGroup* sg2) {
  TORCH_INTERNAL_ASSERT(
      sg1 != nullptr && sg2 != nullptr,
      "This function doesn't handle trivial.");

  auto consumer_edges = sg1->consumer_edges;
  consumer_edges.insert(
      consumer_edges.end(),
      sg2->consumer_edges.begin(),
      sg2->consumer_edges.end());

  consumer_edges.erase(
      std::remove_if(
          consumer_edges.begin(),
          consumer_edges.end(),
          [&sg1, &sg2](SegmentedEdge* se) {
            return (se->to_ == sg1 && se->from_ == sg2) ||
                (se->to_ == sg2 && se->from_ == sg1);
          }),
      consumer_edges.end());

  return consumer_edges;
}

// Returns a determinstic, unique set of inputs of the segment group, sg1, or
// the combined group sg1 + sg2
std::vector<Val*> getAllInputs(
    const SegmentedGroup* sg1,
    const SegmentedGroup* sg2 = nullptr) {
  std::vector<SegmentedEdge*> merged_producer_edges;

  if (sg1 != nullptr && sg2 != nullptr) {
    merged_producer_edges = getMergedProducerEdges(sg1, sg2);
  } else if (sg1 != nullptr) {
    merged_producer_edges = sg1->producer_edges;
  } else if (sg2 != nullptr) {
    merged_producer_edges = sg2->producer_edges;
  }

  std::vector<Val*> producer_edge_vals;

  std::transform(
      merged_producer_edges.begin(),
      merged_producer_edges.end(),
      std::back_inserter(producer_edge_vals),
      [](SegmentedEdge* se) { return se->val_; });

  return uniqueValConcat(
      {sg1 == nullptr ? std::vector<Val*>() : sg1->input_vals,
       sg2 == nullptr ? std::vector<Val*>() : sg2->input_vals,
       producer_edge_vals});
}

// Returns a determinstic, unique set of outputs of the segment group, sg1, or
// the combined group sg1 + sg2
std::vector<Val*> getAllOutputs(
    const SegmentedGroup* sg1,
    const SegmentedGroup* sg2 = nullptr) {
  std::vector<SegmentedEdge*> merged_consumer_edges;

  if (sg1 != nullptr && sg2 != nullptr) {
    merged_consumer_edges = getMergedConsumerEdges(sg1, sg2);
  } else if (sg1 != nullptr) {
    merged_consumer_edges = sg1->consumer_edges;
  } else if (sg2 != nullptr) {
    merged_consumer_edges = sg2->consumer_edges;
  }

  std::vector<Val*> consumer_edge_vals;

  std::transform(
      merged_consumer_edges.begin(),
      merged_consumer_edges.end(),
      std::back_inserter(consumer_edge_vals),
      [](SegmentedEdge* se) { return se->val_; });

  auto output_vals =
      uniqueValConcat({sg1 == nullptr ? std::vector<Val*>() : sg1->output_vals,
                       sg2 == nullptr ? std::vector<Val*>() : sg2->output_vals,
                       consumer_edge_vals});

  return output_vals;
}

} // namespace

// Disconect group from neighbors, and return edges that were disconnected
std::unordered_set<SegmentedEdge*> SegmentCandidateFinder::disconnectGroup(
    SegmentedGroup* group) {
  std::unordered_set<SegmentedEdge*> removed_edges(
      group->producer_edges.begin(), group->producer_edges.end());

  for (auto edge : group->producer_edges) {
    auto from = edge->from_;
    auto& from_edges = from->consumer_edges;
    auto from_edge_it = std::find(from_edges.begin(), from_edges.end(), edge);
    TORCH_INTERNAL_ASSERT(
        from_edge_it != from_edges.end(), "Could not find edge to remove.");
    from_edges.erase(from_edge_it);
  }

  for (auto edge : group->consumer_edges) {
    auto to = edge->to_;
    auto& to_edges = to->producer_edges;
    auto to_edge_it = std::find(to_edges.begin(), to_edges.end(), edge);
    TORCH_INTERNAL_ASSERT(
        to_edge_it != to_edges.end(), "Could not find edge to remove.");
    to_edges.erase(to_edge_it);
  }

  group->producer_edges.clear();
  group->consumer_edges.clear();

  return removed_edges;
}

void SegmentCandidateFinder::mergeNodes() {
  while (!to_merge.empty()) {
    auto group1 = *to_merge.begin();
    auto group2 = group1->merge_with;
    to_merge.erase(group1);
    to_merge.erase(group2);
    clean_up_groups.emplace(group1);
    clean_up_groups.emplace(group2);

    // Make the new joined node
    groups.push_back(SegmentedGroup());
    auto& joined_group = groups.back();

    joined_group.input_vals =
        uniqueValConcat({group1->input_vals, group2->input_vals});

    joined_group.output_vals =
        uniqueValConcat({group1->output_vals, group2->output_vals});

    joined_group.exprs_ = group1->exprs_;
    joined_group.exprs_.insert(
        joined_group.exprs_.end(),
        group2->exprs_.begin(),
        group2->exprs_.end());

    auto producer_edges = getMergedProducerEdges(group1, group2);
    // Connect joined group to resulting neighbors
    for (auto edge : producer_edges) {
      auto from = edge->from_;
      auto val = edge->val_;

      edges.push_back(SegmentedEdge(from, &joined_group, val));

      joined_group.producer_edges.push_back(&edges.back());
      from->consumer_edges.push_back(&edges.back());
    }

    auto consumer_edges = getMergedConsumerEdges(group1, group2);

    for (auto edge : consumer_edges) {
      auto to = edge->to_;
      auto val = edge->val_;

      edges.push_back(SegmentedEdge(&joined_group, to, val));
      joined_group.consumer_edges.push_back(&edges.back());
      edge->to_->producer_edges.push_back(&edges.back());
    }
  }

  for (auto group : clean_up_groups) {
    auto disconnected_edges = disconnectGroup(group);
    clean_up_edges.insert(disconnected_edges.begin(), disconnected_edges.end());
  }

  edges.remove_if([this](SegmentedEdge& edge) {
    return this->clean_up_edges.find(&edge) != this->clean_up_edges.end();
  });

  groups.remove_if([this](SegmentedGroup& group) {
    return this->clean_up_groups.find(&group) != this->clean_up_groups.end();
  });

  clean_up_edges.clear();
  clean_up_groups.clear();
}

std::unique_ptr<Fusion> SegmentCandidateFinder::makeFusion(SegmentedGroup* sg) {
  std::unique_ptr<Fusion> segmented_fusion = std::make_unique<Fusion>();

  auto complete_to_segment_map =
      Fusion::copy(&complete_fusion, segmented_fusion.get());

  for (auto inp : segmented_fusion->inputs()) {
    segmented_fusion->removeInput(inp);
  }
  for (auto out : segmented_fusion->outputs()) {
    segmented_fusion->removeOutput(out);
  }

  for (auto inp : getAllInputs(sg)) {
    segmented_fusion->addInput(complete_to_segment_map.clone(inp));
  }

  for (auto out : getAllOutputs(sg)) {
    segmented_fusion->addOutput(complete_to_segment_map.clone(out));
  }

  return segmented_fusion;
}

namespace {

// Guard to temporarily change the inputs and outputs of a fusion. On
// destruction will return fusion to original state.
class FusionSegmentGuard {
 public:
  FusionSegmentGuard() = delete;
  FusionSegmentGuard(FusionSegmentGuard const&) = delete;
  FusionSegmentGuard& operator=(FusionSegmentGuard const&) = delete;

  FusionSegmentGuard(
      Fusion* fusion,
      const std::vector<Val*>& inputs,
      const std::vector<Val*>& outputs)
      : fusion_(fusion),
        old_inputs(fusion->inputs()),
        old_outputs(fusion->outputs()),
        new_inputs(inputs),
        new_outputs(outputs) {
    for (auto old_inp : old_inputs) {
      fusion_->removeInput(old_inp);
    }

    for (auto old_out : old_outputs) {
      fusion_->removeOutput(old_out);
    }

    for (auto new_inp : new_inputs) {
      fusion_->addInput(new_inp);
    }

    for (auto new_out : new_outputs) {
      fusion_->addOutput(new_out);
    }
  }

  ~FusionSegmentGuard() {
    for (auto new_inp : new_inputs) {
      fusion_->removeInput(new_inp);
    }

    for (auto new_out : new_outputs) {
      fusion_->removeOutput(new_out);
    }

    for (auto old_inp : old_inputs) {
      fusion_->addInput(old_inp);
    }

    for (auto old_out : old_outputs) {
      fusion_->addOutput(old_out);
    }
  }

 private:
  Fusion* const fusion_;
  const std::vector<Val*> old_inputs;
  const std::vector<Val*> old_outputs;
  const std::vector<Val*> new_inputs;
  const std::vector<Val*> new_outputs;
};

} // namespace

bool SegmentCandidateFinder::codeGenSupportedMerge(
    SegmentedGroup* sg1,
    SegmentedGroup* sg2) {
  FusionSegmentGuard fsg(
      &complete_fusion, getAllInputs(sg1, sg2), getAllOutputs(sg1, sg2));

  bool can_gen = this->canGenerateCode(&complete_fusion);

  return can_gen;
}

SegmentCandidateFinder::SegmentCandidateFinder(const Fusion* fusion)
    : complete_fusion(*fusion) {}

void SegmentCandidateFinder::segment() {
  // TODO: Make traversal items local to this function.

  // Need this for initialization of the DAG that is process
  std::unordered_map<Expr*, SegmentedGroup*> expr2group;

  // Initialize DAG, convert each expr to a segment group
  for (auto expr : complete_fusion.exprs()) {
    groups.push_back(SegmentedGroup(expr));
    expr2group.insert(std::make_pair(expr, &groups.back()));
  }

  // Create edges between the Exprs. Mark inputs and outputs of the fusion.
  for (auto expr : complete_fusion.exprs()) {
    auto expr_group = expr2group.at(expr);
    for (auto inp : expr->inputs()) {
      if (inp->isFusionInput()) {
        expr_group->input_vals.push_back(inp);
        continue;
      }

      // Could be something like a constant scalar, definition is nullptr, but
      // isn't an "input" to the fusion. At least not one provided by an
      // external source.
      if (inp->definition() == nullptr) {
        continue;
      }

      auto def_group = expr2group.at(inp->definition());
      edges.push_back(SegmentedEdge(def_group, expr_group, inp));
      expr_group->producer_edges.push_back(&edges.back());
      def_group->consumer_edges.push_back(&edges.back());
    }
    for (auto out : expr->outputs()) {
      if (out->isFusionOutput()) {
        expr_group->output_vals.push_back(out);
      }
    }
  }

  bool merged_nodes = true;
  while (merged_nodes) {
    // Reset stateful traversal details in SegmentedGroups
    resetTraversal();

    resetLevels();

    for (auto& group : groups) {
      if (group.merged) {
        continue;
      }
      auto candidates = group.getMergeCandidates();
      if (candidates.empty()) {
        continue;
      }

      auto candidate_it = candidates.begin();
      while (candidate_it != candidates.end() &&
             !codeGenSupportedMerge(&group, *candidate_it)) {
        candidate_it++;
      }
      if (candidate_it == candidates.end()) {
        continue;
      }

      to_merge.emplace(&group);
      to_merge.emplace(candidates[0]);

      group.merged = true;
      group.merge_with = candidates[0];

      candidates[0]->merged = true;
      candidates[0]->merge_with = &group;
    }

    if (to_merge.empty()) {
      merged_nodes = false;
    }

    mergeNodes();
  }
}

bool SingleReductionSegmenter::canGenerateCode(Fusion* fusion) {
  bool has_reduction = false;
  for (auto expr : fusion->exprs()) {
    if (expr->getExprType().value() == ExprType::ReductionOp) {
      if (has_reduction) {
        return false;
      }
      has_reduction = true;
    }
  }
  return true;
}

void SingleReductionSegmenter::generateFusions() {
  TORCH_INTERNAL_ASSERT(
      groups.size() > 1,
      "Didn't do any segmentation. Don't support trivial segmentation.");

  for (auto group : groups) {
    fusion_executors.push_back(
        std::make_unique<FusionExecutorCache>(makeFusion(&group)));
  }
}

std::vector<at::Tensor> SingleReductionSegmenter::runFusionWithInputs(
    const at::ArrayRef<IValue>& fusion_runtime_inputs) {
  TORCH_INTERNAL_ASSERT(
      fusion_runtime_inputs.size() == complete_fusion.inputs().size(),
      "Inputs were not set up correctly, recieved ",
      fusion_runtime_inputs.size(),
      " inputs but expecting ",
      complete_fusion.inputs().size());

  std::unordered_map<Val*, IValue> tensor_map;
  for (size_t i = 0; i < fusion_runtime_inputs.size(); i++) {
    tensor_map.emplace(
        std::make_pair(complete_fusion.inputs()[i], fusion_runtime_inputs[i]));
  }

  std::vector<bool> group_ran(groups.size(), false);

  while (!std::all_of(
      group_ran.begin(), group_ran.end(), [](bool b) { return b; })) {
    bool one_ran = false;
    auto group_it = groups.begin();
    for (size_t group_i = 0; group_i < groups.size(); group_i++, group_it++) {
      auto& group = *group_it;
      if (group_ran[group_i]) {
        continue;
      }
      auto group_inputs = getAllInputs(&group);
      bool ready_to_run = std::all_of(
          group_inputs.begin(), group_inputs.end(), [&tensor_map](Val* val) {
            return tensor_map.find(val) != tensor_map.end();
          });

      if (ready_to_run) {
        std::vector<IValue> group_runtime_inputs;
        for (auto input : group_inputs) {
          group_runtime_inputs.push_back(tensor_map.at(input));
        }
        std::cout << "Running:" << std::endl;
        fusion_executors[group_i]->printFusion();
        auto group_runtime_outputs =
            fusion_executors[group_i]->runFusionWithInputs(
                group_runtime_inputs);

        auto group_outputs = getAllOutputs(&group);

        for (size_t group_out_i = 0; group_out_i < group_outputs.size();
             group_out_i++) {
          tensor_map.emplace(std::make_pair(
              group_outputs[group_out_i], group_runtime_outputs[group_out_i]));
        }
        group_ran[group_i] = true;
        one_ran = true;
      }
    }
    TORCH_INTERNAL_ASSERT(
        one_ran,
        "Couldn't run all groups, something must have gone wrong in segmentation.");
  }

  std::vector<IValue> fusion_outputs;
  for (auto output : complete_fusion.outputs()) {
    fusion_outputs.push_back(tensor_map.at(output));
  }

  std::vector<at::Tensor> fusion_output_tensors;
  std::transform(
      fusion_outputs.begin(),
      fusion_outputs.end(),
      std::back_inserter(fusion_output_tensors),
      [](IValue ival) {
        TORCH_INTERNAL_ASSERT(
            ival.isTensor(), "Cannot output non-tensor objects from a fusion.");
        return ival.toTensor();
      });

  return fusion_output_tensors;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch