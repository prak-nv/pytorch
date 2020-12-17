#include <torch/csrc/jit/codegen/cuda/segment.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <list>
#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

std::deque<SegmentedGroup*> SegmentedGroup::getNeighbors() {
  std::deque<SegmentedGroup*> neighbors;
  for (auto inp : producer_edges) {
    neighbors.push_back(inp->from_);
  }
  for (auto out : consumer_edges) {
    neighbors.push_back(out->to_);
  }
  return neighbors;
}

std::deque<SegmentedGroup*> SegmentedGroup::getMergeCandidates() {
  std::deque<SegmentedGroup*> neighbors = getNeighbors();

  // Can this node be merged with another? Check if neighbors are merged, if
  // so and merged neighbor, or node merged with neighbor is within 1 level,
  // can't merge this node
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

  std::deque<bool> can_merge(true, neighbors.size());

  // Find neighbors who level is only 1 differant than this groups level
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

  std::deque<SegmentedGroup*> merge_candidates;
  for (size_t i = 0; i < neighbors.size(); i++) {
    if (can_merge[i]) {
      merge_candidates.push_back(neighbors[i]);
    }
  }
  return merge_candidates;
}

void SegmentedGroup::clearTraversalInfo() {
  is_input = false;
  level = -1;
  visited = false;
  merge_with = nullptr;
  merged = false;
}

std::ostream& operator<<(std::ostream& os, SegmentedGroup* group) {
  os << "g{";
  for (size_t i = 0; i < group->exprs_.size(); i++) {
    os << group->exprs_[i]->name();
    if (i + 1 != group->exprs_.size())
      os << ", ";
  }
  os << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, SegmentedEdge* edge) {
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
}

std::string SegmentCandidateFinder::toString() {
  std::stringstream ss;
  for (auto group : groups) {
  }
  for (auto group : groups) {
    ss << &group << "\n";

    if (group.consumer_edges.size() > 0) {
      ss << "  Consumed by groups: { \n";
      for (auto consumer_edge : group.consumer_edges) {
        ss << "    " << consumer_edge->to_ << "\n";
      }
      ss << "  }"
         << "\n";
    }
  }
  return ss.str();
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
    for (auto expr : group1->exprs_) {
      joined_group.exprs_.push_back(expr);
    }

    for (auto expr : group2->exprs_) {
      joined_group.exprs_.push_back(expr);
    }

    // Reconnect producer edges to the rest of the graph
    auto producer_edges = group1->producer_edges;
    producer_edges.insert(
        producer_edges.end(),
        group2->producer_edges.begin(),
        group2->producer_edges.end());

    for (auto edge : producer_edges) {
      auto from = edge->from_;
      auto val = edge->val_;

      if (edge->from_ == group1 && edge->to_ == group2) {
        continue;
      }
      if (edge->to_ == group1 && edge->from_ == group2) {
        continue;
      }

      auto& from_edges = from->consumer_edges;
      auto from_edge_it = std::find(from_edges.begin(), from_edges.end(), edge);

      edges.push_back(SegmentedEdge(from, &joined_group, val));

      joined_group.producer_edges.push_back(&edges.back());
      from_edges.push_back(&edges.back());

      from_edges.erase(from_edge_it);
      clean_up_edges.emplace(edge);
    }

    // Reconnect consumer edges to the rest of the graph
    auto consumer_edges = group1->consumer_edges;
    consumer_edges.insert(
        consumer_edges.end(),
        group2->consumer_edges.begin(),
        group2->consumer_edges.end());

    for (auto edge : consumer_edges) {
      if (edge->from_ == group1 && edge->to_ == group2) {
        continue;
      }
      if (edge->to_ == group1 && edge->from_ == group2) {
        continue;
      }

      auto to = edge->to_;
      auto val = edge->val_;
      auto& to_edges = edge->to_->producer_edges;
      auto to_edge_it = std::find(to_edges.begin(), to_edges.end(), edge);

      edges.push_back(SegmentedEdge(&joined_group, to, val));
      joined_group.consumer_edges.push_back(&edges.back());
      to_edges.push_back(&edges.back());

      to_edges.erase(to_edge_it);
      clean_up_edges.emplace(edge);
    }

    std::cout << "Group: " << &joined_group << std::endl;
    edges.remove_if([this](SegmentedEdge& edge) {
      bool found =
          this->clean_up_edges.find(&edge) != this->clean_up_edges.end();
      this->clean_up_edges.erase(&edge);
      return found;
    });

    groups.remove_if([this](SegmentedGroup& group) {
      bool found =
          this->clean_up_groups.find(&group) != this->clean_up_groups.end();
      this->clean_up_groups.erase(&group);
      return found;
    });
  }
}

bool SegmentCandidateFinder::codeGenSupportedMerge(
    SegmentedGroup* sg1,
    SegmentedGroup* sg2) {
  std::vector<Val*> old_inputs = fusion_.inputs();
  std::vector<Val*> old_outputs = fusion_.outputs();
  std::deque<Expr*> exprs = sg1->exprs_;
  // We will want to retraverse all values, make sure that traversal is
  // deterministic from one run to the next
  std::vector<Val*> deterministic_vals;
  exprs.insert(exprs.end(), sg2->exprs_.begin(), sg2->exprs_.end());
  std::unordered_set<Val*> used;
  std::unordered_set<Val*> produced;
  for (auto expr : exprs) {
    for (auto inp : expr->inputs()) {
      used.emplace(inp);
      deterministic_vals.push_back(inp);
    }
    for (auto out : expr->outputs()) {
      produced.emplace(out);
      deterministic_vals.push_back(out);
    }
  }

  // By property of how segmentation is done, if there is a use of a Val in
  // the fusion group, it is only used in the fusion group

  std::vector<Val*> new_inputs;
  std::vector<Val*> new_outputs;
  for (auto val : deterministic_vals) {
    if (used.find(val) == used.end()) {
      // Only produced, must be output
      new_outputs.push_back(val);
    }

    if (produced.find(val) == produced.end()) {
      // Only used, must be input
      new_inputs.push_back(val);
    }
  }

  for (auto old_inp : old_inputs) {
    fusion_.removeInput(old_inp);
  }

  for (auto old_out : old_outputs) {
    fusion_.removeOutput(old_out);
  }

  for (auto new_inp : new_inputs) {
    fusion_.addInput(new_inp);
  }

  for (auto new_out : new_outputs) {
    fusion_.addOutput(new_out);
  }

  bool can_gen = this->canGenerateCode(&fusion_);
  // if(can_gen){
  //   // std::cout<<"Can generate the fusion:\n";
  //   // fusion_.printMath();
  // }

  for (auto new_inp : new_inputs) {
    fusion_.removeInput(new_inp);
  }

  for (auto new_out : new_outputs) {
    fusion_.removeOutput(new_out);
  }

  for (auto old_inp : old_inputs) {
    fusion_.addInput(old_inp);
  }

  for (auto old_out : old_outputs) {
    fusion_.addOutput(old_out);
  }

  return can_gen;
}

SegmentCandidateFinder::SegmentCandidateFinder(const Fusion* fusion)
    : fusion_(*fusion) {
  // Need this for initialization of the DAG we'll process

  std::unordered_map<Expr*, SegmentedGroup*> expr2group;
  for (auto expr : fusion_.exprs()) {
    groups.push_back(SegmentedGroup(expr));
    expr2group.insert(std::make_pair(expr, &groups.back()));
  }

  for (auto expr : fusion_.exprs()) {
    for (auto inp : expr->inputs()) {
      if (inp->definition() == nullptr) {
        continue;
      }
      auto def_group = expr2group.at(inp->definition());
      auto expr_group = expr2group.at(expr);
      edges.push_back(SegmentedEdge(def_group, expr_group, inp));
      def_group->consumer_edges.push_back(&edges.back());
      expr_group->producer_edges.push_back(&edges.back());
    }
  }
}

void SegmentCandidateFinder::segment() {
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

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch