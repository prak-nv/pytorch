#include <torch/csrc/jit/codegen/cuda/segmenter_helper.h>
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
  if (payload()->merged) {
    return {};
  }

  // Can this node be merged with another? Check if neighbors are merged, if
  // so and merged neighbor is within 1 level or node merged with neighbor is
  // within 1 level, can't merge this node with anything else.
  bool can_merge_this = true;
  for (size_t i = 0; i < neighbors.size(); i++) {
    if (!neighbors[i]->payload()->merged) {
      continue;
    }
    if (std::abs(neighbors[i]->payload()->level - payload()->level) <= 1) {
      can_merge_this = false;
    }
    if (std::abs(
            neighbors[i]->payload()->merge_with->payload()->level -
            payload()->level) <= 1) {
      can_merge_this = false;
    }
  }
  if (!can_merge_this) {
    return {};
  }

  std::vector<bool> can_merge(true, neighbors.size());

  // Find neighbors with a level that is only 1 differant than this groups level
  for (size_t i = 0; i < neighbors.size(); i++) {
    if (std::abs(neighbors[i]->payload()->level - payload()->level) > 1) {
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
      if (neighbor_neighbor->payload()->merged) {
        // check neighbor_neighbor level
        if (std::abs(neighbor_neighbor->payload()->level - payload()->level) <=
            1) {
          can_merge[i] = false;
        }
        if (std::abs(
                neighbor_neighbor->payload()->level -
                neighbors[i]->payload()->level) <= 1) {
          can_merge[i] = false;
        }

        // check neighbor_neighber->merged->level
        if (std::abs(
                neighbor_neighbor->payload()->merge_with->payload()->level -
                payload()->level) <= 1) {
          can_merge[i] = false;
        }
        if (std::abs(
                neighbor_neighbor->payload()->merge_with->payload()->level -
                neighbors[i]->payload()->level) <= 1) {
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
  payload()->level = -1;
  payload()->visited = false;
  payload()->merge_with = nullptr;
  payload()->merged = false;
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
    if (group->producer_edges.empty()) {
      to_visit.push_back(group.get());
    }
    group->payload()->visited = false;
    group->payload()->level = 0;
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
          [&](SegmentedEdge* dep) { return dep->from_->payload()->visited; });
    }

    if (!ready) {
      // In case traversal doesn't complete because there's an error in the
      // DAG topology.
      next_to_visit.push_back(visit);
      continue;
    }

    visit->payload()->visited = true;

    to_visit.insert(to_visit.end(), next_to_visit.begin(), next_to_visit.end());
    next_to_visit.clear();

    for (auto out : visit->consumer_edges) {
      to_visit.push_back(out->to_);
    }

    visit->payload()->level = 0;
    for (auto inp : visit->producer_edges) {
      visit->payload()->level =
          std::max(visit->payload()->level, inp->from_->payload()->level + 1);
    }
  }
  TORCH_INTERNAL_ASSERT(next_to_visit.empty(), "Error in graph, is not a DAG.");
}

SegmentedGroup* SegmentCandidateFinder::makeEmptyGroup() {
  groups.push_back(std::make_unique<SegmentedGroup>());
  return groups.back().get();
}

SegmentedGroup* SegmentCandidateFinder::makeEmptyGroup(Expr* expr) {
  groups.push_back(std::make_unique<SegmentedGroup>());
  groups.back().get()->exprs_.push_back(expr);
  return groups.back().get();
}

std::string SegmentCandidateFinder::toString(int verbosity) const {
  std::stringstream ss;
  for (auto& group : groups) {
    ss << group.get() << "\n";

    if (verbosity > 1) {
      if (group->producer_edges.size() > 0) {
        ss << "  produced by groups: { \n";
        for (auto producer_edge : group->producer_edges) {
          ss << "    " << producer_edge->from_ << " via " << producer_edge->val_
             << "\n";
        }
        ss << "  }"
           << "\n";
      }
    }

    if (verbosity > 0) {
      if (group->consumer_edges.size() > 0) {
        ss << "  Consumed by groups: { \n";
        for (auto consumer_edge : group->consumer_edges) {
          ss << "    " << consumer_edge->to_ << "\n";
        }
        ss << "  }"
           << "\n";
      }
    }

    if (verbosity > 2) {
      ss << "  Exprs{\n";
      for (auto expr : group->exprs_) {
        ss << "    " << expr;
      }
      ss << "  }\n";
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

// Assuming sg1 and sg2 are connected, figure out which is the consumer
const SegmentedGroup* getProducer(
    const SegmentedGroup* sg1,
    const SegmentedGroup* sg2) {
  for (auto producer_edge : sg1->producer_edges) {
    if (producer_edge->from_ == sg2) {
      return sg2;
    }
  }

  for (auto consumer_edge : sg1->consumer_edges) {
    if (consumer_edge->to_ == sg2) {
      return sg1;
    }
  }

  return nullptr;
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

SegmentedGroup* SegmentCandidateFinder::makeMergedNode(
    SegmentedGroup* sg1,
    SegmentedGroup* sg2) {
  // Make the new joined node
  auto joined_group = makeEmptyGroup();

  joined_group->input_vals =
      uniqueValConcat({sg1->input_vals, sg2->input_vals});

  joined_group->output_vals =
      uniqueValConcat({sg1->output_vals, sg2->output_vals});

  // Keep Expr's sorted in topological order.
  auto producer = getProducer(sg1, sg2);
  auto consumer = sg1 == producer ? sg2 : sg1;

  TORCH_INTERNAL_ASSERT(
      producer != nullptr,
      "Tried to merge expr's together that aren't neighbors.");

  joined_group->exprs_ = producer->exprs_;
  joined_group->exprs_.insert(
      joined_group->exprs_.end(),
      consumer->exprs_.begin(),
      consumer->exprs_.end());

  auto producer_edges = getMergedProducerEdges(sg1, sg2);
  // Connect joined group to resulting neighbors
  for (auto& edge : producer_edges) {
    auto from = edge->from_;
    auto val = edge->val_;

    edges.push_back(std::make_unique<SegmentedEdge>(from, joined_group, val));

    joined_group->producer_edges.push_back(edges.back().get());
    from->consumer_edges.push_back(edges.back().get());
  }

  auto consumer_edges = getMergedConsumerEdges(sg1, sg2);

  for (auto& edge : consumer_edges) {
    auto to = edge->to_;
    auto val = edge->val_;

    edges.push_back(std::make_unique<SegmentedEdge>(joined_group, to, val));
    joined_group->consumer_edges.push_back(edges.back().get());
    edge->to_->producer_edges.push_back(edges.back().get());
  }

  return joined_group;
}

void SegmentCandidateFinder::mergeNodes() {
  while (!to_merge.empty()) {
    auto group1 = *to_merge.begin();
    auto group2 = group1->payload()->merge_with;
    to_merge.erase(group1);
    to_merge.erase(group2);
    clean_up_groups.emplace(group1);
    clean_up_groups.emplace(group2);
    auto joined_group = makeMergedNode(group1, group2);
  }

  for (auto group : clean_up_groups) {
    auto disconnected_edges = disconnectGroup(group);
    clean_up_edges.insert(disconnected_edges.begin(), disconnected_edges.end());
  }

  edges.remove_if([this](std::unique_ptr<SegmentedEdge>& edge) {
    return this->clean_up_edges.find(edge.get()) != this->clean_up_edges.end();
  });

  groups.remove_if([this](std::unique_ptr<SegmentedGroup>& group) {
    return this->clean_up_groups.find(group.get()) !=
        this->clean_up_groups.end();
  });

  clean_up_edges.clear();
  clean_up_groups.clear();
}

bool SegmentCandidateFinder::codeGenSupportedMerge(
    SegmentedGroup* sg1,
    SegmentedGroup* sg2) {
  return true;
}

SegmentCandidateFinder::SegmentCandidateFinder(Fusion* fusion)
    : complete_fusion(fusion) {}

void SegmentCandidateFinder::segment() {
  // TODO: Make traversal items local to this function.

  // Need this for initialization of the DAG that is process
  std::unordered_map<Expr*, SegmentedGroup*> expr2group;

  // Initialize DAG, convert each expr to a segment group
  for (auto expr : complete_fusion->exprs()) {
    auto group = makeEmptyGroup(expr);
    expr2group.insert(std::make_pair(expr, group));
  }

  // Create edges between the Exprs. Mark inputs and outputs of the fusion.
  for (auto expr : complete_fusion->exprs()) {
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
      edges.push_back(
          std::make_unique<SegmentedEdge>(def_group, expr_group, inp));
      expr_group->producer_edges.push_back(edges.back().get());
      def_group->consumer_edges.push_back(edges.back().get());
    }
    for (auto out : expr->outputs()) {
      if (out->isFusionOutput()) {
        expr_group->output_vals.push_back(out);
      }
    }
  }

  bool inter_iter_update = true;
  while (inter_iter_update) {
    bool merged_nodes = true;
    while (merged_nodes) {
      // Reset stateful traversal details in SegmentedGroups
      resetTraversal();
      resetLevels();

      for (auto& group : groups) {
        if (group->payload()->merged) {
          continue;
        }
        auto candidates = group->getMergeCandidates();
        if (candidates.empty()) {
          continue;
        }

        auto candidate_it = candidates.begin();
        while (candidate_it != candidates.end() &&
               !codeGenSupportedMerge(group.get(), *candidate_it)) {
          candidate_it++;
        }
        if (candidate_it == candidates.end()) {
          continue;
        }

        to_merge.emplace(group.get());
        to_merge.emplace(*candidate_it);

        group->payload()->merged = true;
        group->payload()->merge_with = *candidate_it;

        (*candidate_it)->payload()->merged = true;
        (*candidate_it)->payload()->merge_with = group.get();
      }

      if (to_merge.empty()) {
        merged_nodes = false;
      }

      mergeNodes();

      // std::cout << this->toString(4) << std::endl;
      inter_iter_update = interIterUpdate();
    }
  }
}

std::ostream& operator<<(std::ostream& os, const SegmentCandidateFinder* scf) {
  return os << scf->toString();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch