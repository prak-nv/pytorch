#include <torch/csrc/jit/codegen/cuda/fusion_segmenter.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

std::vector<std::pair<SegmentedGroup*, SegmentedEdge*>> SegmentedGroup::
    getNeighborsPair() {
  std::vector<std::pair<SegmentedGroup*, SegmentedEdge*>> neighbors;
  for (auto inp : producer_edges) {
    neighbors.emplace_back(inp->from_, inp);
  }
  for (auto out : consumer_edges) {
    neighbors.emplace_back(out->to_, out);
  }
  return neighbors;
}

std::vector<SegmentedGroup*> SegmentedGroup::getNeighbors() {
  std::vector<SegmentedGroup*> neighbors;
  auto neighbors_pair = getNeighborsPair();

  std::transform(
      neighbors_pair.begin(),
      neighbors_pair.end(),
      std::back_inserter(neighbors),
      [](std::pair<SegmentedGroup*, SegmentedEdge*> pair) {
        return pair.first;
      });
  return neighbors;
}

std::vector<std::pair<SegmentedGroup*, SegmentedEdge*>> SegmentedGroup::
    getMergeCandidates() {
  std::vector<std::pair<SegmentedGroup*, SegmentedEdge*>> neighbors =
      getNeighborsPair();

  // Don't look for candidates if already merged
  if (merged) {
    return {};
  }

  // Can this node be merged with another? Check if neighbors are merged, if
  // so and merged neighbor is within 1 level or node merged with neighbor is
  // within 1 level, can't merge this node with anything else.
  bool can_merge_this = true;
  for (size_t i = 0; i < neighbors.size(); i++) {
    if (!neighbors[i].first->merged) {
      continue;
    }
    if (std::abs(neighbors[i].first->level - level) <= 1) {
      can_merge_this = false;
    }
    if (std::abs(neighbors[i].first->merge_with->level - level) <= 1) {
      can_merge_this = false;
    }
  }
  if (!can_merge_this) {
    return {};
  }

  std::vector<bool> can_merge(true, neighbors.size());

  // Find neighbors with a level that is only 1 differant than this groups level
  for (size_t i = 0; i < neighbors.size(); i++) {
    if (std::abs(neighbors[i].first->level - level) > 1) {
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

    for (auto neighbor_neighbor : neighbors[i].first->getNeighbors()) {
      // Don't check self
      if (neighbor_neighbor == neighbors[i].first) {
        continue;
      }
      if (neighbor_neighbor->merged) {
        // check neighbor_neighbor level
        if (std::abs(neighbor_neighbor->level - level) <= 1) {
          can_merge[i] = false;
        }
        if (std::abs(neighbor_neighbor->level - neighbors[i].first->level) <=
            1) {
          can_merge[i] = false;
        }

        // check neighbor_neighber->merged->level
        if (std::abs(neighbor_neighbor->merge_with->level - level) <= 1) {
          can_merge[i] = false;
        }
        if (std::abs(
                neighbor_neighbor->merge_with->level -
                neighbors[i].first->level) <= 1) {
          can_merge[i] = false;
        }
      }
    }
  }

  std::vector<std::pair<SegmentedGroup*, SegmentedEdge*>> merge_candidates;
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

std::vector<Val*> SegmentedGroup::edgesToVals(
    const std::vector<SegmentedEdge*>& se_v) {
  std::vector<Val*> ret_v;
  ret_v.reserve(se_v.size());

  std::transform(
      se_v.cbegin(),
      se_v.cend(),
      std::back_inserter(ret_v),
      [](SegmentedEdge* se) { return se->val_; });
  return ret_v;
}

void SegmentedGroup::print() {
  std::cout << "g{";
  for (size_t i = 0; i < exprs_.size(); i++) {
    exprs_[i]->print();
    if (i + 1 != exprs_.size())
      std::cout << ", ";
  }
  std::cout << "}\n\n";
}

std::ostream& operator<<(std::ostream& os, const SegmentedGroup* group) {
  os << "g{";
  for (size_t i = 0; i < group->exprs_.size(); i++) {
    os << group->exprs_[i]->name();
    if (i + 1 != group->exprs_.size())
      os << ", ";
  }
  os << "}\n";
  return os;
}

void SegmentedEdge::print() {
  std::cout << "e{ " << std::endl << "  ";
  from_->print();
  std::cout << " -> " << std::endl << "  ";
  to_->print();
  std::cout << "through: ";
  val_->print();
  std::cout << "}\\\\e\n\n";
}

std::ostream& operator<<(std::ostream& os, const SegmentedEdge* edge) {
  os << "e{ " << edge->from_ << " -> " << edge->to_ << " }" << std::endl;
  return os;
}

SegmentedFusion::SegmentedFusion(const Fusion* fusion)
    : fusion_(*fusion), impl_(this) {}

void SegmentedFusion::print() {
  std::cout << "Segmented_Fusion{ \n";
  for (auto g : groups()) {
    g->print();
  }

  for (auto e : edges()) {
    std::cout << e << std::endl;
  }
  std::cout << "} //Segmented_Fusion\n";
}

SegmentedGroup* SegmentedFusion::Impl::makeGroup() {
  groups_.emplace_back(std::make_unique<SegmentedGroup>());
  return groups_.back().get();
}

SegmentedGroup* SegmentedFusion::Impl::makeGroup(Expr* expr) {
  groups_.emplace_back(std::make_unique<SegmentedGroup>(expr));
  return groups_.back().get();
}

SegmentedEdge* SegmentedFusion::Impl::makeEdge(
    SegmentedGroup* from,
    SegmentedGroup* to,
    Val* val) {
  edges_.emplace_back(std::make_unique<SegmentedEdge>(from, to, val));
  return edges_.back().get();
}

void SegmentedFusion::Impl::cleanUnused() {
  std::unordered_set<SegmentedGroup*> g_used(
      owning_fusion_->groups().begin(), owning_fusion_->groups().end());
  std::unordered_set<SegmentedEdge*> e_used(
      owning_fusion_->edges().begin(), owning_fusion_->edges().end());

  groups_.erase(
      std::remove_if(
          groups_.begin(),
          groups_.end(),
          [&g_used](auto& g) { return g_used.count(g.get()) == 0; }),
      groups_.end());

  edges_.erase(
      std::remove_if(
          edges_.begin(),
          edges_.end(),
          [&e_used](auto& e) { return e_used.count(e.get()) == 0; }),
      edges_.end());
}

SegmentedGroup* SegmentedFusion::newGroup() {
  SegmentedGroup* g = impl_.makeGroup();
  groups_.push_back(g);
  return g;
}

SegmentedGroup* SegmentedFusion::newGroup(Expr* expr) {
  SegmentedGroup* g = impl_.makeGroup(expr);
  groups_.push_back(g);
  return g;
}

SegmentedEdge* SegmentedFusion::newEdge(
    SegmentedGroup* from,
    SegmentedGroup* to,
    Val* val) {
  SegmentedEdge* e = impl_.makeEdge(from, to, val);
  edges_.push_back(e);
  return e;
}

std::string SegmentedFusion::toString(int verbosity) const {
  std::stringstream ss;
  for (auto& group : groups_) {
    ss << "group " << group->groupId() << "\n";

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
  }
  return ss.str();
}

void SegmentedFusion::finalize() {
  impl_.cleanUnused();
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

std::unique_ptr<Fusion> SegmentedFusion::makeFusion(SegmentedGroup* sg) {
  std::unique_ptr<Fusion> fusion_segment = std::make_unique<Fusion>();

  auto complete_to_segment_map = Fusion::copy(&fusion_, fusion_segment.get());

  for (auto inp : fusion_segment->inputs()) {
    fusion_segment->removeInput(inp);
  }
  for (auto out : fusion_segment->outputs()) {
    fusion_segment->removeOutput(out);
  }

  for (auto inp : getAllInputs(sg)) {
    fusion_segment->addInput(complete_to_segment_map.clone(inp));
  }

  for (auto out : getAllOutputs(sg)) {
    fusion_segment->addOutput(complete_to_segment_map.clone(out));
  }

  return fusion_segment;
}

std::ostream& operator<<(std::ostream& os, const SegmentedFusion* scf) {
  return os << scf->toString(0);
}

void SegmentCandidateFinder::resetTraversal() {
  for (auto group : groups()) {
    // Start traversal at input groups
    if (group->producer_edges.empty()) {
      to_visit.push_back(group);
    }
    group->visited = false;
    group->level = 0;
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
    removed_edges.insert(edge);
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
    auto joined_group = segmented_fusion->newGroup();

    joined_group->input_vals =
        uniqueValConcat({group1->input_vals, group2->input_vals});

    joined_group->output_vals =
        uniqueValConcat({group1->output_vals, group2->output_vals});

    joined_group->exprs_ = group1->exprs_;
    joined_group->exprs_.insert(
        joined_group->exprs_.end(),
        group2->exprs_.begin(),
        group2->exprs_.end());

    auto producer_edges = getMergedProducerEdges(group1, group2);
    // Connect joined group to resulting neighbors
    for (auto edge : producer_edges) {
      auto from = edge->from_;
      auto val = edge->val_;

      auto new_edge = segmented_fusion->newEdge(from, joined_group, val);
      joined_group->producer_edges.push_back(new_edge);
      from->consumer_edges.push_back(new_edge);
    }

    auto consumer_edges = getMergedConsumerEdges(group1, group2);

    for (auto edge : consumer_edges) {
      auto to = edge->to_;
      auto val = edge->val_;

      auto new_edge = segmented_fusion->newEdge(joined_group, to, val);
      joined_group->consumer_edges.push_back(new_edge);
      edge->to_->producer_edges.push_back(new_edge);
    }

    joined_group->setHeuristic(deriveHeuristic(group1->merge_through));
  }

  for (auto group : clean_up_groups) {
    auto disconnected_edges = disconnectGroup(group);
    clean_up_edges.insert(disconnected_edges.begin(), disconnected_edges.end());
  }

  edges().erase(
      std::remove_if(
          edges().begin(),
          edges().end(),
          [this](SegmentedEdge* edge) {
            if (this->clean_up_edges.find(edge) != this->clean_up_edges.end()) {
              return true;
            };
            return false;
          }),
      edges().end());

  groups().erase(
      std::remove_if(
          groups().begin(),
          groups().end(),
          [this](SegmentedGroup* group) {
            if (this->clean_up_groups.find(group) !=
                this->clean_up_groups.end()) {
              return true;
            };
            return false;
          }),
      groups().end());

  clean_up_edges.clear();
  clean_up_groups.clear();
}

namespace {

// Guard to temporarily change the inputs and outputs of a fusion. On
// destruction will return fusion to original state.
// Not used temporarily but will be useful when adding more mergin heuristics
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

// Note:
//  Below two methods should later be generalized once we start
//  supporting more optimized schedules
// Use heuristics based merging rule
bool SegmentCandidateFinder::codeGenSupportedMerge(SegmentedEdge* edge) {
  SegmentedGroup* producer = edge->from_;
  SegmentedGroup* consumer = edge->to_;

  if (producer->heuristic() == ScheduleHeuristic::PointWise) {
    return true;
  }

  // TODO: Normalization rule was duplicated from kernel_cache.cpp,
  // might need to be extended more for safety
  if (producer->heuristic() == ScheduleHeuristic::Reduction) {
    // Detect normalization creation
    Val* val = edge->val_;
    for (auto expr : consumer->exprs_) {
      if (expr->isA<BroadcastOp>()) {
        if (expr->input(0)->sameAs(val)) {
          return true;
        }
      }
    }
    return false;
  }

  // TODO: come up with a concise way to check root domain consistency
  return consumer->heuristic() == ScheduleHeuristic::PointWise;
}

// Manual way of setting heuristics,
//  assuming canMerge has already been checked
ScheduleHeuristic SegmentCandidateFinder::deriveHeuristic(SegmentedEdge* edge) {
  SegmentedGroup* producer = edge->from_;
  SegmentedGroup* consumer = edge->to_;

  if (producer->heuristic() == ScheduleHeuristic::Normalization ||
      producer->heuristic() == ScheduleHeuristic::Reduction) {
    // Note:
    // Assumption here is that we are not allowing fusion through
    // reduction ops unless producing normalization
    return ScheduleHeuristic::Normalization;
  }

  return consumer->heuristic();
}

SegmentCandidateFinder::SegmentCandidateFinder(const Fusion* fusion) {
  segmented_fusion = std::make_unique<SegmentedFusion>(fusion);
  findSegments();
}

void SegmentCandidateFinder::findSegments() {
  // TODO: Make traversal items local to this function.

  // Need this for initialization of the DAG that is process
  std::unordered_map<Expr*, SegmentedGroup*> expr2group;

  // Initialize DAG, convert each expr to a segment group
  for (auto expr : completeFusion().exprs()) {
    auto new_group = segmented_fusion->newGroup(expr);
    expr2group.insert(std::make_pair(expr, new_group));
  }

  // Create edges between the Exprs. Mark inputs and outputs of the fusion.
  for (auto expr : completeFusion().exprs()) {
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
      auto new_edge = segmented_fusion->newEdge(def_group, expr_group, inp);
      expr_group->producer_edges.push_back(new_edge);
      def_group->consumer_edges.push_back(new_edge);
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

    for (auto& group : groups()) {
      if (group->merged) {
        continue;
      }
      auto candidates = group->getMergeCandidates();
      if (candidates.empty()) {
        continue;
      }

      auto candidate_it = candidates.begin();
      while (candidate_it != candidates.end() &&
             !codeGenSupportedMerge(candidate_it->second)) {
        candidate_it++;
      }
      if (candidate_it == candidates.end()) {
        continue;
      }

      to_merge.emplace(group);
      to_merge.emplace(candidates[0].first);

      group->merged = true;
      group->merge_with = candidates[0].first;
      group->merge_through = candidates[0].second;

      candidates[0].first->merged = true;
      candidates[0].first->merge_with = group;
      candidates[0].first->merge_through = candidates[0].second;
    }

    if (to_merge.empty()) {
      merged_nodes = false;
    }

    mergeNodes();
  }

  finalize();
}

void SegmentCandidateFinder::finalize() {
  // Remove unconnected groups
  groups().erase(
      std::remove_if(
          groups().begin(),
          groups().end(),
          [](SegmentedGroup* sg) { return !sg->isConnected(); }),
      groups().end());

  // Add group labeling
  int i = 0;
  for (auto it = groups().begin(); it != groups().end(); it++, i++) {
    (*it)->setID(i);
  }

  segmented_fusion->finalize();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch