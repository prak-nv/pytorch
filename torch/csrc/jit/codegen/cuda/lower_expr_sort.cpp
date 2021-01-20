#include <torch/csrc/jit/codegen/cuda/lower_expr_sort.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <unordered_map>
#include <unordered_set>

#include <deque>
#include <list>
#include <vector>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// TODO: Review const model, and objects
//  ExprSegmentationSorter
//    Responsible for going through DAG and proposing things we could try to
//    merge together, calls "supportedMerge" on these proposed groups to see
//    if they should be merged together, then merges them if so.
//  ExprGroup
//    A group of exprs that are grouped together based on their loop nest
//    structures.
//  ExprGroupConnections
//    Holds vals and what they connect. In other words it's a val that is an
//    output of a ExprSegmentationSorter "from" and an input of
//    ExprSegmentationSorter "to". There's nothing preventing from a val being
//    between groups twice.
//    TODO: make sure there's nothing wrong with grouping of nodes that
//    have the same value input twice. i.e. (B = A*A)

// Selecting segments to propose is based on the theorem 4.2 in the paper which
// makes sure when segment the segmented graph will be a DAG (assumes Fusion is
// already a DAG). The segmentation code relies on assumptions of DAG-ness
// during segmentation, meaning proposed merging of groups must maintain the DAG
// property of the graph.
//
// Julien Herrmann, Yusuf Özkaya, Bora Uçar, Kamer Kaya, Umit Catalyurek.
// Multilevel Algorithms for Acyclic Partitioning of Directed Acyclic Graphs.
// SIAM Journal on Scientific Computing, Society for Industrial and Applied
// Mathematics, 2019, 41 (4), pp.A2117-A2145. ff10.1137/18M1176865ff.
// ffhal02306566f

class ExprGroup;

// Wrapper for values, these are edges between expr groups. Multiple edges can
// exist between expr groups, and the same Val can show up more than once in
// multiple edges.
class ExprGroupConnections {
 public:
  ExprGroupConnections(ExprGroup* from, ExprGroup* to, Val* val)
      : from_(from), to_(to), val_(val) {}
  ExprGroup* from_;
  ExprGroup* to_;
  Val* val_;
};

std::ostream& operator<<(std::ostream& os, const ExprGroupConnections* edge);

class ExprSortPayload : public PolymorphicBase {
 public:
  // Track the active domains that start at the compute at point of the
  // expression and increment outward
  std::vector<IterDomain*> ca_domains;

  // Maximum path distance from an input expr group required for
  // Theorem 4.2
  int level = -1;

  // Traversal marker, marks if this group has been visited by current pass
  bool visited = false;

  // Marks if this group is already selected to merge with another group, marks
  // which group to merge with
  ExprGroup* merge_with = nullptr;

  // Marks if this group is already selected to merge with another group
  bool merged = false;
};

// Groups together expressions which create a expr group
class ExprGroup {
 public:
  explicit ExprGroup() : payload_(std::make_unique<ExprSortPayload>()) {}

  ExprGroup(Expr* expr) : payload_(std::make_unique<ExprSortPayload>()) {
    exprs_.push_back(expr);
  }

  ExprGroup(const ExprGroup& other)
      : payload_(new ExprSortPayload(*(other.payload_))) {}

  ExprGroup& operator=(const ExprGroup& other) {
    *payload_ = *other.payload_;
    exprs_ = other.exprs_;
    return *this;
  }

  // Clears the traversal information in the payload
  void clearTraversalInfo();

  // Returns all neighbors, producers and consumers
  std::vector<ExprGroup*> getNeighbors();

  // Look at all neighbors of this and return who this could merge with based on
  // level values of this, neighbors, and merged neighbors of neighbors
  std::vector<ExprGroup*> getMergeCandidates();

  std::unique_ptr<ExprSortPayload>& payload() {
    return payload_;
  }

 public:
  // "Ancestor nodes", towards inputs of segmentedDAG
  std::vector<ExprGroupConnections*> producer_edges;

  // "Descendent nodes", towards outputs of segmentedDAG
  std::vector<ExprGroupConnections*> consumer_edges;

  std::vector<Val*> input_vals;
  std::vector<Val*> output_vals;

  // Exprs that make up the group
  std::vector<Expr*> exprs_;

  // Stateful traversal information
  std::unique_ptr<ExprSortPayload> payload_;
};

std::ostream& operator<<(std::ostream& os, const ExprGroup* group);

class ExprSegmentationSorter {
 public:
  ExprSegmentationSorter(Fusion* fusion) : complete_fusion(fusion) {}

  void sort();

  std::string toString(int verbosity = 0) const;

  const std::vector<ExprGroup*> getGroups() {
    std::vector<ExprGroup*> group_vec;
    std::transform(
        groups.begin(),
        groups.end(),
        std::back_inserter(group_vec),
        [](std::unique_ptr<ExprGroup>& sg) { return sg.get(); });
    return group_vec;
  }

 private:
  // Allocate an empty expr group and return it
  ExprGroup* makeEmptyGroup();

  // Allocate an expr group with the provided expr and return it
  ExprGroup* makeEmptyGroup(Expr*);

  // Returns if sg1 and sg2 should be merged together, is called if they can
  // based on the current status of the DAG.
  bool supportedMerge(ExprGroup* sg1, ExprGroup* sg2);

  // Merges two ExprGroups and returns the new ExprGroup
  ExprGroup* makeMergedNode(ExprGroup* sg1, ExprGroup* sg2);

  // This is called once no more groups can be merged together. This will lower
  // the compute at position of a segment group if the last dimension of the
  // segment group doesn't map to any of the dimensions of its neighbors.
  bool interIterUpdate();

  // Reset the ExprSortPayload of the groups so we can traverse and identify
  // merge candidates.
  void resetTraversal();

  // Reset the set levels of each group. This is what's used to identify which
  // nodes can be merged together.
  void resetLevels();

  // Go through groups that are marked as to merge and merge them.
  void mergeNodes();

  // Disconnect the edges connecting group to the rest of the graph, and return
  // all the edges that were disconnected
  std::unordered_set<ExprGroupConnections*> disconnectGroup(ExprGroup* group);

  // Track how many groups we have from iteration to iteration so we can track
  // when we've stopped merging nodes.
  size_t n_groups = 0;

 private:
  // Lifetime of the graph view of the fusion and segmentation. Use list to not
  // invalidate any entries on insertion/deletion.
  std::list<std::unique_ptr<ExprGroupConnections>> edges;
  std::list<std::unique_ptr<ExprGroup>> groups;

  std::deque<ExprGroup*> to_visit;
  std::vector<ExprGroup*> next_to_visit;

  std::unordered_set<ExprGroup*> clean_up_groups;
  std::unordered_set<ExprGroupConnections*> clean_up_edges;

  std::unordered_set<ExprGroup*> to_merge;

  // Maintain my own fusion the state of which is not always the same as the
  // original provided fusion.
  Fusion* complete_fusion;
};

std::ostream& operator<<(std::ostream& os, const ExprSegmentationSorter* scf);

std::vector<ExprGroup*> ExprGroup::getNeighbors() {
  std::vector<ExprGroup*> neighbors;
  for (auto inp : producer_edges) {
    neighbors.push_back(inp->from_);
  }
  for (auto out : consumer_edges) {
    neighbors.push_back(out->to_);
  }
  return neighbors;
}

std::vector<ExprGroup*> ExprGroup::getMergeCandidates() {
  std::vector<ExprGroup*> neighbors = getNeighbors();

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

  std::vector<ExprGroup*> merge_candidates;
  for (size_t i = 0; i < neighbors.size(); i++) {
    if (can_merge[i]) {
      merge_candidates.push_back(neighbors[i]);
    }
  }
  return merge_candidates;
}

void ExprGroup::clearTraversalInfo() {
  payload()->level = -1;
  payload()->visited = false;
  payload()->merge_with = nullptr;
  payload()->merged = false;
}

std::ostream& operator<<(std::ostream& os, const ExprGroup* group) {
  os << "g{";
  for (size_t i = 0; i < group->exprs_.size(); i++) {
    os << group->exprs_[i]->name();
    if (i + 1 != group->exprs_.size())
      os << ", ";
  }
  os << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const ExprGroupConnections* edge) {
  os << "e{ " << edge->from_ << " -> " << edge->to_ << " }" << std::endl;
  return os;
}

void ExprSegmentationSorter::resetTraversal() {
  for (auto& group : groups) {
    // Start traversal at input groups
    if (group->producer_edges.empty()) {
      to_visit.push_back(group.get());
    }
    group->clearTraversalInfo();
  }
}

// Level is maximum distance from inputs. It's the metric used to select what
// nodes can be merged while maintaining a DAG
void ExprSegmentationSorter::resetLevels() {
  while (!to_visit.empty()) {
    auto visit = to_visit.front();
    to_visit.pop_front();

    // All inputs processed?
    bool ready = true;
    if (!visit->producer_edges.empty()) {
      ready = std::all_of(
          visit->producer_edges.begin(),
          visit->producer_edges.end(),
          [&](ExprGroupConnections* dep) {
            return dep->from_->payload()->visited;
          });
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

ExprGroup* ExprSegmentationSorter::makeEmptyGroup() {
  groups.push_back(std::make_unique<ExprGroup>());
  return groups.back().get();
}

ExprGroup* ExprSegmentationSorter::makeEmptyGroup(Expr* expr) {
  groups.push_back(std::make_unique<ExprGroup>());
  auto* group = groups.back().get();
  group->exprs_.push_back(expr);
  if (ir_utils::isTVOp(expr)) {
    auto out_tv = expr->outputs()[0]->as<TensorView>();
    // Loop map produces a produce_at_map used specifically for expr sorting
    // when we generate it. Produce at may be a misnomer, as it really marks the
    // inner most loop that is shared with any producers of a tv.
    for (size_t tv_i = 0;
         tv_i < GpuLower::current()->caLoopMap().producedAt(out_tv);
         tv_i++) {
      group->payload()->ca_domains.push_back(out_tv->axis(tv_i));
    }
  }
  return group;
}

// Debug function that prints the current state of the sorter.
std::string ExprSegmentationSorter::toString(int verbosity) const {
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
std::vector<ExprGroupConnections*> getMergedProducerEdges(
    const ExprGroup* sg1,
    const ExprGroup* sg2) {
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
          [&sg1, &sg2](ExprGroupConnections* se) {
            return (se->to_ == sg1 && se->from_ == sg2) ||
                (se->to_ == sg2 && se->from_ == sg1);
          }),
      producer_edges.end());

  return producer_edges;
}

// Concat's consumer edges of sg1 and sg2, but removes any edges from/to sg1/sg2
std::vector<ExprGroupConnections*> getMergedConsumerEdges(
    const ExprGroup* sg1,
    const ExprGroup* sg2) {
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
          [&sg1, &sg2](ExprGroupConnections* se) {
            return (se->to_ == sg1 && se->from_ == sg2) ||
                (se->to_ == sg2 && se->from_ == sg1);
          }),
      consumer_edges.end());

  return consumer_edges;
}

// Assuming sg1 and sg2 are connected, figure out which is the consumer
const ExprGroup* getProducer(const ExprGroup* sg1, const ExprGroup* sg2) {
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
std::unordered_set<ExprGroupConnections*> ExprSegmentationSorter::
    disconnectGroup(ExprGroup* group) {
  std::unordered_set<ExprGroupConnections*> removed_edges(
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

ExprGroup* ExprSegmentationSorter::makeMergedNode(
    ExprGroup* sg1,
    ExprGroup* sg2) {
  std::vector<IterDomain*> resulting_ca_axes;
  auto& domain1 = sg1->payload()->ca_domains;
  auto& domain2 = sg2->payload()->ca_domains;
  auto it1 = domain1.begin();
  auto it2 = domain2.begin();

  while (it1 != domain1.end() && it2 != domain2.end()) {
    if (it1 == domain1.end()) {
      resulting_ca_axes.push_back(*it2++);
    } else if (it2 == domain2.end()) {
      resulting_ca_axes.push_back(*it1++);
    } else if (GpuLower::current()->caIndexMap().areMapped(*it1, *it2)) {
      resulting_ca_axes.push_back(*it1);
      ++it1;
      ++it2;
    } else if (std::any_of(it1 + 1, domain1.end(), [&](IterDomain* id1) {
                 return GpuLower::current()->caIndexMap().areMapped(id1, *it2);
               })) {
      // Increment it1, as a later iter domain matches the current one in
      // domain2
      resulting_ca_axes.push_back(*it1++);

    } else if (std::any_of(it2 + 1, domain2.end(), [&](IterDomain* id2) {
                 return GpuLower::current()->caIndexMap().areMapped(id2, *it1);
               })) {
      // Increment it2, as a later iter domain matches the current one in
      // domain1
      resulting_ca_axes.push_back(*it2++);
    } else {
      resulting_ca_axes.push_back(*it1++);
      resulting_ca_axes.push_back(*it2++);
    }
  }

  // Make the new joined node
  auto joined_groups = makeEmptyGroup();

  joined_groups->input_vals =
      uniqueValConcat({sg1->input_vals, sg2->input_vals});

  joined_groups->output_vals =
      uniqueValConcat({sg1->output_vals, sg2->output_vals});

  // Keep Expr's sorted in topological order.
  auto producer = getProducer(sg1, sg2);
  auto consumer = sg1 == producer ? sg2 : sg1;

  TORCH_INTERNAL_ASSERT(
      producer != nullptr,
      "Tried to merge expr's together that aren't neighbors.");

  joined_groups->exprs_ = producer->exprs_;
  joined_groups->exprs_.insert(
      joined_groups->exprs_.end(),
      consumer->exprs_.begin(),
      consumer->exprs_.end());

  auto producer_edges = getMergedProducerEdges(sg1, sg2);
  // Connect joined group to resulting neighbors
  for (auto& edge : producer_edges) {
    auto from = edge->from_;
    auto val = edge->val_;

    edges.push_back(
        std::make_unique<ExprGroupConnections>(from, joined_groups, val));

    joined_groups->producer_edges.push_back(edges.back().get());
    from->consumer_edges.push_back(edges.back().get());
  }

  auto consumer_edges = getMergedConsumerEdges(sg1, sg2);

  for (auto& edge : consumer_edges) {
    auto to = edge->to_;
    auto val = edge->val_;

    edges.push_back(
        std::make_unique<ExprGroupConnections>(joined_groups, to, val));
    joined_groups->consumer_edges.push_back(edges.back().get());
    edge->to_->producer_edges.push_back(edges.back().get());
  }

  joined_groups->payload()->ca_domains = resulting_ca_axes;

  return joined_groups;
}

// Update in between attempts to segment. This is called once no more groups
// can be merged together. Typically we will want to remove compute at groups
// that have finished being grouped together. However if no gruops have been
// merged after we've done this, we may need to stop as we could have multiple
// disjoint groups that won't be merged.
bool ExprSegmentationSorter::interIterUpdate() {
  // Go through groups and lower compute at domain
  bool lowered_ca_domain = false;
  for (auto& unique_group : groups) {
    auto group = unique_group.get();
    IterDomain* g_last_id = nullptr;
    if (group->payload()->ca_domains.size() > 0) {
      g_last_id = group->payload()->ca_domains.back();
    }
    if (g_last_id == nullptr) {
      continue;
    }

    bool matching_neighbor = false;
    for (auto neighbor : group->getNeighbors()) {
      if (matching_neighbor) {
        break;
      }
      for (auto p_id : neighbor->payload()->ca_domains) {
        if (GpuLower::current()->caIndexMap().areMapped(p_id, g_last_id)) {
          matching_neighbor = true;
          break;
        }
      }
    }

    if (!matching_neighbor) {
      group->payload()->ca_domains.pop_back();
      lowered_ca_domain = true;
    }
  }

  // If we couldn't lower compute at domain any further, and we haven't merged
  // any new groups since the last time we were called, make sure we're done.
  if (!lowered_ca_domain && n_groups == groups.size()) {
    // Make sure none of the groups are still connected, as that would mean we
    // should have been able to merge them.

    TORCH_INTERNAL_ASSERT(
        std::all_of(
            groups.begin(),
            groups.end(),
            [](std::unique_ptr<ExprGroup>& sg) {
              return sg->producer_edges.empty() && sg->consumer_edges.empty();
            }),
        "Couldn't succcessfully sort out the fusion expressions. ",
        "There are remaining connections of the heirarchical segmentation which should have been ",
        "flattened to a single ordered group, or disjoint ordered groups.");

    // Successfully finished
    return false;
  }

  // Initialize n_groups if this is the first pass.
  if (n_groups == 0 && groups.size() > 0) {
    n_groups = groups.size();
  }

  n_groups = groups.size();
  // Not done, continue.
  return true;
}

void ExprSegmentationSorter::mergeNodes() {
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

  edges.remove_if([this](std::unique_ptr<ExprGroupConnections>& edge) {
    return this->clean_up_edges.find(edge.get()) != this->clean_up_edges.end();
  });

  groups.remove_if([this](std::unique_ptr<ExprGroup>& group) {
    return this->clean_up_groups.find(group.get()) !=
        this->clean_up_groups.end();
  });

  clean_up_edges.clear();
  clean_up_groups.clear();
}

bool ExprSegmentationSorter::supportedMerge(ExprGroup* sg1, ExprGroup* sg2) {
  auto domain1 = sg1->payload()->ca_domains;
  auto domain2 = sg2->payload()->ca_domains;

  if (domain1.empty() && domain2.empty()) {
    return true;
  }

  if (domain1.empty() || domain2.empty()) {
    return false;
  }

  return GpuLower::current()->caIndexMap().areMapped(
      domain1.back(), domain2.back());
}

void ExprSegmentationSorter::sort() {
  // Need this for initialization of the DAG that is processed
  std::unordered_map<Expr*, ExprGroup*> expr2group;

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
          std::make_unique<ExprGroupConnections>(def_group, expr_group, inp));
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
    // If we didn't do any update, stop traversal, we're done.
    bool merged_nodes = true;
    // Merge expressions in sorted order
    while (merged_nodes) {
      // Reset stateful traversal details in ExprGroups
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
               !supportedMerge(group.get(), *candidate_it)) {
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

      // Move compute at axes left
      inter_iter_update = interIterUpdate();
    }
  }
}

std::ostream& operator<<(std::ostream& os, const ExprSegmentationSorter* scf) {
  return os << scf->toString();
}

} // namespace

std::vector<Expr*> reorderExprsTest() {
  auto fusion = FusionGuard::getCurFusion();
  TORCH_INTERNAL_ASSERT(fusion != nullptr);
  ExprSegmentationSorter sorter(fusion);
  sorter.sort();
  auto groups = sorter.getGroups();
  TORCH_INTERNAL_ASSERT(
      groups.size() > 0,
      "Error during expression sorting, no expressions produced.");

  // We could have multiple groups if they're disjoint. Simply flatten them in
  // order as they could be in any order.
  std::vector<Expr*> exprs;
  for (auto group : groups) {
    exprs.insert(exprs.end(), group->exprs_.begin(), group->exprs_.end());
  }
  return exprs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
