#pragma once

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>
#include <torch/csrc/jit/codegen/cuda/scheduler_registry.h>

#include <deque>
#include <list>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//  FusionSegmentFinder
//    Responsible for going through DAG and proposing things we could try to
//    fuse together, calls "canGenerateCode" on these proposed segments to see
//    if they are valid and we can generate code for them.
//  FusionSegment
//    A group of exprs that are segmented together
//  FusionSegmentConnections
//    Holds vals and what they connect. In other words it's a val that is an
//    output of a FusionSegment "from" and an input of FusionSegment "to".
//    There's nothing preventing from a val being between segments twice.
//    TODO: make sure there's nothing wrong with segmentation on nodes that
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

class SegmentedGroup;
class SegmentCandidateFinder;

// A directed edge on DAG,
// Wrapper for values, edges between segmented groups which are made up
// of Exprs. Multiple edges can exist between segmented groups.
class SegmentedEdge {
 public:
  SegmentedEdge(SegmentedGroup* from, SegmentedGroup* to, Val* val)
      : from_(from), to_(to), val_(val) {}
  SegmentedGroup* from_;
  SegmentedGroup* to_;
  Val* val_;

  void print();
};

std::ostream& operator<<(std::ostream& os, const SegmentedEdge* edge);

// Groups together expressions which create a segmented group
// Can be used to produce fusions
class TORCH_CUDA_API SegmentedGroup {
 public:
  SegmentedGroup() = default;

  SegmentedGroup(Expr* expr) {
    exprs_.push_back(expr);
  }

  void clearTraversalInfo();

  std::vector<SegmentedGroup*> getNeighbors();

  // TODO: May want to sort this based on size of connections between this and
  // neighbors as well as if the connection is an output of the fusion (has to
  // be saved to gmem anyways)
  std::vector<std::pair<SegmentedGroup*, SegmentedEdge*>> getNeighborsPair();

  // Look at all neighbors of this and return who this could merge with based on
  // level values of this, neighbors, and merged neighbors of neighbors
  std::vector<std::pair<SegmentedGroup*, SegmentedEdge*>> getMergeCandidates();

  bool isInputGroup() {
    return !input_vals.empty();
  };

  bool isConnected() const {
    return !producer_edges.empty() || !consumer_edges.empty();
  }

  int groupId() const {
    return group_id_;
  }

  // Consider Passkey or friend for safety
  void setID(int id) {
    TORCH_INTERNAL_ASSERT(group_id_ == -1);
    group_id_ = id;
  }

  std::vector<Val*> inputs() {
    return input_vals;
  }

  std::vector<Val*> outputs() {
    return output_vals;
  }

  // might just set heuristic as public
  ScheduleHeuristic heuristic() {
    return heuristic_;
  }

  void setHeuristic(ScheduleHeuristic sh) {
    heuristic_ = sh;
  }

  void print();

  //! Generate the use info for groups to avoid multi-use in
  //!  a single group, expect to remove in the future so
  //!  keeping this part separate
  void updateUse();

  //! Check if an edge is used internally within a group
  bool hasUse(Val* se);

  //! To be called at the very end of segment fusion
  //!  no more segment merging should be done beyond
  void finalize();

 public:
  // "Ancestor nodes", towards inputs of segmentedDAG
  std::vector<SegmentedEdge*> producer_edges;

  // "Descendent nodes", towards outputs of segmentedDAG
  std::vector<SegmentedEdge*> consumer_edges;

  std::vector<Val*> input_vals;
  std::vector<Val*> output_vals;

  // Exprs that make up the group
  std::vector<Expr*> exprs_;

  // Doesn't have any producer edges mapped to an Expr, they're all inputs of
  // the original fusion.

  // ==== Stateful traversal information below ====

  // Maximum path distance from an input segmented group required for
  // Theorem 4.2
  int level = -1;

  // traversal marker, has this node already been processed
  bool visited = false;

  // Did we select another group to merge with
  SegmentedGroup* merge_with = nullptr;

  // if we selected another group to merge, which edge is to be contracted
  SegmentedEdge* merge_through = nullptr;

  // Has this node been merged?
  bool merged = false;

 private:
  friend SegmentCandidateFinder;
  // unique identifier in group
  int group_id_ = -1;
  ScheduleHeuristic heuristic_ = ScheduleHeuristic::PointWise;
  std::unordered_set<Val*> internal_use;

 private:
  std::vector<Val*> edgesToVals(const std::vector<SegmentedEdge*>& se_v);
};

std::ostream& operator<<(std::ostream& os, const SegmentedGroup* group);

//! Auxiliary class for managing a list of heuristics instances for the
//!  Segmented Groups
class TORCH_CUDA_API SegmentHeuristics {
  using SchedulerEntryPtr = std::unique_ptr<SchedulerEntry>;

 public:
  explicit SegmentHeuristics() = default;
  void emplace_back(SchedulerEntryPtr&& pt) {
    heuristics_.emplace_back(std::move(pt));
  }

  const std::vector<SchedulerEntryPtr>& heuristics() const {
    return heuristics_;
  }

 private:
  std::vector<SchedulerEntryPtr> heuristics_;
};

//! Exported Interface for representing segmented fusion graph
//!   this class owns the segmented groups
class TORCH_CUDA_API SegmentedFusion {
 public:
  explicit SegmentedFusion(const Fusion* fusion);

  // Is the fusion segmented?
  bool isSegmented() {
    return !groups_.empty();
  }

  std::vector<SegmentedGroup*>& groups() {
    return groups_;
  }

  std::vector<SegmentedEdge*>& edges() {
    return edges_;
  }

  // Returns the original un-segmented fusion
  Fusion& completeFusion() {
    return fusion_;
  }

  // short cuts for accessing complete fusion
  const auto& inputs() const {
    return fusion_.inputs();
  }

  const auto& outputs() const {
    return fusion_.outputs();
  }

  // Make a clone of the group and convert to fusion
  std::unique_ptr<Fusion> makeFusion(SegmentedGroup* sg);

  // Make heuristics for all groups in this segmented fusion
  std::unique_ptr<SegmentHeuristics> makeHeuristics(
      const at::ArrayRef<IValue>& inputs);

  // Inline Debug print for segmented fusion
  std::string toString(int verbosity) const;

  // Debug print for segmented fusions
  void print();

  //! API for adding groups
  SegmentedGroup* newGroup();

  //! API shortcut for adding a singleton group
  SegmentedGroup* newGroup(Expr* expr);

  //! API for adding edges
  SegmentedEdge* newEdge(SegmentedGroup* from, SegmentedGroup* to, Val* val);

 protected:
  //! original full fusion
  Fusion fusion_;

  //! Count total exprs
  size_t total_expr_count_ = 0;

  //! States representing segmentation
  std::vector<SegmentedEdge*> edges_;
  std::vector<SegmentedGroup*> groups_;

  // Owning object to explicitly manage groups and edges
  // Owning object to provide
  class Impl {
   public:
    explicit Impl(SegmentedFusion* sf) : owning_fusion_(sf) {}

    SegmentedGroup* makeGroup();
    SegmentedGroup* makeGroup(Expr*);
    SegmentedEdge* makeEdge(SegmentedGroup* from, SegmentedGroup* to, Val* val);
    void cleanUnused();

   private:
    using GroupPtr = std::unique_ptr<SegmentedGroup>;
    using EdgePtr = std::unique_ptr<SegmentedEdge>;
    std::vector<GroupPtr> groups_;
    std::vector<EdgePtr> edges_;
    SegmentedFusion* owning_fusion_;
  };
  Impl impl_;

 protected:
  friend class SegmentCandidateFinder;
  //! Make a heuristics entry for a group and parameters
  std::unique_ptr<SchedulerEntry> makeSchedulerEntry(
      SegmentedGroup* sg,
      ExpressionEvaluator& ee);

  //! Cleanup function to be call at the end of fusion
  //!  segment pass
  void finalize();
};

class TORCH_CUDA_API SegmentCandidateFinder {
 public:
  // Take a copy of fusion to own
  SegmentCandidateFinder(const Fusion* fusion);

  static std::unique_ptr<SegmentedFusion> segment(const Fusion* fusion) {
    SegmentCandidateFinder scf(fusion);
    return std::move(scf.segmented_fusion);
  }

 private:
  void resetTraversal();

  void resetLevels();

  void mergeNodes();

  bool codeGenSupportedMerge(SegmentedEdge* edge);

  void findSegments();

  std::unordered_set<SegmentedEdge*> disconnectGroup(SegmentedGroup* group);

  std::vector<SegmentedGroup*>& groups() {
    TORCH_INTERNAL_ASSERT(
        segmented_fusion != nullptr, "Segment finder not owinging any fusion");
    return segmented_fusion->groups();
  }

  std::vector<SegmentedEdge*>& edges() {
    TORCH_INTERNAL_ASSERT(
        segmented_fusion != nullptr, "Segment finder not owinging any fusion");
    return segmented_fusion->edges();
  }

  Fusion& completeFusion() {
    TORCH_INTERNAL_ASSERT(
        segmented_fusion != nullptr, "Segment finder not owinging any fusion");
    return segmented_fusion->completeFusion();
  }

  void finalize();

  // Return the resulting heuristic corresponding to the merged
  //  group built by merging the two groups connected by edge
  ScheduleHeuristic deriveHeuristic(SegmentedGroup* edge);

 protected:
  std::deque<SegmentedGroup*> to_visit;
  std::vector<SegmentedGroup*> next_to_visit;

  std::unordered_set<SegmentedGroup*> clean_up_groups;
  std::unordered_set<SegmentedEdge*> clean_up_edges;

  std::unordered_set<SegmentedGroup*> to_merge;

  std::unique_ptr<SegmentedFusion> segmented_fusion;
};

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& os,
    const SegmentedFusion* scf);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch