#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <deque>
#include <list>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// TODO: Clean up deque, use vector when possible
// TODO: Review const model, and objects
// TODO: Rename,
//  segment -> fusion_segmenter.cpp/.h
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

// Wrapper for values values, edges between segmented groups which are made up
// of Exprs. Multiple edges can exist between segmented groups.
class SegmentedEdge {
 public:
  SegmentedEdge(SegmentedGroup* from, SegmentedGroup* to, Val* val)
      : from_(from), to_(to), val_(val) {}
  SegmentedGroup* from_;
  SegmentedGroup* to_;
  Val* val_;
};

std::ostream& operator<<(std::ostream& os, const SegmentedEdge* edge);

class TraversalPayload : public PolymorphicBase {
 public:
  // Maximum path distance from an input segmented group required for
  // Theorem 4.2
  int level = -1;

  // traversal marker, has this node already been processed
  bool visited = false;

  // Did we select another group to merge with
  SegmentedGroup* merge_with = nullptr;

  // Has this node been merged?
  bool merged = false;
};

// Groups together expressions which create a segmented group
class SegmentedGroup {
 public:
  explicit SegmentedGroup(
      std::unique_ptr<TraversalPayload>&& _payload =
          std::make_unique<TraversalPayload>())
      : payload_(std::move(_payload)) {}

  SegmentedGroup(Expr* expr) : payload_(std::make_unique<TraversalPayload>()) {
    exprs_.push_back(expr);
  }

  SegmentedGroup(const SegmentedGroup& other)
      : payload_(new TraversalPayload(*(other.payload_))) {}

  SegmentedGroup& operator=(const SegmentedGroup& other) {
    *payload_ = *other.payload_;
    exprs_ = other.exprs_;
    return *this;
  }

  void clearTraversalInfo();

  // TODO: May want to sort this based on size of connections between this and
  // neighbors as well as if the connection is an output of the fusion (has to
  // be saved to gmem anyways)
  std::vector<SegmentedGroup*> getNeighbors();

  // Look at all neighbors of this and return who this could merge with based on
  // level values of this, neighbors, and merged neighbors of neighbors
  std::vector<SegmentedGroup*> getMergeCandidates();

  // Doesn't have any producer edges mapped to an Expr, they're all inputs of
  // the original fusion.
  bool isInputGroup();

  std::unique_ptr<TraversalPayload>& payload() {
    return payload_;
  }

 public:
  // "Ancestor nodes", towards inputs of segmentedDAG
  std::vector<SegmentedEdge*> producer_edges;

  // "Descendent nodes", towards outputs of segmentedDAG
  std::vector<SegmentedEdge*> consumer_edges;

  std::vector<Val*> input_vals;
  std::vector<Val*> output_vals;

  // Exprs that make up the group
  std::vector<Expr*> exprs_;

  // ==== Stateful traversal information below ====
  std::unique_ptr<TraversalPayload> payload_;
};

std::ostream& operator<<(std::ostream& os, const SegmentedGroup* group);

class TORCH_CUDA_API SegmentCandidateFinder {
 public:
  // Take a copy of fusion to own, it will get reused and copies sent to
  // schedulers.
  SegmentCandidateFinder(Fusion* fusion);

  void segment();

  std::string toString(int verbosity = 0) const;

  const std::vector<SegmentedGroup*> getGroups() {
    std::vector<SegmentedGroup*> group_vec;
    std::transform(
        groups.begin(),
        groups.end(),
        std::back_inserter(group_vec),
        [](std::unique_ptr<SegmentedGroup>& sg) { return sg.get(); });
    return group_vec;
  }

 protected:
  // For payload overload
  virtual SegmentedGroup* makeEmptyGroup();

  // For payload overload
  virtual SegmentedGroup* makeEmptyGroup(Expr*);

  // Mechanism by which we decide if we support a given fusion of nodes, meaning
  // sg1, and sg2 will be segmented together.
  virtual bool codeGenSupportedMerge(SegmentedGroup* sg1, SegmentedGroup* sg2);

  virtual SegmentedGroup* makeMergedNode(
      SegmentedGroup* sg1,
      SegmentedGroup* sg2);

  // Return true if we want to run more iterations of the segmentation after
  // this function is called. It's good if we want to segment, process, then
  // segment more (used in lower_expr_sort).
  virtual bool interIterUpdate() {
    return false;
  };

 private:
  // Reset the TraversalPayload of the groups
  void resetTraversal();

  // Reset the set levels which the analysis of if we can fuse nodes together
  // but maintain the graph is a DAG
  void resetLevels();

  // Go through groups which should me marked with other nodes to merge with,
  // and merges them.
  void mergeNodes();

  // Disconnect the edges connecting group to the rest of the graph, and return
  // all the edges that were disconnected
  std::unordered_set<SegmentedEdge*> disconnectGroup(SegmentedGroup* group);

 protected:
  // Lifetime of the graph view of the fusion and segmentation
  std::list<std::unique_ptr<SegmentedEdge>> edges;
  std::list<std::unique_ptr<SegmentedGroup>> groups;

  std::deque<SegmentedGroup*> to_visit;
  std::vector<SegmentedGroup*> next_to_visit;

  std::unordered_set<SegmentedGroup*> clean_up_groups;
  std::unordered_set<SegmentedEdge*> clean_up_edges;

  std::unordered_set<SegmentedGroup*> to_merge;

  // Maintain my own fusion the state of which is not always the same as the
  // original provided fusion.
  Fusion* complete_fusion;
};

std::ostream& operator<<(std::ostream& os, const SegmentCandidateFinder* scf);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch