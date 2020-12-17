#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <deque>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Segmentation selection is based on the theorem 4.2 in the paper:
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

std::ostream& operator<<(std::ostream& os, SegmentedEdge* edge);

// Groups together expressions which create a segmented group
class SegmentedGroup {
 public:
  SegmentedGroup() = default;

  SegmentedGroup(Expr* expr) {
    exprs_.push_back(expr);
  }

  void clearTraversalInfo();

  // TODO: May want to sort this based on size of connections between this and
  // neighbors as well as if the connection is an output of the fusion (has to
  // be saved to gmem anyways)
  std::deque<SegmentedGroup*> getNeighbors();

  // Look at all neighbors of this and return who this could merge with based on
  // level values of this, neighbors, and merged neighbors of neighbors
  std::deque<SegmentedGroup*> getMergeCandidates();

  // "Ancestor nodes", towards inputs of segmentedDAG
  std::deque<SegmentedEdge*> producer_edges;

  // "Descendent nodes", towards outputs of segmentedDAG
  std::deque<SegmentedEdge*> consumer_edges;

  // Exprs that make up the group
  std::deque<Expr*> exprs_;

  // ==== Stateful traversal information below ====

  bool is_input = false;

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

std::ostream& operator<<(std::ostream& os, SegmentedGroup* group);

class TORCH_CUDA_API SegmentCandidateFinder {
 public:
  // Take a copy of fusion to own, it will get reused and copies sent to
  // schedulers.
  SegmentCandidateFinder(const Fusion* fusion);

  void segment();

  virtual bool canGenerateCode(Fusion* fusion) = 0;

  std::string toString();

 private:
  void resetTraversal();

  void resetLevels();

  void mergeNodes();

  bool codeGenSupportedMerge(SegmentedGroup* sg1, SegmentedGroup* sg2);

 private:
  // Lifetime of the graph view of the fusion and segmentation
  std::list<SegmentedEdge> edges;
  std::list<SegmentedGroup> groups;

  std::deque<SegmentedGroup*> to_visit;
  std::deque<SegmentedGroup*> next_to_visit;

  std::unordered_set<SegmentedGroup*> clean_up_groups;
  std::unordered_set<SegmentedEdge*> clean_up_edges;

  std::unordered_set<SegmentedGroup*> to_merge;

  Fusion fusion_;
};

std::ostream& operator<<(std::ostream& os, SegmentCandidateFinder* scf) {
  return os << scf->toString();
}

class TORCH_CUDA_API SingleReductionKernels : public SegmentCandidateFinder {
 public:
  SingleReductionKernels(const Fusion* fusion)
      : SegmentCandidateFinder(fusion) {}

  // TODO: May be good to have this arg as a const Fusion
  virtual bool canGenerateCode(Fusion* fusion) override {
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
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch