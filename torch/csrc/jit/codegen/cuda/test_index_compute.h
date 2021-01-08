#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TestReplay : public IterVisitor {
 private:
  TestReplay(
      const std::vector<kir::ForLoop*>& loop_structure,
      const ComputeAtMap& ca_maps)
      : loop_structure_(loop_structure), ca_maps_(ca_maps) {}

  void handle(Expr* e) override {
    TORCH_INTERNAL_ASSERT(
        e->isA<Split>() || e->isA<Merge>(),
        "This class doesn't support expressions other than merge and split.");
    IterVisitor::handle(e);
  }

  // We're going to replay this split operation on the corresponding ID
  void handle(Split* s) override;

  // We're going to replay this merge operation on the corresponding IDs
  void handle(Merge* m) override;

  TensorDomain* computeReplay();

 private:
  const std::vector<kir::ForLoop*>& loop_structure_;
  const ComputeAtMap& ca_maps_;

  // Replay map
  std::unordered_map<IterDomain*, IterDomain*> concrete_to_id;

 public:
  static TensorDomain* getReference(
      const std::vector<kir::ForLoop*>& loop_structure,
      const ComputeAtMap& ca_maps) {
    auto replay = TestReplay(loop_structure, ca_maps);
    return replay.computeReplay();
  }
};

IndexCompute getReferenceIndexing(
    const std::vector<kir::ForLoop*>& loop_structure,
    const ComputeAtMap& ca_maps,
    TensorDomain* reference_domain,
    std::unordered_map<kir::IterDomain*, kir::Val*> index_map,
    std::unordered_set<IterDomain*> preferred_path);

// Short cut for global TVs
IndexCompute getReferenceIndexing(
    const std::vector<kir::ForLoop*>& loop_structure,
    const ComputeAtMap& ca_maps,
    TensorDomain* reference_domain);

// When indexing there are sometimes an option to propagate an index down
// multiple paths. This will return the IterDomains in the history of the
// reference domain and mark which paths should be taken (if there's a
// preference) to reach the roots provided in preferred_roots.
std::unordered_set<IterDomain*> buildPreferredPaths(
    TensorDomain* reference_domain,
    std::unordered_set<IterDomain*> preferred_roots);

// TODO: should be a const IrVisitor
class TestIndexing : private kir::IrVisitor {
 public:
  static std::vector<kir::Expr*> getIndexedExprs(
      std::vector<kir::Expr*> incoming_exprs) {
    FUSER_PERF_SCOPE("TestIndexing::getIndexedExprs");
    TestIndexing il;
    il.generate(incoming_exprs);
    return incoming_exprs;
  }

 private:
  explicit TestIndexing();

  void handle(kir::Expr*);

  void visit(kir::ForLoop*) final;
  void visit(kir::IfThenElse*) final;
  // void visit( kir::UnaryOp*) final;
  // void visit(kir::BinaryOp*) final;
  // void visit( kir::TernaryOp*) final;
  // void visit( kir::ReductionOp*) final;
  // void visit( kir::BroadcastOp*) final;
  // void visit( kir::Allocate*) final;
  // void visit( kir::Sync*) final;

  void generate(std::vector<kir::Expr*>& exprs);

 private:
  std::vector<kir::ForLoop*> for_loops;

  GpuLower* gpu_lower;

  kir::IrBuilder ir_builder;

  const ComputeAtMap& ca_maps_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
