#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {

// Inserts syncs in for loops that use smem so that the next iteration of the
// loop doesn't overwrite the smem before it has all been read
class TORCH_CUDA_API SyncInserter : public OptOutDispatch {
 public:
  static void InsertSyncs(Fusion* fusion, std::vector<Expr*> incoming_exprs) {
    FusionGuard fg(fusion);
    SyncInserter si;
    si.modify(incoming_exprs);
  }

 private:
  using OptOutDispatch::handle;

  // Open the for loop.
  void handle(kir::ForLoop*) final;

  void handle(kir::IfThenElse*) final;

  void handle(Expr*) final;

  void modify(const std::vector<Expr*>& exprs);

 private:
  // Fusion pointer for convenience
  Fusion* fusion_;
  bool needs_sync = false;
  kir::ForLoop* active_scope = nullptr;
};

} // namespace fuser
} // namespace jit
} // namespace torch