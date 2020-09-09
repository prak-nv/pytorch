#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <torch/csrc/jit/codegen/cuda/lower_insert_syncs.h>

namespace torch {
namespace jit {
namespace fuser {

void SyncInserter::handle(kir::ForLoop* fl) {
  bool prev_needs_sync = needs_sync;
  active_scope = fl;

  for (auto expr : fl->body().exprs()) {
    handle(expr);
  }

  if (needs_sync && !fl->iter_domain()->isThread() &&
      fl->body().exprs().back()->getExprType().value() != ExprType::Sync) {
    fl->body().push_back(new kir::Sync());
  }

  bool needs_sync = prev_needs_sync;
}

void SyncInserter::handle(kir::IfThenElse* ite) {
  for (auto expr : ite->body().exprs()) {
    handle(expr);
  }

  for (auto expr : ite->elseBody().exprs()) {
    handle(expr);
  }
}

void SyncInserter::handle(Expr* expr) {
  if (ir_utils::isTVOp(expr)) {
    for (auto inp : expr->inputs()) {
      if (ir_utils::isTV(inp)) {
        if (inp->as<TensorView>()->getMemoryType() == MemoryType::Shared) {
          needs_sync = true;
        }
      }
    }
  } else {
    OptOutDispatch::handle(expr);
  }
}

void SyncInserter::modify(const std::vector<Expr*>& exprs) {
  for (auto expr : exprs) {
    handle(expr);
  }
}

} // namespace fuser
} // namespace jit
} // namespace torch