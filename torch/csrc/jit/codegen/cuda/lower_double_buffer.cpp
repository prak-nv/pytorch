#include <torch/csrc/jit/codegen/cuda/lower_double_buffer.h>
#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

kir::Val* newResult(kir::IrBuilder& ir_builder, DataType dtype) {
  switch (dtype) {
    case DataType::Bool:
      return ir_builder.create<kir::Bool>(c10::nullopt);
    case DataType::Double:
      return ir_builder.create<kir::Double>(c10::nullopt);
    case DataType::Int:
      return ir_builder.create<kir::Int>(c10::nullopt);
    default:
      TORCH_CHECK(false, "Unexpected data type");
  }
}

class Replace : private kir::MutableIrVisitor {
 public:
  Replace(kir::Node* old_node, const kir::Val* old_val, kir::Val* new_val)
      : old_node_(old_node),
        old_val_(old_val),
        new_val_(new_val),
        ir_builder_(GpuLower::current()->kernel()) {}

  kir::Node* operator()() {
    old_node_->accept(this);
    return getReplacement(old_node_);
  }

 private:
  void registerReplacement(const kir::Node* old_node, kir::Node* new_node) {
    TORCH_INTERNAL_ASSERT(
        replacement_map_.find(old_node) == replacement_map_.end(),
        "Mapping already exists for :",
        kir::toString(old_node));
    replacement_map_.insert(std::make_pair(old_node, new_node));
  }

  template <typename T>
  T* getReplacement(const T* old_node) const {
    TORCH_INTERNAL_ASSERT(
        replacement_map_.find(old_node) != replacement_map_.end(),
        "Mapping not found: ",
        kir::toString(old_node));
    return replacement_map_.at(old_node)->template as<T>();
  }

  bool isReplaced(const kir::Node* node) const {
    return replacement_map_.find(node) != replacement_map_.end();
  }

  void visit(kir::NamedScalar* node) override {
    if (isReplaced(node)) {
      return;
    }
    registerReplacement(node, node);
  }

  void visit(kir::Int* node) override {
    if (isReplaced(node)) {
      return;
    }
    kir::Int* new_int = nullptr;
    if (static_cast<const kir::Val*>(node) == old_val_) {
      new_int = new_val_->as<kir::Int>();
    } else if (node->definition()) {
      auto def = node->definition();
      def->accept(this);
      new_int = getReplacement(def)->outputs()[0]->as<kir::Int>();
    } else {
      new_int = node;
    }
    registerReplacement(node, new_int);
  }

  void visit(kir::Bool* node) override {
    if (isReplaced(node)) {
      return;
    }
    kir::Bool* new_bool = nullptr;
    if (static_cast<const kir::Val*>(node) == old_val_) {
      new_bool = new_val_->as<kir::Bool>();
    } else if (node->definition()) {
      auto def = node->definition();
      def->accept(this);
      new_bool = getReplacement(def)->outputs()[0]->as<kir::Bool>();
    } else {
      new_bool = node;
    }
    registerReplacement(node, new_bool);
  }

  void visit(kir::Double* node) override {
    if (isReplaced(node)) {
      return;
    }
    kir::Double* new_val = nullptr;
    if (static_cast<const kir::Val*>(node) == old_val_) {
      new_val = new_val_->as<kir::Double>();
    } else if (node->definition()) {
      auto def = node->definition();
      def->accept(this);
      new_val = getReplacement(def)->outputs()[0]->as<kir::Double>();
    } else {
      new_val = node;
    }
    registerReplacement(node, new_val);
  }

  void visit(kir::UnaryOp* node) override {
    node->in()->accept(this);
    auto in_new = getReplacement(node->in());
    kir::Val* out_new = nullptr;
    if (node->out()->isScalar()) {
      out_new = newResult(ir_builder_, node->out()->dtype());
    } else {
      node->out()->accept(this);
      out_new = getReplacement(node->out());
    }
    auto new_node =
        ir_builder_.create<kir::UnaryOp>(node->operation(), out_new, in_new);
    registerReplacement(node, new_node);
  }

  void visit(kir::BinaryOp* node) override {
    node->lhs()->accept(this);
    auto lhs_new = getReplacement(node->lhs());
    node->rhs()->accept(this);
    auto rhs_new = getReplacement(node->rhs());
    kir::Val* out_new = nullptr;
    if (node->out()->isScalar()) {
      out_new = newResult(ir_builder_, node->out()->dtype());
    } else {
      node->out()->accept(this);
      out_new = getReplacement(node->out());
    }
    auto new_node = ir_builder_.create<kir::BinaryOp>(
        node->operation(), out_new, lhs_new, rhs_new);
    registerReplacement(node, new_node);
  }

  void visit(kir::TensorIndex* ti) override {
    if (isReplaced(ti)) {
      return;
    }
    kir::TensorIndex* new_ti = nullptr;
    if (static_cast<const kir::TensorIndex*>(ti) == old_val_) {
      new_ti = new_val_->as<kir::TensorIndex>();
    } else {
      std::vector<kir::Val*> new_indices;
      for (auto index : ti->indices()) {
        index->accept(this);
        auto copied_index = getReplacement(index);
        TORCH_INTERNAL_ASSERT(copied_index != nullptr);
        new_indices.push_back(copied_index);
      }
      new_ti = ir_builder_.create<kir::TensorIndex>(
          ti->view()->fuserTv(), new_indices);
    }
    registerReplacement(ti, new_ti);
  }

  void visit(kir::ForLoop* fl) override {
    fl->index()->accept(this);
    auto index_copy = getReplacement(fl->index());
    // Replacement of IterDomain not supported
    auto id_copy = fl->iter_domain();
    auto new_fl =
        ir_builder_.create<kir::ForLoop>(index_copy, id_copy, active_scope_);
    auto cur_scope = active_scope_;
    active_scope_ = new_fl;
    for (auto body_expr : fl->body().exprs()) {
      body_expr->accept(this);
      new_fl->body().push_back(getReplacement(body_expr));
    }
    active_scope_ = cur_scope;
    registerReplacement(fl, new_fl);
  }

  void visit(kir::IfThenElse* ite) override {
    ite->cond()->accept(this);
    auto new_ite = ir_builder_.create<kir::IfThenElse>(
        getReplacement(ite->cond()), active_scope_);
    auto cur_scope = active_scope_;
    active_scope_ = new_ite;
    for (auto body_expr : ite->thenBody().exprs()) {
      body_expr->accept(this);
      new_ite->thenBody().push_back(getReplacement(body_expr));
    }
    for (auto body_expr : ite->elseBody().exprs()) {
      new_ite->elseBody().push_back(getReplacement(body_expr));
    }
    active_scope_ = cur_scope;
    registerReplacement(ite, new_ite);
  }

 private:
  kir::Node* old_node_ = nullptr;
  const kir::Val* old_val_ = nullptr;
  kir::Val* new_val_ = nullptr;
  kir::IrBuilder ir_builder_;
  kir::Expr* active_scope_ = nullptr;
  std::unordered_map<const kir::Node*, kir::Node*> replacement_map_;
};

struct TensorIndexInfo {
  kir::TensorIndex* ti = nullptr;
  kir::Expr* expr = nullptr;
  kir::Scope* scope = nullptr;
};

struct BufferInfo {
  kir::TensorView* tv = nullptr;
  kir::Allocate* alloc = nullptr;
  kir::ForLoop* alloc_scope = nullptr;
  kir::UnaryOp* load = nullptr;
  kir::IfThenElse* load_predicate = nullptr;
  std::vector<TensorIndexInfo> uses;
};

class DoubleBuffering : private kir::MutableIrVisitor {
 public:
  DoubleBuffering() : ir_builder_(GpuLower::current()->kernel()) {}

  void validateDoubleBufferingUsage() {}

  void apply(const std::vector<kir::Expr*>& exprs) {
    lowered_exprs_ = exprs;
    for (auto expr : lowered_exprs_) {
      expr->accept(this);
    }
    for (auto kv : buffer_info_map_) {
      auto info = kv.second;
      updateBufferOffset(info);
      moveAndExpandAllocate(info);
      insertInitialLoad(info);
      advanceLoadOffset(info);
    }
  }

  const std::vector<kir::Expr*>& loweredExprs() const {
    return lowered_exprs_;
  }

 private:
  void visit(kir::Allocate* node) override {
    kir::TensorView* tv = dynamic_cast<kir::TensorView*>(node->buffer());
    if (tv == nullptr) {
      return;
    }
    auto fuser_tv = tv->fuserTv();
    if (fuser_tv == nullptr || !fuser_tv->isDoubleBuffered()) {
      return;
    }
    kir::ForLoop* alloc_scope = active_scope_expr_->as<kir::ForLoop>();
    BufferInfo info{tv, node, alloc_scope};
    buffer_info_map_.insert({tv, info});
  }

  void visit(kir::UnaryOp* node) override {
    active_arith_expr_ = node;
    node->out()->accept(this);
    node->in()->accept(this);
    active_arith_expr_ = nullptr;
  }

  void visit(kir::BinaryOp* node) override {
    active_arith_expr_ = node;
    node->out()->accept(this);
    node->lhs()->accept(this);
    node->rhs()->accept(this);
    active_arith_expr_ = nullptr;
  }

  void visit(kir::TernaryOp* node) override {
    active_arith_expr_ = node;
    node->out()->accept(this);
    node->in1()->accept(this);
    node->in2()->accept(this);
    node->in3()->accept(this);
    active_arith_expr_ = nullptr;
  }

  void visit(kir::TensorIndex* ti) override {
    auto tv = ti->view();
    if (tv->fuserTv() == nullptr || !tv->fuserTv()->isDoubleBuffered()) {
      return;
    }

    validateDoubleBufferingUsage(ti);

    BufferInfo& info = buffer_info_map_.at(tv);
    TORCH_INTERNAL_ASSERT(active_arith_expr_ != nullptr);

    // Uses
    info.uses.push_back(
        TensorIndexInfo{ti, active_arith_expr_, active_scope_.back()});

    // Load
    if (active_arith_expr_->outputs()[0] == ti) {
      auto uop = dynamic_cast<kir::UnaryOp*>(active_arith_expr_);
      TORCH_INTERNAL_ASSERT(uop != nullptr);
      TORCH_INTERNAL_ASSERT(uop->operation() == UnaryOpType::Set);
      info.load = uop;
      TORCH_INTERNAL_ASSERT(
          active_scope_expr_->isA<kir::IfThenElse>(), "Predicate not found");
      info.load_predicate = active_scope_expr_->as<kir::IfThenElse>();
    }
  }

  void validateDoubleBufferingUsage(kir::TensorIndex* ti) const {
    const auto tv = ti->view();
    const auto def = tv->definition();

    TORCH_CHECK(def->isA<kir::UnaryOp>());
    TORCH_CHECK(def->as<kir::UnaryOp>()->operation() == UnaryOpType::Set);

    TORCH_CHECK(def->as<kir::UnaryOp>()->in()->isA<kir::TensorView>());
    const auto in = def->as<kir::UnaryOp>()->in()->as<kir::TensorView>();

    TORCH_CHECK(
        in->memoryType() == MemoryType::Global ||
        in->memoryType() == MemoryType::Shared);

    TORCH_CHECK(tv->fuserTv()->getThisComputeAtAxis() > 0);
  }

  void moveAndExpandAllocate(const BufferInfo& info) {
    const auto alloc = info.alloc;
    // Double the size
    auto size =
        ir_builder_.mulExpr(alloc->size(), ir_builder_.create<kir::Int>(2));
    // Create a new allocate expr with the expanded size
    auto expanded_alloc = ir_builder_.create<kir::Allocate>(
        info.tv, alloc->memoryType(), size, alloc->zeroInit());

    TORCH_INTERNAL_ASSERT(info.alloc_scope);
    // Insert the new alloc expr
    insertBeforeScope(expanded_alloc, info.alloc_scope);
    // Remove the existing one
    removeExpr(info.alloc, info.alloc_scope);
  }

  void insertInitialLoad(const BufferInfo& info) {
    auto load_copy = copyLoad(info);
    // Replace the loop index with zero
    auto initial_load = Replace(
                            load_copy,
                            info.alloc_scope->index(),
                            ir_builder_.create<kir::Int>(0))()
                            ->as<kir::Expr>();
    insertBeforeScope(initial_load, info.alloc_scope);
  }

  void updateBufferOffset(BufferInfo& info) {
    TORCH_INTERNAL_ASSERT(info.load != nullptr);
    kir::Allocate* allocate = info.alloc;
    kir::Val* size = allocate->size();
    kir::ForLoop* alloc_fl = info.alloc_scope;
    kir::Val* buffer_switch = newResult(ir_builder_, DataType::Int);
    ir_builder_.create<kir::BinaryOp>(
        BinaryOpType::And,
        buffer_switch,
        alloc_fl->index(),
        ir_builder_.create<kir::Int>(1));
    auto buffer_offset = ir_builder_.mulExpr(size, buffer_switch);

    for (const auto& ti_info : info.uses) {
      kir::TensorIndex* old_ti = ti_info.ti;
      kir::Expr* old_expr = ti_info.expr;
      auto old_index = old_ti->indices().back();
      auto new_index = ir_builder_.addExpr(old_index, buffer_offset);
      auto new_ti =
          Replace(old_ti, old_index, new_index)()->as<kir::TensorIndex>();
      auto new_expr = Replace(old_expr, old_ti, new_ti)()->as<kir::Expr>();
      replaceExpr(old_expr, new_expr, ti_info.scope);
      if (old_expr == info.load) {
        info.load = new_expr->as<kir::UnaryOp>();
      }
    }
  }

  kir::Expr* copyLoad(const BufferInfo& info) {
    kir::Expr* cur_scope_expr = info.load_predicate;
    kir::Expr* expr_copy = ir_builder_.create<kir::UnaryOp>(
        info.load->operation(), info.load->out(), info.load->in());
    while (cur_scope_expr != info.alloc_scope) {
      if (auto fl = dynamic_cast<kir::ForLoop*>(cur_scope_expr)) {
        auto fl_copy = ir_builder_.create<kir::ForLoop>(
            fl->index(), fl->iter_domain(), nullptr);
        fl_copy->body().insert(0, expr_copy);
        expr_copy = fl_copy;
      } else if (auto ite = dynamic_cast<kir::IfThenElse*>(cur_scope_expr)) {
        auto ite_copy =
            ir_builder_.create<kir::IfThenElse>(ite->cond(), nullptr);
        ite_copy->thenBody().insert(0, expr_copy);
        expr_copy = ite_copy;
      }
      cur_scope_expr = cur_scope_expr->parentScope();
    }
    return expr_copy;
  }

  void advanceLoadOffset(const BufferInfo& info) {
    auto idx = info.alloc_scope->index();
    auto idx_next = ir_builder_.addExpr(idx, ir_builder_.create<kir::Int>(1));
    auto new_predicated_load =
        Replace(info.load_predicate, idx, idx_next)()->as<kir::IfThenElse>();
    replaceScopeExpr(info.load_predicate, new_predicated_load);
  }

  void replaceScopeExpr(kir::Expr* old_expr, kir::Expr* new_expr) {
    insertBeforeScope(new_expr, old_expr);
    removeExpr(old_expr, old_expr->parentScope());
  }

  void insertBeforeScope(kir::Expr* expr, kir::Expr* scope_expr) {
    TORCH_INTERNAL_ASSERT(
        scope_expr->isA<kir::ForLoop>() || scope_expr->isA<kir::IfThenElse>());
    kir::Expr* outer_scope = scope_expr->parentScope();
    if (outer_scope == nullptr) {
      auto pos =
          std::find(lowered_exprs_.begin(), lowered_exprs_.end(), scope_expr);
      TORCH_INTERNAL_ASSERT(pos != lowered_exprs_.end());
      lowered_exprs_.insert(pos, expr);
    } else if (auto fl = dynamic_cast<kir::ForLoop*>(outer_scope)) {
      fl->body().insert_before(scope_expr, expr);
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(outer_scope)) {
      if (ite->thenBody().contains(expr)) {
        ite->thenBody().insert_before(scope_expr, expr);
      } else {
        TORCH_INTERNAL_ASSERT(ite->elseBody().contains(expr));
        ite->elseBody().insert_before(scope_expr, expr);
      }
    } else {
      TORCH_INTERNAL_ASSERT(false);
    }
  }

  void removeExpr(kir::Expr* expr, kir::Expr* scope_expr) {
    if (scope_expr == nullptr) {
      auto it = std::find(lowered_exprs_.begin(), lowered_exprs_.end(), expr);
      TORCH_INTERNAL_ASSERT(it != lowered_exprs_.end());
      lowered_exprs_.erase(it);
    } else if (auto fl = dynamic_cast<kir::ForLoop*>(scope_expr)) {
      kir::Scope& scope = fl->body();
      TORCH_INTERNAL_ASSERT(scope.contains(expr));
      scope.erase(expr);
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(scope_expr)) {
      if (ite->thenBody().contains(expr)) {
        ite->thenBody().erase(expr);
      } else {
        TORCH_INTERNAL_ASSERT(ite->elseBody().contains(expr));
        ite->elseBody().erase(expr);
      }
    } else {
      TORCH_INTERNAL_ASSERT(false);
    }
  }

  void replaceExpr(kir::Expr* old_expr, kir::Expr* new_expr, kir::Scope* scope)
      const {
    scope->insert_before(old_expr, new_expr);
    scope->erase(old_expr);
  }

  void visit(kir::ForLoop* for_loop) override {
    const auto prev_scope_expr = active_scope_expr_;

    active_scope_expr_ = for_loop;
    active_scope_.push_back(&for_loop->body());

    for (auto expr : for_loop->body().exprs()) {
      expr->accept(this);
    }

    active_scope_.pop_back();
    active_scope_expr_ = prev_scope_expr;
  }

  void visit(kir::IfThenElse* ite) override {
    const auto prev_scope_expr = active_scope_expr_;

    active_scope_expr_ = ite;

    active_scope_.push_back(&ite->thenBody());
    for (auto expr : ite->thenBody().exprs()) {
      expr->accept(this);
    }
    active_scope_.pop_back();
    active_scope_.push_back(&ite->elseBody());
    for (auto expr : ite->elseBody().exprs()) {
      expr->accept(this);
    }

    active_scope_.pop_back();
    active_scope_expr_ = prev_scope_expr;
  }

 private:
  std::vector<kir::Expr*> lowered_exprs_;

  kir::IrBuilder ir_builder_;
  std::deque<kir::Scope*> active_scope_;
  kir::Expr* active_scope_expr_ = nullptr;
  kir::Expr* active_arith_expr_ = nullptr;
  std::unordered_map<const kir::TensorView*, BufferInfo> buffer_info_map_;
};

} // namespace

std::vector<kir::Expr*> applyDoubleBuffering(
    const std::vector<kir::Expr*>& indexed_loops) {
  DoubleBuffering double_buffering;
  double_buffering.apply(indexed_loops);
  return double_buffering.loweredExprs();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
