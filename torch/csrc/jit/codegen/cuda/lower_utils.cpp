#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>

#include <algorithm>

namespace torch {
namespace jit {
namespace fuser {

namespace scope_utils {

// START SCOPE HELPER SYSTEMS
namespace {

class Loops : private OptInDispatch {
 private:
  std::deque<kir::ForLoop*> loops;
  void handle(kir::ForLoop* fl) final {
    loops.insert(loops.begin(), fl);
  }

  void handle(kir::IfThenElse* ite) final {}

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

 public:
  static std::vector<kir::ForLoop*> getLoops(Expr* scope) {
    Loops loops;
    Expr* it = scope;
    while (it != nullptr) {
      loops.handle(it);
      it = scope_utils::getParent(it);
    }
    return std::vector<kir::ForLoop*>(loops.loops.begin(), loops.loops.end());
  }
};

class forLoopCount : private OptInDispatch {
 private:
  unsigned int count_ = 0;

  void handle(kir::ForLoop* fl) final {
    count_++;
  }

  void handle(kir::IfThenElse* ite) final {}

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

 public:
  static unsigned int get(Expr* scope) {
    forLoopCount flc;
    Expr* it = scope;
    while (it != nullptr) {
      flc.handle(it);
      it = scope_utils::getParent(it);
    }
    return flc.count_;
  }
};

class scopePushBack : private OptInDispatch {
 private:
  Expr* expr_;
  void handle(kir::ForLoop* fl) final {
    fl->body().push_back(expr_);
  }

  void handle(kir::IfThenElse* ite) final {
    ite->body().push_back(expr_);
  }

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

  scopePushBack(Expr* expr) : expr_(expr) {}

 public:
  static void push(Expr* scope, Expr* expr) {
    scopePushBack pb(expr);
    TORCH_INTERNAL_ASSERT(
        expr != nullptr && scope != nullptr,
        "Cannot push back, scope or expr is a nullptr.");
    pb.handle(scope);
  }
};

class scopeInsertBefore : private OptInDispatch {
 private:
  Expr* ref_;
  Expr* expr_;
  void handle(kir::ForLoop* fl) final {
    fl->body().insert_before(ref_, expr_);
  }

  void handle(kir::IfThenElse* ite) final {
    ite->body().insert_before(ref_, expr_);
  }

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

  scopeInsertBefore(Expr* ref, Expr* expr) : ref_(ref), expr_(expr) {}

 public:
  static void insert(Expr* scope, Expr* ref, Expr* expr) {
    scopeInsertBefore scb(ref, expr);
    TORCH_INTERNAL_ASSERT(
        expr != nullptr && scope != nullptr,
        "Cannot push back, scope or expr is a nullptr.");
    scb.handle(scope);
  }
};

class parentScope : private OptInDispatch {
 private:
  Expr* parent_ = nullptr;

  void handle(kir::ForLoop* fl) final {
    parent_ = fl->parentScope();
  }

  void handle(kir::IfThenElse* ite) final {
    parent_ = ite->parentScope();
  }

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

 public:
  static Expr* get(Expr* scope) {
    parentScope sp;
    sp.handle(scope);
    return sp.parent_;
  }
};

class scopeClearExprs : private OptInDispatch {
 private:
  void handle(kir::ForLoop* fl) final {
    fl->body().clear();
  }

  void handle(kir::IfThenElse* ite) final {
    ite->body().clear();
  }

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

 public:
  static void clear(Expr* scope) {
    scopeClearExprs sce;
    TORCH_INTERNAL_ASSERT(
        scope != nullptr, "Cannot clear scope, scope is a nullptr.");
    sce.handle(scope);
  }
};

void assertScope(Expr* expr) {
  TORCH_INTERNAL_ASSERT(
      expr->getExprType() == ExprType::ForLoop ||
          expr->getExprType() == ExprType::IfThenElse,
      "Assert Scope failed when calling a scope_util function.");
}

class CloneLoopNest : public OptOutMutator {
 private:
  Expr* parent_scope_ = nullptr;
  Expr* to_clone_ = nullptr;

  Statement* mutate(kir::ForLoop* fl) final {
    std::vector<Expr*> mutated_exprs;
    for (Expr* expr : fl->body().exprs()) {
      mutated_exprs.push_back(ir_utils::asExpr(OptOutMutator::mutate(expr)));
    }
    if (fl == to_clone_)
      return new kir::ForLoop(
          fl->index(), fl->iter_domain(), mutated_exprs, parent_scope_);
    return new kir::ForLoop(
        fl->index(), fl->iter_domain(), mutated_exprs, fl->parentScope());
  }

  CloneLoopNest(Expr* _to_clone, Expr* _parent_scope)
      : parent_scope_(_parent_scope), to_clone_(_to_clone) {}

 public:
  static kir::ForLoop* getClone(kir::ForLoop* _to_clone, Expr* _parent_scope) {
    TORCH_INTERNAL_ASSERT(
        _to_clone != nullptr,
        "Tried to clone a scope, but received a nullptr.");
    CloneLoopNest cln(_to_clone, _parent_scope);
    return ir_utils::asForLoop(ir_utils::asExpr(cln.mutate(_to_clone)));
  }
};

class ReplaceExprsInScope : public OptOutDispatch {
 public:
  static void replace(
      Expr* scope,
      std::unordered_map<Expr*, Expr*> replacement_map) {
    ReplaceExprsInScope reis(std::move(replacement_map));
    reis.handle(scope);
  }

 private:
  explicit ReplaceExprsInScope(std::unordered_map<Expr*, Expr*> replacement_map)
      : replacement_map_(std::move(replacement_map)) {}

  void handleScope(kir::Scope& scope) {
    for (size_t i = 0; i < scope.size(); ++i) {
      const auto it = replacement_map_.find(scope[i]);
      if (it == replacement_map_.end()) {
        handle(scope[i]);
        continue;
      }
      scope[i] = it->second;
    }
  }

  void handle(Expr* expr) final {
    OptOutDispatch::handle(expr);
  }

  void handle(kir::ForLoop* fl) final {
    handleScope(fl->body());
  }

  void handle(kir::IfThenElse* ite) final {
    handleScope(ite->body());
    handleScope(ite->elseBody());
  }

 private:
  std::unordered_map<Expr*, Expr*> replacement_map_;
};

class FirstInnerMostScope : private OptInDispatch {
 private:
  Expr* active_scope = nullptr;

  void handle(kir::ForLoop* fl) final {
    for (auto expr : fl->body().exprs()) {
      if (ir_utils::isScope(expr)) {
        active_scope = expr;
        return;
      }
    }
    active_scope = nullptr;
  }

  void handle(kir::IfThenElse* ite) final {
    for (auto expr : ite->body().exprs()) {
      if (ir_utils::isScope(expr)) {
        active_scope = expr;
        return;
      }
    }
    for (auto expr : ite->elseBody().exprs()) {
      if (ir_utils::isScope(expr)) {
        active_scope = expr;
        return;
      }
    }
    active_scope = nullptr;
  }

  Expr* getInner(Expr* expr) {
    OptInDispatch::handle(expr);
    return active_scope;
  }

 public:
  static Expr* get(Expr* scope) {
    TORCH_INTERNAL_ASSERT(
        scope != nullptr,
        "Tried to get inner most scope, but was provided nullptr.");

    FirstInnerMostScope fims;
    Expr* inner = fims.getInner(scope);

    if (inner == nullptr)
      return scope;

    while (fims.getInner(inner) != nullptr)
      inner = fims.getInner(inner);
    return inner;
  }
};

// END SCOPE HELPER SYSTEMS
} // namespace

// Grab the ForLoop starting from scope working out
std::vector<kir::ForLoop*> getLoops(Expr* scope) {
  if (scope == nullptr)
    return std::vector<kir::ForLoop*>();
  assertScope(scope);
  return Loops::getLoops(scope);
}

// Track how far our for loop scope is
unsigned int computeForDepth(Expr* scope) {
  if (scope == nullptr)
    return 0;
  assertScope(scope);
  return forLoopCount::get(scope);
}

// Push back an expr to scope
void pushBack(Expr* scope, Expr* expr) {
  TORCH_INTERNAL_ASSERT(
      scope != nullptr, "Scope is a nullptr, cannot push an expr to it.");
  assertScope(scope);
  scopePushBack::push(scope, expr);
}

// Insert expr in scope before ref
void insertBefore(Expr* scope, Expr* ref, Expr* expr) {
  scopeInsertBefore::insert(scope, ref, expr);
}

// Return the parent of the active scope
Expr* getParent(Expr* scope) {
  TORCH_INTERNAL_ASSERT(
      scope != nullptr,
      "Tried to close the active scope, but there isn't one set.");
  assertScope(scope);
  return parentScope::get(scope);
}

// Open a new inner most for loop
kir::ForLoop* openFor(Expr* scope, IterDomain* id) {
  kir::ForLoop* new_scope = nullptr;
  if (id->isThread()) {
    std::stringstream ss;
    ss << id->getParallelType();
    new_scope = new kir::ForLoop(
        new NamedScalar(ss.str(), DataType::Int), id, {}, scope);
  } else {
    new_scope = new kir::ForLoop(new Int(), id, {}, scope);
  }
  if (scope != nullptr)
    pushBack(scope, new_scope);
  return new_scope;
}

// Close the inner most for loop
Expr* closeScope(Expr* scope) {
  TORCH_INTERNAL_ASSERT(
      scope != nullptr, "Tried to close a scope but got a nullptr.");
  return getParent(scope);
}

// Clear all expressions from the scope
Expr* clearScope(Expr* scope) {
  TORCH_INTERNAL_ASSERT(
      scope != nullptr, "Tried to clear a scope but got a nullptr.");
  assertScope(scope);
  scopeClearExprs::clear(scope);
  return scope;
}

kir::ForLoop* cloneLoopNest(kir::ForLoop* to_clone, Expr* parent_scope) {
  return CloneLoopNest::getClone(to_clone, parent_scope);
}

void replaceExprsInScope(
    Expr* scope,
    std::unordered_map<Expr*, Expr*> replacement_map) {
  TORCH_INTERNAL_ASSERT(
      replacement_map.find(scope) == replacement_map.end(),
      "Error trying to replace expressions in a scope, scope wants to be replaced entirely.");
  ReplaceExprsInScope::replace(scope, std::move(replacement_map));
}

Expr* firstInnerMostScope(Expr* scope) {
  return FirstInnerMostScope::get(scope);
}

} // namespace scope_utils

namespace ir_utils {

TVDomainGuard::TVDomainGuard(TensorView* _tv, TensorDomain* td)
    : tv_(_tv), prev_domain(tv_->domain()) {
  tv_->setDomain(td);
}

TVDomainGuard::~TVDomainGuard() {
  tv_->setDomain(prev_domain);
}

std::vector<IterDomain*> iterDomainInputsOf(
    const std::vector<IterDomain*>& input_ids) {
  auto inputs = IterVisitor::getInputsTo({input_ids.begin(), input_ids.end()});
  std::vector<IterDomain*> id_inputs(
      ir_utils::filterByType<IterDomain>(inputs).begin(),
      ir_utils::filterByType<IterDomain>(inputs).end());
  return id_inputs;
}

std::vector<IterDomain*> iterDomainInputsOfOrderedAs(
    const std::vector<IterDomain*>& of,
    const std::vector<IterDomain*>& order) {
  auto inputs_vec = iterDomainInputsOf(of);

  std::unordered_set<IterDomain*> inputs_set(
      inputs_vec.begin(), inputs_vec.end());

  std::vector<IterDomain*> ordered_inputs;
  std::copy_if(
      order.begin(),
      order.end(),
      std::back_inserter(ordered_inputs),
      [&inputs_set](const auto& id) {
        return inputs_set.find(id) != inputs_set.end();
      });

  return ordered_inputs;
}

std::vector<Val*> indices(std::vector<kir::ForLoop*> loops) {
  std::vector<Val*> inds(loops.size());
  std::transform(
      loops.begin(), loops.end(), inds.begin(), [](kir::ForLoop* fl) {
        return fl->index();
      });
  return inds;
}

std::vector<IterDomain*> iterDomains(std::vector<kir::ForLoop*> loops) {
  std::vector<IterDomain*> ids(loops.size());
  std::transform(loops.begin(), loops.end(), ids.begin(), [](kir::ForLoop* fl) {
    return fl->iter_domain();
  });
  return ids;
}

bool isTV(const Val* val) {
  return val->getValType().value() == ValType::TensorView;
}

// Check if we're a TensorView op that we can generate code for.
bool isTVOp(const Expr* expr) {
  if (expr->outputs().size() == 1 && isTV(expr->output(0)) &&
      (expr->getExprType().value() == ExprType::BinaryOp ||
       expr->getExprType().value() == ExprType::UnaryOp ||
       expr->getExprType().value() == ExprType::TernaryOp ||
       expr->getExprType().value() == ExprType::ReductionOp ||
       expr->getExprType().value() == ExprType::BroadcastOp))
    return true;
  return false;
}

TensorView* getTVOutput(const Expr* expr) {
  for (auto out : expr->outputs()) {
    if (out->getValType().value() == ValType::TensorView) {
      return out->as<TensorView>();
    }
  }
  return nullptr;
}

bool isScalarOp(const Expr* expr) {
  for (auto out : expr->outputs())
    if (!out->isScalar())
      return false;
  return true;
}

void ASSERT_EXPR(Statement* stmt) {
  TORCH_INTERNAL_ASSERT(
      stmt->isExpr(),
      "Tried to generate a kernel but hit a non expression during lowering: ",
      stmt);
}

Expr* asExpr(Statement* stmt) {
  ASSERT_EXPR(stmt);
  return stmt->as<Expr>();
}

TensorView* asTV(Val* val) {
  TORCH_INTERNAL_ASSERT(isTV(val));
  return val->as<TensorView>();
}

bool isScope(const Expr* expr) {
  return expr->getExprType() == ExprType::ForLoop ||
      expr->getExprType() == ExprType::IfThenElse;
}

kir::ForLoop* asForLoop(Statement* stmt) {
  Expr* expr = asExpr(stmt);
  TORCH_INTERNAL_ASSERT(expr->getExprType() == ExprType::ForLoop);
  return expr->as<kir::ForLoop>();
}

const TensorView* asConstTV(const Val* val) {
  TORCH_INTERNAL_ASSERT(isTV(val));
  return val->as<TensorView>();
}

bool isUnrolledFor(const Expr* expr) {
  if (expr->getExprType() != ExprType::ForLoop) {
    return false;
  }
  return expr->as<kir::ForLoop>()->iter_domain()->getParallelType() ==
      ParallelType::Unroll;
}

const std::unordered_map<ParallelType, int> ParallelTypeBitmap::pt_to_offset_{
    {ParallelType::BIDx, 0},
    {ParallelType::BIDy, 1},
    {ParallelType::BIDz, 2},
    {ParallelType::TIDx, 3},
    {ParallelType::TIDy, 4},
    {ParallelType::TIDz, 5}};

const std::unordered_map<int, ParallelType> ParallelTypeBitmap::offset_to_pt_ =
    {{0, ParallelType::BIDx},
     {1, ParallelType::BIDy},
     {2, ParallelType::BIDz},
     {3, ParallelType::TIDx},
     {4, ParallelType::TIDy},
     {5, ParallelType::TIDz}};

bool ParallelTypeBitmap::get(ParallelType pt) const {
  if (pt_to_offset_.find(pt) == pt_to_offset_.end()) {
    TORCH_INTERNAL_ASSERT(false, "Could not recognize parallel type.");
  }
  return bitset_[pt_to_offset_.at(pt)];
}

bool ParallelTypeBitmap::set(ParallelType pt, bool new_val) {
  if (pt_to_offset_.find(pt) == pt_to_offset_.end()) {
    TORCH_INTERNAL_ASSERT(false, "Could not recognize parallel type.");
  }
  bool old_val = bitset_[pt_to_offset_.at(pt)];
  bitset_[pt_to_offset_.at(pt)] = new_val;
  return old_val;
}

ParallelTypeBitmap ParallelTypeBitmap::operator&=(
    const ParallelTypeBitmap& other) {
  bitset_ &= other.bitset_;
  return *this;
}

ParallelTypeBitmap ParallelTypeBitmap::operator|=(
    const ParallelTypeBitmap& other) {
  bitset_ |= other.bitset_;
  return *this;
}

ParallelTypeBitmap ParallelTypeBitmap::operator^=(
    const ParallelTypeBitmap& other) {
  bitset_ ^= other.bitset_;
  return *this;
}

ParallelTypeBitmap ParallelTypeBitmap::operator~() const {
  return ParallelTypeBitmap(~bitset_);
}

bool ParallelTypeBitmap::none() const {
  return bitset_.none();
}

bool ParallelTypeBitmap::any() const {
  return bitset_.any();
}

bool ParallelTypeBitmap::all() const {
  return bitset_.all();
}

bool ParallelTypeBitmap::operator[](size_t pos) const {
  TORCH_INTERNAL_ASSERT(
      pos < num_p_type, "Invalid index to ParallelTypeBitset: ", pos);
  return bitset_[pos];
}

std::map<ParallelType, bool> ParallelTypeBitmap::getMap() const {
  std::map<ParallelType, bool> map;
  for (const auto& pt_offset : pt_to_offset_) {
    map.emplace(std::make_pair(pt_offset.first, bitset_[pt_offset.second]));
  }
  return map;
}

ParallelTypeBitmap operator&(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs) {
  auto x = lhs;
  x &= rhs;
  return x;
}

ParallelTypeBitmap operator|(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs) {
  auto x = lhs;
  x |= rhs;
  return x;
}

ParallelTypeBitmap operator^(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs) {
  auto x = lhs;
  x ^= rhs;
  return x;
}

ParallelTypeBitmap getParallelBroadcastDomains(
    const Val* bop_out,
    const ThreadPredicateMap& preds) {
  if (bop_out->getValType().value() == ValType::TensorIndex) {
    bop_out = bop_out->as<kir::TensorIndex>()->view();
  }
  TORCH_INTERNAL_ASSERT(
      bop_out->getValType().value() == ValType::TensorView,
      "Out is not tensor view");
  auto out_tv = bop_out->as<TensorView>();
  // If no pred is found for out_tv, no predicate is necessary
  if (preds.find(out_tv) == preds.end()) {
    return ParallelTypeBitmap();
  }
  const ParallelTypeBitmap& out_pred = preds.at(out_tv).first;

  ParallelTypeBitmap parallel_broadcast;
  const auto& iter_domains = out_tv->domain()->domain();
  for (auto id : iter_domains) {
    if (id->isBroadcast() && id->isThread()) {
      parallel_broadcast.set(id->getParallelType(), true);
    }
  }

  return parallel_broadcast & out_pred;
}

} // namespace ir_utils

namespace loop_utils {
bool loopsHasReductions(
    TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& ca_id_map) {
  // If we're initializing a reduction buffer, we won't have the reduction
  // loops. If we're actually performing the reduction, we will. Grab a root
  // dimension in tv and see if it maps to any loop, if it does we need to map
  // reductions, if not, assume we don't.

  if (!tv->hasReduction())
    return false;

  // Grab a reduction ID in the tensor, see if it maps to any loops.
  auto reduction_i = tv->getReductionAxis();

  // shouldn't be possible to hit this as isReduction should just check every
  // ID in tv->domain() for ->isReduction
  TORCH_INTERNAL_ASSERT(reduction_i.has_value());

  auto reduction_ca_id = tv->getComputeAtAxis(*reduction_i).first;
  if (ca_id_map.find(reduction_ca_id) != ca_id_map.end()) {
    reduction_ca_id = ca_id_map.at(reduction_ca_id);
  }

  // find if reduction_ca_id is an iteration domain of any of loops
  return std::any_of(
      loops.begin(), loops.end(), [reduction_ca_id](const auto& loop) {
        return loop->iter_domain() == reduction_ca_id;
      });
}

std::unordered_map<IterDomain*, kir::ForLoop*> computeAtToLoopMap(
    TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& ca_id_map) {
  // Map we're generating.
  std::unordered_map<IterDomain*, kir::ForLoop*> ca_to_loop_map;

  bool need_reduction_axes = loopsHasReductions(tv, loops, ca_id_map);

  auto loops_it = loops.begin();

  // Look at each axis individually in out's domain and find the matching loop
  for (int64_t tv_i = 0; tv_i < (int64_t)tv->nDims(); tv_i++) {
    if (!need_reduction_axes && tv->axis(tv_i)->isReduction()) {
      continue;
    }

    // Grab the axis information
    auto ca_point = tv->getComputeAtAxis(tv_i);
    auto ca_view = ca_point.second;
    auto ca_id = ca_point.first;

    // use ca_id map if we have it.
    if (!ca_id_map.empty()) {
      ca_id = ca_id_map.at(ca_id);
    }

    loops_it = std::find_if(loops_it, loops.end(), [&ca_id](const auto& loop) {
      return ca_id == loop->iter_domain();
    });

    TORCH_INTERNAL_ASSERT(
        loops_it != loops.end(),
        "Could not find all required axes for indexing.");

    ca_to_loop_map[ca_id] = *loops_it;
    ++loops_it;
  }

  return ca_to_loop_map;
}

std::unordered_map<kir::ForLoop*, IterDomain*> loopToComputeAtMap(
    TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& ca_id_map) {
  auto inverse_map = computeAtToLoopMap(tv, loops, ca_id_map);
  std::unordered_map<kir::ForLoop*, IterDomain*> map;
  for (auto entry : inverse_map) {
    map[entry.second] = entry.first;
  }
  return map;
}

std::unordered_map<IterDomain*, IterDomain*> mapIdPtoC(
    TensorView* producer,
    TensorView* consumer) {
  auto p2c_domain_ind_map = TensorDomain::mapDomainPandC(
      producer->domain()->domain(), consumer->domain()->domain());

  std::unordered_map<IterDomain*, IterDomain*> p2c_id_map;

  for (auto entry : p2c_domain_ind_map) {
    auto p_i = entry.first;
    auto c_i = entry.second;
    p2c_id_map[producer->getComputeAtAxis(p_i).first] =
        consumer->getComputeAtAxis(c_i).first;
  }

  return p2c_id_map;
}

std::vector<Val*> getIndicesForTV(
    TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    bool for_predicates,
    const std::unordered_map<IterDomain*, IterDomain*>& ca_id_map) {
  Val* zero = new Int(0);
  std::vector<Val*> indices(tv->nDims(), zero);

  bool is_shared =
      for_predicates ? false : tv->getMemoryType() == MemoryType::Shared;
  bool is_local =
      for_predicates ? false : tv->getMemoryType() == MemoryType::Local;

  // Where is this TV allocated relative to the loop nest and its own axes?
  auto alloc_point = loop_utils::getAllocPoint(tv, loops);
  auto alloc_axis = alloc_point.second;

  // Which loop is this axis associated with?
  auto ca2loop = computeAtToLoopMap(tv, loops, ca_id_map);

  bool need_reduction_axes = loopsHasReductions(tv, loops, ca_id_map);

  // Look at each axis individually in out's domain
  for (int64_t tv_i = 0; tv_i < (int64_t)tv->nDims(); tv_i++) {
    if (tv->axis(tv_i)->isReduction() && !need_reduction_axes) {
      continue;
    }

    // Check if we need to index based on this axis. If outside our allocation
    // point, we don't need to unless we're generating a predicate
    if (tv_i < alloc_axis && !for_predicates) {
      continue;
    }

    auto ca_id = tv->getComputeAtAxis(tv_i).first;
    if (!ca_id_map.empty()) {
      ca_id = ca_id_map.at(ca_id);
    }

    // If bound to a grid dimension and this tv is shared memory we don't need
    // to
    if (ca_id->isBlockDim() && is_shared) {
      continue;
    }

    // If bound to any thread and this is local memory we don't need to
    if (ca_id->isThread() && is_local) {
      continue;
    }

    auto loop = ca2loop.at(ca_id);

    // We're worried below about merged axes in the compute at that aren't in
    // tv, however reduction domains can only merge in themselves, so if
    // computeAt is a reduction domain it can't have a merged in dim tv doesn't
    // have. If we don't short-cut this we can hit issues in rfactor.
    if (ca_id->isReduction()) {
      indices[tv_i] = loop->index();
      continue;
    }

    // Grab the axis information
    auto tv_id = tv->axis(tv_i);

    // Check if tv_id had a broadcast merged that ca_id had an extent for, or
    // tv_id didn't have an iter domain that ca_id has merged into it. If this
    // is the case, we need to modulo the index of that loop by tv_id's extent.

    auto ca_id_inputs = ir_utils::iterDomainInputsOf({ca_id});
    int ca_id_inputs_n_bcast = std::count_if(
        ca_id_inputs.begin(), ca_id_inputs.end(), [](IterDomain* id) {
          return id->isBroadcast();
        });

    // If no broadcasts were in the input to ca_id no modulo necessary
    if (ca_id_inputs_n_bcast == 0) {
      indices[tv_i] = loop->index();
    } else {
      auto tv_id_inputs = ir_utils::iterDomainInputsOf({tv_id});
      int tv_id_inputs_n_bcast = std::count_if(
          tv_id_inputs.begin(), tv_id_inputs.end(), [](IterDomain* id) {
            return id->isBroadcast();
          });

      // tv_id has broadcasts in their input, but so does the ca_id. Also
      // shouldn't need modulo. Can't merge a braodcast/iteration domain with a
      // reduction domain, so ca_id inputs should strictly be >= tv_id's
      if (ca_id_inputs.size() == tv_id_inputs.size() &&
          ca_id_inputs_n_bcast == tv_id_inputs_n_bcast) {
        indices[tv_i] = loop->index();
      } else {
        indices[tv_i] = mod(loop->index(), tv_id->extent());
      }
    }
  }

  return indices;
}

std::vector<Val*> getUnrollPredIndicesForTV(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  Val* zero = new Int(0);
  Val* one = new Int(1);

  std::vector<Val*> indices(consumer_tv->nDims(), zero);

  std::unordered_set<kir::ForLoop*> loops_within_unroll(
      std::find_if(
          loops.begin(),
          loops.end(),
          [](const auto& loop) {
            return loop->iter_domain()->getParallelType() ==
                ParallelType::Unroll;
          }),
      loops.end());

  // Which loop is this axis associated with?
  auto ca2loop = computeAtToLoopMap(consumer_tv, loops);
  bool need_reduction_axes = loopsHasReductions(consumer_tv, loops);

  // Look at each axis individually in out's domain
  for (int64_t tv_i = 0; tv_i < (int64_t)consumer_tv->nDims(); tv_i++) {
    auto ca_id = consumer_tv->getComputeAtAxis(tv_i).first;

    if (consumer_tv->axis(tv_i)->isReduction() && !need_reduction_axes) {
      continue;
    }

    auto loop = ca2loop.at(ca_id);

    auto ind = loop->index();
    if (loops_within_unroll.find(loop) != loops_within_unroll.end() &&
        !ca_id->isThread()) {
      ind = sub(ca_id->extent(), one);
    }

    indices[tv_i] = ind;
    // Normally we'd be worried about merged axes in the compute at (see
    // getIndicesforTV), but this should be picked up by an expr later in the
    // unrolled loop, shouldn't have to worry about it here.
  }

  return indices;
}

std::vector<Val*> getRangesForTV(
    TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& ca_id_map) {
  Val* zero = new Int(0);
  std::vector<Val*> ranges(tv->nDims(), zero);

  bool is_shared = tv->getMemoryType() == MemoryType::Shared;
  bool is_local = tv->getMemoryType() == MemoryType::Local;

  TORCH_INTERNAL_ASSERT(
      is_shared || is_local,
      " Cannot use this function for global memory, the ranges of root domains of global memory are simply rootIDs->extent()");

  // Where is this TV allocated relative to the loop nest and its own axes?
  auto alloc_axis = loop_utils::getAllocPoint(tv, loops).second;

  // Look at each axis individually in out's domain
  for (int64_t tv_i = 0; tv_i < (int64_t)tv->nDims(); tv_i++) {
    // Grab the axis information
    auto tv_id = tv->axis(tv_i);

    // reduction axes don't have an extent
    if (tv_id->isReduction()) {
      continue;
    }

    // Check if we need to index based on this axis
    // If outside our allocation point, we don't need to
    if (tv_i < alloc_axis) {
      continue;
    }

    auto ca_id = tv->getComputeAtAxis(tv_i).first;
    if (!ca_id_map.empty()) {
      ca_id = ca_id_map.at(ca_id);
    }

    // If bound to a grid dimension and this tv is shared memory we don't need
    // to
    if (ca_id->isBlockDim() && is_shared) {
      continue;
    }

    // If bound to any thread and this is local memory we don't need to
    if (ca_id->isThread() && is_local) {
      continue;
    }

    ranges[tv_i] = tv_id->extent();
  }

  return ranges;
}

std::pair<kir::ForLoop*, int64_t> getAllocPoint(
    TensorView* tv,
    const std::vector<kir::ForLoop*>& loops) {
  // If in global memory, it can be all the way outside the loops.
  if (tv->getMemoryType() == MemoryType::Global) {
    return std::make_pair(nullptr, 0);
  }

  // Figure out where we want to place alloc/reduction initialization. We want
  // outside an unroll loop, or inside our computeAt point.
  kir::ForLoop* alloc_loop = nullptr;

  auto loops_it = loops.begin();

  // Look at each axis individually in out's domain
  for (int64_t tv_i = 0; tv_i < (int64_t)tv->getThisComputeAtAxis(); tv_i++) {
    // Grab the axis ID
    auto ca_id = tv->getComputeAtAxis(tv_i).first;

    loops_it = std::find_if(loops_it, loops.end(), [&ca_id](const auto& loop) {
      return ca_id == loop->iter_domain() ||
          loop->iter_domain()->getParallelType() == ParallelType::Unroll;
    });

    TORCH_INTERNAL_ASSERT(
        loops_it != loops.end(),
        "Could not find all required axes for indexing.");

    if ((*loops_it)->iter_domain()->getParallelType() == ParallelType::Unroll) {
      return std::make_pair(alloc_loop, tv_i);
    }

    alloc_loop = *loops_it;
    ++loops_it;
  }

  return std::make_pair(alloc_loop, (int64_t)tv->getThisComputeAtAxis());
}

std::unordered_map<IterDomain*, IterDomain*> p2cRootMap(
    std::vector<Expr*> exprs) {
  std::unordered_map<IterDomain*, IterDomain*> p2c_root_map;

  for (auto expr : exprs) {
    auto out_tv = ir_utils::getTVOutput(expr);
    for (auto inp : expr->inputs()) {
      if (inp->getValType().value() != ValType::TensorView) {
        continue;
      }

      auto root_p2c = TensorDomain::mapRootPtoC(
          inp->as<TensorView>()->domain(), out_tv->domain());
      for (auto entry : root_p2c) {
        auto p_id = entry.first;
        auto c_id = entry.second;
        // Careful we don't allow circular references
        if (p_id != c_id) {
          p2c_root_map[p_id] = c_id;
        }
      }
    }
  }

  return p2c_root_map;
}

IterDomain* getTermIDInMap(
    IterDomain* root_id,
    std::unordered_map<IterDomain*, IterDomain*> p2c_root_map) {
  auto entry = root_id;
  while (p2c_root_map.find(entry) != p2c_root_map.end()) {
    entry = p2c_root_map.at(entry);
  }
  return entry;
}

} // namespace loop_utils

} // namespace fuser
} // namespace jit
} // namespace torch
