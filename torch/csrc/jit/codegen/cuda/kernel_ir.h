
#pragma once

#include <torch/csrc/jit/codegen/cuda/utils.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

// TODO(kir): remove these once the Kernel IR is separated from Fusion IR
#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_internal_nodes.h>

#include <c10/util/Optional.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace kir {

class IrBuilder;
class Kernel;

// Abstract nodes
class Node;
class Val;
class Expr;

// Values
class NamedScalar;
class Bool;
class Float;
class Half;
class Int;
class IterDomain;
class TensorDomain;
class TensorView;
class TensorIndex;

// Expressions
class UnaryOp;
class BinaryOp;
class TernaryOp;
class ReductionOp;
class BroadcastOp;

// Statements
class Allocate;
class Sync;
class ForLoop;
class IfThenElse;
class GridReduction;

using ValueId = int32_t;

//! Token used to restrict the access to Kernel IR creation
//!
//! A token is associated with a kernel, which is passed with the key
//! (Passkey::kernel)
//!
//! It is a "granular friendship" token, used to implement the "passkey" idiom:
//! https://www.spiria.com/en/blog/desktop-software/passkey-idiom-and-better-friendship-c
//! https://arne-mertz.de/2016/10/passkey-idiom
//!
class Passkey {
  friend class IrBuilder;

 public:
  Kernel* const kernel = nullptr;

 private:
  explicit Passkey(Kernel* kernel) : kernel(kernel) {}
};

//! Kernel IR visitor interface
class TORCH_CUDA_API IrVisitor : public NonCopyable, public PolymorphicBase {
 public:
  // Values
  virtual void visit(const NamedScalar* named_scalar) {}
  virtual void visit(const Bool* value) {}
  virtual void visit(const Float* value) {}
  virtual void visit(const Half* value) {}
  virtual void visit(const Int* value) {}
  virtual void visit(const IterDomain* iter_domain) {}
  virtual void visit(const TensorDomain* tensor_domain) {}
  virtual void visit(const TensorView* tensor_view) {}
  virtual void visit(const TensorIndex* tensor_index) {}

  // Expressions
  virtual void visit(const UnaryOp* node) {}
  virtual void visit(const BinaryOp* node) {}
  virtual void visit(const TernaryOp* node) {}
  virtual void visit(const ReductionOp* node) {}
  virtual void visit(const BroadcastOp* node) {}

  // Statements
  virtual void visit(const Allocate* node) {}
  virtual void visit(const Sync* node) {}
  virtual void visit(const ForLoop* node) {}
  virtual void visit(const IfThenElse* node) {}
  virtual void visit(const GridReduction* node) {}
};

//! Base class for Kernel IR nodes
class TORCH_CUDA_API Node : public NonCopyable, public PolymorphicBase {
 public:
  explicit Node(Passkey) {}

  virtual void accept(IrVisitor* visitor) const { visitor->visit(this); }
};

//! Generic value (scalar or tensor)
class TORCH_CUDA_API Val : public Node {
 public:
  Val(Passkey passkey, DataType dtype) : Node(passkey), dtype_(dtype) {
    id_ = passkey.kernel->newValueId(passkey);
  }

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

  StmtNameType name() const {
    return name_;
  }

  ValueId id() const {
    return id_;
  }

  DataType dtype() const {
    return dtype_;
  }

  Expr* definition() const {
    // $$$
    return nullptr;
  }

  virtual bool isScalar() const { return false; }

  virtual bool isConst() const { return false; }

  // TODO(kir): revisit and find a better interface
  virtual bool isZeroInt() const { return false; }
  virtual bool isOneInt() const { return false; }

 private:
  const DataType dtype_;

  // This is a value name preserved from the Fusion IR (optional)
  StmtNameType name_ = kInvalidStmName;

  // All Kernel IR values have IDs (unique within the same Kernel)
  ValueId id_ = -1;
};

//! Base class for expressions and statements
//!
//! Expressions consume inputs and produce outputs (depending on the context
//! this may imply assignments). Currently some of the expressions
//! don't actually produce any outputs (ForLoop, IfThenElse) and they
//! model statements to be executed.
//!
//! TODO(kir): split the expressions, assignments and statements?
//!
class TORCH_CUDA_API Expr : public Node {
 public:
  explicit Expr(Passkey passkey) : Node(passkey) {}

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

 protected:
  void registerInput(Val* input) {
    inputs_.push_back(input);
  }

  void registerOutput(Val* output) {
    outputs_.push_back(output);
  }

 private:
  // TODO(kir): can we avoid this?
  std::vector<Val*> inputs_;
  std::vector<Val*> outputs_;
};

class TORCH_CUDA_API NamedScalar : public Val {
 public:
  NamedScalar(Passkey Passkey passkey, std::string name, DataType dtype)
      : Val(Passkey passkey, dtype), name_(name) {}

  explicit NamedScalar(Passkey passkey, const fuser::NamedScalar* node)
      : Val(node), name_(node->name()) {}

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

  bool isScalar() const override { return true; }

  const std::string& name() const {
    return name_;
  }

  // Return the named scalar extent of a parallel dimension (e.g. blockDim.x)
  static NamedScalar* getParallelDim(ParallelType p_type);

  // Return the named scalar index of a parallel dimension (e.g. threadIdx.x)
  static NamedScalar* getParallelIndex(ParallelType p_type);

  // Return the parallel type of this NamedScalar if it is an extent of a
  // parallel dimension
  c10::optional<ParallelType> getParallelDim() const;

  // Return the parallel type of this NamedScalar if it is an index of a
  // parallel dimension
  c10::optional<ParallelType> getParallelIndex() const;

 private:
  std::string name_;
};

class TORCH_CUDA_API Bool : public Val {
 public:
  explicit Bool(Passkey passkey, const c10::optional<bool>& value)
      : Val(passkey, DataType::Bool, true, true),
        maybe_value_(value) {}

  explicit Bool(Passkey passkey, const fuser::Bool* node)
      : Val(node), maybe_value_(node->value()) {}

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

  bool isScalar() const override { return true; }

  bool isConst() const override {
    return maybe_value_.has_value();
  }

  c10::optional<bool> value() const {
    return maybe_value_;
  }

 private:
  const c10::optional<bool> maybe_value_;
};

class TORCH_CUDA_API Float : public Val {
 public:
  using ScalarType = double;

  explicit Float(Passkey passkey, const c10::optional<ScalarType>& value)
      : Val(passkey, DataType::Float, true, true),
        maybe_value_(value) {}

  explicit Float(Passkey passkey, const fuser::Float* node)
      : Val(node), maybe_value_(node->value()) {}

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

  bool isScalar() const override { return true; }

  bool isConst() const override {
    return maybe_value_.has_value();
  }

  c10::optional<ScalarType> value() const {
    return maybe_value_;
  }

 private:
  const c10::optional<ScalarType> maybe_value_;
};

class TORCH_CUDA_API Half : public Val {
 public:
  explicit Half(Passkey passkey, const c10::optional<float>& value)
      : Val(passkey, DataType::Half, true, true),
        maybe_value_(value) {}

  explicit Half(Passkey passkey, const fuser::Half* node)
      : Val(node), maybe_value_(node->value()) {}

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

  bool isScalar() const override { return true; }

  bool isConst() const override {
    return maybe_value_.has_value();
  }

  c10::optional<float> value() const {
    return maybe_value_;
  }

 private:
  const c10::optional<float> maybe_value_;
};

class TORCH_CUDA_API Int : public Val {
 public:
  using ScalarType = int64_t;

  explicit Int(Passkey passkey, const c10::optional<ScalarType>& value)
      : Val(passkey, DataType::Int, true, true),
        maybe_value_(value) {}

  explicit Int(
      Passkey passkey,
      const fuser::Int* node,
      bool /*avoid_zero_ambiguity*/)
      : Val(node), maybe_value_(node->value()) {}

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

  bool isScalar() const override { return true; }

  bool isConst() const override {
    return maybe_value_.has_value();
  }

  bool isZeroInt() const override {
    return maybe_value_.has_value() && *maybe_value_ == 0;
  }

  bool isOneInt() const override {
    return maybe_value_.has_value() && *maybe_value_ == 1;
  }

  c10::optional<ScalarType> value() const {
    return maybe_value_;
  }

 private:
  const c10::optional<ScalarType> maybe_value_;
};

class TORCH_CUDA_API IterDomain : public Val {
 public:
  IterDomain(Passkey passkey, Val* start, Val* extent);

  explicit IterDomain(Passkey passkey, const fuser::IterDomain* iter_domain);

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

  bool isReduction() const {
    return getIterType() == IterType::Reduction;
  }

  bool isRFactorProduct() const {
    return is_rfactor_domain_;
  }

  bool isBroadcast() const {
    return getIterType() == IterType::BroadcastWithStride ||
        getIterType() == IterType::BroadcastWithoutStride;
  }

  bool isParallelized() const {
    return getParallelType() != ParallelType::Serial;
  }

  // Return if this iter domain is mapped to a grid dimension
  bool isBlockDim() const {
    return (
        getParallelType() == ParallelType::BIDz ||
        getParallelType() == ParallelType::BIDy ||
        getParallelType() == ParallelType::BIDx);
  }

  // Return if this iter domain is mapped to a block dimension
  bool isThreadDim() const {
    return (
        getParallelType() == ParallelType::TIDz ||
        getParallelType() == ParallelType::TIDy ||
        getParallelType() == ParallelType::TIDx);
  }

  // Return if this iter domain is either mapped to a block or grid dimension
  bool isThread() const {
    return (isBlockDim() || isThreadDim());
  }

  ParallelType getParallelType() const {
    return parallel_type_;
  }

  IterType getIterType() const {
    return iter_type_;
  }

  Val* start() const {
    return start_;
  }

  Val* extent() const;

 private:
  Val* const start_ = nullptr;
  Val* const extent_ = nullptr;
  ParallelType parallel_type_ = ParallelType::Serial;
  IterType iter_type_ = IterType::Iteration;
  bool is_rfactor_domain_ = false;
};

class TORCH_CUDA_API TensorDomain : public Val {
 public:
  explicit TensorDomain(Passkey passkey, std::vector<IterDomain*> domain);

  explicit TensorDomain(
      Passkey passkey,
      const fuser::TensorDomain* tensor_domain);

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

  std::vector<IterDomain*>::size_type nDims() const {
    return domain_.size();
  }

  const std::vector<IterDomain*>& domain() const {
    return domain_;
  }

  const std::vector<bool>& contiguity() const {
    return contiguity_;
  }

  std::string getContiguityString() const {
    std::stringstream ss;
    for (auto b : contiguity()) {
      ss << (b ? "t" : "f");
    }
    return ss.str();
  }

  bool hasReduction() const;
  bool hasBlockReduction() const;
  bool hasGridReduction() const;
  bool hasBlockBroadcast() const;
  bool hasBroadcast() const;
  bool hasRFactor() const;

  const std::vector<IterDomain*>& noReductions() const {
    return no_reduction_domain_;
  }

  const std::vector<IterDomain*>& noBroadcasts() const {
    return no_bcast_domain_;
  }

  const std::vector<IterDomain*>& rootDomain() const {
    return root_domain_;
  };

  const std::vector<IterDomain*>& rfactorDomain() const {
    return rfactor_domain_;
  };

  void resetDomains() {
    no_reduction_domain_ = noReductions(domain_);
    no_bcast_domain_ = noBroadcasts(domain_);
  }

  IterDomain* axis(int i) const;

  // TODO(kir): overloading non-static and static methods is not a good idea
  static std::vector<IterDomain*> noReductions(const std::vector<IterDomain*>&);
  static std::vector<IterDomain*> noBroadcasts(const std::vector<IterDomain*>&);

 private:
  std::vector<IterDomain*> root_domain_;
  std::vector<IterDomain*> domain_;
  std::vector<IterDomain*> no_bcast_domain_;
  std::vector<IterDomain*> no_reduction_domain_;
  std::vector<IterDomain*> rfactor_domain_;
  const std::vector<bool> contiguity_;
};

class TORCH_CUDA_API TensorView : public Val {
 public:
  explicit TensorView(Passkey passkey, const fuser::TensorView* tv);

  TensorDomain* domain() const {
    return domain_;
  }

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

  MemoryType memoryType() const {
    return memory_type_;
  }

  const fuser::TensorView* fuserTv() const {
    TORCH_INTERNAL_ASSERT(fuser_tv_ != nullptr);
    return fuser_tv_;
  }

 private:
  TensorDomain* domain_ = nullptr;
  MemoryType memory_type_ = MemoryType::Local;

  // TODO(kir): remove temporary hack
  const fuser::TensorView* fuser_tv_ = nullptr;
};

class TORCH_CUDA_API UnaryOp : public Expr {
 public:
  UnaryOp(Passkey passkey, UnaryOpType operation, Val* out, Val* in);

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

  Val* out() const {
    return out_;
  }

  Val* in() const {
    return in_;
  }

  UnaryOpType operation() const {
    return operation_;
  }

 private:
  const UnaryOpType operation_;
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;
};

class TORCH_CUDA_API BinaryOp : public Expr {
 public:
  BinaryOp(Passkey passkey, BinaryOpType operation, Val* out, Val* lhs, Val* rhs);

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

  Val* out() const {
    return out_;
  }

  Val* lhs() const {
    return lhs_;
  }

  Val* rhs() const {
    return rhs_;
  }

  BinaryOpType operation() const {
    return operation_;
  }

 private:
  const BinaryOpType operation_;
  Val* const out_ = nullptr;
  Val* const lhs_ = nullptr;
  Val* const rhs_ = nullptr;
};

class TORCH_CUDA_API TernaryOp : public Expr {
 public:
  TernaryOp(
      Passkey passkey,
      TernaryOpType operation,
      Val* out,
      Val* in1,
      Val* in2,
      Val* in3);

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

  Val* out() const {
    return out_;
  }

  Val* in1() const {
    return in1_;
  }

  Val* in2() const {
    return in2_;
  }

  Val* in3() const {
    return in3_;
  }

  TernaryOpType operation() const {
    return operation_;
  }

 private:
  const TernaryOpType operation_;
  Val* const out_ = nullptr;
  Val* const in1_ = nullptr;
  Val* const in2_ = nullptr;
  Val* const in3_ = nullptr;
};

class TORCH_CUDA_API ReductionOp : public Expr {
 public:
  ReductionOp(
      Passkey passkey,
      BinaryOpType operation,
      Val* init,
      Val* out,
      Val* in,
      Bool* pred = nullptr);

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

  Val* out() const {
    return out_;
  }

  Val* in() const {
    return in_;
  }

  Val* init() const {
    return init_;
  }

  Bool* pred() const {
    return pred_;
  }

  BinaryOpType operation() const {
    return operation_;
  }

  std::unordered_map<ParallelType, IterDomain*, TypeHash>
  getParallelReductionDomains() const;

 private:
  std::vector<IterDomain*> getReductionDomains() const;

 private:
  const BinaryOpType operation_;
  Val* const init_ = nullptr;
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;
  Bool* const pred_ = nullptr;
};

class TORCH_CUDA_API TensorIndex : public Val {
 public:
  TensorIndex(
      Passkey passkey,
      const fuser::TensorView* view,
      std::vector<Val*> indices);

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

  std::vector<Val*>::size_type nDims() const {
    return indices_.size();
  }

  Val* index(int i) const;

  const std::vector<Val*>& indices() const {
    return indices_;
  }

  const TensorView* view() const {
    return view_;
  }

 private:
  const TensorView* view_ = nullptr;
  std::vector<Val*> indices_;
};

class TORCH_CUDA_API BroadcastOp : public Expr {
 public:
  BroadcastOp(Passkey passkey, Val* out, Val* in);

  Val* out() const {
    return out_;
  }

  Val* in() const {
    return in_;
  }

 private:
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;
};

//! Allocate is a lower level Node that describes a buffer of memory that
//! is required as an intermediate within a kernel. The extent is the expression
//! of the size of the buffer that is generated from the TensorView that
//! describes the output of an operation.
//!
//! TODO: The components of Allocate like Type and Name could be separated from
//! the the assocated TensorView.  Perhaps that is more appropriate?
//!
class TORCH_CUDA_API Allocate : public Expr {
 public:
  explicit Allocate(
      Passkey passkey,
      Val* buffer,
      MemoryType memory_type = MemoryType::Local,
      Val* size = nullptr,
      bool zero_init = false);

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

  Val* buffer() const {
    return buffer_;
  }

  MemoryType getMemoryType() const {
    return memory_type_;
  }

  Val* size() const {
    return size_;
  }

  bool zeroInit() const {
    return zero_init_;
  }

  DataType buffer_type() const {
    return buffer_->getDataType().value();
  }

 private:
  Val* buffer_ = nullptr;
  MemoryType memory_type_ = MemoryType::Local;
  Val* size_ = nullptr;
  bool zero_init_ = false;
};

// Sync represents __syncthreads barrier for block level coordination.
class TORCH_CUDA_API Sync : public Expr {
 public:
  explicit Sync(Passkey passkey, bool war_sync = false);

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

  bool isWarHazardSync() const {
    return war_sync_;
  }

 private:
  // TODO: war_sync_ is only used for testing/validation purposes.
  bool war_sync_ = false;
};

// TODO(kir): promote to IR node
class TORCH_CUDA_API Scope {
 public:
  Scope() = default;

  const std::vector<Expr*>& exprs() const {
    return exprs_;
  }

  void push_back(Expr* e) {
    exprs_.push_back(e);
  }

  void insert(size_t pos, Expr* expr) {
    exprs_.insert(exprs_.begin() + pos, expr);
  }

  void erase(size_t pos) {
    exprs_.erase(exprs_.begin() + pos);
  }

  bool empty() const {
    return exprs_.empty();
  }

  auto size() const {
    return exprs_.size();
  }

/*
  auto& operator[](size_t i) {
    return exprs_[i];
  }

  auto& operator[](size_t i) const {
    return exprs_[i];
  }
*/

  // Insert expr before ref
  void insert_before(Expr* ref, Expr* expr);

  // Insert expr after ref
  void insert_after(Expr* ref, Expr* expr);

  bool contains(Expr* expr) const;

  void erase(Expr* ref);

  void clear();

 private:
  std::vector<Expr*> exprs_;
};

//! ForLoop provides scoping around an int iterator from 0 to range. Exprs placed
//! in its body are considered inside the scope of the for loop. In the future
//! the implementation should look quite different so that we can do proper
//! dependency annalysis like in Fusion.
//!
//! TODO(kir): this is not a real expression
//!
class TORCH_CUDA_API ForLoop : public Expr {
 public:
  ForLoop(
      Passkey passkey,
      Val* index,
      IterDomain* iter_domain,
      Expr* parent_scope);

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

  Val* index() const {
    return index_;
  }

  IterDomain* iter_domain() const {
    return iter_domain_;
  }

  Scope& body() {
    return body_;
  }

  const Scope& body() const {
    return body_;
  }

  Expr* parentScope() const {
    return parent_scope_;
  }

  void setParentScope(Expr* scope);

 private:
  Val* const index_ = nullptr;
  IterDomain* const iter_domain_;
  Scope body_;
  Expr* parent_scope_ = nullptr;
};

//! IfThenElse provides scoping for an boolean operator. Exprs placed in its body
//! are considered inside the scope of the if statement. In the future the
//! implementation should look quite different so that we can do proper
//! dependency annalysis like in Fusion.
//!
//! TODO(kir): this is not a real expression
//!
class TORCH_CUDA_API IfThenElse : public Expr {
 public:
  explicit IfThenElse(Passkey passkey, Bool* cond, Expr* parent_scope);

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

  Bool* cond() const {
    return cond_;
  }

  Scope& thenBody() {
    return then_body_;
  }
  const Scope& thenBody() const {
    return then_body_;
  }

  Scope& elseBody() {
    return else_body_;
  }

  const Scope& elseBody() const {
    return else_body_;
  }

  bool hasElse() const {
    return !else_body_.empty();
  }

  Expr* parentScope() const {
    return parent_scope_;
  }

  void setParentScope(Expr* scope);

 private:
  Bool* const cond_ = nullptr;
  Scope then_body_;
  Scope else_body_;
  Expr* parent_scope_ = nullptr;
};

//! Grid reduction operation
//!
//! This node is used only after lowering a fusion to explicitly mark a grid
//! reduction and the buffer allocation needed to do it.
//!
//! This node provides FusionExecutor the information it needs to allocate the
//! reduction and sync buffers.
class TORCH_CUDA_API GridReduction : public Expr {
 public:
  explicit GridReduction(Passkey passkey, ReductionOp* reduction_op);

  void accept(IrVisitor* visitor) const override { visitor->visit(this); }

  GridReduction(
      Passkey passkey,
      ReductionOp* reduction_op,
      Allocate* reduction_buffer,
      Allocate* sync_buffer,
      Bool* pred = nullptr);

  ReductionOp* reduction_op() const {
    return reduction_op_;
  }

  Allocate* reduction_buffer() const {
    return reduction_buffer_;
  }

  Allocate* sync_buffer() const {
    return sync_buffer_;
  }

  Bool* pred() const {
    return pred_;
  }

  static std::string getPredicateFlagName(const TensorView* val);
  static std::string getPredicateFlagName(const fuser::TensorView* val);

 private:
  ReductionOp* reduction_op_ = nullptr;
  Allocate* reduction_buffer_ = nullptr;
  Allocate* sync_buffer_ = nullptr;
  Bool* pred_ = nullptr;
};

} // namespace kir
} // namespace fuser
} // namespace jit
} // namespace torch
