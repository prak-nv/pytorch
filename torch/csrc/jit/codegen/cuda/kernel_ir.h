#pragma once

#include <torch/csrc/jit/codegen/cuda/type.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

// TODO(kir): remove these once the Kernel IR is separated from Fusion IR
#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_internal_nodes.h>
#include <torch/csrc/jit/codegen/cuda/parallel_type_bitmap.h>

#include <c10/util/Optional.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
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
class Double;
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
class TORCH_CUDA_CU_API IrVisitor : public PolymorphicBase {
 public:
  // TODO(kir): use Node* instead of void*
  virtual void unhandled(const void* node) {}

  // Values
  virtual void visit(const NamedScalar* named_scalar) {
    unhandled(named_scalar);
  }
  virtual void visit(const Bool* value) {
    unhandled(value);
  }
  virtual void visit(const Double* value) {
    unhandled(value);
  }
  virtual void visit(const Int* value) {
    unhandled(value);
  }
  virtual void visit(const IterDomain* iter_domain) {
    unhandled(iter_domain);
  }
  virtual void visit(const TensorDomain* tensor_domain) {
    unhandled(tensor_domain);
  }
  virtual void visit(const TensorView* tensor_view) {
    unhandled(tensor_view);
  }
  virtual void visit(const TensorIndex* tensor_index) {
    unhandled(tensor_index);
  }

  // Expressions
  virtual void visit(const UnaryOp* node) {
    unhandled(node);
  }
  virtual void visit(const BinaryOp* node) {
    unhandled(node);
  }
  virtual void visit(const TernaryOp* node) {
    unhandled(node);
  }
  virtual void visit(const ReductionOp* node) {
    unhandled(node);
  }
  virtual void visit(const BroadcastOp* node) {
    unhandled(node);
  }

  // Statements
  virtual void visit(const Allocate* node) {
    unhandled(node);
  }
  virtual void visit(const Sync* node) {
    unhandled(node);
  }
  virtual void visit(const ForLoop* node) {
    unhandled(node);
  }
  virtual void visit(const IfThenElse* node) {
    unhandled(node);
  }
  virtual void visit(const GridReduction* node) {
    unhandled(node);
  }
};

//! Kernel IR visitor interface
class TORCH_CUDA_CU_API MutableIrVisitor : public PolymorphicBase {
 public:
  // TODO(kir): use Node* instead of void*
  virtual void unhandled(const void*) {}

  // Values
  virtual void visit(NamedScalar* named_scalar) {
    unhandled(named_scalar);
  }
  virtual void visit(Bool* value) {
    unhandled(value);
  }
  virtual void visit(Double* value) {
    unhandled(value);
  }
  virtual void visit(Int* value) {
    unhandled(value);
  }
  virtual void visit(IterDomain* iter_domain) {
    unhandled(iter_domain);
  }
  virtual void visit(TensorDomain* tensor_domain) {
    unhandled(tensor_domain);
  }
  virtual void visit(TensorView* tensor_view) {
    unhandled(tensor_view);
  }
  virtual void visit(TensorIndex* tensor_index) {
    unhandled(tensor_index);
  }

  // Expressions
  virtual void visit(UnaryOp* node) {
    unhandled(node);
  }
  virtual void visit(BinaryOp* node) {
    unhandled(node);
  }
  virtual void visit(TernaryOp* node) {
    unhandled(node);
  }
  virtual void visit(ReductionOp* node) {
    unhandled(node);
  }
  virtual void visit(BroadcastOp* node) {
    unhandled(node);
  }

  // Statements
  virtual void visit(Allocate* node) {
    unhandled(node);
  }
  virtual void visit(Sync* node) {
    unhandled(node);
  }
  virtual void visit(ForLoop* node) {
    unhandled(node);
  }
  virtual void visit(IfThenElse* node) {
    unhandled(node);
  }
  virtual void visit(GridReduction* node) {
    unhandled(node);
  }
};

//! Base class for Kernel IR nodes
class TORCH_CUDA_CU_API Node : public NonCopyable, public PolymorphicBase {
 public:
  explicit Node(Passkey) {}

  //! IR Visitor double-dispatch interface
  //! (https://en.wikipedia.org/wiki/Visitor_pattern)
  virtual void accept(IrVisitor* visitor) const = 0;

  //! Non constant IR Visitor
  virtual void accept(MutableIrVisitor* visitor) = 0;

  //! Debug helper, prints the textual representation of an IR node
  void print() const;
};

//! Generic value (scalar or tensor)
class TORCH_CUDA_CU_API Val : public Node {
 public:
  Val(Passkey passkey, DataType dtype);

  // TODO(kir): consider renaming
  StmtNameType name() const {
    return name_;
  }

  void setName(StmtNameType name) {
    name_ = name;
  }

  ValueId id() const {
    return id_;
  }

  DataType dtype() const {
    return dtype_;
  }

  Expr* definition() const {
    return definition_;
  }

  void setDefinition(Expr* expr) {
    // TODO(kir): extra checks on changing existing definitions?
    definition_ = expr;
  }

  virtual bool isScalar() const {
    return false;
  }

  virtual bool isConst() const {
    return false;
  }

  // TODO(kir): revisit and find a better interface
  virtual bool isZeroInt() const {
    return false;
  }

  virtual bool isOneInt() const {
    return false;
  }

 private:
  const DataType dtype_;

  // The expression which defines this value, or nullptr
  Expr* definition_ = nullptr;

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
class TORCH_CUDA_CU_API Expr : public Node {
 public:
  explicit Expr(Passkey passkey) : Node(passkey) {}

  const auto& inputs() const {
    return inputs_;
  }

  const auto& outputs() const {
    return outputs_;
  }

  Expr* parentScope() const {
    return parent_scope_;
  }

  void setParentScope(Expr* scope);

  Bool* predicate() const {
    return predicate_;
  }

  void setPredicate(Bool* predicate) {
    predicate_ = predicate;
  }

 protected:
  // TODO(kir): try to avoid this protected interface
  void addInput(Val* input) {
    inputs_.push_back(input);
  }

  void addOutput(Val* output) {
    output->setDefinition(this);
    outputs_.push_back(output);
  }

 private:
  // TODO(kir): can we avoid this?
  std::vector<Val*> inputs_;
  std::vector<Val*> outputs_;

  // TODO(kir): revisit scope/nesting data structures
  Expr* parent_scope_ = nullptr;

  Bool* predicate_ = nullptr;
};

class TORCH_CUDA_CU_API NamedScalar final : public Val {
 public:
  NamedScalar(Passkey passkey, std::string name, DataType dtype)
      : Val(passkey, dtype), name_(name) {}

  explicit NamedScalar(Passkey passkey, const fuser::cuda::NamedScalar* node)
      : Val(passkey, node->getDataType().value()) {
    name_ = node->name();
  }

  void accept(IrVisitor* visitor) const override {
    visitor->visit(this);
  }

  void accept(MutableIrVisitor* visitor) override {
    visitor->visit(this);
  }

  bool isScalar() const override {
    return true;
  }

  // TODO(kir): this is hiding and redefining Val::name()
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

class TORCH_CUDA_CU_API Bool final : public Val {
 public:
  explicit Bool(Passkey passkey, const c10::optional<bool>& value)
      : Val(passkey, DataType::Bool), maybe_value_(value) {}

  explicit Bool(Passkey passkey, const fuser::cuda::Bool* node)
      : Val(passkey, DataType::Bool), maybe_value_(node->value()) {
    setName(node->name());
  }

  void accept(IrVisitor* visitor) const override {
    visitor->visit(this);
  }

  void accept(MutableIrVisitor* visitor) override {
    visitor->visit(this);
  }

  bool isScalar() const override {
    return true;
  }

  bool isConst() const override {
    return maybe_value_.has_value();
  }

  c10::optional<bool> value() const {
    return maybe_value_;
  }

 private:
  const c10::optional<bool> maybe_value_;
};

class TORCH_CUDA_CU_API Double final : public Val {
 public:
  using ScalarType = double;

  explicit Double(Passkey passkey, const c10::optional<ScalarType>& value)
      : Val(passkey, DataType::Double), maybe_value_(value) {}

  explicit Double(Passkey passkey, const fuser::cuda::Double* node)
      : Val(passkey, DataType::Double), maybe_value_(node->value()) {
    setName(node->name());
  }

  void accept(IrVisitor* visitor) const override {
    visitor->visit(this);
  }

  void accept(MutableIrVisitor* visitor) override {
    visitor->visit(this);
  }

  bool isScalar() const override {
    return true;
  }

  bool isConst() const override {
    return maybe_value_.has_value();
  }

  c10::optional<ScalarType> value() const {
    return maybe_value_;
  }

 private:
  const c10::optional<ScalarType> maybe_value_;
};

class TORCH_CUDA_CU_API Int final : public Val {
 public:
  using ScalarType = int64_t;

  explicit Int(Passkey passkey, const c10::optional<ScalarType>& value)
      : Val(passkey, DataType::Int), maybe_value_(value) {}

  explicit Int(
      Passkey passkey,
      const fuser::cuda::Int* node,
      bool /*avoid_zero_ambiguity*/)
      : Val(passkey, DataType::Int), maybe_value_(node->value()) {
    setName(node->name());
  }

  void accept(IrVisitor* visitor) const override {
    visitor->visit(this);
  }

  void accept(MutableIrVisitor* visitor) override {
    visitor->visit(this);
  }

  bool isScalar() const override {
    return true;
  }

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

class TORCH_CUDA_CU_API IterDomain final : public Val {
 public:
  IterDomain(Passkey passkey, Val* start, Val* extent);

  explicit IterDomain(Passkey, const fuser::cuda::IterDomain* iter_domain);

  void accept(IrVisitor* visitor) const override {
    visitor->visit(this);
  }

  void accept(MutableIrVisitor* visitor) override {
    visitor->visit(this);
  }

  bool isReduction() const {
    return iterType() == IterType::Reduction;
  }

  bool isRFactorProduct() const {
    return is_rfactor_domain_;
  }

  bool isBroadcast() const {
    return iterType() == IterType::BroadcastWithStride ||
        iterType() == IterType::BroadcastWithoutStride;
  }

  bool isParallelized() const {
    return parallelType() != ParallelType::Serial;
  }

  // Return if this iter domain is mapped to a grid dimension
  bool isBlockDim() const {
    return parallelType() == ParallelType::BIDz ||
        parallelType() == ParallelType::BIDy ||
        parallelType() == ParallelType::BIDx;
  }

  // Return if this iter domain is mapped to a block dimension
  bool isThreadDim() const {
    return parallelType() == ParallelType::TIDz ||
        parallelType() == ParallelType::TIDy ||
        parallelType() == ParallelType::TIDx;
  }

  // Return if this iter domain is either mapped to a block or grid dimension
  bool isThread() const {
    return isBlockDim() || isThreadDim();
  }

  ParallelType parallelType() const {
    return parallel_type_;
  }

  IterType iterType() const {
    return iter_type_;
  }

  Val* start() const {
    return start_;
  }

  Val* extent() const;

  Val* rawExtent() const {
    return extent_;
  }

  bool isSimple() const {
    return is_simple_;
  }

 private:
  Val* const start_ = nullptr;
  Val* const extent_ = nullptr;
  ParallelType parallel_type_ = ParallelType::Serial;
  IterType iter_type_ = IterType::Iteration;
  bool is_rfactor_domain_ = false;

  // An IterDomain is "simple" if the original Fusion IterDomain
  // doesn't have a definition ("definition" expression)
  //
  // TODO(kir): this feels like a hack, revisit
  //
  bool is_simple_ = true;
};

// TODO(kir): is this really a value?
class TORCH_CUDA_CU_API TensorDomain final : public Val {
 public:
  explicit TensorDomain(Passkey, std::vector<IterDomain*> domain);

  explicit TensorDomain(
      Passkey passkey,
      const fuser::cuda::TensorDomain* tensor_domain);

  void accept(IrVisitor* visitor) const override {
    visitor->visit(this);
  }

  void accept(MutableIrVisitor* visitor) override {
    visitor->visit(this);
  }

  std::vector<IterDomain*>::size_type nDims() const {
    return domain_.size();
  }

  // TODO(kir): rename this
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

class TORCH_CUDA_CU_API TensorView final : public Val {
 public:
  explicit TensorView(Passkey, const fuser::cuda::TensorView* tv);

  TensorView(
      Passkey,
      DataType dtype,
      TensorDomain* domain,
      MemoryType memory_type);

  TensorDomain* domain() const {
    return domain_;
  }

  void accept(IrVisitor* visitor) const override {
    visitor->visit(this);
  }

  void accept(MutableIrVisitor* visitor) override {
    visitor->visit(this);
  }

  MemoryType memoryType() const {
    return memory_type_;
  }

  fuser::cuda::TensorView* fuserTv() const {
    TORCH_INTERNAL_ASSERT(fuser_tv_ != nullptr);
    // TODO(kir): remove the need for const_cast
    return const_cast<fuser::cuda::TensorView*>(fuser_tv_); // NOLINT
  }

 private:
  TensorDomain* domain_ = nullptr;
  MemoryType memory_type_ = MemoryType::Local;

  // TODO(kir): remove temporary hack
  const fuser::cuda::TensorView* fuser_tv_ = nullptr;
};

class TORCH_CUDA_CU_API UnaryOp final : public Expr {
 public:
  UnaryOp(Passkey passkey, UnaryOpType operation, Val* out, Val* in);

  void accept(IrVisitor* visitor) const override {
    visitor->visit(this);
  }

  void accept(MutableIrVisitor* visitor) override {
    visitor->visit(this);
  }

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

class TORCH_CUDA_CU_API BinaryOp final : public Expr {
 public:
  BinaryOp(
      Passkey passkey,
      BinaryOpType operation,
      Val* out,
      Val* lhs,
      Val* rhs);

  void accept(IrVisitor* visitor) const override {
    visitor->visit(this);
  }

  void accept(MutableIrVisitor* visitor) override {
    visitor->visit(this);
  }

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

class TORCH_CUDA_CU_API TernaryOp final : public Expr {
 public:
  TernaryOp(
      Passkey passkey,
      TernaryOpType operation,
      Val* out,
      Val* in1,
      Val* in2,
      Val* in3);

  void accept(IrVisitor* visitor) const override {
    visitor->visit(this);
  }

  void accept(MutableIrVisitor* visitor) override {
    visitor->visit(this);
  }

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

class TORCH_CUDA_CU_API ReductionOp final : public Expr {
 public:
  ReductionOp(
      Passkey passkey,
      BinaryOpType operation,
      Val* init,
      Val* out,
      Val* in);

  void accept(IrVisitor* visitor) const override {
    visitor->visit(this);
  }

  void accept(MutableIrVisitor* visitor) override {
    visitor->visit(this);
  }

  Val* out() const {
    return out_;
  }

  Val* in() const {
    return in_;
  }

  Val* init() const {
    return init_;
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
};

class TORCH_CUDA_CU_API TensorIndex final : public Val {
 public:
  TensorIndex(
      Passkey,
      const fuser::cuda::TensorView* view,
      std::vector<Val*> indices);

  void accept(IrVisitor* visitor) const override {
    visitor->visit(this);
  }

  void accept(MutableIrVisitor* visitor) override {
    visitor->visit(this);
  }

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

class TORCH_CUDA_CU_API BroadcastOp final : public Expr {
 public:
  BroadcastOp(Passkey passkey, Val* out, Val* in);

  void accept(IrVisitor* visitor) const override {
    visitor->visit(this);
  }

  void accept(MutableIrVisitor* visitor) override {
    visitor->visit(this);
  }

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
//! TODO(kir): The components of Allocate like Type and Name could be separated
//!   from the the assocated TensorView.  Perhaps that is more appropriate?
//!
class TORCH_CUDA_CU_API Allocate final : public Expr {
 public:
  explicit Allocate(
      Passkey passkey,
      Val* buffer,
      MemoryType memory_type = MemoryType::Local,
      Val* size = nullptr,
      bool zero_init = false);

  void accept(IrVisitor* visitor) const override {
    visitor->visit(this);
  }

  void accept(MutableIrVisitor* visitor) override {
    visitor->visit(this);
  }

  Val* buffer() const {
    return buffer_;
  }

  MemoryType memoryType() const {
    return memory_type_;
  }

  Val* size() const {
    return size_;
  }

  bool zeroInit() const {
    return zero_init_;
  }

  const Allocate* alias() const {
    return alias_;
  }

  void setAlias(const Allocate* alias) {
    TORCH_INTERNAL_ASSERT(alias != this);
    TORCH_INTERNAL_ASSERT(alias->memoryType() == memory_type_);
    alias_ = alias;
  }

 private:
  Val* buffer_ = nullptr;
  MemoryType memory_type_ = MemoryType::Local;
  Val* size_ = nullptr;
  bool zero_init_ = false;

  // This alias tracks the next Allocate node in a linked chain of aliases
  // If the alias is nullptr, then the Allocate node uses memory in the kernel
  const Allocate* alias_ = nullptr;
};

// Sync represents __syncthreads barrier for block level coordination.
//
// TODO(kir): change name to SyncThreads as we could have other barriers.
//
class TORCH_CUDA_CU_API Sync final : public Expr {
 public:
  explicit Sync(Passkey passkey, bool war_sync = false);

  void accept(IrVisitor* visitor) const override {
    visitor->visit(this);
  }

  void accept(MutableIrVisitor* visitor) override {
    visitor->visit(this);
  }

  bool isWarHazardSync() const {
    return war_sync_;
  }

 private:
  // TODO: war_sync_ is only used for testing/validation purposes.
  bool war_sync_ = false;
};

// TODO(kir): promote to IR node
class TORCH_CUDA_CU_API Scope {
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

  auto& operator[](size_t i) {
    return exprs_[i];
  }

  auto& operator[](size_t i) const {
    return exprs_[i];
  }

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

//! ForLoop provides scoping around an int iterator from 0 to range. Exprs
//! placed in its body are considered inside the scope of the for loop. In the
//! future the implementation should look quite different so that we can do
//! proper dependency annalysis like in Fusion.
//!
//! TODO(kir): this is not a real expression
//!
class TORCH_CUDA_CU_API ForLoop final : public Expr {
 public:
  ForLoop(
      Passkey passkey,
      Val* index,
      IterDomain* iter_domain,
      Expr* parent_scope);

  void accept(IrVisitor* visitor) const override {
    visitor->visit(this);
  }

  void accept(MutableIrVisitor* visitor) override {
    visitor->visit(this);
  }

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

 private:
  Val* const index_ = nullptr;
  IterDomain* const iter_domain_;
  Scope body_;
};

//! IfThenElse provides scoping for an boolean operator. Exprs placed in its
//! body are considered inside the scope of the if statement. In the future the
//! implementation should look quite different so that we can do proper
//! dependency annalysis like in Fusion.
//!
//! TODO(kir): this is not a real expression
//!
class TORCH_CUDA_CU_API IfThenElse final : public Expr {
 public:
  explicit IfThenElse(Passkey passkey, Bool* cond, Expr* parent_scope);

  void accept(IrVisitor* visitor) const override {
    visitor->visit(this);
  }

  void accept(MutableIrVisitor* visitor) override {
    visitor->visit(this);
  }

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

 private:
  Bool* const cond_ = nullptr;
  Scope then_body_;
  Scope else_body_;
};

//! Grid reduction operation
//!
//! This node is used only after lowering a fusion to explicitly mark a grid
//! reduction and the buffer allocation needed to do it.
//!
//! This node provides FusionExecutor the information it needs to allocate the
//! reduction and sync buffers.
class TORCH_CUDA_CU_API GridReduction final : public Expr {
 public:
  explicit GridReduction(Passkey passkey, ReductionOp* reduction_op);

  void accept(IrVisitor* visitor) const override {
    visitor->visit(this);
  }

  void accept(MutableIrVisitor* visitor) override {
    visitor->visit(this);
  }

  GridReduction(
      Passkey passkey,
      ReductionOp* reduction_op,
      Allocate* reduction_buffer,
      Allocate* sync_buffer);

  ReductionOp* reduction_op() const {
    return reduction_op_;
  }

  Allocate* reduction_buffer() const {
    return reduction_buffer_;
  }

  Allocate* sync_buffer() const {
    return sync_buffer_;
  }

  const ParallelTypeBitmap& threadPredicate() const {
    return thread_predicate_;
  }

  void setThreadPredicate(const ParallelTypeBitmap& thread_predicate) {
    thread_predicate_ = thread_predicate;
  }

  static std::string getPredicateFlagName(const TensorView* val);
  static std::string getPredicateFlagName(const fuser::cuda::TensorView* val);

 private:
  ReductionOp* reduction_op_ = nullptr;
  Allocate* reduction_buffer_ = nullptr;
  Allocate* sync_buffer_ = nullptr;
  // gridReduce has template flags for thread predicates. In order to
  // use them, the thread predicate is held here separately from
  // Expr::predicate_.
  ParallelTypeBitmap thread_predicate_;
};

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
