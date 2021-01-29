#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/type.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <array>
#include <sstream>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace codegen {

namespace {

class CudaKernelGenerator : private kir::IrVisitor {
  static constexpr const char* kTab = "  ";

 public:
  static std::string generateKernelDefinition(
      const kir::Kernel* kernel,
      const std::string& kernel_name) {
    CudaKernelGenerator codegen(kernel);
    codegen.genDeclaration(kernel_name);
    codegen.startBlock();
    codegen.genPrologue();
    codegen.genBody();
    codegen.endBlock();
    TORCH_CHECK(codegen.block_nest_level_ == 0);
    return codegen.code_.str();
  }

 private:
  explicit CudaKernelGenerator(const kir::Kernel* kernel) : kernel_(kernel) {}

  // Generates the kernel function declaration
  void genDeclaration(const std::string& kernel_name) {
    const auto& kernel_summary = kernel_->summary();

    code_ << "__global__ void " << kernel_name << "(";

    std::vector<kir::Val*> params;

    // Inputs & Outputs
    for (auto val : kernel_->inputs()) {
      params.push_back(val);
    }
    for (auto val : kernel_->outputs()) {
      params.push_back(val);
    }

    // Generate parameter declarations
    for (kir::Val* val : params) {
      if (const auto tv = dynamic_cast<kir::TensorView*>(val)) {
        code_ << "Tensor<" << val->dtype() << ", "
              << TensorDomain::noReductions(
                     tv->fuserTv()->getMaybeRFactorDomain())
                     .size()
              << "> " << varName(tv);
      } else {
        TORCH_INTERNAL_ASSERT(val->isScalar()); // NOLINT (LLVM bug 48525)
        TORCH_INTERNAL_ASSERT(val->definition() == nullptr);
        code_ << val->dtype() << " " << gen(val);
      }

      if (val != params.back()) {
        code_ << ", ";
      }
    }

    // Global buffers
    for (auto allocate : kernel_summary.global_allocations) {
      TORCH_INTERNAL_ASSERT(allocate->buffer()->isA<kir::TensorView>());
      const auto tv = allocate->buffer()->as<kir::TensorView>();
      const auto& maybe_rfactor_domain = tv->domain()->hasRFactor()
          ? tv->domain()->rfactorDomain()
          : tv->domain()->rootDomain();
      const auto nDims = std::count_if(
          maybe_rfactor_domain.begin(),
          maybe_rfactor_domain.end(),
          [](const kir::IterDomain* id) {
            return !id->isReduction() &&
                id->iterType() != IterType::BroadcastWithoutStride;
          });
      code_ << ", Tensor<" << tv->dtype() << ", " << nDims << "> "
            << varName(tv);
    }

    // Kernels generating random numbers take extra (seed, offset) arguments
    if (kernel_summary.is_stochastic) {
      code_ << ", unsigned long long seed, unsigned long long offset";
    }

    code_ << ") ";
  }

  // Generates setup code which is executed before the kernel body
  void genPrologue() {
    const auto& kernel_summary = kernel_->summary();

    // Random number generator (optional)
    if (kernel_summary.is_stochastic) {
      indent() << "const int idx = blockIdx.x*blockDim.x + threadIdx.x;\n";
      indent() << "Philox rnd(seed, idx, offset);\n";
    }

    // Do we have any dynamic shared memory buffers?
    const bool has_dynamic_smem =
        !kernel_summary.dynamic_smem_allocations.empty();

    // Do we have any reductions?
    const bool has_reductions = kernel_summary.has_block_reductions ||
        kernel_summary.number_of_grid_reductions > 0;

    // Shared memory
    if (has_dynamic_smem || has_reductions) {
      indent() << "alignas("
#ifndef __HIP_PLATFORM_HCC__
               << dataTypeSize(kernel_summary.largest_smem_data_type)
#else
               << 8 // for HIP, we want 8-aligned even for smaller datatypes
#endif
               << ") extern __shared__ char array[];\n";

      if (has_dynamic_smem) {
        indent() << "unsigned offset = 0;\n";
      }

      if (has_reductions) {
        indent() << "void* shared_mem = array;\n";
        if (has_dynamic_smem) {
          indent() << "offset += "
                   << "((blockDim.x * blockDim.y * blockDim.z) * sizeof("
                   << kernel_summary.largest_smem_data_type << "));\n";
        }
      }
    }
  }

  void genBody() {
    for (auto expr : kernel_->topLevelExprs()) {
      expr->accept(this);
    }
  }

  void startBlock(bool continuation = false) {
    if (continuation) {
      code_ << "{\n";
    } else {
      indent() << "{\n";
    }
    ++block_nest_level_;
  }

  void endBlock(const char* sep = "\n") {
    --block_nest_level_;
    TORCH_CHECK(block_nest_level_ >= 0);
    indent() << "}" << sep;
  }

  std::ostream& indent() {
    for (int i = 0; i < block_nest_level_; ++i) {
      code_ << kTab;
    }
    return code_;
  }

  std::string gen(const kir::Node* node) {
    std::stringstream tmp_code;
    std::swap(tmp_code, code_);
    node->accept(this);
    std::swap(tmp_code, code_);
    return tmp_code.str();
  }

  // TODO(kir): consider automatic var naming
  std::string varName(const kir::Val* val) {
    std::string prefix = "";
    if (val->isA<kir::TensorView>()) {
      prefix = "T";
    } else {
      prefix = typePrefix(val->dtype());
    }

    std::stringstream value_name;
    if (val->name() != kInvalidStmName) {
      value_name << prefix << val->name();
    } else {
      value_name << "k" << prefix << val->id();
    }
    return value_name.str();
  }

  std::string genInline(const kir::Node* node) {
    const bool saved_inline = print_inline_;
    print_inline_ = true;
    const auto result = gen(node);
    print_inline_ = saved_inline;
    return result;
  }

  void visit(const kir::Bool* node) final {
    const auto def = node->definition();
    if (print_inline_ && def != nullptr) {
      code_ << "(" << gen(def) << ")";
    } else if (node->isConst()) {
      code_ << *node->value();
    } else {
      code_ << varName(node);
    }
  }

  void visit(const kir::Double* node) final {
    const auto def = node->definition();
    if (print_inline_ && def != nullptr) {
      code_ << "(" << gen(def) << ")";
    } else if (node->isConst()) {
      const int digits = std::numeric_limits<Double::ScalarType>::max_digits10;
      code_ << std::setprecision(digits) << *node->value();
    } else {
      code_ << varName(node);
    }
  }

  void visit(const kir::Int* node) final {
    const auto def = node->definition();
    if (print_inline_ && def != nullptr) {
      code_ << "(" << gen(def) << ")";
    } else if (node->isConst()) {
      code_ << *node->value();
    } else {
      code_ << varName(node);
    }
  }

  void visit(const kir::NamedScalar* node) final {
    code_ << node->name();
  }

  void visit(const kir::TensorIndex* node) final {
    code_ << varName(node->view()) << "[";

    bool first = true;
    for (auto* ind : node->indices()) {
      if (!ind->isZeroInt()) {
        if (!first) {
          code_ << " + ";
        }
        code_ << genInline(ind);
        first = false;
      }
    }

    if (first) {
      code_ << "0";
    }

    code_ << "]";
  }

  void visit(const kir::IterDomain* node) final {
    TORCH_INTERNAL_ASSERT(!"Unreachable");
  }

  void visit(const kir::TensorDomain* node) final {
    TORCH_INTERNAL_ASSERT(!"Unreachable");
  }

  void visit(const kir::TensorView* tv) final {
    TORCH_INTERNAL_ASSERT(!"Unreachable");
  }

  void visit(const kir::UnaryOp* node) final {
    const auto op_type = node->operation();
    if (!print_inline_) {
      if (op_type == UnaryOpType::VectorizeRead) {
        indent()
            << "*reinterpret_cast<"
            << "Array<" << node->out()->dtype() << ", "
            << genInline(
                   node->out()->as<kir::TensorIndex>()->view()->vectorSize())
            << ">*>"
            << "(&" << gen(node->out()) << ")";
      } else if (op_type == UnaryOpType::VectorizeWrite) {
        indent()
            << "*reinterpret_cast<"
            << "Array<" << node->out()->dtype() << ", "
            << genInline(
                   node->out()->as<kir::TensorIndex>()->view()->vectorSize())
            << ">*>"
            << "(&" << gen(node->out()) << ")";
      } else {
        indent() << gen(node->out());
      }

      if (!node->out()->isScalar() && !node->in()->isScalar()) {
        code_ << "\n";
        indent() << kTab;
      }
      code_ << " = ";
    }

    if (auto op = inline_op_str(op_type)) {
      if (alsoBooleanOperator(op_type) &&
          node->out()->dtype() == DataType::Bool) {
        code_ << stringifyBooleanOp(op_type) << gen(node->in());
      } else {
        code_ << *op << gen(node->in());
      }
    } else if (op_type == UnaryOpType::VectorizeRead) {
      code_ << "*reinterpret_cast<"
            << "Array<" << node->in()->dtype() << ", "
            << genInline(
                   node->in()->as<kir::TensorIndex>()->view()->vectorSize())
            << ">*>"
            << "(&" << gen(node->in()) << ")";
    } else if (op_type == UnaryOpType::VectorizeWrite) {
      // code_ << "vec_" << gen(node->in());
      code_ << "*reinterpret_cast<"
            << "Array<" << node->in()->dtype() << ", "
            << genInline(
                   node->in()->as<kir::TensorIndex>()->view()->vectorSize())
            << ">*>"
            << "(&" << gen(node->in()) << ")";
    } else {
      if (op_type == UnaryOpType::Cast) {
        const auto cast_str =
            cast_func_str({node->in()->dtype(), node->out()->dtype()});
        code_ << cast_str.value();
      } else {
        code_ << op_type;
        if (needFloatSuffix(op_type) &&
            node->out()->dtype() == DataType::Float) {
          code_ << "f";
        }
      }

      code_ << "(";
      if (op_type == UnaryOpType::RandLike) {
        code_ << "rnd";
      } else {
        code_ << gen(node->in());
      }
      code_ << ")";
    }

    if (!print_inline_) {
      code_ << ";\n";
    }
  }

  std::string genBinaryOp(
      BinaryOpType op_type,
      kir::Val* out,
      const std::string& lhs,
      const std::string& rhs) {
    std::stringstream expr;
    if (auto op = inline_op_str(op_type)) {
      expr << lhs << " ";
      if (alsoBooleanOperator(op_type) && out->dtype() == DataType::Bool) {
        expr << stringifyBooleanOp(op_type);
      } else {
        expr << *op;
      }
      expr << " " << rhs;
    } else {
      expr << op_type;
      if (needFloatSuffix(op_type) && out->dtype() == DataType::Float) {
        expr << "f";
      }
      expr << "(" << lhs << ", " << rhs << ")";
    }
    return expr.str();
  }

  // If one argument is a tensorview and the other is a scalar, make sure we
  // cast the scalar to the tensorview type
  std::string scalarCast(kir::Val* lhs, kir::Val* rhs) {
    // If neither are scalars return
    if (!((lhs->isScalar() || rhs->isScalar()) &&
          (lhs->isA<kir::TensorIndex>() || rhs->isA<kir::TensorIndex>()))) {
      return "";
    }

    // Looking for mixed tensorview scalar options where types don't match
    // but are either both floating or both int types. We should cast
    // scalar to tensorview type in these instances.
    auto lhs_t = lhs->dtype();
    auto rhs_t = rhs->dtype();

    // If same type, don't cast anything
    if (lhs_t == rhs_t) {
      return "";
    }

    // Don't do anything when dealing with bools
    if (lhs_t == DataType::Bool || rhs_t == DataType::Bool) {
      return "";
    }

    // Mixing floating and int combination
    if ((isFloatingPointType(lhs_t) != isFloatingPointType(rhs_t)) ||
        (isIntegralType(lhs_t) != isIntegralType(rhs_t))) {
      return "";
    }

    std::stringstream cast;
    cast << "(" << (lhs->isA<TensorView>() ? rhs_t : lhs_t) << ") ";
    return cast.str();
  }

  void visit(const kir::BinaryOp* node) final {
    const auto op_type = node->operation();
    if (print_inline_) {
      // Inline expression: `lhs op rhs`
      code_ << genBinaryOp(
          op_type, node->out(), gen(node->lhs()), gen(node->rhs()));
    } else {
      indent() << gen(node->out());
      if (node->out()->isScalar()) {
        // Single line: `out = lhs op rhs;`
        code_ << " = "
              << genBinaryOp(
                     op_type, node->out(), gen(node->lhs()), gen(node->rhs()));
      } else {
        // Split TensorView expressions across multiple lines:
        //
        // out
        //    =  lhs
        //    op rhs;
        //

        auto cast = scalarCast(node->lhs(), node->rhs());
        if (auto op = inline_op_str(op_type)) {
          code_ << "\n";
          indent() << kTab << "= " << (node->lhs()->isScalar() ? cast : "")
                   << gen(node->lhs()) << "\n";
          indent() << kTab;
          if (alsoBooleanOperator(op_type) &&
              node->out()->dtype() == DataType::Bool) {
            code_ << stringifyBooleanOp(op_type);
          } else {
            code_ << *op;
          }
          code_ << " " << (node->rhs()->isScalar() ? cast : "")
                << gen(node->rhs());
        } else {
          if (integer_op_str(op_type) && isIntegralType(node->out()->dtype())) {
            auto int_op = integer_op_str(op_type);
            code_ << " = " << *int_op << "(\n";
          } else {
            code_ << " = " << op_type << "(\n";
          }
          indent() << kTab << (node->lhs()->isScalar() ? cast : "")
                   << gen(node->lhs()) << ",\n";
          indent() << kTab << (node->rhs()->isScalar() ? cast : "")
                   << gen(node->rhs()) << ")";
        }
      }
      code_ << ";\n";
    }
  }

  void visit(const kir::TernaryOp* node) final {
    if (!print_inline_) {
      indent() << gen(node->out());
      if (!node->out()->isScalar()) {
        code_ << "\n";
        indent() << kTab;
      }
      code_ << " = ";
    }

    code_ << node->operation() << "(" << gen(node->in1()) << ", "
          << gen(node->in2()) << ", " << gen(node->in3()) << ")";

    if (!print_inline_) {
      code_ << ";\n";
    }
  }

  std::string genReductionOp(BinaryOpType op_type, kir::Val* out) {
    std::stringstream lambda;
    DataType data_type = out->dtype();
    lambda << "[](" << data_type << " &a, " << data_type << " b) "
           << "{ a = " << genBinaryOp(op_type, out, "a", "b") << "; }";
    return lambda.str();
  }

  void visit(const kir::BroadcastOp* node) final {
    TORCH_INTERNAL_ASSERT(node->out()->isA<kir::TensorIndex>());
    const auto tensor_index = node->out()->as<kir::TensorIndex>();

    const ParallelTypeBitmap domains = ir_utils::getParallelBroadcastDomains(
        tensor_index->view()->fuserTv(), kernel_->predicateMap());

    const bool thread_x = domains.get(ParallelType::TIDx);
    const bool thread_y = domains.get(ParallelType::TIDy);
    const bool thread_z = domains.get(ParallelType::TIDz);
    const bool block_x = domains.get(ParallelType::BIDx);
    const bool block_y = domains.get(ParallelType::BIDy);
    const bool block_z = domains.get(ParallelType::BIDz);

    const bool grid_broadcast_needed = block_x || block_y || block_z;
    const bool block_broadcast_needed = thread_x || thread_y || thread_z;

    TORCH_INTERNAL_ASSERT(
        !grid_broadcast_needed,
        "Parallel broadcast across blocks not supported");

    if (block_broadcast_needed) {
      const auto data_type = node->out()->dtype();
      indent() << "broadcast::blockBroadcast<" << (thread_x ? "true" : "false")
               << ", " << (thread_y ? "true" : "false") << ", "
               << (thread_z ? "true" : "false") << ">(\n";
      indent() << kTab << gen(node->out()) << ",\n";
      indent() << kTab << gen(node->in()) << ",\n";
      indent() << kTab << "static_cast<" << data_type << "*>(shared_mem));\n";
    } else {
      indent() << gen(node->out()) << "\n";
      indent() << kTab << " = " << gen(node->in()) << ";\n";
    }
  }

  void visit(const kir::ReductionOp* node) final {
    TORCH_INTERNAL_ASSERT(node->out()->isA<kir::TensorIndex>());

    const auto out = node->out()->as<kir::TensorIndex>();
    const auto domain = out->view()->domain();

    const bool has_block_reduce = domain->hasBlockReduction();
    const bool has_grid_reduce = domain->hasGridReduction();

    if (!has_block_reduce && !has_grid_reduce) {
      const auto gen_out = gen(out);
      const auto op_type = node->operation();
      indent() << gen_out << " = "
               << genBinaryOp(op_type, out, gen_out, gen(node->in())) << ";\n";
      return;
    }

    const auto par_domains = node->getParallelReductionDomains();
    const bool tidx = par_domains.find(ParallelType::TIDx) != par_domains.end();
    const bool tidy = par_domains.find(ParallelType::TIDy) != par_domains.end();
    const bool tidz = par_domains.find(ParallelType::TIDz) != par_domains.end();

    const auto data_type = node->out()->dtype();
    const auto op_type = node->operation();

    if (has_block_reduce) {
      if (has_grid_reduce) {
        indent() << data_type << " "
                 << "block_result"
                 << ";\n";
      }
      indent() << "blockReduce<" << (tidx ? "true" : "false") << ", "
               << (tidy ? "true" : "false") << ", " << (tidz ? "true" : "false")
               << ">(\n";
      if (has_grid_reduce) {
        indent() << kTab << "block_result"
                 << ",\n";
      } else {
        indent() << kTab << gen(node->out()) << ",\n";
      }
      indent() << kTab << gen(node->in()) << ",\n";
      indent() << kTab << genReductionOp(op_type, node->out()) << ",\n";
      indent() << kTab << "threadIdx,\n";
      indent() << kTab << "blockDim,\n";
      indent() << kTab << "static_cast<" << data_type << "*>(shared_mem),\n";
      if (node->predicate() == nullptr) {
        indent() << kTab << "true,\n";
      } else {
        indent() << kTab << genInline(node->predicate()) << ",\n";
      }
      indent() << kTab << data_type << "(" << genInline(node->init())
               << "));\n";
    }
  }

  std::string generateGridReduceTemplateFlags(
      const kir::ReductionOp* rop,
      const ParallelTypeBitmap& thread_pred) {
    const auto par_domains = rop->getParallelReductionDomains();
    const std::array<ParallelType, 6> ptypes{
        ParallelType::BIDx,
        ParallelType::BIDy,
        ParallelType::BIDz,
        ParallelType::TIDx,
        ParallelType::TIDy,
        ParallelType::TIDz};
    std::stringstream flags;
    for (const ParallelType pt : ptypes) {
      const bool parallel_reduction = par_domains.find(pt) != par_domains.end();
      const bool pred = thread_pred.get(pt);
      TORCH_INTERNAL_ASSERT(
          !(parallel_reduction && pred), "Cannot reduce predicated axis: ", pt);
      bool flag = false;
      // Currently assumed that no dimensions parallelized with blocks
      // are predicated. This assumption may be lifted, but
      // gridReduction would need some changes.
      if (isParallelTypeBlockDim(pt)) {
        TORCH_INTERNAL_ASSERT(
            !pred, "Predication on block dimensions not allowed: ", pt);
        flag = parallel_reduction;
      } else {
        flag = !pred && !parallel_reduction;
      }
      if (pt != ptypes[0]) {
        flags << ", ";
      }
      flags << (flag ? "true" : "false");
    }
    return flags.str();
  }

  void visit(const kir::GridReduction* node) final {
    const auto rop = node->reduction_op();
    TORCH_INTERNAL_ASSERT(rop->out()->isA<kir::TensorIndex>());

    const auto out = rop->out()->as<kir::TensorIndex>();
    const auto domain = out->view()->domain();
    TORCH_INTERNAL_ASSERT(domain->hasGridReduction());

    const auto data_type = rop->out()->dtype();
    const auto op_type = rop->operation();

    TORCH_INTERNAL_ASSERT(
        node->reduction_buffer()->buffer()->isA<kir::TensorView>());
    TORCH_INTERNAL_ASSERT(
        node->sync_buffer()->buffer()->isA<kir::TensorView>());
    const auto work_buffer =
        node->reduction_buffer()->buffer()->as<kir::TensorView>();
    const auto sync_buffer =
        node->sync_buffer()->buffer()->as<kir::TensorView>();

    const std::string flags_str =
        generateGridReduceTemplateFlags(rop, node->threadPredicate());

    // Since block-level reduction is already done, those dimensions
    // with tidx/y/z being true do not participate in the grid reduction.
    indent() << kir::GridReduction::getPredicateFlagName(out->view()) << " = "
             << "reduction::gridReduce<" << flags_str << ">(\n";
    indent() << kTab << gen(rop->out()) << ",\n";
    if (domain->hasBlockReduction()) {
      indent() << kTab << "block_result"
               << ",\n";
    } else {
      indent() << kTab << gen(rop->in()) << ",\n";
    }
    indent() << kTab << genReductionOp(op_type, out) << ",\n";
    indent() << kTab << "&" << varName(work_buffer) << "[0],\n";
    indent() << kTab << varName(sync_buffer) << ",\n";
    indent() << kTab << "static_cast<" << data_type << "*>(shared_mem),\n";
    if (node->predicate() == nullptr) {
      indent() << kTab << "true,\n";
    } else {
      indent() << kTab << genInline(node->predicate()) << ",\n";
    }
    indent() << kTab << data_type << "("
             << genInline(node->reduction_op()->init()) << "));\n";
  }

  void handleScope(const kir::Scope& scope) {
    for (auto expr : scope.exprs()) {
      expr->accept(this);
    }
  }

  void visit(const kir::ForLoop* node) final {
    // TODO(kir): handle this during lowering
    if (node->iter_domain()->isThread() || node->iter_domain()->isBroadcast()) {
      handleScope(node->body());
      return;
    }

    const auto gen_index = gen(node->index());
    const auto gen_start = genInline(node->iter_domain()->start());
    const auto gen_extent = genInline(node->iter_domain()->extent());
    const auto gen_offset = genInline(node->offset());
    indent() << "for(size_t " << gen_index << " = " << gen_start << "; "
             << gen_index << " < " << gen_extent << "; " << gen_index
             << " += " << gen_offset << ") ";

    startBlock(true);
    handleScope(node->body());
    endBlock();
  }

  void visit(const kir::IfThenElse* node) final {
    // If predicate condition is true, print only the "then" body
    if (node->cond()->value().value_or(false)) {
      handleScope(node->thenBody());
      return;
    }

    indent() << "if (" << genInline(node->cond()) << ") ";

    // "then" block
    startBlock(true);
    handleScope(node->thenBody());

    // "else" block (optional)
    if (node->hasElse()) {
      endBlock(" else ");
      startBlock(true);
      handleScope(node->elseBody());
    }

    endBlock();
  }

  // TODO(kir): fold initialization into Allocate
  void visit(const kir::Allocate* node) final {
    const auto buffer_dtype = node->buffer()->dtype();

    if (!node->buffer()->isA<kir::TensorView>()) {
      indent() << buffer_dtype << " " << gen(node->buffer()) << ";\n";
      return;
    }

    const auto tv = node->buffer()->as<kir::TensorView>();
    TORCH_INTERNAL_ASSERT(tv->domain()->nDims() > 0);

    const auto size = node->size();
    TORCH_INTERNAL_ASSERT(size != nullptr);

    if (node->alias() != nullptr) {
      // Allocate alias another Allocate node
      const auto alias_tv = node->alias()->buffer()->as<kir::TensorView>();
      indent() << "// Alias Allocation - " << node->memoryType() << "\n";
      indent() << buffer_dtype << "* " << varName(tv) << " = "
               << varName(alias_tv) << ";\n";
    } else {
      // Standard Memory Allocation
      switch (node->memoryType()) {
        case MemoryType::Global:
          indent() << "// Allocate global tensor " << varName(tv) << "\n";
          break;
        case MemoryType::Shared:
          if (kir::ExpressionEvaluator::isConst(size)) {
            // Static shared memory
            indent() << "__shared__ " << buffer_dtype << " " << varName(tv)
                     << "[" << genInline(size) << "];\n";
          } else {
            // Align Offset Position
            indent() << "offset = alignBufferSize(offset,"
                     << dataTypeSize(buffer_dtype) << ");\n";
            // Shared Memory Pointer
            indent() << buffer_dtype << "* " << varName(tv)
                     << " = reinterpret_cast<" << buffer_dtype << "*>"
                     << "(array + offset);\n";
            // Increment Offset Position
            indent() << "offset += (" << genInline(size) << " * sizeof("
                     << buffer_dtype << "));\n";
          }
          break;
        case MemoryType::Local:
          indent() << buffer_dtype << " " << varName(tv) << "["
                   << genInline(size) << "];\n";
          break;
        default:
          TORCH_INTERNAL_ASSERT(false, "Unexpected memory type");
      }
    }
  }

  void visit(const kir::Sync* node) final {
    indent() << "__syncthreads();\n";
  }

 private:
  std::stringstream code_;
  const kir::Kernel* kernel_;
  int block_nest_level_ = 0;

  // TODO(kir): replace with explicit assignment statements
  bool print_inline_ = false;
};

} // namespace

std::string generateCudaKernel(
    const kir::Kernel* kernel,
    const std::string& kernel_name) {
  FUSER_PERF_SCOPE("generateCudaKernel");
  return CudaKernelGenerator::generateKernelDefinition(kernel, kernel_name);
}

} // namespace codegen
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
