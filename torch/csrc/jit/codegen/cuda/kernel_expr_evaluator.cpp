
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>

#include "glfdc/expr_builder.hh"
#include "glfdc/eval.hh"

#include <iostream>

namespace {

namespace fuser_cuda = torch::jit::fuser::cuda;
using namespace torch::jit::fuser::cuda::kir;

class ExprBuilderVisitor : IrVisitor
{
public:
  explicit ExprBuilderVisitor(glfdc::ExpressionBuilder& bld);

  c10::optional<glfdc::Expr> build(const Val* value) {
    auto ref = build_operand(value);
    if (!glfdc::is_sexpr(ref))
       return c10::nullopt;
    return glfdc::Expr{&builder_->dag(), c10::get<glfdc::SExprRef>(ref)};
  }

  glfdc::Operand build_operand(const Val* value) {
    TORCH_CHECK(value->isScalar());
    TORCH_CHECK(value->dtype() == fuser_cuda::DataType::Int);

    if (value->isA<Int>())
    {
      if (value->isConst())
      {
         auto opt_val = value->as<Int>()->value();
         if (opt_val.has_value())
           return glfdc::Value(opt_val.value());
         else
           return builder_->get_binding(reinterpret_cast<uintptr_t>(value));
      }
    }

    current_operand_ = c10::nullopt;
    value->accept(this);
    TORCH_CHECK(current_operand_.has_value());
    auto ret = current_operand_.value();
    current_operand_ = c10::nullopt;
    return ret;
  }

private:
  void unhandled(const void*) final;
  void visit(const Int* value) final;
  void visit(const NamedScalar* named_scalar) final;
  void visit(const UnaryOp* unary_op) final;
  void visit(const BinaryOp* binary_op) final;

  glfdc::ExpressionBuilder *builder_;
  c10::optional<glfdc::Operand> current_operand_ = c10::nullopt;
};

ExprBuilderVisitor::ExprBuilderVisitor(glfdc::ExpressionBuilder& bld) : builder_(&bld)
{
}

void ExprBuilderVisitor::unhandled(const void*) {
  TORCH_INTERNAL_ASSERT(
      false, "Kernel IR expression evaluation reached an unsupported node");
}

void ExprBuilderVisitor::visit(const Int* value) {
  TORCH_INTERNAL_ASSERT(!value->isConst());
  if (auto def = value->definition()) {
    def->accept(this);
  }
}

void ExprBuilderVisitor::visit(const UnaryOp* unary_op) {

  using fuser_cuda::UnaryOpType;
  glfdc::Operand oper = build_operand(unary_op->in());
  switch (unary_op->operation())
  {
  case UnaryOpType::Neg:
    current_operand_ = builder_->create_sexpr(glfdc::OperatorKind::sub, glfdc::Value(0), oper);
    break;
  case UnaryOpType::Cast:
    current_operand_ = oper;
  default:
    break;
  }

  TORCH_INTERNAL_ASSERT(current_operand_.has_value() && "Unexpected operator type");
}

class KnownValuesCallback
{
  using hashmap_t = std::unordered_map<const Val*, Int::ScalarType>;

public:
  c10::optional<Int::ScalarType> operator() (uintptr_t binding) {
    auto it = known_values_->find(reinterpret_cast<const Val*>(binding));
    if (it == known_values_->end())
      return c10::nullopt;
    return it->second;
  }

  explicit KnownValuesCallback(hashmap_t& values) : known_values_(&values) {}

private:
  hashmap_t* known_values_ = nullptr;
};

void ExprBuilderVisitor::visit(const BinaryOp* binary_op) {
  auto lhs = build_operand(binary_op->lhs());
  auto rhs = build_operand(binary_op->rhs());

  using fuser_cuda::BinaryOpType;
  c10::optional<glfdc::OperatorKind> oper;

  switch (binary_op->operation())
  {
    case BinaryOpType::Add:
      oper = glfdc::OperatorKind::add;
      break;
    case BinaryOpType::Sub:
      oper = glfdc::OperatorKind::sub;
      break;
    case BinaryOpType::Mul:
      oper = glfdc::OperatorKind::mul;
      break;
    case BinaryOpType::Div:
      oper = glfdc::OperatorKind::div;
      break;
    case BinaryOpType::Mod:
      oper = glfdc::OperatorKind::mod;
      break;
    case BinaryOpType::CeilDiv:
      oper = glfdc::OperatorKind::ceildiv;
      break;
  }

  TORCH_INTERNAL_ASSERT(oper.has_value() && "Unexpected operator type");

  current_operand_ = builder_->create_sexpr(oper.value(), lhs, rhs);
}

void ExprBuilderVisitor::visit(const NamedScalar* scalar) {
  current_operand_ = builder_->get_binding(reinterpret_cast<uintptr_t>(scalar));
  // It's a legal expresison node so we must handle it
}
}

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {


ExpressionEvaluator::ExpressionEvaluator(std::unique_ptr<glfdc::EvalState> es) : builder_(new glfdc::ExpressionBuilder) {
  eval_state_ = std::move(es);
}

ExpressionEvaluator::~ExpressionEvaluator() = default;

ExpressionEvaluator::ExpressionEvaluator(ExpressionEvaluator&&) = default;
ExpressionEvaluator& ExpressionEvaluator::operator=(ExpressionEvaluator&&) = default;

void ExpressionEvaluator::bind(
    const Val* value,
    Int::ScalarType concrete_value) {
  TORCH_CHECK(value->isScalar());
  TORCH_CHECK(value->dtype() == DataType::Int);
  TORCH_CHECK(!value->isConst(), "Tried to bind to a constant value");
  TORCH_CHECK(
      value->definition() == nullptr,
      "Tried to bind to a value that is computed in the kernel IR");
  known_values_[value] = concrete_value;
}

c10::optional<Int::ScalarType> ExpressionEvaluator::evaluateMemoized(const glfdc::ExprEvaluator& expr)
{
  FUSER_PERF_SCOPE("kir::ExpressionEvaluator::evaluateMemoized");

  KnownValuesCallback cb{known_values_};
  TORCH_CHECK(eval_state_ != nullptr);

  return expr.evaluate(*eval_state_, cb);
}

c10::optional<Int::ScalarType> ExpressionEvaluator::evaluate(const Val* value) {
  FUSER_PERF_SCOPE("kir::ExpressionEvaluator::evaluate");

  TORCH_CHECK(value != nullptr);
  TORCH_CHECK(value->isScalar());
  TORCH_CHECK(value->dtype() == DataType::Int);

  // Const scalar?
  if (value->isScalar() && value->isConst()) {
    return value->as<Int>()->value();
  }

  // Is the value known (either explicit binding)?
  const auto pre_eval_it = known_values_.find(value);
  if (pre_eval_it != known_values_.end()) {
    return pre_eval_it->second;
  }
#if defined(FUSER_ALWAYS_GLFDC_EVAL)
  ExprBuilderVisitor visitor(*builder_);

  auto expr = visitor.build(value);
  if (!expr)
    return c10::nullopt;

  using namespace glfdc;

  if (eval_state_ == nullptr || builder_->reuse_count() != eval_size_)
  {
    auto mapping = glfdc::ReusedExprMapping::create_lazy_mapping(builder_->reuses());
    eval_state_.reset(new glfdc::EvalState(mapping));
    eval_size_ = builder_->reuse_count();
  }

  glfdc::ExprEvaluator eval{expr.value()};

  KnownValuesCallback cb{known_values_};

  return eval.evaluate(*eval_state_, cb);
#else
  value->accept(this);
  const auto post_eval_it = known_values_.find(value);
  return post_eval_it != known_values_.end()
      ? c10::optional<Int::ScalarType>(post_eval_it->second)
      : c10::nullopt;

#endif
}

bool ExpressionEvaluator::isConst(const Val* value) {
  return ExpressionEvaluator().evaluate(value).has_value();
}

void ExpressionEvaluator::print() const {
  std::cout << "\nEvaluation context\n";
  std::cout << "--------------------\n";
  for (const auto& kv : known_values_) {
    std::cout << toString(kv.first) << " = " << kv.second << "\n";
  }
  std::cout << "--------------------\n\n";
}

void ExpressionEvaluator::unhandled(const void*) {
  TORCH_INTERNAL_ASSERT(
      false, "Kernel IR expression evaluation reached an unsupported node");
}

void ExpressionEvaluator::visit(const Int* value) {
  TORCH_INTERNAL_ASSERT(!value->isConst());
  if (auto def = value->definition()) {
    def->accept(this);
  }
}

void ExpressionEvaluator::visit(const NamedScalar* named_scalar) {
  // It's a legal expresison node so we must handle it
}

void ExpressionEvaluator::visit(const UnaryOp* unary_op) {
  const auto in = evaluate(unary_op->in());
  if (in.has_value()) {
    switch (unary_op->operation()) {
      case UnaryOpType::Neg:
        known_values_[unary_op->out()] = -*in;
        break;
      case UnaryOpType::Cast:
        known_values_[unary_op->out()] = *in;
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

void ExpressionEvaluator::visit(const BinaryOp* binary_op) {
  const auto lhs = evaluate(binary_op->lhs());
  const auto rhs = evaluate(binary_op->rhs());
  if (lhs.has_value() && rhs.has_value()) {
    switch (binary_op->operation()) {
      case BinaryOpType::Add:
        known_values_[binary_op->out()] = *lhs + *rhs;
        break;
      case BinaryOpType::Sub:
        known_values_[binary_op->out()] = *lhs - *rhs;
        break;
      case BinaryOpType::Mul:
        known_values_[binary_op->out()] = *lhs * *rhs;
        break;
      case BinaryOpType::Div:
        TORCH_CHECK(*rhs != 0);
        known_values_[binary_op->out()] = *lhs / *rhs;
        break;
      case BinaryOpType::Mod:
        TORCH_CHECK(*rhs != 0);
        known_values_[binary_op->out()] = *lhs % *rhs;
        break;
      case BinaryOpType::CeilDiv:
        TORCH_CHECK(*rhs != 0);
        known_values_[binary_op->out()] = (*lhs + *rhs - 1) / *rhs;
        break;
      case BinaryOpType::And:
        known_values_[binary_op->out()] = Int::ScalarType(*lhs && *rhs);
        break;
      default:
        TORCH_CHECK(!"Unexpected operator type");
    }
  }
}

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
