
#include <torch/csrc/jit/codegen/cuda/executor.h>

#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/executor_kernel_arg.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

int FusionExecutor::fusion_id_counter_ = 0; // NOLINT

std::string FusionExecutor::getStructuredCode(const std::string& kernel) {
  // generating cuda code;
  std::string code = "";
#ifdef __HIP_PLATFORM_HCC__
  code += std::string("#include <hip/hip_runtime.h>\n") +
      std::string("#include <hip/hip_fp16.h>\n");
#endif
  code += std::string("namespace ") + FusionExecutor::kernelNamespace() +
      " {\n" + executor_utils::kernelPreamble() + kernel + "}\n";

  if (isDebugDumpEnabled(DebugDumpOption::CudaKernel)) {
    std::cout << "\n======= Codegen output for kernel: " << kernelName()
              << " =======\n\n"
              << kernel << "\n======================================\n\n";
  } else if (isDebugDumpEnabled(DebugDumpOption::CudaFull)) {
    std::cout << "\n======= Codegen output for kernel: " << kernelName()
              << " =======\n\n"
              << code << "\n======================================\n\n";
  }

  return code;
}

// TODO: come up with a more user friendly interface
void FusionExecutor::debugCompileFusionFromStr(
    Fusion* fusion,
    const std::string& code,
    const std::string& name,
    int id,
    CompileOptions options) {
  fusion_ = *fusion;
  FusionGuard fg(&fusion_);
  options_ = options;

  if (isDebugDumpEnabled(DebugDumpOption::FusionIr)) {
    fusion->print();
  } else if (isDebugDumpEnabled(DebugDumpOption::FusionIrMath)) {
    fusion->printMath();
  }

  if (isDebugDumpEnabled(DebugDumpOption::CudaFull)) {
    std::cout << "\n==== codegen output for kernel: " << kernelName()
              << " ====" << std::endl
              << code << std::endl
              << "======================================\n"
              << std::endl;
  }

  setUsedTVs();

  fusion_id_ = id;
  lowered_ = GpuLower(&fusion_);
  const auto kernel = lowered_.kernel();

  if (isDebugDumpEnabled(DebugDumpOption::KernelIr)) {
    kernel->print();
  }

  const auto& kernel_summary = kernel->summary();

  if (!kernel_summary.static_smem_allocations.empty()) {
    kir::ExpressionEvaluator static_evaluator;
    const auto static_smem_size = computeSharedMemory(
        static_evaluator, kernel_summary.static_smem_allocations);
    TORCH_INTERNAL_ASSERT(
        static_smem_size < max_device_smem,
        "The static shared memory allocation is larger than available memory.");
  }

  compiled_kernel_ = executor_utils::nvrtcCompile(code, name, fusion_id_);
  TORCH_INTERNAL_ASSERT(
      fusion_id_ > 0, "assign a fusion_id_ <= 0 is not accepted.");
}

void FusionExecutor::compileFusion(Fusion* fusion, CompileOptions options) {
  FUSER_PERF_SCOPE("compileFusion");

  TORCH_INTERNAL_ASSERT(
      !fusion->outputs().empty(), "No output found for this kernel, aborting.");

  for (auto out : fusion->outputs()) {
    TORCH_INTERNAL_ASSERT(
        out->getValType() == ValType::TensorView,
        "Output types from fusions that are not tensors are not supported at this point.");
  }

  if (isDebugDumpEnabled(DebugDumpOption::FusionIr)) {
    fusion->print();
  } else if (isDebugDumpEnabled(DebugDumpOption::FusionIrMath)) {
    fusion->printMath();
  }

  // Clone the fusion so we can store it
  fusion_ = *fusion;
  FusionGuard fg(&fusion_);
  options_ = options;
  c10::DeviceGuard dg(options_.device);

  TORCH_INTERNAL_ASSERT(
      options.device.is_cuda(), "Provided device to CUDA fuser is the CPU.");
  max_device_smem =
      at::cuda::getDeviceProperties(options.device.index())->sharedMemPerBlock;

  setUsedTVs();

  fusion_id_ = ++fusion_id_counter_;
  lowered_ = GpuLower(&fusion_);
  const auto kernel = lowered_.kernel();

  if (isDebugDumpEnabled(DebugDumpOption::KernelIr)) {
    kernel->print();
  }

  const auto kernel_code = codegen::generateCudaKernel(kernel, kernelName());
  const auto structured_code = getStructuredCode(kernel_code);

  const auto& kernel_summary = kernel->summary();

  if (!kernel_summary.static_smem_allocations.empty()) {
    kir::ExpressionEvaluator static_evaluator;
    const auto static_smem_size = computeSharedMemory(
        static_evaluator, kernel_summary.static_smem_allocations);
    TORCH_INTERNAL_ASSERT(
        static_smem_size < max_device_smem,
        "The static shared memory allocation is larger than available memory.");
  }

  TORCH_INTERNAL_ASSERT(
      !kernel_summary.has_dynamic_local_memory_allocations,
      "Allocations must be based on constant integers for local memory.");

  TORCH_CHECK(
      kernel_summary.number_of_grid_reductions <= 1,
      "Multiple grid reductions in a fusion is not supported");

  TORCH_CHECK(
      !kernel_summary.has_grid_reduction_in_loop,
      "Grid reduction must not be placed inside a loop.");

  compiled_kernel_ = executor_utils::nvrtcCompile(
      structured_code,
      (kernelNamespace() + "::" + kernelName()).c_str(),
      fusion_id_);
  TORCH_INTERNAL_ASSERT(
      fusion_id_ > 0, "failed to assign a fusion_id_ after compilation.");
}

namespace {

at::Tensor inferAndAlloc(
    const kir::TensorView* tv,
    kir::ExpressionEvaluator& expr_eval,
    const CompileOptions& options,
    bool zero_init = false) {
  FUSER_PERF_SCOPE("inferAndAlloc");

  std::vector<int64_t> sizes;

  const auto domain = tv->domain();
  const auto maybe_rfactor_domain =
      domain->hasRFactor() ? domain->rfactorDomain() : domain->rootDomain();

  for (const auto id : maybe_rfactor_domain) {
    if (id->isReduction() ||
        id->iterType() == IterType::BroadcastWithoutStride) {
      continue;
    }
    const auto inferred_val = expr_eval.evaluate(id->rawExtent());
    TORCH_INTERNAL_ASSERT(
        inferred_val.has_value(),
        "Could not launch kernel as program could not infer ",
        kir::toString(id->rawExtent()),
        " for the buffer ",
        kir::toString(tv));
    sizes.push_back(inferred_val.value());
  }

  const auto at_type = data_type_to_aten(tv->dtype());

  if (zero_init) {
    const auto tensor_options =
        at::TensorOptions().dtype(at_type).device(options.device);
    c10::IntArrayRef isizes(sizes);
    return at::zeros(isizes, tensor_options);
  } else {
    c10::IntArrayRef isizes(sizes);
    // Non Variable type guard for empty_cuda call
    at::AutoNonVariableTypeMode non_variable_type_mode;
    return at::native::empty_cuda(
        isizes, at_type, c10::nullopt, options.device, c10::nullopt);
  }
}

} // namespace

uint64_t FusionExecutor::computeSharedMemory(
    kir::ExpressionEvaluator& expr_eval,
    const std::vector<const kir::Allocate*>& buffers,
    bool align_padding,
    uint64_t total) {
  FUSER_PERF_SCOPE("computeSharedMemory");
  for (auto smem_alloc : buffers) {
    // If this buffer aliases another buffer,
    // then do not allocate memory for this buffer.
    if (smem_alloc->alias() == nullptr) {
      const auto inferred_val = expr_eval.evaluate(smem_alloc->size());
      if (inferred_val.has_value()) {
        const uint64_t data_size = dataTypeSize(smem_alloc->buffer()->dtype());
        // Add padding to align dynamic shared memory
        if (align_padding) {
          total = ceilDiv(total, data_size) * data_size;
        }
        total += inferred_val.value() * data_size;
      } else {
        TORCH_INTERNAL_ASSERT(
            false,
            "Failed to evaluate the size ",
            smem_alloc->size(),
            " of shared memory buffer - T",
            smem_alloc->buffer()->name());
      }
    }
  }
  return total;
}

LaunchParams FusionExecutor::computeLaunchParams(
    const LaunchParams& launch_constraints,
    kir::ExpressionEvaluator& expr_eval) {
  FUSER_PERF_SCOPE("computeLaunchParams");

  LaunchParams launch_params;

  // Lets collect all IterDomains that are bound to a thread binding
  std::unordered_map<ParallelType, std::vector<const kir::Val*>, TypeHash>
      parallel_iter_extents;
  for (auto tv : getUsedTVs()) {
    for (auto id : tv->domain()->domain()) {
      if (id->isThread() && !id->isBroadcast()) {
        // TODO(kir): we should rewrite this logic based on the Kernel object
        auto kir_extent = lowered_.lowerValue(id->rawExtent());
        const auto it = parallel_iter_extents.find(id->getParallelType());
        if (it != parallel_iter_extents.end()) {
          it->second.push_back(kir_extent);
        } else {
          parallel_iter_extents[id->getParallelType()] = {kir_extent};
        }
      }
    }
  }

  // If any dimension was set in launch constraints we need to run through
  // IterDomains that have been parallelized, and bind those values. Or make
  // sure if they could be inferred the inference matches what was set.
  for (auto& entry : parallel_iter_extents) {
    auto p_type = entry.first;
    if (launch_constraints.hasDim(p_type)) {
      auto parallel_extents = entry.second;
      for (auto extent : parallel_extents) {
        auto inferred_val = expr_eval.evaluate(extent);
        if (inferred_val.has_value()) {
          // This value could have been inferred, make sure it was set right.
          bool valid =
              inferred_val.value() == launch_constraints.getDim(p_type) ||
              launch_constraints.getRawVal(p_type) == -1;
          if (!useFallback() && !valid) {
            TORCH_WARN_ONCE(
                "Cannot validate parallelization scheme, "
                "this may be due to mixed broadcast axes that are parallelized.");
          }
        } else {
          // Bind the launch constraint into our evaluation context
          expr_eval.bind(extent, launch_constraints.getDim(p_type));
          launch_params.bind(launch_constraints.getDim(p_type), p_type);
        }
      }
    }
  }

  // Run through the rest of the parallel IterDomains and infer their size
  for (auto& entry : parallel_iter_extents) {
    auto p_type = entry.first;
    auto parallel_extents = entry.second;
    // Select the maxmimum value out of all the parallel extents
    int64_t maximum_value = std::numeric_limits<int64_t>::min();
    for (auto extent : parallel_extents) {
      const auto val = expr_eval.evaluate(extent);
      TORCH_INTERNAL_ASSERT(
          val.has_value(),
          "Tried to evaluate the extent of ",
          p_type,
          " to set launch bounds but could not.");
      maximum_value = std::max(maximum_value, *val);
    }
    launch_params.bind(maximum_value, p_type);
  }

  const auto kernel = lowered_.kernel();
  const auto& kernel_summary = kernel->summary();

  // Calculate Dynamic Shared Memory Size
  // Add workspace for reduction and broadcast
  uint64_t reduction_broadcast_workspace = 0;
  const bool has_workspace = kernel_summary.has_block_reductions ||
      kernel_summary.number_of_grid_reductions > 0 ||
      kernel_summary.has_block_broadcasts;
  if (has_workspace &&
      kernel_summary.largest_smem_data_type != DataType::Null) {
    // Not using nThreads here since it does not handle uninitialized value

    // TODO: here is an optimization opportunity since welford uses int64_t for
    // N while the data type is not neccessarily double. But it may need more
    // work on the alignment
    const int welford_factor =
        kernel_summary.has_block_welford || kernel_summary.has_grid_welford ? 3
                                                                            : 1;
    reduction_broadcast_workspace =
        dataTypeSize(kernel_summary.largest_smem_data_type) * welford_factor *
        launch_params.bdimx() * launch_params.bdimy() * launch_params.bdimz();
  }

  const uint64_t dynamic_smem_size = computeSharedMemory(
      expr_eval,
      kernel_summary.dynamic_smem_allocations,
      true,
      reduction_broadcast_workspace);

  const uint64_t static_smem_size =
      computeSharedMemory(expr_eval, kernel_summary.static_smem_allocations);

  TORCH_INTERNAL_ASSERT(
      (dynamic_smem_size + static_smem_size) < max_device_smem,
      "The total shared memory allocation is larger than available memory.");
  launch_params.setSmem(dynamic_smem_size);

  return launch_params;
}

FusionExecutor::GlobalBuffers FusionExecutor::allocGlobalVals(
    kir::ExpressionEvaluator& expr_eval) {
  FUSER_PERF_SCOPE("allocGlobalVals");
  GlobalBuffers global_buffers;
  const auto& kernel_summary = lowered_.kernel()->summary();
  for (auto alloc : kernel_summary.global_allocations) {
    TORCH_INTERNAL_ASSERT(
        alloc->buffer()->isA<kir::TensorView>(),
        "Cannot allocate global buffers that are not tensors.");
    if (!alloc->zeroInit()) {
      global_buffers.empty_buffers.push_back(inferAndAlloc(
          alloc->buffer()->as<kir::TensorView>(), expr_eval, options_, false));
    } else {
      global_buffers.zero_buffers.push_back(inferAndAlloc(
          alloc->buffer()->as<kir::TensorView>(), expr_eval, options_, true));
    }
  }

  return global_buffers;
}

std::vector<at::Tensor> FusionExecutor::allocOutputs(
    kir::ExpressionEvaluator& expr_eval) {
  FUSER_PERF_SCOPE("allocOutputs");
  const auto kernel = lowered_.kernel();
  std::vector<at::Tensor> outputs;
  for (auto output : kernel->outputs()) {
    TORCH_INTERNAL_ASSERT(
        output->isA<kir::TensorView>(),
        "Cannot allocate outputs that are not tensors.");
    outputs.push_back(inferAndAlloc(
        output->as<kir::TensorView>(), expr_eval, options_, false));
  }
  return outputs;
}

void FusionExecutor::setUsedTVs() {
  used_tvs_.clear();
  auto used_vals = DependencyCheck::getAllValsBetween(
      {fusion_.inputs().begin(), fusion_.inputs().end()}, fusion_.outputs());
  for (auto val : used_vals) {
    if (val->getValType().value() == ValType::TensorView) {
      used_tvs_.push_back(val->as<TensorView>());
    }
  }
}

std::vector<at::Tensor> FusionExecutor::runFusion(
    const at::ArrayRef<IValue>& inputs,
    const std::vector<at::Tensor>& outputs,
    const LaunchParams& launch_constraints,
    const c10::optional<size_t>& opt_code) {
  FUSER_PERF_SCOPE("runFusion");

  TORCH_INTERNAL_ASSERT(
      fusion_id_ > 0, "Cannot run fusion, it was not compiled.");
  TORCH_INTERNAL_ASSERT(
      !opt_code.has_value() || outputs.empty(),
      "short cut input cache is not compatible with pre-allocated output");

  ExecutorEntry* executor_entry = nullptr;
  if (opt_code.has_value()) {
    executor_entry = &executor_entry_lookup_[*opt_code];
  }

  FusionGuard fg(&fusion_);
  c10::DeviceGuard dg(options_.device);
  auto stream = at::cuda::getCurrentCUDAStream();

  LaunchParams launch_params;
  std::vector<at::Tensor> allocated_outputs = outputs;
  GlobalBuffers global_buffers;
  uint64_t rand_offset = 0;

  if (executor_entry && executor_entry->init) {
    {
      // context manager to disable auto grad for `empty_cuda` calls later
      at::AutoNonVariableTypeMode non_variable_type_mode;
      // take the short-cut for launch if we see a recorded input set again
      launch_params = executor_entry->launch_params;
      for (size_t i = 0; i < executor_entry->output_sizes.size(); i++) {
        allocated_outputs.push_back(at::native::empty_cuda(
            executor_entry->output_sizes[i],
            executor_entry->output_types[i],
            c10::nullopt,
            options_.device,
            c10::nullopt));
      }
      for (size_t i = 0; i < executor_entry->empty_buffer_sizes.size(); i++) {
        global_buffers.empty_buffers.push_back(at::native::empty_cuda(
            executor_entry->empty_buffer_sizes[i],
            executor_entry->empty_buffer_types[i],
            c10::nullopt,
            options_.device,
            c10::nullopt));
      }
    }
    for (size_t i = 0; i < executor_entry->zero_buffer_sizes.size(); i++) {
      auto tensor_options = at::TensorOptions()
                                .dtype(executor_entry->zero_buffer_types[i])
                                .device(options_.device);
      global_buffers.zero_buffers.push_back(
          at::zeros(executor_entry->zero_buffer_sizes[i], tensor_options));
    }
    rand_offset = executor_entry->rand_offset;
  } else {
    // code path to take when either:
    //   1. no opt_code is provided or
    //   2. `executor_entry` is not initialized
    executor_utils::validateKernelInputs(&fusion_, inputs, options_.device);

    const auto kernel = lowered_.kernel();

    auto expr_eval = executor_utils::bindKernelInputs(inputs, kernel);

    launch_params = computeLaunchParams(launch_constraints, expr_eval);
    if (isDebugDumpEnabled(DebugDumpOption::LaunchParam)) {
      launch_params.print();
    }

    executor_utils::validateVectorizedTensors(
        &fusion_, inputs, outputs, lowered_, expr_eval);

    if (outputs.empty() || outputs.size() != fusion_.outputs().size()) {
      allocated_outputs = allocOutputs(expr_eval);
    } else {
      executor_utils::validateKernelOutputs(
          &fusion_, allocated_outputs, options_.device);
    }

    global_buffers = allocGlobalVals(expr_eval);

    if (kernel->summary().is_stochastic) {
      // NOTE: this is how we map offset to PW kernels in order to have
      // identical random number generator to match native PyTorch results.
      // But it doesn't really work as it takes assumption how threads are
      // binded but is not generally how we handle that in scheduler.
      // Refer to `Philox` in generated kernel to understand how the mapping
      // works.
      rand_offset = 4 *
          (std::ceil(
               allocated_outputs[0].numel() /
               (4.0 * 128 * launch_params.gdimx())) + // NOLINT
           1);
    }

    // This is the entry when we have provided `opt_code` but the entry has not
    // been initialized yet.
    if (executor_entry) {
      // record the the short-cut executor entry for the given input set;
      executor_entry->launch_params = launch_params;
      for (const auto& output : allocated_outputs) {
        executor_entry->output_sizes.push_back(output.sizes().vec());
        executor_entry->output_types.push_back(output.scalar_type());
      }
      for (const auto& buffer : global_buffers.empty_buffers) {
        executor_entry->empty_buffer_sizes.push_back(buffer.sizes().vec());
        executor_entry->empty_buffer_types.push_back(buffer.scalar_type());
      }
      for (const auto& buffer : global_buffers.zero_buffers) {
        executor_entry->zero_buffer_sizes.push_back(buffer.sizes().vec());
        executor_entry->zero_buffer_types.push_back(buffer.scalar_type());
      }
      executor_entry->rand_offset = rand_offset;
      executor_entry->init = true;
    }
  }

  KernelArgumentHolder kernel_arguments;
  kernel_arguments.push(inputs);
  kernel_arguments.push(allocated_outputs);
  kernel_arguments.push(global_buffers.empty_buffers);
  kernel_arguments.push(global_buffers.zero_buffers);
  if (lowered_.kernel()->summary().is_stochastic) {
    kernel_arguments.appendPhiloxRNGSeed(rand_offset);
  }

  cudaEvent_t start_event = {};
  cudaEvent_t finish_event = {};

  if (measure_kernel_time_) {
    cudaEventCreate(&start_event);
    cudaEventCreate(&finish_event);
    cudaEventRecord(start_event);
  }

  if (execute_kernel_) {
    FUSER_PERF_SCOPE("cuLaunchKernel");
    AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLaunchKernel(
        compiled_kernel_.function,
        launch_params.gdimx(),
        launch_params.gdimy(),
        launch_params.gdimz(),
        launch_params.bdimx(),
        launch_params.bdimy(),
        launch_params.bdimz(),
        launch_params.smem(),
        stream,
        kernel_arguments.getBuffer(),
        nullptr));
  }

  if (measure_kernel_time_) {
    cudaEventRecord(finish_event);
    cudaEventSynchronize(start_event);
    cudaEventSynchronize(finish_event);
    cudaEventElapsedTime(&kernel_time_ms_, start_event, finish_event);
  }

  return allocated_outputs;
}

void FusionExecutor::compileRtc(
    const std::string& code,
    const std::string& name,
    bool structured) {
  std::string scode;
  if (!structured) {
    scode = getStructuredCode(code);
  } else {
    scode = code;
  }
  fusion_id_ = 1;
  options_ = CompileOptions();
  compiled_kernel_ = executor_utils::nvrtcCompile(scode, name, fusion_id_);
}

void FusionExecutor::runRtc(
    const LaunchParams& launch_params,
    const std::vector<at::Tensor>& args) {
  FUSER_PERF_SCOPE("runFusion");

  c10::DeviceGuard dg(options_.device);
  auto stream = at::cuda::getCurrentCUDAStream();

  KernelArgumentHolder kernel_arguments;
  kernel_arguments.push(args);
  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLaunchKernel(
      compiled_kernel_.function,
      launch_params.gdimx(),
      launch_params.gdimy(),
      launch_params.gdimz(),
      launch_params.bdimx(),
      launch_params.bdimy(),
      launch_params.bdimz(),
      launch_params.smem(),
      stream,
      kernel_arguments.getBuffer(),
      nullptr));
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
