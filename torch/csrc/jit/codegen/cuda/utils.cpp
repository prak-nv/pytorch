
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <c10/util/string_view.h>

#include <cstdlib>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

auto parseDebugDumpOptions() {
  std::unordered_map<DebugDumpOption, bool> options_map = {
      {DebugDumpOption::FusionIr, false},
      {DebugDumpOption::FusionIrMath, false},
      {DebugDumpOption::KernelIr, false},
      {DebugDumpOption::CudaKernel, false},
      {DebugDumpOption::CudaFull, false},
      {DebugDumpOption::LaunchParam, false}};

  if (const char* dump_options = std::getenv("PYTORCH_NVFUSER_DUMP")) {
    c10::string_view options_view(dump_options);
    while (!options_view.empty()) {
      const auto end_pos = options_view.find_first_of(',');
      const auto token = options_view.substr(0, end_pos);
      if (token == "fusion_ir") {
        options_map[DebugDumpOption::FusionIr] = true;
      } else if (token == "fusion_ir_math") {
        options_map[DebugDumpOption::FusionIrMath] = true;
      } else if (token == "kernel_ir") {
        options_map[DebugDumpOption::KernelIr] = true;
      } else if (token == "cuda_kernel") {
        options_map[DebugDumpOption::CudaKernel] = true;
      } else if (token == "cuda_full") {
        options_map[DebugDumpOption::CudaFull] = true;
      } else if (token == "launch_param") {
        options_map[DebugDumpOption::LaunchParam] = true;
      } else {
        TORCH_CHECK(
            false,
            "Invalid debug dump option: '",
            token,
            "'\n  Available options: ",
            "fusion_ir, fusion_ir_math, kernel_ir, cuda_kernel, cuda_full, launch_param\n");
      }
      options_view = (end_pos != c10::string_view::npos)
          ? options_view.substr(end_pos + 1)
          : "";
    }
  }

  return options_map;
}

} // namespace

bool isDebugDumpEnabled(DebugDumpOption option) {
  const static auto dump_options = parseDebugDumpOptions();
  return dump_options.at(option);
}

bool useFallback() {
  const char* disable_fb_env = getenv("PYTORCH_NVFUSER_DISABLE_FALLBACK");
  return !(disable_fb_env ? atoi(disable_fb_env) : 0);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
