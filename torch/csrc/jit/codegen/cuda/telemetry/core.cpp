#if defined(TRACY_ENABLE)

#include <torch/csrc/jit/codegen/cuda/telemetry/telemetry.h>

#include "tracy/client/TracyProfiler.hpp"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>


namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace telemetry {

namespace {

void wait_for_tracy() {
  using namespace std::literals::chrono_literals;

  if (!tracy::GetProfiler().IsConnected()) {
    fprintf(stderr, "Wait for tracy profiler.\n");
    do {
      std::this_thread::sleep_for(10ms);
    } while (!tracy::GetProfiler().IsConnected());
  }
}

struct CheckCallstackSupport
{
  CheckCallstackSupport() {
#if !defined(TRACY_HAS_CALLSTACK)
    fprintf(stderr, "Tracy has callstacks disabled!"
            "Those won't be available. No cookies for you today.\n");
#endif
  }
} g_check_callstack_once;

// NB: Tracy manual asks for string literals to be interned ie value of pointer is same for same string literals.
const char* g_fusion_cgen_mark = "Fusion::Codegen";
const char* g_fusion_exec_mark = "Fusion::Exec";

const char* mark(frame_mark m) noexcept {
  switch (m) {
  case frame_mark::codegen: return g_fusion_cgen_mark;
  case frame_mark::exec: return g_fusion_exec_mark;
  }
}

} // namespace anonymous

void fusion_start(telemetry::frame_mark m) {
  // Little convinence, allows to start tracy after we lunched test payload.
  // to enable:
  // $ export NVFUSER_WAIT_FOR_TRACY=1
  const char* wait_env_var = std::getenv("NVFUSER_WAIT_FOR_TRACY");

  if (wait_env_var != nullptr) {
    std::fprintf(stderr, "Warning! Waiting for tracy profiler, "
                 "your wall time measurements may be skewed for initial test run.\n");

    wait_for_tracy();
  }

  tracy::Profiler::SendFrameMark(mark(m), tracy::QueueType::FrameMarkMsgStart);
}

void fusion_end(frame_mark m) {
  tracy::Profiler::SendFrameMark(mark(m), tracy::QueueType::FrameMarkMsgEnd);
}

//For debugger
//[[clang::optnone]]
void trace_container_size(const char* name, std::size_t sz, const char* kind)
{
  tracy::Profiler::PlotData(name, std::int64_t(sz));
}

} // namespace telemetry
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

#endif
