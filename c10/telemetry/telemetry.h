#pragma once

// Thin abstraction layer over [tracy](https://github.com/wolfpld) profiler.
//
#include <torch/csrc/WindowsTorchApiMacro.h>

//#include <bitset>
//#include <list>
#include <map>
#include <set>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#if defined(TRACY_ENABLE)
#  include <tracy/client/TracyScoped.hpp>
#endif

// TODO: refactor, so c10 interfaces are more generic and build ours on top of that
namespace c10 {
namespace telemetry {

enum class frame_mark { codegen, exec };

// TODO: remove and replace by check in FUSER_PERF_SCOPE for "runFusion"/"compileFusion"???
#define FUSER_MARK_START_CGEN() ::c10::telemetry::fusion_start(\
  ::c10::telemetry::frame_mark::codegen)
#define FUSER_MARK_START_EXEC() ::c10::telemetry::fusion_start(\
  ::c10::telemetry::frame_mark::exec)

#define FUSER_MARK_END_CGEN() ::c10::telemetry::fusion_end(\
  ::c10::telemetry::frame_mark::codegen)
#define FUSER_MARK_END_EXEC() ::c10::telemetry::fusion_end(\
  ::c10::telemetry::frame_mark::exec)

#define FUSER_PERF_TRACE_SIZE(c) ::c10::telemetry::trace_container_size(#c, c);

#if defined(TRACY_ENABLE)

// TODO: extend and move out of this namespace
template <typename> struct ContainerKind { static constexpr const char* kind = "other"; };

template <typename Ty_> struct ContainerKind<std::vector<Ty_>> { static constexpr const char* kind = "std::vector"; };
template <> struct ContainerKind<std::vector<bool>> { static constexpr const char* kind = "std::vector<bool>"; };

template <typename K_, typename V_> struct ContainerKind<std::map<K_, V_>> { static constexpr const char* kind = "std::map"; };
template <typename K_, typename V_, typename H_> struct ContainerKind<std::unordered_map<K_, V_, H_>> { static constexpr const char* kind = "std::unordered_map"; };

template <typename K_> struct ContainerKind<std::set<K_>> { static constexpr const char* kind = "std::set"; };
template <typename K_, typename H_> struct ContainerKind<std::unordered_set<K_, H_>> { static constexpr const char* kind = "std::unordered_set"; };

using ZoneScope = tracy::ScopedZone;

void fusion_start(frame_mark mark);
void fusion_end(frame_mark mark);

void trace_container_size(const char*, std::size_t sz, const char* kind);

template <typename Collection_>
void trace_container_size(const char* name, const Collection_& c) {
  trace_container_size(name, c.size(), ContainerKind<Collection_>::kind);
}

#else

// Those are dummy implementations that are used when telemetry is disabled.
// See telemetry/core.cpp for real thing.
inline static void fusion_start(frame_mark) {}
inline static void fusion_end(frame_mark) {}

struct ZoneScope {

  ZoneScope(int, const char*, std::size_t, const char*, std::size_t, const char*, std::size_t, bool) {};

  ZoneScope(const ZoneScope&) = delete;
  ZoneScope(const ZoneScope&&) = delete;

  ZoneScope& operator= (const ZoneScope&) = delete;
  ZoneScope& operator= (ZoneScope&&) = delete;
};

template <typename Collection_>
void trace_container_size(const char* name, const Collection_& c) {(void)name, (void)c; }

#endif // TRACY_ENABLE

} // namespace telemetry
} // namespace c10

