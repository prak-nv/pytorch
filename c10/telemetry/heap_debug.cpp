#ifndef C10_TRACY_TELEMETRY_NO_HEAP_DEBUG

/*
 * Unless cmake option 'DISABLE_TRACY_HEAP_DEBUG' is enabled we intercept memory
 * allocation functions for heap profiling.
 *
 * This is currently only supported with gnu based toolchain and ELF executable
 * as it is achieved by linkers --wrap option.
 */

#include <c10/util/Exception.h>

#include <tracy/client/TracyProfiler.hpp>

#include <atomic>
#include <cassert>
#include <cstdlib>


// Dummy function weak declarations to be overidden by linker
//
// Those real be pointing to original libc allocation functions
// (aligned_alloc, free, memalign, malloc and posix_memalign) instead

[[gnu::weak]] extern "C" void* __real_aligned_alloc(std::size_t, std::size_t);
[[gnu::weak]] extern "C" void  __real_free(void*);
[[gnu::weak]] extern "C" void* __real_malloc(std::size_t);
[[gnu::weak]] extern "C" void* __real_memalign(std::size_t, std::size_t);
[[gnu::weak]] extern "C" void* __real_posix_memalign(std::size_t, std::size_t);

namespace {

bool has_intercepted_alloc_fns() noexcept {

   //return &__real_malloc != nullptr;
   return true;
}

// NB: DTLS initilization of thread_local constructor uses malloc
// (via operator new in our particular case).
// This ensures we won't touch TLS value before it's constructed.
// Approach taken is simply guarding it with atomic flag that is set during constructor.
//
// Please note this class is one way singleton, due to the use static storage in its constructor.
template <bool DefaultVal>
class GuardedTLSFlag {
  static std::atomic<bool> tls_constructed;

public:
  GuardedTLSFlag() noexcept {
    tls_constructed.store(true, std::memory_order_release);
  }

  bool& value() noexcept {
    thread_local bool tls_value = DefaultVal;
    return tls_value;
  }

  bool value() const noexcept {
    return const_cast<GuardedTLSFlag<DefaultVal>*>(this)->value();
  }

  static bool is_constructed() noexcept {
    return tls_constructed.load(std::memory_order_acquire);
  }

  explicit operator bool() const noexcept {
    if (!tls_constructed.load(std::memory_order_acquire))
      return DefaultVal;
    return value();
  }

  GuardedTLSFlag& operator=(bool val) noexcept {
    assert(is_constructed());
    value() = val;
    return *this;
  }
};

template <bool Default>
std::atomic<bool> GuardedTLSFlag<Default>::tls_constructed{false};

GuardedTLSFlag<false> in_alloc{};

// Tracy hooks use malloc and friends.
// This helper that ensures we don't call tracy client allocation hooks recursively.
template <typename HookFn>
void call_alloc_hook_nonrec(HookFn fn) {
   if (in_alloc.is_constructed() && !in_alloc) {
     in_alloc = true;
     // NB: Avoid using tracy before it is really initialized.
     // FIXME: that should have better solution, without it we're getting crashes
     // while malloc is call during tracy::Profiler ctor.
     if (tracy::ProfilerAvailable())
       fn();
     in_alloc = false;
   }
}

void profiler_mem_alloc(void* ptr, std::size_t size) {
  if (ptr) tracy::Profiler::MemAlloc(ptr, size, /*secure*/true);
}

void profiler_mem_free(void* ptr) {
  if (ptr) tracy::Profiler::MemFree(ptr, /*secure*/true);
}

// Declare custom assembly labels intercept functions.
// Those function names shouldn't be mangled according to C++ mangling rules, since they are used as
// the alias targets of wrapped by functions linker like __wrap_malloc, __wrap_free etc.
//
#define GNU_ASM_LABEL(name) asm(#name)

void* intercepted_aligned_alloc(std::size_t, std::size_t) GNU_ASM_LABEL(intercepted_aligned_alloc);
void intercepted_free(void*) GNU_ASM_LABEL(intercepted_free);
void* intercepted_memalign(std::size_t, std::size_t) GNU_ASM_LABEL(intercepted_memalign);
void* intercepted_malloc(std::size_t) GNU_ASM_LABEL(intercepted_malloc);
void* intercepted_posix_memalign(std::size_t, std::size_t) GNU_ASM_LABEL(intercepted_posix_memalign);

// Those functions are used as liker alias targets and compiler diagnostics is incorrect about them being unused.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

void* intercepted_aligned_alloc(std::size_t alignment, std::size_t size) {
  assert(has_intercepted_alloc_fns());
  void* mem = __real_aligned_alloc(alignment, size);
  call_alloc_hook_nonrec([=]() {
    profiler_mem_alloc(mem, size);
  });

  return mem;
}

void intercepted_free(void* ptr) {
  assert(has_intercepted_alloc_fns());
  call_alloc_hook_nonrec([=] () {
    tracy::Profiler::MemFree(ptr, /*secure*/true);
  });
  __real_free(ptr);
}

void* intercepted_memalign(std::size_t alignment, std::size_t size) {
  assert(has_intercepted_alloc_fns());
  void* mem = __real_memalign(alignment, size);
  call_alloc_hook_nonrec([=]() {
    tracy::Profiler::MemAlloc(mem, size, /*secure*/true);
  });
  return mem;
}

void* intercepted_malloc(std::size_t size) {
  assert(has_intercepted_alloc_fns());
  void* mem = __real_malloc(size);
  call_alloc_hook_nonrec([=]() {
    tracy::Profiler::MemAlloc(mem, size, /*secure*/true);
  });
  return mem;
}


#pragma GCC diagnostic pop

}

#if 0
void* operator new(std::size_t count) noexcept(false) {
  auto* ptr = __real_malloc(count);
  call_alloc_hook_nonrec([=]() {
    tracy::Profiler::MemAlloc(ptr, count, /*secure*/true);
  });
  return ptr;
}

void operator delete(void* ptr) noexcept(true) {
  call_alloc_hook_nonrec([=]() {
    tracy::Profiler::MemFree(ptr, /*secure*/true);
  });
  __real_free(ptr);
}
#endif

// Declarations below redirectes function to intercepted_ counterparts.
// __wrap_ and __real {aligned_alloc, free, memalign, malloc, posix_memalgin} function aliases
// Those come from linker when using -Wl,--wrap option.
extern "C" {

[[gnu::alias("intercepted_aligned_alloc")]] void* __wrap_aligned_alloc(std::size_t, std::size_t);
[[gnu::alias("intercepted_free")]] void __wrap_free(void*);
[[gnu::alias("intercepted_malloc")]] void* __wrap_malloc(std::size_t);
[[gnu::alias("intercepted_memalign")]] void* __wrap_memalign(std::size_t, std::size_t);
/*[[gnu::alias("intercepted_posix_memalign")]]*/ void* __wrap_posix_memalign(std::size_t, std::size_t);

}

#if 1
void* __wrap_posix_memalign(std::size_t alignment, std::size_t size) {
  assert(&__real_posix_memalign != nullptr);
  void* mem = __real_posix_memalign(alignment, size);
  call_alloc_hook_nonrec([=]() {
    tracy::Profiler::MemAlloc(mem, size, /*secure*/true);
  });
  return mem;
}
#endif

#endif
