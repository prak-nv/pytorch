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
#include <chrono>
#include <limits>
#include <mutex>
#include <thread>
#include <type_traits>

#include <unistd.h>
#include <sys/syscall.h>

#if defined(__x86_64__) // SSE2 extension is included in 64-bit ISA as per amd64/ia64 specifications
#include <immintrin.h>
#endif

// Dummy function weak declarations to be overidden by linker
//
// Those real be pointing to original libc allocation functions
// (aligned_alloc, free, memalign, malloc and posix_memalign) instead
//

[[gnu::weak]] extern "C" void* __real_aligned_alloc(std::size_t, std::size_t);
[[gnu::weak]] extern "C" void  __real_free(void*);
[[gnu::weak]] extern "C" void* __real_malloc(std::size_t);
[[gnu::weak]] extern "C" void* __real_memalign(std::size_t, std::size_t);
[[gnu::weak]] extern "C" void* __real_posix_memalign(std::size_t, std::size_t);
[[gnu::weak]] extern "C" void* __real_realloc(void*, std::size_t);

namespace {

// gettid - returns current thread thread id
// This function was added in glibc 2.30 and is unavailable in erlier versions
// for simplicity implemented using syscall
pid_t cached_gettid() {
  // NB: thread local memoization here considerably reduces number of sycalls
  thread_local pid_t tid = syscall(SYS_gettid);
  return tid;
}

// spin_wait_backoff - Performs n iterations of active waiting in spinlock loop
void spin_wait_backoff(std::size_t backoff) {
  for (std::size_t i=0; i<backoff; ++i) {
#ifdef __x86_64__
    _mm_pause();
#else
    using namespace std::literals::chrono_literals;
    // We really mean c++17 yield() here
    std::this_thread::sleep_for(0ms);
#endif
  }
}

// NB: Value 64 should cover 99.99% cases for real world cacheline size (in general 32 or 64)
// We don't need correct value, as intent here is only avoid cache fighting in member access.
// Given that, its multiple will be fine.
constexpr std::size_t LIKELY_CACHLINE_SIZE = 64;

// AtomicSpinlock - Very simple spinlock using c++ atomics
// It uses ticket locking algorithm based losely on Algorithm 2 from:
// "Algorithms for scalable synchronization on shared-memory multiprocessors"
// John M. Mellor-Crummey , Michael L. Scott
//
// https://doi.org/10.1145/103727.103729
//
// Sligth deviation from proposed algorithm is that we use exponential back-off, rather than
// propotional
class AtomicSpinlock {
public:
  AtomicSpinlock() noexcept = default;

  void lock() noexcept {
    auto ticket = next_ticket_.fetch_add(1, std::memory_order_relaxed);
    std::size_t backoff = 1;
    while (ticket != now_serving_.load(std::memory_order_acquire)) {
      // Wait exponential number of iterations for thread turn to enter critical section.
      spin_wait_backoff(backoff);
      backoff *= 2;
    }
  }

  void unlock() noexcept {
    size_t next_served = now_serving_.load(std::memory_order_acquire) + 1;
    // Since we are in critical section, no update to now_serving_ may occur and atomic store is
    // sufficient.
    now_serving_.store(next_served, std::memory_order_release);
  }

private:
  // Ticket numbers - as per general implementation advice members are aligned to cacheline
  // to avoid potential cache fighting effect between different threads.
  alignas(LIKELY_CACHLINE_SIZE) std::atomic_size_t now_serving_ = {0};
  alignas(LIKELY_CACHLINE_SIZE) std::atomic_size_t next_ticket_ = {0};
};

// recursive_lock_adaptor - adds recursive property to binary lock
template <typename LockType_>
class RecursiveLockAdaptor : private LockType_ {
  // Locking order:
  //  underlaying lock -> atomic owner_thread_ == our_tid value
  static constexpr pid_t NOT_OWNED = pid_t{};

  using base_lock_t = LockType_;

public:
  RecursiveLockAdaptor() noexcept = default;

  void lock() {
    thread_local auto our_tid = cached_gettid();

    // Check whether we are entring critical section recursivly.
    if (owner_thread_.load(std::memory_order_acquire) != our_tid) {

       // Try enter critical section of base_lock
       // Please note however that acquiring lock here is not required for correctness, since
       // setting owner_thread in CAS loop below is sufficient to achive mutual exclusion.
       // Entring critical section of base lock, only enforces its additional properties,
       // like fairness, backoff strategy etc.
       base_lock_t::lock();

       // Loop until previos thread revoked ownership of this lock
       pid_t tid;
       do {
         tid = NOT_OWNED;
       } while(owner_thread_.compare_exchange_weak(tid, our_tid,
             std::memory_order_acquire, std::memory_order_relaxed));
    }

    // We own the lock for both recursive and nonrecursive case
    // Increment count of recursive calls to this lock.
    ++rec_count_;
  }

  void unlock() {
     // Do we really own this lock?
     assert(owner_thread_.load() == cached_gettid());
     // Are lock/unlock calls ?
     assert(rec_count_ > 0);

     --rec_count_;
     // Last recursive call
     if (rec_count_ == 0) {
       base_lock_t::unlock();
       // Give back final ownership over lock to potentially waiting or future thread.
       owner_thread_.store(NOT_OWNED, std::memory_order_release);
     }
  }

public:
  std::atomic<pid_t> owner_thread_ = {NOT_OWNED};
  std::size_t rec_count_ = 0;
};

using RecursiveSpinlock = RecursiveLockAdaptor<AtomicSpinlock>;

// c++11 style allocator using __real_malloc and __real_free directly, to avoid calling allocation hooks
template <typename T>
class real_malloc_allocator;

template <>
class real_malloc_allocator<void> {
public:
  using value_type = void;

  real_malloc_allocator() noexcept = default;

  template <typename T>
  constexpr real_malloc_allocator(const real_malloc_allocator<T>& other) noexcept {};

  void* allocate(std::size_t sz) {
    void* ptr = __real_malloc(sz);
    if (ptr == nullptr) throw std::bad_alloc();
    return ptr;
  }

  void deallocate(void* ptr, std::size_t sz) noexcept {
    __real_free(ptr);
  }
};

template <typename T>
class real_malloc_allocator : private real_malloc_allocator<void> {
  // NB: real_malloc_allocator<void> is base to limit template bloat
  // (ie. generation of same functions for each type).
  // allocate/deallocate implmentation diffrences are only in type c++ system
  using base_allocator = real_malloc_allocator<void>;

public:
  using value_type = T;

  // Affects c++17 dialect and beyond
  using is_always_equal = std::true_type;

  real_malloc_allocator() noexcept = default;

  template <typename U>
  constexpr real_malloc_allocator(const real_malloc_allocator<U>& other) noexcept {};

  T* allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_alloc();
    return static_cast<T*>(base_allocator::allocate(sizeof(T)*n));
  }

  void swap(real_malloc_allocator&) {
    // no-op swap, since this allocator is stateless
  }

  void deallocate(T* ptr, std::size_t n) {
    base_allocator::deallocate(ptr, n);
  }
};

// Allocators can free memory allocated by other instance:
template <typename T, typename U>
bool operator== (const real_malloc_allocator<T>&, const real_malloc_allocator<U>&) {
  return true;
}

template <typename T, typename U>
bool operator!= (const real_malloc_allocator<T>&, const real_malloc_allocator<U>&) {
  return false;
}

enum class MemEventKind {
  Aligned, /* allocation with specified alignment */
  Normal, /* (naturally aligned, like malloc, free) */
  Reallocation,
  Free,

  AlignedAlloc = Aligned,
  Delete = Free,
  Malloc = Normal,
  MemAlign = Aligned,
  New = Normal,
  PosixMemalign = Aligned,
  Realloc = Reallocation,
};

const char* allocation_name(MemEventKind kind) {
  switch (kind) {
  case MemEventKind::Aligned: return "aligned malloc";
  case MemEventKind::Free: return "malloc";
  case MemEventKind::Normal: return "malloc";
  case MemEventKind::Reallocation: return "malloc";
  }
  assert(false);
  return "";
}

constexpr bool is_allocation_event(MemEventKind kind) noexcept {
  return kind != MemEventKind::Free;
}

struct MemEvent {
  void* ptr_;
  std::size_t size_;
  MemEventKind kind_;
};

template <typename T>
using Vector = std::vector<T, real_malloc_allocator<T>>;

// Backlog of memory events that we can't log to early on during initialization
// Those are resubmitted once all facilities are available.
class MemEventBacklog {
  using lock_type = RecursiveSpinlock;

public:
  MemEventBacklog();

  bool has_pending_events() const noexcept;
  void report_event(void* ptr, std::size_t sz, MemEventKind k);
  void submit_pending_events();

private:
  RecursiveSpinlock lock_; // protects events_ from concurrent access
  Vector<MemEvent> events_;
};

MemEventBacklog g_event_backlog;

bool has_intercepted_alloc_fns() noexcept {
   return &__real_malloc != nullptr;
}

// NB: DTLS initilization of thread_local constructor uses malloc
// (via operator new in our particular case).
// This ensures we won't touch TLS value before it's constructed.
// Approach taken is simply guarding it with atomic flag that is set during constructor.
//
// Please note this class is one way singleton, due to the use static storage in its constructor.
template <bool Default>
class GuardedTLSFlag {
  static std::atomic<bool> tls_constructed;

  static bool constexpr default_value = Default;

public:
  GuardedTLSFlag() noexcept {
    tls_constructed.store(true, std::memory_order_release);
  }

  bool& value() noexcept {
    thread_local bool tls_value = default_value;
    return tls_value;
  }

  bool value() const noexcept {
    return const_cast<GuardedTLSFlag<default_value>*>(this)->value();
  }

  static bool is_constructed() noexcept {
    return tls_constructed.load(std::memory_order_acquire);
  }

  explicit operator bool() const noexcept {
    if (!tls_constructed.load(std::memory_order_acquire))
      return default_value;
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

// This helper that ensures we don't call tracy client allocation hooks recursively.
template <typename HookFn>
bool call_mem_hook_nonrec(HookFn fn) {
   if (!in_alloc.is_constructed() || in_alloc)
     return false;

   assert(tracy::ProfilerAvailable());
   in_alloc = true;
   g_event_backlog.submit_pending_events();
   fn();
   in_alloc = false;
   return true;
}

constexpr int CALLSTACK_DEPTH = 16; // tunable as needed

void report_mem_alloc(void* ptr, std::size_t size, MemEventKind kind) {
  if (ptr) {
    const char* name = allocation_name(kind);
    tracy::Profiler::MemAllocCallstackNamed(ptr, size, CALLSTACK_DEPTH, /*secure*/true, name);
  }
}

void report_mem_free(void* ptr, MemEventKind kind) {
  if (ptr) {
    const char* name = allocation_name(kind);
    tracy::Profiler::MemFreeCallstackNamed(ptr, CALLSTACK_DEPTH, /*secure*/true, name);
  }
}

void profiler_report_event(const MemEvent& ev) {
   if (is_allocation_event(ev.kind_)) {
     report_mem_alloc(ev.ptr_, ev.size_, ev.kind_);
   } else {
     report_mem_free(ev.ptr_, ev.kind_);
   }
}

bool report_to_backlock(void* ptr, MemEventKind kind) {
  assert(!is_allocation_event(kind) && "Allocations should have size");
  g_event_backlog.report_event(ptr, 0, kind);
  return true;
}

bool report_to_backlock(void* ptr, std::size_t size, MemEventKind kind) {
  g_event_backlog.report_event(ptr, size, kind);
  return true;
}
// Declare custom assembly labels intercept functions.
// Those function names shouldn't be mangled according to C++ mangling rules, since they are used as
// the alias targets of wrapped by functions linker like __wrap_malloc, __wrap_free etc.
//
#define GNU_ASM_LABEL(name) asm(#name)

// Allocation interceptors
void* intercepted_aligned_alloc(std::size_t, std::size_t) GNU_ASM_LABEL(intercepted_aligned_alloc);
void intercepted_free(void*) GNU_ASM_LABEL(intercepted_free);
void* intercepted_memalign(std::size_t, std::size_t) GNU_ASM_LABEL(intercepted_memalign);
void* intercepted_malloc(std::size_t) GNU_ASM_LABEL(intercepted_malloc);
void* intercepted_posix_memalign(std::size_t, std::size_t) GNU_ASM_LABEL(intercepted_posix_memalign);
void* intercepted_realloc(void*, std::size_t) GNU_ASM_LABEL(intercepted_realloc);

// Those functions are used as liker alias targets and compiler diagnostics is incorrect
// about them being unused.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

void* intercepted_aligned_alloc(std::size_t alignment, std::size_t size) {
  assert(has_intercepted_alloc_fns());
  void* mem = __real_aligned_alloc(alignment, size);

  call_mem_hook_nonrec([=]() {
    report_mem_alloc(mem, size, MemEventKind::AlignedAlloc);
  }) || report_to_backlock(mem, size, MemEventKind::AlignedAlloc);

  return mem;
}

void intercepted_free(void* ptr) {
  assert(has_intercepted_alloc_fns());

  call_mem_hook_nonrec([=] () {
    report_mem_free(ptr, MemEventKind::Free);
  }) || report_to_backlock(ptr, MemEventKind::Free);

  __real_free(ptr);
}

void* intercepted_memalign(std::size_t alignment, std::size_t size) {
  assert(has_intercepted_alloc_fns());
  void* mem = __real_memalign(alignment, size);

  call_mem_hook_nonrec([=]() {
    report_mem_alloc(mem, size, MemEventKind::MemAlign);
  }) || report_to_backlock(mem, size, MemEventKind::MemAlign);

  return mem;
}

void* intercepted_malloc(std::size_t size) {
  assert(has_intercepted_alloc_fns());
  void* mem = __real_malloc(size);

  call_mem_hook_nonrec([=]() {
    report_mem_alloc(mem, size, MemEventKind::Malloc);
  }) || report_to_backlock(mem, size, MemEventKind::Malloc);

  return mem;
}

void* intercepted_posix_memalign(std::size_t alignment, std::size_t size) {
  assert(has_intercepted_alloc_fns());
  void* mem = __real_posix_memalign(alignment, size);

  call_mem_hook_nonrec([=]() {
    report_mem_alloc(mem, size, MemEventKind::PosixMemalign);
  }) || report_to_backlock(mem, size, MemEventKind::PosixMemalign);

  return mem;
}

void* intercepted_realloc(void* ptr, std::size_t size) {
  assert(has_intercepted_alloc_fns());
  void* mem = __real_realloc(ptr, size);

  call_mem_hook_nonrec([=]() {
    report_mem_free(ptr, MemEventKind::Realloc);
  }) || report_to_backlock(ptr, MemEventKind::Free);

  call_mem_hook_nonrec([=]() {
    report_mem_alloc(mem, size, MemEventKind::Realloc);
  }) || report_to_backlock(mem, size, MemEventKind::Realloc);

  return mem;
}

#pragma GCC diagnostic pop

} // anonymous namespace

// User provided operators new/delete operators
void* operator new(std::size_t count) noexcept(false) {
  void* ptr = __real_malloc(count);
  bool ok = call_mem_hook_nonrec([=]() {
    report_mem_alloc(ptr, count, MemEventKind::New);
  }) || report_to_backlock(ptr, count, MemEventKind::New);

  return ptr;
}

void* operator new(std::size_t count, const std::nothrow_t&) noexcept(true) {
  void* ptr = __real_malloc(count);
  call_mem_hook_nonrec([=]() {
    report_mem_alloc(ptr, count, MemEventKind::New);
  }) || report_to_backlock(ptr, count, MemEventKind::New);

  return ptr;
}

void* operator new[](std::size_t count) noexcept(false) {
  void* ptr = __real_malloc(count);
  call_mem_hook_nonrec([=]() {
    report_mem_alloc(ptr, count, MemEventKind::New);
  }) || report_to_backlock(ptr, count, MemEventKind::New);
  return ptr;
}

void* operator new[](std::size_t count, const std::nothrow_t&) noexcept(true) {
  void* ptr = __real_malloc(count);
  call_mem_hook_nonrec([=]() {
    report_mem_alloc(ptr, count, MemEventKind::New);
  }) || report_to_backlock(ptr, MemEventKind::New);
  return ptr;
}

void operator delete(void* ptr) noexcept(true) {
  call_mem_hook_nonrec([=]() {
    report_mem_free(ptr, MemEventKind::Delete);
  }) || report_to_backlock(ptr, MemEventKind::Delete);
  __real_free(ptr);
}

void operator delete(void* ptr, const std::nothrow_t&) noexcept(true) {
  call_mem_hook_nonrec([=]() {
    report_mem_free(ptr, MemEventKind::Delete);
  }) || report_to_backlock(ptr, MemEventKind::Delete);
  __real_free(ptr);
}

void operator delete[](void* ptr) noexcept(true) {
  call_mem_hook_nonrec([=]() {
    report_mem_free(ptr, MemEventKind::Delete);
  }) || report_to_backlock(ptr, MemEventKind::Delete);
  __real_free(ptr);
}

void operator delete[](void* ptr, const std::nothrow_t&) noexcept(true) {
  call_mem_hook_nonrec([=]() {
    report_mem_free(ptr, MemEventKind::Delete);
  }) || report_to_backlock(ptr, MemEventKind::Delete);
  __real_free(ptr);
}

using LockGuard = std::lock_guard<RecursiveSpinlock>;

MemEventBacklog::MemEventBacklog() = default;

void MemEventBacklog::report_event(void* ptr, std::size_t size, MemEventKind kind) {
  LockGuard gaurd(lock_);
  {
    // TODO: tracy currently doesn't provide public facility to manually specify thread and timestamp
    // of allocation/deallocation call, once such is available, it's best place to do it
    events_.push_back({ptr, size, kind });
  }
}

bool MemEventBacklog::has_pending_events() const noexcept {
  LockGuard guard(const_cast<RecursiveSpinlock&>(lock_));
  {
    return !events_.empty();
  }
}

void MemEventBacklog::submit_pending_events() {
  LockGuard guard(lock_);
  {
    // Please note that additional events may be added to events_ vector as the result of
    // tracy client library allocations during profiler_report_event() calls
    for (std::size_t i=0; i < events_.size(); ++i) {
      profiler_report_event(events_[i]);
    }
    events_.clear();
  }
}

// Declarations below redirectes function to intercepted_ counterparts.
// __wrap_ and __real {aligned_alloc, free, memalign, malloc, posix_memalign, realloc} function aliases
// Those come from linker when using -Wl,--wrap option.
extern "C" {

[[gnu::alias("intercepted_aligned_alloc")]] void* __wrap_aligned_alloc(std::size_t, std::size_t);
[[gnu::alias("intercepted_free")]] void __wrap_free(void*);
[[gnu::alias("intercepted_malloc")]] void* __wrap_malloc(std::size_t);
[[gnu::alias("intercepted_memalign")]] void* __wrap_memalign(std::size_t, std::size_t);
[[gnu::alias("intercepted_posix_memalign")]] void* __wrap_posix_memalign(std::size_t, std::size_t);
[[gnu::alias("intercepted_realloc")]] void* __wrap_realloc(std::size_t, std::size_t);

}

#endif
