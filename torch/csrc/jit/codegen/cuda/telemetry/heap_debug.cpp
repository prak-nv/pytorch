#if defined(TRACY_ENABLE) && !defined(FUSER_TRACY_TELEMETRY_NO_HEAP_DEBUG)

/*
 * Unless cmake option 'DISABLE_TRACY_HEAP_DEBUG' is enabled we override
 * global new/delete to obtain heap traffic data
 */

#include "tracy/Tracy.hpp"

#include <cstdlib>

#if 0

// FIXME: Those fail since torch does allocations before we get loaded and dynamic linker
// overrides those operators.
// This causes SEGV in glibc malloc (due to call of free on memory that was allocated previously with new)
//
void* operator new(std::size_t count) noexcept(false)
{
  auto* ptr = std::malloc(count);
  TracyAlloc(ptr, count);
  return ptr;
}

void operator delete(void* ptr) noexcept(true)
{
  TracyFree(ptr);
  std::free(ptr);
}

#elif 0
// FIXME: malloc debugging broken due to unpaired free which tracy doesn't like
// This is caused by dynamic linker its own memory allocations before we get load

// glibc malloc hooks
#include <malloc.h>

#include <dlfcn.h>
#include <sys/syscall.h>


struct InitTracyMallocHooks
{
  static decltype(__malloc_hook) old_malloc_hook_;
  static decltype(__free_hook) old_free_hook_;

  // FIXME: Those should be thread-safe

  InitTracyMallocHooks()
  {
    old_malloc_hook_ = __malloc_hook;
    old_free_hook_ = __free_hook;

    __malloc_hook = tracy_malloc;
    __free_hook = tracy_free;
  }

  static void* tracy_malloc(size_t sz, const void*)
  {
    __malloc_hook = old_malloc_hook_;
    __free_hook = old_free_hook_;
    auto *mem = ::malloc(sz);
    TracyAlloc(mem, sz);
    __free_hook = &tracy_free;
    __malloc_hook = &tracy_malloc;

    return mem;
  }

  static void tracy_free(void *ptr, const void* orig)
  {
    if (!ptr) return;

    __malloc_hook = old_malloc_hook_;
    __free_hook = old_free_hook_;
    Dl_info nfo;
    //::dladdr(orig, &nfo);
    // printf("free %p %p %s %s\n", ptr, orig, nfo.dli_fname, nfo.dli_sname);
    TracyFree(ptr);
    ::free(ptr);
    __free_hook = &tracy_free;
    __malloc_hook = &tracy_malloc;
  }

};

__attribute__((constructor)) static void early_init_malloc_hooks()
{
   static InitTracyMallocHooks init_tracy_hooks;
}

decltype(__malloc_hook) InitTracyMallocHooks::old_malloc_hook_ = nullptr;
decltype(__free_hook) InitTracyMallocHooks::old_free_hook_ = nullptr;

#endif

#endif
