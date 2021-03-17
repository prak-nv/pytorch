
add_executable(nvfuser_bench
  batch_norm.cpp
  gelu_backward.cpp
  layer_norm.cpp
  lstm_cell.cpp
  reduction.cpp
  softmax.cpp
  main.cpp)

#if(NOT DISABLE_TRACY_HEAP_DEBUG)
#  target_sources(nvfuser_bench PRIVATE ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/telemetry/heap_debug.cpp)
# Intercept memory allocation functions by means of linker magic.
# -Wl,--wrap=name option redirects all calls to __wrap_name equivalent at static linking level.
# Conviniently it also introduces __real_ symbol pointing to old target.
# This option is supported by all 3 major likers: bfd, gold and lld.
#  target_link_options(nvfuser_bench PRIVATE "-Wl,--wrap=aligned_alloc")
#  target_link_options(nvfuser_bench PRIVATE "-Wl,--wrap=free")
#  target_link_options(nvfuser_bench PRIVATE "-Wl,--wrap=memalign")
#  target_link_options(nvfuser_bench PRIVATE "-Wl,--wrap=malloc")
#  target_link_options(nvfuser_bench PRIVATE "-Wl,--wrap=posix_memalign")
#else()
#  target_compile_definitions(torch_cuda PRIVATE "-DNVFUSER_TRACY_TELEMETRY_NO_HEAP_DEBUG")
#endif()

target_link_libraries(nvfuser_bench PRIVATE torch_library benchmark)
