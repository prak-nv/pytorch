
add_executable(nvfuser_bench
  batch_norm.cpp
  gelu_backward.cpp
  layer_norm.cpp
  lstm_cell.cpp
  reduction.cpp
  softmax.cpp
  bert.cpp
  main.cpp)

target_link_libraries(nvfuser_bench PRIVATE torch_library benchmark)
