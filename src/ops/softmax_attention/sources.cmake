target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dense/plain/softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/causal_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/bf16/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/bf16/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/int8/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/int8/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/packed_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/context_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/sliding_window_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/plan.cpp"
)

if(NOT NINFER_TARGET STREQUAL "jetson_orin")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/tiled_launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/plan.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/plan.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/tiled_launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/plan.cpp")
endif()

target_sources(ninfer_ops_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dense/plain/softmax_attention.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/launch.cu"
)
if(NOT NINFER_TARGET STREQUAL "jetson_orin")
  target_sources(ninfer_ops_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/tiled_launch.cu")
endif()
if(NINFER_TARGET STREQUAL "jetson_orin")
  target_sources(ninfer_ops_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dense/packed/rtx5090/uniform.cu")
else()
  target_sources(ninfer_ops_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dense/packed/${NINFER_TARGET}/uniform.cu")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/qsa/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kpool_mla/sources.cmake")
