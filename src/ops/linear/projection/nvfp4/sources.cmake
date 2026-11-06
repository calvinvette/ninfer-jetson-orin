target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n34816_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k17408.cu"
)

if(NINFER_TARGET STREQUAL "jetson_orin")
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a4_unsupported.cpp")
endif()

if(NOT NINFER_TARGET STREQUAL "jetson_orin")
  target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a4_tma.cu")
endif()

if(NINFER_TARGET STREQUAL "dgx_spark")
  include("${CMAKE_CURRENT_LIST_DIR}/dgx_spark/sources.cmake")
endif()
