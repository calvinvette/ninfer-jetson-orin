if(NOT NINFER_TARGET STREQUAL "jetson_orin")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a4.cu"
  )
endif()
