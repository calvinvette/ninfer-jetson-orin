include("${CMAKE_CURRENT_LIST_DIR}/control/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/input_conv/sources.cmake")
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/recurrence/chunked/prepare.cu"
  "${CMAKE_CURRENT_LIST_DIR}/recurrence/chunked/recurrence.cu"
  "${CMAKE_CURRENT_LIST_DIR}/recurrence/gated_delta_net.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/recurrence/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/replay/fold.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/replay/record.cpp"
)
if(NINFER_TARGET STREQUAL "jetson_orin")
  set(NINFER_GDN_BACKEND rtx5090)
else()
  set(NINFER_GDN_BACKEND ${NINFER_TARGET})
endif()
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/recurrence/${NINFER_GDN_BACKEND}/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/recurrence/${NINFER_GDN_BACKEND}/recurrent.cu"
  "${CMAKE_CURRENT_LIST_DIR}/replay/${NINFER_GDN_BACKEND}/fold.cu"
  "${CMAKE_CURRENT_LIST_DIR}/replay/${NINFER_GDN_BACKEND}/record.cu"
)
