target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/tail_extract.cu"
  "${CMAKE_CURRENT_LIST_DIR}/prepare/prepare.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/prepare/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/indexer/indexer.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/indexer/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/sparse_gqa.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/launch.cu"
)

if(NINFER_TARGET STREQUAL "jetson_orin")
  set(NINFER_QSA_BACKEND rtx5090)
else()
  set(NINFER_QSA_BACKEND ${NINFER_TARGET})
endif()
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/indexer/${NINFER_QSA_BACKEND}/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/${NINFER_QSA_BACKEND}/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/${NINFER_QSA_BACKEND}/launch.cu"
)

if(NINFER_TARGET STREQUAL "dgx_spark")
  target_sources(ninfer_ops_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/indexer/dgx_spark/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/dgx_spark/grouped.cu")
else()
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/indexer/${NINFER_QSA_BACKEND}/launch.cu")
endif()
