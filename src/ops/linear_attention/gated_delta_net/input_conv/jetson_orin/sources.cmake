# SM87 uses the existing Ampere-compatible shape and launch definitions.
include("${CMAKE_CURRENT_LIST_DIR}/../rtx5090/sources.cmake")
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/unsupported.cpp")
