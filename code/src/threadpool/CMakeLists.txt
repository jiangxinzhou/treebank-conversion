aux_source_directory(. DIR_SRCS)
add_library(thread ${DIR_SRCS})
target_link_libraries(thread ${LIBS})
