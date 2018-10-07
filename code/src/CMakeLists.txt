include_directories(
	basic
	basic-features
	basic-optimization
	util
	string-util
	threadpool
)


add_subdirectory(basic-features)
add_subdirectory(basic-optimization)
add_subdirectory(string-util)
add_subdirectory(threadpool)
add_subdirectory(util)

set(LIBS ${LIBS} basic-features basic-optimization util string-util thread
	)

message("libs: "  ${LIBS})
aux_source_directory(. DIR_SRCS)
add_executable(biaffine-parser ${DIR_SRCS})
target_link_libraries(biaffine-parser ${LIBS})
