message(STATUS "bmnn_link_dirs=${bmnn_link_dirs}")

if (NOT WIN32)
	link_libraries(sail ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnn_link_libs} pthread)
	# build executable
    add_executable(det_mtcnn det_mtcnn.cpp processor.cpp)
else()
	link_libraries(libsail ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnn_link_libs})
	# build executable
    add_executable(det_mtcnn det_mtcnn.cpp processor.cpp getopt_win.c)
endif()

