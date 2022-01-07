message(STATUS "bmnn_link_dirs=${bmnn_link_dirs}")

if (NOT WIN32)
	link_libraries(sail ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnn_link_libs} pthread)
	# build executable
	add_executable(det_yolov3_0 det_yolov3_0.cpp processor.cpp)
	if(USE_FFMPEG)
		add_executable(det_yolov3_1 det_yolov3_1.cpp processor.cpp frame_provider.cpp)
	endif()
else()
	link_libraries(libsail ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnn_link_libs})
	# build executable
	add_executable(det_yolov3_0 det_yolov3_0.cpp processor.cpp getopt_win.c)
	if(USE_FFMPEG)
		add_executable(det_yolov3_1 det_yolov3_1.cpp processor.cpp frame_provider.cpp getopt_win.c)
	endif()
endif()


