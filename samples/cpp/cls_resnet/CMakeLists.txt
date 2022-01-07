
message(STATUS "bmnn_link_dirs=${bmnn_link_dirs}")
if (NOT WIN32)
	link_libraries(sail ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnn_link_libs} pthread)
	add_executable(cls_resnet_0 processor.cpp cls_resnet_0.cpp)
    add_executable(cls_resnet_1 processor.cpp cls_resnet_1.cpp)
    add_executable(cls_resnet_2 processor.cpp cls_resnet_2.cpp)   
    add_executable(cls_resnet_3 processor.cpp cls_resnet_3.cpp)
else()
	link_libraries(libsail ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnn_link_libs})
	
    add_executable(cls_resnet_0 processor.cpp cls_resnet_0.cpp getopt_win.c)
	add_executable(cls_resnet_1 processor.cpp cls_resnet_1.cpp getopt_win.c)
    add_executable(cls_resnet_2 processor.cpp cls_resnet_2.cpp getopt_win.c)
    add_executable(cls_resnet_3 processor.cpp cls_resnet_3.cpp getopt_win.c)
endif()
# build executable

