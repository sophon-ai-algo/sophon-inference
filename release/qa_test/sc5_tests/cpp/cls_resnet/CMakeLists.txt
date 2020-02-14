# find opencv
find_package(OpenCV 3 REQUIRED)
# include
include_directories(${BMSDK_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/include)
link_directories(${OpenCV_LIBRARY_DIRS})
# bm lib path
link_libraries(pthread ${BMNNSDK2_LIBRARIES} ${OpenCV_LIBRARIES} ${SAIL_LIBRARY})

#if(USE_FFMPEG)
#  include_directories(${FFMPEG_INCLUDE_DIRS})
#  link_libraries(${FFMPEG_LIBRARIES} ${BMDECODE_LIBRARIES})
#endif()

# build executable
add_executable(cls_resnet_0 processor.cpp cls_resnet_0.cpp)
add_executable(cls_resnet_1 processor.cpp cls_resnet_1.cpp)
add_executable(cls_resnet_2 processor.cpp cls_resnet_2.cpp)
add_executable(cls_resnet_3 processor.cpp cls_resnet_3.cpp)
