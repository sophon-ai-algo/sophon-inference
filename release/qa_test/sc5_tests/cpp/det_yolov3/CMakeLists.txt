# find opencv
find_package(OpenCV 3 REQUIRED)

# include
include_directories(${BMSDK_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/include)
link_directories(${OpenCV_LIBRARY_DIRS})
# bm lib path
link_libraries(pthread ${BMNNSDK2_LIBRARIES} ${OpenCV_LIBRARIES} ${SAIL_LIBRARY})

# build executable
add_executable(det_yolov3_0 det_yolov3_0.cpp processor.cpp)
add_executable(det_yolov3_1 det_yolov3_1.cpp processor.cpp frame_provider.cpp)
