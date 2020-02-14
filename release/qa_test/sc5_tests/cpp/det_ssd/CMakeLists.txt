# find opencv
find_package(OpenCV 3 REQUIRED)
# include
include_directories(${BMSDK_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/include)
link_directories(${OpenCV_LIBRARY_DIRS})
# bm lib path
link_libraries(${BMNNSDK2_LIBRARIES} ${OpenCV_LIBRARIES} ${SAIL_LIBRARY})

# build executable
add_executable(det_ssd_0 det_ssd_0.cpp processor.cpp cvdecoder.cpp)
add_executable(det_ssd_1 det_ssd_1.cpp processor.cpp)
add_executable(det_ssd_2 det_ssd_2.cpp processor.cpp)
