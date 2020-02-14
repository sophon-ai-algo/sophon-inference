# find opencv
find_package(OpenCV REQUIRED)

# include
include_directories(${BMSDK_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/include)
link_directories(${OpenCV_LIBRARY_DIRS}
                 ${BMSDK_LIBRARY_DIRS})
# bm lib path
link_libraries(${OpenCV_LIBRARIES} ${BMSDK_LIBRARIES} sail pthread)

if(USE_FFMPEG)
  include_directories(${FFMPEG_INCLUDE_DIRS})
  link_libraries(${FFMPEG_LIBRARIES} ${BMDECODE_LIBRARIES})
endif()

# build executable
add_executable(det_yolov3_0 det_yolov3_0.cpp processor.cpp)
if(USE_FFMPEG)
  add_executable(det_yolov3_1 det_yolov3_1.cpp processor.cpp frame_provider.cpp)
endif()
