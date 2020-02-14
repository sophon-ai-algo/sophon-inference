# find opencv
find_package(OpenCV REQUIRED)
# include
include_directories(${BMSDK_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/include)
link_directories(${OpenCV_LIBRARY_DIRS})
# bm lib path
link_libraries(sail ${BMSDK_LIBRARIES} ${OpenCV_LIBRARIES})

if(USE_FFMPEG)
  include_directories(${FFMPEG_INCLUDE_DIRS})
  link_libraries(${FFMPEG_LIBRARIES} ${BMDECODE_LIBRARIES})
endif()

# build executable
add_executable(det_mtcnn det_mtcnn.cpp processor.cpp)
