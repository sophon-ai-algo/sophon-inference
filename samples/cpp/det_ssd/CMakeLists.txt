set(CMAKE_CXX_FLAGS "-Wall")

# find opencv
find_package(OpenCV REQUIRED)
# include
include_directories(${BMSDK_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/include)
link_directories(${OpenCV_LIBRARY_DIRS}
                 ${BMSDK_LIBRARY_DIRS})
# bm lib path
link_libraries(${OpenCV_LIBRARIES} ${BMSDK_LIBRARIES} sail)

if(USE_FFMPEG)
  include_directories(${FFMPEG_INCLUDE_DIRS})
  link_libraries(${FFMPEG_LIBRARIES} ${BMDECODE_LIBRARIES})
endif()

# build executable
add_executable(det_ssd_0 det_ssd_0.cpp processor.cpp cvdecoder.cpp)

if(USE_FFMPEG)
  if(NOT USE_PCIE)
    add_executable(det_ssd_3 det_ssd_3.cpp processor.cpp cvdecoder.cpp)
    add_executable(det_ssd_4 det_ssd_4.cpp processor.cpp cvdecoder.cpp)
  endif()
  add_executable(det_ssd_1 det_ssd_1.cpp processor.cpp)
  add_executable(det_ssd_2 det_ssd_2.cpp processor.cpp)
endif()
