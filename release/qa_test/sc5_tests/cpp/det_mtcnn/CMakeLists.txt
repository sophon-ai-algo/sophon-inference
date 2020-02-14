# find opencv
find_package(OpenCV REQUIRED)
# include
include_directories(${BMSDK_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/include)
link_directories(${OpenCV_LIBRARY_DIRS})
# bm lib path
link_libraries(${BMSDK_LIBRARIES} ${OpenCV_LIBRARIES} ${SAIL_LIBRARY})

if(USE_FFMPEG)
  include_directories(${FFMPEG_INCLUDE_DIRS})
  link_libraries(${FFMPEG_LIBRARIES} ${BMDECODE_LIBRARIES})
endif()

#add source files
aux_source_directory(./ SRC)
# build executable
add_executable(det_mtcnn ${SRC})
