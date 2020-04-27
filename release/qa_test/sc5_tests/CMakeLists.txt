project(sc5_tests)
cmake_minimum_required(VERSION 3.0)

# Set compile options
option(BUILD_ON_ARM "ON for arm and OFF for x86" OFF)

# print log
add_definitions(-DPRINT_DETAILS)

# Assign bmnnsdk2 installation path
if(DEFINED ENV{REL_TOP})
  set(BMNNSDK2_PATH $ENV{REL_TOP})
else()
  set(BMNNSDK2_PATH bmnnsdk2_default_path)
endif()

# global set
set(CMAKE_CXX_STANDARD  11)
set(CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS} -g -w")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH    ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_MODULE_PATH      ${CMAKE_SOURCE_DIR}/cmake)

# include
add_definitions(-DUSE_PCIE)
add_definitions(-DUSE_FFMPEG)
add_definitions(-DUSE_BMCV)
if(BUILD_ON_ARM)
  set(CMAKE_MODULE_PATH      ${CMAKE_SOURCE_DIR}/cmake/arm)
else()
  set(CMAKE_MODULE_PATH      ${CMAKE_SOURCE_DIR}/cmake/x86)
endif()

# find opencv3
find_package(OpenCV 3 REQUIRED)
# find BMSDK
find_package(BMNNSDK2 REQUIRED)

include_directories(${BMNNSDK2_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/include
                    ${OpenCV_INCLUDE_DIRS})

# lib
link_libraries(${OpenCV_LIBRARIES} ${BMNNSDK2_LIBRARIES} ${SAIL_LIBRARY})

if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0.0)
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
  # libstdc++
  if(BUILD_ON_ARM)
    set(STDCPP_LIB_DIR ${BMNNSDK2_PATH}/lib/sys/aarch64)
  else()
    set(STDCPP_LIB_DIR ${BMNNSDK2_PATH}/lib/sys/x86)
  endif()
  file(GLOB STDCPP_LIBS "${STDCPP_LIB_DIR}/lib*.so*")
  link_libraries(${STDCPP_LIBS})
endif()

# cpp sail target
add_subdirectory(cpp/cls_resnet)
add_subdirectory(cpp/det_ssd)
add_subdirectory(cpp/det_yolov3)
add_subdirectory(cpp/det_mtcnn)
