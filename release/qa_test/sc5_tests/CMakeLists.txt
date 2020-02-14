project(sc5_tests)
cmake_minimum_required(VERSION 3.0)

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

if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0.0)
  message("-- use _GLIBCXX_USE_CXX11_ABI=0")
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
  # libstdc++
  set(STDCPP_LIB_DIR ${BMNNSDK2_PATH}/lib/sys)
  file(GLOB STDCPP_LIBS "${STDCPP_LIB_DIR}/lib*.so*")
  foreach (lib_name ${STDCPP_LIBS})
    set(STDCPP_LINK_LIBS ${STDCPP_LIBS} ${lib_name})
  endforeach ()
endif ()

# include
add_definitions(-DUSE_FFMPEG)
add_definitions(-DUSE_BMCV)
add_definitions(-DUSE_PCIE)
#if(USE_PCIE)
set(CMAKE_MODULE_PATH      ${CMAKE_SOURCE_DIR}/cmake)
link_directories(${BMNNSDK2_PATH}/lib/ffmpeg)
link_directories(${BMNNSDK2_PATH}/lib/decode)


find_package(OpenCV 3 REQUIRED)
find_package(BMNNSDK2 REQUIRED)

include_directories(${BMNNSDK2_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/include
                    ${OpenCV_INCLUDE_DIRS})


# lib
link_libraries(${OpenCV_LIBRARIES} ${BMNNSDK2_LIBRARIES} ${SAIL_LIBRARY})

if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0.0)
  link_libraries(${STDCPP_LINK_LIBS})
endif ()

# cpp sail target
add_subdirectory(cpp/cls_resnet)
add_subdirectory(cpp/det_ssd)
add_subdirectory(cpp/det_yolov3)
add_subdirectory(cpp/det_mtcnn)
