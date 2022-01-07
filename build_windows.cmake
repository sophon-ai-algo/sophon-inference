# project configuration
set(RUNTIME_LIB "MT")
set(TARGET_TYPE "release")

# windows enable all options
option(USE_BMCV   "Use BMCV option" ON)
option(USE_FFMPEG "Use FFMPEG option" ON)
option(USE_OPENCV "Use OPENCV option" ON)

# windows is PCIE_MODE
add_definitions(-DPCIE_MODE=1)

if (USE_BMCV)
    add_definitions(-DUSE_BMCV=1)
endif()

if (USE_FFMPEG)
    add_definitions(-DUSE_FFMPEG=1)
endif()

if (USE_OPENCV)
    add_definitions(-DUSE_OPENCV=1)
endif()


if(TARGET_TYPE STREQUAL "release")
		if (RUNTIME_LIB STREQUAL "MD")
			set(CMAKE_CXX_FLAGS_RELEASE "/MD")
			set(CMAKE_C_FLAGS_RELEASE "/MD")
		else()
			set(CMAKE_CXX_FLAGS_RELEASE "/MT")
			set(CMAKE_C_FLAGS_RELEASE "/MT")
		endif()
else()
		if (RUNTIME_LIB STREQUAL "MD")
			set(CMAKE_CXX_FLAGS_DEBUG "/MDd")
			set(CMAKE_C_FLAGS_DEBUG "/MDd")
		else()
			set(CMAKE_CXX_FLAGS_DEBUG "/MTd")
			set(CMAKE_C_FLAGS_DEBUG "/MTd")
        endif()
endif()


if(DEFINED WINSDK_PATH)
	set(BMNNSDK2_TOP ${WINSDK_PATH})
else()
	set(BMNNSDK2_TOP ..\\bm168x\\out\\sc5_windows_release_MT)
endif()

message(STATUS "BMNNSDK2_TOP: " ${BMNNSDK2_TOP})

set(bmnnsdk_inc_dirs ${BMNNSDK2_TOP}/include
            ${BMNNSDK2_TOP}/include/bmruntime
            ${BMNNSDK2_TOP}/include/bmlib
            ${BMNNSDK2_TOP}/include/third_party
            ${BMNNSDK2_TOP}/include/opencv
            ${BMNNSDK2_TOP}/include/ffmpeg
            CACHE INTERNAL "")
set(ffmpeg_link_libs avdevice avfilter avformat avcodec avutil swresample swscale)
set(opencv_link_libs libopencv_core410 libopencv_imgproc410 libopencv_imgcodecs410 libopencv_videoio410)
if (${RUNTIME_LIB} STREQUAL "MT")
        set(bmnn_link_libs libbmrt-static libbmlib-static libbmcv-static
            libbmvideo-static libbmjpuapi-static libbmjpulite-static libvpp-static libbmion-static
            CACHE INTERNAL "")
else()
        set(bmnn_link_libs libbmrt libbmlib libbmcv
            libbmvideo libbmjpuapi libbmjpulite libvpp libbmion
            CACHE INTERNAL "")
endif()
			
if (USE_FFMPEG)
			set(bmnn_link_libs ${ffmpeg_link_libs} ${bmnn_link_libs} CACHE INTERNAL "")
endif()

if (USE_OPENCV)
			set(bmnn_link_libs ${opencv_link_libs} ${bmnn_link_libs} CACHE INTERNAL "")
endif()

set(bmnn_link_dirs ${BMNNSDK2_TOP}/libs
            ${BMNNSDK2_TOP}/libs/decode
            ${BMNNSDK2_TOP}/libs/ffmpeg
            ${BMNNSDK2_TOP}/libs/jpeg
            ${BMNNSDK2_TOP}/libs/opencv
            ${BMNNSDK2_TOP}/libs/bmnn
            ${BMNNSDK2_TOP}/libs/vpp
            ${BMNNSDK2_TOP}/libs/third_party/boost
            CACHE INTERNAL "")	

include_directories(${bmnnsdk_inc_dirs})
link_directories(${bmnn_link_dirs})

add_subdirectory(3rdparty/pybind11)
# set sail source file
set(SAIL_SRC src/cvwrapper.cpp src/engine.cpp src/graph.cpp src/tensor.cpp src/tools.cpp src/base64.cpp)
set(SAIL_TOP ${CMAKE_CURRENT_SOURCE_DIR})

# build cpp api lib
include_directories(${SAIL_TOP}/include ${SAIL_TOP}/3rdparty)
add_library(libsail ${SAIL_SRC})
add_library(libsail_dll SHARED ${SAIL_SRC})
target_link_libraries(libsail_dll ${bmnn_link_libs})

if(CUSTOM_PY_LIBDIR)
  link_directories(${CUSTOM_PY_LIBDIR})
  message(STATUS "CUSTOM_PY_LIBDIR = ${CUSTOM_PY_LIBDIR}")
else()
  link_directories(c:\\Pythons\\Python38\\libs)
endif()

# build sail.xxx.pyd
add_library(pysail MODULE src/bind.cpp ${SAIL_SRC})
target_include_directories(pysail PUBLIC ${PYBIND11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
target_compile_definitions(pysail PRIVATE -DPYTHON=1)		  
target_link_libraries(pysail PRIVATE ${bmnn_link_libs} pybind11::module pybind11::embed)
set_target_properties(pysail PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                        SUFFIX "${PYTHON_MODULE_EXTENSION}" OUTPUT_NAME "sail")

# build example
add_subdirectory(samples)		