project(sophon-inference)
cmake_minimum_required(VERSION 3.0)

# Set compile options
option(USE_PCIE              "ON for PCIE OFF for SOC"          ON)
option(USE_FFMPEG            "ON for USE BM-FFMPEG"             ON)
option(USE_LOCAL             "develop locally"                  ON)
option(USE_CENTOS            "compile ubuntu lib"               OFF)
option(USE_BMNNSDK2          "complie with bmnnsdk2"            ON)
option(USE_ARM_PCIE          "complie arm_pcie version"         OFF)
option(WITH_TEST             "build unit tests"                 OFF)
option(WITH_DOC              "build html or pdf"                OFF)

# Assign environment variables
set(CMAKE_CXX_STANDARD     14)
set(CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS} -w -O2")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH    ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_MODULE_PATH      ${CMAKE_SOURCE_DIR}/cmake)
# set for centos
if(USE_CENTOS)
  set(CMAKE_CXX_FLAGS      "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

# Set nntoolchain installation path
if(DEFINED ENV{REL_TOP})
  # Set for sophon-inference dev
  set(NNT_PATH $ENV{REL_TOP})
  set(BM168X_PATH "$ENV{REL_TOP}/../release")
  if(USE_PCIE)
    set(BMDECODER_DIR "$ENV{REL_TOP}/../release")
  else()
    set(BMDECODER_DIR "$ENV{REL_TOP}/../soc_bm1684_asic")
  endif()
  # set for bmnnsdk2 compile
  set(BMNNSDK2_PATH "$ENV{REL_TOP}")
else()
  # Set for all-in-one release
  set(NNT_PATH "/workspace/nntoolchain")
  set(BM168X_PATH "/workspace/bm168x")
  set(SOC_BM1684 "/workspace/soc_bm1684_asic")
  if(USE_PCIE)
    set(BMDECODER_DIR "/workspace/bm168x")
  else()
    set(BMDECODER_DIR "/workspace/soc_bm1684_asic")
  endif()
endif()

# Set cmake config file path
if(USE_PCIE)
  if(USE_CMODEL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/BM1684_CMODEL)
    link_directories(${NNT_PATH}/lib/thirdparty/x86)
  elseif(USE_ARM_PCIE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
    set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/BM1684_ARM_PCIE)
    if(USE_BMNNSDK2)
      set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/BMNNSDK2_BM1684_ARM_PCIE)
    endif()
    add_definitions(-DUSE_BMCV)
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/BM1684_PCIE)
    if(USE_BMNNSDK2)
      set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/BMNNSDK2_BM1684_PCIE)
    endif()
    link_directories(${NNT_PATH}/lib/thirdparty/x86)
    # Set BMCV definition
    add_definitions(-DUSE_BMCV)
  endif()
  # Set PCIE definition
  add_definitions(-DUSE_PCIE)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
  set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/BM1684_SOC)
  link_directories(${SOC_BM1684}/ffmpeg/usr/local/lib
                   ${SOC_BM1684}/decode/lib)
  # Set BMCV definition
  add_definitions(-DUSE_BMCV)
endif()

# Find BMSDK
find_package(BMSDK REQUIRED)

# Set FFMPEG definition
if(USE_FFMPEG)
  add_definitions(-DUSE_FFMPEG)
  find_package(BMDecode REQUIRED)
  find_package(FFMPEG REQUIRED)
endif()

add_subdirectory(src)
if((NOT USE_CENTOS) AND (NOT USE_ARM_PCIE))
  add_subdirectory(samples)
endif()

# Add test
if(WITH_TEST)
  add_subdirectory(tests)
  enable_testing()
endif()

# use "make cpp" to build all pure cpp targets
# use "make pysail" to build lib for python3
if(USE_FFMPEG)
  if(NOT USE_PCIE)
    add_custom_target(cpp DEPENDS sail cls_resnet_0 cls_resnet_1 cls_resnet_2 cls_resnet_3 det_mtcnn det_ssd_0 det_ssd_1 det_ssd_2 det_ssd_3 det_ssd_4 det_yolov3_0 det_yolov3_1)
  else()
    add_custom_target(cpp DEPENDS sail cls_resnet_0 cls_resnet_1 cls_resnet_2 cls_resnet_3 det_mtcnn det_ssd_0 det_ssd_1 det_ssd_2 det_yolov3_0 det_yolov3_1)
  endif()
else()
  add_custom_target(cpp DEPENDS sail cls_resnet_0 cls_resnet_1 cls_resnet_2 cls_resnet_3 det_mtcnn det_ssd_0 det_yolov3_0)
endif()

# Add doc
if(WITH_DOC)
  add_subdirectory(docs)
endif()
