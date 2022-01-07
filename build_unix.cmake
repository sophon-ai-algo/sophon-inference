# Set compile options
option(BUILD_X86_PCIE          "ON for X86 PCIE"                 ON)
option(BUILD_ARM_PCIE          "ON for arm_pcie version"         OFF)
option(BUILD_SOC               "ON for soc version"              OFF)
option(BUILD_MIPS64            "ON for mips64 version"           OFF)
option(BUILD_CMODEL            "ON for cmodel version"           OFF)
option(BUILD_SW64              "ON for sw64 version"             OFF)
option(BUILD_LOONGARCH64       "ON for loongarch64"              OFF)

# ffmpeg is on if not cmodel
option(USE_OPENCV            "ON for USE OpenCV"                ON)
option(USE_FFMPEG            "ON for USE BM-FFMPEG"             ON)
option(USE_BMCV              "ON for USE BM-FFMPEG"             ON)

# base on local by default
option(USE_LOCAL             "develop locally"                  OFF)
option(USE_CENTOS            "ON for centos OFF for ubuntu "    OFF)
option(USE_BMNNSDK2          "complie with bmnnsdk2"            ON)
option(USE_ALLINONE          "on for All-in-one build"          OFF)
option(WITH_TEST             "build unit tests"                 ON)
option(WITH_DOC              "build html or pdf"                OFF)

if (NOT DEFINED JINKINS_TOP)
    set(JINKINS_TOP /workspace)
endif()


if ("${BUILD_TYPE}" STREQUAL "arm_pcie")
    set(BUILD_X86_PCIE OFF)
    set(BUILD_ARM_PCIE ON)
    set(BUILD_SOC OFF)
    set(BUILD_MIPS64 OFF)
    set(BUILD_CMODEL OFF)
    set(BUILD_SW64 OFF)
    set(BUILD_LOONGARCH64 OFF)
elseif ("${BUILD_TYPE}" STREQUAL "soc")
    set(BUILD_X86_PCIE OFF)
    set(BUILD_ARM_PCIE OFF)
    set(BUILD_SOC ON)
    set(BUILD_MIPS64 OFF)
    set(BUILD_CMODEL OFF)
    set(BUILD_SW64 OFF)
    set(BUILD_LOONGARCH64 OFF)
elseif ("${BUILD_TYPE}" STREQUAL "mips64")
    set(BUILD_X86_PCIE OFF)
    set(BUILD_ARM_PCIE OFF)
    set(BUILD_SOC OFF)
    set(BUILD_MIPS64 ON)
    set(BUILD_CMODEL OFF)
    set(BUILD_SW64 OFF)
    set(BUILD_LOONGARCH64 OFF)
elseif ("${BUILD_TYPE}" STREQUAL "cmodel")
    set(BUILD_X86_PCIE OFF)
    set(BUILD_ARM_PCIE OFF)
    set(BUILD_SOC OFF)
    set(BUILD_MIPS64 OFF)
    set(BUILD_CMODEL ON)
    set(USE_FFMPEG   OFF)
    set(USE_BMCV  OFF)
    set(BUILD_SW64 OFF)
    set(BUILD_LOONGARCH64 OFF)
elseif ("${BUILD_TYPE}" STREQUAL "pcie")
    set(BUILD_X86_PCIE ON)
    set(BUILD_ARM_PCIE OFF)
    set(BUILD_SOC OFF)
    set(BUILD_MIPS64 OFF)
    set(BUILD_CMODEL OFF)
    set(BUILD_SW64 OFF)
    set(BUILD_LOONGARCH64 OFF)
elseif ("${BUILD_TYPE}" STREQUAL "sw64")
    set(BUILD_X86_PCIE OFF)
    set(BUILD_ARM_PCIE OFF)
    set(BUILD_SOC OFF)
    set(BUILD_MIPS64 OFF)
    set(BUILD_CMODEL OFF)
    set(BUILD_SW64 ON)
    set(BUILD_LOONGARCH64 OFF)
elseif ("${BUILD_TYPE}" STREQUAL "loongarch64")
    set(BUILD_X86_PCIE OFF)
    set(BUILD_ARM_PCIE OFF)
    set(BUILD_SOC OFF)
    set(BUILD_MIPS64 OFF)
    set(BUILD_CMODEL OFF)
    set(BUILD_SW64 OFF)
    set(BUILD_LOONGARCH64 ON)
else()
    message("BUILD_TYPE=${BUILD_TYPE}, default is pcie")
endif()

if ("${SDK_TYPE}" STREQUAL "local")
    set(USE_LOCAL ON)
    set(USE_BMNNSDK2 OFF)
    set(USE_ALLINONE OFF)
elseif("${SDK_TYPE}" STREQUAL "bmnnsdk2")
    set(USE_LOCAL OFF)
    set(USE_BMNNSDK2 ON)
    set(USE_ALLINONE OFF)
elseif("${SDK_TYPE}" STREQUAL "allinone")
    set(USE_LOCAL OFF)
    set(USE_BMNNSDK2 OFF)
    set(USE_ALLINONE ON)
else()
    message("SDK_TYPE=${SDK_TYPE},default is local")
endif()

# Assign environment variables
set(CMAKE_CXX_STANDARD     14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
if (BUILD_MIPS64)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -mabi=64 -mxgot -O")
else()
    set(CMAKE_CXX_FLAGS_DEBUG    "-g")
    set(CMAKE_CXX_FLAGS_RELEASE  "-O3")
endif()
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH    ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_MODULE_PATH      ${CMAKE_SOURCE_DIR}/cmake)

# set for centos
if(USE_CENTOS)
  set(CMAKE_CXX_FLAGS      "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

if (USE_BMCV)
   add_definitions(-DUSE_BMCV=1)
endif()

if (USE_FFMPEG)
    add_definitions(-DUSE_FFMPEG=1)
endif()

if (USE_OPENCV)
    add_definitions(-DUSE_OPENCV=1)
endif()

if (BUILD_SOC)
    add_definitions(-DIS_SOC_MODE=1)
    set(IS_SOC_MODE ON)
endif()

# Set nntoolchain installation path
if (USE_LOCAL)
  # use develop local as build root
  if (DEFINED ENV{BMWS})
    set(BMWS $ENV{BMWS})
  else ()
    set(BMWS /home/yuan/bitmain/work)
  endif ()
  set(NNTC_PATH "${BMWS}/nntoolchain/net_compiler/out/install_bmruntime")

  if (BUILD_X86_PCIE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(MW_PATH "${BMWS}/middleware-soc/install/pcie_bm1684_asic")
    set(BM168X_PATH "${BMWS}/bm168x/out/bm1684_x86_pcie_device/release")
  elseif (BUILD_ARM_PCIE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
    set(MW_PATH "${BMWS}/middleware-soc/install/pcie_arm64_bm1684_asic")
    set(BM168X_PATH "${BMWS}/bm168x/out/bm1684_aarch64_pcie_device/release")
  elseif (BUILD_SOC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
    set(MW_PATH "${BMWS}/middleware-soc/install/soc_bm1684_asic")
    set(BM168X_PATH "${BMWS}/bm168x/out/bm1684_soc_device/release")
  elseif (BUILD_MIPS64)
    set(MW_PATH "${BMWS}/middleware-soc/install/mips_bm1684_asic")
  elseif (BUILD_SW64)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    set(MW_PATH "${BMWS}/middleware-soc/install/pcie_sw64_bm1684_asic")
    set(BM168X_PATH "${BMWS}/bm168x/out/bm1684_sw64_pcie_device/release")
  elseif (BUILD_LOONGARCH64)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    set(MW_PATH "${BMWS}/middleware-soc/install/pcie_loongarch64_bm1684_asic")
    set(BM168X_PATH "${BMWS}/bm168x/out/bm1684_loongarch64_pcie_device/release")
  elseif (BUILD_CMODEL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  else()
    message(ERROR "not supported")
  endif ()

  # shared configurations
  ## ffmpeg configuration
  set(ffmpeg_inc_dirs ${MW_PATH}/ffmpeg/usr/local/include)
  set(ffmpeg_link_dirs ${MW_PATH}/ffmpeg/usr/local/lib ${MW_PATH}/decode/lib)

  ## opencv configuration
  set(opencv_inc_dirs ${MW_PATH}/opencv/include/opencv4)
  set(opencv_link_dirs ${MW_PATH}/opencv/lib)

  ## bmlib bmrt
  set(bmnn_inc_dirs ${BM168X_PATH}/include ${NNTC_PATH}/include)
  set(bmnn_link_dirs ${BM168X_PATH}/libs ${NNTC_PATH}/lib)

elseif (USE_BMNNSDK2)
  # use bmnnsdk2 as build root
  if (DEFINED ENV{REL_TOP})
    set(BMNNSDK2_PATH $ENV{REL_TOP})
  else ()
    set(BMNNSDK2_PATH /home/yuan/bmnnsdk2/bmnnsdk2-latest)
  endif ()
    ## bmnnsdk2 common include
  set(bmnn_inc_dirs ${BMNNSDK2_PATH}/include
          ${BMNNSDK2_PATH}/include/bmlib
          ${BMNNSDK2_PATH}/include/bmruntime)
  set(opencv_inc_dirs ${BMNNSDK2_PATH}/include/opencv/opencv4)
  set(ffmpeg_inc_dirs ${BMNNSDK2_PATH}/include/ffmpeg)

  if (BUILD_X86_PCIE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(bmnn_link_dirs ${BMNNSDK2_PATH}/lib/bmnn/pcie
            ${BMNNSDK2_PATH}/lib/decode/x86)
    ## OpenCV lib dirs
    set(opencv_link_dirs ${BMNNSDK2_PATH}/lib/opencv/x86)
    ## Ffmpeg
    set(ffmpeg_link_dirs ${BMNNSDK2_PATH}/lib/ffmpeg/x86)

  elseif (BUILD_ARM_PCIE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
    set(bmnn_link_dirs ${BMNNSDK2_PATH}/lib/bmnn/arm_pcie
            ${BMNNSDK2_PATH}/lib/decode/arm_pcie)
    ## OpenCV
    set(opencv_link_dirs ${BMNNSDK2_PATH}/lib/opencv/arm_pcie)
    ## Ffmpeg
    set(ffmpeg_link_dirs ${BMNNSDK2_PATH}/lib/ffmpeg/arm_pcie)
  elseif (BUILD_SOC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
    set(bmnn_link_dirs ${BMNNSDK2_PATH}/lib/bmnn/soc
            ${BMNNSDK2_PATH}/lib/decode/soc)
    ## OpenCV
    set(opencv_link_dirs ${BMNNSDK2_PATH}/lib/opencv/soc)
    ## Ffmpeg
    set(ffmpeg_link_dirs ${BMNNSDK2_PATH}/lib/ffmpeg/soc)
  elseif (BUILD_MIPS64)

  elseif (BUILD_CMODEL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(bmnn_link_dirs ${BMNNSDK2_PATH}/lib/bmnn/cmodel
            ${BMNNSDK2_PATH}/lib/decode/cmodel
            )
    ## OpenCV
    #find_package(OpenCV 3 REQUIRED)
    set(opencv_link_dirs ${BMNNSDK2_PATH}/lib/opencv/cmodel)
    ## Ffmpeg
    set(ffmpeg_link_dirs ${BMNNSDK2_PATH}/lib/ffmpeg/cmodel)
  elseif (BUILD_SW64)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    set(bmnn_link_dirs ${BMNNSDK2_PATH}/lib/bmnn/sw64
            ${BMNNSDK2_PATH}/lib/decode/sw64
            )
    ## OpenCV
    #find_package(OpenCV 3 REQUIRED)
    set(opencv_link_dirs ${BMNNSDK2_PATH}/lib/opencv/sw64)
    ## Ffmpeg
    set(ffmpeg_link_dirs ${BMNNSDK2_PATH}/lib/ffmpeg/sw64)
  elseif (BUILD_LOONGARCH64)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    set(bmnn_link_dirs ${BMNNSDK2_PATH}/lib/bmnn/loongarch64
            ${BMNNSDK2_PATH}/lib/decode/loongarch64
            )
    ## OpenCV
    #find_package(OpenCV 3 REQUIRED)
    set(opencv_link_dirs ${BMNNSDK2_PATH}/lib/opencv/loongarch64)
    ## Ffmpeg
    set(ffmpeg_link_dirs ${BMNNSDK2_PATH}/lib/ffmpeg/loongarch64)
  else ()
    message(ERROR "not supported")
  endif ()
else()
  # all-in-one release mode
      if(USE_CENTOS)
          set(CXX_ABI CXX11_ABI0)
      else()
          set(CXX_ABI CXX11_ABI1)
      endif()
  set(NNTC_PATH /${JINKINS_TOP}/nntoolchain)
  set(BM168X_PATH /${JINKINS_TOP}/bm168x)
  set(bmnn_inc_dirs ${NNTC_PATH}/include
          ${BM168X_PATH}/include
          ${NNTC_PATH}/include/bmruntime)

  if (BUILD_X86_PCIE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
      set(bmnn_link_dirs ${NNTC_PATH}/lib/bmnn/pcie
              ${NNTC_PATH}/lib/bmnn/pcie/lib_${CXX_ABI}
              ${BM168X_PATH}/libs/pcie
              )

      ## OpenCV inc/lib dirs
      set(opencv_inc_dirs /${JINKINS_TOP}/pcie_bm1684_asic/opencv/include/opencv4)
      set(opencv_link_dirs /${JINKINS_TOP}/pcie_bm1684_asic/opencv/lib)
      ## Ffmpeg
      set(ffmpeg_inc_dirs /${JINKINS_TOP}/pcie_bm1684_asic/ffmpeg/usr/local/include)
      set(ffmpeg_link_dirs /${JINKINS_TOP}/pcie_bm1684_asic/ffmpeg/usr/local/lib
              /${JINKINS_TOP}/pcie_bm1684_asic/decode/lib)

  elseif (BUILD_ARM_PCIE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
      set(bmnn_link_dirs ${NNTC_PATH}/lib/bmnn/arm_pcie
              ${NNTC_PATH}/lib/bmnn/arm_pcie/lib_${CXX_ABI}
              ${BM168X_PATH}/libs/arm_pcie)
      ## OpenCV inc/lib dirs
      set(opencv_inc_dirs /${JINKINS_TOP}/pcie_arm64_bm1684_asic/opencv/include/opencv4)
      set(opencv_link_dirs /${JINKINS_TOP}/pcie_arm64_bm1684_asic/opencv/lib)
      ## Ffmpeg
      set(ffmpeg_inc_dirs /${JINKINS_TOP}/pcie_arm64_bm1684_asic/ffmpeg/usr/local/include)
      set(ffmpeg_link_dirs /${JINKINS_TOP}/pcie_arm64_bm1684_asic/ffmpeg/usr/local/lib
              /${JINKINS_TOP}/pcie_arm64_bm1684_asic/decode/lib)
  elseif (BUILD_SOC)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
      set(bmnn_link_dirs ${NNTC_PATH}/lib/bmnn/soc
              ${BM168X_PATH}/libs/soc)
      ## OpenCV inc/lib dirs
      set(opencv_inc_dirs /${JINKINS_TOP}/soc_bm1684_asic/opencv/include/opencv4)
      set(opencv_link_dirs /${JINKINS_TOP}/soc_bm1684_asic/opencv/lib)
      ## Ffmpeg
      set(ffmpeg_inc_dirs /${JINKINS_TOP}/soc_bm1684_asic/ffmpeg/usr/local/include)
      set(ffmpeg_link_dirs /${JINKINS_TOP}/soc_bm1684_asic/ffmpeg/usr/local/lib
              /${JINKINS_TOP}/soc_bm1684_asic/decode/lib)
  elseif (BUILD_MIPS64)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++14")
      set(bmnn_link_dirs ${NNTC_PATH}/lib/bmnn/mips64/lib_${CXX_ABI}
              ${BM168X_PATH}/libs/mips64)
      ## OpenCV inc/lib dirs
      set(opencv_inc_dirs /${JINKINS_TOP}/pcie_mips64_bm1684_asic/opencv/include/opencv4)
      set(opencv_link_dirs /${JINKINS_TOP}/pcie_mips64_bm1684_asic/opencv/lib)
      ## Ffmpeg
      set(ffmpeg_inc_dirs /${JINKINS_TOP}/pcie_mips64_bm1684_asic/ffmpeg/usr/local/include)
      set(ffmpeg_link_dirs /${JINKINS_TOP}/pcie_mips64_bm1684_asic/ffmpeg/usr/local/lib
              /${JINKINS_TOP}/pcie_mips64_bm1684_asic/decode/lib)
  elseif (BUILD_CMODEL)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
      set(bmnn_link_dirs ${NNTC_PATH}/lib/bmnn/cmodel
              ${BM168X_PATH}/libs/cmodel)
      ## OpenCV
      #find_package(OpenCV 3 REQUIRED)
      #set(opencv_link_dirs ${BMNNSDK2_PATH}/lib/opencv/cmodel)
      ## Ffmpeg
      #set(ffmpeg_link_dirs ${BMNNSDK2_PATH}/lib/ffmpeg/cmodel)
  elseif (BUILD_SW64)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++14")
      set(bmnn_link_dirs ${NNTC_PATH}/lib/bmnn/sw64/lib_${CXX_ABI}
              ${BM168X_PATH}/libs/sw64)
      ## OpenCV inc/lib dirs
      set(opencv_inc_dirs /${JINKINS_TOP}/pcie_sw64_bm1684_asic/opencv/include/opencv4)
      set(opencv_link_dirs /${JINKINS_TOP}/pcie_sw64_bm1684_asic/opencv/lib)
      ## Ffmpeg
      set(ffmpeg_inc_dirs /${JINKINS_TOP}/pcie_sw64_bm1684_asic/ffmpeg/usr/local/include)
      set(ffmpeg_link_dirs /${JINKINS_TOP}/pcie_sw64_bm1684_asic/ffmpeg/usr/local/lib
              /${JINKINS_TOP}/pcie_sw64_bm1684_asic/decode/lib)
  elseif (BUILD_LOONGARCH64)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++11")
      set(bmnn_link_dirs ${NNTC_PATH}/lib/bmnn/loongarch64/lib_${CXX_ABI}
              ${BM168X_PATH}/libs/loongarch64)
      ## OpenCV inc/lib dirs
      set(opencv_inc_dirs /${JINKINS_TOP}/pcie_loongarch64_bm1684_asic/opencv/include/opencv4)
      set(opencv_link_dirs /${JINKINS_TOP}/pcie_loongarch64_bm1684_asic/opencv/lib)
      ## Ffmpeg
      set(ffmpeg_inc_dirs /${JINKINS_TOP}/pcie_loongarch64_bm1684_asic/ffmpeg/usr/local/include)
      set(ffmpeg_link_dirs /${JINKINS_TOP}/pcie_loongarch64_bm1684_asic/ffmpeg/usr/local/lib
              /${JINKINS_TOP}/pcie_loongarch64_bm1684_asic/decode/lib)
  else ()
      message(ERROR "not supported")
  endif ()
endif()


#common shared args for use_bmnnsdk2/use_local
if (NOT BUILD_CMODEL)
    set(bmnn_link_libs bmrt bmcv bmlib)
    set(opencv_link_libs opencv_calib3d opencv_core opencv_dnn opencv_features2d opencv_flann
            opencv_gapi opencv_highgui opencv_imgcodecs opencv_imgproc opencv_ml opencv_objdetect
            opencv_photo opencv_stitching opencv_video opencv_videoio)
    if (BUILD_LOONGARCH64)
      set(opencv_link_libs rt dl resolv opencv_calib3d opencv_core opencv_dnn opencv_features2d opencv_flann
            opencv_gapi opencv_highgui opencv_imgcodecs opencv_imgproc opencv_ml opencv_objdetect
            opencv_photo opencv_stitching opencv_video opencv_videoio)    
    endif ()
    set(ffmpeg_link_libs avutil swresample avcodec avformat avfilter avdevice swscale bmvideo bmvpuapi bmion bmjpuapi bmjpulite bmvpulite bmvppapi yuv)
    #set(ffmpeg_link_libs avcodec avformat avfilter avdevice swscale bmvideo)
else ()
    set(bmnn_link_libs bmrt bmcv bmlib)
    find_package(OpenCV)
    set(opencv_inc_dirs ${OpenCV_INCLUDE_DIRS})
    set(opencv_link_libs ${OpenCV_LIBS})
    set(ffmpeg_link_libs "")
endif ()

set(common_inc_dirs ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty ${CMAKE_CURRENT_SOURCE_DIR}/include ${bmnn_inc_dirs} ${opencv_inc_dirs} ${ffmpeg_inc_dirs})
set(common_link_dirs ${bmnn_link_dirs} ${opencv_link_dirs} ${ffmpeg_link_dirs})

message(STATUS "[sail]opencv_link_libs=${opencv_link_libs}")
message(STATUS "[sail]common_inc_dirs=${common_inc_dirs}")
message(STATUS "[sail]common_link_dirs=${common_link_dirs}")


if (NOT DEFINED CUSTOM_PY_LIBDIR)
    set(PYTHON_BIN ${CMAKE_CURRENT_SOURCE_DIR}/../bm_prebuilt_toolchains/pythons/Python-3.8.2/python_3.8.2/bin/python3.8)
    set(PYTHON_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../bm_prebuilt_toolchains/pythons/Python-3.8.2/python_3.8.2/lib)
    set(PYTHON_EXECUTABLE ${PYTHON_BIN})
    set(CUSTOM_PY_LIBDIR ${PYTHON_LIB})
    set(ENV{LD_LIBRARY_PATH} ${PYTHON_LIB})
endif()

message(STATUS "[sail] CUSTOM_PY_LIBDIR=${CUSTOM_PY_LIBDIR}")
include_directories(${common_inc_dirs})
link_directories(${common_link_dirs})

add_subdirectory(src)

# Add test
if(WITH_TEST)
  add_subdirectory(samples)
  enable_testing()
endif()

# use "make cpp" to build all pure cpp targets
# use "make pysail" to build lib for python3
if(USE_FFMPEG)
  if(IS_SOC_MODE)
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
