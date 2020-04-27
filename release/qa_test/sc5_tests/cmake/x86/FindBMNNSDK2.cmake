message(STATUS "Finding BMNNSDK2 SDK")
set(BMNNSDK2_COMPONETS BASE CV DNN BMLIB RUNTIME BMVIDEO BMJPUAPI BMJPULITE AVCODEC AVDEVICE AVFILTER AVFORMAT AVUTIL POSTPROC AVSWRESAMPLE AVSWSCALE SAIL)
# find sub dir include
find_path(BASE_INCLUDE_DIR
          NAMES op_code.h
          PATHS "${BMNNSDK2_PATH}/include"
          NO_DEFAULT_PATH)
find_path(BMLIB_INCLUDE_DIR
          NAMES bmlib_runtime.h bmlib_utils.h
          PATHS "${BMNNSDK2_PATH}/include/bmlib"
          NO_DEFAULT_PATH)
find_path(RUNTIME_INCLUDE_DIR
          NAMES bm_mem_desc.h bmruntime_interface.h
          PATHS "${BMNNSDK2_PATH}/include/bmruntime"
          NO_DEFAULT_PATH)
find_path(SAIL_INCLUDE_DIR
          NAMES engine.h graph.h
          PATHS "${BMNNSDK2_PATH}/include/sail"
          NO_DEFAULT_PATH)
find_path(BMJPUAPI_INCLUDE_DIR
          NAMES bmjpuapi.h
          PATHS "${BMNNSDK2_PATH}/include/decode"
          NO_DEFAULT_PATH)
find_path(AVCODEC_INCLUDE_DIR
          NAMES libavcodec
          PATHS "${BMNNSDK2_PATH}/include/ffmpeg"
          NO_DEFAULT_PATH)
find_path(AVDEVICE_INCLUDE_DIR
          NAMES libavdevice
          PATHS "${BMNNSDK2_PATH}/include/ffmpeg"
          NO_DEFAULT_PATH)
find_path(AVFILTER_INCLUDE_DIR
          NAMES libavfilter
          PATHS "${BMNNSDK2_PATH}/include/ffmpeg"
          NO_DEFAULT_PATH)
find_path(AVFORMAT_INCLUDE_DIR
          NAMES libavformat
          PATHS "${BMNNSDK2_PATH}/include/ffmpeg"
          NO_DEFAULT_PATH)
find_path(AVUTIL_INCLUDE_DIR
          NAMES libavutil
          PATHS "${BMNNSDK2_PATH}/include/ffmpeg"
          NO_DEFAULT_PATH)
find_path(AVSWRESAMPLE_INCLUDE_DIR
          NAMES libswresample
          PATHS "${BMNNSDK2_PATH}/include/ffmpeg"
          NO_DEFAULT_PATH)
find_path(AVSWSCALE_INCLUDE_DIR
          NAMES libswscale
          PATHS "${BMNNSDK2_PATH}/include/ffmpeg"
          NO_DEFAULT_PATH)
# find sub dir library
find_library(CV_LIBRARY
             NAMES bmcv
             PATHS "${BMNNSDK2_PATH}/lib/bmnn/pcie")
find_library(BMLIB_LIBRARY
             NAMES bmlib
             PATHS "${BMNNSDK2_PATH}/lib/bmnn/pcie")
find_library(RUNTIME_LIBRARY
             NAMES bmrt
             PATHS "${BMNNSDK2_PATH}/lib/bmnn/pcie")
find_library(SAIL_LIBRARY
             NAMES sail
             PATHS "${BMNNSDK2_PATH}/lib/sail/pcie")
find_library(BMVIDEO_LIBRARY
             NAMES bmvideo
             PATHS "${BMNNSDK2_PATH}/lib/decode/x86")
find_library(BMJPUAPI_LIBRARY
             NAMES bmjpuapi
             PATHS "${BMNNSDK2_PATH}/lib/decode/x86")
find_library(BMJPULITE_LIBRARY
             NAMES bmjpulite
             PATHS "${BMNNSDK2_PATH}/lib/decode/x86")
find_library(AVFORMAT_LIBRARY
             NAMES avformat
             PATHS "${BMNNSDK2_PATH}/lib/ffmpeg/x86"
             NO_DEFAULT_PATH)
find_library(AVCODEC_LIBRARY
             NAMES avcodec
             PATHS "${BMNNSDK2_PATH}/lib/ffmpeg/x86"
             NO_DEFAULT_PATH)
find_library(AVFILTER_LIBRARY
             NAMES avfilter
             PATHS "${BMNNSDK2_PATH}/lib/ffmpeg/x86"
             NO_DEFAULT_PATH)
find_library(AVDEVICE_LIBRARY
             NAMES avdevice
             PATHS "${BMNNSDK2_PATH}/lib/ffmpeg/x86"
             NO_DEFAULT_PATH)
find_library(AVUTIL_LIBRARY
             NAMES avutil
             PATHS "${BMNNSDK2_PATH}/lib/ffmpeg/x86"
             NO_DEFAULT_PATH)
find_library(AVSWRESAMPLE_LIBRARY
             NAMES swresample
             PATHS "${BMNNSDK2_PATH}/lib/ffmpeg/x86"
             NO_DEFAULT_PATH)
find_library(AVSWSCALE_LIBRARY
             NAMES swscale
             PATHS "${BMNNSDK2_PATH}/lib/ffmpeg/x86"
             NO_DEFAULT_PATH)

foreach(ITEM ${BMNNSDK2_COMPONETS})
  list(APPEND BMNNSDK2_INCLUDE_DIRS "${${ITEM}_INCLUDE_DIR}")
  list(APPEND BMNNSDK2_LIBRARIES    "${${ITEM}_LIBRARY}")
endforeach(ITEM)

if(BMNNSDK2_INCLUDE_DIRS AND BMNNSDK2_LIBRARIES)
  set(BMNNSDK2_FOUND TRUE)
  message(STATUS "BMNNSDK2 found")
  #message(STATUS ${BMNNSDK2_INCLUDE_DIRS})
  #message(STATUS ${BMNNSDK2_LIBRARIES})
else(BMNNSDK2_INCLUDE_DIRS AND BMNNSDK2_LIBRARIES)
  message(STATUS "BMNNSDK2 not found")
endif(BMNNSDK2_INCLUDE_DIRS AND BMNNSDK2_LIBRARIES)
