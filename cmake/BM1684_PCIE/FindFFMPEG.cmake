message(STATUS "Finding Bitmain FFMPEG")

set(FFMPEG_COMPONENTS
    swresample
    swscale
    avformat
    avutil
    avdevice
    avfilter
    avcodec)

set(FFMPEG_INCLUDE_DIRS ${BMDECODER_DIR}/include/ffmpeg)

foreach(ITEM ${FFMPEG_COMPONENTS})
  list(APPEND FFMPEG_LIBRARIES "${BMDECODER_DIR}/libs/pcie/ffmpeg/lib${ITEM}.so")
endforeach()

if(FFMPEG_INCLUDE_DIRS AND FFMPEG_LIBRARIES)
  set(FFMPEG_FOUND TRUE)
  message(STATUS "Bitmain FFMPEG found")
else(FFMPEG_INCLUDE_DIRS AND FFMPEG_LIBRARIES)
  message(STATUS "Bitmain FFMPEG not found")
endif(FFMPEG_INCLUDE_DIRS AND FFMPEG_LIBRARIES)
