message(STATUS "Finding Bitmain FFMPEG")

set(FFMPEG_COMPONENTS
    swresample
    swscale
    avformat
    avutil
    avdevice
    avfilter
    avcodec)

set(FFMPEG_INCLUDE_DIRS
  ${BMNNSDK2_PATH}/include/ffmpeg)

foreach(ITEM ${FFMPEG_COMPONENTS})
  list(APPEND FFMPEG_LIBRARIES "${BMNNSDK2_PATH}/lib/ffmpeg/arm_pcie/lib${ITEM}.so")
endforeach()

if(FFMPEG_INCLUDE_DIRS AND FFMPEG_LIBRARIES)
    set(FFMPEG_FOUND TRUE)
    message(STATUS "Bitmain FFMPEG found")
else(FFMPEG_INCLUDE_DIRS AND FFMPEG_LIBRARIES)
    message(STATUS "Bitmain FFMPEG not found")
endif(FFMPEG_INCLUDE_DIRS AND FFMPEG_LIBRARIES)
