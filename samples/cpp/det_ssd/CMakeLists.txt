
message(STATUS "[det_ssd]bmnn_link_dirs=${bmnn_link_dirs}")

if (NOT WIN32)
	link_libraries(sail ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnn_link_libs} pthread)
  # build executable
add_executable(det_ssd_0 det_ssd_0.cpp processor.cpp cvdecoder.cpp)

if (USE_FFMPEG)
  if (USE_BMCV)
    add_executable(det_ssd_3 det_ssd_3.cpp processor.cpp cvdecoder.cpp)
    add_executable(det_ssd_4 det_ssd_4.cpp processor.cpp cvdecoder.cpp)
  endif ()

  add_executable(det_ssd_1 det_ssd_1.cpp processor.cpp)
  add_executable(det_ssd_2 det_ssd_2.cpp processor.cpp)
endif ()
else()
	link_libraries(libsail ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnn_link_libs})
  # build executable
add_executable(det_ssd_0 det_ssd_0.cpp processor.cpp cvdecoder.cpp getopt_win.c)

if (USE_FFMPEG)
  if (USE_BMCV)
    add_executable(det_ssd_3 det_ssd_3.cpp processor.cpp cvdecoder.cpp getopt_win.c)
    add_executable(det_ssd_4 det_ssd_4.cpp processor.cpp cvdecoder.cpp getopt_win.c)
  endif ()

  add_executable(det_ssd_1 det_ssd_1.cpp processor.cpp getopt_win.c)
  add_executable(det_ssd_2 det_ssd_2.cpp processor.cpp getopt_win.c)
endif ()
endif()


