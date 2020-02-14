message(STATUS "Finding Bitmain OpenCV")
set(OpenCV_PATH ${SOC_BM1684}/opencv)
set(OpenCV_COMPONENTS
    opencv_calib3d
    opencv_core
    opencv_dnn
    opencv_features2d
    opencv_flann
    opencv_gapi
    opencv_highgui
    opencv_imgcodecs
    opencv_imgproc
    opencv_ml
    opencv_objdetect
    opencv_photo
    opencv_stitching
    opencv_video
    opencv_videoio)
set(OpenCV_INCLUDE_DIRS
    "${OpenCV_PATH}/include/opencv4")
foreach(ITEM ${OpenCV_COMPONENTS})
    list(APPEND OpenCV_LIBRARIES "${OpenCV_PATH}/lib/lib${ITEM}.so")
endforeach()
message(${OpenCV_LIBRARIES})
if(OpenCV_INCLUDE_DIRS AND OpenCV_LIBRARIES)
    set(OpenCV_FOUND TRUE)
    message(STATUS "Bitmain OpenCV found")
else(OpenCV_INCLUDE_DIRS AND OpenCV_LIBRARIES)
    message(STATUS "Bitmain OpenCV not found")
endif(OpenCV_INCLUDE_DIRS AND OpenCV_LIBRARIES)
