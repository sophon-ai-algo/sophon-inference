message(STATUS "Finding Bitmain OpenCV")
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
    "$ENV{REL_TOP}/include/opencv/opencv4")
message(STATUS ${OpenCV_INCLUDE_DIRS})
foreach(ITEM ${OpenCV_COMPONENTS})
    list(APPEND OpenCV_LIBRARIES "$ENV{REL_TOP}/lib/opencv/arm_pcie/lib${ITEM}.so")
endforeach()
if(OpenCV_INCLUDE_DIRS AND OpenCV_LIBRARIES)
    set(OpenCV_FOUND TRUE)
    message(STATUS "Bitmain OpenCV found")
else(OpenCV_INCLUDE_DIRS AND OpenCV_LIBRARIES)
    message(STATUS "Bitmain OpenCV not found")
endif(OpenCV_INCLUDE_DIRS AND OpenCV_LIBRARIES)
