#!/bin/bash

function judge_ret() {
  if [[ $1 == 0 ]]; then
    echo "Passed: $2"
    echo ""
  else
    echo "Failed: $2"
    exit 1
  fi
}

function cpp_test() {
  # test cls_resnet
  ./build/bin/cls_resnet_0 --bmodel $DATA_DIR/resnet50_fp32.bmodel --input $DATA_DIR/cls.jpg --loops 1 --compare $DATA_DIR/verify/cpp/verify_resnet50.ini
  judge_ret $? "cls_resnet_0 fp32"
  ./build/bin/cls_resnet_0 --bmodel $DATA_DIR/resnet50_int8.bmodel --input $DATA_DIR/cls.jpg --loops 1 --compare $DATA_DIR/verify/cpp/verify_resnet50.ini
  judge_ret $? "cls_resnet_0 int8"
  ./build/bin/cls_resnet_1 --bmodel $DATA_DIR/resnet50_fp32.bmodel --input $DATA_DIR/cls.jpg --threads 2 --loops 1 --compare $DATA_DIR/verify/cpp/verify_resnet50.ini
  judge_ret $? "cls_resnet_1 fp32"
  ./build/bin/cls_resnet_1 --bmodel $DATA_DIR/resnet50_int8.bmodel --input $DATA_DIR/cls.jpg --threads 2 --loops 1 --compare $DATA_DIR/verify/cpp/verify_resnet50.ini
  judge_ret $? "cls_resnet_1 int8"
  ./build/bin/cls_resnet_2 --bmodel $DATA_DIR/resnet50_fp32.bmodel --bmodel $DATA_DIR/resnet50_int8.bmodel --input $DATA_DIR/cls.jpg --loops 1 --compare $DATA_DIR/verify/cpp/verify_resnet50.ini
  judge_ret $? "cls_resnet_2"
  ./build/bin/cls_resnet_3 --bmodel $DATA_DIR/resnet50_fp32.bmodel --input $DATA_DIR/cls.jpg --tpu_id 0 --loops 1 --compare $DATA_DIR/verify/cpp/verify_resnet50.ini
  judge_ret $? "cls_resnet_3 fp32"
  ./build/bin/cls_resnet_3 --bmodel $DATA_DIR/resnet50_int8.bmodel --input $DATA_DIR/cls.jpg --tpu_id 0 --loops 1 --compare $DATA_DIR/verify/cpp/verify_resnet50.ini
  judge_ret $? "cls_resnet_3 int8"
  # test det_ssd
  ./build/bin/det_ssd_0 --bmodel $DATA_DIR/ssd_fp32.bmodel --input $DATA_DIR/det.jpg --loops 1 --compare $DATA_DIR/verify/cpp/verify_ssd_0_fp32_image.ini
  judge_ret $? "det_ssd_0 fp32 image"
  ./build/bin/det_ssd_0 --bmodel $DATA_DIR/ssd_fp32.bmodel --input $DATA_DIR/det.h264 --loops 1 --compare $DATA_DIR/verify/cpp/verify_ssd_0_fp32_video.ini
  judge_ret $? "det_ssd_0 fp32 video"
  ./build/bin/det_ssd_0 --bmodel $DATA_DIR/ssd_int8.bmodel --input $DATA_DIR/det.jpg --loops 1 --compare $DATA_DIR/verify/cpp/verify_ssd_0_int8_image.ini
  judge_ret $? "det_ssd_0 int8 image"
  ./build/bin/det_ssd_0 --bmodel $DATA_DIR/ssd_int8.bmodel --input $DATA_DIR/det.h264 --loops 1 --compare $DATA_DIR/verify/cpp/verify_ssd_0_int8_video.ini
  judge_ret $? "det_ssd_0 int8 video"
  ./build/bin/det_ssd_1 --bmodel $DATA_DIR/ssd_fp32.bmodel --input $DATA_DIR/det.jpg --loops 1 --compare $DATA_DIR/verify/cpp/verify_ssd_1_fp32_image.ini
  judge_ret $? "det_ssd_1 fp32 image"
  ./build/bin/det_ssd_1 --bmodel $DATA_DIR/ssd_fp32.bmodel --input $DATA_DIR/det.h264 --loops 1 --compare $DATA_DIR/verify/cpp/verify_ssd_1_fp32_video.ini
  judge_ret $? "det_ssd_1 fp32 video"
  ./build/bin/det_ssd_1 --bmodel $DATA_DIR/ssd_int8.bmodel --input $DATA_DIR/det.jpg --loops 1 --compare $DATA_DIR/verify/cpp/verify_ssd_1_int8_image.ini
  judge_ret $? "det_ssd_1 int8 image"
  ./build/bin/det_ssd_1 --bmodel $DATA_DIR/ssd_int8.bmodel --input $DATA_DIR/det.h264 --loops 1 --compare $DATA_DIR/verify/cpp/verify_ssd_1_int8_video.ini
  judge_ret $? "det_ssd_1 int8 video"
  ./build/bin/det_ssd_2 --bmodel $DATA_DIR/ssd_fp32.bmodel --input $DATA_DIR/det.h264 --loops 1 --compare $DATA_DIR/verify/cpp/verify_ssd_2_fp32_video.ini
  judge_ret $? "det_ssd_2 fp32 video"
  ./build/bin/det_ssd_2 --bmodel $DATA_DIR/ssd_int8.bmodel --input $DATA_DIR/det.h264 --loops 1 --compare $DATA_DIR/verify/cpp/verify_ssd_2_int8_video.ini
  judge_ret $? "det_ssd_2 int8 video"
  # test det_yolov3
  ./build/bin/det_yolov3_0 --bmodel $DATA_DIR/yolov3_fp32.bmodel --input $DATA_DIR/det.h264 --loops 1 --compare $DATA_DIR/verify/cpp/verify_yolov3_fp32_0.ini
  judge_ret $? "det_yolov3_0 fp32"
  ./build/bin/det_yolov3_0 --bmodel $DATA_DIR/yolov3_int8.bmodel --input $DATA_DIR/det.h264 --loops 1 --compare $DATA_DIR/verify/cpp/verify_yolov3_int8_0.ini
  judge_ret $? "det_yolov3_0 int8"
  ./build/bin/det_yolov3_1 --bmodel $DATA_DIR/yolov3_fp32.bmodel --input $DATA_DIR/det.h264 --loops 1 --compare $DATA_DIR/verify/cpp/verify_yolov3_fp32_1.ini
  judge_ret $? "det_yolov3_1 fp32"
  ./build/bin/det_yolov3_1 --bmodel $DATA_DIR/yolov3_int8.bmodel --input $DATA_DIR/det.h264 --loops 1 --compare $DATA_DIR/verify/cpp/verify_yolov3_int8_1.ini
  judge_ret $? "det_yolov3_1 int8"
  # test det_mtcnn
  ./build/bin/det_mtcnn --bmodel $DATA_DIR/mtcnn_fp32.bmodel --input $DATA_DIR/face.jpg --loops 1 --compare $DATA_DIR/verify/cpp/verify_mtcnn.ini
  judge_ret $? "det_mtcnn"
  echo "C++ tests passed!"
}

function python_test() {
  # test cls_resnet
  python3 ./samples/python/cls_resnet/cls_resnet_0.py --bmodel $DATA_DIR/resnet50_fp32.bmodel --input $DATA_DIR/cls.jpg --loops 1 --compare $DATA_DIR/verify/python/verify_resnet50.json
  judge_ret $? "cls_resnet_0 fp32"
  python3 ./samples/python/cls_resnet/cls_resnet_0.py --bmodel $DATA_DIR/resnet50_int8.bmodel --input $DATA_DIR/cls.jpg --loops 1 --compare $DATA_DIR/verify/python/verify_resnet50.json
  judge_ret $? "cls_resnet_0 int8"
  python3 ./samples/python/cls_resnet/cls_resnet_1.py --bmodel $DATA_DIR/resnet50_fp32.bmodel --input $DATA_DIR/cls.jpg --threads 2 --loops 1 --compare $DATA_DIR/verify/python/verify_resnet50.json
  judge_ret $? "cls_resnet_1 fp32"
  python3 ./samples/python/cls_resnet/cls_resnet_1.py --bmodel $DATA_DIR/resnet50_int8.bmodel --input $DATA_DIR/cls.jpg --threads 2 --loops 1 --compare $DATA_DIR/verify/python/verify_resnet50.json
  judge_ret $? "cls_resnet_1 int8"
  python3 ./samples/python/cls_resnet/cls_resnet_2.py --bmodel $DATA_DIR/resnet50_fp32.bmodel --bmodel $DATA_DIR/resnet50_int8.bmodel --input $DATA_DIR/cls.jpg --loops 1 --compare $DATA_DIR/verify/python/verify_resnet50.json
  judge_ret $? "cls_resnet_2"
  python3 ./samples/python/cls_resnet/cls_resnet_3.py --bmodel $DATA_DIR/resnet50_fp32.bmodel --input $DATA_DIR/cls.jpg --tpu_id 0 --loops 1 --compare $DATA_DIR/verify/python/verify_resnet50.json
  judge_ret $? "cls_resnet_3 fp32"
  python3 ./samples/python/cls_resnet/cls_resnet_3.py --bmodel $DATA_DIR/resnet50_int8.bmodel --input $DATA_DIR/cls.jpg --tpu_id 0 --loops 1 --compare $DATA_DIR/verify/python/verify_resnet50.json
  judge_ret $? "cls_resnet_3 int8"
  # test det_ssd
  python3 ./samples/python/det_ssd/det_ssd_0.py --bmodel $DATA_DIR/ssd_fp32.bmodel --input $DATA_DIR/det.jpg --loops 1 --compare $DATA_DIR/verify/python/verify_ssd_0_fp32_image.json
  judge_ret $? "det_ssd_0 fp32 image"
  python3 ./samples/python/det_ssd/det_ssd_0.py --bmodel $DATA_DIR/ssd_fp32.bmodel --input $DATA_DIR/det.h264 --loops 1 --compare $DATA_DIR/verify/python/verify_ssd_0_fp32_video.json
  judge_ret $? "det_ssd_0 fp32 video"
  python3 ./samples/python/det_ssd/det_ssd_0.py --bmodel $DATA_DIR/ssd_int8.bmodel --input $DATA_DIR/det.jpg --loops 1 --compare $DATA_DIR/verify/python/verify_ssd_0_int8_image.json
  judge_ret $? "det_ssd_0 int8 image"
  python3 ./samples/python/det_ssd/det_ssd_0.py --bmodel $DATA_DIR/ssd_int8.bmodel --input $DATA_DIR/det.h264 --loops 1 --compare $DATA_DIR/verify/python/verify_ssd_0_int8_video.json
  judge_ret $? "det_ssd_0 int8 video"
  python3 ./samples/python/det_ssd/det_ssd_1.py --bmodel $DATA_DIR/ssd_fp32.bmodel --input $DATA_DIR/det.jpg --loops 1 --compare $DATA_DIR/verify/python/verify_ssd_1_fp32_image.json
  judge_ret $? "det_ssd_1 fp32 image"
  python3 ./samples/python/det_ssd/det_ssd_1.py --bmodel $DATA_DIR/ssd_fp32.bmodel --input $DATA_DIR/det.h264 --loops 1 --compare $DATA_DIR/verify/python/verify_ssd_1_fp32_video.json
  judge_ret $? "det_ssd_1 fp32 video"
  python3 ./samples/python/det_ssd/det_ssd_1.py --bmodel $DATA_DIR/ssd_int8.bmodel --input $DATA_DIR/det.jpg --loops 1 --compare $DATA_DIR/verify/python/verify_ssd_1_int8_image.json
  judge_ret $? "det_ssd_1 int8 image"
  python3 ./samples/python/det_ssd/det_ssd_1.py --bmodel $DATA_DIR/ssd_int8.bmodel --input $DATA_DIR/det.h264 --loops 1 --compare $DATA_DIR/verify/python/verify_ssd_1_int8_video.json
  judge_ret $? "det_ssd_1 int8 video"
  python3 ./samples/python/det_ssd/det_ssd_2.py --bmodel $DATA_DIR/ssd_fp32.bmodel --input $DATA_DIR/det.h264 --loops 1 --compare $DATA_DIR/verify/python/verify_ssd_2_fp32_video.json
  judge_ret $? "det_ssd_2 fp32 video"
  python3 ./samples/python/det_ssd/det_ssd_2.py --bmodel $DATA_DIR/ssd_int8.bmodel --input $DATA_DIR/det.h264 --loops 1 --compare $DATA_DIR/verify/python/verify_ssd_2_int8_video.json
  judge_ret $? "det_ssd_2 int8 video"
  # test det_yolov3
  python3 ./samples/python/det_yolov3/det_yolov3.py --bmodel $DATA_DIR/yolov3_fp32.bmodel --input $DATA_DIR/det.h264 --loops 1 --compare $DATA_DIR/verify/python/verify_yolov3_fp32.json
  judge_ret $? "det_yolov3 fp32 video"
  python3 ./samples/python/det_yolov3/det_yolov3.py --bmodel $DATA_DIR/yolov3_int8.bmodel --input $DATA_DIR/det.h264 --loops 1 --compare $DATA_DIR/verify/python/verify_yolov3_int8.json
  judge_ret $? "det_yolov3 int8 video"
  # test det_mtcnn
  python3 ./samples/python/det_mtcnn/det_mtcnn.py --bmodel $DATA_DIR/mtcnn_fp32.bmodel --input $DATA_DIR/face.jpg --loops 1 --compare $DATA_DIR/verify/python/verify_mtcnn.json
  judge_ret $? "det_mtcnn fp32 image"
  echo "Python tests passed!"
}

if [ $# != 1 ]; then
  echo "Usage: $0 data_dir(which contains all bmodels and input data)"
  exit 2
fi

DATA_DIR=$1
if [ ! -d $DATA_DIR ]; then
  echo "data_dir : $DATA_DIR is not an existed directory."
  exit 3
fi

cpp_test
python_test

echo "All tests passed!"
