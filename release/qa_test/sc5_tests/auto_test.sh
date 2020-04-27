#!/bin/bash

# Usage:
#     Option 1: ./auto_test.sh
#         use './data' as default directory to save test models and data
#     Option 2: ./auto_test.sh $DATA_DIR
#         use specified directory to save test models and data
#
# Attention: bmnnsdk2 should be installed by hand as following.
# cd bmnnsdk2-bm1684_vx.x.x/scripts
# ./install_lib.sh nntc
#
# # for x86
# source envsetup_pcie.sh
# sudo ./remove_driver_pcie.sh
# sudo ./install_driver_pcie.sh
# pip3 uninstall -y sophon
# # get your python version
# python3 -V
# # choose the same verion of sophon wheel to install
# # the following py3x maybe py35, py36, py37 or py38
# pip3 install ../lib/sail/python3/pcie/py3x/sophon-2.0.3-py3-none-any.whl --user
#
# # for arm
# source envsetup_arm_pcie.sh
# sudo ./remove_driver_arm_pcie.sh
# sudo ./install_driver_arm_pcie.sh
# pip3 uninstall -y sophon
# # get your python version
# python3 -V
# # choose the same verion of sophon wheel to install
# # the following py3x maybe py35, py36, py37 or py38
# pip3 install ../lib/sail/python3/arm_pcie/py3x/sophon-2.0.3-py3-none-any.whl --user

function judge_ret() {
  if [[ $1 == 0 ]]; then
    echo "Passed: $2"
    echo ""
  else
    echo "Failed: $2"
    exit 1
  fi
  sleep 1
}

function get_tpu_num() {
  local tpu_num=$(python3 -c "from sophon import sail; print(sail.get_available_tpu_num())")
  echo $tpu_num
}

function get_tpu_ids() {
  local tpu_num=$(python3 -c "from sophon import sail; print(sail.get_available_tpu_num())")
  declare -a tpus
  for i in $( seq 0 $(($tpu_num-1)) )
  do
    tpus+=($i)
  done
  echo ${tpus[*]}
}

function prepare_data() {
  if [ ! -d $DATA_DIR ]; then
    mkdir $DATA_DIR
  fi
  python3 ./download.py all --save_path $DATA_DIR
  judge_ret $? "download models and data"
}

function build_cpp() {
  if [ ! -d ./build ]; then
    mkdir ./build
  else
    rm -rf ./build/*
  fi
  pushd ./build
  local arch=$(arch)
  if [[ $arch == "aarch64" ]]; then
    cmake -DBUILD_ON_ARM=ON ..
  else
    cmake ..
  fi
  make -j
  judge_ret $? "build cpp"
  popd
}

function cpp_test() {
  local tpu_id=$1
  local tpu_num=$2
  local loops=$3
  local arch=$(arch)
  # test cls_resnet
  ./build/bin/cls_resnet_0 --bmodel $DATA_DIR/resnet50_fp32_191115.bmodel --input $DATA_DIR/cls.jpg --tpu_id $tpu_id --loops $loops --compare ./cpp/cls_resnet/verify_files/verify_resnet50.ini
  judge_ret $? "cls_resnet_0 fp32"
  ./build/bin/cls_resnet_0 --bmodel $DATA_DIR/resnet50_int8_191115.bmodel --input $DATA_DIR/cls.jpg --tpu_id $tpu_id --loops $loops --compare ./cpp/cls_resnet/verify_files/verify_resnet50.ini
  judge_ret $? "cls_resnet_0 int8"
  ./build/bin/cls_resnet_1 --bmodel $DATA_DIR/resnet50_fp32_191115.bmodel --input $DATA_DIR/cls.jpg --tpu_id $tpu_id --threads 2 --loops $loops --compare ./cpp/cls_resnet/verify_files/verify_resnet50.ini
  judge_ret $? "cls_resnet_1 fp32"
  ./build/bin/cls_resnet_1 --bmodel $DATA_DIR/resnet50_int8_191115.bmodel --input $DATA_DIR/cls.jpg --tpu_id $tpu_id --threads 2 --loops $loops --compare ./cpp/cls_resnet/verify_files/verify_resnet50.ini
  judge_ret $? "cls_resnet_1 int8"
  ./build/bin/cls_resnet_2 --bmodel $DATA_DIR/resnet50_fp32_191115.bmodel --bmodel $DATA_DIR/resnet50_int8_191115.bmodel --input $DATA_DIR/cls.jpg --tpu_id $tpu_id --loops $loops --compare ./cpp/cls_resnet/verify_files/verify_resnet50.ini
  judge_ret $? "cls_resnet_2"
  if [[ $tpu_id == $(($tpu_num-1)) ]]; then
    ./build/bin/cls_resnet_3 --bmodel $DATA_DIR/resnet50_fp32_191115.bmodel --input $DATA_DIR/cls.jpg --tpu_id $tpu_id --loops $loops --compare ./cpp/cls_resnet/verify_files/verify_resnet50.ini
    judge_ret $? "cls_resnet_3 fp32"
    ./build/bin/cls_resnet_3 --bmodel $DATA_DIR/resnet50_int8_191115.bmodel --input $DATA_DIR/cls.jpg --tpu_id $tpu_id --loops $loops --compare ./cpp/cls_resnet/verify_files/verify_resnet50.ini
    judge_ret $? "cls_resnet_3 int8"
  else
    ./build/bin/cls_resnet_3 --bmodel $DATA_DIR/resnet50_fp32_191115.bmodel --input $DATA_DIR/cls.jpg --tpu_id $tpu_id --tpu_id $(($tpu_id+1)) --loops $loops --compare ./cpp/cls_resnet/verify_files/verify_resnet50.ini
    judge_ret $? "cls_resnet_3 fp32"
    ./build/bin/cls_resnet_3 --bmodel $DATA_DIR/resnet50_int8_191115.bmodel --input $DATA_DIR/cls.jpg --tpu_id $tpu_id --tpu_id $(($tpu_id+1)) --loops $loops --compare ./cpp/cls_resnet/verify_files/verify_resnet50.ini
    judge_ret $? "cls_resnet_3 int8"
  fi
  # test det_ssd
  ./build/bin/det_ssd_0 --bmodel $DATA_DIR/ssd_fp32_191115.bmodel --input $DATA_DIR/det.jpg --tpu_id $tpu_id --loops $loops --compare ./cpp/det_ssd/verify_files/verify_ssd_0_fp32_image.ini
  judge_ret $? "det_ssd_0 fp32 image"
  if [[ $arch == "aarch64" ]]; then
    ./build/bin/det_ssd_0 --bmodel $DATA_DIR/ssd_fp32_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./cpp/det_ssd/verify_files/verify_ssd_0_fp32_video_arm.ini
  else
    ./build/bin/det_ssd_0 --bmodel $DATA_DIR/ssd_fp32_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./cpp/det_ssd/verify_files/verify_ssd_0_fp32_video.ini
  fi
  judge_ret $? "det_ssd_0 fp32 video"
  ./build/bin/det_ssd_0 --bmodel $DATA_DIR/ssd_int8_191115.bmodel --input $DATA_DIR/det.jpg --tpu_id $tpu_id --loops $loops --compare ./cpp/det_ssd/verify_files/verify_ssd_0_int8_image.ini
  judge_ret $? "det_ssd_0 int8 image"
  if [[ $arch == "aarch64" ]]; then
    ./build/bin/det_ssd_0 --bmodel $DATA_DIR/ssd_int8_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./cpp/det_ssd/verify_files/verify_ssd_0_int8_video_arm.ini
  else
    ./build/bin/det_ssd_0 --bmodel $DATA_DIR/ssd_int8_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./cpp/det_ssd/verify_files/verify_ssd_0_int8_video.ini
  fi
  judge_ret $? "det_ssd_0 int8 video"
  ./build/bin/det_ssd_1 --bmodel $DATA_DIR/ssd_fp32_191115.bmodel --input $DATA_DIR/det.jpg --tpu_id $tpu_id --loops $loops --compare ./cpp/det_ssd/verify_files/verify_ssd_1_fp32_image.ini
  judge_ret $? "det_ssd_1 fp32 image"
  ./build/bin/det_ssd_1 --bmodel $DATA_DIR/ssd_fp32_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./cpp/det_ssd/verify_files/verify_ssd_1_fp32_video.ini
  judge_ret $? "det_ssd_1 fp32 video"
  ./build/bin/det_ssd_1 --bmodel $DATA_DIR/ssd_int8_191115.bmodel --input $DATA_DIR/det.jpg --tpu_id $tpu_id --loops $loops --compare ./cpp/det_ssd/verify_files/verify_ssd_1_int8_image.ini
  judge_ret $? "det_ssd_1 int8 image"
  ./build/bin/det_ssd_1 --bmodel $DATA_DIR/ssd_int8_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./cpp/det_ssd/verify_files/verify_ssd_1_int8_video.ini
  judge_ret $? "det_ssd_1 int8 video"
  ./build/bin/det_ssd_2 --bmodel $DATA_DIR/ssd_fp32_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./cpp/det_ssd/verify_files/verify_ssd_2_fp32_video.ini
  judge_ret $? "det_ssd_2 fp32 video"
  ./build/bin/det_ssd_2 --bmodel $DATA_DIR/ssd_int8_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./cpp/det_ssd/verify_files/verify_ssd_2_int8_video.ini
  judge_ret $? "det_ssd_2 int8 video"
  # test det_yolov3
  if [[ $arch == "aarch64" ]]; then
    ./build/bin/det_yolov3_0 --bmodel $DATA_DIR/yolov3_fp32_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./cpp/det_yolov3/verify_files/verify_yolov3_fp32_0_arm.ini
  else
    ./build/bin/det_yolov3_0 --bmodel $DATA_DIR/yolov3_fp32_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./cpp/det_yolov3/verify_files/verify_yolov3_fp32_0.ini
  fi
  judge_ret $? "det_yolov3_0 fp32"
  if [[ $arch == "aarch64" ]]; then
    ./build/bin/det_yolov3_0 --bmodel $DATA_DIR/yolov3_int8_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./cpp/det_yolov3/verify_files/verify_yolov3_int8_0_arm.ini
  else
    ./build/bin/det_yolov3_0 --bmodel $DATA_DIR/yolov3_int8_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./cpp/det_yolov3/verify_files/verify_yolov3_int8_0.ini
  fi
  judge_ret $? "det_yolov3_0 int8"
  ./build/bin/det_yolov3_1 --bmodel $DATA_DIR/yolov3_fp32_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./cpp/det_yolov3/verify_files/verify_yolov3_fp32_1.ini
  judge_ret $? "det_yolov3_1 fp32"
  ./build/bin/det_yolov3_1 --bmodel $DATA_DIR/yolov3_int8_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./cpp/det_yolov3/verify_files/verify_yolov3_int8_1.ini
  judge_ret $? "det_yolov3_1 int8"
  # test det_mtcnn
  ./build/bin/det_mtcnn --bmodel $DATA_DIR/mtcnn_fp32_191115.bmodel --input $DATA_DIR/face.jpg --tpu_id $tpu_id --loops $loops --compare ./cpp/det_mtcnn/verify_files/verify_mtcnn.ini
  judge_ret $? "det_mtcnn"
  echo "C++ tests passed on TPU $tpu_id and loops of $loops!"
}

function python_test() {
  local tpu_id=$1
  local tpu_num=$2
  local loops=$3
  local arch=$(arch)
  # test cls_resnet
  python3 ./python/cls_resnet/cls_resnet_0.py --bmodel $DATA_DIR/resnet50_fp32_191115.bmodel --input $DATA_DIR/cls.jpg --tpu_id $tpu_id --loops $loops --compare ./python/cls_resnet/verify_files/verify_resnet50.json
  judge_ret $? "cls_resnet_0 fp32"
  python3 ./python/cls_resnet/cls_resnet_0.py --bmodel $DATA_DIR/resnet50_int8_191115.bmodel --input $DATA_DIR/cls.jpg --tpu_id $tpu_id --loops $loops --compare ./python/cls_resnet/verify_files/verify_resnet50.json
  judge_ret $? "cls_resnet_0 int8"
  python3 ./python/cls_resnet/cls_resnet_1.py --bmodel $DATA_DIR/resnet50_fp32_191115.bmodel --input $DATA_DIR/cls.jpg --threads 2 --tpu_id $tpu_id --loops $loops --compare ./python/cls_resnet/verify_files/verify_resnet50.json
  judge_ret $? "cls_resnet_1 fp32"
  python3 ./python/cls_resnet/cls_resnet_1.py --bmodel $DATA_DIR/resnet50_int8_191115.bmodel --input $DATA_DIR/cls.jpg --threads 2 --tpu_id $tpu_id --loops $loops --compare ./python/cls_resnet/verify_files/verify_resnet50.json
  judge_ret $? "cls_resnet_1 int8"
  python3 ./python/cls_resnet/cls_resnet_2.py --bmodel $DATA_DIR/resnet50_fp32_191115.bmodel --bmodel $DATA_DIR/resnet50_int8_191115.bmodel --input $DATA_DIR/cls.jpg --tpu_id $tpu_id --loops $loops --compare ./python/cls_resnet/verify_files/verify_resnet50.json
  judge_ret $? "cls_resnet_2"
  if [[ $tpu_id == $(($tpu_num-1)) ]]; then
    python3 ./python/cls_resnet/cls_resnet_3.py --bmodel $DATA_DIR/resnet50_fp32_191115.bmodel --input $DATA_DIR/cls.jpg --tpu_id $tpu_id --loops $loops --compare ./python/cls_resnet/verify_files/verify_resnet50.json
    judge_ret $? "cls_resnet_3 fp32"
    python3 ./python/cls_resnet/cls_resnet_3.py --bmodel $DATA_DIR/resnet50_int8_191115.bmodel --input $DATA_DIR/cls.jpg --tpu_id $tpu_id --loops $loops --compare ./python/cls_resnet/verify_files/verify_resnet50.json
    judge_ret $? "cls_resnet_3 int8"
  else
    python3 ./python/cls_resnet/cls_resnet_3.py --bmodel $DATA_DIR/resnet50_fp32_191115.bmodel --input $DATA_DIR/cls.jpg --tpu_id $tpu_id --tpu_id $(($tpu_id+1)) --loops $loops --compare ./python/cls_resnet/verify_files/verify_resnet50.json
    judge_ret $? "cls_resnet_3 fp32"
    python3 ./python/cls_resnet/cls_resnet_3.py --bmodel $DATA_DIR/resnet50_int8_191115.bmodel --input $DATA_DIR/cls.jpg --tpu_id $tpu_id --tpu_id $(($tpu_id+1)) --loops $loops --compare ./python/cls_resnet/verify_files/verify_resnet50.json
    judge_ret $? "cls_resnet_3 int8"
  fi
  # test det_ssd
  python3 ./python/det_ssd/det_ssd_0.py --bmodel $DATA_DIR/ssd_fp32_191115.bmodel --input $DATA_DIR/det.jpg --tpu_id $tpu_id --loops $loops --compare ./python/det_ssd/verify_files/verify_ssd_0_fp32_image.json
  judge_ret $? "det_ssd_0 fp32 image"
  if [[ $arch == "aarch64" ]]; then
    python3 ./python/det_ssd/det_ssd_0.py --bmodel $DATA_DIR/ssd_fp32_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./python/det_ssd/verify_files/verify_ssd_0_fp32_video_arm.json
  else
    python3 ./python/det_ssd/det_ssd_0.py --bmodel $DATA_DIR/ssd_fp32_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./python/det_ssd/verify_files/verify_ssd_0_fp32_video.json
  fi
  judge_ret $? "det_ssd_0 fp32 video"
  python3 ./python/det_ssd/det_ssd_0.py --bmodel $DATA_DIR/ssd_int8_191115.bmodel --input $DATA_DIR/det.jpg --tpu_id $tpu_id --loops $loops --compare ./python/det_ssd/verify_files/verify_ssd_0_int8_image.json
  judge_ret $? "det_ssd_0 int8 image"
  if [[ $arch == "aarch64" ]]; then
    python3 ./python/det_ssd/det_ssd_0.py --bmodel $DATA_DIR/ssd_int8_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./python/det_ssd/verify_files/verify_ssd_0_int8_video_arm.json
  else
    python3 ./python/det_ssd/det_ssd_0.py --bmodel $DATA_DIR/ssd_int8_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./python/det_ssd/verify_files/verify_ssd_0_int8_video.json
  fi
  judge_ret $? "det_ssd_0 int8 video"
  python3 ./python/det_ssd/det_ssd_1.py --bmodel $DATA_DIR/ssd_fp32_191115.bmodel --input $DATA_DIR/det.jpg --tpu_id $tpu_id --loops $loops --compare ./python/det_ssd/verify_files/verify_ssd_1_fp32_image.json
  judge_ret $? "det_ssd_1 fp32 image"
  python3 ./python/det_ssd/det_ssd_1.py --bmodel $DATA_DIR/ssd_fp32_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./python/det_ssd/verify_files/verify_ssd_1_fp32_video.json
  judge_ret $? "det_ssd_1 fp32 video"
  python3 ./python/det_ssd/det_ssd_1.py --bmodel $DATA_DIR/ssd_int8_191115.bmodel --input $DATA_DIR/det.jpg --tpu_id $tpu_id --loops $loops --compare ./python/det_ssd/verify_files/verify_ssd_1_int8_image.json
  judge_ret $? "det_ssd_1 int8 image"
  python3 ./python/det_ssd/det_ssd_1.py --bmodel $DATA_DIR/ssd_int8_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./python/det_ssd/verify_files/verify_ssd_1_int8_video.json
  judge_ret $? "det_ssd_1 int8 video"
  python3 ./python/det_ssd/det_ssd_2.py --bmodel $DATA_DIR/ssd_fp32_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./python/det_ssd/verify_files/verify_ssd_2_fp32_video.json
  judge_ret $? "det_ssd_2 fp32 video"
  python3 ./python/det_ssd/det_ssd_2.py --bmodel $DATA_DIR/ssd_int8_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./python/det_ssd/verify_files/verify_ssd_2_int8_video.json
  judge_ret $? "det_ssd_2 int8 video"
  # test det_yolov3
  if [[ $arch == "aarch64" ]]; then
    python3 ./python/det_yolov3/det_yolov3.py --bmodel $DATA_DIR/yolov3_fp32_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./python/det_yolov3/verify_files/verify_yolov3_fp32_arm.json
  else
    python3 ./python/det_yolov3/det_yolov3.py --bmodel $DATA_DIR/yolov3_fp32_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./python/det_yolov3/verify_files/verify_yolov3_fp32.json
  fi
  judge_ret $? "det_yolov3 fp32 video"
  if [[ $arch == "aarch64" ]]; then
    python3 ./python/det_yolov3/det_yolov3.py --bmodel $DATA_DIR/yolov3_int8_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./python/det_yolov3/verify_files/verify_yolov3_int8_arm.json
  else
    python3 ./python/det_yolov3/det_yolov3.py --bmodel $DATA_DIR/yolov3_int8_191115.bmodel --input $DATA_DIR/det.h264 --tpu_id $tpu_id --loops $loops --compare ./python/det_yolov3/verify_files/verify_yolov3_int8.json
  fi
  judge_ret $? "det_yolov3 int8 video"
  # test det_mtcnn
  python3 ./python/det_mtcnn/det_mtcnn.py --bmodel $DATA_DIR/mtcnn_fp32_191115.bmodel --input $DATA_DIR/face.jpg --tpu_id $tpu_id --loops $loops --compare ./python/det_mtcnn/verify_files/verify_mtcnn.json
  judge_ret $? "det_mtcnn fp32 image"
  echo "Python tests passed on TPU $tpu_id and loops of $loops!"
}

DATA_DIR=./data
if [ $# == 1 ]; then
  DATA_DIR=$1
fi

TPU_NUM=$(get_tpu_num)
TPU_IDS=$(get_tpu_ids)
prepare_data
build_cpp

# test on all TPUS for loops 1 and 200
LOOPS=(1 200)
for loops in ${LOOPS[@]}
do
  for tpu_id in ${TPU_IDS[@]}
  do
    cpp_test $tpu_id $TPU_NUM $loops
    python_test $tpu_id $TPU_NUM $loops
  done
done

echo "All tests passed!"
