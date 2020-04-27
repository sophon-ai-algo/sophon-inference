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

function prepare_data() {
  if [ ! -d data ]; then
    mkdir data
  fi
  python3 ./scripts/download.py --save_path ./data resnet50_fp32.bmodel
  python3 ./scripts/download.py --save_path ./data resnet50_int8.bmodel
  python3 ./scripts/download.py --save_path ./data ssd_fp32.bmodel
  python3 ./scripts/download.py --save_path ./data ssd_int8.bmodel
  python3 ./scripts/download.py --save_path ./data cls.jpg
  python3 ./scripts/download.py --save_path ./data det.h264
  judge_ret $? "download models and data"
}

function test_resnet50() {
  export LD_LIBRARY_PATH=../../lib/sail/soc:$LD_LIBRARY_PATH
  ./cls_resnet/cls_resnet_0 --bmodel ./data/resnet50_fp32_191115.bmodel --input ./data/cls.jpg
  judge_ret $? "cls_resnet_0 with FP32 model"
  ./cls_resnet/cls_resnet_0 --bmodel ./data/resnet50_int8_191115.bmodel --input ./data/cls.jpg
  judge_ret $? "cls_resnet_0 with INT8 model"
  ./cls_resnet/cls_resnet_1 --bmodel ./data/resnet50_fp32_191115.bmodel --input ./data/cls.jpg --threads 2
  judge_ret $? "cls_resnet_1 with FP32 model in 2 threads"
  ./cls_resnet/cls_resnet_1 --bmodel ./data/resnet50_int8_191115.bmodel --input ./data/cls.jpg --threads 2
  judge_ret $? "cls_resnet_1 with INT8 model in 2 threads"
  echo "test_resnet50 passed!"
}

function test_ssd() {
  python3 ./det_ssd/det_ssd_1.py --bmodel ./data/ssd_fp32_191115.bmodel --input ./data/det.h264
  judge_ret $? "det_ssd_0 with FP32 model"
  python3 ./det_ssd/det_ssd_1.py --bmodel ./data/ssd_int8_191115.bmodel --input ./data/det.h264
  judge_ret $? "det_ssd_0 with INT8 model"
  python3 ./det_ssd/det_ssd_2.py --bmodel ./data/ssd_fp32_191115.bmodel --input ./data/det.h264
  judge_ret $? "det_ssd_1 with FP32 model"
  python3 ./det_ssd/det_ssd_2.py --bmodel ./data/ssd_int8_191115.bmodel --input ./data/det.h264
  judge_ret $? "det_ssd_1 with INT8 model"
  echo "test_ssd passed!"
}

prepare_data
test_resnet50
test_ssd
echo "All tests passed!"
