## Object Detection Example of YOLOv3

Run yolov3 object detection with:

Name|Input|Decoder|Preprocessor|Data Type|Model|Mode|Batch Size|Multi-Thread
-|-|-|-|-|-|-|-|-
det_yolov3.py|video|opencv|opencv|fp32/int8|yolov3|static|1|N

## Prepare Bmodel

* Download bmodels, which are generated from the trained model.

```shell
# download bmodel to current directory.
# download.py is in directory of source repo: tools/
```shell
# download data to current directory.
python3 ./download.py yolov3_fp32.bmodel
python3 ./download.py yolov3_int8.bmodel
# you can also download the raw model
python3 ./download.py yolov3_caffe.tgz
```

* Check the bmodel information.

```shell
bm_model.bin --info yolov3_fp32.bmodel
# print info as following:
# bmodel version: B.2.2
# chip: BM1684                                              // run on sophon products with chip of BM1684
# create time: Mon Nov 18 13:28:16 2019
#
# ==========================================                // only one model inside the bmdoel, and its name is "yolo3_dark53_coco"
# net 0: [yolo3_dark53_coco]  static                        // "static" means input shape is fixed and can not change.
# ------------
# stage 0:
# subnet number: 2                                          // only one input tensor named "data", data type is float32, layout is NCHW.
# input: data, [1, 3, 416, 416], float32, scale: 1          // only one output tensor named "detection_out", data type is float32, output
# output: detection_out, [1, 1, 100, 7], float32, scale: 1  // shape may change respects to detected boxes.

bm_model.bin --info yolov3_int8.bmodel
# print info as following:
# bmodel version: B.2.2
# chip: BM1684                                              // run on sophon products with chip of BM1684
# create time: Wed Nov  6 14:21:23 2019
#
# ==========================================                // only one model inside the bmdoel, and its name is "yolo3_dark53_coco"
# net 0: [yolo3_dark53_coco]  static                        // "static" means input shape is fixed and can not change.
# ------------
# stage 0:                                                  // only one input tensor named "data", data type is int8, layout is NCHW,
# subnet number: 2                                          // scale factor is 127.986. only one output tensor named "detection_out",
# input: data, [1, 3, 416, 416], int8, scale: 127.986       // data type is float32, scale factor is 0.0078125.
# output: detection_out, [1, 1, 100, 7], float32, scale: 0.0078125  // output shape may change respects to detected boxes.
```

## Prepare Test Data

```shell
# download data to current directory.
python3 ./download.py det.h264
```

## Run Examples

```shell
# usage: python3 det_yolov3.py [-h] --bmodel BMODEL_PATH --input INPUT_PATH
#                              [--loops LOOPS_NUMBER(default:1)]
#                              [--tpu_id TPU_ID(default:0)]
#                              [--compare COMPARE_FILE_PATH]
# run fp32 bmodel with input of image
python3 ./det_yolov3.py --bmodel ./yolov3_fp32.bmodel --input ./det.h264
# run int8 bmodel with input of video
python3 ./det_yolov3.py --bmodel ./yolov3_int8.bmodel --input ./det.h264
```
