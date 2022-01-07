### Object Detection Examples of SSD

Name|Input|Decoder|Batch Size|Preprocessor|Data Type|Model
-|-|-|-|-|-|-
det_ssd_0.py|image/video|opencv|1|opencv|fp32/int8|ssd_vgg
det_ssd_1.py|image/video|ffmpeg|1|bmcv|fp32/int8|ssd_vgg
det_ssd_2.py|video|ffmpeg|4|bmcv|fp32/int8|ssd_vgg

## Prepare Bmodel

* Download bmodels, which are generated from the trained model.

```shell
# download bmodel to current directory.
# download.py is in directory examples/sail/sc5_tests
python3 ./download.py ssd_fp32.bmodel
python3 ./download.py ssd_int8.bmodel
# you can also download the raw model
python3 ./download.py ssd_vgg_caffe.tgz
```

* Check the bmodel information.

```shell
bm_model.bin --info ssd_fp32.bmodel
# print info as following:
# bmodel version: B.2.2
# chip: BM1684                                              // run on sophon products with chip of BM1684
# create time: Mon Nov 25 20:01:28 2019
#
# ==========================================                // only one model inside the bmdoel, and its name is "VGG_VOC0712_SSD_300x300_deploy".
# net 0: [VGG_VOC0712_SSD_300x300_deploy]  static           // "static" means input shape is fixed and can not change. For this model, only two
# ------------                                              // input shapes are allowed, see stages.
# stage 0:
# subnet number: 2
# input: data, [1, 3, 300, 300], float32, scale: 1          // only one input tensor named "data", data type is float32, layout is NCHW.
# output: detection_out, [1, 1, 200, 7], float32, scale: 1  // only one output tensor named "detection_out", data type is float32, output shape
# ------------                                              // may change respects to detected boxes.
# stage 1:
# subnet number: 2                                          // the second allowd input shape is [4, 3, 300, 300].
# input: data, [4, 3, 300, 300], float32, scale: 1
# output: detection_out, [1, 1, 800, 7], float32, scale: 1

# print info as following:
bm_model.bin --info ssd_int8.bmodel
# bmodel version: B.2.2
# chip: BM1684                                              // run on sophon products with chip of BM1684
# create time: Mon Nov 25 20:20:01 2019
#
# ==========================================                // only one model inside the bmdoel, and its name is "VGG_VOC0712_SSD_300x300_deploy".
# net 0: [VGG_VOC0712_SSD_300x300_deploy]  static           // "static" means input shape is fixed and can not change. For this model, only two
# ------------                                              // input shapes are allowed, see stages.
# stage 0:                                                  // only one input tensor named "data", data type is int8, layout is NCHW, scale factor
# subnet number: 2                                          // is 0.847682.
# input: data, [4, 3, 300, 300], int8, scale: 0.847682      // only one output tensor named "detection_out", data type is float32, scale factor is
# output: detection_out, [1, 1, 800, 7], float32, scale: 0.117188 // 0.117188, output shape may change respects to detected boxes.
# ------------
# stage 1:
# subnet number: 2
# input: data, [1, 3, 300, 300], int8, scale: 0.847682      // the second allowd input shape is [4, 3, 300, 300].
# output: detection_out, [1, 1, 200, 7], float32, scale: 0.117188
```

## Prepare Test Data

```shell
# download data to current directory.
python3 ./download.py det.jpg
python3 ./download.py det.h264
```

## Run Examples

* A SSD example using opencv to decode and using opencv to preprocess, with batch size is 1.

```shell
# usage: python3 det_ssd_0.py [-h] --bmodel BMODEL_PATH --input INPUT_PATH
#                             [--loops LOOPS_NUMBER(default:1)]
#                             [--tpu_id TPU_ID(default:0)]
#                             [--compare COMPARE_FILE_PATH]
# run fp32 bmodel with input of image
python3 ./det_ssd_0.py --bmodel ./ssd_fp32.bmodel --input ./det.jpg
# run int8 bmodel with input of video
python3 ./det_ssd_0.py --bmodel ./ssd_int8.bmodel --input ./det.h264
```

* A SSD example using bm-ffmpeg to decode and using bmcv to preprocess, with batch size is 1.

```shell
# usage: python3 det_ssd_1.py [-h] --bmodel BMODEL_PATH --input INPUT_PATH
#                             [--loops LOOPS_NUMBER(default:1)]
#                             [--tpu_id TPU_ID(default:0)]
#                             [--compare COMPARE_FILE_PATH]
# run fp32 bmodel with input of image
python3 ./det_ssd_1.py --bmodel ./ssd_fp32.bmodel --input ./det.jpg
# run int8 bmodel with input of video
python3 ./det_ssd_1.py --bmodel ./ssd_int8.bmodel --input ./det.h264
```

* A SSD example with batch size is 4 for acceleration of int8 model, using bm-ffmpeg to decode and using bmcv to preprocess.

```shell
# usage: python3 det_ssd_2.py [-h] --bmodel BMODEL_PATH --input INPUT_PATH
#                             [--loops LOOPS_NUMBER(default:1)]
#                             [--tpu_id TPU_ID(default:0)]
#                             [--compare COMPARE_FILE_PATH]
# run int8 bmodel with input of video
python3 ./det_ssd_2.py --bmodel ./ssd_int8.bmodel --input ./det.h264
```
