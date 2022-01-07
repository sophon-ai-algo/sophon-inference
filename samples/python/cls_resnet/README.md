### Image Classification Example

Name|Input|Decoder|Preprocessor|Data Type|Model|Mode|Model Number|TPU Number|Multi-Thread
-|-|-|-|-|-|-|-|-|-
cls_resnet_0|image|opencv|opencv|fp32/int8|resnet-50|static|1|1|N
cls_resnet_1|image|opencv|opencv|fp32/int8|resnet-50|static|1|1|Y
cls_resnet_2|image|opencv|opencv|fp32/int8|resnet-50|static|2|1|Y
cls_resnet_3|image|opencv|opencv|fp32/int8|resnet-50|static|1|2|Y

## Prepare Bmodel

* Download bmodels, which are generated from the trained model.

```shell
# download bmodel to current directory.
# download.py is in directory examples/sail/sc5_tests
python3 ./download.py resnet50_fp32.bmodel
python3 ./download.py resnet50_int8.bmodel
# you can also download the raw model
python3 ./download.py resnet50_caffe.tgz
```

* Check the bmodel information.

```shell
bm_model.bin --info resnet50_fp32.bmodel
# print info as following:
# bmodel version: B.2.2
# chip: BM1684                                         // run on sophon products with chip of BM1684
# create time: Sat Nov 23 14:37:37 2019
#
# ==========================================           // only one model inside the bmdoel, and its name is "ResNet-50_fp32";
# net 0: [ResNet-50_fp32]  static                      // "static" means input shape is fixed and can not change.
# ------------                                         // only one input tensor named "data", data type is float32, shape is [1, 3, 224, 224]
# stage 0:                                             // with layout of NCHW, scale factor is 1.0 for data type of float32.
# input: data, [1, 3, 224, 224], float32, scale: 1     // only one output tensor named "fc1000", data type is float32, shape is [1, 1000], scale
# output: fc1000, [1, 1000], float32, scale: 1         // factor is 1.0 for data type of float32.

bm_model.bin --info resnet50_int8.bmodel
# print info as following:
# bmodel version: B.2.2
# chip: BM1684                                         // run on sophon products with chip of BM1684
# create time: Sat Nov 23 14:38:50 2019
#
# ==========================================           // only one model inside the bmdoel, and its name is "ResNet-50_int8";
# net 0: [ResNet-50_int8]  static                      // "static" means input shape is fixed and can not change.
# ------------                                         // only one input tensor named "data", data type is int8, shape is [1, 3, 224, 224]
# stage 0:                                             // with layout of NCHW, scale factor is 0.84734.
# input: data, [1, 3, 224, 224], int8, scale: 0.84734  // only one output tensor named "fc1000", data type is int8, shape is [1, 1000], scale
# output: fc1000, [1, 1000], int8, scale: 0.324477     // factor is 0.324477.
```

## Prepare Test Data

```shell
# download data to current directory.
python3 ./download.py cls.jpg
```

## Run Examples

* The simplest case for inference of one model on one TPU.

```shell
# usage: python3 cls_resnet_0.py [-h] --bmodel BMODEL_PATH --input INPUT_PATH
#                                [--loops LOOPS_NUMBER(default:1)]
#                                [--tpu_id TPU_ID(default:0)]
#                                [--compare COMPARE_FILE_PATH]
# run fp32 bmodel
python3 ./cls_resnet_0.py --bmodel ./resnet50_fp32.bmodel --input ./cls.jpg
# run int8 bmodel
python3 ./cls_resnet_0.py --bmodel ./resnet50_int8.bmodel --input ./cls.jpg
```

* Inference of one model by multiple threads on one TPU.

```shell
# usage: python3 cls_resnet_1.py [-h] --bmodel BMODEL_PATH --input INPUT_PATH
#                                --threads THREADS_NUMBER
#                                [--loops LOOPS_NUMBER(default:1)]
#                                [--tpu_id TPU_ID(default:0)]
#                                [--compare COMPARE_FILE_PATH]
# run fp32 bmodel
python3 ./cls_resnet_1.py --bmodel ./resnet50_fp32.bmodel --input ./cls.jpg --threads 2
# run int8 bmodel
python3 ./cls_resnet_1.py --bmodel ./resnet50_int8.bmodel --input ./cls.jpg --threads 2
```

* Inference of two models in two threads on one TPU.

```shell
# usage: python3 cls_resnet_2.py [-h] --bmodel BMODEL_PATH [--bmodel BMODEL_PATH]...
#                                --input INPUT_PATH
#                                [--loops LOOPS_NUMBER(default:1)]
#                                [--tpu_id TPU_ID(default:0)]
#                                [--compare COMPARE_FILE_PATH]
# run fp32 bmodel and int8 bmodel in two threads
python3 ./cls_resnet_2.py --bmodel ./resnet50_fp32.bmodel --bmodel ./resnet50_int8.bmodel --input ./cls.jpg
```

* Inference of one model on multiple TPUs.

```shell
# usage: python3 cls_resnet_3.py [-h] --bmodel BMODEL_PATH --input INPUT_PATH
#                                --tpu_id TPU_ID [--tpu_id TPU_ID]...
#                                [--loops LOOPS_NUMBER(default:1)]
#                                [--compare COMPARE_FILE_PATH]
# run fp32 bmodel
python3 ./cls_resnet_3.py --bmodel ./resnet50_fp32.bmodel --input ./cls.jpg --tpu_id 0 --tpu_id 1
# run int8 bmodel
python3 ./cls_resnet_3.py --bmodel ./resnet50_int8.bmodel --input ./cls.jpg --tpu_id 0 --tpu_id 1
```
