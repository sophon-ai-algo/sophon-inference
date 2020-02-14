### Face Detection Example for Dynamic Model

An example of inference for dynamic model, whose input shapes may change.
There are 3 graphs in the MTCNN model: PNet, RNet and ONet. Input height and width may change for Pnet while input batch_szie may change for RNet and Onet.

Name|Input|Decoder|Preprocessor|Data Type|Model|Mode
-|-|-|-|-|-|-
det_mtcnn|image|opencv|opencv|fp32|mtcnn|dynamic

## Prepare Bmodel

* Download bmodels, which are generated from the trained model.

```shell
# download bmodel to current directory.
# download.py is in directory of source repo: tools/
python3 ./download.py mtcnn_fp32.bmodel
# you can also download the raw model
python3 ./download.py mtcnn_caffe.tgz
```

* Check the bmodel information.

```shell
# bm_model.bin --info mtcnn_fp32.bmodel
# print info as following:
# bmodel version: B.2.2
# chip: BM1684                                                  // run on sophon products with chip of BM1684
# create time: Tue Nov 26 13:07:44 2019
#
# ==========================================                    // there are 3 submodels, or 3 stages in MTCNN
# net 0: [PNet]  dynamic                                        // first stage named PNet, "dynamic" means input shapes may change.
# ------------
# stage 0:                                                      // only one input tensor named "data", data type is float32, layout is NCHW.
# input: data, [max:1, 3, max:576, max:324], float32, scale: 1  // H and W may change but should satisfy: 0 < H <= 576, 0 < W <= 324.
# output: conv4-2, [1, 4, 283, 157], float32, scale: 1          // there are 2 output tensors: "conv4-2" and "prob1", data types are float32.
# output: prob1, [1, 2, 283, 157], float32, scale: 1            // output shapes here are respect to the max input shape for dynamic models.
# ==========================================
# net 1: [RNet]  dynamic                                        // second stage named RNet, "dynamic" means input shapes may change.
# ------------
# stage 0:                                                      // only one input tensor named "data", data type is float32, layout is NCHW.
# input: data, [max:32, 3, 24, 24], float32, scale: 1           // N may change but should satisfy: 0 < N <= 32.
# output: conv5-2, [32, 4], float32, scale: 1                   // there are 2 output tensors: "conv5-2" and "prob1", data types are float32.
# output: prob1, [32, 2], float32, scale: 1                     // output shapes here are respect to the max input shape for dynamic models.
# ==========================================
# net 2: [ONet]  dynamic                                        // third stage named ONet, "dynamic" means input shapes may change.
# ------------
# stage 0:                                                      // only one input tensor named "data", data type is float32, layout is NCHW.
# input: data, [max:32, 3, 48, 48], float32, scale: 1           // N may change but should satisfy: 0 < N <= 32.
# output: conv6-2, [32, 4], float32, scale: 1                   // there are 3 output tensors: "conv6-2", "conv6-3" and "prob1".
# output: conv6-3, [32, 10], float32, scale: 1                  // data types are float32. output shapes here are respect to the max input
# output: prob1, [32, 2], float32, scale: 1                     // max input shape for dynamic models.
```

## Prepare Test Data

```shell
# download data to current directory.
python3 ./download.py face.jpg
```

## Run Examples

```shell
./det_mtcnn --bmodel ./mtcnn_fp32.bmodel --input ./face.jpg
```
