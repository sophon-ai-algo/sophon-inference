# AutoSplit (of AutoDeploy)

AutoDeploy is an open source library for automatically deploying models on Sophon TPU (https://sophon.ai).<br>
When a model can't totally be deployed on Sophon TPU, it will be splitted into several parts by AutoDeploy. After splitting, some parts can be compiled to bmodel.
AutoDeploy can infer the splitted/compiled sub-models on both cpu and tpu, while won't change the final result.<br>

For details, please refer to Sophon Inference Documentation.<br>

![image](../../docs/autodeploy.png)

## Features
* Use a general algorithm to split models from different frameworks;
* Support mxnet/tensorflow currnetly, caffe/pytorch for future;
* Integrate BMCompilers (bmnetc/t/m/p) and SAIL (Sophon Artifical Intelligence Library) to make a pipeline;


## Prerequisites

* **python2/python3, numpy** Required
* **sail** Required
* **bmnetc/t/m/p** Optional, depends on which framework you are using
* **opencv-python** Optional

**Tips**: Make sure your sail (for python api) has compiled correctly and set your PYTHONPATH to where you place your sail.so in.


## Demos

### Mxnet

* AutoDeploy Mxnet models.
```shell
# Run python3 tests/auto_deploy/test_mxnet.py --help
usage: test_mxnet.py [-h] [--save_dir SAVE_DIR] --json_path JSON_PATH
                     --params_path PARAMS_PATH --model_type MODEL_TYPE --size
                     SIZE [--mode MODE]
```

### Tensorflow

* AutoDeploy Tensorflow models.

```shell
# Run python3 tests/auto_deploy/test_tensorflow.py --help
usage: test_tensorflow.py [-h] --save_dir SAVE_DIR --model_path MODEL_PATH
                          --image_path IMAGE_PATH --model_type MODEL_TYPE
                          --width WIDTH --height HEIGHT --dynamic DYNAMIC
                          [--mode MODE]

```
## Tested models
Testing Environment:<br>
* CPU: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz;
* RAM: 16G + 16G swap;
* TPU: Sophon SC3;
* Tensorflow: 1.13.1
* Mxnet: 1.4.0
* NNToolChain: 1.1;
* SAIL: 2.0.2.

### mxnet

GluonCV Models | Time/ms<br>(pure cpu) | Time/ms<br>(cpu and tpu) | Number<br>(splitted)  | Number<br>(tpu) | Flops ratio of Supports | Params ratio of Supports
:-----:|:-----:|:-----:|:----------:|:----:|:-----:|:--------:|
ssd_512_mobilenet1.0_voc| | | 2 | 1 |
ssd_512_resnet50_v1_voc| | | 2 | 1 |
ssd_300_vgg16_atrous_voc| | | 2 | 1 |
yolo3_darknet53_voc| | | 2 | 1 |
yolo3_mobilenet1.0_voc| | | 2 | 1 |
faster_rcnn_resnet50_v1b_coco| | | 4 | 2 |
faster_rcnn_resnet101_v1d_coco| | | 4 | 2 |
faster_rcnn_fpn_resnet50_v1b_coco| | | 4 | 2 |
faster_rcnn_fpn_resnet101_v1d_coco| | | 4 | 2 |
deeplab_resnet50_ade| | | 1 | 1 |
psp_resnet101_citys| | | 1 | 1 |
simple_pose_resnet50_v1b| | | 2 | 1 |
simple_pose_resnet50_v1d| | | 2 | 1 |

### tensorflow

**Tips**: input shape is (1,500,500,3), tpu models are compiled on server with 256G RAM.

Models | Time/ms<br>(pure cpu) | Time/ms<br>(cpu and tpu) | Number<br>(splitted)  | Number<br>(tpu) | Flops ratio of Supports | Params ratio of Supports
:-----:|:-----:|:-----:|:----------:|:----:|:-----:|:--------:|
[ssd_mobilenet_v1](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)| 25 | 12 | 3 | 1 |
[ssd_mobilenet_v2](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)| 33 | 14 | 3 | 1 |
[ssdlite_mobilenet_v2](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)| 24 | 10 | 3 | 1 |
[ssd_inception_v2](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)| 53 | 19 | 3 | 1 |
[ssd_resnet50_v1](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)| 825 | 163 | 3 | 1 |
[rfcn_resnet101](http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_2018_01_28.tar.gz)| 845 | 254 | 3 | 1 |
[faster_rcnn_inception_resnet_v2](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz)| 8115 | 3276 | 5 | 2 |
[faster_rcnn_nasnet](http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz)| 22077 | | 5 | 2 |
[faster_rcnn_resnet50](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz)| 956 | 256 | 5 | 2 |
[faster_rcnn_resnet101](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz)| 1192 | 290 | 5 | 2 |
[mask_rcnn_inception_v2](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz)| 1046 | 280 | 7 | 3 |
[mask_rcnn_resnet50](http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet50_atrous_coco_2018_01_28.tar.gz)| 8498 | 2294 | 7 | 3 |
[mask_rcnn_resnet101](http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet101_atrous_coco_2018_01_28.tar.gz)| 10285 | 5103 | 5 | 2 |
[mask_rcnn_inception_resnet_v2](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz)| 12784 | 5150 | 7 | 3 |
[deeplab_v3_mobilenet_v2](http://download.tensorflow.org/models/deeplabv3_mnv2_dm05_pascal_trainaug_2018_10_01.tar.gz)| 109 | 254 | 3 | 1 |
[deeplab_v3_xception](http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz)| 2566 | | 3 | 1 |

## Contributing

TODO

## Authors

* Hong Liu      - Initial work
* Lian He       - Initial work
* Zhenpeng Xiao - Initial work

See also the list of contributors who participated in this project.

## License

This project is licensed under the Apache License, Version 2.0 - see the LICENSE.md file for details

## Acknowledgments

TODO
