# Utils

This is a tool for downloading and converting models. 

## Install Sophon

```shell
cd sophon-inference/python
# install sophon
bash install_sophon.sh

```

## Download And Convert Model 

```shell
# download model and convert to bmodel
python3 tools/download_and_convert.py ${MODEL_NAME}
```

## Model list

ID|Domain|Model|Platform|Split|Mode|CV Lib|Input
-|-|-|-|-|-|-|-
1|img_cls|googlenet|caffe|N|Static|OpenCV|image
2|img_cls|retnet50|caffe|N|Static|OpenCV|image
3|img_cls|mobilenetv1|caffe|N|Static|OpenCV|image/video
4|img_cls|vgg16|caffe|N|Static|OpenCV|image
5|img_cls|mobilenetv1_tf|tensorflow|N|Static|OpenCV|image
6|img_cls|resnext50_mx|mxnet|N|Static|OpenCV|image
7|img_cls|resnet50_pt|pytorch|N|Static|OpenCV|image
8|obj_det|yolov3|caffe|N|Static|OpenCV|image
9|obj_det|mobilenetyolov3|caffe|N|Static|OpenCV|image
10|obj_det|mobilenetssd|caffe|N|Static|OpenCV|image
11|obj_det|fasterrcnn_vgg|caffe|Y|Static|OpenCV|image
12|obj_det|fasterrcnn_resnet50_tf|tensorflow|Y|Static|OpenCV|image
13|face_det|mtcnn|caffe|N|Dynamic|OpenCV|image
14|face_det|ssh|caffe|N|Static|OpenCV|image
15|sem_seg|deeplabv3_mobilenetv2_tf|tensorflow|Y|Static|OpenCV|