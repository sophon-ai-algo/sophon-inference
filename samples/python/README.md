## Summary

### Image Classification

ID|Input|Decoder|Preprocessor|Data Type|Model|Mode|Model Number|TPU Number|Multi-Thread
-|-|-|-|-|-|-|-|-|-
1|image|opencv|opencv|fp32/int8|resnet-50|static|1|1|N
2|image|opencv|opencv|fp32/int8|resnet-50|static|1|1|Y
3|image|opencv|opencv|fp32/int8|resnet-50|static|1|2|Y
4|image|opencv|opencv|fp32/int8|resnet-50|static|2|1|Y

[cls_resnet](cls_resnet/README.md)

### Object Detection

ID|Input|Decoder|Preprocessor|Data Type|Model|Mode|Batch Size|Multi-Thread
-|-|-|-|-|-|-|-|-
1|video/image|opencv|opencv|fp32/int8|ssd_vgg|static|1|N
2|video/image|bm-ffmpeg|bmcv|fp32/int8|ssd_vgg|static|1|N
3|video|bm-ffmpeg|bmcv|fp32/int8|ssd_vgg|static|4|N
4|multi-video|opencv|opencv|fp32/int8|yolov3|static|1|Y
5|multi-video|bm-ffmpeg|bmcv|fp32/int8|yolov3|static|1|Y

[det_ssd](det_ssd/README.md)
[det_yolov3](det_yolov3/README.md)

### Face Detection

ID|Input|Decoder|Preprocessor|Data Type|Model|Mode
-|-|-|-|-|-|-
1|image|opencv|opencv|fp32|mtcnn|dynamic

[det_mtcnn](det_mtcnn/README.md)
