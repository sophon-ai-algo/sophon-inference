Preface
=======

Demo Brief
__________

+--------------+-------------+-----------+--------------+-----------+--------------+---------+--------------+------------+------------+-------------+
| Binary       | Input       | Decoder   | Preprocessor | Data Type |  Model       | Mode    | Model Number | TPU Number | Batch Size | Multi-Thread|
+==============+=============+===========+==============+===========+==============+=========+==============+============+============+=============+
| cls-resnet-0 | image       | opencv    | opencv       | fp32 int8 | resnet-50    | static  | 1            | 1          | 1          | N           |
+--------------+-------------+-----------+--------------+-----------+--------------+---------+--------------+------------+------------+-------------+
| cls-resnet-1 | image       | opencv    | opencv       | fp32 int8 | resnet-50    | static  | 1            | 1          | 1          | Y           |
+--------------+-------------+-----------+--------------+-----------+--------------+---------+--------------+------------+------------+-------------+
| cls-resnet-2 | image       | opencv    | opencv       | fp32 int8 | resnet-50    | static  | 1            | 2          | 1          | Y           |
+--------------+-------------+-----------+--------------+-----------+--------------+---------+--------------+------------+------------+-------------+
| cls-resnet-3 | image       | opencv    | opencv       | fp32 int8 | resnet-50    | static  | 2            | 1          | 1          | Y           |
+--------------+-------------+-----------+--------------+-----------+--------------+---------+--------------+------------+------------+-------------+
| det-ssd-0    | video image | opencv    | opencv       | fp32 int8 | ssd300-vgg16 | static  | 1            | 1          | 1          | N           |
+--------------+-------------+-----------+--------------+-----------+--------------+---------+--------------+------------+------------+-------------+
| det-ssd_1    | video image | bm-ffmpeg | bmcv         | fp32 int8 | ssd300-vgg16 | static  | 1            | 1          | 1          | N           |
+--------------+-------------+-----------+--------------+-----------+--------------+---------+--------------+------------+------------+-------------+
| det-ssd-2    | video image | bm-ffmpeg | bmcv         | fp32 int8 | ssd300-vgg16 | static  | 1            | 1          | 4          | N           |
+--------------+-------------+-----------+--------------+-----------+--------------+---------+--------------+------------+------------+-------------+
| det-ssd-3    | video image | bm-opencv | bm-opencv    | fp32 int8 | ssd300-vgg16 | static  | 1            | 1          | 1          | N           |
+--------------+-------------+-----------+--------------+-----------+--------------+---------+--------------+------------+------------+-------------+
| det-ssd-4    | video image | bm-opencv | bmcv         | fp32 int8 | ssd300-vgg16 | static  | 1            | 1          | 1          | N           |
+--------------+-------------+-----------+--------------+-----------+--------------+---------+--------------+------------+------------+-------------+
| det-yolov3-0 | multi-video | opencv    | opencv       | fp32 int8 | yolov3       | static  | 1            | 1          | 1          | Y           |
+--------------+-------------+-----------+--------------+-----------+--------------+---------+--------------+------------+------------+-------------+
| det-yolov3-1 | multi-video | bm-ffmpeg | bmcv         | fp32 int8 | yolov3       | static  | 1            | 1          | 1          | Y           |
+--------------+-------------+-----------+--------------+-----------+--------------+---------+--------------+------------+------------+-------------+
| det-mtcnn    | image       | opencv    | opencv       | fp32      | MTCNN        | dynamic | 1            | 1          | 1          | N           |
+--------------+-------------+-----------+--------------+-----------+--------------+---------+--------------+------------+------------+-------------+

As the above table shown, 
we prepared several demos to let you get familiar with Sophon-Inference more quickly.
Both c++ and python are supported.
For each demo, we implemented different cases to adapt to multiple applications.
we have four kinds of demos by now:
    
    **cls_resnet(classification with resnet50)**
    
    **det_ssd(detection with ssd300-vgg16)**
    
    **det_yolov3(detection with yolov3)**
    
    **det_mtcnn(detection with mtcnn)**


The meanings of properties of the cases are explained as follows:

    **Binary:** the name of binary file(c++) or script(python) of the case.

    **Input:** input date type, image or video.

    **Decoder:** libs for decoding the input. 
    "opencv" is the public release version of opencv which using CPU for decoding.
    "bm-opencv" and "ffmpeg" are the bitmain versions of opencv and ffmpeg which using VPU for decoding.

    **Preprocessor:** libs for processing image or tensor.
    "opencv" is the public release version of opencv which using CPU for caculating.
    "bm-opencv" and "bmcv" are the bitmain versions for processing image or tensor.

    **Data Type:** data type of the bmodel to be used, fp32 or int8.

    **Model:** the name of deep learning model used in this case.

    **Mode:** two modes, static mean input tensor shapes of bmodel are unchanged, 
    while dynamic means input tensor shapes of bmodel can be changed.

    **Model Number:** how many models are supported concurrently in this case.

    **TPU Number:** how many TPUs are supported at the same time.

    **Batch Size:** the batchsize of the bmodel we used.

    **Multi Thread:**: how many threads are supported at the same time.


Return Values
_____________
We also defined a return value list for each case, for reference.

+-----+------------------+
| ret | meaning          |
+=====+==================+
|  0  | normal           |
+-----+------------------+
|  1  | comparing failed |
+-----+------------------+
|  2  | invalid tpu id   |
+-----+------------------+

