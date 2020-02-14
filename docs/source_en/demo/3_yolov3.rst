Detection with Yolov3
=====================

In this Demo, we use yolov3 to detect objects in multiple videos. 
The bmodels used in this demo are already converted from official yolov3, to both fp32 and int8 data type.

The differences between the two cases are the Decoder and Preprocessor.

+----+-------------+-----------+--------------+-----------+--------+--------+--------------+------------+-------------+
| ID | Input       | Decoder   | Preprocessor | Data Type | Model  | Mode   | Model Number | Batch Size | Multi-Thread|
+====+=============+===========+==============+===========+========+========+==============+============+=============+
| 0  | multi-video | opencv    | opencv       | fp32 int8 | yolov3 | static | 1            | 1          | Y           |
+----+-------------+-----------+--------------+-----------+--------+--------+--------------+------------+-------------+
| 1  | multi-video | bm-ffmpeg | bmcv         | fp32 int8 | yolov3 | static | 1            | 1          | Y           |
+----+-------------+-----------+--------------+-----------+--------+--------+--------------+------------+-------------+


.. toctree::
   :glob:

   yolov3_cases/usage
   yolov3_cases/cpp
   yolov3_cases/python




