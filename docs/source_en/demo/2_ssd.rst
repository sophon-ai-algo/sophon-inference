Detection with SSD
==================

In this Demo, we use ssd300-vgg16 to detect objects in both images and videos. 
The bmodels used in this demo are already converted from official ssd300-vgg16, to both fp32 and int8 data type.

The main differences among these cases are decoder and preprocessor we choosen, except case 2, 
which is just the 4-N mode (batch_size is the multiples of 4) of case 1.

+----+-------------+-----------+--------------+-----------+--------------+--------+--------------+------------+-------------+
| ID | Input       | Decoder   | Preprocessor | Data Type | Model        | Mode   | Model Number | Batch Size | Multi-Thread|
+====+=============+===========+==============+===========+==============+========+==============+============+=============+
| 0  | video image | opencv    | opencv       | fp32 int8 | ssd300-vgg16 | static |          1   |        1   |           N |
+----+-------------+-----------+--------------+-----------+--------------+--------+--------------+------------+-------------+
| 1  | video image | bm-ffmpeg | bmcv         | fp32 int8 | ssd300-vgg16 | static |          1   |        1   |           N |
+----+-------------+-----------+--------------+-----------+--------------+--------+--------------+------------+-------------+
| 2  | video image | bm-ffmpeg | bmcv         | fp32 int8 | ssd300-vgg16 | static |          1   |        4   |           N |
+----+-------------+-----------+--------------+-----------+--------------+--------+--------------+------------+-------------+
| 3  | video image | bm-opencv | bm-opencv    | fp32 int8 | ssd300-vgg16 | static |          1   |        1   |           N |
+----+-------------+-----------+--------------+-----------+--------------+--------+--------------+------------+-------------+
| 4  | video image | bm-opencv | bmcv         | fp32 int8 | ssd300-vgg16 | static |          1   |        1   |           N |
+----+-------------+-----------+--------------+-----------+--------------+--------+--------------+------------+-------------+

.. toctree::
   :glob:

   ssd_cases/usage
   ssd_cases/cpp
   ssd_cases/python



