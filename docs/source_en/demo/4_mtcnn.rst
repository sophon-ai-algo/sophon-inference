Detection with MTCNN
====================

In this Demo, we use mtcnn to detect faces in images. 
The bmodel used in this demo are already converted from official yolov3, to fp32 data type.

+----+-------+---------+--------------+-----------+-------+---------+--------------+------------+-------------+
| ID | Input | Decoder | Preprocessor | Data Type | Model | Mode    | Model Number | TPU Number | Multi-Thread|
+====+=======+=========+==============+===========+=======+=========+==============+============+=============+
| 0  | image | opencv  | opencv       | fp32      | MTCNN | dynamic | 1            | 1          | N           |
+----+-------+---------+--------------+-----------+-------+---------+--------------+------------+-------------+


.. toctree::
   :glob:

   mtcnn_cases/usage
   mtcnn_cases/cpp
   mtcnn_cases/python
