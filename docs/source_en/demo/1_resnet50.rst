Classification with Resnet
==========================
In this Demo, we use resnet-50 to classify images.
The bmodels used in this demo are already converted from official caffe resnet-50, to both fp32 and int8 data type.
We implemented four cases, they are all using public released opencv for image decoding and preprocessing.
The input tensor shape of each bmodel is valid, which is the common used 1*3*224*224.
The differences among the four cases are the "Model Number", "TPU Number" and "Multi-Thread".

+---+-------+---------+--------------+-----------+-----------+--------+--------------+------------+-------------+
|ID | Input | Decoder | Preprocessor | Data Type |  Model    | Mode   | Model Number | TPU Number | Multi-Thread|
+===+=======+=========+==============+===========+===========+========+==============+============+=============+
| 0 | image | opencv  | opencv       | fp32 int8 | resnet-50 | static | 1            | 1          | N           |
+---+-------+---------+--------------+-----------+-----------+--------+--------------+------------+-------------+
| 1 | image | opencv  | opencv       | fp32 int8 | resnet-50 | static | 1            | 1          | Y           |
+---+-------+---------+--------------+-----------+-----------+--------+--------------+------------+-------------+
| 2 | image | opencv  | opencv       | fp32 int8 | resnet-50 | static | 1            | 2          | Y           |
+---+-------+---------+--------------+-----------+-----------+--------+--------------+------------+-------------+
| 3 | image | opencv  | opencv       | fp32 int8 | resnet-50 | static | 2            | 1          | Y           |
+---+-------+---------+--------------+-----------+-----------+--------+--------------+------------+-------------+


.. toctree::
   :glob:

   resnet_cases/usage
   resnet_cases/cpp
   resnet_cases/python
