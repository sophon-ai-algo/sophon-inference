Crash Course
============

In this course, we will help you deploy a tensorflow frozen model of mobilenet on Sophon SC5.
Before starting the course, you should prepare a personal computer with a Sophon SC5 being plugined in its PCIE slot,
and the BMNNSDK, Bitmain Neural Network Software Development Kit.

There are three steps in this course.
First, we should install the driver, bmnett and the python module of sophon-inference. The three modules are all contained in BMNNSDK.
Then, we are going to convert a mobilent which is trained from tensorflow to a bmodel using bmnett.
Finally, we will deploy the converted bmodel on Sophon SC5 by sophon-inference.

.. toctree::
   :glob:

   courses/install
   courses/convert
   courses/deploy



