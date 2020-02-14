Model deployment
________________

.. image:: ../../../images/sophon_inference.png


Model deployment includes two steps: model offline compilation and online reasoning.
Softwares shown in above picture are all included in BMNNSDK.

**a).Offline Compilation**

This process corresponds to the blue part in the above figure.
Suppose the user has obtained a trained FP32 precision deep learning model,
then the user can directly compile the model to bmodel using BMCompiler.
The bmodel generated in this way can be reasoned using the FP32 computing units on the TPU.
The BMCompiler is a general term here. It contains four front-end tools that support four deep learning frameworks.
They are bmnetc(caffe), bmnett(tensorflow), bmnetm(mxnet), bmnetp(pytorch).

If the user wants to use the INT8 computing units on the TPU for reasoning,
Quantization & Calibration tool can be used to quantify the original FP32 precision model to an INT8 precision model.
Finally, user can Compile the generated int8_umodel to bmodel using the bmnetu tool in BMCompiler.

The generation of bmodel does not depend on TPU.
Users only need to install the corresponding BBMCompiler and Quantization & Calibration tools as needed to complete this step.
In theory, a deep learning model, as long as the bmodel can be finally generated, the bmodel can be deployed on Sophon TPUs.

**b).Online Reasoning**

This process corresponds to the process from input to output in the red part of the above figure.
Users can do images/video decoding, tensor processing and calculations, and bmodel operations based on the SAIL module in Sophon Inference.

This process needs to be performed in the environment where the TPU and driver are installed.

