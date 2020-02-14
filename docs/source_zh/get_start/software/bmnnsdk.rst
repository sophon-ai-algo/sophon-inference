BMNNSDK
_______

.. image:: ../../../images/sophon_inference.png



BMNNSDK 是比特大陆自研的软件包。
上图中提及的所有软件模块都包含在 BMNNSDK 中，
包括了 Quantization & Calibration Tool, BMCompiler, BMDriver, BMLib, BMDecoder, BMCV, BMRuntime, Sophon Inference.

**Quantization & Calibration Tool**
:该模块可以将 FP32 精度的模型转换成 INT8 精度的模型

在线文档: https://sophon-ai-algo.github.io/calibration_tools-doc/

**BMCompiler**
:目前包括了五种模型编译工具。其中，
bmnett 可以将 tensorflow 下训练生成的模型编译成 fp32_bmodel。
bmnetp 可以将 pytorch 下训练生成的模型编译成 fp32_bmodel。
bmnetm 可以将 mxnet 下训练生成的模型编译成 fp32_bmodel。
bmnetc 可以将 caffe 下训练生成的模型编译成 fp32_bmodel。
bmnetu 可以将 Quantization & Calibration Tool 下生成的 int8_umodel 编译成 int8_bmodel。

在线文档: https://sophon-ai-algo.github.io/bmnnsdk-doc/

**BMDriver**
:是算丰 TPU 的驱动程序，将会通过 insmod 的方式安装到系统内核中。

**BMLib**
:提供了一些基础接口，用来控制 TPU 与主机的内存交互。

在线文档: https://sophon-ai-algo.github.io/bmlib_1684-doc/

**BMDecoder**
:提供了一些应用接口，用来驱动 TPU 上的硬件单元进行图像和视频的编解码。

在线文档: https://sophon-ai-algo.github.io/bm_multimedia/

**BMCV**
:提供了一些应用接口，用来驱动 TPU 上的硬件单元进行张量计算和图像处理。

在线文档: https://sophon-ai-algo.github.io/bmcv_1684-doc/

**BMRuntime**
:提供了一些应用接口，用来驱动 TPU 加载 bmodel 并进行模型推理。

在线文档: https://sophon-ai-algo.github.io/bmnnsdk-doc/

**SAIL**
:提供了一些高级接口，主要是对 BMRuntime、BMCV、BMdecoder 等运行时模块的封装。
