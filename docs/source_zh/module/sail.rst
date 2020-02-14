SAIL
====

SAIL 是 Sophon Inference 中的核心模块，

SAIL 对 BMNNSDK 中的 BMLib、BMDecoder、BMCV、BMRuntime 进行了封装，
将 BMNNSDK 中原有的 “加载 bmodel 并驱动 TPU 推理”、
“驱动 TPU 做图像处理”、“驱动 VPU 做图像和视频解码”
等功能抽象成更为简单的 C++ 接口对外提供；
并且支持使用 pybind11 再次封装，提供简洁的 python 接口。

目前，SAIL 模块中所有的类、枚举、函数都在 “sail” 名字空间下，
本单元中的文档将深入介绍您可能用到的 SAIL 中的模块和类。
核心的类包括：

* Handle：

BMNNSDK 中 BMLib 的 bm_handle_t 的包装类，设备句柄，上下文信息，用来和内核驱动交互信息。

* Tensor：

BMNNSDK 中 BMLib 的包装类，封装了对 device memroy 的管理以及与 system memory 的同步。

* Engine：

BMNNSDK 中 BMRuntime 的包装类，可以加载 bmodel 并驱动 TPU 进行推理。
一个 Engine 实例可以加载一个任意的 bmodel，
自动地管理输入张量与输出张量对应的内存。

* Decoder

使用VPU解码视频，JPU解码图像，均为硬件解码。

* Bmcv：

BMNNSDK 中 BMCV 的包装类，封装了一系列的图像处理函数，可以驱动 TPU 进行图像处理。

