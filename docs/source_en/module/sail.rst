SAIL
====

SAIL is the core module in the Sophon Inference.

SAIL encapsulates BMRuntime, BMDecoder, BMCV, and BMLib in BMNNSDK.
It abstracts the original functions in BMNNSDK such as "loading bmodel and driving TPU reasoning",
"Drive TPU for image processing", "Drive VPU for image and video decoding" into simpler C++ interfaces for external use.
And it re-encapsulate with pybind11, providing the most compact Python interfaces.

Currently, all classes, enumerations, and functions in the SAIL module are in the "sail" namespace.
The documentation in this module provides an in-depth look at the modules and classes in SAIL that you might use.

The core classes include:

* Handle:

The wrapper class of bm_handle_t in BMLib.
Contains device handles, contextual information, used to interact with the kernel driver information of TPU.

* Tensor:

BMLib wrapper class that encapsulates management of device memroy and synchronization with system memory.

* Engine：

The wrapper class of BMRuntime, which loads bmodel and drives the TPU for reasoning.
An Instance of Engine can load an arbitrary bmodel.
The memory corresponding to the input tensor and the output tensor is automatically managed.

* Decoder:

Decoder to decode videos by VPU and images by JPU.

* Bmcv：

It encapsulates a series of image processing functions that can drive the TPU for image processing.

