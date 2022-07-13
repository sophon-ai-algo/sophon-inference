<pre>
   _____             __                      ____      ____
  / ___/____  ____  / /_  ____  ____        /  _/___  / __/__  ________  ____  ________
  \__ \/ __ \/ __ \/ __ \/ __ \/ __ \______ / // __ \/ /_/ _ \/ ___/ _ \/ __ \/ ___/ _ \
 ___/ / /_/ / /_/ / / / / /_/ / / / /_____// // / / / __/  __/ /  /  __/ / / / /__/  __/
/____/\____/ .___/_/ /_/\____/_/ /_/     /___/_/ /_/_/  \___/_/   \___/_/ /_/\___/\___/
          /_/
</pre>

# Sophon-Inference

Guide to deploying deep-learning inference networks and deep vision primitives on Sophon TPU.
![image](docs/images/sophon_inference.png)

### SAIL: Sophon Artificial Intelligent Library for online deployment.
* It's a wrapper of bmruntime, bmdecoder, bmcv, bmlib;
* Provide both C++ and Python APIs;
* Automatically manage memory of tensors;

```
## Prerequisites and Compilation

- **SophonSDK3**                        Required
- **CMake**                           Required
- **OpenCV3(at least 3.4.6)**         Required for C++ samples
- **Python3**                         Optional for python samples
- **Sphinx**                          Optional for documents

```shell
# install sophonsdk3
cd sophonsdk3/scripts
# extract libs adaptable to the OS
./install_lib.sh nntc
# remove old driver and install the new one, 'pcie' for x86_64 and 'arm_pcie' for aarch64
sudo ./remove_driver_pcie.sh      # for x86_64
sudo ./uninstall_driver_pcie.sh   # for x86_64
# configure the environment. 'pcie' for x86_64 and 'arm_pcie' for aarch64
source ./envsetup_pcie.sh         # for x86_64

# compilation
1. change python configuration from compile.sh.
2. ./compile.sh

```

## Samples

### Image Classification

ID|Input|Decoder|Preprocessor|Data Type|Model|Mode|Model Number|TPU Number|Multi-Thread
-|-|-|-|-|-|-|-|-|-
1|image|opencv|opencv|fp32/int8|resnet-50|static|1|1|N
2|image|opencv|opencv|fp32/int8|resnet-50|static|1|1|Y
3|image|opencv|opencv|fp32/int8|resnet-50|static|1|2|Y
4|image|opencv|opencv|fp32/int8|resnet-50|static|2|1|Y

### Object Detection

ID|Input|Decoder|Preprocessor|Data Type|Model|Mode|Batch Size|Multi-Thread
-|-|-|-|-|-|-|-|-
1|video/image|opencv|opencv|fp32/int8|ssd_vgg|static|1|N
2|video/image|bm-ffmpeg|bmcv|fp32/int8|ssd_vgg|static|1|N
3|video|bm-ffmpeg|bmcv|fp32/int8|ssd_vgg|static|4|N
4|video/image|bm-opencv|bm-opencv|fp32/int8|ssd_vgg|static|1|N
5|video/image|bm-opencv|bmcv|fp32/int8|ssd_vgg|static|1|N
6|video|opencv|opencv|fp32/int8|yolov3|static|1|Y
7|video|bm-ffmpeg|bmcv|fp32/int8|yolov3|static|1|Y

* Attention: 4 and 5 are only for SOC mode, 7 is only for CPP test.

### Face Detection

ID|Input|Decoder|Preprocessor|Data Type|Model|Mode
-|-|-|-|-|-|-
1|image|opencv|opencv|fp32|mtcnn|dynamic

[C++ Samples Usage Instructions](samples/cpp/README.md)

[Python Samples Usage Instructions](samples/python/README.md)

## Documnets

[PDF English doc](docs/Sophon_Inference_en.pdf)

[PDF Chinese doc](docs/Sophon_Inference_zh.pdf)

## Testing

Result files of each examples to compare are in directories: ./release/qa_test/sc5_tests/cpp_OR_python/CASE_NAME/verify_files/

## FAQ
[FAQ link](docs/FAQ.md)

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* haipeng.wan   - Initial work
* hong.liu      - Initial work
* juntao.tong   - Initial work
* lian.he       - Initial work
* mike.wu       - Initial work
* tong.liu      - Initial work
* zhenpeng.xiao - Initial work

See also the list of [Contributors](CODEOWNERS) who participated in this project.

## Coding Style Guide

This project contains codes written in C++, Python and Shell. We refer to Google Style Guides with some minor modifications.

[Coding Style Guide](docs/CODING_STYLE_GUIDE.md)

## License

This project is licensed under the Apache License, Version 2.0.

[License Detail](LICENSE)
