## Summary

```shell
sc5_tests
├── cmake                    # BMNNSDK2 should be installed and sourced envsetup.sh
│   └── FindBMNNSDK2.cmake   # find dependences to build C++ examples
├── CMakeLists.txt
├── cpp                      # cpp examples source code
│   ├── cls_resnet
│   ├── det_mtcnn
│   ├── det_ssd
│   └── det_yolov3
├── download.py              # python script to download test data and models
├── python                   # python examples source code
│   ├── cls_resnet
│   ├── det_mtcnn
│   ├── det_ssd
│   └── det_yolov3
└── README.md
```

### Image Classification(Python && C++)

ID|Input|Decoder|Preprocessor|Data Type|Model|Mode|Model Number|TPU Number|Multi-Thread
-|-|-|-|-|-|-|-|-|-
1|image|opencv|opencv|fp32/int8|resnet-50|static|1|1|N
2|image|opencv|opencv|fp32/int8|resnet-50|static|1|1|Y
3|image|opencv|opencv|fp32/int8|resnet-50|static|1|2|Y
4|image|opencv|opencv|fp32/int8|resnet-50|static|2|1|Y

### Object Detection(Python && C++)

ID|Input|Decoder|Preprocessor|Data Type|Model|Mode|Batch Size|Multi-Thread
-|-|-|-|-|-|-|-|-
1|video/image|opencv|opencv|fp32/int8|ssd_vgg|static|1|N
2|video/image|bm-ffmpeg|bmcv|fp32/int8|ssd_vgg|static|1|N
3|video/image|bm-ffmpeg|bmcv|fp32/int8|ssd_vgg|static|4|N
4|multi-video|opencv|opencv|fp32/int8|yolov3|static|1|Y
5|multi-video|bm-ffmpeg|bmcv|fp32/int8|yolov3|static|1|Y

### Face Detection(Python && C++)

ID|Input|Decoder|Preprocessor|Data Type|Model|Mode
-|-|-|-|-|-|-
1|image|opencv|opencv|fp32|mtcnn|dynamic


## Notes

* Environment configuration

```shell
# bmnnsdk2 should be download and uncompressed
cd bmnnsdk2-bm1684_vx.x.x/scripts
./install_lib.sh nntc
source envsetup_pcie.sh

# set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:bmnnsdk2-bm1684_vx.x.x/lib/ffmpeg/x86:bmnnsdk2-bm1684_vx.x.x/lib/decode/x86
```

* Python module named sophon is needed to install

```shell
# the wheel package is in the bmnnsdk2:
# for centos: ./examples/sail/python3/x86/lib_CXX11_ABI0/sophon-2.0.2-py3-none-any.whl
# for ubuntu: ./examples/sail/python3/x86/lib_CXX11_ABI1/sophon-2.0.2-py3-none-any.whl
pip3 install sophon-2.0.2-py3-none-any.whl --user
```

* Build C++ examples

```shell
mkdir build
cd build
cmake ..
make -j
```
