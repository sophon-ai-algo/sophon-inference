## Summary

```shell
sc5_tests
├── cmake                    # BMNNSDK2 should be installed and sourced envsetup.sh
│   └── FindBMNNSDK2.cmake  # find dependences to build C++ examples
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
3|video|bm-ffmpeg|bmcv|fp32/int8|ssd_vgg|static|4|N
4|video|opencv|opencv|fp32/int8|yolov3|static|1|Y
5|video|bm-ffmpeg|bmcv|fp32/int8|yolov3|static|1|Y

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

# for x86
# remove old driver and install new driver for pice mode
sudo ./remove_driver_pcie.sh
sudo ./install_driver_pcie.sh
# setup environment paths
source envsetup_pcie.sh

# for arm
# remove old driver and install new driver for pice mode
sudo ./remove_driver_arm_pcie.sh
sudo ./install_driver_arm_pcie.sh
# setup environment paths
source envsetup_arm_pcie.sh
```

* Python module named sophon is needed to install

```shell
# the wheel package is in the bmnnsdk2:
pip3 uninstall -y sophon
# get your python version
python3 -V
# choose the same verion of sophon wheel to install
# the following py3x maybe py35, py36, py37 or py38
# for x86
pip3 install ../lib/sail/python3/pcie/py3x/sophon-2.0.3-py3-none-any.whl --user
# for arm
pip3 install ../lib/sail/python3/arm_pcie//py3x/sophon-2.0.3-py3-none-any.whl --user
```

* Test cpp and python cases

```shell
# you can modify the script to add tests
# c++ readmes: cpp/CASE_NAME/README.md
# python readmes: python/CASE_NAME/README.md
./auto_test.sh
```
