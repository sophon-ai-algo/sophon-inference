# sophon-inference package instruction

```shell
sophon-inference/
|-- README.md
|-- docs
|   `-- Sophon_Inference_zh.pdf              # sophon-inference document
|-- include
|   `-- sail                                 # sail header
|-- lib
|   `-- sail
|       |-- soc                              # sail soc lib
|       |-- arm_pcie                         # sail arm_pcie lib
|       |-- cmodel                           # sail cmodel lib (x86 mode without decode && bmcv)
|       |   |-- lib_CXX11_ABI0               # centos lib
|       |   `-- lib_CXX11_ABI1               # ubuntu lib
|       `-- pcie                             # sail pcie lib
|           |-- lib_CXX11_ABI0               # centos lib
|           `-- lib_CXX11_ABI1               # ubuntu lib
|-- python3
|   |-- soc
|   |   `-- sophon                           # pre-installed on soc /system/lib
|   |-- arm_pcie                             # arm_pcie sophon-inference python whl
|   |-- cmodel
|   |   |-- lib_CXX11_ABI0                   # (centos) cmodel sophon-inference python whl (x86 mode without decode && bmcv)
|   |   `-- lib_CXX11_ABI1                   # (ubuntu) cmodel sophon-inference python whl (x86 mode without decode && bmcv)
|   `-- pcie
|       |-- lib_CXX11_ABI0                   # (centos) pcie sophon-inference python whl
|       `-- lib_CXX11_ABI1                   # (ubuntu) pcie sophon-inference python whl
`-- samples
|   |-- cpp
|   |   |-- cpp_cv_bmcv_sail                 # run ssd: opencv decode, bmcv preprocess
|   |   |-- cpp_cv_cv+bmcv_sail              # run ssd: opencv decode, cv::bmcv && bmcv preprocess
|   |   `-- cpp_ffmpeg_bmcv_sail             # run ssd: ffmpeg decode, bmcv preprocess
|   `-- python
|       `-- det_ssd_bmcv                     # python sample det_ssd_bmcv(bmcv)
`-- test
|   |-- sa5_tests
|   |   |-- test_resnet50                    # sophon-inference cpp test unit
|   |   `-- test_ssd                         # sophon-inference python test unit
|   `-- sc5_tests
`-- scripts
    `-- install_sail.sh                      # install sail lib by linux system version
```

# dependent env (base_path=nntoolchain_path)
* Install nntc lib

```shell
cd base_path/scripts
./install_lib.sh nntc
```

* Install bmcompiler && Set the BMNNSDK system environment

```shell
cd base_path/test
source envsetup_pcie.sh | envsetup_cmodel.sh | envsetup_soc.sh | envsetup_arm_pcie.sh
```

* Install sophon-inference on X86

```shell
cd ./scripts && ./install_sail.sh

cd ./python3/pcie
pip3 install sophon-2.0.0-py3-none-any.whl --user
```
