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

## SAIL: Sophon Artificial Intelligent Library for online deployment.
* It's a wrapper of bmruntime, bmdecoder, bmcv, bmlib;
* Provide both C++ and Python APIs;
* Automatically manage memory of tensors;

### 如何编译
Sophon-Inference依赖SophonSDK,可以在x86架构的Linux/Windows10主机上进行编译,生成对应版本的动态库及Python Wheel文件,也可通过配置交叉编译器及编译出其它架构下的动态库及Python Wheel文件。本文档后续内容均以x86架构Linux下的使用操作为例。
#### 获取SDK
SophonSDK开发包由两部分组成：
* 基于x86的Ubuntu开发Docker
   我们提供统一的Docker基础开发镜像以规避环境不一致带来的隐性问题，基础Docker镜像的版本为Ubuntu18.04。
* SophonSDK
   sophonsdk_v<x.y.z>.tar.gz，其中x.y.z为版本号。SophonSDK将以目录映射的方式挂载到Docker容器内部供用户使用。

您可以访问[算能官网](https://developer.sophgo.com/site/index.html)来下载相关资料
|项目                            |下载页面|
|:------------------------------:|:---------------------------:|
|Ubuntu18.04开发Docker - Python3.7|[点击前往官网下载界面](https://developer.sophgo.com/site/index/material/11/all.html)|
|SophonSDK3.0.0  开发包           |[点击前往官网下载界面](https://developer.sophgo.com/site/index/material/17/all.html)|

#### 安装Sophon设备驱动并配置基础开发环境
我们提供了docker开发镜像供用户在x86主机下开发和部署使用，docker中已安装好了交叉编译环境以及开发工具依赖的库和软件。
* 解压SDK压缩包
``` bash
   tar zxvf sophonsdk_v<x.y.z>.tar.gz
   cd sophonsdk_v<x.y.z>
```

* 驱动安装
驱动安装请在docker之外宿主机上进行,如果宿主机上没有Sophon PCIe加速卡,请跳过本节。
  - 1. 检查PCIe加速卡是否正常被系统识别
  打开终端执行 ``lspci | grep Sophon`` 检查卡是否能够识别，正常情况应该输出如下信息：
  ```bash
    01:00.0 Processing accelerators: Bitmain Technologies Inc. BM1684, Sophon Series Deep Learning Accelerator (rev 01)
  ```
  - 2. PCIe环境驱动安装
  ```bash
    cd sophonsdk_v<x.x.x>/scripts
    sudo ./install_driver_pcie.sh
  ```

  - 3. 检查驱动是否安装成功
  打开终端执行 ``ls /dev/bm* `` 看看是否有/dev/bm-sohponX (X表示0-N），如果有表示安装成功。 正常情况下输出如下信息：
  ```bash
    /dev/bmdev-ctl /dev/bm-sophon0
  ```
* Docker安装
若已安装docker，请跳过本节。
```bash
   # 安装docker
   sudo apt-get install docker.io
   # docker命令免root权限执行
   # 创建docker用户组，若已有docker组会报错，没关系可忽略
   sudo groupadd docker
   # 将当前用户加入docker组
   sudo gpasswd -a ${USER} docker
   # 重启docker服务
   sudo service docker restart
   # 切换当前会话到新group或重新登录重启X会话
   newgrp docker​ 
```
```提示：需要logout系统然后重新登录，再使用docker就不需要sudo了。```

* 加载docker镜像
   ```bash
   docker load -i x86_sophonsdk3_ubuntu18.04_py37_dev_22.06.docker
   ```
```注意：后续版本的Docker名字可能会变化，请根据实际Docker名字做输入。```

* 将sophon inference源码拷贝至SophonSDK根目录,也即是```sophonsdk_v<x.x.x>```
```注意：拷贝源码到SophonSDK根目录是为了方便映射到docker, 也可以通过修改创建docker的脚本映射到其它位置。```

* 创建docker容器进入docker

   ```bash
    cd sophonsdk_v<x.x.x>
    # 若您没有执行前述关于docker命令免root执行的配置操作，需在命令前添加sudo
    ./docker_run_sophonsdk.sh
   ```
**(以下步骤在docker中进行)**

* 工具安装
   ```bash
   cd  /workspace/scripts/
   ./install_lib.sh nntc 
   ```
```注意：此步骤只需要在创建docker容器之后运行一次即可。```

#### 编译SAIL
**(以下步骤在docker中进行)**
* 配置环境变量 
   如果宿主机上有Sophon PCIe加速卡可以执行pcie环境变量的脚本。
   ```bash
   cd  /workspace/scripts/
   source envsetup_pcie.sh
   ```
   如果宿主机上没有Sophon PCIe加速卡可以执行cmodel环境变量的脚本。
   ```bash
   cd  /workspace/scripts/
   source envsetup_cmodel.sh
   ```
```注意：配置环境变量的步骤没打开一个新的终端都需要重新设置一下。```

* 编译在x86架构上的运行的动态库及python wheel
   ```bash
   cd /workspace/sophon-inference
   ./compile.sh pcie
   ```
   编译结果的目录为:```/workspace/sophon-inference/out/sophon-inference```
```注意：目前docker内的python版本为python3.7,所以编译出的wheel包也是针对python3.7的,如果需要其它版本的wheel包,则需要下载安装相应的python包,[参考链接](http://219.142.246.77:65000/sharing/8MlSKnV8x),将其映射到docker内,并在'compile.sh'文件中修改python的可执行程序路径'PYTHON_BIN'及对应依赖库的目录'PYTHON_LIB',然后重新执行上述步骤即可。```

* 编译在Sophon边缘设备上运行的动态库及python wheel
   ```bash
   cd /workspace/sophon-inference
   ./compile.sh soc
   ```
   编译结果的目录为:```/workspace/sophon-inference/out/sophon-inference```
```注意：目前docker内的python版本为python3.7,所以编译出的wheel包也是针对python3.7的,由于目前Sophon边缘设备的python版本为3.5,所以需要先下载python3.5的包,[参考连接](http://219.142.246.77:65000/sharing/8MlSKnV8x),将其映射到docker内,并在'compile.sh'文件中修改python的可执行程序路径'PYTHON_BIN'及对应依赖库的目录'PYTHON_LIB',然后重新执行上述步骤即可。```


### 官方论坛
[论坛链接](https://developer.sophgo.com/forum/index.html)


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

## Coding Style Guide

This project contains codes written in C++, Python and Shell. We refer to Google Style Guides with some minor modifications.

[Coding Style Guide](docs/CODING_STYLE_GUIDE.md)

## License

This project is licensed under the Apache License, Version 2.0.

[License Detail](LICENSE)
