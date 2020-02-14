软件安装指南
____________


在 “1.1 算丰硬件产品概览” 中，我们介绍了我们目前的四种产品形态：SC、SE、SA、SM。
其中，SM 属于定制化的产品，因此在这里不做详细介绍。
SC 系列产品为 PCIE 模式的加速卡，作为协处理器接受 X86 主机 CPU 的调用。
SE 和 SA 系列产品为 SOC 模式，该模式下，操作系统运行在 TPU 内存上，由 TPU 上的 ARM 处理器负责管理和调度。

对于 SE 和 SA 系列产品的模型部署，我们通常在 X86 系统上编译模型生成 bmodel，再在 SE 和 SA 产品上部署。
而在 SE 和 SA 系列产品上，我们已经预装了 BMNNSDK 中的全部运行时模块。
因此，在这里我们只介绍 X86 主机下 PCIE 模式的 BMNNSDK 的安装。
如果你希望你的模型最终运行在 SE 或 SA 产品上，那么只需了解 X86 主机下 BMNNSDK 中的 离线模型编译工具的安装过程。


获取软件包并安装链接库
^^^^^^^^^^^^^^^^^^^^^^

BMNNSDK 以 tar 包的形式发布。
命名方式为 bmnnsdk2-bm1684_vx.x.x.tar.gz。
其中，bmnnsdk2 代表版本号为 2，bm1684 代表支持的芯片编号，x.x.x 为详细版本号。
解压该软件包后，我们用 ${BMNNSDK} 来代替软件包的主目录。

由于 BMNNSDK 中存在由不同版本内核编译生成的链接库，因此在解压完之后，
我们需要根据当前主机的内核版本来选择适当的链接库。
对此，我们提供了对应脚本。每次解压完之后只需运行一次下列命令即可。

    .. code-block:: shell

       cd ${BMNNSDK}/scripts/
       ./install_lib.sh nntc


离线模型编译工具安装
^^^^^^^^^^^^^^^^^^^^

在 ”1.2.2 BMNNSDK“ 中我们介绍了 BMNNSDK 中的所有软件模块。
离线模型编译工具包括了 Quantization & Calibration tool 和 BMCompiler。
我们提供了一个脚本来完成离线工具的安装，每次进入终端之后运行以下命令即可完成安装。

    .. code-block:: shell
    
       cd ${BMNNSDK}/scripts/
       source envsetup_pcie.sh

需要注意的是，由于 BMCompiler 依赖有比较多的依赖包，比如 bmnett 依赖 tensorflow，
bmnetp 依赖 pytorch，bmnetm 依赖 mxnet。
因此，如果你只需要其中某一个工具，可以带参数运行该脚本，如下：

    .. code-block:: shell

       cd ${BMNNSDK}/scrips/
       # 安装 Quantization & Calibration Tool
       source envsetup_pcie.sh ufw
       # 安装 bmnetu
       source envsetup_pcie.sh bmnetu
       # 安装 bmnett
       source envsetup_pcie.sh bmnett
       # 安装 bmnetp
       source envsetup_pcie.sh bmnetp
       # 安装 bmnetm
       source envsetup_pcie.sh bmnetm
       # 安装 bmnetc
       source envsetup_pcie.sh bmnetc


运行时工具安装
^^^^^^^^^^^^^^

目前，需要安装的运行时工具只有 BMDriver 和 Sophon Inference 的 python 包。

安装 BMDriver 需要 root 权限，BMDriver 会在主机上编译并安装到系统内核中。

    .. code-block:: shell

       cd ${BMNNSDK}/scripts/
       sudo ./install_driver_pcie.sh

安装 Sophon Inference：

    .. code-block:: shell

       cd ${BMNNSDK}/examples/sail/x86/
       pip3 install --user sophon-x.x.x-py3-none-any.whl



