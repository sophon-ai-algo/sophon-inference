依赖软件安装
____________

**硬件环境确认**

    .. code-block:: shell

       # 确认系统版本
         bitmain@bitmain:~$ lsb_release -a
         # No LSB modules are available.
         # Distributor ID: Ubuntu
         # Description:    Ubuntu 16.04.6 LTS
         # Release:        16.04
         # Codename:       xenial

       # 确认内核版本
         bitmain@bitmain:~$ uname -r
         # 4.15.0-45-generic

       # 确认 Sophon SC5 已正确插入 PCIE 插槽
         bitmain@bitmain:~$ lspci | grep 1684
         # 01:00.0 Processing accelerators: Device 1e30:1684 (rev 01)


**解压并安装链接库**

    .. code-block:: shell

       cd ${BMNNSDK}/scripts/
       ./install_lib.sh nntc


**安装驱动**

    .. code-block:: shell

       cd ${BMNNSDK}/scripts/
       sudo ./install_driver_pcie.sh


**安装bmnett**

    .. code-block:: shell

       cd ${BMNNSDK}/scripts/
       source envsetup_pcie.sh bmnett

**安装Sophon Inference**

    .. code-block:: shell

       cd ${BMNNSDK}/examples/sail/python3/x86/
       pip3 install sophon-x.x.x-py3-none-any.whl --user





