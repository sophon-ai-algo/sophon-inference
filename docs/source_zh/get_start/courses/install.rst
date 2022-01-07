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

**在宿主机上解压软件包**

    .. code-block:: shell
      
      # 解压软件包操作应当在宿主机上执行，解压后的目录使用${BMNNSDK} 来标识
       tar -xvf bmnnsdk2-bm1684_vx.x.x.tar.gz

**在宿主机上安装驱动**

    .. code-block:: shell

      # 编译安装驱动应当在宿主机上执行
       cd ${BMNNSDK}/scripts/
       sudo ./install_driver_pcie.sh

**在宿主机上加载docker开发镜像并创建运行容器**

    .. code-block:: shell
      
       docker load -i bmnnsdk_ubuntu_docker.tar.gz
       cd ${BMNNSDK}
       ./docker_run_bmnnsdk.sh
       # 启动运行容器后，宿主机上的${BMNNSDK}目录将被映射到docker容器中的/workspace目录，用户可根据需求修改上述启动脚本的映射目录参数

**在docker开发容器中安装链接库**

    .. code-block:: shell

       cd /workspace/scripts/
       ./install_lib.sh nntc

**在docker开发容器中安装bmnett**

    .. code-block:: shell

       cd /workspace/scripts/
       source envsetup_pcie.sh bmnett

**在docker开发容器中安装Sophon Inference**

    .. code-block:: shell
    
      # 确认平台及python版本，然后进入相应目录，比如x86平台，python3.5
       cd /workspace/lib/sail/python3/pcie/py35
       pip3 install sophon-x.x.x-py3-none-any.whl --user





