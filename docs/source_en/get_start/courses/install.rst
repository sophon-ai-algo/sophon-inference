install required softwares
__________________________
BMNNSDK is the original deep learning development toolkit of Bitmain.
You can contact us to get it.
It is a tarfile named as bmnnsdk2-bm1684_vx.x.x.tar.gz.
The bmnnsdk2 means that it is the second version of bmnnsdk.
The bm1684 means that it is suitable for the chip of bm1684.
After you uncompress this tarfile, 
all modules of bmnnsdk will placed in the folder of bmnnsdk2-bm1684_vx.x.x,
which we are going to use ${BMNNSDK} to represent.

**Installing libs** should only be done one time after you uncompress this tarfile.
The purpose of this installation is that we will choose the correct version of libs depends on your kernel version.

    .. code-block:: shell

       cd ${BMNNSDK}/scripts/ && ./install_lib.sh nntc


**Installing driver** also should only be done one time.
After installing the driver, you can see "bmdev-ctl" and "bm-sophon0" under your "/dev/" path.

    .. code-block:: shell

       cd ${BMNNSDK}/scripts/ && sudo ./install_driver_pcie.sh

       # check if installing successfully
       ls /dev/ | grep "bm"


**Installing bmnett and Configuring environment** should be done as long as you open a new terminal.
Bmnett is a python module and python3.5 is recommanded.

    .. code-block:: shell

       cd ${BMNNSDK}/scripts/ && source envsetup_pcie.sh bmnett


Sophon-Inference is a submodule of BMNNSDK which supplies a bunch of hight level APIs.
With Sophon-Inference, we can rapidly deploy our deep learning models on Sophon TPU products.
**Install Sophon-Inference:**
    .. code-block:: shell

       cd ${BMNNSDK}/exsamples/sail/python3/x86/ && pip3 install sophon-2.0.2-py3-none-any.whl --user





