Sophon Inference Installation
_____________________________

In the section of "1.1 Sophon TPU Choices", we introduced four kinds of Sophon TPU products: SC, SE, SA, SM.
SM serials are customized, which we will not mention in this article.
SC serials are PCIE accelerators working on x86 platform.
SE and SA serials are all SOC accelerators working on ARM platform.
In SOC mode, operating system is running on the TPU memory, and schelued by the ARM core on TPU itself.

For the model deployment on SE and SA products, 
driving TPU to do inference is running on it, 
while the procedure of converting original models to bmodels are excecuted on x86 servers.
And the runtime libs like BMDriver, BMRuntime, BMCV, BMdecoder are all pre-installed on SE and SA products,
so, we only introduce the installation of BMMNSDK on x86 servers.
If you want to deploy your application on SE and SA products finally,
the installation of offline-tools in BMMNSDK is just the all you need to know.


Get BMNNSDK and Choose Link Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    BMNNSDK is released as a tarfile, which names as bmnnsdk2-bm1684_vx.x.x.tar.gz.
    bmnnsdk2 means that it is the second version serials.
    bm1684 represents the suitable chip version.
    x.x.x is the detailed version tag.
    We use ${BMNNSDK} as the root folder of BMNNSDK after uncompression.

    Correct libraries should be choosen according to the kernel version of your system.
    So, a script named "install_lib.sh" should be executed once you uncomress the BMNNSDK.

        .. code-block:: shell

           cd ${BMNNSDK}/scripts/
           ./install_lib.sh nntc



Install Offline Tools
^^^^^^^^^^^^^^^^^^^^^

    In the section of "1.2.2 BMNNSDK", we introduced all softwares in BMNNSDK.
    Offline tools includes Quantization & Calibration tool and BMCompilers.
    A script which should be executed once you open a new terminal is supplied for you to install all the offline tools.

        .. code-block:: shell

           cd ${BMNNSDK}/scripts/
           source envsetup_pcie.sh

    Attention that, there are many dependencies for BMCompilers.
    If you only want one of the BMCompilers(bmnetc/t/m/p) or Quantization & Calibration tool,
    just install one of them is OK.

        .. code-block:: shell

           cd ${BMNNSDK}/scripts
           # intall Quantization & Calibration tool
           source envsetup_pcie.sh ufw
           # install bmnett
           source envsetup_pcie.sh bmnett
           # install bmnetc
           source envsetup_pcie.sh bmnetc
           # install bmnetm
           source envsetup_pcie.sh bmnetm
           # install bmnetp
           source envsetup_pcie.sh bmnetp





Install Runtime Tools
^^^^^^^^^^^^^^^^^^^^^

    The runtime libraries to be installed are BMDriver and Sophon Inference by now.
    The installation of BMDriver needs root's priority.
    BMDriver will be compiler based on your kernel source and installed on system kernel after follow commands.

        .. code-block:: shell

           cd ${BMNNSDK}/scripts/
           sudo ./install_driver_pcie.sh

    Install sophon inference:

        .. code-block:: shell

           cd ${BMNNSDK}/examples/sail/x86/
           pip3 install sophon-x.x.x-py3-none-any.whl --user





