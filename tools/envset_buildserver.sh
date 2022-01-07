#!/bin/bash

export PATH=/workspace/bm_prebuilt_toolchains/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/:/workspace/bm_prebuilt_toolchains/mips-loongson-gcc7.3-linux-gnu/2019.06-29/bin/:/workspace/bm_prebuilt_toolchains/x86-64-core-i7--glibc--stable/bin/:/usr/sw/swgcc830_cross_tools-bb623bc9a/usr/bin/:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:
export PATH=/workspace/bm_prebuilt_toolchains/loongarch64-linux-gnu-2021-04-22-vector/bin:$PATH
echo ${PATH}
export LD_LIBRARY_PATH=/workspace/bm_prebuilt_toolchains/x86-64-core-i7--glibc--stable/x86_64-buildroot-linux-gnu/lib64:/workspace/bm_prebuilt_toolchains/loongarch64-linux-gnu-2021-04-22-vector/loongarch64-linux-gnu/sysroot/lib64:/usr/local/lib: