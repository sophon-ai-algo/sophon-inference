#!/usr/bin/env bash

rm -fr build
mkdir build;cd build

# params: <pcie|arm_pcie|soc|mips|cmodel> <local|bmnnsdk2|allinone>
target_arch=pcie
sdk_type=bmnnsdk2

if [ -n "$1" ]; then
   target_arch=$1
fi

if [ -n "$2" ]; then
   sdk_type=$1
fi

echo sdk_type=$sdk_type
# judge param1
if [ "$target_arch" = "pcie" ]; then
    cmake_param1="-DBUILD_TYPE=pcie"
elif [ "$target_arch" = "arm_pcie" ]; then
    cmake_param1="-DBUILD_TYPE=arm_pcie -DCMAKE_TOOLCHAIN_FILE=./cmake/aarch64-linux-toolchain.cmake"
elif [ "$target_arch" = "soc" ]; then
    cmake_param1="-DBUILD_TYPE=soc -DCMAKE_TOOLCHAIN_FILE=./cmake/BM1684_SOC/ToolChain_aarch64_linux.cmake"
elif [ "$target_arch" = "mips" ]; then
    cmake_param1="-DBUILD_TYPE=mips -DCMAKE_TOOLCHAIN_FILE=./cmake/mips64-linux-toolchain.cmake"
fi

# judge param2
if [ "$sdk_type" = "bmnnsdk2" ]; then
    cmake_param2="-DUSE_BMNNSDK2=ON -DUSE_LOCAL=OFF -DUSE_ALLINONE=OFF"
elif [ "$target_arch" = "local" ]; then
    cmake_param2="-DUSE_BMNNSDK2=OFF -DUSE_LOCAL=ON -DUSE_ALLINONE=OFF"
elif [ "$target_arch" = "allinone" ]; then
    cmake_param2="-DUSE_BMNNSDK2=OFF -DUSE_LOCAL=OFF -DUSE_ALLINONE=ON"
fi

cmake_param2="$cmake_param2 -DSDK_TYPE=$sdk_type"
echo $cmake_param2

PYTHON_BIN=$HOME/bitmain/work/bm_prebuilt_toolchains/pythons/Python-3.8.2/python_3.8.2/bin/python3.8
PYTHON_LIB=$HOME/bitmain/work/bm_prebuilt_toolchains/pythons/Python-3.8.2/python_3.8.2/lib
PYBIND11_PYTHON_VERSION=3.8.2

export LD_LIBRARY_PATH=$PYTHON_LIB
cmake -DCMAKE_TOOLCHAIN_FILE=./toolchain-x86_64-linux.cmake $cmake_param1 $cmake_param2 -DPYBIND11_PYTHON_VERSION=3.8.2 -DPYTHON_EXECUTABLE=$PYTHON_BIN -DCUSTOM_PY_LIBDIR=$PYTHON_LIB ..
make -j4

cd ..

