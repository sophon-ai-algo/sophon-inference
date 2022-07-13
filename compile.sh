#!/usr/bin/env bash

rm -fr build
mkdir build;

# params: <pcie|arm_pcie|soc|mips|cmodel> <local|sophonsdk3|allinone>
target_arch=soc
sdk_type=sophonsdk3

PYTHON_BIN=/workspace/pythons/Python-3.8.2/python_3.8.2/bin/python3
PYTHON_LIB=/workspace/pythons/Python-3.8.2/python_3.8.2/lib/

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
elif [ "$target_arch" = "soc" ]; then
    cmake_param1="-DBUILD_TYPE=soc -DCMAKE_TOOLCHAIN_FILE=./cmake/BM1684_SOC/ToolChain_aarch64_linux.cmake"
elif [ "$target_arch" = "mips" ]; then
    cmake_param1="-DBUILD_TYPE=mips -DCMAKE_TOOLCHAIN_FILE=./cmake/mips64-linux-toolchain.cmake"
elif [ "$target_arch" = "arm_pcie" ]; then
    cmake_param1="-DBUILD_TYPE=arm_pcie -DCMAKE_TOOLCHAIN_FILE=./cmake/aarch64-linux-toolchain.cmake"
fi

# judge param2
if [ "$sdk_type" = "sophonsdk3" ]; then
    cmake_param2="-DUSE_SOPHONSDK3=ON -DUSE_LOCAL=OFF -DUSE_ALLINONE=OFF"
elif [ "$target_arch" = "local" ]; then
    cmake_param2="-DUSE_SOPHONSDK3=OFF -DUSE_LOCAL=ON -DUSE_ALLINONE=OFF"
elif [ "$target_arch" = "allinone" ]; then
    cmake_param2="-DUSE_SOPHONSDK3=OFF -DUSE_LOCAL=OFF -DUSE_ALLINONE=ON"
fi

cmake_param2="$cmake_param2 -DSDK_TYPE=$sdk_type"
echo $cmake_param2

function get_python3_version() {
    U_V1=`$PYTHON_BIN -V 2>&1|awk '{print $2}'|awk -F '.' '{print $1}'`
    U_V2=`$PYTHON_BIN -V 2>&1|awk '{print $2}'|awk -F '.' '{print $2}'`
    U_V3=`$PYTHON_BIN -V 2>&1|awk '{print $2}'|awk -F '.' '{print $3}'`
    echo $U_V1"."$U_V2"."$U_V3
    return $?
}

function create_whl_path() {
  U_V1=`$PYTHON_BIN -V 2>&1|awk '{print $2}'|awk -F '.' '{print $1}'`
  U_V2=`$PYTHON_BIN -V 2>&1|awk '{print $2}'|awk -F '.' '{print $2}'`
  mkdir -p out/sophon-inference/python$U_V1$U_V2
  echo out/sophon-inference/python$U_V1$U_V2
  return $?
}

function create_release_directory() {
  echo "----------------------------- Create release folder ---------------------------"
  if [ -d "out" ]  ; then
    rm -rf out
  fi
  mkdir -p out/sophon-inference/
  mkdir -p out/sophon-inference/include/sail
  mkdir -p out/sophon-inference/lib
}

function build_lib(){
    echo "----------------------------- Start build lib ---------------------------------"
    if [ ! -f "CMakeLists.txt" ]; then
        echo "Error: Please excute the command at project root path!"
        exit 1
    fi
    if [ ! -d "build" ]; then
        mkdir build
    else
        rm -rf ./build/*
    fi
    export LD_LIBRARY_PATH=$PYTHON_LIB:$LD_LIBRARY_PATH
    PYBIND11_PYTHON_VERSION=$(get_python3_version)
    pushd build
    cmake $cmake_param1 $cmake_param2 -DPYBIND11_PYTHON_VERSION=$PYBIND11_PYTHON_VERSION -DPYTHON_EXECUTABLE=$PYTHON_BIN -DCUSTOM_PY_LIBDIR=$PYTHON_LIB ..
    core_nums=`cat /proc/cpuinfo| grep "processor"| wc -l`
    make -j${core_nums}
    if [ $? -eq 0 ];then
        echo "Build succeed!"
    else
        echo "Build Failed!"
        exit 1
    fi
    popd
}

function build_whl_pcie() {
    echo "----------------------------- Start build wheel -------------------------------"
    pushd python/pcie
    ./sophon_pcie_whl.sh
    popd 
    echo "----------------------------- Start copy wheel --------------------------------"
    whl_res_path=$(create_whl_path)/pcie/
    mkdir -p $whl_res_path
    cp ./python/pcie/dist/*.whl $whl_res_path
}

function build_whl_soc() {
    echo "----------------------------- Start build wheel -------------------------------"
    pushd python/soc
    ./sophon_soc_whl.sh
    popd 
    echo "----------------------------- Start copy wheel --------------------------------"
    whl_res_path=$(create_whl_path)/soc/
    mkdir -p $whl_res_path
    cp ./python/soc/arm/dist/*.whl $whl_res_path
}

function fill_headers() {
    echo "----------------------------- Fill headers ------------------------------------"
    cp ./include/*.h ./out/sophon-inference/include/sail
    cp -r ./3rdparty/spdlog ./out/sophon-inference/include/sail
    cp ./3rdparty/inireader* ./out/sophon-inference/include/sail
    cp ./build/lib/libsail.so ./out/sophon-inference/lib/
}


shell_dir=$(dirname $(readlink -f "$0"))

export LD_LIBRARY_PATH=$PYTHON_LIB:$LD_LIBRARY_PATH
create_release_directory
build_lib
if [ "$target_arch" = "pcie" ]; then
    build_whl_pcie
elif [ "$target_arch" = "soc" ]; then
    build_whl_soc
fi
fill_headers