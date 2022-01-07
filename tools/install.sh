#!/bin/bash

function usage() {
  echo "Usage: bash ./tools/install.sh"
}

function judge_ret() {
  if [[ $1 == 0 ]]; then
    echo "Passed: $2"
    echo ""
  else
    echo "Failed: $2"
    exit 5
  fi
}

function install_third_part_lib() {
  # install cmake and gnu tools
  ${use_sudo} apt install -y build-essential cmake
  ${use_sudo} apt-get install -y pkg-config
  # install python3, numpy, pip3, required for Python API
  ${use_sudo} apt install -y python3-dev python3-numpy python3-pip
  pip3 install gluoncv --user
  pip3 install opencv-contrib-python --user
  pip3 install PyYAML --user
  pip3 install numpy --user
  pip3 install opencv-python --user

  # install Sphinx, required for Python API document.
  pip3 install sphinx sphinx-autobuild sphinx_rtd_theme recommonmark --user
}

function install_opencv3_from_source() {
  # build opencv3 from source if needed
  workspace=$(cd "$(dirname "$0")";pwd)
  cd "${workspace}"
  opencv_version=$(pkg-config opencv --modversion)
  opencv_version=${opencv_version:0:3}
  if [ "${opencv_version}" == "3.4" ];then
    echo "opencv3.4 already install.."
  else
    git clone https://github.com/opencv/opencv
    cd opencv
    git fetch && git checkout 3.4
    mkdir build && cd build
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
    make -j4 && ${use_sudo} make install
    cd ../.. && rm -rf opencv
  fi
}

function install_bmsdk() {
  python3 -c "import bmnetc"
  if [[ $? == 0 ]]; then
    echo "found bmnetc in python3"
    pip3 uninstall bmnetc -y
    echo ""
  fi

  pushd "${BMSDK_PATH}"
  pip3 install bmnetc/bmnetc-?.?.?-py2.py3-none-any.whl --user
  popd

  python3 -c "import bmnett"
  if [[ $? == 0 ]]; then
    echo "found bmnett in python3"
    pip3 uninstall bmnett -y
    echo ""
  fi
  pushd "${BMSDK_PATH}"
  pip3 install bmnett/bmnett-?.?.?-py2.py3-none-any.whl --user
  popd

  python3 -c "import bmnetm"
  if [[ $? == 0 ]]; then
    echo "found bmnetm in python3"
    pip3 uninstall bmnetm -y
    echo ""
  fi
  pushd "${BMSDK_PATH}"
  pip3 install bmnetm/bmnetm-?.?.?-py2.py3-none-any.whl --user
  popd

  python3 -c "import bmnetp"
  if [[ $? == 0 ]]; then
    echo "found bmnetp in python3"
    pip3 uninstall bmnetp -y
    echo ""
  fi
  pushd "${BMSDK_PATH}"
  pip3 install bmnetp/bmnetp-?.?.?-py2.py3-none-any.whl --user
  popd
}

BMSDK_PATH="$REL_TOP"
if [ -z "$BMSDK_PATH" ]; then
  echo "Error: $BMSDK_PATH not exists!"
  echo "Please 'cd bmsdk_path && source envsetup.sh'"
fi
if [ ! -d "$BMSDK_PATH" ]; then
  echo "Error: $BMSDK_PATH not exists!"
  usage
  exit 3
fi
if [[ ${BMSDK_PATH:0:1} != "/" ]]; then
  echo "Error: $BMSDK_PATH is not absolute path!"
  usage
  exit 4
fi
echo "BMNNSDK_PATH = $BMSDK_PATH"

who=`whoami`
use_sudo=sudo
if [ "${who}" == "root" ];then
  use_sudo= 
fi 
install_third_part_lib
install_opencv3_from_source
install_bmsdk
