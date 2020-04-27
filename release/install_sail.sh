##############################################
#install sail lib with Correct version of g++#
##############################################
#!/bin/bash

SAIL_TOP=`pwd`/..

function sail_install() {
  i=1
  if [ $1 -eq "1" ]; then
    i=0
  fi
  pushd $SAIL_TOP/lib/sail/python3
  # install cmodel python wheel
  rm -rf cmodel/lib_CXX11_ABI$i
  mv cmodel/lib_CXX11_ABI$1/* ./cmodel/
  rm -rf cmodel/lib_CXX11_ABI$1

  # install pcie python wheel
  rm -rf pcie/lib_CXX11_ABI$i
  mv pcie/lib_CXX11_ABI$1/* ./pcie/
  rm -rf pcie/lib_CXX11_ABI$1

  # install arm_pcie python wheel
  rm -rf arm_pcie/lib_CXX11_ABI$i
  mv arm_pcie/lib_CXX11_ABI$1/* ./arm_pcie/
  rm -rf arm_pcie/lib_CXX11_ABI$1
  popd

  pushd $SAIL_TOP/lib/sail
  # install cmodel lib
  rm -rf cmodel/lib_CXX11_ABI$i
  mv cmodel/lib_CXX11_ABI$1/* ./cmodel/
  rm -rf cmodel/lib_CXX11_ABI$1

  # install pcie lib
  rm -rf pcie/lib_CXX11_ABI$i
  mv pcie/lib_CXX11_ABI$1/* ./pcie/
  rm -rf pcie/lib_CXX11_ABI$1

  # install arm_pcie lib
  rm -rf arm_pcie/lib_CXX11_ABI$i
  mv arm_pcie/lib_CXX11_ABI$1/* ./arm_pcie/
  rm -rf arm_pcie/lib_CXX11_ABI$1
  popd
}

function sail_install_local() {
  i=1
  if [ $1 -eq "1" ]; then
    i=0
  fi
  pushd $SAIL_TOP/python3
  # install cmodel python wheel
  rm -rf cmodel/lib_CXX11_ABI$i
  mv cmodel/lib_CXX11_ABI$1/* ./cmodel/
  rm -rf cmodel/lib_CXX11_ABI$1

  # install pice python wheel
  rm -rf pcie/lib_CXX11_ABI$i
  mv pcie/lib_CXX11_ABI$1/* ./pcie/
  rm -rf pcie/lib_CXX11_ABI$1

  # install pice python wheel
  rm -rf arm_pcie/lib_CXX11_ABI$i
  mv arm_pcie/lib_CXX11_ABI$1/* ./arm_pcie/
  rm -rf arm_pcie/lib_CXX11_ABI$1
  popd

  pushd $SAIL_TOP/lib/sail
  # install cmodel lib
  rm -rf cmodel/lib_CXX11_ABI$i
  mv cmodel/lib_CXX11_ABI$1/* ./cmodel/
  rm -rf cmodel/lib_CXX11_ABI$1

  # install pcie lib
  rm -rf pcie/lib_CXX11_ABI$i
  mv pcie/lib_CXX11_ABI$1/* ./pcie/
  rm -rf pcie/lib_CXX11_ABI$1

  # install arm_pcie lib
  rm -rf arm_pcie/lib_CXX11_ABI$i
  mv arm_pcie/lib_CXX11_ABI$1/* ./arm_pcie/
  rm -rf arm_pcie/lib_CXX11_ABI$1
  popd
}

# get linux info
linux=`sed -n '1p' /etc/issue`

#get gcc version(only for centos)
gcc=`gcc --version | grep gcc`
gcc_version=${gcc:0-9:3}

linux=`echo $linux | sed 's/ //g'`

version_flag=1
if [ $linux == "\S" ]; then
  if [ $gcc_version == "4.8" ]; then
    version_flag=0
  else
    version_flag=1
  fi
else
  version_flag=1
fi

if [ -n "$1" ]; then
  if [ $1 == "nntc" ]; then
    sail_install $version_flag
  else
    echo "Usage: ./install_sail.sh nntc"
  fi
else
  sail_install_local $version_flag
fi
