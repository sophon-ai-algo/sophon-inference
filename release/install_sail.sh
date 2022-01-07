##############################################
#install sail lib with Correct version of g++#
##############################################
#!/bin/bash

function gettop()
{
    echo $(dirname $(readlink -f ${BASH_SOURCE[0]}))
}

CMD_DIR=$(gettop)
SAIL_TOP=$CMD_DIR/..

function safe_rm()
{
    if [ -d "$1" ];then
	rm -fr $1
    fi
}

function safe_mv()
{
    if [ -d "$1" ];then
	local ll=$(ls $1)
	if [ ${#ll} != 0 ]; then
	    mv $1/* $2
        fi
    fi
}

function sail_install() {
  i=1
  if [ $1 -eq "1" ]; then
    i=0
  fi
  pushd $SAIL_TOP/lib/sail/python3
  # install cmodel python wheel
  #safe_rm  cmodel/lib_CXX11_ABI$i
  #safe_mv cmodel/lib_CXX11_ABI$1 ./cmodel/
  #safe_rm  cmodel/lib_CXX11_ABI$1

  # install pcie python wheel
  safe_rm pcie/lib_CXX11_ABI$i
  safe_mv pcie/lib_CXX11_ABI$1 ./pcie/
  safe_rm pcie/lib_CXX11_ABI$1

  # install asafe_rm_pcie python wheel
  safe_rm  arm_pcie/lib_CXX11_ABI$i
  safe_mv arm_pcie/lib_CXX11_ABI$1 ./arm_pcie/
  safe_rm  arm_pcie/lib_CXX11_ABI$1
  
  # install mips64 python wheel
  if [ $1 -eq "0" ];then
    safe_rm  mips64/lib_CXX11_ABI$i
    safe_mv mips64/lib_CXX11_ABI$1 ./mips64/
    safe_rm  mips64/lib_CXX11_ABI$1
  fi
  #install sw64 python wheel
  safe_rm  sw64/lib_CXX11_ABI$i
  safe_mv sw64/lib_CXX11_ABI$1 ./sw64/
  safe_rm  sw64/lib_CXX11_ABI$1

  #install loongarch64 python wheel
  safe_rm  loongarch64/lib_CXX11_ABI$i
  safe_mv loongarch64/lib_CXX11_ABI$1 ./loongarch64/
  safe_rm  loongarch64/lib_CXX11_ABI$1
  popd

  pushd $SAIL_TOP/lib/sail
  # install cmodel lib
  #safe_rm  cmodel/lib_CXX11_ABI$i
  #safe_mv cmodel/lib_CXX11_ABI$1 ./cmodel/
  #safe_rm  cmodel/lib_CXX11_ABI$1

  # install pcie lib
  safe_rm  pcie/lib_CXX11_ABI$i
  safe_mv pcie/lib_CXX11_ABI$1 ./pcie/
  safe_rm  pcie/lib_CXX11_ABI$1

  # install asafe_rm_pcie lib
  safe_rm  arm_pcie/lib_CXX11_ABI$i
  safe_mv arm_pcie/lib_CXX11_ABI$1 ./arm_pcie/
  safe_rm  arm_pcie/lib_CXX11_ABI$1
  
  # install mips64 lib
  if [ $1 -eq "0" ];then
    safe_rm  mips64/lib_CXX11_ABI$i
    safe_mv mips64/lib_CXX11_ABI$1 ./mips64/
    safe_rm  mips64/lib_CXX11_ABI$1
  fi
  #install sw64 lib 
  safe_rm  sw64/lib_CXX11_ABI$i
  safe_mv sw64/lib_CXX11_ABI$1 ./sw64/
  safe_rm  sw64/lib_CXX11_ABI$1

  #install loongarch64 lib 
  safe_rm  loongarch64/lib_CXX11_ABI$i
  safe_mv loongarch64/lib_CXX11_ABI$1 ./loongarch64/
  safe_rm  loongarch64/lib_CXX11_ABI$1
  popd
}

function sail_install_local() {
  i=1
  if [ $1 -eq "1" ]; then
    i=0
  fi
  pushd $SAIL_TOP/python3
  # install cmodel python wheel
  #safe_rm  cmodel/lib_CXX11_ABI$i
  #safe_mv  cmodel/lib_CXX11_ABI$1 ./cmodel/
  #safe_rm  cmodel/lib_CXX11_ABI$1

  # install pice python wheel
  safe_rm pcie/lib_CXX11_ABI$i
  safe_mv pcie/lib_CXX11_ABI$1 ./pcie/
  safe_rm pcie/lib_CXX11_ABI$1

  # install pice python wheel
  safe_rm arm_pcie/lib_CXX11_ABI$i
  safe_mv arm_pcie/lib_CXX11_ABI$1 ./arm_pcie/
  safe_rm arm_pcie/lib_CXX11_ABI$1

  # install mips64 python wheel
  safe_rm mips64/lib_CXX11_ABI$i
  safe_mv mips64/lib_CXX11_ABI$1 ./mips64/
  safe_rm mips64/lib_CXX11_ABI$1

  # install sw64 python wheel
  safe_rm sw64/lib_CXX11_ABI$i
  safe_mv sw64/lib_CXX11_ABI$1 ./sw64/
  safe_rm sw64/lib_CXX11_ABI$1

  # install loongarch64 python wheel
  safe_rm loongarch64/lib_CXX11_ABI$i
  safe_mv loongarch64/lib_CXX11_ABI$1 ./loongarch64/
  safe_rm loongarch64/lib_CXX11_ABI$1
  popd

  pushd $SAIL_TOP/lib/sail
  # install cmodel lib
  #safe_rm  cmodel/lib_CXX11_ABI$noabi
  #safe_mv cmodel/lib_CXX11_ABI$1 ./cmodel/
  #safe_rm  cmodel/lib_CXX11_ABI$1

  # install pcie lib
  safe_rm pcie/lib_CXX11_ABI$i
  safe_mv pcie/lib_CXX11_ABI$1 ./pcie/
  safe_rm pcie/lib_CXX11_ABI$1

  # install asafe_rm_pcie lib
  safe_rm arm_pcie/lib_CXX11_ABI$i
  safe_mv arm_pcie/lib_CXX11_ABI$1 ./arm_pcie/
  safe_rm arm_pcie/lib_CXX11_ABI$1

  # install mips64 python wheel
  safe_rm mips64/lib_CXX11_ABI$i
  safe_mv mips64/lib_CXX11_ABI$1 ./mips64/
  safe_rm mips64/lib_CXX11_ABI$1

  # install sw64 lib
  safe_rm sw64/lib_CXX11_ABI$i
  safe_mv sw64/lib_CXX11_ABI$1 ./sw64/
  safe_rm sw64/lib_CXX11_ABI$1

  # install loongarch64 lib
  safe_rm loongarch64/lib_CXX11_ABI$i
  safe_mv loongarch64/lib_CXX11_ABI$1 ./loongarch64/
  safe_rm loongarch64/lib_CXX11_ABI$1

  popd
}

# get linux info
linux=`sed -n '1p' /etc/issue`

#get gcc version(only for centos)
gcc=`gcc --version | grep gcc`
gcc_version=${gcc:0-9:3}

linux=`echo $linux | sed 's/ //g'`
echo $linux
USING_CXX11_ABI=1
if [ x"$linux" == x"\S" ]; then
  echo "linux is centos"
  if [ $gcc_version == "4.8" ]; then
     echo "gcc version is $gcc_version"
     echo "USING CXX11_ABI=0"
     USING_CXX11_ABI=0
  else
     echo "gcc version is not 4.8"
     echo "USING CXX11_ABI=0"
     USING_CXX11_ABI=0
  fi
else
  echo "linux is $linux"
  echo "USING CXX11_ABI=1"
  USING_CXX11_ABI=1
fi

if [ -n "$1" ]; then
  if [ $1 == "nntc" ]; then
    sail_install $USING_CXX11_ABI
  else
    echo "Usage: ./install_sail.sh nntc"
  fi
else
  sail_install_local ${USING_CXX11_ABI}
fi
