#! /usr/bin
#navigate to script path
DIRNAME=$(cd $(dirname $0);pwd)
cd $DIRNAME

NAME=$(curl ftp://AI:dailybuild@10.30.40.51/all_in_one/release_build/ | sort | tail -n 1 | awk '{print $9}')
echo "the latest bmsdk release time is " ${NAME}
touch last_version

last_time=`cat last_version`
if [[ "${last_time}" != ${NAME} ]];then
   rm ~/bm-driver/*.tar.gz
   echo "download the latest bmsdk version"
   TAR_FILE=$(curl ftp://AI:dailybuild@10.30.40.51/all_in_one/release_build/${NAME}/bmnnsdk2/ |grep tar |awk '{print $NF}')
   wget ftp://AI:dailybuild@10.30.40.51/all_in_one/release_build/${NAME}/bmnnsdk2/${TAR_FILE}
   tar -zvxf ${TAR_FILE}
   TAR_DIR_NAME="${TAR_FILE/.tar.gz}"
   if [[ -d release ]]; then
     rm -r release
   fi
   if [[ -f release ]]; then
     rm -r release
   fi
   mv $TAR_DIR_NAME release

   # install driver
   pushd release 
   cd ./test && sudo ./install_driver_pcie.sh
   popd
   # update lib
   pushd release 
   cd ./scripts && ./install_lib.sh nntc
   popd
   # set env variable
   pushd release 
   cd ./test && source envsetup_pcie.sh
   popd
else
   echo "already the latest"
#   cd release/test && source envsetup_pcie.sh
fi
echo $NAME >last_version
