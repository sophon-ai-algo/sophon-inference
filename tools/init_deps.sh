#!/bin/bash
set +e
function gettop()
{
    echo $(dirname $(readlink -f ${BASH_SOURCE[0]}))
}

GIT_SERVER=172.28.141.75
VERSION=2.6.0

TOP=$(gettop)
echo top=$TOP
if [ -n "$1" ]; then
   NAME=$1
else
   NAME=$(curl ftp://AI:dailybuild@${GIT_SERVER}/all_in_one/release_build/ | sort | tail -n 1 | awk '{print $9}')
fi

if [ -n "$2" ]; then
    VERSION=$2
fi

echo "the latest bmsdk release time is " ${NAME}
echo "the latest bmsdk VERSION is " ${VERSION}
touch last_version

last_time=`cat last_version`
deps_modules=(nntoolchain/nntoolchain-bm1684-all-$VERSION.tar.gz \
	      hd_server/middleware-soc_bm1684_v$VERSION.tgz \
	      hd_server/middleware-pcie_bm1684_v$VERSION.tgz \
	      hd_server/middleware-pcie_arm64_bm1684_v$VERSION.tgz \
	      hd_server/middleware-pcie_loongarch64_bm1684_v$VERSION.tgz \
	      hd_server/middleware-pcie_sw64_bm1684_v$VERSION.tgz \
	      hd_server/middleware-pcie_mips64_bm1684_v$VERSION.tgz \
	      hd_server/bm168x_bm1684_v$VERSION.tgz)

rm *.tgz *.tar.gz
rm -fr pcie_* soc_* nntoolchain* bm168x

for m in ${deps_modules[@]}
do
	url=ftp://QA:dailybuild@${GIT_SERVER}/all_in_one/release_build/${NAME}/${m}
	wget $url
        tar_file=${url##*/}
	bn=$(basename $tar_file .tar.gz)
	echo "bn=$bn"
	if [[ "$bn" =~ "nntoolchain" ]]; then
		tar zxf $tar_file
		ln -s $bn nntoolchain
	elif [[ "$bn" =~ "bm168x" ]]; then
		tar zxf $tar_file
	        ln -s release bm168x
	else
		tar zxf $tar_file
        fi	       
done
	      
rm *.tgz *.tar.gz

