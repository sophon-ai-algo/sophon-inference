#if [ $# != 2 ]; then
#    echo "$0 <git top dir>"
#    exit 1
#fi

function gettop()
{
    echo $(dirname $(readlink -f ${BASH_SOURCE[0]}))
}

TOOLS=$(gettop)
TOP=$TOOLS/..
TOOLS=$TOOLS

echo "deps_top=$TOOLS"

docker run -it --rm \
       -v $TOP:/workspace/sophon-inference \
       -v $TOOLS/release:/workspace/bm168x \
       -v $TOOLS/soc_bm1684_asic:/workspace/soc_bm1684_asic \
       -v $TOOLS/pcie_bm1684_asic:/workspace/pcie_bm1684_asic \
       -v $TOOLS/pcie_sw64_bm1684_asic:/workspace/pcie_sw64_bm1684_asic \
       -v $TOOLS/pcie_loongarch64_bm1684_asic:/workspace/pcie_loongarch64_bm1684_asic \
       -v $TOOLS/pcie_mips64_bm1684_asic:/workspace/pcie_mips64_bm1684_asic \
       -v $TOOLS/pcie_arm64_bm1684_asic:/workspace/pcie_arm64_bm1684_asic \
       -v $TOOLS/nntoolchain:/workspace/nntoolchain \
       -v $TOP/../bm_prebuilt_toolchains/pythons:/workspace/pythons \
       -v $TOP/../bm_prebuilt_toolchains:/workspace/bm_prebuilt_toolchains \
       -v $TOP/../bm_prebuilt_toolchains/swgcc830_cross_tools-bb623bc9a:/usr/sw/swgcc830_cross_tools-bb623bc9a \
       --workdir=/workspace/sophon-inference \
       bmnnsdk2-bm1684/dev:ubuntu18.04 \
       bash -c "./tools/release_bm1684.sh $1"

