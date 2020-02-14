#!/bin/bash

function install_nnt_lib() {
  echo "============================== install nntlib ============================="
  # install nnt lib
  pushd /workspace/nntoolchain/scripts
  ./install_lib.sh nntc
  popd
}

function create_release_directory() {
  echo "============================== create release folder ============================="
  if [ -d "out" ]  ; then
    rm -rf out
  fi
  mkdir -p out/sophon-inference
  mkdir -p out/sophon-inference/include/sail
  mkdir -p out/sophon-inference/lib/sail/pcie/lib_CXX11_ABI1
  mkdir -p out/sophon-inference/lib/sail/pcie/lib_CXX11_ABI0
  mkdir -p out/sophon-inference/lib/sail/soc
  mkdir -p out/sophon-inference/lib/sail/arm_pcie
  mkdir -p out/sophon-inference/lib/sail/cmodel/lib_CXX11_ABI1
  mkdir -p out/sophon-inference/lib/sail/cmodel/lib_CXX11_ABI0
  mkdir -p out/sophon-inference/python3/pcie/lib_CXX11_ABI1
  mkdir -p out/sophon-inference/python3/pcie/lib_CXX11_ABI0
  mkdir -p out/sophon-inference/python3/soc
  mkdir -p out/sophon-inference/python3/arm_pcie
  mkdir -p out/sophon-inference/python3/cmodel/lib_CXX11_ABI1
  mkdir -p out/sophon-inference/python3/cmodel/lib_CXX11_ABI0
  mkdir -p out/sophon-inference/docs
  mkdir -p out/sophon-inference/scripts
  mkdir -p out/sophon-inference/samples/cpp
  mkdir -p out/sophon-inference/samples/python
  # add test for bm1684
  mkdir -p out/sophon-inference/test
}

function build_sail_pcie_u() {
  # build sail pcie lib
  if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Please excute the command at project root path!"
    exit 1
  fi
  if [ ! -d "build" ]; then
    mkdir build
  else
    rm -rf ./build/*
  fi
  pushd build
  cmake -DUSE_BMCV=ON -DUSE_LOCAL=OFF ..
  echo "============================== build pcie (Ubuntu) ============================="
  make -j
  if [ $? -eq 0 ];then
    echo "pcie(Ubuntu) build finished!"
  else
    echo "pcie(Ubuntu) build error!"
    exit 1
  fi
  popd
  # cp sail pcie lib
  cp ./build/lib/libsail.so ./out/sophon-inference/lib/sail/pcie/lib_CXX11_ABI1
}

function build_sail_python_pcie_u() {
  echo "============================ build pcie (Ubuntu) python ============================="
  # build pcie python wheel
  pushd python/pcie
  ./sophon_pcie_whl.sh
  popd
  # cp pcie python wheel
  cp ./python/pcie/dist/*whl ./out/sophon-inference/python3/pcie/lib_CXX11_ABI1
}

function build_sail_pcie_c() {
  # build sail pcie lib
  if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Please excute the command at project root path!"
    exit 1
  fi
  if [ ! -d "build" ]; then
    mkdir build
  else
    rm -rf ./build/*
  fi
  pushd build
  cmake -DUSE_BMCV=ON -DUSE_LOCAL=OFF -DUSE_CENTOS=ON ..
  echo "============================ build pcie (CentOS) ============================="
  make -j
  if [ $? -eq 0 ];then
    echo "pcie(CentOS) build finished!"
  else
    echo "pcie(CenOS) build error!"
    exit 1
  fi
  popd
  # cp sail pcie lib
  cp ./build/lib/libsail.so ./out/sophon-inference/lib/sail/pcie/lib_CXX11_ABI0
}

function build_sail_python_pcie_c() {
  echo "============================ build pcie (CentOS) python ============================="
  # build pcie python wheel
  pushd python/pcie
  ./sophon_pcie_whl.sh
  popd
  # cp pcie python wheel
  cp ./python/pcie/dist/*whl ./out/sophon-inference/python3/pcie/lib_CXX11_ABI0
}

function build_sail_cmodel_u() {
  # build sail cmodel lib
  if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Please excute the command at project root path!"
    exit 1
  fi
  if [ ! -d "build" ]; then
    mkdir build
  else
    rm -rf ./build/*
  fi
  pushd build
  cmake -DUSE_CMODEL=ON -DUSE_BMCV=OFF -DUSE_FFMPEG=OFF -DUSE_LOCAL=OFF ..
  echo "============================ build cmodel (Ubuntu) ============================="
  make -j
  if [ $? -eq 0 ];then
    echo "cmodel(Ubuntu) build finished!"
  else
    echo "cmodel(Ubuntu) build error!"
    exit 1
  fi
  popd
  # cp sail cmodel lib
  cp ./build/lib/libsail.so ./out/sophon-inference/lib/sail/cmodel/lib_CXX11_ABI1
}

function build_sail_python_cmodel_u() {
  echo "============================ build cmodel (Ubuntu) python ============================="
  # build cmodel python wheel
  pushd python/pcie
  ./sophon_pcie_whl.sh
  popd
  # cp pcie python wheel
  cp ./python/pcie/dist/*whl ./out/sophon-inference/python3/cmodel/lib_CXX11_ABI1
}

function build_sail_cmodel_c() {
  # build sail cmodel lib
  if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Please excute the command at project root path!"
    exit 1
  fi
  if [ ! -d "build" ]; then
    mkdir build
  else
    rm -rf ./build/*
  fi
  pushd build
  cmake -DUSE_CMODEL=ON -DUSE_BMCV=OFF -DUSE_FFMPEG=OFF -DUSE_LOCAL=OFF -DUSE_CENTOS=ON ..
  echo "============================ build cmodel (CentOS) ============================="
  make -j
  if [ $? -eq 0 ];then
    echo "cmodel(CentOS) build finished!"
  else
    echo "cmodel(CentOS) build error!"
    exit 1
  fi
  popd
  # cp sail cmodel lib
  cp ./build/lib/libsail.so ./out/sophon-inference/lib/sail/cmodel/lib_CXX11_ABI0
}

function build_sail_python_cmodel_c() {
  echo "============================ build cmodel (CentOS) python ============================="
  # build cmodel python wheel
  pushd python/pcie
  ./sophon_pcie_whl.sh
  popd
  # cp pcie python wheel
  cp ./python/pcie/dist/*whl ./out/sophon-inference/python3/cmodel/lib_CXX11_ABI0
}

function build_sail_arm_pcie() {
  # build sail arm_pcie lib
  if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Please excute the command at project root path!"
    exit 1
  fi
  if [ ! -d "build" ]; then
    mkdir build
  else
    rm -rf ./build/*
  fi
  pushd build
  cmake -DUSE_FFMPEG=OFF -DUSE_LOCAL=OFF -DUSE_ARM_PCIE=ON -DCMAKE_TOOLCHAIN_FILE=../cmake/BM1684_ARM_PCIE/ToolChain_aarch64_linux.cmake ..
  echo "============================ build arm_pcie ============================="
  make -j
  if [ $? -eq 0 ];then
    echo "arm_pcie build finished!"
  else
    echo "arm_pcie build error!"
    exit 1
  fi
  popd
  # cp sail arm_pcie lib
  cp ./build/lib/libsail.so ./out/sophon-inference/lib/sail/arm_pcie
}

function build_sail_python_arm_pcie() {
  echo "============================ build arm_pcie python ============================="
  # build arm_pcie python wheel
  pushd python/arm_pcie
  ./sophon_arm_pcie_whl.sh
  popd
  # cp pcie python wheel
  cp ./python/arm_pcie/dist/*whl ./out/sophon-inference/python3/arm_pcie
}

function build_sail_soc() {
  # build sail soc lib
  if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Please excute the command at project root path!"
    exit 1
  fi
  if [ ! -d "build" ]; then
    mkdir build
  else
    rm -rf ./build/*
  fi
  pushd build
  cmake -DUSE_BMCV=ON -DUSE_PCIE=OFF -DCMAKE_TOOLCHAIN_FILE=../cmake/BM1684_SOC/ToolChain_aarch64_linux.cmake ..
  echo "============================ build soc ============================="
  make -j
  if [ $? -eq 0 ];then
    echo "soc build finished!"
  else
    echo "soc build error!"
    exit 1
  fi
  popd
  # cp sail soc lib
  cp ./build/lib/libsail.so ./out/sophon-inference/lib/sail/soc
  # cp sa5 test
  cp ./build/bin/cls_resnet_0 ./out/sophon-inference/test/sa5_tests/test_resnet50/soc
  cp ./build/bin/cls_resnet_1 ./out/sophon-inference/test/sa5_tests/test_resnet50/soc
}

function build_sail_python_soc() {
  echo "============================ build soc python ============================="
  # build soc python wheel
  pushd python/soc
  ./sophon_soc_whl.sh
  popd
  # cp soc arm python module
  cp -r ./python/soc/arm/sophon ./out/sophon-inference/python3/soc
  rm ./out/sophon-inference/python3/soc/sophon/auto_runner
  rm ./out/sophon-inference/python3/soc/sophon/algokit
  rm ./out/sophon-inference/python3/soc/sophon/utils
  cp -r ./modules/auto_runner ./out/sophon-inference/python3/soc/sophon
  cp -r ./modules/algokit ./out/sophon-inference/python3/soc/sophon
  cp -r ./modules/utils ./out/sophon-inference/python3/soc/sophon
}

function fill_sail_header() {
  echo "============================ fill header ============================="
  # fill sail header
  cp ./include/*.h ./out/sophon-inference/include/sail
  cp -r ./3rdparty/spdlog ./out/sophon-inference/include/sail
  cp ./3rdparty/inireader* ./out/sophon-inference/include/sail
}

function fill_samples() {
  echo "============================= fill samples ============================"
  # fill samples
  # cpp: det_ssd_cv_bmcv
  cp -r ./release/release_case/cpp_cv_bmcv_sail ./out/sophon-inference/samples/cpp
  cp ./samples/cpp/det_ssd/processor* \
     ./samples/cpp/det_ssd/cvdecoder* \
     ./out/sophon-inference/samples/cpp/cpp_cv_bmcv_sail
  cp ./samples/cpp/det_ssd/det_ssd_4.cpp ./out/sophon-inference/samples/cpp/cpp_cv_bmcv_sail/main.cpp
  # cpp: det_ssd_cv_cvbmcv
  cp -r ./release/release_case/cpp_cv_cv+bmcv_sail ./out/sophon-inference/samples/cpp
  cp ./samples/cpp/det_ssd/processor* \
     ./samples/cpp/det_ssd/cvdecoder* \
     ./out/sophon-inference/samples/cpp/cpp_cv_cv+bmcv_sail
  cp ./samples/cpp/det_ssd/det_ssd_3.cpp ./out/sophon-inference/samples/cpp/cpp_cv_cv+bmcv_sail/main.cpp
  # cpp: det_ssd_ffmpeg_bmcv
  cp -r ./release/release_case/cpp_ffmpeg_bmcv_sail ./out/sophon-inference/samples/cpp
  cp ./samples/cpp/det_ssd/processor* \
     ./samples/cpp/det_ssd/cvdecoder* \
     ./out/sophon-inference/samples/cpp/cpp_ffmpeg_bmcv_sail
  cp ./samples/cpp/det_ssd/det_ssd_1.cpp ./out/sophon-inference/samples/cpp/cpp_ffmpeg_bmcv_sail/main.cpp

  # python: det_ssd_bmcv
  cp -r ./release/release_case/py_ffmpeg_bmcv_sail ./out/sophon-inference/samples/python/det_ssd_bmcv
  cp -r ./samples/python/det_ssd/det_ssd_1.py ./out/sophon-inference/samples/python/det_ssd_bmcv/det_ssd_bmcv.py
  cp -r ./samples/python/det_ssd/det_ssd_2.py ./out/sophon-inference/samples/python/det_ssd_bmcv/det_ssd_bmcv_4b.py
}

function fill_sa5_tests() {
  echo "=========================== fill sa5_tests ============================="
  # cp sa5_tests
  cp -r ./release/qa_test/sa5_tests ./out/sophon-inference/test
  # cp test
  cp ./build/bin/cls_resnet_0 ./out/sophon-inference/test/sa5_tests/test_resnet50/x86
  cp ./build/bin/cls_resnet_1 ./out/sophon-inference/test/sa5_tests/test_resnet50/x86
  cp ./samples/python/det_ssd/det_ssd_1.py ./out/sophon-inference/test/sa5_tests/test_ssd
  cp ./samples/python/det_ssd/det_ssd_2.py ./out/sophon-inference/test/sa5_tests/test_ssd
}

function fill_sc5_tests() {
  echo "============================ fill sc5_tests ============================"
  cp -r ./release/qa_test/sc5_tests ./out/sophon-inference/test
  # copy download.py
  cp ./tools/download.py ./out/sophon-inference/test/sc5_tests
  # python samples
  for i in cls_resnet det_ssd det_yolov3 det_mtcnn;
  do
    cp -rf ./samples/python/${i}/* ./out/sophon-inference/test/sc5_tests/python/${i}
  done
  # cpp samples
  for i in cls_resnet det_yolov3 det_mtcnn;
  do
    cp -f ./samples/cpp/${i}/*.cpp ./out/sophon-inference/test/sc5_tests/cpp/${i};
    cp -f ./samples/cpp/${i}/*.h ./out/sophon-inference/test/sc5_tests/cpp/${i};
    cp -f ./samples/cpp/${i}/*.md ./out/sophon-inference/test/sc5_tests/cpp/${i};
  done
  cp -f ./samples/cpp/det_ssd/*.h ./out/sophon-inference/test/sc5_tests/cpp/det_ssd
  cp -f ./samples/cpp/det_ssd/cvdecoder.cpp ./out/sophon-inference/test/sc5_tests/cpp/det_ssd
  cp -f ./samples/cpp/det_ssd/processor.cpp ./out/sophon-inference/test/sc5_tests/cpp/det_ssd
  cp -f ./samples/cpp/det_ssd/*.md ./out/sophon-inference/test/sc5_tests/cpp/det_ssd
  cp -f ./samples/cpp/det_ssd/det_ssd_0.cpp ./out/sophon-inference/test/sc5_tests/cpp/det_ssd
  cp -f ./samples/cpp/det_ssd/det_ssd_1.cpp ./out/sophon-inference/test/sc5_tests/cpp/det_ssd
  cp -f ./samples/cpp/det_ssd/det_ssd_2.cpp ./out/sophon-inference/test/sc5_tests/cpp/det_ssd
}

function fill_docs() {
  echo "============================ fill documentation ======================"
  # fill release README.md
  cp ./release/README.md ./out/sophon-inference
  # fill sophon-inference document
  cp ./docs/Sophon_Inference_zh.pdf ./out/sophon-inference/docs
}

function fill_scripts() {
  echo "============================= fill scripts ==========================="
  cp ./release/install_sail.sh ./out/sophon-inference/scripts
}

function release_bm1684_init() {
  # release init
  # install_nnt_lib
  create_release_directory
}

function release_bm1684() {
  if [ -z "${mode}" ];then
    echo "Need to select a release mode all(default) | soc | pcie | cmodel | arm_pcie"
    mode="all"
  fi
  echo "Release mode: ${mode}"
  release_bm1684_init
  if [ "${mode}" == "all" ]; then
    # pcie (ubuntu/centos)
    build_sail_pcie_u
    fill_sa5_tests
    build_sail_python_pcie_u
    build_sail_pcie_c
    build_sail_python_pcie_c
    # arm_pcie
    build_sail_arm_pcie
    build_sail_python_arm_pcie
    # soc
    build_sail_soc
    build_sail_python_soc
    # cmodel (ubuntu/centos)
    build_sail_cmodel_u
    build_sail_python_cmodel_u
    build_sail_cmodel_c
    build_sail_python_cmodel_c
  elif [ "${mode}" == "pcie" ]; then
    build_sail_pcie_u
    fill_sa5_tests
    build_sail_python_pcie_u
    build_sail_pcie_c
    build_sail_python_pcie_c
  elif [ "${mode}" == "soc" ]; then
    build_sail_soc
    build_sail_python_soc
  elif [ "${mode}" == "cmodel" ]; then
    build_sail_cmodel_u
    build_sail_python_cmodel_u
    build_sail_cmodel_c
    build_sail_python_cmodel_c
  elif [ "${mode}" == "arm_pcie" ]; then
    build_sail_arm_pcie
    build_sail_python_arm_pcie
  else
    echo "${mode} is not a valid mode!"
    exit 1
  fi
  fill_sail_header
  fill_samples
  fill_sc5_tests
  fill_scripts
  fill_docs
}

mode=$1
release_bm1684
