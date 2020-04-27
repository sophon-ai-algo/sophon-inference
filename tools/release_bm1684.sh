#!/bin/bash

function create_release_directory() {
  echo "------------------------------ create release folder ------------------------------"
  if [ -d "out" ]  ; then
    rm -rf out
  fi
  mkdir -p out/sophon-inference
  mkdir -p out/sophon-inference/include/sail
  mkdir -p out/sophon-inference/lib/sail/pcie/lib_CXX11_ABI1
  mkdir -p out/sophon-inference/lib/sail/pcie/lib_CXX11_ABI0
  mkdir -p out/sophon-inference/lib/sail/soc
  mkdir -p out/sophon-inference/lib/sail/arm_pcie/lib_CXX11_ABI1
  mkdir -p out/sophon-inference/lib/sail/arm_pcie/lib_CXX11_ABI0
  mkdir -p out/sophon-inference/lib/sail/cmodel/lib_CXX11_ABI1
  mkdir -p out/sophon-inference/lib/sail/cmodel/lib_CXX11_ABI0
  mkdir -p out/sophon-inference/python3/pcie/lib_CXX11_ABI1
  mkdir -p out/sophon-inference/python3/pcie/lib_CXX11_ABI0
  mkdir -p out/sophon-inference/python3/soc
  mkdir -p out/sophon-inference/python3/arm_pcie/lib_CXX11_ABI1
  mkdir -p out/sophon-inference/python3/arm_pcie/lib_CXX11_ABI0
  mkdir -p out/sophon-inference/python3/cmodel/lib_CXX11_ABI1
  mkdir -p out/sophon-inference/python3/cmodel/lib_CXX11_ABI0
  mkdir -p out/sophon-inference/docs
  mkdir -p out/sophon-inference/scripts
  mkdir -p out/sophon-inference/samples/cpp
  mkdir -p out/sophon-inference/samples/python
  # add test for bm1684
  mkdir -p out/sophon-inference/test
}

function build_lib {
  echo "------------------------------ build_lib: $* ------------------------------"
  local pcie_or_soc=""
  local cpp_or_py="all"
  local x86_or_arm_or_cmodel=""
  local ubuntu_or_centos=""
  local py_version_long=""
  local py_version_middle=""
  local py_version_short=""
  if [[ $# -ge 1 ]]; then
    pcie_or_soc=$1
    if [[ $# -ge 4 ]]; then
      cpp_or_py=$2
      x86_or_arm_or_cmodel=$3
      ubuntu_or_centos=$4
      if [[ $# -eq 5 ]]; then
        py_version_long=$5
        local array=(${py_version_long//./ })
        py_version_middle=${array[0]}.${array[1]}
        py_version_short=py${array[0]}${array[1]}
      fi
    fi
  fi

  # build sail lib
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

  # cmake config
  local lib_path=""
  local py_keyword=""
  local cmake_params=""
  if [[ "$pcie_or_soc" == "soc" ]]; then
    lib_path="soc"
    py_keyword="soc"
    cmake_params="-DUSE_BMCV=ON -DUSE_PCIE=OFF -DCMAKE_TOOLCHAIN_FILE=../cmake/BM1684_SOC/ToolChain_aarch64_linux.cmake"
  elif [[ "$pcie_or_soc" == "pcie" ]]; then
    if [[ "$x86_or_arm_or_cmodel" == "arm" ]]; then
      py_keyword="arm_pcie"
      cmake_params="-DUSE_BMCV=ON -DUSE_ARM_PCIE=ON -DCMAKE_TOOLCHAIN_FILE=../cmake/BM1684_ARM_PCIE/ToolChain_aarch64_linux.cmake"
      if [[ "$ubuntu_or_centos" == "ubuntu" ]]; then
        lib_path="arm_pcie/lib_CXX11_ABI1"
      elif [[ "$ubuntu_or_centos" == "centos" ]]; then
        lib_path="arm_pcie/lib_CXX11_ABI0"
        cmake_params="$cmake_params -DUSE_CENTOS=ON"
      else
        echo "Invalid parameter ubuntu_or_centos: $ubuntu_or_centos"
        exit 1
      fi
    elif [[ "$x86_or_arm_or_cmodel" == "x86" ]]; then
      py_keyword="pcie"
      if [[ "$ubuntu_or_centos" == "ubuntu" ]]; then
        lib_path="pcie/lib_CXX11_ABI1"
        cmake_params="-DUSE_BMCV=ON"
      elif [[ "$ubuntu_or_centos" == "centos" ]]; then
        lib_path="pcie/lib_CXX11_ABI0"
        cmake_params="-DUSE_BMCV=ON -DUSE_CENTOS=ON"
      else
        echo "Invalid parameter ubuntu_or_centos: $ubuntu_or_centos"
        exit 1
      fi
    elif [[ "$x86_or_arm_or_cmodel" == "cmodel" ]]; then
      py_keyword="pcie"
      if [[ "$ubuntu_or_centos" == "ubuntu" ]]; then
        lib_path="cmodel/lib_CXX11_ABI1"
        cmake_params="-DUSE_CMODEL=ON -DUSE_BMCV=OFF -DUSE_FFMPEG=OFF"
      elif [[ "$ubuntu_or_centos" == "centos" ]]; then
        lib_path="cmodel/lib_CXX11_ABI0"
        cmake_params="-DUSE_CMODEL=ON -DUSE_CENTOS=ON -DUSE_BMCV=OFF -DUSE_FFMPEG=OFF"
      else
        echo "Invalid parameter ubuntu_or_centos: $ubuntu_or_centos"
        exit 1
      fi
    else
      echo "Invalid parameter x86_or_arm_or_cmodel: $x86_or_arm_or_cmodel"
      exit 1
    fi
    if [[ ("$cpp_or_py" == "all" || "$cpp_or_py" == "py") && -n "$py_version_short" ]]; then
      local py_path=/workspace/pythons/Python-$py_version_long/python_$py_version_long
      if [[ -x $py_path ]]; then
        local py_bin_path=$py_path/bin/python$py_version_middle
        export LD_LIBRARY_PATH=$py_path/lib
        $py_bin_path -V
        if [[ $? == 0 ]]; then
          cmake_params="$cmake_params -DPYTHON_EXECUTABLE=$py_bin_path -DCUSTOM_PY_LIBDIR=$py_path/lib"
        fi
      else
        echo "Error: file not exist: $py_path/bin/python$py_version_middle"
        py_version_short=""
      fi
    fi
  else
    echo "Invalid parameter pcie_or_soc: $pcie_or_soc"
    exit 1
  fi
  cmake -DUSE_LOCAL=OFF $cmake_params ..

  # make targets
  if [[ "$cpp_or_py" == "all" ]]; then
    make -j
  elif [[ "$cpp_or_py" == "cpp" ]]; then
    make cpp -j
  elif [[ "$cpp_or_py" == "py" ]]; then
    make pysail -j
  else
    echo "Invalid parameter x86_or_arm_or_cmodel: $cpp_or_py"
    exit 1
  fi
  if [ $? -eq 0 ];then
    echo "Build succeed!"
  else
    echo "Build Failed!"
    exit 1
  fi
  popd

  # collect cpp lib
  if [[ "$cpp_or_py" == "all" || "$cpp_or_py" == "cpp" ]]; then
    cp ./build/lib/libsail.so ./out/sophon-inference/lib/sail/$lib_path
  fi

  # collect python wheel
  if [[ "$cpp_or_py" == "all" || "$cpp_or_py" == "py" ]]; then
    # build python wheel
    pushd python/$py_keyword
    ./sophon_${py_keyword}_whl.sh
    popd
    if [[ "$pcie_or_soc" == "soc" ]]; then
      cp -r ./python/soc/arm/sophon ./out/sophon-inference/python3/soc
      rm ./out/sophon-inference/python3/soc/sophon/auto_runner
      rm ./out/sophon-inference/python3/soc/sophon/algokit
      rm ./out/sophon-inference/python3/soc/sophon/utils
      cp -r ./modules/auto_runner ./out/sophon-inference/python3/soc/sophon
      cp -r ./modules/algokit ./out/sophon-inference/python3/soc/sophon
      cp -r ./modules/utils ./out/sophon-inference/python3/soc/sophon
    else
      # cp python wheel
      local py_lib_dir="./out/sophon-inference/python3/$lib_path/$py_version_short"
      if [[ ! -d $py_lib_dir ]]; then
        mkdir -p $py_lib_dir
      fi
      cp ./python/$py_keyword/dist/*whl $py_lib_dir
    fi
  fi
}

function fill_headers() {
  echo "------------------------------ fill headers ------------------------------"
  # fill sail header
  cp ./include/*.h ./out/sophon-inference/include/sail
  cp -r ./3rdparty/spdlog ./out/sophon-inference/include/sail
  cp ./3rdparty/inireader* ./out/sophon-inference/include/sail
}

function fill_samples() {
  echo "------------------------------ fill samples ------------------------------"
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

function fill_se5_tests() {
  echo "------------------------------ fill se5_tests ------------------------------"
  # cp se5_tests
  cp -r ./release/qa_test/se5_tests ./out/sophon-inference/test
  # copy download.py
  cp ./tools/download.py ./out/sophon-inference/test/se5_tests/scripts
  # cp test
  cp ./build/bin/cls_resnet_0 ./out/sophon-inference/test/se5_tests/cls_resnet
  cp ./build/bin/cls_resnet_1 ./out/sophon-inference/test/se5_tests/cls_resnet
  cp ./samples/python/det_ssd/det_ssd_1.py ./out/sophon-inference/test/se5_tests/det_ssd
  cp ./samples/python/det_ssd/det_ssd_2.py ./out/sophon-inference/test/se5_tests/det_ssd
}

function fill_sc5_tests() {
  echo "------------------------------ fill sc5_tests ------------------------------"
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

function fill_document() {
  echo "------------------------------ fill document ------------------------------"
  # fill release README.md
  cp ./release/README.md ./out/sophon-inference
  # fill sophon-inference document
  cp ./docs/Sophon_Inference_zh.pdf ./out/sophon-inference/docs
}

function fill_scripts() {
  echo "------------------------------ fill scripts ------------------------------"
  cp ./release/install_sail.sh ./out/sophon-inference/scripts
}

function release_bm1684_init() {
  # release init
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
    build_lib pcie all x86 ubuntu 3.5.9
    build_lib pcie py x86 ubuntu 3.6.5
    build_lib pcie py x86 ubuntu 3.7.3
    build_lib pcie py x86 ubuntu 3.8.2
    build_lib pcie all x86 centos 3.5.9
    build_lib pcie py x86 centos 3.6.5
    build_lib pcie py x86 centos 3.7.3
    build_lib pcie py x86 centos 3.8.2
    # arm_pcie
    build_lib pcie all arm ubuntu 3.5.9
    build_lib pcie py arm ubuntu 3.6.5
    build_lib pcie py arm ubuntu 3.7.3
    build_lib pcie py arm ubuntu 3.8.2
    build_lib pcie all arm centos 3.5.9
    build_lib pcie py arm centos 3.6.5
    build_lib pcie py arm centos 3.7.3
    build_lib pcie py arm centos 3.8.2
    # soc
    build_lib soc
    fill_se5_tests
    # cmodel (ubuntu/centos)
    build_lib pcie all cmodel ubuntu
    build_lib pcie all cmodel centos
  elif [ "${mode}" == "pcie" ]; then
    build_lib pcie all x86 ubuntu 3.5.9
    build_lib pcie py x86 ubuntu 3.6.5
    build_lib pcie py x86 ubuntu 3.7.3
    build_lib pcie py x86 ubuntu 3.8.2
    build_lib pcie all x86 centos 3.5.9
    build_lib pcie py x86 centos 3.6.5
    build_lib pcie py x86 centos 3.7.3
    build_lib pcie py x86 centos 3.8.2
  elif [ "${mode}" == "soc" ]; then
    build_lib soc
    fill_se5_tests
  elif [ "${mode}" == "cmodel" ]; then
    build_lib pcie all cmodel ubuntu
    build_lib pcie all cmodel centos
  elif [ "${mode}" == "arm_pcie" ]; then
    build_lib pcie all arm ubuntu 3.5.9
    build_lib pcie py arm ubuntu 3.6.5
    build_lib pcie py arm ubuntu 3.7.3
    build_lib pcie py arm ubuntu 3.8.2
    build_lib pcie all arm centos 3.5.9
    build_lib pcie py arm centos 3.6.5
    build_lib pcie py arm centos 3.7.3
    build_lib pcie py arm centos 3.8.2
  else
    echo "${mode} is not a valid mode!"
    exit 1
  fi
  fill_headers
  fill_samples
  fill_sc5_tests
  fill_scripts
  fill_document
}

mode=$1
release_bm1684
