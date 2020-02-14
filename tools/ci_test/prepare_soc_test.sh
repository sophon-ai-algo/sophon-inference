#!/bin/bash

SOC_DATA_DIR=/workspace/soc_data

function clean_data() {
  if [ -d $SOC_DATA_DIR ]; then
    rm -rf  $SOC_DATA_DIR/*
  else
    mkdir $SOC_DATA_DIR
  fi
}

function fill_data() {
  pushd $SOC_DATA_DIR
  mkdir -p cpp/lib cpp/bin python3/samples
  popd
  cp ./build/lib/libsail.so $SOC_DATA_DIR/cpp/lib
  cp ./build/bin/* $SOC_DATA_DIR/cpp/bin
  cp -r ./out/sophon-inference/python3/soc/sophon $SOC_DATA_DIR/python3
  cp -r ./samples/python/* $SOC_DATA_DIR/python3/samples
  cp ./tools/ci_test/test_all_soc.sh $SOC_DATA_DIR
}

clean_data
fill_data
