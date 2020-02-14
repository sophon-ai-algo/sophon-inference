#!/bin/bash

# soft link sail
echo "link sail python lib"

pushd ./sophon

if [ -f "../../build/lib/sail.cpython-35m-x86_64-linux-gnu.so" ];then
    rm -f ./sail.cpython-35m-x86_64-linux-gnu.so
    cp ../../build/lib/sail.cpython-35m-x86_64-linux-gnu.so ./sail.cpython-35m-x86_64-linux-gnu.so
fi
if [ -f "../../build/lib/sail.cpython-36m-x86_64-linux-gnu.so" ];then
    rm -f ./sail.cpython-36m-x86_64-linux-gnu.so
    cp ../../build/lib/sail.cpython-36m-x86_64-linux-gnu.so ./sail.cpython-36m-x86_64-linux-gnu.so
fi

if [ -f "../../build/lib/sail.cpython-35m-aarch64-linux-gnu.so" ];then
    rm -f ./sail.cpython-35m-aarch64-linux-gnu.so
    cp ../../build/lib/sail.cpython-35m-aarch64-linux-gnu.so ./sail.cpython-35m-aarch64-linux-gnu.so
fi

popd

# add autodeploy && algokit
export PYTHONPATH=$PYTHONPATH:${PWD}:${PWD}/sophon
