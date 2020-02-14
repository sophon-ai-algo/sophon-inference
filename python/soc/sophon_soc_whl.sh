#!/bin/bash

# wrap x86 new version
pushd ./x86
python3 setup_x86.py bdist_wheel
if [[ $? != 0 ]];then
  echo "Failed to build sophon wheel"
  exit 1
fi
echo "---- setup sophon x86 wheel"
# rm intermediate file
rm -rf ./sophon_x86.egg-info ./build
popd

# wrap arm new version
pushd ./arm
python3 setup_arm.py bdist_wheel
if [[ $? != 0 ]];then
  echo "Failed to build sophon wheel"
  exit 1
fi
echo "---- setup sophon arm wheel"
# rm intermediate file
rm -rf ./sophon_arm.egg-info ./build
popd
