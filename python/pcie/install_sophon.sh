#!/bin/bash

echo "install sophon python library (default: python3)"

# rm old version
pip3 uninstall sophon -y
echo "---- uninstall old sophon if exists"
# wrap new version
python3 setup.py bdist_wheel
if [[ $? != 0 ]];then
  echo "Failed to build sophon wheel"
  exit 1
fi
echo "---- setup sophon wheel"
# install new version
pip3 install ./dist/sophon-2.0.2-py3-none-any.whl --user
echo "---- install sophon"
# rm intermediate file
rm -rf ./sophon.egg-info ./build

