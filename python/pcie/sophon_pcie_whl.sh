#!/bin/bash

# wrap pcie new version
python3 setup.py bdist_wheel
if [[ $? != 0 ]];then
  echo "Failed to build sophon wheel"
  exit 1
fi
echo "---- setup sophon wheel"
rm -rf ./sophon.egg-info ./build

