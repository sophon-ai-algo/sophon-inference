@echo off

:: wrap pcie new version
python setup.py bdist_wheel
if %errorlevel% equ 1 (
  echo "Failed to build sophon wheel"
  exit  /B 1
)

echo "---- setup sophon wheel"
rd /q /s .\sophon.egg-info
rd /q /s .\build

