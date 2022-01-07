
echo "install sophon python library (default: python3)"

:: remove old version
pip3 uninstall sophon -y
echo "---- uninstall old sophon if exists"
# wrap new version
python setup.py bdist_wheel
if %errorlevel% equ 1 (
  echo "Failed to build sophon wheel"
  exit 1
)

echo "---- setup sophon wheel"
:: install new version
pip3 install ./dist/sophon-master-py3-none-any.whl --user
echo "---- install sophon"
:: rm intermediate file
rd /q /s ./sophon.egg-info ./build

