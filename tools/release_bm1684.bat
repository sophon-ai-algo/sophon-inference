@echo off
set build_path=%cd%
set compile_type="Release|x64"
set runtime_lib=MT
set PYTHONS_PATH=C:\Pythons
set SYS_PATH=%PATH%

set TARGET=winx64

::debug
rem call :create_release_dirs
rem call :fill_samples
rem exit /B 0
::debug

call :get_gitversion
echo err=%errorlevel%
if %errorlevel% NEQ 0 (
  echo "get_gitversion failed!"
  exit /B 1
) else (
  echo "get_gitversion success!"
)

call :create_release_dirs
if %errorlevel% NEQ 0 (
  echo "create release directories failed!"
  exit /B 1
) else (
  echo "create release directories success!"
)

call :build_lib 36
if %errorlevel% NEQ 0 (
  echo "build_lib py36 failed!"
  exit /B 1
) else (
  echo "build_lib py36 success!"
)

call :build_lib 38
if %errorlevel% NEQ 0 (
  echo "build_lib py38 failed!"
  exit /B 1
) else (
  echo "build_lib py38 success!"
)

call :fill_headers
if %errorlevel% NEQ 0 (
  echo "fill_headers failed!"
  exit /B 1
) else (
  echo "fill_headers success!"
)

call :fill_headers
call :fill_samples
call :fill_sc5_tests
call :fill_document

:: exit this program
exit /B 0


:get_gitversion
for /F "tokens=*" %%i in ('git branch') do (set gitbranch=%%i)
set /p=%gitbranch:~2%<nul > git_version
set errorlevel=0
goto:eof

:create_release_dirs
echo "------------------------------ create release folder ------------------------------"

if exist "out" (
     rd /Q /S out
)

mkdir out\sophon-inference
mkdir out\sophon-inference\include\sail
mkdir out\sophon-inference\lib\sail\%TARGET%
mkdir out\sophon-inference\python3\%TARGET%
mkdir out\sophon-inference\docs
mkdir out\sophon-inference\scripts
mkdir out\sophon-inference\samples\cpp
mkdir out\sophon-inference\samples\python
::add test for bm1684
mkdir out\sophon-inference\test

goto:EOF

:build_lib
echo "build_lib: <python version>"
set py_version=%1
if "%py_version%" == "" (
   py_version=38
)

set py_path=%PYTHONS_PATH%\Python%py_version%
set py_bin_path=%py_path%
set PATH=%py_bin_path%;%SYS_PATH%
echo path=%PATH%

if not exist "CMakeLists.txt" (
    echo "ERROR: Please execute the command at project root path!"
	exit /B 1
)

if exist "build_winx64" (
    rd /q /s build_winx64
)

set cmake_params="-DPYTHON_EXECUTABLE=$py_bin_path -DCUSTOM_PY_LIBDIR=$py_path/libs"

echo release
cmake . -B .\build_winx64 -G "Visual Studio 16 2019" -DTARGET_TYPE=release -DRUNTIME_LIB=%runtime_type%
if %errorlevel% NEQ 0 (
  echo "cmake failed!"
  exit /B 1
) else (
  echo "cmake success!"
)

devenv %build_path%\build_winx64\sophon-inference.sln /Build %compile_type%
if %errorlevel% NEQ 0 (
  echo "build failed!"
  exit /B 1
) else (
  echo "build success!"
)

copy /Y build_winx64\Release\libsail.lib out\sophon-inference\lib\sail\%TARGET%\
copy /Y build_winx64\Release\libsail_dll.lib out\sophon-inference\lib\sail\%TARGET%\
copy /Y build_winx64\Release\libsail_dll.dll out\sophon-inference\lib\sail\%TARGET%\
cd python\%TARGET%
call sophon_pcie_whl.bat
if %errorlevel% neq 0 (
   echo "build python wheel failed!"
   exit /B 1
)
cd ..\..

mkdir .\out\sophon-inference\python3\%TARGET%\%py_version%
copy /Y .\python\%TARGET%\dist\*whl .\out\sophon-inference\python3\%TARGET%\%py_version%\
if %errorlevel% neq 0 (
   echo "copy python wheel failed!"
   exit /B 1
)

goto:eof

:fill_headers
echo "------------- fill headers -------------"
copy /Y .\include\*.h out\sophon-inference\include\sail\
copy /Y .\3rdparty\spdlog .\out\sophon-inference\include\sail\
copy /Y .\3rdparty\inireader* .\out\sophon-inference\include\sail\
goto:eof

:fill_samples
  echo "------------------------------ fill samples ------------------------------"
  ::fill samples
  ::cpp: det_ssd_cv_bmcv
  mkdir .\out\sophon-inference\samples\cpp\cpp_cv_bmcv_sail
  xcopy /E /Y .\release\release_case\cpp_cv_bmcv_sail .\out\sophon-inference\samples\cpp\cpp_cv_bmcv_sail
  copy /Y .\samples\cpp\det_ssd\processor* .\out\sophon-inference\samples\cpp\cpp_cv_bmcv_sail\
  copy /Y .\samples\cpp\det_ssd\cvdecoder* .\out\sophon-inference\samples\cpp\cpp_cv_bmcv_sail\
  copy /Y  .\samples\cpp\det_ssd\det_ssd_4.cpp .\out\sophon-inference\samples\cpp\cpp_cv_bmcv_sail\main.cpp
  
  ::cpp: det_ssd_cv_cvbmcv
  mkdir ".\out\sophon-inference\samples\cpp\cpp_cv_cv+bmcv_sail"
  xcopy /Y /E  ".\release\release_case\cpp_cv_cv+bmcv_sail" ".\out\sophon-inference\samples\cpp\cpp_cv_cv+bmcv_sail"
  copy /Y  .\samples\cpp\det_ssd\processor* "out\sophon-inference\samples\cpp\cpp_cv_cv+bmcv_sail\"
  copy /Y .\samples\cpp\det_ssd\cvdecoder* ".\out\sophon-inference\samples\cpp\cpp_cv_cv+bmcv_sail\"
  copy /Y  .\samples\cpp\det_ssd\det_ssd_3.cpp ".\out\sophon-inference\samples\cpp\cpp_cv_cv+bmcv_sail\main.cpp"
  ::cpp: det_ssd_ffmpeg_bmcv
  mkdir ".\out\sophon-inference\samples\cpp\cpp_ffmpeg_bmcv_sail"
  xcopy /Y /E .\release\release_case\cpp_ffmpeg_bmcv_sail .\out\sophon-inference\samples\cpp\cpp_ffmpeg_bmcv_sail
  copy /Y  .\samples\cpp\det_ssd\processor* + .\samples\cpp\det_ssd\cvdecoder* .\out\sophon-inference\samples\cpp\cpp_ffmpeg_bmcv_sail\
  copy /Y  .\samples\cpp\det_ssd\det_ssd_1.cpp .\out\sophon-inference\samples\cpp\cpp_ffmpeg_bmcv_sail\main.cpp

  ::python: det_ssd_bmcv
  mkdir ".\out\sophon-inference\samples\python\det_ssd_bmcv"
  xcopy /Y /E  .\release\release_case\py_ffmpeg_bmcv_sail .\out\sophon-inference\samples\python\det_ssd_bmcv\
  copy /Y  .\samples\python\det_ssd\det_ssd_1.py .\out\sophon-inference\samples\python\det_ssd_bmcv\det_ssd_bmcv.py
  copy /Y  .\samples\python\det_ssd\det_ssd_2.py .\out\sophon-inference\samples\python\det_ssd_bmcv\det_ssd_bmcv_4b.py
goto:eof

:fill_sc5_tests
  echo "------------------------------ fill sc5_tests ------------------------------"
  mkdir .\out\sophon-inference\test\sc5_tests
  xcopy /Y /E .\release\qa_test\sc5_tests .\out\sophon-inference\test\sc5_tests
  ::copy download.py
  copy /Y  .\tools\download.py .\out\sophon-inference\test\sc5_tests
  ::python samples
  for %%i in (cls_resnet,det_ssd,det_yolov3,det_mtcnn) do (
      xcopy /Y /E .\samples\python\%%i .\out\sophon-inference\test\sc5_tests\python\%%i
  )
  ::cpp samples
  for %%i in (cls_resnet,det_yolov3,det_mtcnn) do (
    copy /Y .\samples/cpp\%%i\*.cpp .\out\sophon-inference\test\sc5_tests\cpp\%%i
    copy /Y .\samples\cpp\%%i\*.h .\out\sophon-inference\test\sc5_tests\cpp\%%i
    copy /Y .\samples\cpp\%%i\*.md .\out\sophon-inference\test\sc5_tests\cpp\%%i
  )
  copy /Y  .\samples\cpp\det_ssd\*.h .\out\sophon-inference\test\sc5_tests\cpp\det_ssd
  copy /Y  .\samples\cpp\det_ssd\cvdecoder.cpp .\out\sophon-inference\test\sc5_tests\cpp\det_ssd
  copy /Y  .\samples\cpp\det_ssd\processor.cpp .\out\sophon-inference\test\sc5_tests\cpp\det_ssd
  copy /Y  .\samples\cpp\det_ssd\*.md .\out\sophon-inference\test\sc5_tests\cpp\det_ssd
  copy /Y  .\samples\cpp\det_ssd\det_ssd_0.cpp .\out\sophon-inference\test\sc5_tests\cpp\det_ssd
  copy /Y  .\samples\cpp\det_ssd\det_ssd_1.cpp .\out\sophon-inference\test\sc5_tests\cpp\det_ssd
  copy /Y  .\samples\cpp\det_ssd\det_ssd_2.cpp .\out\sophon-inference\test\sc5_tests\cpp\det_ssd
goto:eof

:fill_document
  echo "------------------------------ fill document ------------------------------"
  ::fill release README.md
  copy /Y  .\release\README.md .\out\sophon-inference
  ::fill sophon-inference document
  copy /Y  .\docs\Sophon_Inference_zh.pdf .\out\sophon-inference\docs
goto:eof

