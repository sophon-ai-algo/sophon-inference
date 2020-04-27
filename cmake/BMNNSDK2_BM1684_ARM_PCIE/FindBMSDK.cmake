message(STATUS "finding Bitmain Sophon SDK(BM1684)")
set(BMSDK_INCLUDE_COMPONETS BASE BMCPU BMLIB BMRT)
set(BMSDK_LIB_COMPONETS BMLIB BMRT BMCPU USERCPU BMCV)

# find sub dir include
find_path(BMSDK_BASE_INCLUDE_DIR NAMES op_code.h
  PATHS "${BMNNSDK2_PATH}/include")
find_path(BMSDK_BMCPU_INCLUDE_DIR NAMES bmcpu.h
  PATHS "${BMNNSDK2_PATH}/include/bmcpu")
find_path(BMSDK_BMLIB_INCLUDE_DIR NAMES bmlib_runtime.h
  PATHS "${BMNNSDK2_PATH}/include/bmlib")
find_path(BMSDK_BMRT_INCLUDE_DIR NAMES bmruntime_interface.h
  PATHS "${BMNNSDK2_PATH}/include/bmruntime")

find_library(BMSDK_BMLIB_LIBRARY NAMES bmlib
  PATHS "${BMNNSDK2_PATH}/lib/bmnn/arm_pcie")
find_library(BMSDK_BMCV_LIBRARY NAMES bmcv
  PATHS "${BMNNSDK2_PATH}/lib/bmnn/arm_pcie")
# set for centos
if(USE_CENTOS)
  set(BMRT_INNER_PATH "bmnn/pcie/lib_CXX11_ABI0")
else()
  if(USE_LOCAL)
    set(BMRT_INNER_PATH "bmnn/arm_pcie")
  else()
    set(BMRT_INNER_PATH "bmnn/pcie/lib_CXX11_ABI1")
  endif()
endif()
find_library(BMSDK_BMRT_LIBRARY NAMES bmrt
  PATHS "${BMNNSDK2_PATH}/lib/${BMRT_INNER_PATH}")
find_library(BMSDK_BMCPU_LIBRARY NAMES bmcpu
  PATHS "${BMNNSDK2_PATH}/lib/${BMRT_INNER_PATH}")
find_library(BMSDK_USERCPU_LIBRARY NAMES usercpu
  PATHS "${BMNNSDK2_PATH}/lib/${BMRT_INNER_PATH}")

foreach(ITEM ${BMSDK_INCLUDE_COMPONETS})
    list(APPEND BMSDK_INCLUDE_DIRS "${BMSDK_${ITEM}_INCLUDE_DIR}")
endforeach(ITEM)
foreach(ITEM ${BMSDK_LIB_COMPONETS})
    list(APPEND BMSDK_LIBRARIES "${BMSDK_${ITEM}_LIBRARY}")
endforeach(ITEM)

message(STATUS "BMSDK_INCLUDE_DIRS: ${BMSDK_INCLUDE_DIRS}")
message(STATUS "BMSDK_LIBRARIES: ${BMSDK_LIBRARIES}")

if(BMSDK_INCLUDE_DIRS AND BMSDK_LIBRARIES)
  set(BMSDK_FOUND TRUE)
  message(STATUS "BMSDK found")
else(BMSDK_INCLUDE_DIRS AND BMSDK_LIBRARIES)
  message(STATUS "BMSDK not found")
endif(BMSDK_INCLUDE_DIRS AND BMSDK_LIBRARIES)
