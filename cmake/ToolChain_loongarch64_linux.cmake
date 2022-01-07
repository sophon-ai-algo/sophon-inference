# set system name and processor type
set(CMAKE_SYSTEM_NAME        Linux)
set(CMAKE_SYSTEM_PROCESSOR   loongarch64)

# set cross compiler
set(CROSS_COMPILE loongarch64-linux-gnu-)
set(CMAKE_C_COMPILER         ${CROSS_COMPILE}gcc)
set(CMAKE_CXX_COMPILER       ${CROSS_COMPILE}g++)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -Wno-error=deprecated-declarations -ffunction-sections -fdata-sections -ffp-contract=off -fPIC -Wno-unused-function -funwind-tables -rdynamic -fno-short-enums" CACHE STRING "cflags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -funwind-tables -rdynamic -fno-short-enums -Wall -Werror -Wno-error=deprecated-declarations -ffunction-sections -fdata-sections -fno-strict-aliasing -ffp-contract=off -std=c++11 -Wno-unused-function -Wno-format-truncation -fPIC -Wno-stringop-truncation" CACHE STRING "c++flags")

# search for programs in the build host dir
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# search lib and header in the target dir
set(CMAKE_FIND_ROOT_PATH_LIBRARY      ONLY)
set(CMAKE_FIND_ROOT_PATH_INCLUDE      ONLY)