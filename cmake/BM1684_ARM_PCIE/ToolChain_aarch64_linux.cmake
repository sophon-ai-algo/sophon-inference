# set system name and processor type
set(CMAKE_SYSTEM_NAME        Linux)
set(CMAKE_SYSTEM_PROCESSOR   aarch64)

# set cross compiler
set(CROSS_COMPILE aarch64-linux-gnu-)
set(CMAKE_C_COMPILER         ${CROSS_COMPILE}gcc)
set(CMAKE_CXX_COMPILER       ${CROSS_COMPILE}g++)

# set target env
# set(CMAKE_FIND_ROOT_PATH "/data/release/toolchains/gcc/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu")

# search for programs in the build host dir
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# search lib and header in the target dir
set(CMAKE_FIND_ROOT_PATH_LIBRARY      ONLY)
set(CMAKE_FIND_ROOT_PATH_INCLUDE      ONLY)
