# set system name and processor type


#INCLUDE(CMakeForceCompiler)
set(CMAKE_SYSTEM_NAME        Linux)
set(CMAKE_SYSTEM_PROCESSOR   sw64)
#CMAKE_FORCE_C_COMPILER(sw_64-sunway-linux-gnu-gcc GNU)
#CMAKE_FORCE_CXX_COMPILER(sw_64-sunway-linux-gnu-g++ GNU)
#set(CMAKE_C_COMPILER_WORKS ON)
#set(CMAKE_CXX_COMPILER_WORKS ON)
# set cross compiler
set(CROSS_COMPILE sw_64-sunway-linux-gnu-)
set(CMAKE_C_COMPILER         ${CROSS_COMPILE}gcc)
set(CMAKE_CXX_COMPILER       ${CROSS_COMPILE}g++ )
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -Wno-stringop-truncation -Wno-format-truncation -Wno-error=deprecated-declarations -ffunction-sections -fdata-sections -fno-strict-aliasing -ffp-contract=off -Wno-unused-function -fPIC --machine=sw6b" CACHE STRING "cflags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-stringop-truncation -Wno-format-truncation -Wno-error=deprecated-declarations -fexceptions -funwind-tables -rdynamic -fno-short-enums -ffunction-sections -fdata-sections -fno-strict-aliasing -ffp-contract=off -std=c++11 -Wno-unused-function -fPIC --machine=sw6b" CACHE STRING "c++flags")

# search for programs in the build host dir
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# search lib and header in the target dir
set(CMAKE_FIND_ROOT_PATH_LIBRARY      ONLY)
set(CMAKE_FIND_ROOT_PATH_INCLUDE      ONLY)
