cmake_minimum_required(VERSION 3.2)
project(sophon-inference)

#debug
#set(BUILD_TYPE mips64)
#set(SDK_TYPE allinone)
if (WIN32)
   include(build_windows.cmake)
else()
   include(build_unix.cmake)
endif()


