include_directories(${PROJECT_SOURCE_DIR}/include
                    ${BMSDK_INCLUDE_DIRS})
link_libraries(${BMSDK_LIBRARIES})

if(USE_FFMPEG)
  message(STATUS "USE_FFMPEG")
  find_package(BMDecode REQUIRED)
  find_package(FFMPEG REQUIRED)
  include_directories(${FFMPEG_INCLUDE_DIRS})
  link_libraries(${BMDECODE_LIBRARIES} ${FFMPEG_LIBRARIES})
endif()

# set sail source file
set(CORE_SRC cvwrapper.cpp engine.cpp graph.cpp tensor.cpp tools.cpp)

if(USE_BMCV)
  link_libraries(${BMSDK_BMCV_LIBRARY})
endif()

if(NOT USE_PCIE)
  find_package(OpenCV REQUIRED)
  include_directories(${OpenCV_INCLUDE_DIRS})
  link_libraries(${OpenCV_LIBRARIES})
endif()

# build cpp api lib
add_library(sail SHARED ${CORE_SRC})

add_subdirectory(pybind11)

if((NOT USE_PCIE) OR USE_ARM_PCIE)
  include_directories(${PROJECT_SOURCE_DIR}/3rdparty/prebuild/include)
endif()

add_library(pysail MODULE bind.cpp ${CORE_SRC})
if(USE_PCIE)
  target_include_directories(pysail PUBLIC ${PYBIND11_INCLUDE_DIR}
                             ${PYTHON_INCLUDE_DIRS})
else()
  target_include_directories(pysail PUBLIC ${PYBIND11_INCLUDE_DIR})
endif()

# build python api lib
target_compile_definitions(pysail PUBLIC -DPYTHON)
if(USE_PCIE AND (NOT USE_ARM_PCIE))
  target_link_libraries(pysail PRIVATE pybind11::module pybind11::embed)
  set_target_properties(pysail PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                        SUFFIX "${PYTHON_MODULE_EXTENSION}" OUTPUT_NAME "sail")
else()
  target_link_libraries(pysail PRIVATE pybind11::module)
  set_target_properties(pysail PROPERTIES PREFIX ""
                        SUFFIX ".so" OUTPUT_NAME "sail")
endif()
