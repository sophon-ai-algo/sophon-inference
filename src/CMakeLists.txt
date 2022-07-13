
# set sail source file
set(CORE_SRC cvwrapper.cpp engine.cpp graph.cpp tensor.cpp tools.cpp base64.cpp engine_multi.cpp)

# build cpp api lib
add_library(sail SHARED ${CORE_SRC})
target_link_libraries(sail ${bmnn_link_libs} ${opencv_link_libs} ${ffmpeg_link_libs})

# build python api lib
add_subdirectory(../3rdparty/pybind11 pybind.dir)

if(CUSTOM_PY_LIBDIR)
  link_directories(${CUSTOM_PY_LIBDIR})
  message(STATUS "CUSTOM_PY_LIBDIR = ${CUSTOM_PY_LIBDIR}")
endif()
add_library(pysail MODULE bind.cpp ${CORE_SRC})

if(IS_SOC_MODE)
  target_include_directories(pysail PUBLIC ${PYBIND11_INCLUDE_DIR}
                             )
else()
  target_include_directories(pysail PUBLIC ${PYBIND11_INCLUDE_DIR}
          ${PYTHON_INCLUDE_DIRS})
endif()

target_compile_definitions(pysail PUBLIC -DPYTHON=1)

if(BUILD_X86_PCIE OR BUILD_CMODEL)
  message(STATUS "[pysail]opencv_link_libs=${opencv_link_libs}")
  target_link_libraries(pysail PRIVATE ${bmnn_link_libs} ${opencv_link_libs} ${ffmpeg_link_libs} pybind11::module pybind11::embed)
  set_target_properties(pysail PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                        SUFFIX "${PYTHON_MODULE_EXTENSION}" OUTPUT_NAME "sail")
else()
  target_link_libraries(pysail PRIVATE ${bmnn_link_libs} ${opencv_link_libs} ${ffmpeg_link_libs} pybind11::module)
  set_target_properties(pysail PROPERTIES PREFIX ""
                        SUFFIX ".so" OUTPUT_NAME "sail")
endif()
