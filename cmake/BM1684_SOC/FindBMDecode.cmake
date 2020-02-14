message(STATUS "Finding Bitmain Decode")

set(BMDECODE_COMPONENTS
    bmvideo
    bmjpuapi
    bmjpulite)

set(BMDECODE_INCLUDE_DIRS
    ${BMDECODER_DIR}/decode/include)

foreach(ITEM ${BMDECODE_COMPONENTS})
    list(APPEND BMDECODE_LIBRARIES "${BMDECODER_DIR}/decode/lib/lib${ITEM}.so")
endforeach()

if(BMDECODE_INCLUDE_DIRS AND BMDECODE_LIBRARIES)
    set(BMDECODE_FOUND TRUE)
    message(STATUS "Bitmain Decode found")
else(BMDECODE_INCLUDE_DIRS AND BMDECODE_LIBRARIES)
    message(STATUS "Bitmain Decode not found")
endif(BMDECODE_INCLUDE_DIRS AND BMDECODE_LIBRARIES)
