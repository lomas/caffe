cmake_minimum_required (VERSION 2.8)
project (demo)

set(CAFFE_ROOT_DIR /home/codes/caffe/)

add_subdirectory(lib/ inference.out/)

aux_source_directory(./ SRC_LIST)

include_directories (./ ${CMAKE_SOURCE_DIR}/lib/ )



#add_library(inference SHARED ${SRC_LIST})
add_executable(demo ${SRC_LIST})


target_link_libraries(demo
    inference
    #${CMAKE_SOURCE_DIR}/../build/lib/libcaffe.so
    ${CAFFE_ROOT_DIR}/build/lib/libcaffe.so
    /usr/lib/x86_64-linux-gnu/libopencv_highgui.so 
    /usr/lib/x86_64-linux-gnu/libopencv_core.so 
    /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so )
