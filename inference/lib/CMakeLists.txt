cmake_minimum_required (VERSION 2.8)
project (inference)


set(CAFFE_ROOT_DIR /home/codes/caffe/)

aux_source_directory(./ SRC_LIST)


include_directories ( 
    ${CAFFE_ROOT_DIR}/include/
    ${CAFFE_ROOT_DIR}/build/include/
    /usr/include
    /usr/local/include 
    /usr/include/python2.7
    /usr/local/cuda/include )



add_library(inference SHARED ${SRC_LIST})


target_link_libraries(inference 
    ${CAFFE_ROOT_DIR}/build/lib/libcaffe.so
    /usr/lib/x86_64-linux-gnu/libopencv_highgui.so 
    /usr/lib/x86_64-linux-gnu/libopencv_core.so 
    /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so 
   # /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so
    /usr/lib/x86_64-linux-gnu/libglog.so
    /usr/lib/x86_64-linux-gnu/libboost_system.so
    /usr/lib/x86_64-linux-gnu/libboost_python.so
    /usr/lib/x86_64-linux-gnu/libpython2.7.so
    )
