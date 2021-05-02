cmake_minimum_required(VERSION 3.16)
project(RedLight)

find_package(OpenCV)

MESSAGE(STATUS "OpenCV version: "${OpenCV_VERSION})

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/)
link_directories(/usr/local/lib/)
#link_directories(/Users/yh-mac/3rdparty-cpp/3rdparty-cpp/NCNN/build-mac/install/lib)

include_directories(/Users/yh-mac/3rdparty-cpp/3rdparty-cpp/NCNN/build-mac/install/include)
include_directories(${OpenCV_INCLUDE_DIRS} )
include_directories(/Users/yh-mac/3rdparty-cpp/3rdparty-cpp/MNN/include)
set(CMAKE_CXX_STANDARD 14)


#add_executable(RedLight main.cpp persondet/PersonDetect.cpp persondet/PersonDetect.h common/common.cpp common/common.h FaceDetector/FaceDetector.cpp FaceDetector/FaceDetector.h model_test.cpp model_test.h model_infer.h model_person.h ldmark/ldmark.h model_infer/mnn_infer.h object/ObjectDetect.cpp object/ObjectDetect.h)
#target_link_libraries(RedLight ncnn mnn ${OpenCV_LIBS} )
#
add_library(python_ext_c MODULE ${SOURCE_FILES} ext_c.cpp persondet/PersonDetect.cpp FaceDetector/FaceDetector.cpp common/common.cpp )
target_link_libraries(python_ext_c mnn ncnn ${OpenCV_LIBS} )