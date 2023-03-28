
set(CMAKE_CXX_STANDARD 14)

cmake_minimum_required(VERSION 3.21)
project(Content_based_Image_Retrieval)

find_package(OpenCV)
include_directories(${/usr/local/include/})

add_executable(Content_based_Image_Retrieval main.cpp getFeature.cpp test.cpp getFeature.h matching.cpp csv_util.cpp csv_util.h matching.h filter.cpp filter.h test2.cpp test3.cpp)
target_link_libraries(Content_based_Image_Retrieval ${OpenCV_LIBS})


