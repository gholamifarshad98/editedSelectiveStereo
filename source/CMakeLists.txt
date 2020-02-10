cmake_minimum_required(VERSION 3.10)
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/../Install CACHE STRING "..." FORCE)
project (ParallelSelectiveStereoCuda)
add_executable (ParallelSelectiveStereoCuda main.cu)
set(CMAKE_BINARY_DIR "${PROJECT_SOURCE_DIR}/../Install/")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
find_package(OpenCV REQUIRED )
find_library(opencv_world342_LIB opencv_world342)
find_library(opencv_world342d_LIB opencv_world342d)
target_link_libraries(${CMAKE_PROJECT_NAME} "${opencv_world342_LIB}")
target_link_libraries(${CMAKE_PROJECT_NAME} "${opencv_world342d_LIB}")
include_directories(${OpenCV_INCLUDE_DIRS})


