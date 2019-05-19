cmake_minimum_required(VERSION 3.14)
project(sba)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
include_directories("/usr/include/eigen3")
include_directories(${OpenCV_INCLUDE_DIR})

add_executable(sba main.cpp readData.cpp ProjectFactor.cpp CameraParameterization.cpp)
target_link_libraries(sba ${OpenCV_LIBRARIES} ${CERES_LIBRARIES})

