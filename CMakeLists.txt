cmake_minimum_required(VERSION 2.8)
project(rvio2)

# Set build type
IF(NOT CMAKE_BUILD_TYPE)
#SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_BUILD_TYPE Release)
ENDIF()
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  add_definitions(-DCOMPILEDWITHC11)
  message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  add_definitions(-DCOMPILEDWITHC0X)
  message(STATUS "Using flag -std=c++0x.")
else()
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Enable debug flags (if you want to debug in gdb)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g3")

# Include libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp roslib rosbag tf std_msgs sensor_msgs geometry_msgs nav_msgs cv_bridge
)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3.0 QUIET)

# Include directories for headers
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)

catkin_package()

add_library(${PROJECT_NAME} SHARED
  src/rvio2/System.cc
  src/rvio2/Ransac.cc
  src/rvio2/Tracker.cc
  src/rvio2/Updater.cc
  src/rvio2/Propagator.cc
  src/rvio2/InputBuffer.cc
  src/rvio2/Feature.cc
  src/rvio2/FeatureDetector.cc
)

add_executable(rvio2_mono src/rvio2_mono.cc)
target_link_libraries(rvio2_mono
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${catkin_LIBRARIES}
)

add_executable(rvio2_mono_eval src/rvio2_mono_eval.cc)
target_link_libraries(rvio2_mono_eval
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${catkin_LIBRARIES}
)
