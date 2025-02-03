cmake_minimum_required(VERSION 3.12.0)
project(NSSolverTest LANGUAGES CXX C)

include(../common/cmake-common.cmake)

# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE Release)

add_executable(NSSolver src/test.cpp src/NSSolver.cpp)
add_executable(StationaryNSSolver src/testStationary.cpp src/NSSolverStationary.cpp)
deal_ii_setup_target(NSSolver)
deal_ii_setup_target(StationaryNSSolver)