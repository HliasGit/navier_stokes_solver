cmake_minimum_required(VERSION 3.12.0)
project(NSSolverTest LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(StationaryNSSolver src/testStationary.cpp src/NSSolverStationary.cpp)
deal_ii_setup_target(StationaryNSSolver)
