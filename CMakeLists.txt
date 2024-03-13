cmake_minimum_required(VERSION 3.12.0)
project(navier_stokes LANGUAGES CXX C)

include(common/cmake-common.cmake)

add_executable(navier_stokes src/navier_stokes.cpp src/Stokes.cpp)
deal_ii_setup_target(navier_stokes)