cmake_minimum_required(VERSION 3.12.0)
project(step-57 LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(step-57 step-57.cc)
deal_ii_setup_target(step-57)