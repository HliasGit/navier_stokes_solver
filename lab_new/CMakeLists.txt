cmake_minimum_required(VERSION 3.12.0)
project(NSSolverTest LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(NSSolver src/test.cpp src/NSSolver.cpp)
add_executable(StationaryNSSolver src/testStationary.cpp src/NSSolverStationary.cpp)
deal_ii_setup_target(NSSolver)
deal_ii_setup_target(StationaryNSSolver)

#add_executable(lab-07-exercise2 src/lab-07-exercise2.cpp src/LinearElasticity.cpp)
#deal_ii_setup_target(lab-07-exercise2)
