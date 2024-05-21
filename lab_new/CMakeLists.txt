cmake_minimum_required(VERSION 3.12.0)
project(NSSolverTest LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(NSSolverTest src/test.cpp src/NSSolver.cpp)
deal_ii_setup_target(NSSolverTest)

#add_executable(lab-07-exercise2 src/lab-07-exercise2.cpp src/LinearElasticity.cpp)
#deal_ii_setup_target(lab-07-exercise2)
