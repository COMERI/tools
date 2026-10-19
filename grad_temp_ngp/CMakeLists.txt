cmake_minimum_required(VERSION 3.20)

project(cvfem_gradient_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#
# Trilinos
#
find_package(Trilinos REQUIRED)

#
# MPI
#
find_package(MPI REQUIRED)

message(STATUS "Trilinos_DIR = ${Trilinos_DIR}")
message(STATUS "Trilinos_VERSION = ${Trilinos_VERSION}")

#
# Source files
#
set(SOURCES
    main.cpp
    src/ComputeNodalGradTemperatureAlg.cpp
)

#
# Executable
#
add_executable(cvfem_gradient_test
    ${SOURCES}
)

#
# Include directories
#
target_include_directories(cvfem_gradient_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${Trilinos_DIR}/../../../include
)

#
# Trilinos libraries
#
target_link_libraries(cvfem_gradient_test
    PRIVATE
        ${Trilinos_LIBRARIES}
        MPI::MPI_CXX
)
