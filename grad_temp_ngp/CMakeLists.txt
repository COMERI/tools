
cmake_minimum_required(VERSION 3.20)

project(cvfem LANGUAGES CXX)

enable_testing()

#
# Dependencies
#

# Trilinos provides STK, Kokkos, and the other required libraries.
find_package(Trilinos REQUIRED)

# MPI is required by the STK/Trilinos configuration.
find_package(MPI REQUIRED)

message(STATUS "Trilinos_DIR = ${Trilinos_DIR}")
message(STATUS "Trilinos_VERSION = ${Trilinos_VERSION}")


#
# Common include directories
#
# These directories are shared by the application and all unit tests.
#
set(CVFEM_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Trilinos_DIR}/../../../include
)


#
# Common libraries
#
# All CVFEM executables link against Trilinos and MPI.
#
set(CVFEM_LIBRARIES
    ${Trilinos_LIBRARIES}
    MPI::MPI_CXX
)


#
# Main application
#
# The application has its own main.cpp and includes the source files
# for the CVFEM algorithms.
#
file(GLOB APP_SOURCES CONFIGURE_DEPENDS
    src/*.cpp
)

add_executable(cvfem_ngp_app
    main.cpp
    ${APP_SOURCES}
)

target_include_directories(cvfem_ngp_app
    PRIVATE
        ${CVFEM_INCLUDE_DIRS}
)

target_link_libraries(cvfem_ngp_app
    PRIVATE
        ${CVFEM_LIBRARIES}
)


#
# Unit-test helper function
#
# Defines a standalone unit-test executable.
#
# Arguments:
#   test_name  - Name of the executable
#   ARGN       - Source files required by that test
#
# Each test receives the common include directories and libraries.
# This avoids duplicating configuration as the test suite grows.
#
function(add_cvfem_unit_test test_name)

    add_executable(${test_name}
        ${ARGN}
    )

    target_include_directories(${test_name}
        PRIVATE
            ${CVFEM_INCLUDE_DIRS}
    )

    target_compile_definitions(${test_name}
        PRIVATE
        CVFEM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    
    target_link_libraries(${test_name}
        PRIVATE
            ${CVFEM_LIBRARIES}
    )

endfunction()


#
# Unit tests
#
# Each test is registered by calling add_cvfem_unit_test().
# Only provide the test source and the algorithm source files it needs.
#

# nodal volume
add_cvfem_unit_test(cvfem_nodal_volume_unit_test
    tests/ComputeNodalVolumeElemTest.cpp
    src/ComputeNodalVolumeElemAlg.cpp
)

add_test(
    NAME cvfem_nodal_volume
    COMMAND cvfem_nodal_volume_unit_test
)

# nodal SCV gradTemp
add_cvfem_unit_test(cvfem_grad_temperature_unit_test
    tests/ComputeGradTemperatureElemTest.cpp
    src/ComputeNodalGradTemperatureElemAlg.cpp
    src/ComputeNodalVolumeElemAlg.cpp
)

add_test(
    NAME cvfem_grad_temperature
    COMMAND cvfem_grad_temperature_unit_test
)

# nodal SCS gradTemp
add_cvfem_unit_test(cvfem_grad_temperature_scs_unit_test
    tests/ComputeGradTemperatureScsElemTest.cpp
    src/ComputeNodalGradTemperatureScsElemAlg.cpp
    src/ComputeNodalVolumeElemAlg.cpp
)

add_test(
    NAME cvfem_grad_temperature_scs
    COMMAND cvfem_grad_temperature_scs_unit_test
)
