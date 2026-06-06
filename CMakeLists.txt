cmake_minimum_required(VERSION 3.24)

project(
    FuriaOptimizer
    VERSION 0.1.0
    DESCRIPTION "C++ nonlinear constrained optimization library"
    LANGUAGES CXX
)

# =====================================================
# Options
# =====================================================

option(FURIAOPTIMIZER_BUILD_TESTS "Build unit tests" OFF)
option(FURIAOPTIMIZER_BUILD_EXAMPLES "Build examples" ON)
option(FURIAOPTIMIZER_BUILD_BENCHMARKS "Build benchmarks" OFF)

# =====================================================
# Global settings
# =====================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =====================================================
# Dependencies
# =====================================================

find_package(Eigen3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# =====================================================
# Library
# =====================================================

add_library(furiaoptimizer
    src/solvers/unconstrained_solver.cpp
    src/solvers/lp_solver.cpp
    src/solvers/qp_solver.cpp
    src/direction_strategy.cpp
)

add_library(FuriaOptimizer::furiaoptimizer ALIAS furiaoptimizer)

target_include_directories(furiaoptimizer
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(furiaoptimizer
    PUBLIC
        Eigen3::Eigen
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

target_compile_features(furiaoptimizer PUBLIC cxx_std_20)

target_compile_options(furiaoptimizer
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# =====================================================
# Examples
# =====================================================

if(FURIAOPTIMIZER_BUILD_EXAMPLES)

    add_executable(furiaoptimizer_examples
        examples/rosenbrock.cpp
    )

    target_link_libraries(furiaoptimizer_examples
        PRIVATE FuriaOptimizer::furiaoptimizer Eigen3::Eigen
    )

endif()

# =====================================================
# Benchmarks
# =====================================================

if(FURIAOPTIMIZER_BUILD_BENCHMARKS)

    add_executable(furiaoptimizer_benchmarks
        #benchmarks/example_benchmark.cpp
    )

endif()

# =====================================================
# Tests
# =====================================================

if(FURIAOPTIMIZER_BUILD_TESTS)

    enable_testing()

    find_package(Catch2 CONFIG REQUIRED)

    add_executable(furiaoptimizer_tests
        #tests/example_test.cpp
    )

    target_link_libraries(furiaoptimizer_tests
        PRIVATE
            FuriaOptimizer::furiaoptimizer
            Catch2::Catch2WithMain
    )

    include(Catch)

    catch_discover_tests(furiaoptimizer_tests)

endif()