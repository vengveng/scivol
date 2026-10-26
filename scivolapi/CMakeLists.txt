cmake_minimum_required(VERSION 3.20)
project(scivolapi LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

file(GLOB SCIVOLAPI_OPTIMIZE_SOURCES CONFIGURE_DEPENDS src/optimize/*.cpp src/optimize/*.c)
file(GLOB SCIVOLAPI_MATH_SOURCES CONFIGURE_DEPENDS src/math/*.cpp src/density/*.cpp)
file(GLOB SCIVOLAPI_JET_SOURCES CONFIGURE_DEPENDS src/jet_*.cpp)
add_library(scivolapi
    src/model.cpp
    src/jet.cpp
    ${SCIVOLAPI_JET_SOURCES}
    src/fit.cpp
    src/finalize.cpp
    src/statistics.cpp
    src/simulate.cpp
    src/correlation.cpp
    src/correlation_paths.cpp
    ${SCIVOLAPI_OPTIMIZE_SOURCES}
    ${SCIVOLAPI_MATH_SOURCES}
)
add_library(scivolapi::scivolapi ALIAS scivolapi)
target_include_directories(scivolapi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(scivolapi PROPERTIES POSITION_INDEPENDENT_CODE ON)
# IEEE semantics: no fast-math (NaN and non-finite checks are part of the contract).
target_compile_options(scivolapi PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -fno-fast-math>
)

option(SCIVOLAPI_TESTS "Build native tests" ON)
if(SCIVOLAPI_TESTS)
    enable_testing()
    file(GLOB_RECURSE SCIVOLAPI_TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
    foreach(source ${SCIVOLAPI_TEST_SOURCES})
        get_filename_component(name ${source} NAME_WE)
        add_executable(${name} ${source})
        target_link_libraries(${name} PRIVATE scivolapi)
        target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        add_test(NAME ${name} COMMAND ${name})
    endforeach()
endif()

option(SCIVOLAPI_PYTHON "Build the scivol._scivolapi Python module" OFF)
if(SCIVOLAPI_PYTHON)
    find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
    find_package(nanobind CONFIG REQUIRED)
    nanobind_add_module(_scivolapi binding/module.cpp)
    target_link_libraries(_scivolapi PRIVATE scivolapi)
endif()
