cmake_minimum_required(VERSION 3.20)
project(scivolapi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(scivolapi
    src/model.cpp
    src/jet.cpp
)
add_library(scivolapi::scivolapi ALIAS scivolapi)
target_include_directories(scivolapi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(scivolapi PROPERTIES POSITION_INDEPENDENT_CODE ON)
# IEEE semantics: no fast-math (NaN and non-finite checks are part of the contract).
target_compile_options(scivolapi PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -fno-fast-math>
)

option(SCIVOLAPI_PYTHON "Build the scivol._scivolapi Python module" OFF)
if(SCIVOLAPI_PYTHON)
    find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
    find_package(nanobind CONFIG REQUIRED)
    nanobind_add_module(_scivolapi python/module.cpp)
    target_link_libraries(_scivolapi PRIVATE scivolapi)
endif()
