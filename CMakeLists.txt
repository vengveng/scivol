cmake_minimum_required(VERSION 3.20)
project(scivol LANGUAGES C CXX)

# Builds the Python package's native module scivol._scivolapi: the standalone
# C++20 library in scivolapi/ and its binding.

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

set(SCIVOLAPI_PYTHON ON CACHE BOOL "" FORCE)
set(SCIVOLAPI_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(scivolapi)
install(TARGETS _scivolapi LIBRARY DESTINATION scivol)
install(FILES scivolapi/src/optimize/third_party/LICENSE.txt scivolapi/src/optimize/third_party/UPSTREAM.md
        DESTINATION scivol/_licenses/scivolapi_scipy_slsqp)
