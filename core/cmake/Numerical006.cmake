option(SCIVOLAPI_BUILD_NUMERICAL006_PROBES "Build the private numerical006 transfer bridges" OFF)
if(NOT SCIVOLAPI_BUILD_NUMERICAL006_PROBES)
    return()
endif()

# Private backend symbols stay hidden in the production shared-library ABI.
get_target_property(numerical006_core_type scivolapi TYPE)
if(NOT numerical006_core_type STREQUAL "STATIC_LIBRARY")
    message(FATAL_ERROR "Numerical006 private bridges require the canonical static scivolapi target")
endif()

set(numerical006_root "${CMAKE_CURRENT_LIST_DIR}/../..")
add_library(scivolapi_numerical_006_probe SHARED
    "${numerical006_root}/tests/bridges/numerical_006/numerical_bridge.cpp")
add_library(scivolapi_numerical_006_paths_probe SHARED
    "${numerical006_root}/tests/bridges/numerical_006/joint_paths.cpp")
foreach(target IN ITEMS scivolapi_numerical_006_probe scivolapi_numerical_006_paths_probe)
    target_link_libraries(${target} PRIVATE scivolapi::scivolapi)
    target_include_directories(${target} PRIVATE
        "${numerical006_root}/core/src"
        "${numerical006_root}/core/src/math"
        "${numerical006_root}/core/src/math/power/third_party/boost"
        "${numerical006_root}/core/src/math/power/third_party/tlfloat")
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
    scivolapi_production_flags(${target})
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -UNDEBUG -fvisibility=hidden -fno-exceptions -fno-rtti)
        if(SCIVOLAPI_NATIVE_ARCH)
            target_compile_options(${target} PRIVATE -funroll-loops -march=native)
        endif()
    endif()
endforeach()
