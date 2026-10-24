cmake_minimum_required(VERSION 3.20)
project(scivol LANGUAGES C CXX)

# Builds the Python package's native modules:
#   scivol._scivolapi  - the new standalone library (scivolapi/) and its binding
#   scivol._core, scivol._native_core - legacy extensions, kept until the rebuild
#                       replaces every model they serve (then deleted).

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

set(SCIVOLAPI_PYTHON ON CACHE BOOL "" FORCE)
set(SCIVOLAPI_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(scivolapi)
install(TARGETS _scivolapi LIBRARY DESTINATION scivol)
install(FILES scivolapi/src/optimize/third_party/LICENSE.txt scivolapi/src/optimize/third_party/UPSTREAM.md
        DESTINATION scivol/_licenses/scivolapi_scipy_slsqp)

# Portable wheels omit host-specific tuning (SCIVOL_PORTABLE_BUILD=1 or cibuildwheel).
set(SCIVOL_PORTABLE_DEFAULT OFF)
if(DEFINED ENV{SCIVOL_PORTABLE_BUILD} AND "$ENV{SCIVOL_PORTABLE_BUILD}" STREQUAL "1")
    set(SCIVOL_PORTABLE_DEFAULT ON)
endif()
if(DEFINED ENV{CIBUILDWHEEL} AND "$ENV{CIBUILDWHEEL}" STREQUAL "1")
    set(SCIVOL_PORTABLE_DEFAULT ON)
endif()
option(SCIVOL_PORTABLE_BUILD "Omit host-specific compiler tuning" ${SCIVOL_PORTABLE_DEFAULT})

# ---- legacy: scivol._core (C) -------------------------------------------------
Python_add_library(_core MODULE WITH_SOABI
    scivol/_core.c
    scivol/_csrc/variance_garch.c
    scivol/_csrc/likelihood_garch.c
    scivol/_csrc/likelihood_normal.c
    scivol/_csrc/likelihood_studentt.c
    scivol/_csrc/likelihood_ged.c
    scivol/_csrc/likelihood_skewt.c
    scivol/_csrc/gh_math.c
    scivol/_csrc/likelihood_gh.c
    scivol/_csrc/likelihood_aparch.c
    scivol/_csrc/arma_vol_gh.c
    scivol/_csrc/errors_garch.c
    scivol/_csrc/transforms_logspace.c
    scivol/_csrc/arma_garch.c
    scivol/_csrc/arma_egarch.c
    scivol/_csrc/arma_gjr_garch.c
    scivol/_csrc/arma.c
    scivol/_csrc/mean_x.c
    scivol/_csrc/mean_garch.c
    scivol/_csrc/mean_gjr_garch.c
    scivol/_csrc/mean_egarch.c
    scivol/_csrc/variance_gjr_garch.c
    scivol/_csrc/likelihood_gjr_garch.c
    scivol/_csrc/likelihood_egarch.c
    scivol/_csrc/errors_gjr_garch.c
    scivol/_csrc/log_wrappers.c
    scivol/_csrc/dcc_gaussian.c
)
target_include_directories(_core PRIVATE scivol/_csrc)
if(MSVC)
    target_compile_options(_core PRIVATE /O2 /fp:fast)
elseif(SCIVOL_PORTABLE_BUILD)
    target_compile_options(_core PRIVATE -O3)
else()
    target_compile_options(_core PRIVATE -O3 -ffast-math -ffp-contract=fast -funroll-loops -march=native)
endif()
if(NOT WIN32)
    target_link_libraries(_core PRIVATE m)
endif()
install(TARGETS _core LIBRARY DESTINATION scivol)

# ---- legacy: scivol._native_core (C++) ----------------------------------------
Python_add_library(_native_core MODULE WITH_SOABI
    scivol/_native/native_core.cpp
    scivol/_native/mean_runtime.cpp
    scivol/_native/linear_joint_runtime.cpp
    scivol/_native/linear_garch_runtime.cpp
    scivol/_native/linear_power_runtime.cpp
    scivol/_native/garch_generated_api.cpp
    scivol/_native/garch_runtime.cpp
    scivol/_native/egarch_runtime.cpp
    scivol/_native/arma_egarch_runtime.cpp
    scivol/_native/aparch_runtime.cpp
    scivol/_native/aparch_multi_phase.cpp
    scivol/_native/aparch_moment_precision.cpp
    scivol/_native/arma_garch_runtime.cpp
    scivol/_native/arma_garch_runtime_reformulation.cpp
    scivol/_native/portable_linalg.cpp
    scivol/_native/portable_nnls.cpp
    scivol/_native/slsqp_translation.cpp
    scivol/_native/gh_math_translation.cpp
)
target_include_directories(_native_core PRIVATE
    scivol/_native
    scivol/_csrc
    scivol/_native/third_party/boost
    scivol/_native/third_party/tlfloat
)
set_target_properties(_native_core PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
if(MSVC)
    target_compile_options(_native_core PRIVATE /O2 /fp:precise /EHs-c- /GR-)
else()
    target_compile_options(_native_core PRIVATE -O3 -fno-exceptions -fno-rtti)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(_native_core PRIVATE -Wframe-larger-than=16384 -Werror=frame-larger-than)
    else()
        target_compile_options(_native_core PRIVATE -Werror=frame-larger-than=16384)
    endif()
    if(NOT SCIVOL_PORTABLE_BUILD)
        target_compile_options(_native_core PRIVATE -ffp-contract=fast -funroll-loops -march=native)
    endif()
endif()
if(NOT WIN32)
    target_link_libraries(_native_core PRIVATE m)
endif()
install(TARGETS _native_core LIBRARY DESTINATION scivol)

# Licenses of third-party code compiled into the legacy native module.
foreach(dir scipy_slsqp boost tlfloat)
    file(GLOB notices scivol/_native/third_party/${dir}/LICENSE*.txt scivol/_native/third_party/${dir}/UPSTREAM.md)
    install(FILES ${notices} DESTINATION scivol/_licenses/${dir})
endforeach()
