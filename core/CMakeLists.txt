cmake_minimum_required(VERSION 3.20)
project(scivolapi VERSION 0.3.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SCIVOLAPI_BUILD_PROBE "Build the private independent-oracle probe" ${BUILD_TESTING})
option(SCIVOLAPI_NATIVE_ARCH "Use the existing local native architecture/unrolling flags" OFF)

set(scivolapi_backend_sources
    src/backend.cpp
    src/mean_backend.cpp
    src/profile_backend.cpp
    src/dependence_backend.cpp
    src/power_backend.cpp
    src/joint_backend.cpp
    src/joint_power_backend.cpp
    src/preparation_plan.cpp
    src/models/joint/coordinate_map.cpp
    src/models/power/aparch_moment_precision.cpp
    src/models/garch_runtime.cpp
    src/models/garch_generated_api.cpp
    src/math/gh_math_translation.cpp)
set(scivolapi_frontend_sources
    src/model.cpp src/problem.cpp src/jet.cpp src/program.cpp src/sample/causal_mean.cpp)

add_library(scivolapi ${scivolapi_frontend_sources} ${scivolapi_backend_sources})
add_library(scivolapi::scivolapi ALIAS scivolapi)
target_compile_features(scivolapi PUBLIC cxx_std_20)
target_include_directories(scivolapi PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(scivolapi PRIVATE
    src/math/power/third_party/boost src/math/power/third_party/tlfloat)
set_target_properties(scivolapi PROPERTIES
    CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
    WINDOWS_EXPORT_ALL_SYMBOLS ON)

include(cmake/ProductionFlags.cmake)
scivolapi_production_flags(scivolapi)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(${scivolapi_backend_sources}
        PROPERTIES COMPILE_OPTIONS "-fvisibility=hidden;-fno-exceptions;-fno-rtti")
    if(SCIVOLAPI_NATIVE_ARCH)
        set_property(SOURCE ${scivolapi_backend_sources} APPEND PROPERTY
            COMPILE_OPTIONS -funroll-loops -march=native)
    endif()
endif()

if(BUILD_TESTING)
    foreach(test_name IN ITEMS test_core_contract test_core_allocations test_fixed_runtime
            test_review_regressions test_skew_negative_moment test_bessel_curvature test_mean_contract test_mean_allocations
            test_power_contract test_power_allocations test_profile_backend
            test_profile_range test_profile_range_backend test_profile_ordinary test_profile_admitted
            test_joint_contract test_joint_allocations
            test_joint_power_contract test_joint_power_allocations
            test_dependence_backend test_causal_mean test_unit_map test_garch_chart_endpoints
            test_point_preparation test_mean_student_preparation test_shared_preparation test_shared_gh_moment_banks test_public_composition test_skew_splice
            test_preparation_enrollment test_public_preparation_enrollment
            test_causal_mean_integration test_dynamic_arx_chart test_generated_ar_fixed_metadata test_egarch_fixed_chart_metadata
            test_mean_fixed_chart_metadata test_profile_public test_arx_reflection_integration test_objective_projection
            test_mean_binding_metadata test_mean_selected_binding test_profile_fixed_units
            test_active_mean_basis test_signed_residual_germ test_selected_unit_seeds
            test_selected_density_preparation
            test_active_garch_basis test_active_number_lifetime
            test_selected_ordinary_product test_selected_mapped_omega test_selected_tier_retry
            test_public_unit_dispatch
            test_normal_factored_score test_selected_normal_factored_score
            test_unit_map_public test_unit_restriction_public test_joint_fixed_map_metadata
            test_softplus_original_z test_egarch_fixed_coefficient_metadata
            test_egarch_coefficient_consumers test_egarch_coefficient_static
            test_egarch_fixed_workspace_metadata test_conditioning_history
            test_skew_splice_public_status test_mean_softplus_original_z
            test_softplus_hessian_row test_garch_map_single_owner
            test_gjr_public_admission
            test_skew_splice_consumers test_dependence_public
            test_active_dependence test_active_dependence_public
            test_gjr_shared_map_owner test_unit_map_range test_profile_units_public test_subnormal_sample
            test_exact_dyadic_affine test_gjr_exact_admission test_gjr_half_identity
            test_unit_map_actions test_selected_unit_chain_range test_selected_unit_gaussian_range
            test_unit_map_rebind test_unit_map_magnitude_enclosure)
        set(test_source "${CMAKE_CURRENT_SOURCE_DIR}/../tests/native/${test_name}.cpp")
        add_executable(${test_name} "${test_source}")
        target_link_libraries(${test_name} PRIVATE scivolapi::scivolapi)
        target_include_directories(${test_name} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/math/power/third_party/boost"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/math/power/third_party/tlfloat")
        scivolapi_production_flags(${test_name})
        if(test_name STREQUAL "test_public_unit_dispatch")
            foreach(density_code RANGE 0 7)
                add_test(NAME ${test_name}_${density_code} COMMAND ${test_name} ${density_code})
                set_tests_properties(${test_name}_${density_code} PROPERTIES TIMEOUT 120)
            endforeach()
        elseif(test_name STREQUAL "test_public_preparation_enrollment")
            foreach(density_code RANGE 1 7)
                add_test(NAME ${test_name}_${density_code} COMMAND ${test_name} ${density_code})
                set_tests_properties(${test_name}_${density_code} PROPERTIES TIMEOUT 120)
            endforeach()
        else()
            add_test(NAME ${test_name} COMMAND ${test_name})
            set_tests_properties(${test_name} PROPERTIES TIMEOUT 120)
        endif()
    endforeach()
    # Instrumented inline density owner: keep this TU unlinked from scivolapi.
    add_executable(test_active_dependence_density ../tests/native/test_active_dependence_density.cpp)
    target_compile_features(test_active_dependence_density PRIVATE cxx_std_20)
    target_include_directories(test_active_dependence_density PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/math/power/third_party/boost"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/math/power/third_party/tlfloat")
    set_target_properties(test_active_dependence_density PROPERTIES CXX_EXTENSIONS OFF UNITY_BUILD OFF)
    scivolapi_production_flags(test_active_dependence_density)
    add_test(NAME test_active_dependence_density COMMAND test_active_dependence_density)
    set_tests_properties(test_active_dependence_density PROPERTIES TIMEOUT 120)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../tests/native/transfer_native_006.cmake")
    include(cmake/Coordinate006.cmake)
    add_library(scivolapi_profile_ordinary_probe OBJECT ../tests/native/profile_ordinary_probe.cpp)
    target_link_libraries(scivolapi_profile_ordinary_probe PRIVATE scivolapi::scivolapi)
    target_include_directories(scivolapi_profile_ordinary_probe PRIVATE
        src src/math/power/third_party/boost src/math/power/third_party/tlfloat)
    set_target_properties(scivolapi_profile_ordinary_probe PROPERTIES
        CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
    scivolapi_production_flags(scivolapi_profile_ordinary_probe)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(scivolapi_profile_ordinary_probe PRIVATE
            -fvisibility=hidden -fno-exceptions -fno-rtti)
        if(SCIVOLAPI_NATIVE_ARCH)
            target_compile_options(scivolapi_profile_ordinary_probe PRIVATE -funroll-loops -march=native)
        endif()
    endif()
    target_include_directories(test_mean_binding_metadata PRIVATE src/models/mean)
    target_include_directories(test_mean_selected_binding PRIVATE src/models/mean)
    foreach(test_name IN ITEMS test_profile_ordinary test_profile_admitted test_profile_fixed_units)
        target_link_libraries(${test_name} PRIVATE scivolapi_profile_ordinary_probe)
    endforeach()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_conditioning_history PRIVATE -fno-exceptions -fno-rtti)
    endif()
    set_tests_properties(test_skew_splice_public_status PROPERTIES TIMEOUT 30)
    target_compile_definitions(test_garch_chart_endpoints PRIVATE SCIVOLAPI_LINKED_ENDPOINT_TEST=1)
    add_executable(scivolapi_jet_example ../examples/native/jet.cpp)
    target_link_libraries(scivolapi_jet_example PRIVATE scivolapi::scivolapi)
    scivolapi_production_flags(scivolapi_jet_example)
    add_test(NAME scivolapi_jet_example COMMAND scivolapi_jet_example)
    add_executable(scivolapi_fixed_input_dcc ../examples/native/fixed_input_dcc.cpp)
    target_link_libraries(scivolapi_fixed_input_dcc PRIVATE scivolapi::scivolapi)
    scivolapi_production_flags(scivolapi_fixed_input_dcc)
    add_test(NAME scivolapi_fixed_input_dcc COMMAND scivolapi_fixed_input_dcc)
endif()

if(SCIVOLAPI_BUILD_PROBE)
    include(cmake/Numerical006.cmake)
    foreach(probe IN ITEMS jet dependence profile profile_units)
        add_library(scivolapi_${probe}_probe SHARED ../tests/bridges/${probe}_probe.cpp)
        target_link_libraries(scivolapi_${probe}_probe PRIVATE scivolapi::scivolapi)
        scivolapi_production_flags(scivolapi_${probe}_probe)
    endforeach()
    target_include_directories(scivolapi_profile_probe PRIVATE
        src/math/power/third_party/boost src/math/power/third_party/tlfloat)
endif()

install(TARGETS scivolapi EXPORT scivolapiTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/scivolapi DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ../LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/scivolapi)
install(FILES src/math/power/third_party/boost/LICENSE_1_0.txt
    src/math/power/third_party/boost/UPSTREAM.md
    DESTINATION ${CMAKE_INSTALL_DATADIR}/scivolapi/third_party/boost)
install(FILES src/math/power/third_party/tlfloat/LICENSE.txt
    src/math/power/third_party/tlfloat/UPSTREAM.md
    DESTINATION ${CMAKE_INSTALL_DATADIR}/scivolapi/third_party/tlfloat)
install(EXPORT scivolapiTargets NAMESPACE scivolapi::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scivolapi)
configure_package_config_file(cmake/scivolapiConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/scivolapiConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scivolapi)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/scivolapiConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/scivolapiConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/scivolapiConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scivolapi)
