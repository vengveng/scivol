# Unqualified native assertion-transfer candidates; run/review before disposition changes.
foreach(transfer_target IN ITEMS
    transfer_006_gh_quadrature
    transfer_006_joint_garch_workspace
    transfer_006_location_scale_assembly
    transfer_006_mean_workspace
    transfer_006_polygamma
    transfer_006_reflection_composition
    transfer_006_scaled_derivative_product
    transfer_006_scaled_logistic
    transfer_006_wide_precision
    transfer_006_jet_math_density_range
    transfer_006_jet_math_normal_curvature
    transfer_006_student_normalization
    transfer_006_linear_power_workspace
    transfer_006_softmax_recovery)
    add_executable(${transfer_target} "${CMAKE_CURRENT_LIST_DIR}/${transfer_target}.cpp")
endforeach()
foreach(transfer_target IN ITEMS
    transfer_006_jet_math_mean_recovery
    transfer_006_jet_math_moment_validity
    transfer_006_aparch_subset_curvature
    transfer_006_jet_math_aparch_range)
    add_library(${transfer_target} SHARED "${CMAKE_CURRENT_LIST_DIR}/${transfer_target}.cpp")
endforeach()
foreach(transfer_target IN ITEMS
    transfer_006_gh_quadrature
    transfer_006_jet_math_mean_recovery
    transfer_006_jet_math_moment_validity
    transfer_006_joint_garch_workspace
    transfer_006_location_scale_assembly
    transfer_006_mean_workspace
    transfer_006_polygamma
    transfer_006_reflection_composition
    transfer_006_scaled_derivative_product
    transfer_006_scaled_logistic
    transfer_006_wide_precision
    transfer_006_jet_math_density_range
    transfer_006_jet_math_normal_curvature
    transfer_006_student_normalization
    transfer_006_linear_power_workspace
    transfer_006_aparch_subset_curvature
    transfer_006_jet_math_aparch_range
    transfer_006_softmax_recovery)
    target_link_libraries(${transfer_target} PRIVATE scivolapi::scivolapi ${CMAKE_DL_LIBS})
    target_include_directories(${transfer_target} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/math/power/third_party/boost"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/math/power/third_party/tlfloat")
    scivolapi_production_flags(${transfer_target})
    set_target_properties(${transfer_target} PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${transfer_target} PRIVATE -fno-exceptions -fno-rtti -UNDEBUG)
    elseif(MSVC)
        target_compile_options(${transfer_target} PRIVATE /UNDEBUG)
    endif()
endforeach()
foreach(transfer_case IN ITEMS gh_quadrature joint_garch_workspace location_scale_assembly
        mean_workspace reflection_composition wide_precision linear_power_workspace
        jet_math_normal_curvature)
    add_test(NAME transfer_006_${transfer_case} COMMAND transfer_006_${transfer_case})
    set_tests_properties(transfer_006_${transfer_case} PROPERTIES TIMEOUT 120 LABELS "transfer_006;unqualified")
endforeach()
add_test(NAME transfer_006_student_domain COMMAND transfer_006_student_normalization --moment)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    foreach(transfer_target IN ITEMS transfer_006_gh_quadrature transfer_006_jet_math_mean_recovery transfer_006_jet_math_moment_validity transfer_006_location_scale_assembly transfer_006_mean_workspace transfer_006_polygamma transfer_006_reflection_composition transfer_006_scaled_derivative_product transfer_006_scaled_logistic transfer_006_wide_precision transfer_006_jet_math_density_range transfer_006_jet_math_normal_curvature transfer_006_student_normalization transfer_006_linear_power_workspace transfer_006_aparch_subset_curvature transfer_006_jet_math_aparch_range transfer_006_softmax_recovery)
        target_compile_options(${transfer_target} PRIVATE -funroll-loops)
    endforeach()
    target_compile_options(transfer_006_joint_garch_workspace PRIVATE -O2)
    target_compile_options(transfer_006_softmax_recovery PRIVATE -fno-builtin-exp -fstack-usage)
    foreach(transfer_target IN ITEMS transfer_006_jet_math_density_range
            transfer_006_jet_math_moment_validity transfer_006_jet_math_normal_curvature
            transfer_006_scaled_derivative_product transfer_006_scaled_logistic
            transfer_006_softmax_recovery transfer_006_wide_precision transfer_006_polygamma)
        target_compile_options(${transfer_target} PRIVATE -march=native)
    endforeach()
    if(NOT "$ENV{SCIVOL_PORTABLE_BUILD}" STREQUAL "1")
        target_compile_options(transfer_006_jet_math_mean_recovery PRIVATE -march=native)
    endif()
endif()
# Keep this header-only limb variant unlinked from differently compiled private owners.
add_executable(transfer_006_wide_precision_portable "${CMAKE_CURRENT_LIST_DIR}/transfer_006_wide_precision.cpp")
target_compile_features(transfer_006_wide_precision_portable PRIVATE cxx_std_20)
target_include_directories(transfer_006_wide_precision_portable PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/power/third_party/boost"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/power/third_party/tlfloat")
target_compile_definitions(transfer_006_wide_precision_portable PRIVATE SCIVOL_WIDE_PORTABLE_LIMBS)
scivolapi_production_flags(transfer_006_wide_precision_portable)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(transfer_006_wide_precision_portable PRIVATE -fno-exceptions -fno-rtti -UNDEBUG -funroll-loops -march=native)
elseif(MSVC)
    target_compile_options(transfer_006_wide_precision_portable PRIVATE /UNDEBUG)
endif()
set_target_properties(transfer_006_wide_precision transfer_006_wide_precision_portable PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF)
add_test(NAME transfer_006_wide_precision_portable COMMAND transfer_006_wide_precision_portable)
set_tests_properties(transfer_006_wide_precision_portable transfer_006_student_domain PROPERTIES TIMEOUT 120 LABELS "transfer_006;unqualified")
# Emitters/probes require the unchanged old caller case matrices and oracle review.
