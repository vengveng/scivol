function(scivolapi_production_flags target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -fno-fast-math -O3 -ffp-contract=fast -fno-math-errno
            -Wframe-larger-than=16384 -Werror=frame-larger-than
            -Werror=vla -Werror=alloca)
    elseif(MSVC)
        target_compile_options(${target} PRIVATE /O2 /fp:precise /fp:contract)
        message(WARNING "MSVC stack-frame evidence remains an explicit qualification gate")
    else()
        message(FATAL_ERROR "Unqualified compiler: production IEEE/frame flags are not defined")
    endif()
endfunction()
