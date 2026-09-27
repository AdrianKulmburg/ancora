# Top-level CMakeLists.txt for the ancora library.
#
# Build flavors:
#
#   SAFE:
#       -DANCORA_MODE_SAFE=ON
#       -> libancora
#       -> FLINT/ARB
#
#   FAST:
#       -DANCORA_MODE_SAFE=OFF
#       -> libancora_fast
#       -> BLAS + HiGHS
#
#   FAST + GPU:
#       -DANCORA_MODE_SAFE=OFF
#       -DANCORA_USE_GPU=ON
#       -> libancora_fast_gpu
#       -> HIP + hipBLAS + HiGHS
#
# IMPORTANT:
# ANCORA_MODE_SAFE is a CMake option. It is NOT passed as a compiler
# definition. The source code uses ANCORA_MODE, whose values are defined
# by ancora_config.h:
#
#     ANCORA_MODE_SAFE = 0
#     ANCORA_MODE_FAST = 1
#
# Therefore we pass:
#
#     ANCORA_MODE=ANCORA_MODE_SAFE
#
# or:
#
#     ANCORA_MODE=ANCORA_MODE_FAST
#
# SAFE and FAST are ABI-incompatible and must never be mixed in one process.
#
# GPU is a separate axis. It is only supported with FAST mode.
# The GPU implementation is written once in HIP and can target:
#
#     ANCORA_GPU_PLATFORM=amd
#     ANCORA_GPU_PLATFORM=nvidia
#
# ---------------------------------------------------------------------------


cmake_minimum_required(VERSION 3.21)


# ===========================================================================
# User options
# ===========================================================================

option(
    ANCORA_MODE_SAFE
    "Build the safe FLINT/ARB-backed flavor"
    OFF
)

option(
    ANCORA_USE_GPU
    "Build the FAST flavor with the HIP GPU implementation"
    OFF
)

set(
    ANCORA_GPU_PLATFORM
    "amd"
    CACHE STRING
    "GPU platform for HIP: 'amd' or 'nvidia'"
)

set_property(
    CACHE ANCORA_GPU_PLATFORM
    PROPERTY STRINGS amd nvidia
)

set(
    ANCORA_HIP_ARCHITECTURES
    ""
    CACHE STRING
    "HIP GPU architecture(s), e.g. gfx90a or 80"
)


# ===========================================================================
# Validate options
# ===========================================================================

if(NOT ANCORA_GPU_PLATFORM STREQUAL "amd"
   AND NOT ANCORA_GPU_PLATFORM STREQUAL "nvidia")

    message(
        FATAL_ERROR
        "ANCORA_GPU_PLATFORM must be either 'amd' or 'nvidia'. "
        "Got '${ANCORA_GPU_PLATFORM}'."
    )

endif()


if(ANCORA_USE_GPU AND ANCORA_MODE_SAFE)

    message(
        FATAL_ERROR
        "ANCORA_USE_GPU=ON requires ANCORA_MODE_SAFE=OFF. "
        "There is no GPU-accelerated FLINT/ARB path."
    )

endif()


# ===========================================================================
# HIP platform selection
#
# ROCm's hip package looks at HIP_PLATFORM. Set it before project() so that
# the HIP/ROCm CMake machinery sees the correct platform.
# ===========================================================================

if(ANCORA_USE_GPU)

    if(ANCORA_GPU_PLATFORM STREQUAL "nvidia")
        set(ENV{HIP_PLATFORM} "nvidia")
    else()
        set(ENV{HIP_PLATFORM} "amd")
    endif()

endif()


# ===========================================================================
# Project
# ===========================================================================

# The project only needs C unless the GPU implementation is enabled.
if(ANCORA_USE_GPU)
    project(ancora C CXX)
else()
    project(ancora C)
endif()


set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)


# ===========================================================================
# Custom CMake modules
# ===========================================================================

# FindFLINT.cmake and FindHIGHS.cmake live here.
list(
    APPEND
    CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
)


# ===========================================================================
# Dependencies
# ===========================================================================

# ---------------------------------------------------------------------------
# SAFE mode
# ---------------------------------------------------------------------------

if(ANCORA_MODE_SAFE)

    find_package(FLINT REQUIRED)

endif()


# ---------------------------------------------------------------------------
# FAST mode
# ---------------------------------------------------------------------------

if(NOT ANCORA_MODE_SAFE)

    # HiGHS is required by the FAST zonotope containment implementation.
    find_package(HIGHS REQUIRED)

endif()


# ---------------------------------------------------------------------------
# GPU mode
# ---------------------------------------------------------------------------
#
# ROCm's packages are normally installed under:
#
#     /opt/rocm/
#
# which may itself be a symlink to something such as:
#
#     /opt/rocm-6.2.2/
#
# ANCORA_ROCM_ROOT can be overridden with:
#
#     -DANCORA_ROCM_ROOT=/some/other/rocm
#
# We deliberately find HIP and hipBLAS exactly once here.
# In particular, do NOT call find_package(hipblas) again later.
# ---------------------------------------------------------------------------

if(ANCORA_USE_GPU)

    set(
        ANCORA_ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )

    message(
        STATUS
        "ancora: ROCm root = ${ANCORA_ROCM_ROOT}"
    )

    message(
        STATUS
        "ancora: GPU platform = ${ANCORA_GPU_PLATFORM}"
    )

    list(
        APPEND CMAKE_PREFIX_PATH
        "${ANCORA_ROCM_ROOT}"
        "${ANCORA_ROCM_ROOT}/lib/cmake"
    )



    # -----------------------------------------------------------------------
    # HIP package
    #
    # This package is responsible for providing targets such as:
    #
    #     hip::host
    #     hip::device
    #
    # hipBLAS depends on hip::host, so HIP must be found before hipBLAS.
    # -----------------------------------------------------------------------

    find_package(
        hip
        CONFIG
        REQUIRED
        PATHS
            "${ANCORA_ROCM_ROOT}/lib/cmake/hip"
        NO_DEFAULT_PATH
    )


    # -----------------------------------------------------------------------
    # hipBLAS package
    #
    # roc::hipblas depends on hip::host, which was provided by find_package
    # above.
    # -----------------------------------------------------------------------

    find_package(
        hipblas
        CONFIG
        REQUIRED
        PATHS
            "${ANCORA_ROCM_ROOT}/lib/cmake/hipblas"
        NO_DEFAULT_PATH
    )


    # -----------------------------------------------------------------------
    # HIP platform
    #
    # CMake versions that support CMAKE_HIP_PLATFORM use this to select the
    # backend. HIP_PLATFORM above is also set because ROCm's own package
    # machinery uses it.
    # -----------------------------------------------------------------------

    set(
        CMAKE_HIP_PLATFORM
        "${ANCORA_GPU_PLATFORM}"
    )


    # -----------------------------------------------------------------------
    # Enable CMake's native HIP language support.
    #
    # Do NOT set CMAKE_HIP_COMPILER to hipcc. CMake's HIP support expects
    # the underlying compiler (normally ROCm's clang).
    # -----------------------------------------------------------------------

    enable_language(HIP)


    # -----------------------------------------------------------------------
    # Explicit HIP platform macros.
    #
    # This avoids relying solely on compiler/package autodetection.
    # -----------------------------------------------------------------------

    if(ANCORA_GPU_PLATFORM STREQUAL "nvidia")

        add_compile_definitions(
            $<$<COMPILE_LANGUAGE:HIP>:__HIP_PLATFORM_NVIDIA__>
        )

    else()

        add_compile_definitions(
            $<$<COMPILE_LANGUAGE:HIP>:__HIP_PLATFORM_AMD__>
        )

    endif()


endif()


# ===========================================================================
# Core library
# ===========================================================================

add_library(
    ancora

    src/ancora_error.c
    src/ancora_random.c

    src/logic/ancora_logic.c

    src/linalg/matrices/ancora_mat_arithmetic.c
    src/linalg/matrices/ancora_mat_concatenation.c
    src/linalg/matrices/ancora_mat_constructorDestructor.c
    src/linalg/matrices/ancora_mat_getSet.c
    src/linalg/matrices/ancora_mat_properties.c
    src/linalg/matrices/ancora_mat_templates.c

    src/linalg/vectors/ancora_vec_arithmetic.c
    src/linalg/vectors/ancora_vec_norms.c

    src/sets/interval/ancora_interval_arithmetic.c
    src/sets/interval/ancora_interval_constructorDestructor.c
    src/sets/interval/ancora_interval_containment.c
    src/sets/interval/ancora_interval_properties.c
    src/sets/interval/ancora_interval_templates.c
    src/sets/interval/ancora_interval_duality.c
    src/sets/interval/ancora_interval_randomPoints.c

    src/sets/zonotope/ancora_zonotope_arithmetic.c
    src/sets/zonotope/ancora_zonotope_constructorDestructor.c
    src/sets/zonotope/ancora_zonotope_containment.c
    src/sets/zonotope/ancora_zonotope_properties.c
    src/sets/zonotope/ancora_zonotope_templates.c
    src/sets/zonotope/ancora_zonotope_duality.c
    src/sets/zonotope/ancora_zonotope_randomPoints.c
)


# ===========================================================================
# Include directories
# ===========================================================================

target_include_directories(
    ancora
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)


# ===========================================================================
# Mode selection
# ===========================================================================

if(ANCORA_MODE_SAFE)

    # -----------------------------------------------------------------------
    # SAFE
    # -----------------------------------------------------------------------

    target_compile_definitions(
        ancora
        PUBLIC
            ANCORA_MODE=ANCORA_MODE_SAFE
    )

    target_link_libraries(
        ancora
        PUBLIC
            FLINT::FLINT
    )

    set_target_properties(
        ancora
        PROPERTIES
            OUTPUT_NAME ancora
    )

    message(
        STATUS
        "ancora: SAFE mode enabled"
    )


else()

    # -----------------------------------------------------------------------
    # FAST
    # -----------------------------------------------------------------------

    target_compile_definitions(
        ancora
        PUBLIC
            ANCORA_MODE=ANCORA_MODE_FAST
    )


    if(ANCORA_USE_GPU)

        # ===================================================================
        # FAST + GPU
        # ===================================================================

        # ---------------------------------------------------------------
        # Default architecture
        # ---------------------------------------------------------------

        if(NOT ANCORA_HIP_ARCHITECTURES)

            if(ANCORA_GPU_PLATFORM STREQUAL "nvidia")

                set(
                    ANCORA_HIP_ARCHITECTURES
                    "80"
                )

            else()

                set(
                    ANCORA_HIP_ARCHITECTURES
                    "gfx90a"
                )

            endif()

        endif()


        message(
            STATUS
            "ancora: HIP architecture = ${ANCORA_HIP_ARCHITECTURES}"
        )


        # ---------------------------------------------------------------
        # HIP GPU sources
        #
        # Each source is added exactly once.
        # ---------------------------------------------------------------

        set(
            ANCORA_GPU_SOURCES

            src/linalg/matrices/ancora_mat_arithmetic_gpu.hip.cpp

            src/sets/interval/ancora_interval_duality_gpu.hip.cpp

            src/sets/zonotope/ancora_zonotope_duality_gpu.hip.cpp

            src/sets/zonotope/ancora_zonotope_randomPoints_gpu.hip.cpp
        )


        set_source_files_properties(
            ${ANCORA_GPU_SOURCES}
            PROPERTIES
                LANGUAGE HIP
        )


        target_sources(
            ancora
            PRIVATE
                ${ANCORA_GPU_SOURCES}
        )


        set_target_properties(
            ancora
            PROPERTIES
                HIP_ARCHITECTURES
                    "${ANCORA_HIP_ARCHITECTURES}"

                OUTPUT_NAME
                    ancora_fast_gpu
        )


        # ---------------------------------------------------------------
        # GPU dependencies
        #
        # roc::hipblas itself depends on hip::host. Because find_package(hip)
        # happened before find_package(hipblas), that dependency should now
        # resolve correctly.
        # ---------------------------------------------------------------

        target_link_libraries(
            ancora
            PRIVATE
                hip::device
                roc::hipblas
        )


        target_compile_definitions(
            ancora
            PUBLIC
                ANCORA_USE_GPU=1
        )


        message(
            STATUS
            "ancora: FAST mode with GPU enabled"
        )


    else()

        # ===================================================================
        # FAST, CPU only
        # ===================================================================

        # ---------------------------------------------------------------
        # BLAS
        # ---------------------------------------------------------------

        find_package(BLAS REQUIRED)

        target_link_libraries(
            ancora
            PUBLIC
                BLAS::BLAS
        )


        # ---------------------------------------------------------------
        # Optional CBLAS domatcopy support
        # ---------------------------------------------------------------

        include(CheckSymbolExists)

        set(
            CMAKE_REQUIRED_LIBRARIES
            ${BLAS_LIBRARIES}
        )

        check_symbol_exists(
            cblas_domatcopy
            "cblas.h"
            ANCORA_HAVE_CBLAS_DOMATCOPY
        )

        if(ANCORA_HAVE_CBLAS_DOMATCOPY)

            target_compile_definitions(
                ancora
                PRIVATE
                    ANCORA_HAVE_CBLAS_DOMATCOPY=1
            )

        endif()


        set_target_properties(
            ancora
            PROPERTIES
                OUTPUT_NAME ancora_fast
        )


        message(
            STATUS
            "ancora: FAST mode enabled (CPU only)"
        )

    endif()


    # -----------------------------------------------------------------------
    # HiGHS
    # -----------------------------------------------------------------------

    target_link_libraries(
        ancora
        PUBLIC
            HIGHS::HIGHS
    )

endif()


# ===========================================================================
# Installation
# ===========================================================================

include(GNUInstallDirs)


install(
    TARGETS ancora
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)


install(
    DIRECTORY
        include/ancora
    DESTINATION
        "${CMAKE_INSTALL_INCLUDEDIR}"
)


# ===========================================================================
# Tests
# ===========================================================================

option(
    ANCORA_BUILD_TESTS
    "Build the C unit tests"
    ON
)


if(ANCORA_BUILD_TESTS)

    enable_testing()


    add_executable(
        test_ancora

        tests/test_ancora.c
        tests/test_ancora_error.c
    )


    target_link_libraries(
        test_ancora
        PRIVATE
            ancora
    )


    target_include_directories(
        test_ancora
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/tests"
    )


    add_test(
        NAME test_ancora
        COMMAND test_ancora
    )

endif()


# ===========================================================================
# Optional examples
# ===========================================================================

# option(
#     ANCORA_BUILD_EXAMPLES
#     "Build the C examples"
#     OFF
# )
#
# if(ANCORA_BUILD_EXAMPLES)
#
#     add_executable(
#         example_interval
#         examples/example_interval.c
#     )
#
#     target_link_libraries(
#         example_interval
#         PRIVATE
#             ancora
#     )
#
# endif()


# ===========================================================================
# Optional Python bindings
# ===========================================================================

# option(
#     ANCORA_BUILD_PYTHON
#     "Build the pybind11 Python bindings"
#     OFF
# )
#
# if(ANCORA_BUILD_PYTHON)
#     add_subdirectory(bindings/python)
# endif()
