# ============================================================================
# Top-level CMakeLists.txt for the ancora library
# ============================================================================
#
# Build flavors:
#
#   SAFE:
#       -DANCORA_MODE_SAFE=ON
#       -> libancora.a
#       -> FLINT/ARB
#
#   FAST:
#       -DANCORA_MODE_SAFE=OFF
#       -> libancora_fast.a
#       -> BLAS + HiGHS
#
#   FAST + NVIDIA GPU:
#       -DANCORA_MODE_SAFE=OFF
#       -DANCORA_USE_GPU=ON
#       -DANCORA_GPU_PLATFORM=nvidia
#       -> libancora_fast_gpu.a
#       -> HIP-over-CUDA + hipBLAS + HiGHS
#
# SAFE and FAST are ABI-incompatible and must never be mixed.
#
# GPU is only supported with FAST mode.
#
# For NVIDIA/HIP builds, ANCORA_HIP_ARCHITECTURES must be known BEFORE
# enable_language(HIP), because CMake needs it while initializing HIP.
#
# Example for NVIDIA A100:
#
#   -DANCORA_GPU_PLATFORM=nvidia
#   -DANCORA_HIP_ARCHITECTURES=80
#
# ============================================================================

# NVIDIA HIP language support (HIP-over-CUDA via NVCC) requires CMake 3.28+.
cmake_minimum_required(VERSION 3.28)


# ============================================================================
# Options
# ============================================================================

option(
    ANCORA_MODE_SAFE
    "Build the safe FLINT/ARB-backed flavor"
    OFF
)

option(
    ANCORA_USE_GPU
    "Build the FAST flavor with HIP GPU support"
    OFF
)

set(
    ANCORA_GPU_PLATFORM
    "amd"
    CACHE STRING
    "GPU platform: amd or nvidia"
)

set_property(
    CACHE ANCORA_GPU_PLATFORM
    PROPERTY STRINGS amd nvidia
)

set(
    ANCORA_HIP_ARCHITECTURES
    ""
    CACHE STRING
    "HIP GPU architecture(s), e.g. gfx90a for AMD or 80 for NVIDIA A100"
)

set(
    ANCORA_ROCM_ROOT
    "/opt/rocm"
    CACHE PATH
    "ROCm/HIP installation root"
)


# ============================================================================
# Validate options
# ============================================================================

if(
    NOT ANCORA_GPU_PLATFORM STREQUAL "amd"
    AND NOT ANCORA_GPU_PLATFORM STREQUAL "nvidia"
)
    message(
        FATAL_ERROR
        "ANCORA_GPU_PLATFORM must be either 'amd' or 'nvidia'. "
        "Got '${ANCORA_GPU_PLATFORM}'."
    )
endif()


if(ANCORA_USE_GPU AND ANCORA_MODE_SAFE)
    message(
        FATAL_ERROR
        "ANCORA_USE_GPU=ON requires ANCORA_MODE_SAFE=OFF."
    )
endif()


# ============================================================================
# GPU/HIP environment
# ============================================================================
#
# HIP's package configuration uses HIP_PLATFORM.
#
# This must be established before the HIP package is found and before HIP
# language support is initialized.
# ============================================================================

if(ANCORA_USE_GPU)

    if(ANCORA_GPU_PLATFORM STREQUAL "nvidia")
        set(ENV{HIP_PLATFORM} "nvidia")
    else()
        set(ENV{HIP_PLATFORM} "amd")
    endif()

    message(
        STATUS
        "ancora: GPU platform = ${ANCORA_GPU_PLATFORM}"
    )

    message(
        STATUS
        "ancora: ROCm root = ${ANCORA_ROCM_ROOT}"
    )

endif()


# ============================================================================
# Project
# ============================================================================
#
# CMake's native HIP language support is enabled later, after the HIP package
# has been found and the architecture has been established.
# ============================================================================

if(ANCORA_USE_GPU)
    project(ancora C CXX)
else()
    project(ancora C)
endif()


set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)


# ============================================================================
# Custom CMake modules
# ============================================================================

list(
    APPEND
    CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
)


# ============================================================================
# Dependencies: SAFE
# ============================================================================

if(ANCORA_MODE_SAFE)

    find_package(FLINT REQUIRED)

endif()


# ============================================================================
# Dependencies: FAST
# ============================================================================

if(NOT ANCORA_MODE_SAFE)

    find_package(HIGHS REQUIRED)

endif()


# ============================================================================
# GPU dependency setup
# ============================================================================
#
# Important:
#
#   find_package(hip)
#   MUST happen before find_package(hipblas)
#
# because roc::hipblas has a dependency on hip::host.
# ============================================================================

if(ANCORA_USE_GPU)

    # ------------------------------------------------------------------------
    # Make ROCm packages discoverable.
    # ------------------------------------------------------------------------

    list(
        APPEND
        CMAKE_PREFIX_PATH
        "${ANCORA_ROCM_ROOT}"
        "${ANCORA_ROCM_ROOT}/lib/cmake"
    )

    # Some ROCm installations use versioned directories such as:
    #
    #   /opt/rocm-6.3.1
    #
    # while /opt/rocm may be a symlink.
    #
    file(
        GLOB
        ANCORA_ROCM_VERSIONED_DIRS
        "/opt/rocm-*"
    )

    if(ANCORA_ROCM_VERSIONED_DIRS)
        list(
            APPEND
            CMAKE_PREFIX_PATH
            ${ANCORA_ROCM_VERSIONED_DIRS}
        )
    endif()


    # ------------------------------------------------------------------------
    # HIP
    # ------------------------------------------------------------------------

    find_package(
        hip
        CONFIG
        REQUIRED
        PATHS
            "${ANCORA_ROCM_ROOT}/lib/cmake/hip"
        NO_DEFAULT_PATH
    )


    # ------------------------------------------------------------------------
    # hipBLAS
    # ------------------------------------------------------------------------
    #
    # HIP is intentionally found first because hipBLAS's imported targets
    # depend on hip::host.
    # ------------------------------------------------------------------------

    find_package(
        hipblas
        CONFIG
        REQUIRED
        PATHS
            "${ANCORA_ROCM_ROOT}/lib/cmake/hipblas"
        NO_DEFAULT_PATH
    )


    # ------------------------------------------------------------------------
    # Determine HIP architecture BEFORE enabling HIP
    # ------------------------------------------------------------------------
    #
    # For NVIDIA, the install script should normally pass:
    #
    #     -DANCORA_HIP_ARCHITECTURES=80
    #
    # for an A100.
    #
    # If it wasn't provided, use a conservative NVIDIA default of 80.
    # ------------------------------------------------------------------------

    if(NOT ANCORA_HIP_ARCHITECTURES)

        if(ANCORA_GPU_PLATFORM STREQUAL "nvidia")
            set(ANCORA_HIP_ARCHITECTURES "80")
        else()
            set(ANCORA_HIP_ARCHITECTURES "gfx90a")
        endif()

    endif()


    # ------------------------------------------------------------------------
    # CMake's HIP architecture variable MUST be set before enable_language(HIP)
    # ------------------------------------------------------------------------

    set(
        CMAKE_HIP_ARCHITECTURES
        "${ANCORA_HIP_ARCHITECTURES}"
        CACHE STRING
        "HIP architecture(s)"
        FORCE
    )


    message(
        STATUS
        "ancora: HIP architecture = ${ANCORA_HIP_ARCHITECTURES}"
    )


    # ------------------------------------------------------------------------
    # CMake 3.28+ is required for NVIDIA HIP language support.
    #
    # On NVIDIA, CMake's native HIP language support uses NVCC as the HIP
    # compiler. HIP_PLATFORM is still set above so ROCm's HIP package and
    # hipBLAS configuration select the NVIDIA backend.
    # ------------------------------------------------------------------------

    set(
        CMAKE_HIP_PLATFORM
        "${ANCORA_GPU_PLATFORM}"
    )


    # ------------------------------------------------------------------------
    # Enable HIP
    #
    # For NVIDIA, CMake 3.28+ selects NVCC for the HIP language when
    # CMAKE_HIP_PLATFORM is "nvidia". The install script also passes
    # CMAKE_HIP_COMPILER explicitly to avoid compiler autodetection ambiguity.
    # ------------------------------------------------------------------------

    enable_language(HIP)


    # ------------------------------------------------------------------------
    # Explicit HIP platform macro
    # ------------------------------------------------------------------------

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


# ============================================================================
# Core library
# ============================================================================

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


# ============================================================================
# Include directories
# ============================================================================

target_include_directories(
    ancora
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)


# ============================================================================
# SAFE mode
# ============================================================================

if(ANCORA_MODE_SAFE)

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


# ============================================================================
# FAST mode
# ============================================================================

else()

    target_compile_definitions(
        ancora
        PUBLIC
            ANCORA_MODE=ANCORA_MODE_FAST
    )


    # ========================================================================
    # FAST + GPU
    # ========================================================================

    if(ANCORA_USE_GPU)

        set(
            ANCORA_GPU_SOURCES

            src/linalg/matrices/ancora_mat_arithmetic_gpu.hip.cpp

            src/sets/interval/ancora_interval_duality_gpu.hip.cpp

            src/sets/zonotope/ancora_zonotope_duality_gpu.hip.cpp

            src/sets/zonotope/ancora_zonotope_randomPoints_gpu.hip.cpp
        )


        # CMake should treat these files as HIP sources.
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


        # hip::device provides the HIP device-side build/link support.
        #
        # roc::hipblas is the hipBLAS imported target. On NVIDIA this is the
        # HIP-over-CUDA implementation.
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
            "ancora: FAST + GPU enabled"
        )


    # ========================================================================
    # FAST CPU
    # ========================================================================

    else()

        find_package(BLAS REQUIRED)


        target_link_libraries(
            ancora
            PUBLIC
                BLAS::BLAS
        )


        # --------------------------------------------------------------------
        # Optional cblas_domatcopy
        # --------------------------------------------------------------------

        include(CheckSymbolExists)

        set(
            CMAKE_REQUIRED_LIBRARIES
            ${BLAS_LIBRARIES}
        )

        if(BLAS_INCLUDE_DIRS)

            set(
                CMAKE_REQUIRED_INCLUDES
                ${BLAS_INCLUDE_DIRS}
            )

        endif()


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
            "ancora: FAST CPU mode enabled"
        )

    endif()


    # ------------------------------------------------------------------------
    # HiGHS
    # ------------------------------------------------------------------------

    target_link_libraries(
        ancora
        PUBLIC
            HIGHS::HIGHS
    )

endif()


# ============================================================================
# Installation
# ============================================================================

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


# ============================================================================
# Tests
# ============================================================================

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


# ============================================================================
# Optional examples
# ============================================================================

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


# ============================================================================
# Optional Python bindings
# ============================================================================

# option(
#     ANCORA_BUILD_PYTHON
#     "Build the pybind11 Python bindings"
#     OFF
# )
#
# if(ANCORA_BUILD_PYTHON)
#     add_subdirectory(bindings/python)
# endif()
