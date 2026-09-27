# Top-level CMakeLists.txt for the ancora library.
#
# IMPORTANT (compute mode): the numeric payload types (ancora_mat / ancora_vec)
# are mode-conditional. A build with ANCORA_MODE_SAFE=ON (FLINT/ARB-backed) is
# ABI-INCOMPATIBLE with one built with ANCORA_MODE_SAFE=OFF (plain C types).
# They must never be mixed. Set ANCORA_MODE_SAFE once (or via -D) and use the
# same value for the core library and every binding layer.
#
# The code branches on the selector macro ANCORA_MODE (see
# include/ancora/ancora_config.h), whose two values are ANCORA_MODE_SAFE (=0)
# and ANCORA_MODE_FAST (=1). The CMake option ANCORA_MODE_SAFE below is the
# user-facing switch; it is translated into the ANCORA_MODE compile definition
# that the source actually tests. Do NOT redefine ANCORA_MODE_SAFE itself as a
# compile definition -- the header already defines it as a value, and
# redefining it breaks the `#if ANCORA_MODE == ANCORA_MODE_SAFE` branches.
#
# IMPORTANT (GPU): ANCORA_USE_GPU is a SEPARATE axis, only meaningful when
# ANCORA_MODE_SAFE=OFF (there is no GPU-accelerated FLINT path). It does
# NOT change ancora_mat/ancora_vec's layout, only which implementation
# ancora_mat_add (and friends) dispatches to internally, so it is not an
# ABI concern the way ANCORA_MODE_SAFE is.
#
# The GPU kernel is written ONCE in HIP (src/linalg/matrices/
# ancora_mat_arithmetic_gpu.hip.cpp) and can target either platform via
# ANCORA_GPU_PLATFORM:
#   - "amd"    : compiles+runs natively on AMD/ROCm hardware (hipcc -> real
#                execution and testing on machines with an AMD GPU)
#   - "nvidia" : HIP's headers wrap the CUDA runtime; hipcc invokes nvcc
#                underneath, producing real CUDA code from the SAME source
#                file
#
# There are three ways to build ancora, each producing a differently-named
# static library:
#   - SAFE mode            (ANCORA_MODE_SAFE=ON)              -> libancora
#   - FAST mode, no GPU    (ANCORA_MODE_SAFE=OFF, default)    -> libancora_fast
#   - FAST mode, with GPU  (ANCORA_MODE_SAFE=OFF
#                           + ANCORA_USE_GPU=ON)              -> libancora_fast_gpu
# The three are mutually exclusive configurations of a single build: pick one
# and configure with it. They must never be mixed in one process (SAFE and
# FAST are ABI-incompatible; the GPU and non-GPU FAST builds differ only in
# which implementation the arithmetic dispatches to, but are still built
# separately so the GPU kernel is only ever linked where it is used).
cmake_minimum_required(VERSION 3.21)

# Declare the GPU options BEFORE project() so HIP_PLATFORM can be set in the
# environment before find_package(hip) runs, and so a non-GPU build never
# needs hipcc/nvcc at all.
option(ANCORA_USE_GPU "Build the fast flavor's GPU kernel (requires HIP + ROCm or HIP-over-CUDA, and ANCORA_MODE_SAFE=OFF)" OFF)
set(ANCORA_GPU_PLATFORM "amd" CACHE STRING "GPU platform for the HIP kernel: 'amd' or 'nvidia'")
set_property(CACHE ANCORA_GPU_PLATFORM PROPERTY STRINGS amd nvidia)

option(ANCORA_MODE_SAFE "Build the safe flavor (requires FLINT)" OFF)

if(ANCORA_USE_GPU AND ANCORA_MODE_SAFE)
    message(FATAL_ERROR
        "ANCORA_USE_GPU=ON requires ANCORA_MODE_SAFE=OFF: there is no "
        "GPU-accelerated FLINT/ARB path. Configure with "
        "-DANCORA_MODE_SAFE=OFF -DANCORA_USE_GPU=ON.")
endif()

# HIP's CMake package reads the HIP_PLATFORM *environment* variable, so it
# must be set before find_package(hip) runs (below, after project()).
if(ANCORA_USE_GPU)
    if(ANCORA_GPU_PLATFORM STREQUAL "nvidia")
        set(ENV{HIP_PLATFORM} "nvidia")
    else()
        set(ENV{HIP_PLATFORM} "amd")
    endif()
endif()

# CXX is needed whenever the GPU kernel is built, since hipcc compiles the
# .hip.cpp file as C++; a non-GPU build stays pure C.
if(ANCORA_USE_GPU)
    project(ancora C CXX)
else()
    project(ancora C)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Our custom Find modules (FLINT, HIGHS) live in the cmake/ directory.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# FLINT/ARB is required in safe mode only.
if(ANCORA_MODE_SAFE)
    find_package(FLINT REQUIRED)
endif()

# HiGHS is required in fast mode only (used by the zonotope containment code).
if(NOT ANCORA_MODE_SAFE)
    find_package(HIGHS REQUIRED)
endif()

# Core sources are always C. The GPU kernel source (.hip.cpp) is added
# separately below so a non-GPU build never even sees it or needs hipcc.
#
# NOTE: the ancora_mat type lives in the header (include/ancora/linalg/
# matrices/ancora_mat.h); there is no ancora_mat.c. The operations are split
# across the .c files listed here.
#
# This list reflects the files that actually exist under src/. The vector
# operations (ancora_vec_init/free/copy/add/... ) are header-only #define
# macros delegating to the matrix layer, so there are no
# ancora_vec_concatenation.c / _constructorDestructor.c / _getSet.c /
# _templates.c files
add_library(ancora
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

if(ANCORA_USE_GPU)
    # Find HIP itself. This first tries ROCm's own CMake package config
    # (typically under /opt/rocm/lib/cmake/hip); if that is not
    # discoverable, it falls back to our Findhip.cmake module in cmake/.
    find_package(hip REQUIRED)

    # Explicitly define the platform macro hip_runtime.h needs, rather than
    # relying on CMAKE_HIP_PLATFORM (CMake >= 3.28 only) or hipconfig
    # --platform autodetection, either of which may silently not apply on
    # an older CMake.
    if(ANCORA_GPU_PLATFORM STREQUAL "nvidia")
        add_compile_definitions($<$<COMPILE_LANGUAGE:HIP>:__HIP_PLATFORM_NVIDIA__>)
    else()
        add_compile_definitions($<$<COMPILE_LANGUAGE:HIP>:__HIP_PLATFORM_AMD__>)
    endif()

    # The kernels use __global__ / hipLaunchKernelGGL, which only a HIP
    # compiler can compile, so compile the .hip.cpp files with CMake's
    # native HIP language support (available since CMake 3.21) rather than
    # the plain CXX compiler.
    #
    # IMPORTANT: do NOT set CMAKE_HIP_COMPILER to the hipcc wrapper. CMake
    # 3.28+ rejects hipcc as the HIP compiler ("Use Clang directly, or let
    # CMake pick a default"). Let CMake pick its default clang-based HIP
    # compiler. If CMake cannot find one, set CMAKE_HIP_COMPILER to a clang
    # explicitly, e.g. -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++.
    #
    # Tell CMake's HIP support which platform to target. This is what makes
    # it add -D__HIP_PLATFORM_AMD__ (or __HIP_PLATFORM_NVIDIA__) to the
    # compile line; without it hip_runtime.h fails with "Must define exactly
    # one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__". Note this is
    # the CMake variable CMAKE_HIP_PLATFORM, NOT the HIP_PLATFORM environment
    # variable that ROCm's hipcc/hip-config.cmake read.
    set(CMAKE_HIP_PLATFORM "${ANCORA_GPU_PLATFORM}")
    enable_language(HIP)

    # CMake requires a target GPU architecture for HIP sources; pick a
    # platform-appropriate default if the user did not set one.
    if(NOT ANCORA_HIP_ARCHITECTURES)
        if(ANCORA_GPU_PLATFORM STREQUAL "nvidia")
            set(ANCORA_HIP_ARCHITECTURES "80")
        else()
            set(ANCORA_HIP_ARCHITECTURES "gfx90a")
        endif()
    endif()
    set_target_properties(ancora PROPERTIES HIP_ARCHITECTURES "${ANCORA_HIP_ARCHITECTURES}")

    set_source_files_properties(src/linalg/matrices/ancora_mat_arithmetic_gpu.hip.cpp PROPERTIES LANGUAGE HIP)
    target_sources(ancora PRIVATE src/linalg/matrices/ancora_mat_arithmetic_gpu.hip.cpp)
    target_link_libraries(ancora PRIVATE hip::device)
    set_source_files_properties(src/sets/interval/ancora_interval_duality_gpu.hip.cpp PROPERTIES LANGUAGE HIP)
    target_sources(ancora PRIVATE src/sets/interval/ancora_interval_duality_gpu.hip.cpp)
endif()

target_include_directories(ancora PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Translate the user-facing ANCORA_MODE_SAFE option into the ANCORA_MODE
# selector macro that the source actually branches on. ANCORA_MODE_SAFE and
# ANCORA_MODE_FAST are defined as 0 and 1 in ancora_config.h, so passing the
# symbolic name keeps the two sides in sync with the header.
#
# The library's OUTPUT_NAME is chosen here so that each of the three build
# configurations produces a distinct, self-describing artifact:
#   - SAFE mode            -> libancora
#   - FAST mode, no GPU    -> libancora_fast
#   - FAST mode, with GPU  -> libancora_fast_gpu
if(ANCORA_MODE_SAFE)
    target_compile_definitions(ancora PUBLIC ANCORA_MODE=ANCORA_MODE_SAFE)
    target_link_libraries(ancora PUBLIC FLINT::FLINT)
else()
    target_compile_definitions(ancora PUBLIC ANCORA_MODE=ANCORA_MODE_FAST)
    if(ANCORA_USE_GPU)
        set_target_properties(ancora PROPERTIES OUTPUT_NAME ancora_fast_gpu)
    else()
        set_target_properties(ancora PROPERTIES OUTPUT_NAME ancora_fast)
    endif()
    # Link HiGHS (used by the zonotope containment code in FAST mode).
    target_link_libraries(ancora PUBLIC HIGHS::HIGHS)
endif()

if(ANCORA_USE_GPU)
    target_compile_definitions(ancora PUBLIC ANCORA_USE_GPU=1)
    message(STATUS "ancora: GPU kernel enabled, platform = ${ANCORA_GPU_PLATFORM}")
endif()

# Installation rules. `cmake --install` (or `make install`) puts the headers
# under <prefix>/include/ancora and the static library under <prefix>/lib.
# This is what the Docker image uses to make ancora available system-wide.
include(GNUInstallDirs)
install(TARGETS ancora
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ancora DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Optional tests.
option(ANCORA_BUILD_TESTS "Build the C unit tests" ON)
if(ANCORA_BUILD_TESTS)
    enable_testing()

    add_executable(test_ancora
        tests/test_ancora.c
        tests/test_ancora_error.c)

    target_link_libraries(test_ancora PRIVATE ancora)
    target_include_directories(test_ancora PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

    add_test(NAME test_ancora COMMAND test_ancora)
endif()

# Optional examples.
#option(ANCORA_BUILD_EXAMPLES "Build the C examples" OFF)
#if(ANCORA_BUILD_EXAMPLES)
#    add_executable(example_interval examples/example_interval.c)
#    target_link_libraries(example_interval PRIVATE ancora)
#endif()

# Optional Python bindings.
#option(ANCORA_BUILD_PYTHON "Build the pybind11 Python bindings" OFF)
#if(ANCORA_BUILD_PYTHON)
#    add_subdirectory(bindings/python)
#endif()
