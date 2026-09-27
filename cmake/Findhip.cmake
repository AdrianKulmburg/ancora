# Findhip.cmake
#
# Finds the HIP runtime (AMD ROCm, or NVIDIA via HIP-over-CUDA).
#
# This module is a fallback for when ROCm's own CMake package config
# (hip-config.cmake, normally under /opt/rocm/lib/cmake/hip) is not
# discoverable by find_package(hip). It provides the SAME imported target
# name (hip::device) that ROCm's config provides, so existing
# target_link_libraries(... hip::device) calls keep working unchanged.
#
# Defines:
#   hip_FOUND          - True if HIP was found
#   HIP_INCLUDE_DIRS   - HIP include directories
#   HIP_LIBRARIES      - HIP runtime libraries to link against
#   HIP_HIPCC          - Path to the hipcc wrapper (informational only; see
#                        the note below)
#   HIP_CLANG          - Path to a clang-based HIP compiler (the kind CMake's
#                        enable_language(HIP) accepts as CMAKE_HIP_COMPILER)
#   HIP_VERSION        - HIP version string (if determinable)
#
# Imported target:
#   hip::device        - the HIP runtime library + include dirs
#
# The platform (AMD vs NVIDIA) is taken from the ANCORA_GPU_PLATFORM cache
# variable when it is set (as the top-level CMakeLists.txt does); otherwise
# it is probed by looking for libamdhip64 (AMD) vs libcudart (NVIDIA).
#
# NOTE on HIP_HIPCC vs HIP_CLANG: CMake 3.28+ rejects the hipcc wrapper as
# CMAKE_HIP_COMPILER ("Use Clang directly, or let CMake pick a default").
# HIP_HIPCC is therefore reported for information only and must NOT be fed
# into CMAKE_HIP_COMPILER. If you need to set CMAKE_HIP_COMPILER explicitly,
# use HIP_CLANG (a clang-based HIP compiler), or let CMake pick its default.

# ---------------------------------------------------------------------------
# 1. Locate the HIP root directory.
# ---------------------------------------------------------------------------
set(_hip_hints)
if(DEFINED HIP_PATH)
    list(APPEND _hip_hints "${HIP_PATH}")
endif()
if(DEFINED ROCM_PATH)
    list(APPEND _hip_hints "${ROCM_PATH}")
endif()
list(APPEND _hip_hints
    /opt/rocm
    /opt/rocm/hip
    /usr
    /usr/local
)

find_path(HIP_INCLUDE_DIR
    NAMES hip/hip_runtime.h
    HINTS ${_hip_hints}
    PATH_SUFFIXES include
)

# ---------------------------------------------------------------------------
# 2. Determine the platform and locate the runtime library.
# ---------------------------------------------------------------------------
set(_hip_platform "${ANCORA_GPU_PLATFORM}")
if(NOT _hip_platform)
    # Probe: if libamdhip64 exists it is an AMD/ROCm install; otherwise fall
    # back to the CUDA runtime (HIP-over-CUDA on NVIDIA).
    find_library(_hip_amd_lib NAMES amdhip64
        HINTS ${_hip_hints}
        PATH_SUFFIXES lib lib64)
    if(_hip_amd_lib)
        set(_hip_platform "amd")
    else()
        set(_hip_platform "nvidia")
    endif()
endif()

if(_hip_platform STREQUAL "nvidia")
    find_library(HIP_LIBRARY NAMES cudart
        HINTS ${_hip_hints}
        PATH_SUFFIXES lib lib64)
else()
    find_library(HIP_LIBRARY NAMES amdhip64
        HINTS ${_hip_hints}
        PATH_SUFFIXES lib lib64)
endif()

# ---------------------------------------------------------------------------
# 3. Locate the HIP compilers.
# ---------------------------------------------------------------------------
# hipcc wrapper (informational only - CMake 3.28+ rejects it as
# CMAKE_HIP_COMPILER).
find_program(HIP_HIPCC
    NAMES hipcc hipcc.bin
    HINTS ${_hip_hints}
    PATH_SUFFIXES bin
)

# A clang-based HIP compiler, which CMake's enable_language(HIP) accepts as
# CMAKE_HIP_COMPILER. On a ROCm install this is typically the LLVM clang++
# shipped with ROCm (e.g. /opt/rocm/llvm/bin/clang++).
find_program(HIP_CLANG
    NAMES clang++ clang++-HIP clang
    HINTS ${_hip_hints}
    PATH_SUFFIXES llvm/bin bin
)

# ---------------------------------------------------------------------------
# 4. Best-effort version detection.
# ---------------------------------------------------------------------------
set(HIP_VERSION "")
if(HIP_INCLUDE_DIR)
    set(_hip_version_header "${HIP_INCLUDE_DIR}/hip/hip_version.h")
    if(EXISTS "${_hip_version_header}")
        file(STRINGS "${_hip_version_header}" _hip_ver_major
            REGEX "#define HIP_VERSION_MAJOR[ \t]+[0-9]+")
        file(STRINGS "${_hip_version_header}" _hip_ver_minor
            REGEX "#define HIP_VERSION_MINOR[ \t]+[0-9]+")
        file(STRINGS "${_hip_version_header}" _hip_ver_patch
            REGEX "#define HIP_VERSION_PATCH[ \t]+[0-9]+")
        if(_hip_ver_major AND _hip_ver_minor AND _hip_ver_patch)
            string(REGEX REPLACE ".*[ \t]([0-9]+)" "\\1" _hip_ver_major "${_hip_ver_major}")
            string(REGEX REPLACE ".*[ \t]([0-9]+)" "\\1" _hip_ver_minor "${_hip_ver_minor}")
            string(REGEX REPLACE ".*[ \t]([0-9]+)" "\\1" _hip_ver_patch "${_hip_ver_patch}")
            set(HIP_VERSION "${_hip_ver_major}.${_hip_ver_minor}.${_hip_ver_patch}")
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# 5. Standard find_package handling.
# ---------------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(hip
    REQUIRED_VARS HIP_LIBRARY HIP_INCLUDE_DIR
    VERSION_VAR HIP_VERSION
)

mark_as_advanced(HIP_INCLUDE_DIR HIP_LIBRARY HIP_HIPCC HIP_CLANG)

if(hip_FOUND)
    set(HIP_INCLUDE_DIRS ${HIP_INCLUDE_DIR})
    set(HIP_LIBRARIES ${HIP_LIBRARY})

    if(NOT TARGET hip::device)
        add_library(hip::device UNKNOWN IMPORTED)
        set_target_properties(hip::device PROPERTIES
            IMPORTED_LOCATION "${HIP_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIR}"
        )
    endif()
endif()
