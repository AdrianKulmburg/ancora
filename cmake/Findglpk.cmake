# ============================================================================
# FindGLPK.cmake
# ============================================================================
#
# Locates the GNU Linear Programming Kit (GLPK) and exposes it as the
# imported target GLPK::GLPK, mirroring how FindFLINT.cmake/FindHIGHS.cmake
# in this same directory expose FLINT::FLINT / HIGHS::HIGHS.
#
# GLPK does not ship its own CMake package config, so this is a classic
# find_path/find_library module rather than a find_package(... CONFIG)
# passthrough.
#
# Result variables:
#   GLPK_FOUND
#   GLPK_INCLUDE_DIR
#   GLPK_LIBRARY
#
# Imported target:
#   GLPK::GLPK
#
# ============================================================================

find_path(
    GLPK_INCLUDE_DIR
    NAMES glpk.h
    PATHS
        /usr/include
        /usr/local/include
)

find_library(
    GLPK_LIBRARY
    NAMES glpk
    PATHS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
    GLPK
    REQUIRED_VARS
        GLPK_LIBRARY
        GLPK_INCLUDE_DIR
)

if(GLPK_FOUND AND NOT TARGET GLPK::GLPK)

    add_library(GLPK::GLPK UNKNOWN IMPORTED)

    set_target_properties(
        GLPK::GLPK
        PROPERTIES
            IMPORTED_LOCATION "${GLPK_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${GLPK_INCLUDE_DIR}"
    )

endif()

mark_as_advanced(GLPK_INCLUDE_DIR GLPK_LIBRARY)
