# FindHIGHS.cmake
# Finds the HiGHS linear programming solver
#
# This will define the following variables:
#   HIGHS_FOUND        - True if HiGHS is found
#   HIGHS_INCLUDE_DIRS  - HiGHS include directories
#   HIGHS_LIBRARIES     - HiGHS libraries to link against
#
# and the following imported target:
#   HIGHS::HIGHS
#
# HiGHS is used by the zonotope containment code (ancora_zonotope_containment.c)
# in ANCORA_MODE_FAST mode only; SAFE mode does not need it.

find_path(HIGHS_INCLUDE_DIR
    NAMES highs/interfaces/highs_c_api.h
    PATHS /usr/include /usr/local/include
)

# The directory that contains the "highs" subdirectory (i.e. <prefix>/include).
get_filename_component(HIGHS_INCLUDE_ROOT "${HIGHS_INCLUDE_DIR}" DIRECTORY)

# The directory that contains the relative includes used by highs_c_api.h
# (i.e. <prefix>/include/highs).
set(HIGHS_INCLUDE_HIGHS_DIR "${HIGHS_INCLUDE_DIR}/highs")

find_library(HIGHS_LIBRARY
    NAMES highs
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HIGHS
    REQUIRED_VARS HIGHS_LIBRARY HIGHS_INCLUDE_DIR
)

mark_as_advanced(HIGHS_INCLUDE_DIR HIGHS_LIBRARY)

if(HIGHS_FOUND)
    set(HIGHS_INCLUDE_DIRS
        "${HIGHS_INCLUDE_ROOT}"
        "${HIGHS_INCLUDE_HIGHS_DIR}")
    set(HIGHS_LIBRARIES ${HIGHS_LIBRARY})

    if(NOT TARGET HIGHS::HIGHS)
        add_library(HIGHS::HIGHS UNKNOWN IMPORTED)
        set_target_properties(HIGHS::HIGHS PROPERTIES
            IMPORTED_LOCATION "${HIGHS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${HIGHS_INCLUDE_DIRS}"
        )
    endif()
endif()
