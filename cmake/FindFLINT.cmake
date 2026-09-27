# FindFLINT.cmake
# Finds the FLINT library
#
# This will define the following variables:
#   FLINT_FOUND        - True if FLINT is found
#   FLINT_INCLUDE_DIRS  - FLINT include directory
#   FLINT_LIBRARIES     - FLINT libraries to link against
#
# and the following imported target:
#   FLINT::FLINT

find_path(FLINT_INCLUDE_DIR
    NAMES flint/flint.h
    PATHS /usr/include /usr/local/include
)

find_library(FLINT_LIBRARY
    NAMES flint
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
)

# FLINT depends on GMP and MPFR - find them too
find_library(GMP_LIBRARY NAMES gmp)
find_library(MPFR_LIBRARY NAMES mpfr)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FLINT
    REQUIRED_VARS FLINT_LIBRARY FLINT_INCLUDE_DIR
)

mark_as_advanced(FLINT_INCLUDE_DIR FLINT_LIBRARY)

if(FLINT_FOUND)
    set(FLINT_INCLUDE_DIRS ${FLINT_INCLUDE_DIR})
    set(FLINT_LIBRARIES ${FLINT_LIBRARY})

    if(GMP_LIBRARY)
        list(APPEND FLINT_LIBRARIES ${GMP_LIBRARY})
    endif()
    if(MPFR_LIBRARY)
        list(APPEND FLINT_LIBRARIES ${MPFR_LIBRARY})
    endif()

    if(NOT TARGET FLINT::FLINT)
        add_library(FLINT::FLINT UNKNOWN IMPORTED)
        set_target_properties(FLINT::FLINT PROPERTIES
            IMPORTED_LOCATION "${FLINT_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${FLINT_INCLUDE_DIR}"
        )
        if(GMP_LIBRARY OR MPFR_LIBRARY)
            set_property(TARGET FLINT::FLINT APPEND PROPERTY
                INTERFACE_LINK_LIBRARIES "${GMP_LIBRARY};${MPFR_LIBRARY}"
            )
        endif()
    endif()
endif()
