/*
 * ancora_interval_properties.c
 *
 * Description
 * -----------
 * Elementary properties for intervals.
 *
 * File Information
 * ----------------
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Authors:       Adrian Kulmburg
 *
 * License
 * -------
 * Copyright (c) 2026 Adrian Kulmburg <adrian.kulmburg@kit.edu>
 * SPDX-License-Identifier: MIT
 */

#include "ancora/sets/zonotope/ancora_zonotope_properties.h"

ancora_status ancora_zonotope_dimension(const ancora_zonotope *Z, slong *dimension)
/* Determines the dimension of an initialized zonotope.
 *
 * INPUT:
 *      Z               : ancora_zonotope instance
 *      dimension       : Pointer to an slong that will be filled with the
 *                        dimension of Z
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(1)
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */
{
    // Check that the zonotope is well-defined
    if (Z == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z is NULL; it should point to a valid ancora_zonotope instance.");
    }

    *dimension = Z->G.nrows;
    return ANCORA_OK;
}
