/*
 * ancora_mat_properties.c
 *
 * Description
 * -----------
 * Elementary properties of an ancora_mat.
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

#include "ancora/linalg/matrices/ancora_mat_properties.h"

ancora_status ancora_mat_isVector(const ancora_mat *m, bool *isVector)
/* Checks whether a matrix is actually a vector (single column).
 * INPUT:
 *      m               : ancora_mat instance
 *      isVector        : Pointer to a Boolean that will be set to true if m is
 *                        a vector, false otherwise
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
    // Check that the matrix is valid
    if (m == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer m is NULL; it should point to a valid ancora_mat instance.");
    }

    if (m->ncols == 1)
    {
        *isVector = true;
    }
    else
    {
        *isVector = false;
    }

    return ANCORA_OK;
}

