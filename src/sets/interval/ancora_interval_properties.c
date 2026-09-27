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

#include "ancora/sets/interval/ancora_interval_properties.h"

ancora_status ancora_interval_dimension(const ancora_interval *I, slong *dimension)
/* Determines the dimension of an initialized interval.
 *
 * INPUT:
 *      I               : ancora_interval instance
 *      dimension       : Pointer to an slong that will be filled with the
 *                        dimension of I
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(1)
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):       Adrian Kulmburg
 */
{
    // Check that interval is well-defined
    if (I == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I is NULL; it should point to a valid ancora_interval instance.");
    }

    *dimension = I->lowerBound.nrows;
    return ANCORA_OK;
}
