/*
 * ancora_zonotope_templates.c
 *
 * Description
 * -----------
 * Constructors for canonical and random zonotopes.
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

#include "ancora/sets/zonotope/ancora_zonotope_templates.h"

ancora_status ancora_zonotope_initRandom_uniform(ancora_zonotope *Z)
/* Fills an already-initialized zonotope with a random center and generators.
 * Each entry of the center vector and generator matrix is sampled from a
 * uniform distribution on [-1,1].
 *
 * INPUT:
 *      Z               : Pointer to an initialized ancora_zonotope instance
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*m*ANCORA_DEFAULT_PREC)
 *      where n is the dimension of the zonotope, m the number of generators.
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */
{
    // Check that Z is well-defined
    if (Z == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z is NULL; it should point to a valid ancora_zonotope instance.");
    }

    // Do the random generation
    ANCORA_TRY(ancora_vec_randomUniform(&Z->c));
    ANCORA_TRY(ancora_mat_randomUniform(&Z->G));

    return ANCORA_OK;
}
