/*
 * ancora_zonotope_constructorDestructor.h
 *
 * Description
 * -----------
 * Constructor and destructor for ancora_zonotope.
 *
 * File Information
 * ----------------
 * Created:       2026-09-25
 * Last modified: 2026-09-28
 * Authors:       Adrian Kulmburg
 *
 * License
 * -------
 * Copyright (c) 2026 Adrian Kulmburg <adrian.kulmburg@kit.edu>
 * SPDX-License-Identifier: MIT
 */

#include "ancora/sets/zonotope/ancora_zonotope_constructorDestructor.h"

ancora_status ancora_zonotope_init(ancora_zonotope *Z,
                                   const ancora_vec *c,
                                   const ancora_mat *G)
/* Initializes a zonotope from a given center vector c and generator matrix G.
 * INPUT:
 *      Z               : Pointer to an (uninitialized) ancora zonotope, to be
 *                        initialized
 *      c               : Center vector (length n)
 *      G               : Generator matrix (n x p)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*p*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */
{
    // Check that Z, c, and G are well-defined
    if (Z == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z is NULL; it should point to a valid ancora_zonotope instance.");
    }
    if (c == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer c is NULL; it should point to a valid ancora_vec instance.");
    }
    if (G == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer G is NULL; it should point to a valid ancora_mat instance.");
    }

    // Verify that c is a vector
    bool isVector;
    ANCORA_TRY(ancora_mat_isVector(c, &isVector));
    if (!isVector)
    {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer c is not a vector; it should point to a valid ancora_vec instance.");
    }

    // Dimension check
    if (G->nrows != c->nrows) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "G has %ld rows, but c has dimension %ld; they need to be the same.",
                      (long)G->nrows, (long)c->nrows);
    }

    /* Validate c and G: no NaN, no +-inf.
     *
     * There is no arb_is_nan in Arb's API. An arb_t is NaN when its midpoint
     * (an arf_t) is NaN, so the check goes through the midpoint explicitly
     * via arf_is_nan(arb_midref(x)). Strictly, arb_is_finite(x) already
     * implies "not NaN" (a NaN midpoint makes the ball non-finite too), so
     * the arf_is_nan(...) check below is redundant with the arb_is_finite
     * check right next to it -- kept anyway to make the NaN case explicit
     * in the error path, matching the original intent of this check.
     */
    for (slong i = 0; i < c->nrows; i++)
    {
#if ANCORA_MODE == ANCORA_MODE_SAFE
        if (arf_is_nan(arb_midref(arb_mat_entry(c->repr, i, 0))) || !arb_is_finite(arb_mat_entry(c->repr, i, 0)))
        {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Vector c has NaN or +-inf in entry %ld; must not be NaN or +-inf.", (long)i);
        }
        for (slong j = 0; j < G->ncols; j++)
        {
            if (arf_is_nan(arb_midref(arb_mat_entry(G->repr, i, j))) || !arb_is_finite(arb_mat_entry(G->repr, i, j)))
            {
                ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Matrix G has NaN or +-inf in entry (%ld,%ld); must not be NaN or +-inf.", (long)i, (long)j);
            }
        }
#elif ANCORA_MODE == ANCORA_MODE_FAST
        if (!isfinite(c->repr[i]))
        {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Vector c has NaN or +-inf in entry %ld; must not be NaN or +-inf.", (long)i);
        }
        for (slong j = 0; j < G->ncols; j++)
        {
            if (!isfinite(G->repr[i*G->ncols + j]))
            {
                ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Matrix G has NaN or +-inf in entry (%ld,%ld); must not be NaN or +-inf.", (long)i, (long)j);
            }
        }
#endif
    }

    ANCORA_TRY(ancora_vec_init(&Z->c, G->nrows));
    ANCORA_TRY(ancora_mat_init(&Z->G, G->nrows, G->ncols));

    ANCORA_TRY(ancora_vec_copy(&Z->c, c));
    ANCORA_TRY(ancora_mat_copy(&Z->G, G));

    return ANCORA_OK;
}

ancora_status ancora_zonotope_free(ancora_zonotope *Z)
/* Frees a previously initialized ancora zonotope.
 * INPUT:
 *      Z               : Pointer to an initialized ancora_zonotope instance,
 *                        to be freed
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(dimension) in ANCORA_MODE_FAST mode.
 *      O(dimension*nrGenerators) in ANCORA_MODE_SAFE mode (one arb_clear per
 *      entry; independent of ANCORA_DEFAULT_PREC).
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):       Adrian Kulmburg
 */
{
    if (Z == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z is NULL; it should point to a valid ancora_zonotope instance.");
    }

    ANCORA_TRY(ancora_vec_free(&Z->c));
    ANCORA_TRY(ancora_mat_free(&Z->G));

    return ANCORA_OK;
}
