/*
 * ancora_interval_constructorDestructor.c
 *
 * Description
 * -----------
 * Constructor and destructor for ancora_interval. The bound vectors are
 * managed through the linalg/vectors layer (ancora_vec_init/free/copy).
 *
 * File Information
 * ----------------
 * Created:       2026-09-23
 * Last modified: 2026-09-24
 * Authors:       Adrian Kulmburg
 *
 * License
 * -------
 * Copyright (c) 2026 Adrian Kulmburg <adrian.kulmburg@kit.edu>
 * SPDX-License-Identifier: MIT
 */

#include <math.h>

#include "ancora/sets/interval/ancora_interval_constructorDestructor.h"

ancora_status ancora_interval_init(ancora_interval *I,
                                   const ancora_vec *lowerBound,
                                   const ancora_vec *upperBound)
/* Initializes an interval from given lower and upper bound vectors.
 *
 * INPUT:
 *      I               : Pointer to an (uninitialized) ancora interval, to be
 *                        initialized
 *      lowerBound      : Lower bound vector (length n)
 *      upperBound      : Upper bound vector (length n)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */
{
    // Verify that I, lowerBound, upperBound are well-defined
    if (I == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I is NULL; it should point to a valid ancora_interval instance.");
    }
    if (lowerBound == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer lowerBound is NULL; it should point to a valid ancora_vec instance.");
    }
    if (upperBound == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer upperBound is NULL; it should point to a valid ancora_vec instance.");
    }

    // Verify that lowerBound and upperBound are vectors
    bool isVector;
    ANCORA_TRY(ancora_mat_isVector(lowerBound, &isVector));
    if (!isVector)
    {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer lowerBound is not a vector; it should point to a valid ancora_vec instance.");
    }
    ANCORA_TRY(ancora_mat_isVector(upperBound, &isVector));
    if (!isVector)
    {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer upperBound is not a vector; it should point to a valid ancora_vec instance.");
    }

    // Dimension checks
    if (lowerBound->nrows != upperBound->nrows) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "lowerBound has dimension %ld, but upperBound has dimension %ld; they need to be the same.",
                      (long)lowerBound->nrows, (long)upperBound->nrows);
    }

    /* Validate bounds: no NaN. Bounds may be +inf or -inf, and an interval
     * with lowerBound[i] > upperBound[i] is allowed (it is the empty set).
     */
    for (slong i = 0; i < lowerBound->nrows; i++)
    {
#if ANCORA_MODE == ANCORA_MODE_SAFE
        if (arf_is_nan(arb_midref(arb_mat_entry(lowerBound->repr, i, 0)))) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Lower bound entry %ld is NaN; bounds must not be NaN (they may be +inf or -inf).", (long)i);
        }
        if (arf_is_nan(arb_midref(arb_mat_entry(upperBound->repr, i, 0)))) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Upper bound entry %ld is NaN; bounds must not be NaN (they may be +inf or -inf).", (long)i);
        }
#elif ANCORA_MODE == ANCORA_MODE_FAST
        if (isnan(lowerBound->repr[i])) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Lower bound entry %ld is NaN; bounds must not be NaN (they may be +inf or -inf).", (long)i);
        }
        if (isnan(upperBound->repr[i])) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Upper bound entry %ld is NaN; bounds must not be NaN (they may be +inf or -inf).", (long)i);
        }
#endif
    }

    // Set the bounds
    ANCORA_TRY(ancora_vec_init(&I->lowerBound, lowerBound->nrows));
    ANCORA_TRY(ancora_vec_init(&I->upperBound, upperBound->nrows));

    /* Copy the validated bounds into the interval. */
    ANCORA_TRY(ancora_vec_copy(&I->lowerBound, lowerBound));
    ANCORA_TRY(ancora_vec_copy(&I->upperBound, upperBound));

    return ANCORA_OK;
}

ancora_status ancora_interval_free(ancora_interval *I)
/* Frees a previously initialized ancora interval.
 * INPUT:
 *      I               : Pointer to an initialized ancora_interval instance,
 *                        to be freed
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(1) in ANCORA_MODE_FAST mode.
 *      O(n) in ANCORA_MODE_SAFE mode (one arb_clear per bound entry;
 *      independent of ANCORA_DEFAULT_PREC), where n is the dimension of the
 *      interval.
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */
{
    if (I == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I is NULL; it should point to a valid ancora_interval instance.");
    }

    ANCORA_TRY(ancora_vec_free(&I->lowerBound));
    ANCORA_TRY(ancora_vec_free(&I->upperBound));

    return ANCORA_OK;
}
