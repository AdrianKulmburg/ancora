/*
 * ancora_interval_templates.c
 *
 * Description
 * -----------
 * Constructors for canonical and random intervals.
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

#include "ancora/sets/interval/ancora_interval_templates.h"


ancora_status ancora_interval_initRandom_uniform(ancora_interval *I)
/* Fills an already-initialized interval with random bounds.
 * For each dimension i, two samples are drawn from the uniform distribution on
 * [-1, 1]. The smallest number is assigned to lowerBound[i], the largest to
 * upperBound[i], ensuring that the interval is never empty.
 *
 * INPUT:
 *      I               : Pointer to an initialized ancora_interval instance
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*ANCORA_DEFAULT_PREC)
 *      where n is the dimension of the interval.
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */
{
    // Check that I is well-defined
    if (I == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I is NULL; it should point to a valid ancora_interval instance.");
    }

    slong n;
    ANCORA_TRY(ancora_interval_dimension(I, &n));

    for (slong i = 0; i < n; i++)
    {
        double s1, s2;

        ANCORA_TRY(ancora_random_uniform(-1.0, 1.0, &s1));
        ANCORA_TRY(ancora_random_uniform(-1.0, 1.0, &s2));

        if (s1 <= s2)
        {
#if ANCORA_MODE == ANCORA_MODE_SAFE
            arb_set_d(arb_mat_entry(I->lowerBound.repr, i, 0), s1);
            arb_set_d(arb_mat_entry(I->upperBound.repr, i, 0), s2);
#elif ANCORA_MODE == ANCORA_MODE_FAST
            I->lowerBound.repr[i] = s1;
            I->upperBound.repr[i] = s2;
#endif
        }
        else
        {
#if ANCORA_MODE == ANCORA_MODE_SAFE
            arb_set_d(arb_mat_entry(I->lowerBound.repr, i, 0), s2);
            arb_set_d(arb_mat_entry(I->upperBound.repr, i, 0), s1);
#elif ANCORA_MODE == ANCORA_MODE_FAST
            I->lowerBound.repr[i] = s2;
            I->upperBound.repr[i] = s1;
#endif
        }
    }

    return ANCORA_OK;
}
