/*
 * ancora_interval_containment.c
 *
 * Description
 * -----------
 * Containment checks for ancora_interval.
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

#include "ancora/sets/interval/ancora_interval_containment.h"

ancora_status ancora_interval_containsPoint(const ancora_interval *I,
                                       const ancora_vec *p,
                                       ancora_truth *contained)
/* Checks whether the point p is contained in the interval I.
 *
 * INPUT:
 *      I               : Pointer to an initialized ancora_interval instance
 *      p               : Point to test, a column vector of length I->dimension
 *      contained       : Output; set to true iff p is contained in I
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(dimension*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */
{
    // Check that I and p are well-defined
    if (I == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I is NULL; it should point to a valid ancora_interval instance.");
    }
    if (p == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer p is NULL; it should point to a valid ancora_vec instance.");
    }

    // Verify that p is a vector
    bool isVector;
    ANCORA_TRY(ancora_mat_isVector(p, &isVector));
    if (!isVector)
    {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer p is not a vector; it should point to a valid ancora_vec instance.");
    }

    slong n;
    ANCORA_TRY(ancora_interval_dimension(I, &n));

    // Dimension check
    if (n != p->nrows) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Interval I has dimension %ld, point p has dimension %ld; they need to be the same.",
                      (long)n, (long)p->nrows);
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_t lo, hi, pi;
    arb_init(lo);
    arb_init(hi);
    arb_init(pi);
    for (slong i = 0; i < n; i++) {
        ANCORA_TRY(ancora_vec_get(&I->lowerBound, i, lo));
        ANCORA_TRY(ancora_vec_get(&I->upperBound, i, hi));
        ANCORA_TRY(ancora_vec_get(p, i, pi));
        if (!arb_ge(pi, lo) || !arb_le(pi, hi)) {
            ancora_setNo(contained);
            arb_clear(lo);
            arb_clear(hi);
            arb_clear(pi);
            return ANCORA_OK;
        }
    }
    ancora_setYes(contained);
    arb_clear(lo);
    arb_clear(hi);
    arb_clear(pi);
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double lo, hi, pi;
    for (slong i = 0; i < n; i++) {
        ANCORA_TRY(ancora_vec_get(&I->lowerBound, i, &lo));
        ANCORA_TRY(ancora_vec_get(&I->upperBound, i, &hi));
        ANCORA_TRY(ancora_vec_get(p, i, &pi));
        if (!(lo-ANCORA_TOL <= pi && pi <= hi+ANCORA_TOL)) {
            ancora_setNo(contained);
            return ANCORA_OK;
        }
    }
    ancora_setYes(contained);
#endif
    return ANCORA_OK;
}

ancora_status ancora_interval_containsPoints(const ancora_interval *I,
                                              const ancora_mat *P,
                                              ancora_truth *contained)
/* Checks whether every column of P is contained in the interval I. P is an
 * (n x k) matrix (where n is the dimension of I) whose j-th column is the
 * j-th point; contained is true iff all k points are contained in I.
 *
 * INPUT:
 *      I               : Pointer to an initialized ancora_interval instance
 *      P               : Matrix of points, (I->dimension x k), one point per
 *                        column
 *      contained       : Output; set to true iff every column of P is
 *                        contained in I
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(dimension*k*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */
{
    // Check that I and P are well-defined
    if (I == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I is NULL; it should point to a valid ancora_interval instance.");
    }
    if (P == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer P is NULL; it should point to a valid ancora_mat instance.");
    }

    // Dimension check
    slong n;
    ANCORA_TRY(ancora_interval_dimension(I, &n));
    if (P->nrows != n) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Interval has dimension %ld, each point (column of P) has length %ld; they need to be the same.",
                      (long)n, (long)P->nrows);
    }

    slong k = P->ncols;

    ancora_truth containedIndividual;

    /* Temporary column vector holding the current point. */
    ancora_vec tmp;
    ANCORA_TRY(ancora_vec_init(&tmp, n));

    for (slong j = 0; j < k; j++) {
        /* Copy column j of P into tmp. */
        ANCORA_TRY(ancora_mat_getColumn(P, j, &tmp));

        ANCORA_TRY(ancora_interval_containsPoint(I, &tmp, &containedIndividual));

        if (!ancora_yes(containedIndividual)) {
            ancora_setNo(contained);
            ancora_vec_free(&tmp);
            return ANCORA_OK;
        }
    }

    ancora_vec_free(&tmp);

    ancora_setYes(contained);

    return ANCORA_OK;
}

ancora_status ancora_interval_batched_containsPoints(
    const ancora_interval **I_batch,
    const ancora_mat **P_batch,
    slong B,
    ancora_truth *contained)
/* Checks whether, for every b in [0, B), every column of P_batch[b] is
 * contained in I_batch[b]. contained is set to true iff ALL points in
 * ALL pairs are contained in their respective interval.
 *
 * Unlike ancora_interval_batched_randomPoints_uniform (stacking) and
 * ancora_interval_batched_supportFunction (fused reduction), pair b's
 * work here does not depend on pair b' != b at all, so - unlike those
 * two - the intervals in the batch are NOT required to share a common
 * dimension; each I_batch[b]/P_batch[b] pair may have its own dimension
 * n_b and its own point count k_b.
 *
 * This fuses everything into one triple loop (pair b, then point j, then
 * dimension i) directly against I_batch[b]->lowerBound/upperBound.repr
 * and P_batch[b]->repr, rather than calling
 * ancora_interval_containsPoints (or containsPoint) once per pair. That
 * avoids two real costs the per-pair functions pay redundantly across a
 * batch: (1) ancora_interval_containsPoint re-validates NULL/dimension on
 * EVERY point - calling it per point across the whole batch means
 * n_b*k_b such re-checks per pair, all but the first of which are
 * wasted, since a pair's dimension does not change between its own
 * points; here validation happens once per pair, not once per point; and
 * (2) ancora_interval_containsPoints allocates one temporary ancora_vec
 * per call - calling it B times means B such allocations, all avoided
 * here since points are read directly from P_batch[b]->repr with no
 * intermediate vector at all.
 *
 * Short-circuits across the ENTIRE batch, not just within one pair: the
 * moment any point in any pair is found not contained, this returns
 * immediately without checking the rest of that pair's points or any
 * later pair at all.
 *
 * INPUT:
 *      I_batch         : Array of B pointers to initialized
 *                        ancora_interval instances (dimensions may differ
 *                        across pairs)
 *      P_batch         : Array of B pointers to point matrices; P_batch[b]
 *                        is (I_batch[b]'s dimension x k_b), one point per
 *                        column, k_b may differ across pairs
 *      B               : Number of (interval, points) pairs (>= 0)
 *      contained       : Output; set to true iff every point in every
 *                        pair is contained in its interval
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(sum_b n_b*k_b*ANCORA_DEFAULT_PREC) worst case (all points
 *      contained, so nothing short-circuits) -- identical total work to B
 *      separate calls to ancora_interval_containsPoints; the savings are
 *      the eliminated per-point re-validation and per-pair temporary
 *      allocation described above, plus batch-wide short-circuiting,
 *      which can make this substantially faster than B separate calls in
 *      practice whenever an early pair/point fails containment.
 *
 * Created:       2026-09-26
 * Last modified: 2026-09-26
 * Author(s):     Adrian Kulmburg
 */
{
    if (I_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I_batch is NULL; it should point to a valid array of ancora_interval pointers.");
    }
    if (P_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer P_batch is NULL; it should point to a valid array of ancora_mat pointers.");
    }
    if (contained == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer contained is NULL; it should point to a valid ancora_truth instance.");
    }
    if (B < 0) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "B is negative (%ld); it should be nonnegative.", (long)B);
    }

    for (slong b = 0; b < B; b++) {
        if (I_batch[b] == NULL) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I_batch[%ld] is NULL; it should point to a valid ancora_interval instance.", (long)b);
        }
        if (P_batch[b] == NULL) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer P_batch[%ld] is NULL; it should point to a valid ancora_mat instance.", (long)b);
        }

        slong n;
        ANCORA_TRY(ancora_interval_dimension(I_batch[b], &n));
        if (P_batch[b]->nrows != n) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "I_batch[%ld] has dimension %ld, each point (column of P_batch[%ld]) has length %ld; they need to be the same.",
                          (long)b, (long)n, (long)b, (long)P_batch[b]->nrows);
        }

        slong k = P_batch[b]->ncols;

#if ANCORA_MODE == ANCORA_MODE_SAFE
        for (slong j = 0; j < k; j++) {
            for (slong i = 0; i < n; i++) {
                arb_ptr lo = arb_mat_entry(I_batch[b]->lowerBound.repr, i, 0);
                arb_ptr hi = arb_mat_entry(I_batch[b]->upperBound.repr, i, 0);
                arb_ptr pij = arb_mat_entry(P_batch[b]->repr, i, j);
                if (!arb_ge(pij, lo) || !arb_le(pij, hi)) {
                    ancora_setNo(contained);
                    return ANCORA_OK;
                }
            }
        }
#elif ANCORA_MODE == ANCORA_MODE_FAST
        for (slong j = 0; j < k; j++) {
            for (slong i = 0; i < n; i++) {
                double lo = I_batch[b]->lowerBound.repr[i];
                double hi = I_batch[b]->upperBound.repr[i];
                double pij = P_batch[b]->repr[i * k + j];
                if (!(lo - ANCORA_TOL <= pij && pij <= hi + ANCORA_TOL)) {
                    ancora_setNo(contained);
                    return ANCORA_OK;
                }
            }
        }
#endif
    }

    ancora_setYes(contained);
    return ANCORA_OK;
}
