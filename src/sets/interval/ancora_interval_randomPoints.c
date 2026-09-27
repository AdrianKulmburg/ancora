/*
 * ancora_interval_randomPoints.c
 *
 * Description
 * -----------
 * Random point generation for intervals.
 *
 * File Information
 * ----------------
 * Created:       2026-09-26
 * Last modified: 2026-09-26
 * Authors:       Adrian Kulmburg
 *
 * License
 * -------
 * Copyright (c) 2026 Adrian Kulmburg <adrian.kulmburg@kit.edu>
 * SPDX-License-Identifier: MIT
 */

#include "ancora/sets/interval/ancora_interval_randomPoints.h"

ancora_status ancora_interval_randomPoints_uniform(ancora_mat *P,
                                                   const ancora_interval *I,
                                                   slong N)
/* Fills P with N points drawn independently and uniformly at random from the
 * interval I. Each column of P is one point: P is an (n x N) matrix, where n
 * is the dimension of I. For each dimension i, the i-th coordinate of every
 * point is sampled uniformly from [lowerBound[i], upperBound[i]] via
 * ancora_random_uniform (one draw per coordinate per point).
 *
 * The interval must be bounded: uniform sampling is ill-defined on an
 * unbounded box, so an interval with any +-inf bound is rejected with
 * ANCORA_ERROR_NOT_IMPLEMENTED. An empty
 * interval (lowerBound[i] > upperBound[i] for some i) is rejected with
 * ANCORA_ERROR_INVALID_ARG.
 *
 * INPUT:
 *      P               : Result matrix, already initialized as (n x N), where
 *                        n is the dimension of I; column j holds the j-th
 *                        point
 *      I               : Pointer to an initialized ancora_interval instance
 *      N               : Number of points to generate (>= 0)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*N*ANCORA_DEFAULT_PREC)
 *      where n is the dimension of the interval.
 *
 * Created:       2026-09-26
 * Last modified: 2026-09-26
 * Author(s):     Adrian Kulmburg
 */
{
    // Check that P and I are well-defined
    if (P == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer P is NULL; it should point to a valid ancora_mat instance.");
    }
    if (I == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I is NULL; it should point to a valid ancora_interval instance.");
    }
    if (N < 0) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "N is negative (%ld); it should be nonnegative.", (long)N);
    }

    slong n;
    ANCORA_TRY(ancora_interval_dimension(I, &n));

    // Check that P has the right shape: (n x N), one point per column
    if (P->nrows != n || P->ncols != N) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Result should be (%ld x %ld), but P is (%ld x %ld).",
                      (long)n, (long)N, (long)P->nrows, (long)P->ncols);
    }

    /* Uniform sampling is only well-defined on a bounded box, so reject any
     * +-inf bound up front (mirroring ancora_interval_affine). */
    for (slong i = 0; i < n; i++) {
#if ANCORA_MODE == ANCORA_MODE_SAFE
        if (!arb_is_finite(arb_mat_entry(I->lowerBound.repr, i, 0)) ||
            !arb_is_finite(arb_mat_entry(I->upperBound.repr, i, 0))) {
            ANCORA_ERROR(ANCORA_ERROR_NOT_IMPLEMENTED,
                          "Unbounded interval bounds (component %ld) are not yet supported.",
                          (long)i);
        }
#elif ANCORA_MODE == ANCORA_MODE_FAST
        if (isinf(I->lowerBound.repr[i]) || isnan(I->lowerBound.repr[i]) ||
            isinf(I->upperBound.repr[i]) || isnan(I->upperBound.repr[i])) {
            ANCORA_ERROR(ANCORA_ERROR_NOT_IMPLEMENTED,
                          "Unbounded interval bounds (component %ld) are not yet supported.",
                          (long)i);
        }
#endif
    }

    /* For each point, sample each coordinate independently and uniformly from
     * its [lower, upper] box. The bounds are read as doubles (arb_get_d in
     * SAFE mode, direct in FAST mode) and fed to ancora_random_uniform, which
     * is the mode-independent uniform sampler used by the existing set
     * templates (e.g. ancora_interval_initRandom_uniform). */
#if ANCORA_MODE == ANCORA_MODE_SAFE
    double lo, hi, sample;
    for (slong j = 0; j < N; j++) {
        for (slong i = 0; i < n; i++) {
            // TODO: That's not quite ok, correct in the future
            lo = arb_get_d(arb_mat_entry(I->lowerBound.repr, i, 0));
            hi = arb_get_d(arb_mat_entry(I->upperBound.repr, i, 0));

            /* ancora_random_uniform handles lo == hi (returns lo) and rejects
             * lo > hi (an empty interval) with ANCORA_ERROR_INVALID_ARG. */
            ANCORA_TRY(ancora_random_uniform(lo, hi, &sample));

            arb_set_d(arb_mat_entry(P->repr, i, j), sample);
        }
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
    ancora_random_ensureRandSeeded();
    double lo, hi, range;
    for (slong i = 0; i < n; i++) {
        lo = I->lowerBound.repr[i];
        hi = I->upperBound.repr[i];
        if (lo > hi) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG,
                          "Invalid interval at component %ld: lower bound (%g) must be <= upper bound (%g).",
                          (long)i, lo, hi);
        }
        range = hi - lo;
        for (slong j = 0; j < N; j++) {
            /* Direct xorshiftUnit draw, not a full ancora_random_uniform
             * call: lo/hi/range are loop-invariant across j, and the
             * validation ancora_random_uniform would repeat every call
             * (NULL check, a>b check) has already been done once per row
             * above - redoing it N times per row was pure waste. */
            P->repr[i * N + j] = lo + xorshiftUnit() * range;
        }
    }
#endif


    return ANCORA_OK;
}

ancora_status ancora_interval_batched_randomPoints_uniform(
    ancora_mat **P_batch,
    const ancora_interval **I_batch,
    slong B,
    slong N)
/* Fills P_batch[0..B-1] with N points each, drawn independently and
 * uniformly at random from I_batch[0..B-1] respectively (all intervals
 * must share the same dimension n). P_batch[b] is (n x N); column j of
 * P_batch[b] is the j-th random point from I_batch[b].
 *
 * Implementation: rather than looping B separate times over
 * ancora_interval_randomPoints_uniform, the B intervals' bounds are
 * vertically stacked (via ancora_mat_vcat, since sampling is coordinate-
 * independent) into one interval of dimension n*B, sampled ONCE for N
 * points, and the resulting (n*B x N) matrix is sliced back apart by row
 * block. This does NOT reduce the total number of random draws (still
 * exactly n*B*N, the inherent cost of one draw per coordinate per point
 * per interval) - it reduces this to one validation pass and one call
 * into the core sampling loop instead of B of each, and fills one
 * contiguous block of memory rather than B smaller ones.
 *
 * INPUT:
 *      P_batch         : Array of B pointers, each already initialized
 *                        as (n x N); P_batch[b] receives interval b's
 *                        points
 *      I_batch         : Array of B pointers to initialized
 *                        ancora_interval instances, all of the same
 *                        dimension n
 *      B               : Number of intervals in the batch (>= 0)
 *      N               : Number of points to generate per interval (>= 0)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*B*N*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-26
 * Last modified: 2026-09-26
 * Author(s):     Adrian Kulmburg
 */
{
    if (P_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer P_batch is NULL; it should point to a valid array of ancora_mat pointers.");
    }
    if (I_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I_batch is NULL; it should point to a valid array of ancora_interval pointers.");
    }
    if (B < 0) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "B is negative (%ld); it should be nonnegative.", (long)B);
    }
    if (N < 0) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "N is negative (%ld); it should be nonnegative.", (long)N);
    }

    if (B == 0) {
        return ANCORA_OK; /* vacuously nothing to do */
    }

    // Validate every entry up front, and determine/check the shared dimension n
    if (I_batch[0] == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I_batch[0] is NULL; it should point to a valid ancora_interval instance.");
    }
    slong n;
    ANCORA_TRY(ancora_interval_dimension(I_batch[0], &n));

    for (slong b = 0; b < B; b++) {
        if (I_batch[b] == NULL) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I_batch[%ld] is NULL; it should point to a valid ancora_interval instance.", (long)b);
        }
        if (P_batch[b] == NULL) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer P_batch[%ld] is NULL; it should point to a valid ancora_mat instance.", (long)b);
        }
        slong nb;
        ANCORA_TRY(ancora_interval_dimension(I_batch[b], &nb));
        if (nb != n) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "I_batch[0] has dimension %ld, I_batch[%ld] has dimension %ld; every interval in the batch must share the same dimension.",
                          (long)n, (long)b, (long)nb);
        }
        if (P_batch[b]->nrows != n || P_batch[b]->ncols != N) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "P_batch[%ld] should be (%ld x %ld), but is (%ld x %ld).",
                          (long)b, (long)n, (long)N, (long)P_batch[b]->nrows, (long)P_batch[b]->ncols);
        }
    }

    ancora_vec lbStack, ubStack;
    ancora_mat Pbig;
    bool lbStackInit = false, ubStackInit = false, PbigInit = false;
    ancora_status status = ANCORA_OK;

    /* Allocate the full (n*B x 1) stacked bounds ONCE, then copy each
     * interval's bounds directly into its own slice -- genuinely O(n*B),
     * not the O(n*B^2) a repeated-vcat-growth loop would cost (see the
     * NOTE above). */
    status = ancora_vec_init(&lbStack, n * B);
    if (status != ANCORA_OK) goto cleanup;
    lbStackInit = true;
    status = ancora_vec_init(&ubStack, n * B);
    if (status != ANCORA_OK) goto cleanup;
    ubStackInit = true;

    for (slong b = 0; b < B; b++) {
#if ANCORA_MODE == ANCORA_MODE_SAFE
        for (slong i = 0; i < n; i++) {
            arb_set(arb_mat_entry(lbStack.repr, b * n + i, 0), arb_mat_entry(I_batch[b]->lowerBound.repr, i, 0));
            arb_set(arb_mat_entry(ubStack.repr, b * n + i, 0), arb_mat_entry(I_batch[b]->upperBound.repr, i, 0));
        }
#elif ANCORA_MODE == ANCORA_MODE_FAST
        memcpy(&lbStack.repr[b * n], I_batch[b]->lowerBound.repr, (size_t)n * sizeof(double));
        memcpy(&ubStack.repr[b * n], I_batch[b]->upperBound.repr, (size_t)n * sizeof(double));
#endif
    }

    /* Assemble the stacked interval directly, taking ownership of
     * lbStack/ubStack rather than deep-copying into a freshly-initialized
     * ancora_interval - freed via lbStack/ubStack below, NOT via a
     * separate ancora_interval_free call, to avoid a double free. */
    {
        ancora_interval bigI;
        bigI.lowerBound = lbStack;
        bigI.upperBound = ubStack;

        status = ancora_mat_init(&Pbig, n * B, N);
        if (status != ANCORA_OK) goto cleanup;
        PbigInit = true;

        status = ancora_interval_randomPoints_uniform(&Pbig, &bigI, N);
        if (status != ANCORA_OK) goto cleanup;
    }

    /* Slice Pbig's row-blocks back apart: rows [b*n, (b+1)*n) of Pbig are
     * exactly P_batch[b]. Row-major storage means each row is contiguous,
     * so FAST mode copies a whole row per memcpy rather than per entry. */
    for (slong b = 0; b < B; b++) {
        for (slong i = 0; i < n; i++) {
#if ANCORA_MODE == ANCORA_MODE_SAFE
            for (slong j = 0; j < N; j++) {
                arb_set(arb_mat_entry(P_batch[b]->repr, i, j),
                        arb_mat_entry(Pbig.repr, b * n + i, j));
            }
#elif ANCORA_MODE == ANCORA_MODE_FAST
            memcpy(&P_batch[b]->repr[i * N], &Pbig.repr[(b * n + i) * N], (size_t)N * sizeof(double));
#endif
        }
    }

cleanup:
    if (lbStackInit) ancora_vec_free(&lbStack);
    if (ubStackInit) ancora_vec_free(&ubStack);
    if (PbigInit) ancora_mat_free(&Pbig);

    if (status != ANCORA_OK) {
        ANCORA_ERROR(status, "ancora_interval_batched_randomPoints_uniform: stacking/sampling failed (see above).");
    }
    return ANCORA_OK;
}
