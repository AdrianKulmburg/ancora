/*
 * ancora_interval_duality.c
 *
 * Description
 * -----------
 * Duality operations for intervals.
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

#include "ancora/sets/interval/ancora_interval_duality.h"

#ifdef ANCORA_USE_GPU
#include "ancora/sets/interval/ancora_interval_duality_gpu.hip.h"
#endif

ancora_status ancora_interval_supportFunction(
#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_t res,
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double *res,
#endif
    const ancora_interval *I,
    const ancora_vec *d)
/* Computes the support function of the interval I in the direction d, i.e.,
 * h_I(d) = sup { d^T x | x in I }.
 * INPUT:
 *      res             : Output scalar (arb_t in SAFE mode, double* in FAST
 *                        mode), set to the support value
 *      I               : Pointer to an initialized ancora_interval instance
 *      d               : Direction, a column vector of length I->lowerBound.nrows
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
    // Check that res, I, and d are well-defined
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid output scalar.");
    }
    if (I == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I is NULL; it should point to a valid ancora_interval instance.");
    }
    if (d == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer d is NULL; it should point to a valid ancora_vec instance.");
    }

    // Verify that d is a vector
    bool isVector;
    ANCORA_TRY(ancora_mat_isVector(d, &isVector));
    if (!isVector) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Direction d is not a vector; it should be a column vector.");
    }

    // Dimension check
    if (d->nrows != I->lowerBound.nrows) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Interval I has dimension %ld, direction d has length %ld; they need to be the same.",
                      (long)I->lowerBound.nrows, (long)d->nrows);
    }

    slong n = I->lowerBound.nrows;

    /* h_I(d) = center^T d + ||B^T d||_1, where I = B*[-1,1]^n + center and
     * B = diag(radius). Since B is diagonal, B^T d is just the elementwise
     * product radius .* d - no matrix-vector product, and no need to
     * materialize B as an (n x n) matrix at all (unlike ancora_interval_affine,
     * where B genuinely got multiplied by a non-diagonal A). */

    ancora_vec center, radius, weighted;
    ANCORA_TRY(ancora_vec_init(&center, n));
    ANCORA_TRY(ancora_vec_init(&radius, n));
    ANCORA_TRY(ancora_vec_init(&weighted, n));

    // center = (lowerBound + upperBound) / 2
    ANCORA_TRY(ancora_vec_add(&center, &I->lowerBound, &I->upperBound));
#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_t half;
    arb_init(half);
    arb_set_d(half, 0.5);
    ANCORA_TRY(ancora_vec_scalarMul(&center, &center, half));
#elif ANCORA_MODE == ANCORA_MODE_FAST
    ANCORA_TRY(ancora_vec_scalarMul(&center, &center, 0.5));
#endif

    // radius = (upperBound - lowerBound) / 2
    ANCORA_TRY(ancora_vec_sub(&radius, &I->upperBound, &I->lowerBound));
#if ANCORA_MODE == ANCORA_MODE_SAFE
    ANCORA_TRY(ancora_vec_scalarMul(&radius, &radius, half));
    arb_clear(half);
#elif ANCORA_MODE == ANCORA_MODE_FAST
    ANCORA_TRY(ancora_vec_scalarMul(&radius, &radius, 0.5));
#endif

    // weighted_j = radius_j * d_j (elementwise)
#if ANCORA_MODE == ANCORA_MODE_SAFE
    for (slong j = 0; j < n; j++) {
        arb_mul(arb_mat_entry(weighted.repr, j, 0),
                arb_mat_entry(radius.repr, j, 0),
                arb_mat_entry(d->repr, j, 0),
                ANCORA_DEFAULT_PREC);
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
    for (slong j = 0; j < n; j++) {
        weighted.repr[j] = radius.repr[j] * d->repr[j];
    }
#endif

#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_t dotVal, normVal;
    arb_init(dotVal);
    arb_init(normVal);
    ANCORA_TRY(ancora_vec_dot(dotVal, &center, d));
    ANCORA_TRY(ancora_vec_1norm(normVal, &weighted));
    arb_add(res, dotVal, normVal, ANCORA_DEFAULT_PREC);
    arb_clear(dotVal);
    arb_clear(normVal);
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double dotVal, normVal;
    ANCORA_TRY(ancora_vec_dot(&dotVal, &center, d));
    ANCORA_TRY(ancora_vec_1norm(&normVal, &weighted));
    *res = dotVal + normVal;
#endif

    // Free all temporary variables
    ANCORA_TRY(ancora_vec_free(&center));
    ANCORA_TRY(ancora_vec_free(&radius));
    ANCORA_TRY(ancora_vec_free(&weighted));

    return ANCORA_OK;
}

ancora_status ancora_interval_batched_supportFunction(
#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_ptr res,
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double *res,
#endif
    const ancora_interval **I_batch,
    const ancora_vec **d_batch,
    slong B)
/* Computes the support function of each interval I_batch[b] in the
 * direction d_batch[b], for b = 0..B-1, i.e. res[b] = h_{I_batch[b]}(d_batch[b]).
 * All intervals must share the same dimension n; each d_batch[b] must be a
 * column vector of length n.
 *
 * See ancora_interval_supportFunction for the underlying formula
 * (h_b = center^T d + ||radius .* d||_1) and why stacking-then-slicing
 * (as in ancora_interval_batched_randomPoints_uniform) does not apply to
 * this reduction.
 *
 * In ANCORA_MODE_FAST with ANCORA_USE_GPU enabled, this dispatches to
 * ancora_interval_batched_supportFunction_gpu: one GPU thread per batch
 * element b, each looping over n internally (mirroring how
 * ancora_mat_mul_kernel's threads loop over the inner dimension). This
 * requires packing I_batch/d_batch's scattered per-interval storage into
 * flat (B*n)-length host arrays first, since GPU kernels need contiguous
 * memory - an O(n*B) packing pass that the CPU path does not need (it
 * reads each interval's own storage directly), but which does not change
 * the overall asymptotic cost.
 *
 * INPUT:
 *      res             : Output array (arb_ptr in SAFE mode, double* in
 *                        FAST mode) of length B; res[b] receives
 *                        h_{I_batch[b]}(d_batch[b])
 *      I_batch         : Array of B pointers to initialized
 *                        ancora_interval instances, all of the same
 *                        dimension n
 *      d_batch         : Array of B pointers to column vectors of length n
 *      B               : Number of (interval, direction) pairs (>= 0)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*B*ANCORA_DEFAULT_PREC) in ANCORA_MODE_SAFE or the
 *      GPU-less ANCORA_MODE_FAST path (see
 *      ancora_interval_supportFunction; a reduction, so this is the
 *      minimum possible total work). With ANCORA_USE_GPU, an additional
 *      O(n*B) host-side packing pass precedes the GPU dispatch.
 *
 * Created:       2026-09-26
 * Last modified: 2026-09-26
 * Author(s):     Adrian Kulmburg
 */
{
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid output array.");
    }
    if (I_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I_batch is NULL; it should point to a valid array of ancora_interval pointers.");
    }
    if (d_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer d_batch is NULL; it should point to a valid array of ancora_vec pointers.");
    }
    if (B < 0) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "B is negative (%ld); it should be nonnegative.", (long)B);
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
        if (d_batch[b] == NULL) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer d_batch[%ld] is NULL; it should point to a valid ancora_vec instance.", (long)b);
        }
        bool isVector;
        ANCORA_TRY(ancora_mat_isVector(d_batch[b], &isVector));
        if (!isVector) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "d_batch[%ld] is not a vector; it should be a column vector.", (long)b);
        }
        slong nb;
        ANCORA_TRY(ancora_interval_dimension(I_batch[b], &nb));
        if (nb != n) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "I_batch[0] has dimension %ld, I_batch[%ld] has dimension %ld; every interval in the batch must share the same dimension.",
                          (long)n, (long)b, (long)nb);
        }
        if (d_batch[b]->nrows != n) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "Interval I_batch[%ld] has dimension %ld, direction d_batch[%ld] has length %ld; they need to be the same.",
                          (long)b, (long)n, (long)b, (long)d_batch[b]->nrows);
        }
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    // Fused per-interval accumulation, same as ancora_interval_supportFunction
    // but written B times into a flat output array instead of an ancora_vec.
    arb_t dotAcc, normAcc, centerVal, radiusVal, term;
    arb_init(dotAcc);
    arb_init(normAcc);
    arb_init(centerVal);
    arb_init(radiusVal);
    arb_init(term);

    for (slong b = 0; b < B; b++) {
        arb_zero(dotAcc);
        arb_zero(normAcc);
        for (slong i = 0; i < n; i++) {
            arb_ptr lb = arb_mat_entry(I_batch[b]->lowerBound.repr, i, 0);
            arb_ptr ub = arb_mat_entry(I_batch[b]->upperBound.repr, i, 0);
            arb_ptr di = arb_mat_entry(d_batch[b]->repr, i, 0);

            arb_add(centerVal, lb, ub, ANCORA_DEFAULT_PREC);
            arb_mul_2exp_si(centerVal, centerVal, -1);
            arb_sub(radiusVal, ub, lb, ANCORA_DEFAULT_PREC);
            arb_mul_2exp_si(radiusVal, radiusVal, -1);

            arb_mul(term, centerVal, di, ANCORA_DEFAULT_PREC);
            arb_add(dotAcc, dotAcc, term, ANCORA_DEFAULT_PREC);

            arb_mul(term, radiusVal, di, ANCORA_DEFAULT_PREC);
            arb_abs(term, term);
            arb_add(normAcc, normAcc, term, ANCORA_DEFAULT_PREC);
        }
        arb_add(res + b, dotAcc, normAcc, ANCORA_DEFAULT_PREC);
    }

    arb_clear(dotAcc);
    arb_clear(normAcc);
    arb_clear(centerVal);
    arb_clear(radiusVal);
    arb_clear(term);
#elif ANCORA_MODE == ANCORA_MODE_FAST
    #ifdef ANCORA_USE_GPU
        // Pack the B intervals' bounds and directions into flat (B*n)-length
        // buffers, since ancora_interval_batched_supportFunction_gpu needs
        // contiguous memory to copy to the device.
        double *lb_flat = (double *)malloc((size_t)(B * n) * sizeof(double));
        double *ub_flat = (double *)malloc((size_t)(B * n) * sizeof(double));
        double *d_flat = (double *)malloc((size_t)(B * n) * sizeof(double));
        if (lb_flat == NULL || ub_flat == NULL || d_flat == NULL) {
            free(lb_flat);
            free(ub_flat);
            free(d_flat);
            ANCORA_ERROR(ANCORA_ERROR_ALLOC, "ancora_interval_batched_supportFunction: failed to allocate GPU packing buffers.");
        }
        for (slong b = 0; b < B; b++) {
            memcpy(&lb_flat[b * n], I_batch[b]->lowerBound.repr, (size_t)n * sizeof(double));
            memcpy(&ub_flat[b * n], I_batch[b]->upperBound.repr, (size_t)n * sizeof(double));
            memcpy(&d_flat[b * n], d_batch[b]->repr, (size_t)n * sizeof(double));
        }

        int gpu_status = ancora_interval_batched_supportFunction_gpu(
            lb_flat, ub_flat, d_flat, res, n, B);

        free(lb_flat);
        free(ub_flat);
        free(d_flat);

        if (gpu_status != 0) {
            ANCORA_ERROR(ANCORA_ERROR_GPU_LAUNCH, "ancora_interval_batched_supportFunction: GPU kernel launch failed.");
        }
    #else
        for (slong b = 0; b < B; b++) {
            double dotAcc = 0.0;
            double normAcc = 0.0;
            for (slong i = 0; i < n; i++) {
                double lb = I_batch[b]->lowerBound.repr[i];
                double ub = I_batch[b]->upperBound.repr[i];
                double di = d_batch[b]->repr[i];

                double centerVal = (lb + ub) * 0.5;
                double radiusVal = (ub - lb) * 0.5;

                dotAcc += centerVal * di;
                normAcc += fabs(radiusVal * di);
            }
            res[b] = dotAcc + normAcc;
        }
    #endif
#endif

    return ANCORA_OK;
}
