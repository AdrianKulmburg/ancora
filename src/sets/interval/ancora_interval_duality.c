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

#include <math.h>
#include <string.h>
#include <stdlib.h>

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
 *
 * This computes h_I(d) = center^T d + ||radius .* d||_1 directly in ONE
 * fused loop, with NO temporary ancora_vec allocations and NO delegation
 * to the generic ancora_vec_add/sub/scalarMul/dot/1norm functions (each
 * of which re-validates NULL/dimensions on every call, overhead that
 * matters disproportionately for the small O(n) vectors typical here).
 * This mirrors the fused per-pair loop already used by
 * ancora_interval_batched_supportFunction's SAFE-mode branch -- this
 * single-instance function is now exactly that loop with B=1, rather than
 * a separately (and less efficiently) implemented version of the same
 * math via 3 vector allocations and 5+ generic function calls.
 *
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
 *      where n is the dimension of the interval. No allocations.
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-27
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

    bool isVector;
    ANCORA_TRY(ancora_mat_isVector(d, &isVector));
    if (!isVector) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Direction d is not a vector; it should be a column vector.");
    }

    if (d->nrows != I->lowerBound.nrows) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Interval I has dimension %ld, direction d has length %ld; they need to be the same.",
                      (long)I->lowerBound.nrows, (long)d->nrows);
    }

    slong n = I->lowerBound.nrows;

#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_t dotAcc, normAcc, centerVal, radiusVal, term;
    arb_init(dotAcc);
    arb_init(normAcc);
    arb_init(centerVal);
    arb_init(radiusVal);
    arb_init(term);
    arb_zero(dotAcc);
    arb_zero(normAcc);

    for (slong i = 0; i < n; i++) {
        arb_ptr lb = arb_mat_entry(I->lowerBound.repr, i, 0);
        arb_ptr ub = arb_mat_entry(I->upperBound.repr, i, 0);
        arb_ptr di = arb_mat_entry(d->repr, i, 0);

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
    arb_add(res, dotAcc, normAcc, ANCORA_DEFAULT_PREC);

    arb_clear(dotAcc);
    arb_clear(normAcc);
    arb_clear(centerVal);
    arb_clear(radiusVal);
    arb_clear(term);
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double dotAcc = 0.0;
    double normAcc = 0.0;
    for (slong i = 0; i < n; i++) {
        double lb = I->lowerBound.repr[i];
        double ub = I->upperBound.repr[i];
        double di = d->repr[i];

        double centerVal = (lb + ub) * 0.5;
        double radiusVal = (ub - lb) * 0.5;

        dotAcc += centerVal * di;
        normAcc += fabs(radiusVal * di);
    }
    *res = dotAcc + normAcc;
#endif

    return ANCORA_OK;
}

/* Below this n*B product, the fused CPU loop runs even when
 * ANCORA_USE_GPU is defined: the GPU kernel is B-way parallel with each
 * thread doing an O(n) sequential reduction, so for small batches or low
 * dimension, host<->device transfer and kernel launch overhead can exceed
 * just running the reduction directly. Starting point, not a measured
 * optimum -- tune against real hardware/problem sizes if available. */
#define ANCORA_SF_GPU_MIN_ELEMENTS 100000

#if ANCORA_MODE == ANCORA_MODE_FAST
/* Shared fused-CPU implementation, used both when ANCORA_USE_GPU is not
 * defined at all, and as the fallback below ANCORA_SF_GPU_MIN_ELEMENTS
 * when it is -- avoids maintaining the same loop written out twice. */
static void ancora_interval_batched_supportFunction_cpu(
    double *res, const ancora_interval **I_batch, const ancora_vec **d_batch,
    slong n, slong B)
{
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
}
#endif

#ifdef ANCORA_USE_GPU
/* Persistent host-side packing buffers: grow-only, reused across calls,
 * never freed until process exit -- avoids a fresh malloc/free of these
 * (potentially large) buffers on every call when repeated calls share the
 * same (or smaller) n*B, exactly the pattern ancora_mat_gpu.hip.cpp's
 * device buffer cache addresses on the GPU side. NOT thread-safe (see
 * that file's equivalent note); fine for a single-threaded caller. */
static double *g_sf_lb_flat = NULL, *g_sf_ub_flat = NULL, *g_sf_d_flat = NULL;
static size_t g_sf_capacity_bytes = 0;

static int ancora_sf_buf_ensure(size_t needed_bytes)
{
    if (g_sf_capacity_bytes >= needed_bytes) {
        return 0;
    }
    free(g_sf_lb_flat);
    free(g_sf_ub_flat);
    free(g_sf_d_flat);
    g_sf_lb_flat = (double *)malloc(needed_bytes);
    g_sf_ub_flat = (double *)malloc(needed_bytes);
    g_sf_d_flat = (double *)malloc(needed_bytes);
    if (g_sf_lb_flat == NULL || g_sf_ub_flat == NULL || g_sf_d_flat == NULL) {
        free(g_sf_lb_flat); g_sf_lb_flat = NULL;
        free(g_sf_ub_flat); g_sf_ub_flat = NULL;
        free(g_sf_d_flat); g_sf_d_flat = NULL;
        g_sf_capacity_bytes = 0;
        return -1;
    }
    g_sf_capacity_bytes = needed_bytes;
    return 0;
}
#endif

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
 * See ancora_interval_supportFunction for the underlying formula and why
 * stacking-then-slicing does not apply to this reduction.
 *
 * In ANCORA_MODE_FAST with ANCORA_USE_GPU enabled AND n*B >=
 * ANCORA_SF_GPU_MIN_ELEMENTS, dispatches to
 * ancora_interval_batched_supportFunction_gpu (one GPU thread per batch
 * element b, looping over n internally). Packing buffers are persistent,
 * grow-only host allocations reused across calls (see
 * ancora_sf_buf_ensure) rather than freshly malloc'd/freed every call.
 * Below the threshold, or without ANCORA_USE_GPU, falls back to
 * ancora_interval_batched_supportFunction_cpu -- the fused per-pair loop,
 * since kernel launch + transfer overhead can exceed the reduction's own
 * cost for small batches/dimensions.
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
 *      O(n*B*ANCORA_DEFAULT_PREC) (a reduction; minimum possible total
 *      work). With ANCORA_USE_GPU and n*B >= ANCORA_SF_GPU_MIN_ELEMENTS,
 *      an O(n*B) host-side packing pass precedes the GPU dispatch (using
 *      persistent, not freshly-allocated, buffers).
 *
 * Created:       2026-09-26
 * Last modified: 2026-09-27
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
        size_t total = (size_t)(n * B);
        if (total >= ANCORA_SF_GPU_MIN_ELEMENTS) {
            size_t bytes = total * sizeof(double);
            if (ancora_sf_buf_ensure(bytes) != 0) {
                ANCORA_ERROR(ANCORA_ERROR_ALLOC, "ancora_interval_batched_supportFunction: failed to allocate GPU packing buffers.");
            }
            for (slong b = 0; b < B; b++) {
                memcpy(&g_sf_lb_flat[b * n], I_batch[b]->lowerBound.repr, (size_t)n * sizeof(double));
                memcpy(&g_sf_ub_flat[b * n], I_batch[b]->upperBound.repr, (size_t)n * sizeof(double));
                memcpy(&g_sf_d_flat[b * n], d_batch[b]->repr, (size_t)n * sizeof(double));
            }

            int gpu_status = ancora_interval_batched_supportFunction_gpu(
                g_sf_lb_flat, g_sf_ub_flat, g_sf_d_flat, res, n, B);

            if (gpu_status != 0) {
                ANCORA_ERROR(ANCORA_ERROR_GPU_LAUNCH, "ancora_interval_batched_supportFunction: GPU kernel launch failed.");
            }
        } else {
            ancora_interval_batched_supportFunction_cpu(res, I_batch, d_batch, n, B);
        }
    #else
        ancora_interval_batched_supportFunction_cpu(res, I_batch, d_batch, n, B);
    #endif
#endif

    return ANCORA_OK;
}
