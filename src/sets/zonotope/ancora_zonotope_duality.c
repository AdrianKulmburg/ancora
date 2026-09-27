/*
 * ancora_zonotope_duality.c
 *
 * Description
 * ------
 * Duality operations for ancora_zonotope, in particular the support function.
 *
 * File Information
 * --------
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Authors:       Adrian Kulmburg
 *
 * License
 * ----
 * Copyright (c) 2026 Adrian Kulmburg <adrian.kulmburg@kit.edu>
 * SPDX-License-Identifier: MIT
 */

#include <math.h>

#include "ancora/sets/zonotope/ancora_zonotope_duality.h"

#ifdef ANCORA_USE_GPU
#include "ancora/sets/zonotope/ancora_zonotope_duality_gpu.hip.h"
#endif

ancora_status ancora_zonotope_supportFunction(
#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_t res,
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double *res,
#endif
    const ancora_zonotope *Z,
    const ancora_vec *d)
/* Computes the support function of the zonotope Z in the direction d, i.e.
 * h_Z(d) = sup { d^T x | x in Z }.
 *
 * This no longer transposes w (the 1 x p row d^T*G) back into a p x 1
 * column before computing its 1-norm: ancora_vec_1norm requires a column,
 * but a 1-norm is just the sum of absolute values, which does not care
 * about row-vs-column layout. Summing |w[0][j]| directly removes an
 * entire allocation (the old wCol) and an entire O(p) transpose call that
 * existed solely to satisfy ancora_vec_1norm's column-only interface.
 *
 * INPUT:
 *      res             : Output scalar (arb_t in SAFE mode, double* in FAST
 *                        mode), set to the support value
 *      Z               : Pointer to an initialized ancora_zonotope instance
 *      d               : Direction, a column vector of length Z->c.nrows
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*p*ANCORA_DEFAULT_PREC)
 *      where n is the dimension of the zonotope and p its number of
 *      generators.
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-27
 * Author(s):     Adrian Kulmburg
 */
{
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid output scalar.");
    }
    if (Z == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z is NULL; it should point to a valid ancora_zonotope instance.");
    }
    if (d == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer d is NULL; it should point to a valid ancora_vec instance.");
    }

    if (d->ncols != 1) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Direction d is not a vector; it should be a column vector.");
    }

    if (d->nrows != Z->c.nrows) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Zonotope Z has dimension %ld, direction d has length %ld; they need to be the same.",
                      (long)Z->c.nrows, (long)d->nrows);
    }

    slong n = Z->c.nrows;
    slong p = Z->G.ncols;

    /* h_Z(d) = d^T c + ||d^T G||_1. Transpose d (O(n)) rather than G
     * (O(n*p)), then a single (1 x n)*(n x p) product gives all p column
     * dot-products at once. */
    ancora_mat dT, w;
    ANCORA_TRY(ancora_mat_init(&dT, 1, n));
    ANCORA_TRY(ancora_vec_transpose(&dT, d));
    ANCORA_TRY(ancora_mat_init(&w, 1, p));
    ANCORA_TRY(ancora_mat_mul(&w, &dT, &Z->G));

#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_t dotVal, normVal, term;
    arb_init(dotVal);
    arb_init(normVal);
    arb_init(term);
    ANCORA_TRY(ancora_vec_dot(dotVal, d, &Z->c));
    arb_zero(normVal);
    for (slong j = 0; j < p; j++) {
        arb_abs(term, arb_mat_entry(w.repr, 0, j));
        arb_add(normVal, normVal, term, ANCORA_DEFAULT_PREC);
    }
    arb_add(res, dotVal, normVal, ANCORA_DEFAULT_PREC);
    arb_clear(dotVal);
    arb_clear(normVal);
    arb_clear(term);
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double dotVal, normVal = 0.0;
    ANCORA_TRY(ancora_vec_dot(&dotVal, d, &Z->c));
    for (slong j = 0; j < p; j++) {
        normVal += fabs(w.repr[j]);
    }
    *res = dotVal + normVal;
#endif

    ANCORA_TRY(ancora_mat_free(&dT));
    ANCORA_TRY(ancora_mat_free(&w));
    return ANCORA_OK;
}

/* Below this n*P product, the CPU loop runs even when ANCORA_USE_GPU is
 * defined -- placeholder, tune against real hardware. */
#define ANCORA_SF_GPU_MIN_ELEMENTS 100000

#ifdef ANCORA_USE_GPU
/* Persistent host-side packing buffers for the GPU path: grow-only,
 * reused across calls, never freed until process exit -- avoids a fresh
 * malloc/free of d_flat/G_flat/owner on every call. NOT thread-safe
 * (module-level static state); fine for a single-threaded caller. */
static double *g_zsf_d_flat = NULL, *g_zsf_G_flat = NULL;
static slong *g_zsf_owner = NULL;
static size_t g_zsf_d_capacity = 0, g_zsf_G_capacity = 0, g_zsf_owner_capacity = 0;

static int ancora_zsf_buf_ensure(size_t d_bytes, size_t G_bytes, size_t owner_bytes)
{
    if (g_zsf_d_capacity < d_bytes) {
        free(g_zsf_d_flat);
        g_zsf_d_flat = (double *)malloc(d_bytes);
        if (g_zsf_d_flat == NULL) { g_zsf_d_capacity = 0; return -1; }
        g_zsf_d_capacity = d_bytes;
    }
    if (g_zsf_G_capacity < G_bytes) {
        free(g_zsf_G_flat);
        g_zsf_G_flat = (double *)malloc(G_bytes);
        if (g_zsf_G_flat == NULL) { g_zsf_G_capacity = 0; return -1; }
        g_zsf_G_capacity = G_bytes;
    }
    if (g_zsf_owner_capacity < owner_bytes) {
        free(g_zsf_owner);
        g_zsf_owner = (slong *)malloc(owner_bytes);
        if (g_zsf_owner == NULL) { g_zsf_owner_capacity = 0; return -1; }
        g_zsf_owner_capacity = owner_bytes;
    }
    return 0;
}
#endif

#if ANCORA_MODE == ANCORA_MODE_FAST
/* Shared fused-CPU implementation (no wCol/transpose, direct row-norm
 * summation), used both when ANCORA_USE_GPU is not defined, and as the
 * fallback below ANCORA_SF_GPU_MIN_ELEMENTS when it is. */
static ancora_status ancora_zonotope_batched_supportFunction_cpu(
    double *res, const ancora_zonotope **Z_batch, const ancora_vec **d_batch,
    const slong *p, slong n, slong B)
{
    ancora_status status = ANCORA_OK;
    ancora_mat dT;
    ANCORA_TRY(ancora_mat_init(&dT, 1, n));

    for (slong b = 0; b < B; b++) {
        status = ancora_vec_transpose(&dT, d_batch[b]);
        if (status != ANCORA_OK) goto cleanup;

        ancora_mat w;
        status = ancora_mat_init(&w, 1, p[b]);
        if (status != ANCORA_OK) goto cleanup;
        status = ancora_mat_mul(&w, &dT, &Z_batch[b]->G);
        if (status != ANCORA_OK) { ancora_mat_free(&w); goto cleanup; }

        double dotVal, normVal = 0.0;
        status = ancora_vec_dot(&dotVal, d_batch[b], &Z_batch[b]->c);
        for (slong j = 0; status == ANCORA_OK && j < p[b]; j++) {
            normVal += fabs(w.repr[j]);
        }
        if (status == ANCORA_OK) res[b] = dotVal + normVal;
        ancora_mat_free(&w);
        if (status != ANCORA_OK) goto cleanup;
    }

cleanup:
    ancora_mat_free(&dT);
    return status;
}
#endif

ancora_status ancora_zonotope_batched_supportFunction(
#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_ptr res,
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double *res,
#endif
    const ancora_zonotope **Z_batch,
    const ancora_vec **d_batch,
    slong B)
/* Computes res[b] = h_{Z_batch[b]}(d_batch[b]) for every b in [0, B). All
 * zonotopes and directions share dimension n, but each Z_batch[b] may have
 * its OWN generator count p_b.
 *
 * Like ancora_zonotope_supportFunction, this no longer transposes each
 * pair's w (1 x p_b row) back into a p_b x 1 column before summing its
 * 1-norm -- the row is summed directly, removing an allocation and an
 * O(p_b) transpose call per pair. The FAST-mode GPU path also uses
 * persistent, grow-only host packing buffers (see ancora_zsf_buf_ensure)
 * instead of a fresh malloc/free every call, and only dispatches to the
 * GPU when n*P >= ANCORA_SF_GPU_MIN_ELEMENTS (the kernel is P-way
 * parallel with O(n) work per thread; below that, transfer/launch
 * overhead can exceed the reduction's own cost).
 *
 * INPUT:
 *      res             : Output array (arb_ptr in SAFE mode, double* in
 *                        FAST mode) of length B; res[b] receives
 *                        h_{Z_batch[b]}(d_batch[b])
 *      Z_batch         : Array of B pointers to initialized
 *                        ancora_zonotope instances, all of dimension n
 *                        (generator counts p_b may differ)
 *      d_batch         : Array of B pointers to column vectors of length n
 *      B               : Number of (zonotope, direction) pairs (>= 0)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*P*ANCORA_DEFAULT_PREC) where P = sum_b p_b (a reduction; no
 *      less work is possible). With ANCORA_USE_GPU and n*P >=
 *      ANCORA_SF_GPU_MIN_ELEMENTS, an O(n*P) host-side packing pass
 *      precedes the GPU dispatch, using persistent buffers.
 *
 * Created:       2026-09-27
 * Last modified: 2026-09-27
 * Author(s):     Adrian Kulmburg
 */
{
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid output array.");
    }
    if (Z_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z_batch is NULL; it should point to a valid array of ancora_zonotope pointers.");
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

    if (Z_batch[0] == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z_batch[0] is NULL; it should point to a valid ancora_zonotope instance.");
    }
    slong n = Z_batch[0]->c.nrows;

    slong *p = (slong *)malloc((size_t)B * sizeof(slong));
    slong *offset = (slong *)malloc((size_t)B * sizeof(slong));
    if (p == NULL || offset == NULL) {
        free(p);
        free(offset);
        ANCORA_ERROR(ANCORA_ERROR_ALLOC, "ancora_zonotope_batched_supportFunction: failed to allocate per-pair bookkeeping arrays.");
    }

    ancora_status status = ANCORA_OK;
    slong P = 0;

    for (slong b = 0; b < B; b++) {
        if (Z_batch[b] == NULL) {
            status = ANCORA_ERROR_INVALID_ARG;
            ANCORA_ERROR(status, "Pointer Z_batch[%ld] is NULL; it should point to a valid ancora_zonotope instance.", (long)b);
        }
        if (d_batch[b] == NULL) {
            status = ANCORA_ERROR_INVALID_ARG;
            ANCORA_ERROR(status, "Pointer d_batch[%ld] is NULL; it should point to a valid ancora_vec instance.", (long)b);
        }
        if (d_batch[b]->ncols != 1) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "d_batch[%ld] is not a vector; it should be a column vector.", (long)b);
        }
        if (Z_batch[b]->c.nrows != n) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "Z_batch[0] has dimension %ld, Z_batch[%ld] has dimension %ld; every zonotope in the batch must share the same dimension.",
                          (long)n, (long)b, (long)Z_batch[b]->c.nrows);
        }
        if (d_batch[b]->nrows != n) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "Zonotope Z_batch[%ld] has dimension %ld, direction d_batch[%ld] has length %ld; they need to be the same.",
                          (long)b, (long)n, (long)b, (long)d_batch[b]->nrows);
        }
        p[b] = Z_batch[b]->G.ncols;
        offset[b] = P;
        P += p[b];
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    ancora_mat dT;
    ANCORA_TRY(ancora_mat_init(&dT, 1, n));

    for (slong b = 0; b < B; b++) {
        status = ancora_vec_transpose(&dT, d_batch[b]);
        if (status != ANCORA_OK) goto cleanup_safe;

        ancora_mat w;
        status = ancora_mat_init(&w, 1, p[b]);
        if (status != ANCORA_OK) goto cleanup_safe;
        status = ancora_mat_mul(&w, &dT, &Z_batch[b]->G);
        if (status != ANCORA_OK) { ancora_mat_free(&w); goto cleanup_safe; }

        arb_t dotVal, normVal, term;
        arb_init(dotVal);
        arb_init(normVal);
        arb_init(term);
        status = ancora_vec_dot(dotVal, d_batch[b], &Z_batch[b]->c);
        if (status == ANCORA_OK) {
            arb_zero(normVal);
            for (slong j = 0; j < p[b]; j++) {
                arb_abs(term, arb_mat_entry(w.repr, 0, j));
                arb_add(normVal, normVal, term, ANCORA_DEFAULT_PREC);
            }
            arb_add(res + b, dotVal, normVal, ANCORA_DEFAULT_PREC);
        }
        arb_clear(dotVal);
        arb_clear(normVal);
        arb_clear(term);
        ancora_mat_free(&w);
        if (status != ANCORA_OK) goto cleanup_safe;
    }

cleanup_safe:
    ancora_mat_free(&dT);
    free(p);
    free(offset);
    if (status != ANCORA_OK) {
        ANCORA_ERROR(status, "ancora_zonotope_batched_supportFunction: per-pair support function computation failed (see above).");
    }
    return ANCORA_OK;
#elif ANCORA_MODE == ANCORA_MODE_FAST
    #ifdef ANCORA_USE_GPU
        size_t total = (size_t)(n * P);
        if (total >= ANCORA_SF_GPU_MIN_ELEMENTS) {
            for (slong b = 0; b < B; b++) {
                double dotAcc = 0.0;
                for (slong i = 0; i < n; i++) {
                    dotAcc += d_batch[b]->repr[i] * Z_batch[b]->c.repr[i];
                }
                res[b] = dotAcc;
            }

            size_t d_bytes = (size_t)(B * n) * sizeof(double);
            size_t G_bytes = (size_t)(n * P) * sizeof(double);
            size_t owner_bytes = (size_t)P * sizeof(slong);
            if (ancora_zsf_buf_ensure(d_bytes, G_bytes, owner_bytes) != 0) {
                free(p); free(offset);
                ANCORA_ERROR(ANCORA_ERROR_ALLOC, "ancora_zonotope_batched_supportFunction: failed to allocate GPU packing buffers.");
            }

            for (slong b = 0; b < B; b++) {
                memcpy(&g_zsf_d_flat[b * n], d_batch[b]->repr, (size_t)n * sizeof(double));
                for (slong i = 0; i < n; i++) {
                    memcpy(&g_zsf_G_flat[i * P + offset[b]], &Z_batch[b]->G.repr[i * p[b]], (size_t)p[b] * sizeof(double));
                }
                for (slong j = 0; j < p[b]; j++) {
                    g_zsf_owner[offset[b] + j] = b;
                }
            }

            int gpu_status = ancora_zonotope_batched_supportFunction_gpu(
                g_zsf_d_flat, g_zsf_G_flat, g_zsf_owner, res, n, P, B);

            free(p);
            free(offset);

            if (gpu_status != 0) {
                ANCORA_ERROR(ANCORA_ERROR_GPU_LAUNCH, "ancora_zonotope_batched_supportFunction: GPU kernel launch failed.");
            }
            return ANCORA_OK;
        } else {
            status = ancora_zonotope_batched_supportFunction_cpu(res, Z_batch, d_batch, p, n, B);
            free(p);
            free(offset);
            if (status != ANCORA_OK) {
                ANCORA_ERROR(status, "ancora_zonotope_batched_supportFunction: per-pair support function computation failed (see above).");
            }
            return ANCORA_OK;
        }
    #else
        status = ancora_zonotope_batched_supportFunction_cpu(res, Z_batch, d_batch, p, n, B);
        free(p);
        free(offset);
        if (status != ANCORA_OK) {
            ANCORA_ERROR(status, "ancora_zonotope_batched_supportFunction: per-pair support function computation failed (see above).");
        }
        return ANCORA_OK;
    #endif
#endif
}
