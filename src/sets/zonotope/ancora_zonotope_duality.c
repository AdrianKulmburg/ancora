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
 *      where n is the dimension of the zonotope and p its number of generators.
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */
{
    // Check that res, Z, and d are well-defined
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid output scalar.");
    }
    if (Z == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z is NULL; it should point to a valid ancora_zonotope instance.");
    }
    if (d == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer d is NULL; it should point to a valid ancora_vec instance.");
    }

    // Verify that d is a vector
    if (d->ncols != 1) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Direction d is not a vector; it should be a column vector.");
    }

    // Dimension check
    if (d->nrows != Z->c.nrows) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Zonotope Z has dimension %ld, direction d has length %ld; they need to be the same.",
                      (long)Z->c.nrows, (long)d->nrows);
    }

    slong n = Z->c.nrows;
    slong p = Z->G.ncols;

    /* h_Z(d) = d^T c + ||d^T G||_1. Transpose d (O(n)) rather than G
     * (O(n*p)), then a single (1 x n)*(n x p) product gives all p column
     * dot-products at once via ancora_mat_mul's own access pattern.
     * ancora_vec_1norm only accepts an (n x 1) column, so the (1 x p)
     * product is transposed back into a (p x 1) column afterward - this
     * transpose is O(p) (it only has p entries), not O(n*p), so it does
     * not undo the saving from avoiding a transpose of G itself. */
    ancora_mat dT;
    ANCORA_TRY(ancora_mat_init(&dT, 1, n));
    ANCORA_TRY(ancora_vec_transpose(&dT, d));

    ancora_mat w, wCol;
    ANCORA_TRY(ancora_mat_init(&w, 1, p));
    ANCORA_TRY(ancora_mat_mul(&w, &dT, &Z->G));
    ANCORA_TRY(ancora_mat_init(&wCol, p, 1));
    ANCORA_TRY(ancora_mat_transpose(&wCol, &w));

#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_t dotVal, normVal;
    arb_init(dotVal);
    arb_init(normVal);
    ANCORA_TRY(ancora_vec_dot(dotVal, d, &Z->c));
    ANCORA_TRY(ancora_vec_1norm(normVal, &wCol));
    arb_add(res, dotVal, normVal, ANCORA_DEFAULT_PREC);
    arb_clear(dotVal);
    arb_clear(normVal);
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double dotVal, normVal;
    ANCORA_TRY(ancora_vec_dot(&dotVal, d, &Z->c));
    ANCORA_TRY(ancora_vec_1norm(&normVal, &wCol));
    *res = dotVal + normVal;
#endif

    ANCORA_TRY(ancora_mat_free(&dT));
    ANCORA_TRY(ancora_mat_free(&w));
    ANCORA_TRY(ancora_mat_free(&wCol));
    return ANCORA_OK;
}

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
 *      O(n*P*ANCORA_DEFAULT_PREC) where P = sum_b p_b - identical total
 *      work to B separate ancora_zonotope_supportFunction calls (a
 *      reduction; no less work is possible). The CPU path's improvement
 *      is one validation pass and B-1 fewer dT allocations; the GPU
 *      path's improvement is one kernel launch (P-way parallel) instead
 *      of B separate ones.
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
    // No dedicated ARB GPU path exists (nor should one - see project
    // convention on FLINT/GPU). Loop, reusing dT across iterations.
    ancora_mat dT;
    ANCORA_TRY(ancora_mat_init(&dT, 1, n));

    for (slong b = 0; b < B; b++) {
        status = ancora_vec_transpose(&dT, d_batch[b]);
        if (status != ANCORA_OK) goto cleanup_safe;

        ancora_mat w, wCol;
        status = ancora_mat_init(&w, 1, p[b]);
        if (status != ANCORA_OK) goto cleanup_safe;
        status = ancora_mat_mul(&w, &dT, &Z_batch[b]->G);
        if (status != ANCORA_OK) { ancora_mat_free(&w); goto cleanup_safe; }
        status = ancora_mat_init(&wCol, p[b], 1);
        if (status != ANCORA_OK) { ancora_mat_free(&w); goto cleanup_safe; }
        status = ancora_mat_transpose(&wCol, &w);
        if (status != ANCORA_OK) { ancora_mat_free(&w); ancora_mat_free(&wCol); goto cleanup_safe; }

        arb_t dotVal, normVal;
        arb_init(dotVal);
        arb_init(normVal);
        status = ancora_vec_dot(dotVal, d_batch[b], &Z_batch[b]->c);
        if (status == ANCORA_OK) status = ancora_vec_1norm(normVal, &wCol);
        if (status == ANCORA_OK) arb_add(res + b, dotVal, normVal, ANCORA_DEFAULT_PREC);
        arb_clear(dotVal);
        arb_clear(normVal);
        ancora_mat_free(&w);
        ancora_mat_free(&wCol);
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
        // Compute the dot(d_b, c_b) term on the host first (O(n*B),
        // negligible next to the O(n*P) generator work) and use it as the
        // GPU accumulator's initial value.
        for (slong b = 0; b < B; b++) {
            double dotAcc = 0.0;
            for (slong i = 0; i < n; i++) {
                dotAcc += d_batch[b]->repr[i] * Z_batch[b]->c.repr[i];
            }
            res[b] = dotAcc;
        }

        // Pack d and G into flat buffers, plus an owner[] array mapping
        // each of the P global generator columns back to its batch index,
        // since the kernel is one-thread-per-column.
        double *d_flat = (double *)malloc((size_t)(B * n) * sizeof(double));
        double *G_flat = (double *)malloc((size_t)(n * P) * sizeof(double));
        slong *owner = (slong *)malloc((size_t)P * sizeof(slong));
        if (d_flat == NULL || G_flat == NULL || owner == NULL) {
            free(d_flat); free(G_flat); free(owner); free(p); free(offset);
            ANCORA_ERROR(ANCORA_ERROR_ALLOC, "ancora_zonotope_batched_supportFunction: failed to allocate GPU packing buffers.");
        }
        // (G_flat packing done properly below, per-row, since each pair's
        // G_b is (n x p_b) and columns must land at the right global
        // offset within each row of the (n x P) conceptual layout.)
        for (slong b = 0; b < B; b++) {
            for (slong i = 0; i < n; i++) {
                memcpy(&G_flat[i * P + offset[b]], &Z_batch[b]->G.repr[i * p[b]], (size_t)p[b] * sizeof(double));
            }
            for (slong j = 0; j < p[b]; j++) {
                owner[offset[b] + j] = b;
            }
        }

        int gpu_status = ancora_zonotope_batched_supportFunction_gpu(
            d_flat, G_flat, owner, res, n, P, B);

        free(d_flat);
        free(G_flat);
        free(owner);
        free(p);
        free(offset);

        if (gpu_status != 0) {
            ANCORA_ERROR(ANCORA_ERROR_GPU_LAUNCH, "ancora_zonotope_batched_supportFunction: GPU kernel launch failed.");
        }
        return ANCORA_OK;
    #else
        ancora_mat dT;
        ANCORA_TRY(ancora_mat_init(&dT, 1, n));

        for (slong b = 0; b < B; b++) {
            status = ancora_vec_transpose(&dT, d_batch[b]);
            if (status != ANCORA_OK) goto cleanup_fast;

            ancora_mat w, wCol;
            status = ancora_mat_init(&w, 1, p[b]);
            if (status != ANCORA_OK) goto cleanup_fast;
            status = ancora_mat_mul(&w, &dT, &Z_batch[b]->G);
            if (status != ANCORA_OK) { ancora_mat_free(&w); goto cleanup_fast; }
            status = ancora_mat_init(&wCol, p[b], 1);
            if (status != ANCORA_OK) { ancora_mat_free(&w); goto cleanup_fast; }
            status = ancora_mat_transpose(&wCol, &w);
            if (status != ANCORA_OK) { ancora_mat_free(&w); ancora_mat_free(&wCol); goto cleanup_fast; }

            double dotVal, normVal;
            status = ancora_vec_dot(&dotVal, d_batch[b], &Z_batch[b]->c);
            if (status == ANCORA_OK) status = ancora_vec_1norm(&normVal, &wCol);
            if (status == ANCORA_OK) res[b] = dotVal + normVal;
            ancora_mat_free(&w);
            ancora_mat_free(&wCol);
            if (status != ANCORA_OK) goto cleanup_fast;
        }

    cleanup_fast:
        ancora_mat_free(&dT);
        free(p);
        free(offset);
        if (status != ANCORA_OK) {
            ANCORA_ERROR(status, "ancora_zonotope_batched_supportFunction: per-pair support function computation failed (see above).");
        }
        return ANCORA_OK;
    #endif
#endif
}
