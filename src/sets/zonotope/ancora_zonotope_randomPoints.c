/*
 * ancora_zonotope_randomPoints.h
 *
 * Description
 * -----------
 * Random point generation for zonotopes.
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
#include "ancora/sets/zonotope/ancora_zonotope_randomPoints.h"

#ifdef ANCORA_USE_GPU
#include "ancora/sets/zonotope/ancora_zonotope_randomPoints_gpu.hip.h"
#endif

ancora_status ancora_zonotope_randomPoints_standard(ancora_mat *P,
                                                    const ancora_zonotope *Z,
                                                    slong N)
/* Fills P with N points on the zonotope Z. Each column of P is one point: P
 * is an (n x N) matrix, where n is the dimension of Z. For each point, a
 * vector x is drawn uniformly at random from the unit cube [-1,1]^m (m = the
 * number of generators of Z, one draw per entry via ancora_random_uniform /
 * ancora_random_uniform_arb), and the point G*x + c is stored. This yields
 * points on the zonotope, though they are not uniformly distributed on it
 * (the map from the cube to the zonotope is not measure-preserving in
 * general).
 *
 * INPUT:
 *      P               : Result matrix, already initialized as (n x N), where
 *                        n is the dimension of Z; column j holds the j-th
 *                        point
 *      Z               : Pointer to an initialized ancora_zonotope instance
 *      N               : Number of points to generate (>= 0)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*m*N*ANCORA_DEFAULT_PREC)
 *      where n is the dimension of the zonotope and m its number of
 *      generators.
 *
 * Created:       2026-09-26
 * Last modified: 2026-09-26
 * Author(s):     Adrian Kulmburg
 */
{
    // Check that P and Z are well-defined
    if (P == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer P is NULL; it should point to a valid ancora_mat instance.");
    }
    if (Z == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z is NULL; it should point to a valid ancora_zonotope instance.");
    }
    if (N < 0) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "N is negative (%ld); it should be nonnegative.", (long)N);
    }

    slong n = Z->G.nrows;
    slong m = Z->G.ncols;

    // Check that P has the right shape: (n x N), one point per column
    if (P->nrows != n || P->ncols != N) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Result should be (%ld x %ld), but P is (%ld x %ld).",
                      (long)n, (long)N, (long)P->nrows, (long)P->ncols);
    }

    /* Temporaries: the [-1,1]^m cube interval I (with its bound vectors), and
     * the (m x N) matrix X of cube samples. All are freed on every path via
     * the cleanup label below, so a mid-function failure does not leak them. */
    // TODO: Implement that cleanup also in other functions
    ancora_status status = ANCORA_OK;
    ancora_vec lower, upper;
    ancora_interval I;
    ancora_mat X;
    bool lower_init = false, upper_init = false, I_init = false, X_init = false;

    status = ancora_vec_init(&lower, m);
    if (status != ANCORA_OK) {
        goto cleanup;
    }
    lower_init = true;

    status = ancora_vec_init(&upper, m);
    if (status != ANCORA_OK) {
        goto cleanup;
    }
    upper_init = true;

    /* lower = -1, upper = +1 in every component. */
    status = ancora_vec_ones(&upper);
    if (status != ANCORA_OK) {
        goto cleanup;
    }
    status = ancora_vec_neg(&lower, &upper);
    if (status != ANCORA_OK) {
        goto cleanup;
    }

    status = ancora_interval_init(&I, &lower, &upper);
    if (status != ANCORA_OK) {
        goto cleanup;
    }
    I_init = true;

    status = ancora_mat_init(&X, m, N);
    if (status != ANCORA_OK) {
        goto cleanup;
    }
    X_init = true;

    /* Draw N points uniformly from the cube [-1,1]^m. */
    status = ancora_interval_randomPoints_uniform(&X, &I, N);
    if (status != ANCORA_OK) {
        goto cleanup;
    }

    /* P = G*X (an (n x N) matrix; G is (n x m), X is (m x N)). */
    status = ancora_mat_mul(P, &Z->G, &X);
    if (status != ANCORA_OK) {
        goto cleanup;
    }

    /* P += c, broadcast across every column. */
    // TODO: Implement a dedicated function that does this
    for (slong j = 0; j < N; j++) {
        for (slong i = 0; i < n; i++) {
#if ANCORA_MODE == ANCORA_MODE_SAFE
            arb_add(arb_mat_entry(P->repr, i, j),
                    arb_mat_entry(P->repr, i, j),
                    arb_mat_entry(Z->c.repr, i, 0), ANCORA_DEFAULT_PREC);
#elif ANCORA_MODE == ANCORA_MODE_FAST
            P->repr[i * N + j] += Z->c.repr[i];
#endif
        }
    }

cleanup:
    if (X_init) {
        ancora_status free_status = ancora_mat_free(&X);
        if (status == ANCORA_OK) {
            status = free_status;
        }
    }
    if (I_init) {
        ancora_status free_status = ancora_interval_free(&I);
        if (status == ANCORA_OK) {
            status = free_status;
        }
    }
    if (upper_init) {
        ancora_status free_status = ancora_vec_free(&upper);
        if (status == ANCORA_OK) {
            status = free_status;
        }
    }
    if (lower_init) {
        ancora_status free_status = ancora_vec_free(&lower);
        if (status == ANCORA_OK) {
            status = free_status;
        }
    }
    return status;
}

ancora_status ancora_zonotope_batched_randomPoints_standard(
    ancora_mat **P_batch,
    const ancora_zonotope **Z_batch,
    slong B,
    slong N)
/* Fills P_batch[b] with N points on Z_batch[b], for every b in [0, B), the
 * same way as ancora_zonotope_randomPoints_standard: a vector x_b is drawn
 * uniformly from [-1,1]^{m_b} (m_b = Z_batch[b]'s generator count) and
 * G_b*x_b + c_b is stored, for each of the N points. All zonotopes share
 * dimension n and the point count N; generator counts m_b may differ.
 *
 * INPUT:
 *      P_batch         : Array of B pointers, each already initialized as
 *                        (n x N); P_batch[b] receives N points on
 *                        Z_batch[b]
 *      Z_batch         : Array of B pointers to initialized
 *                        ancora_zonotope instances, all of dimension n
 *                        (generator counts m_b may differ)
 *      B               : Number of zonotopes in the batch (>= 0)
 *      N               : Number of points to generate per zonotope (>= 0)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(N*sum_b(n*m_b)*ANCORA_DEFAULT_PREC) in ANCORA_MODE_SAFE or the
 *      GPU-less ANCORA_MODE_FAST path - identical total work to B
 *      separate ancora_zonotope_randomPoints_standard calls (matmul, no
 *      possible reduction). The improvements are: one cube-sampling call
 *      instead of B, and, with ANCORA_USE_GPU, one fused kernel launch
 *      (doing the SAME total O(N*sum_b(n*m_b)) work, no zero-padding
 *      waste) instead of B separate matmul kernel launches.
 *
 * Created:       2026-09-26
 * Last modified: 2026-09-26
 * Author(s):     Adrian Kulmburg
 */
{
    if (P_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer P_batch is NULL; it should point to a valid array of ancora_mat pointers.");
    }
    if (Z_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z_batch is NULL; it should point to a valid array of ancora_zonotope pointers.");
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

    if (Z_batch[0] == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z_batch[0] is NULL; it should point to a valid ancora_zonotope instance.");
    }
    slong n = Z_batch[0]->G.nrows;

    slong *m = (slong *)malloc((size_t)B * sizeof(slong));
    slong *offset = (slong *)malloc((size_t)B * sizeof(slong));
    if (m == NULL || offset == NULL) {
        free(m);
        free(offset);
        ANCORA_ERROR(ANCORA_ERROR_ALLOC, "ancora_zonotope_batched_randomPoints_standard: failed to allocate per-pair bookkeeping arrays.");
    }

    ancora_status status = ANCORA_OK;
    slong M = 0;

    for (slong b = 0; b < B; b++) {
        if (Z_batch[b] == NULL) {
            status = ANCORA_ERROR_INVALID_ARG;
            ANCORA_ERROR(status, "Pointer Z_batch[%ld] is NULL; it should point to a valid ancora_zonotope instance.", (long)b);
        }
        if (P_batch[b] == NULL) {
            status = ANCORA_ERROR_INVALID_ARG;
            ANCORA_ERROR(status, "Pointer P_batch[%ld] is NULL; it should point to a valid ancora_mat instance.", (long)b);
        }
        if (Z_batch[b]->G.nrows != n) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "Z_batch[0] has dimension %ld, Z_batch[%ld] has dimension %ld; every zonotope in the batch must share the same dimension.",
                          (long)n, (long)b, (long)Z_batch[b]->G.nrows);
        }
        if (P_batch[b]->nrows != n || P_batch[b]->ncols != N) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "P_batch[%ld] should be (%ld x %ld), but is (%ld x %ld).",
                          (long)b, (long)n, (long)N, (long)P_batch[b]->nrows, (long)P_batch[b]->ncols);
        }
        m[b] = Z_batch[b]->G.ncols;
        offset[b] = M;
        M += m[b];
    }

    // Stage 1: cube sampling, ONE call for the whole batch
    ancora_vec lower, upper;
    ancora_interval I;
    ancora_mat Xbig;
    bool lower_init = false, upper_init = false, I_init = false, Xbig_init = false;

    status = ancora_vec_init(&lower, M);
    if (status != ANCORA_OK) goto cleanup;
    lower_init = true;
    status = ancora_vec_init(&upper, M);
    if (status != ANCORA_OK) goto cleanup;
    upper_init = true;
    status = ancora_vec_ones(&upper);
    if (status != ANCORA_OK) goto cleanup;
    status = ancora_vec_neg(&lower, &upper);
    if (status != ANCORA_OK) goto cleanup;
    status = ancora_interval_init(&I, &lower, &upper);
    if (status != ANCORA_OK) goto cleanup;
    I_init = true;
    status = ancora_mat_init(&Xbig, M, N);
    if (status != ANCORA_OK) goto cleanup;
    Xbig_init = true;

    status = ancora_interval_randomPoints_uniform(&Xbig, &I, N);
    if (status != ANCORA_OK) goto cleanup;

    // Stage 2: G_b*X_b + c_b
#if ANCORA_MODE == ANCORA_MODE_SAFE
    // No dedicated ARB GPU path (see project convention: no GPU-accelerated
    // FLINT path). Per-pair matmul, reusing ancora_mat_mul directly against
    // Xbig's row-blocks; each pair's slice is read straight out of Xbig
    // without any copy, since ancora_mat_mul only needs a view - but
    // ancora_mat is not a view type in this project, so a small per-pair
    // (m_b x N) temporary is still needed to hold the extracted rows.
    for (slong b = 0; b < B && status == ANCORA_OK; b++) {
        ancora_mat Xb;
        status = ancora_mat_init(&Xb, m[b], N);
        if (status != ANCORA_OK) break;
        for (slong k = 0; k < m[b]; k++) {
            for (slong j = 0; j < N; j++) {
                arb_set(arb_mat_entry(Xb.repr, k, j), arb_mat_entry(Xbig.repr, offset[b] + k, j));
            }
        }
        status = ancora_mat_mul(P_batch[b], &Z_batch[b]->G, &Xb);
        ancora_mat_free(&Xb);
        if (status != ANCORA_OK) break;

        for (slong j = 0; j < N; j++) {
            for (slong i = 0; i < n; i++) {
                arb_add(arb_mat_entry(P_batch[b]->repr, i, j),
                        arb_mat_entry(P_batch[b]->repr, i, j),
                        arb_mat_entry(Z_batch[b]->c.repr, i, 0), ANCORA_DEFAULT_PREC);
            }
        }
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
    #ifdef ANCORA_USE_GPU
        // Pack G (n x M) and c (B x n) into flat buffers; Xbig is already
        // exactly the (M x N) shape the kernel needs, no repacking.
        double *G_flat = (double *)malloc((size_t)(n * M) * sizeof(double));
        double *c_flat = (double *)malloc((size_t)(B * n) * sizeof(double));
        double *res_flat = (double *)malloc((size_t)(B * n * N) * sizeof(double));
        if (G_flat == NULL || c_flat == NULL || res_flat == NULL) {
            free(G_flat); free(c_flat); free(res_flat);
            status = ANCORA_ERROR_ALLOC;
            goto cleanup;
        }
        for (slong b = 0; b < B; b++) {
            for (slong i = 0; i < n; i++) {
                memcpy(&G_flat[i * M + offset[b]], &Z_batch[b]->G.repr[i * m[b]], (size_t)m[b] * sizeof(double));
            }
            memcpy(&c_flat[b * n], Z_batch[b]->c.repr, (size_t)n * sizeof(double));
        }

        int gpu_status = ancora_zonotope_batched_randomPoints_standard_gpu(
            G_flat, Xbig.repr, c_flat, offset, m, res_flat, n, M, N, B);

        if (gpu_status == 0) {
            for (slong b = 0; b < B; b++) {
                memcpy(P_batch[b]->repr, &res_flat[b * n * N], (size_t)(n * N) * sizeof(double));
            }
        }

        free(G_flat); free(c_flat); free(res_flat);

        if (gpu_status != 0) {
            status = ANCORA_ERROR_GPU_LAUNCH;
        }
    #else
        for (slong b = 0; b < B && status == ANCORA_OK; b++) {
            ancora_mat Xb;
            status = ancora_mat_init(&Xb, m[b], N);
            if (status != ANCORA_OK) break;
            memcpy(Xb.repr, &Xbig.repr[offset[b] * N], (size_t)(m[b] * N) * sizeof(double));

            status = ancora_mat_mul(P_batch[b], &Z_batch[b]->G, &Xb);
            ancora_mat_free(&Xb);
            if (status != ANCORA_OK) break;

            for (slong j = 0; j < N; j++) {
                for (slong i = 0; i < n; i++) {
                    P_batch[b]->repr[i * N + j] += Z_batch[b]->c.repr[i];
                }
            }
        }
    #endif
#endif

cleanup:
    if (Xbig_init) ancora_mat_free(&Xbig);
    if (I_init) ancora_interval_free(&I);
    if (upper_init) ancora_vec_free(&upper);
    if (lower_init) ancora_vec_free(&lower);
    free(m);
    free(offset);

    if (status != ANCORA_OK) {
        ANCORA_ERROR(status, "ancora_zonotope_batched_randomPoints_standard: cube sampling / mapping failed (see above).");
    }
    return ANCORA_OK;
}
