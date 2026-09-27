/*
 * ancora_zonotope_arithmetic.c
 *
 * Description
 * -----------
 * Arithmetic operations for zonotopes.
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

#include "ancora/sets/zonotope/ancora_zonotope_arithmetic.h"

ancora_status ancora_zonotope_minkowskiSum(ancora_zonotope *res,
                                           const ancora_zonotope *Z1,
                                           const ancora_zonotope *Z2)
/* Minkowski sum of two zonotopes: res = Z1 + Z2.
 *
 * INPUT:
 *      res             : Result zonotope, already initialized as the same
 *                        dimension as Z1 and Z2, and with m generators, where
 *                        m = m1 + m2, with mi the number of generators of Zi
 *      Z1              : First summand
 *      Z2              : Second summand
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*(m1+m2)*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */
{
    // Check that res, Z1, and Z2 are well-defined
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_zonotope instance.");
    }
    if (Z1 == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z1 is NULL; it should point to a valid ancora_zonotope instance.");
    }
    if (Z2 == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z2 is NULL; it should point to a valid ancora_zonotope instance.");
    }

    // Check dimensions
    slong n, n1, n2;
    ANCORA_TRY(ancora_zonotope_dimension(res, &n));
    ANCORA_TRY(ancora_zonotope_dimension(Z1, &n1));
    ANCORA_TRY(ancora_zonotope_dimension(Z2, &n2));

    if (n1 != n2) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Zonotope Z1 has dimension %ld, zonotope Z2 has dimension %ld; they need to be the same.",
                      (long)n1, (long)n2);
    }
    if (n != n1) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Zonotopes Z1 and Z2 have dimension %ld, zonotope res has dimension %ld; they need to be the same.",
                      (long)n1, (long)n);
    }

    // TODO: Perhaps create a dedicated function for this someday?
    slong m, m1, m2;
    m = res->G.ncols;
    m1 = Z1->G.ncols;
    m2 = Z2->G.ncols;

    if (m != m1 + m2) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Result should have %ld generators, but res has %ld.",
                      (long)(m1 + m2), (long)m);
    }

    /* res.c = Z1.c + Z2.c */
    ANCORA_TRY(ancora_vec_add(&res->c, &Z1->c, &Z2->c));
    /* res.G = [Z1.G Z2.G] */
    ANCORA_TRY(ancora_mat_hcat(&res->G, &Z1->G, &Z2->G));

    return ANCORA_OK;
}

ancora_status ancora_zonotope_batched_minkowskiSum(
    ancora_zonotope **res_batch,
    const ancora_zonotope **Z1_batch,
    const ancora_zonotope **Z2_batch,
    slong B)
/* Computes res_batch[b] = Z1_batch[b] + Z2_batch[b] (Minkowski sum) for
 * every b in [0, B). All zonotopes across the batch share dimension n;
 * each pair may have its own generator counts m1_b, m2_b, and
 * res_batch[b] must have m1_b + m2_b generators (same requirement as the
 * single-instance ancora_zonotope_minkowskiSum).
 *
 * INPUT:
 *      res_batch       : Array of B pointers, each already initialized
 *                        with dimension n and (m1_b + m2_b) generators;
 *                        res_batch[b] receives Z1_batch[b] + Z2_batch[b]
 *      Z1_batch        : Array of B pointers to initialized
 *                        ancora_zonotope instances, all of dimension n
 *      Z2_batch        : Array of B pointers to initialized
 *                        ancora_zonotope instances, all of dimension n
 *      B               : Number of (Z1, Z2) pairs (>= 0)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*B) for the batched center addition (identical total work to B
 *      separate additions; the win is one call/kernel launch instead of
 *      B), plus O(n*sum_b(m1_b+m2_b)) for the generator concatenation
 *      (identical total work to B separate ancora_mat_hcat calls; no
 *      possible reduction, the win is one validation pass instead of B
 *      and no per-pair temporary allocation).
 *
 * Created:       2026-09-27
 * Last modified: 2026-09-27
 * Author(s):     Adrian Kulmburg
 */
{
    if (res_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res_batch is NULL; it should point to a valid array of ancora_zonotope pointers.");
    }
    if (Z1_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z1_batch is NULL; it should point to a valid array of ancora_zonotope pointers.");
    }
    if (Z2_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z2_batch is NULL; it should point to a valid array of ancora_zonotope pointers.");
    }
    if (B < 0) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "B is negative (%ld); it should be nonnegative.", (long)B);
    }

    if (B == 0) {
        return ANCORA_OK; /* vacuously nothing to do */
    }

    if (Z1_batch[0] == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z1_batch[0] is NULL; it should point to a valid ancora_zonotope instance.");
    }
    slong n = Z1_batch[0]->c.nrows;

    // Validate every entry up front.
    for (slong b = 0; b < B; b++) {
        if (res_batch[b] == NULL) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res_batch[%ld] is NULL; it should point to a valid ancora_zonotope instance.", (long)b);
        }
        if (Z1_batch[b] == NULL) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z1_batch[%ld] is NULL; it should point to a valid ancora_zonotope instance.", (long)b);
        }
        if (Z2_batch[b] == NULL) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z2_batch[%ld] is NULL; it should point to a valid ancora_zonotope instance.", (long)b);
        }

        slong nRes, n1, n2;
        ANCORA_TRY(ancora_zonotope_dimension(res_batch[b], &nRes));
        ANCORA_TRY(ancora_zonotope_dimension(Z1_batch[b], &n1));
        ANCORA_TRY(ancora_zonotope_dimension(Z2_batch[b], &n2));

        if (n1 != n) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "Every zonotope must have dimension %ld, but Z1_batch[%ld] has dimension %ld.",
                          (long)n, (long)b, (long)n1);
        }
        if (n2 != n) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "Every zonotope must have dimension %ld, but Z2_batch[%ld] has dimension %ld.",
                          (long)n, (long)b, (long)n2);
        }
        if (nRes != n) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "Every zonotope must have dimension %ld, but res_batch[%ld] has dimension %ld.",
                          (long)n, (long)b, (long)nRes);
        }

        slong m1 = Z1_batch[b]->G.ncols;
        slong m2 = Z2_batch[b]->G.ncols;
        slong mRes = res_batch[b]->G.ncols;
        if (mRes != m1 + m2) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "res_batch[%ld] should have %ld generators, but has %ld.",
                          (long)b, (long)(m1 + m2), (long)mRes);
        }
    }

    // Centers: entrywise addition, batched like ancora_interval_batched_minkowskiSum
#if ANCORA_MODE == ANCORA_MODE_SAFE
    for (slong b = 0; b < B; b++) {
        for (slong i = 0; i < n; i++) {
            arb_add(arb_mat_entry(res_batch[b]->c.repr, i, 0),
                    arb_mat_entry(Z1_batch[b]->c.repr, i, 0),
                    arb_mat_entry(Z2_batch[b]->c.repr, i, 0),
                    ANCORA_DEFAULT_PREC);
        }
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
    #ifdef ANCORA_USE_GPU
        double *c1_flat = (double *)malloc((size_t)(B * n) * sizeof(double));
        double *c2_flat = (double *)malloc((size_t)(B * n) * sizeof(double));
        double *cres_flat = (double *)malloc((size_t)(B * n) * sizeof(double));
        if (c1_flat == NULL || c2_flat == NULL || cres_flat == NULL) {
            free(c1_flat); free(c2_flat); free(cres_flat);
            ANCORA_ERROR(ANCORA_ERROR_ALLOC, "ancora_zonotope_batched_minkowskiSum: failed to allocate GPU packing buffers for centers.");
        }
        for (slong b = 0; b < B; b++) {
            memcpy(&c1_flat[b * n], Z1_batch[b]->c.repr, (size_t)n * sizeof(double));
            memcpy(&c2_flat[b * n], Z2_batch[b]->c.repr, (size_t)n * sizeof(double));
        }

        int gpu_status = ancora_mat_add_gpu(c1_flat, c2_flat, cres_flat, (size_t)(B * n));

        if (gpu_status == 0) {
            for (slong b = 0; b < B; b++) {
                memcpy(res_batch[b]->c.repr, &cres_flat[b * n], (size_t)n * sizeof(double));
            }
        }

        free(c1_flat); free(c2_flat); free(cres_flat);

        if (gpu_status != 0) {
            ANCORA_ERROR(ANCORA_ERROR_GPU_LAUNCH, "ancora_zonotope_batched_minkowskiSum: GPU kernel launch failed (centers).");
        }
    #else
        for (slong b = 0; b < B; b++) {
            for (slong i = 0; i < n; i++) {
                res_batch[b]->c.repr[i] = Z1_batch[b]->c.repr[i] + Z2_batch[b]->c.repr[i];
            }
        }
    #endif
#endif

    // Generators: pure concatenation, no arithmetic, no GPU angle
    for (slong b = 0; b < B; b++) {
        slong m1 = Z1_batch[b]->G.ncols;
        slong m2 = Z2_batch[b]->G.ncols;
        slong mRes = m1 + m2;
#if ANCORA_MODE == ANCORA_MODE_SAFE
        for (slong i = 0; i < n; i++) {
            for (slong j = 0; j < m1; j++) {
                arb_set(arb_mat_entry(res_batch[b]->G.repr, i, j), arb_mat_entry(Z1_batch[b]->G.repr, i, j));
            }
            for (slong j = 0; j < m2; j++) {
                arb_set(arb_mat_entry(res_batch[b]->G.repr, i, m1 + j), arb_mat_entry(Z2_batch[b]->G.repr, i, j));
            }
        }
#elif ANCORA_MODE == ANCORA_MODE_FAST
        for (slong i = 0; i < n; i++) {
            memcpy(&res_batch[b]->G.repr[i * mRes], &Z1_batch[b]->G.repr[i * m1], (size_t)m1 * sizeof(double));
            memcpy(&res_batch[b]->G.repr[i * mRes + m1], &Z2_batch[b]->G.repr[i * m2], (size_t)m2 * sizeof(double));
        }
#endif
    }

    return ANCORA_OK;
}

ancora_status ancora_zonotope_affine(ancora_zonotope *res,
                                     const ancora_mat *A,
                                     const ancora_vec *c,
                                     const ancora_zonotope *Z)
/* Affine map of a zonotope: res = A*Z + c = { A*x + c | x in Z }.
 *
 * This concatenates Z.c and Z.G into one (m x (p+1)) matrix and performs
 * A*[Z.c | Z.G] as a SINGLE matmul, rather than two separate matmuls
 * (res.c = A*Z.c, res.G = A*Z.G) as an earlier version of this function
 * did. Since both products share the same left operand A, this halves
 * the matmul dispatch count (and, with ANCORA_USE_GPU, the GPU kernel
 * launch count) for identical total arithmetic -- the concatenation/
 * splitting overhead is O(n*m + n*(p+1)), negligible next to the
 * O(n*m*p) matmul itself.
 *
 * INPUT:
 *      res             : Result zonotope, already initialized as dimension n
 *                        with p generators
 *      A               : Matrix (n x m)
 *      c               : Translation vector (length n)
 *      Z               : Zonotope to map (dimension m, p generators)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*m*p*ANCORA_DEFAULT_PREC) (one (n x m)*(m x (p+1)) matmul,
 *      instead of two separate (n x m)*(m x 1) and (n x m)*(m x p)
 *      matmuls of the same combined cost, but issued as one dispatch).
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-27
 * Author(s):     Adrian Kulmburg
 */
{
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_zonotope instance.");
    }
    if (A == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer A is NULL; it should point to a valid ancora_mat instance.");
    }
    if (c == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer c is NULL; it should point to a valid ancora_vec instance.");
    }
    if (Z == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z is NULL; it should point to a valid ancora_zonotope instance.");
    }
    bool isVector;
    ANCORA_TRY(ancora_mat_isVector(c, &isVector));
    if (!isVector) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer c is not a vector; it should point to a valid ancora_vec instance.");
    }

    slong mZ;
    ANCORA_TRY(ancora_zonotope_dimension(Z, &mZ));
    slong n;
    ANCORA_TRY(ancora_zonotope_dimension(res, &n));

    slong p = res->G.ncols;
    slong pZ = Z->G.ncols;

    if (mZ != A->ncols) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Matrix A has %ld columns, zonotope Z has dimension %ld; they need to be the same.",
                      (long)A->ncols, (long)mZ);
    }
    if (c->nrows != A->nrows) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Matrix A has %ld rows, translation vector c has length %ld; they need to be the same.",
                      (long)A->nrows, (long)c->nrows);
    }
    if (n != A->nrows) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Result should have dimension %ld, but res has dimension %ld.",
                      (long)A->nrows, (long)n);
    }
    if (p != pZ) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Result should have %ld generators, but res has %ld.",
                      (long)pZ, (long)p);
    }
    if (res == Z) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "res must not alias Z for ancora_zonotope_affine.");
    }

    ancora_mat CG, RCG;
    ANCORA_TRY(ancora_mat_init(&CG, mZ, p + 1));
    ANCORA_TRY(ancora_mat_init(&RCG, n, p + 1));

    // CG = [Z.c | Z.G]
    for (slong i = 0; i < mZ; i++) {
#if ANCORA_MODE == ANCORA_MODE_SAFE
        arb_set(arb_mat_entry(CG.repr, i, 0), arb_mat_entry(Z->c.repr, i, 0));
        for (slong j = 0; j < p; j++) {
            arb_set(arb_mat_entry(CG.repr, i, 1 + j), arb_mat_entry(Z->G.repr, i, j));
        }
#elif ANCORA_MODE == ANCORA_MODE_FAST
        CG.repr[i * (p + 1)] = Z->c.repr[i];
        memcpy(&CG.repr[i * (p + 1) + 1], &Z->G.repr[i * p], (size_t)p * sizeof(double));
#endif
    }

    // ONE matmul: RCG = A * [Z.c | Z.G]
    ANCORA_TRY(ancora_mat_mul(&RCG, A, &CG));

    // Split back apart: column 0 (+c) is res.c, columns [1, p+1) are res.G.
    for (slong i = 0; i < n; i++) {
#if ANCORA_MODE == ANCORA_MODE_SAFE
        arb_add(arb_mat_entry(res->c.repr, i, 0), arb_mat_entry(RCG.repr, i, 0), arb_mat_entry(c->repr, i, 0), ANCORA_DEFAULT_PREC);
        for (slong j = 0; j < p; j++) {
            arb_set(arb_mat_entry(res->G.repr, i, j), arb_mat_entry(RCG.repr, i, 1 + j));
        }
#elif ANCORA_MODE == ANCORA_MODE_FAST
        res->c.repr[i] = RCG.repr[i * (p + 1)] + c->repr[i];
        memcpy(&res->G.repr[i * p], &RCG.repr[i * (p + 1) + 1], (size_t)p * sizeof(double));
#endif
    }

    ANCORA_TRY(ancora_mat_free(&CG));
    ANCORA_TRY(ancora_mat_free(&RCG));

    return ANCORA_OK;
}

ancora_status ancora_zonotope_batched_affine(
    ancora_zonotope **res_batch,
    const ancora_mat *A,
    const ancora_vec *c,
    const ancora_zonotope **Z_batch,
    slong B)
/* Computes res_batch[b] = A*Z_batch[b] + c for every b in [0, B). A (n x m)
 * and c (length n) are SHARED across the whole batch; only the zonotopes
 * differ. All Z_batch[b] must have dimension m (but may each have a
 * DIFFERENT number of generators p_b); res_batch[b] must have dimension n
 * and p_b generators (matching Z_batch[b]'s own generator count, same
 * requirement as the single-instance ancora_zonotope_affine).
 *
 * This packs every pair's generators AND center into ONE combined
 * (m x (P+B)) matrix (P = sum_b p_b) -- columns [0, P) hold every pair's
 * generators (as before), columns [P, P+B) hold every pair's center, one
 * per column -- and performs A*Combined as a SINGLE matmul, rather than
 * two separate matmuls (Gres = A*Gstack, Cres = A*Cstack) as an earlier
 * version of this function did. Since both share the same left operand
 * A, this consolidates the batch's entire linear-algebra cost into
 * exactly ONE matmul dispatch (and, with ANCORA_USE_GPU, one GPU kernel
 * launch) instead of two.
 *
 * INPUT:
 *      res_batch       : Array of B pointers, each already initialized
 *                        with dimension n and p_b generators (p_b =
 *                        Z_batch[b]->G.ncols); res_batch[b] receives
 *                        A*Z_batch[b] + c
 *      A               : Linear map (n x m), shared across the batch
 *      c               : Translation vector (length n), shared across
 *                        the batch
 *      Z_batch         : Array of B pointers to initialized
 *                        ancora_zonotope instances, all of dimension m
 *                        (generator counts may differ)
 *      B               : Number of zonotopes in the batch (>= 0)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*m*P*ANCORA_DEFAULT_PREC) where P = sum_b p_b -- identical total
 *      arithmetic to B separate ancora_zonotope_affine calls; the
 *      packing/slicing passes add O(m*(P+B) + n*(P+B)), the same order as
 *      the work itself. The improvement is consolidating the batch's
 *      matmul work into exactly ONE dispatch (and one GPU kernel launch,
 *      if ANCORA_USE_GPU) instead of two.
 *
 * Created:       2026-09-27
 * Last modified: 2026-09-27
 * Author(s):     Adrian Kulmburg
 */
{
    if (res_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res_batch is NULL; it should point to a valid array of ancora_zonotope pointers.");
    }
    if (A == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer A is NULL; it should point to a valid ancora_mat instance.");
    }
    if (c == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer c is NULL; it should point to a valid ancora_vec instance.");
    }
    if (Z_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z_batch is NULL; it should point to a valid array of ancora_zonotope pointers.");
    }
    if (B < 0) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "B is negative (%ld); it should be nonnegative.", (long)B);
    }

    bool isVector;
    ANCORA_TRY(ancora_mat_isVector(c, &isVector));
    if (!isVector) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer c is not a vector; it should point to a valid ancora_vec instance.");
    }

    slong n = A->nrows;
    slong m = A->ncols;

    if (c->nrows != n) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Matrix A has %ld rows, translation vector c has length %ld; they need to be the same.",
                      (long)n, (long)c->nrows);
    }

    if (B == 0) {
        return ANCORA_OK; /* vacuously nothing to do */
    }

    slong *p = (slong *)malloc((size_t)B * sizeof(slong));
    slong *offset = (slong *)malloc((size_t)B * sizeof(slong));
    if (p == NULL || offset == NULL) {
        free(p);
        free(offset);
        ANCORA_ERROR(ANCORA_ERROR_ALLOC, "ancora_zonotope_batched_affine: failed to allocate per-pair bookkeeping arrays.");
    }

    ancora_status status = ANCORA_OK;
    slong P = 0;

    for (slong b = 0; b < B; b++) {
        if (Z_batch[b] == NULL) {
            status = ANCORA_ERROR_INVALID_ARG;
            ANCORA_ERROR(status, "Pointer Z_batch[%ld] is NULL; it should point to a valid ancora_zonotope instance.", (long)b);
        }
        if (res_batch[b] == NULL) {
            status = ANCORA_ERROR_INVALID_ARG;
            ANCORA_ERROR(status, "Pointer res_batch[%ld] is NULL; it should point to a valid ancora_zonotope instance.", (long)b);
        }
        if (res_batch[b] == Z_batch[b]) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "res_batch[%ld] must not alias Z_batch[%ld] for ancora_zonotope_batched_affine.", (long)b, (long)b);
        }

        slong mZ, nRes, pRes;
        ANCORA_TRY(ancora_zonotope_dimension(Z_batch[b], &mZ));
        ANCORA_TRY(ancora_zonotope_dimension(res_batch[b], &nRes));
        pRes = res_batch[b]->G.ncols;
        p[b] = Z_batch[b]->G.ncols;

        if (mZ != m) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "Matrix A has %ld columns, Z_batch[%ld] has dimension %ld; they need to be the same.",
                          (long)m, (long)b, (long)mZ);
        }
        if (nRes != n) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "res_batch[%ld] should have dimension %ld, but has dimension %ld.",
                          (long)b, (long)n, (long)nRes);
        }
        if (pRes != p[b]) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "res_batch[%ld] should have %ld generators, but has %ld.",
                          (long)b, (long)p[b], (long)pRes);
        }

        offset[b] = P;
        P += p[b];
    }

    ancora_mat Combined, ResCombined;
    bool CombinedInit = false, ResCombinedInit = false;

    // Combined has P+B columns: [0, P) generators, [P, P+B) centers.
    status = ancora_mat_init(&Combined, m, P + B);
    if (status != ANCORA_OK) goto cleanup;
    CombinedInit = true;
    status = ancora_mat_init(&ResCombined, n, P + B);
    if (status != ANCORA_OK) goto cleanup;
    ResCombinedInit = true;

    for (slong b = 0; b < B; b++) {
        for (slong i = 0; i < m; i++) {
#if ANCORA_MODE == ANCORA_MODE_SAFE
            arb_set(arb_mat_entry(Combined.repr, i, P + b), arb_mat_entry(Z_batch[b]->c.repr, i, 0));
            for (slong j = 0; j < p[b]; j++) {
                arb_set(arb_mat_entry(Combined.repr, i, offset[b] + j), arb_mat_entry(Z_batch[b]->G.repr, i, j));
            }
#elif ANCORA_MODE == ANCORA_MODE_FAST
            Combined.repr[i * (P + B) + P + b] = Z_batch[b]->c.repr[i];
            memcpy(&Combined.repr[i * (P + B) + offset[b]], &Z_batch[b]->G.repr[i * p[b]], (size_t)p[b] * sizeof(double));
#endif
        }
    }

    // Exactly ONE matmul for the entire batch (generators AND centers).
    status = ancora_mat_mul(&ResCombined, A, &Combined);
    if (status != ANCORA_OK) goto cleanup;

    for (slong b = 0; b < B; b++) {
        for (slong i = 0; i < n; i++) {
#if ANCORA_MODE == ANCORA_MODE_SAFE
            arb_add(arb_mat_entry(res_batch[b]->c.repr, i, 0),
                    arb_mat_entry(ResCombined.repr, i, P + b),
                    arb_mat_entry(c->repr, i, 0),
                    ANCORA_DEFAULT_PREC);
            for (slong j = 0; j < p[b]; j++) {
                arb_set(arb_mat_entry(res_batch[b]->G.repr, i, j), arb_mat_entry(ResCombined.repr, i, offset[b] + j));
            }
#elif ANCORA_MODE == ANCORA_MODE_FAST
            res_batch[b]->c.repr[i] = ResCombined.repr[i * (P + B) + P + b] + c->repr[i];
            memcpy(&res_batch[b]->G.repr[i * p[b]], &ResCombined.repr[i * (P + B) + offset[b]], (size_t)p[b] * sizeof(double));
#endif
        }
    }

cleanup:
    if (CombinedInit) ancora_mat_free(&Combined);
    if (ResCombinedInit) ancora_mat_free(&ResCombined);
    free(p);
    free(offset);

    if (status != ANCORA_OK) {
        ANCORA_ERROR(status, "ancora_zonotope_batched_affine: concatenation/matmul/slicing failed (see above).");
    }
    return ANCORA_OK;
}

ancora_status ancora_zonotope_matMul(ancora_zonotope *res,
                                     const ancora_mat *A,
                                     const ancora_zonotope *Z)
/* Matrix map of a zonotope, without translation: res = A*Z = { A*x | x in Z }.
 *
 * Same computation as ancora_zonotope_affine with c omitted entirely (not
 * even a zero vector is allocated). Like that function, this concatenates
 * Z.c and Z.G into one (m x (p+1)) matrix and performs A*[Z.c | Z.G] as a
 * SINGLE matmul rather than two separate ones.
 *
 * INPUT:
 *      res             : Result zonotope, already initialized as dimension n
 *                        with p generators
 *      A               : Matrix (n x m)
 *      Z               : Zonotope to map (dimension m, p generators)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*m*p*ANCORA_DEFAULT_PREC) (one (n x m)*(m x (p+1)) matmul; see
 *      ancora_zonotope_affine).
 *
 * Created:       2026-09-27
 * Last modified: 2026-09-27
 * Author(s):     Adrian Kulmburg
 */
{
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_zonotope instance.");
    }
    if (A == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer A is NULL; it should point to a valid ancora_mat instance.");
    }
    if (Z == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z is NULL; it should point to a valid ancora_zonotope instance.");
    }

    slong mZ;
    ANCORA_TRY(ancora_zonotope_dimension(Z, &mZ));
    slong n;
    ANCORA_TRY(ancora_zonotope_dimension(res, &n));

    slong p = res->G.ncols;
    slong pZ = Z->G.ncols;

    if (mZ != A->ncols) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Matrix A has %ld columns, zonotope Z has dimension %ld; they need to be the same.",
                      (long)A->ncols, (long)mZ);
    }
    if (n != A->nrows) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Result should have dimension %ld, but res has dimension %ld.",
                      (long)A->nrows, (long)n);
    }
    if (p != pZ) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Result should have %ld generators, but res has %ld.",
                      (long)pZ, (long)p);
    }
    if (res == Z) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "res must not alias Z for ancora_zonotope_matMul.");
    }

    ancora_mat CG, RCG;
    ANCORA_TRY(ancora_mat_init(&CG, mZ, p + 1));
    ANCORA_TRY(ancora_mat_init(&RCG, n, p + 1));

    // CG = [Z.c | Z.G]
    for (slong i = 0; i < mZ; i++) {
#if ANCORA_MODE == ANCORA_MODE_SAFE
        arb_set(arb_mat_entry(CG.repr, i, 0), arb_mat_entry(Z->c.repr, i, 0));
        for (slong j = 0; j < p; j++) {
            arb_set(arb_mat_entry(CG.repr, i, 1 + j), arb_mat_entry(Z->G.repr, i, j));
        }
#elif ANCORA_MODE == ANCORA_MODE_FAST
        CG.repr[i * (p + 1)] = Z->c.repr[i];
        memcpy(&CG.repr[i * (p + 1) + 1], &Z->G.repr[i * p], (size_t)p * sizeof(double));
#endif
    }

    // ONE matmul: RCG = A * [Z.c | Z.G]
    ANCORA_TRY(ancora_mat_mul(&RCG, A, &CG));

    // Split back apart: no c to add here, column 0 is res.c directly.
    for (slong i = 0; i < n; i++) {
#if ANCORA_MODE == ANCORA_MODE_SAFE
        arb_set(arb_mat_entry(res->c.repr, i, 0), arb_mat_entry(RCG.repr, i, 0));
        for (slong j = 0; j < p; j++) {
            arb_set(arb_mat_entry(res->G.repr, i, j), arb_mat_entry(RCG.repr, i, 1 + j));
        }
#elif ANCORA_MODE == ANCORA_MODE_FAST
        res->c.repr[i] = RCG.repr[i * (p + 1)];
        memcpy(&res->G.repr[i * p], &RCG.repr[i * (p + 1) + 1], (size_t)p * sizeof(double));
#endif
    }

    ANCORA_TRY(ancora_mat_free(&CG));
    ANCORA_TRY(ancora_mat_free(&RCG));

    return ANCORA_OK;
}

ancora_status ancora_zonotope_batched_matMul(
    ancora_zonotope **res_batch,
    const ancora_mat *A,
    const ancora_zonotope **Z_batch,
    slong B)
/* Computes res_batch[b] = A*Z_batch[b] for every b in [0, B) (no
 * translation). A (n x m) is SHARED across the batch; only the zonotopes
 * differ (generator counts p_b may differ). Same as
 * ancora_zonotope_batched_affine with c omitted: packs every pair's
 * generators AND center into one combined (m x (P+B)) matrix and performs
 * A*Combined as a SINGLE matmul for the whole batch.
 *
 * INPUT:
 *      res_batch       : Array of B pointers, each already initialized
 *                        with dimension n and p_b generators (p_b =
 *                        Z_batch[b]->G.ncols); res_batch[b] receives
 *                        A*Z_batch[b]
 *      A               : Linear map (n x m), shared across the batch
 *      Z_batch         : Array of B pointers to initialized
 *                        ancora_zonotope instances, all of dimension m
 *                        (generator counts may differ)
 *      B               : Number of zonotopes in the batch (>= 0)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*m*P*ANCORA_DEFAULT_PREC) where P = sum_b p_b (see
 *      ancora_zonotope_batched_affine); one matmul dispatch for the
 *      entire batch.
 *
 * Created:       2026-09-27
 * Last modified: 2026-09-27
 * Author(s):     Adrian Kulmburg
 */
{
    if (res_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res_batch is NULL; it should point to a valid array of ancora_zonotope pointers.");
    }
    if (A == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer A is NULL; it should point to a valid ancora_mat instance.");
    }
    if (Z_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z_batch is NULL; it should point to a valid array of ancora_zonotope pointers.");
    }
    if (B < 0) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "B is negative (%ld); it should be nonnegative.", (long)B);
    }

    slong n = A->nrows;
    slong m = A->ncols;

    if (B == 0) {
        return ANCORA_OK; /* vacuously nothing to do */
    }

    slong *p = (slong *)malloc((size_t)B * sizeof(slong));
    slong *offset = (slong *)malloc((size_t)B * sizeof(slong));
    if (p == NULL || offset == NULL) {
        free(p);
        free(offset);
        ANCORA_ERROR(ANCORA_ERROR_ALLOC, "ancora_zonotope_batched_matMul: failed to allocate per-pair bookkeeping arrays.");
    }

    ancora_status status = ANCORA_OK;
    slong P = 0;

    for (slong b = 0; b < B; b++) {
        if (Z_batch[b] == NULL) {
            status = ANCORA_ERROR_INVALID_ARG;
            ANCORA_ERROR(status, "Pointer Z_batch[%ld] is NULL; it should point to a valid ancora_zonotope instance.", (long)b);
        }
        if (res_batch[b] == NULL) {
            status = ANCORA_ERROR_INVALID_ARG;
            ANCORA_ERROR(status, "Pointer res_batch[%ld] is NULL; it should point to a valid ancora_zonotope instance.", (long)b);
        }
        if (res_batch[b] == Z_batch[b]) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "res_batch[%ld] must not alias Z_batch[%ld] for ancora_zonotope_batched_matMul.", (long)b, (long)b);
        }

        slong mZ, nRes, pRes;
        ANCORA_TRY(ancora_zonotope_dimension(Z_batch[b], &mZ));
        ANCORA_TRY(ancora_zonotope_dimension(res_batch[b], &nRes));
        pRes = res_batch[b]->G.ncols;
        p[b] = Z_batch[b]->G.ncols;

        if (mZ != m) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "Matrix A has %ld columns, Z_batch[%ld] has dimension %ld; they need to be the same.",
                          (long)m, (long)b, (long)mZ);
        }
        if (nRes != n) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "res_batch[%ld] should have dimension %ld, but has dimension %ld.",
                          (long)b, (long)n, (long)nRes);
        }
        if (pRes != p[b]) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "res_batch[%ld] should have %ld generators, but has %ld.",
                          (long)b, (long)p[b], (long)pRes);
        }

        offset[b] = P;
        P += p[b];
    }

    ancora_mat Combined, ResCombined;
    bool CombinedInit = false, ResCombinedInit = false;

    status = ancora_mat_init(&Combined, m, P + B);
    if (status != ANCORA_OK) goto cleanup;
    CombinedInit = true;
    status = ancora_mat_init(&ResCombined, n, P + B);
    if (status != ANCORA_OK) goto cleanup;
    ResCombinedInit = true;

    for (slong b = 0; b < B; b++) {
        for (slong i = 0; i < m; i++) {
#if ANCORA_MODE == ANCORA_MODE_SAFE
            arb_set(arb_mat_entry(Combined.repr, i, P + b), arb_mat_entry(Z_batch[b]->c.repr, i, 0));
            for (slong j = 0; j < p[b]; j++) {
                arb_set(arb_mat_entry(Combined.repr, i, offset[b] + j), arb_mat_entry(Z_batch[b]->G.repr, i, j));
            }
#elif ANCORA_MODE == ANCORA_MODE_FAST
            Combined.repr[i * (P + B) + P + b] = Z_batch[b]->c.repr[i];
            memcpy(&Combined.repr[i * (P + B) + offset[b]], &Z_batch[b]->G.repr[i * p[b]], (size_t)p[b] * sizeof(double));
#endif
        }
    }

    // ONE matmul for the entire batch (generators AND centers, no c to add).
    status = ancora_mat_mul(&ResCombined, A, &Combined);
    if (status != ANCORA_OK) goto cleanup;

    for (slong b = 0; b < B; b++) {
        for (slong i = 0; i < n; i++) {
#if ANCORA_MODE == ANCORA_MODE_SAFE
            arb_set(arb_mat_entry(res_batch[b]->c.repr, i, 0), arb_mat_entry(ResCombined.repr, i, P + b));
            for (slong j = 0; j < p[b]; j++) {
                arb_set(arb_mat_entry(res_batch[b]->G.repr, i, j), arb_mat_entry(ResCombined.repr, i, offset[b] + j));
            }
#elif ANCORA_MODE == ANCORA_MODE_FAST
            res_batch[b]->c.repr[i] = ResCombined.repr[i * (P + B) + P + b];
            memcpy(&res_batch[b]->G.repr[i * p[b]], &ResCombined.repr[i * (P + B) + offset[b]], (size_t)p[b] * sizeof(double));
#endif
        }
    }

cleanup:
    if (CombinedInit) ancora_mat_free(&Combined);
    if (ResCombinedInit) ancora_mat_free(&ResCombined);
    free(p);
    free(offset);

    if (status != ANCORA_OK) {
        ANCORA_ERROR(status, "ancora_zonotope_batched_matMul: concatenation/matmul/slicing failed (see above).");
    }
    return ANCORA_OK;
}
