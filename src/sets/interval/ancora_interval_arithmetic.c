/*
 * ancora_interval_arithmetic.c
 *
 * Description
 * ------
 * Arithmetic operations for ancora_interval.
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

#include "ancora/sets/interval/ancora_interval_arithmetic.h"

ancora_status ancora_interval_minkowskiSum(ancora_interval *res,
                                           const ancora_interval *I,
                                           const ancora_interval *J)
/* Minkowski sum of two intervals: res = I + J = { x + y | x in I, y in J }.
 *
 * This computes the entrywise addition directly against
 * lowerBound/upperBound.repr rather than delegating to ancora_vec_add:
 * that generic path re-validates NULL/dimensions this function has
 * already checked, and -- critically -- if ANCORA_USE_GPU is defined,
 * ancora_vec_add's underlying ancora_mat_add unconditionally dispatches
 * to the GPU, meaning every call here would pay TWO full GPU round trips
 * (hipMalloc/hipMemcpy/kernel launch/hipMemcpy back/hipFree, once each
 * for lowerBound and upperBound) for what is typically a tiny O(n)
 * addition -- GPU dispatch overhead dominating a cheap elementwise op is
 * exactly the caveat flagged for ancora_interval_batched_minkowskiSum;
 * unlike that function, this single-instance one has no size threshold
 * to justify paying it at all, so it never should.
 *
 * INPUT:
 *      res             : Result interval, already initialized with the same
 *                        dimension as I and J
 *      I               : First summand
 *      J               : Second summand
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-27
 * Author(s):     Adrian Kulmburg
 */
{
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_interval instance.");
    }
    if (I == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I is NULL; it should point to a valid ancora_interval instance.");
    }
    if (J == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer J is NULL; it should point to a valid ancora_interval instance.");
    }

    slong Idim, Jdim, resDim;
    ANCORA_TRY(ancora_interval_dimension(I, &Idim));
    ANCORA_TRY(ancora_interval_dimension(J, &Jdim));
    ANCORA_TRY(ancora_interval_dimension(res, &resDim));

    if (Idim != Jdim) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Interval I has dimension %ld, interval J has dimension %ld; they need to be the same.",
                      (long)Idim, (long)Jdim);
    }
    if (resDim != Idim) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Intervals I and J have dimension %ld, interval res has dimension %ld; they need to be the same.",
                      (long)Idim, (long)Jdim);
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    for (slong i = 0; i < Idim; i++) {
        arb_add(arb_mat_entry(res->lowerBound.repr, i, 0),
                arb_mat_entry(I->lowerBound.repr, i, 0),
                arb_mat_entry(J->lowerBound.repr, i, 0), ANCORA_DEFAULT_PREC);
        arb_add(arb_mat_entry(res->upperBound.repr, i, 0),
                arb_mat_entry(I->upperBound.repr, i, 0),
                arb_mat_entry(J->upperBound.repr, i, 0), ANCORA_DEFAULT_PREC);
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
    for (slong i = 0; i < Idim; i++) {
        res->lowerBound.repr[i] = I->lowerBound.repr[i] + J->lowerBound.repr[i];
        res->upperBound.repr[i] = I->upperBound.repr[i] + J->upperBound.repr[i];
    }
#endif

    return ANCORA_OK;
}


ancora_status ancora_interval_batched_minkowskiSum(
    ancora_interval **res_batch,
    const ancora_interval **I_batch,
    const ancora_interval **J_batch,
    slong B)
/* Computes res_batch[b] = I_batch[b] + J_batch[b] (Minkowski sum) for every
 * b in [0, B). All intervals across the whole batch (every I_batch[b],
 * J_batch[b], res_batch[b]) must share the same dimension n.
 *
 * Minkowski sum on intervals is a purely entrywise operation (unlike
 * ancora_interval_batched_supportFunction's reduction, or
 * ancora_interval_batched_containsPoints's per-point branching), so this
 * fuses directly into per-coordinate addition with no intermediate
 * ancora_vec_add calls. Note that ancora_interval_minkowskiSum itself is
 * already very cheap (two dimension lookups, two ancora_vec_add calls, NO
 * temporary allocations) - so unlike the support-function/containment
 * batches, this fusion does not eliminate a real per-call allocation or
 * redundant-revalidation cost; its benefit is amortizing the small FIXED
 * per-call overhead (function calls, dimension lookups, NULL checks)
 * across the batch, which only matters when n is small relative to B.
 *
 * In ANCORA_MODE_FAST with ANCORA_USE_GPU enabled, this packs every
 * interval's lowerBound (and separately, upperBound) into one flat
 * (B*n)-length array and reuses the EXISTING ancora_mat_add_gpo kernel
 * directly (elementwise addition of two flat double arrays is exactly
 * what Minkowski sum on intervals reduces to) - no new GPU kernel is
 * needed for this operation.
 *
 * INPUT:
 *      res_batch       : Array of B pointers, each already initialized
 *                        with dimension n; res_batch[b] receives
 *                        I_batch[b] + J_batch[b]
 *      I_batch         : Array of B pointers to initialized
 *                        ancora_interval instances, all of dimension n
 *      J_batch         : Array of B pointers to initialized
 *                        ancora_interval instances, all of dimension n
 *      B               : Number of (I, J) pairs (>= 0)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*B*ANCORA_DEFAULT_PREC): identical total work to B separate
 *      calls to ancora_interval_minkowskiSum (this is entrywise addition,
 *      so there is no less work possible) - the improvement is one
 *      validation pass instead of B, and, with ANCORA_USE_GPU, reuse of
 *      the existing elementwise-add kernel via an O(n*B) packing pass.
 *
 * Created:       2026-09-26
 * Last modified: 2026-09-26
 * Author(s):     Adrian Kulmburg
 */
{
    if (res_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res_batch is NULL; it should point to a valid array of ancora_interval pointers.");
    }
    if (I_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I_batch is NULL; it should point to a valid array of ancora_interval pointers.");
    }
    if (J_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer J_batch is NULL; it should point to a valid array of ancora_interval pointers.");
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
        if (res_batch[b] == NULL) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res_batch[%ld] is NULL; it should point to a valid ancora_interval instance.", (long)b);
        }
        if (I_batch[b] == NULL) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I_batch[%ld] is NULL; it should point to a valid ancora_interval instance.", (long)b);
        }
        if (J_batch[b] == NULL) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer J_batch[%ld] is NULL; it should point to a valid ancora_interval instance.", (long)b);
        }
        slong nI, nJ, nRes;
        ANCORA_TRY(ancora_interval_dimension(I_batch[b], &nI));
        ANCORA_TRY(ancora_interval_dimension(J_batch[b], &nJ));
        ANCORA_TRY(ancora_interval_dimension(res_batch[b], &nRes));
        if (nI != n || nJ != n || nRes != n) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "Every interval in the batch must share dimension %ld, but pair %ld has I: %ld, J: %ld, res: %ld.",
                          (long)n, (long)b, (long)nI, (long)nJ, (long)nRes);
        }
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    for (slong b = 0; b < B; b++) {
        for (slong i = 0; i < n; i++) {
            arb_add(arb_mat_entry(res_batch[b]->lowerBound.repr, i, 0),
                    arb_mat_entry(I_batch[b]->lowerBound.repr, i, 0),
                    arb_mat_entry(J_batch[b]->lowerBound.repr, i, 0),
                    ANCORA_DEFAULT_PREC);
            arb_add(arb_mat_entry(res_batch[b]->upperBound.repr, i, 0),
                    arb_mat_entry(I_batch[b]->upperBound.repr, i, 0),
                    arb_mat_entry(J_batch[b]->upperBound.repr, i, 0),
                    ANCORA_DEFAULT_PREC);
        }
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
        #ifdef ANCORA_USE_GPU
        // Pack BOTH lowerBound and upperBound into ONE combined buffer per
        // operand (length 2*n*B: first n*B entries are every lowerBound,
        // next n*B are every upperBound), rather than two separate buffers
        // dispatched as two separate ancora_mat_add_gpu calls. Since both
        // halves go through the identical elementwise-add kernel, this
        // halves the GPU round-trip overhead (one hipMalloc/hipMemcpy/
        // kernel-launch/hipMemcpy/hipFree cycle instead of two) for
        // identical total arithmetic -- meaningful here specifically
        // because Minkowski sum's actual compute is so cheap that dispatch
        // overhead, not arithmetic, is the dominant cost.
        slong half = n * B;
        double *I_combined = (double *)malloc((size_t)(2 * half) * sizeof(double));
        double *J_combined = (double *)malloc((size_t)(2 * half) * sizeof(double));
        double *res_combined = (double *)malloc((size_t)(2 * half) * sizeof(double));
        if (I_combined == NULL || J_combined == NULL || res_combined == NULL) {
            free(I_combined); free(J_combined); free(res_combined);
            ANCORA_ERROR(ANCORA_ERROR_ALLOC, "ancora_interval_batched_minkowskiSum: failed to allocate GPU packing buffers.");
        }
        for (slong b = 0; b < B; b++) {
            memcpy(&I_combined[b * n], I_batch[b]->lowerBound.repr, (size_t)n * sizeof(double));
            memcpy(&I_combined[half + b * n], I_batch[b]->upperBound.repr, (size_t)n * sizeof(double));
            memcpy(&J_combined[b * n], J_batch[b]->lowerBound.repr, (size_t)n * sizeof(double));
            memcpy(&J_combined[half + b * n], J_batch[b]->upperBound.repr, (size_t)n * sizeof(double));
        }

        int gpu_status = ancora_mat_add_gpu(I_combined, J_combined, res_combined, (size_t)(2 * half));

        if (gpu_status == 0) {
            for (slong b = 0; b < B; b++) {
                memcpy(res_batch[b]->lowerBound.repr, &res_combined[b * n], (size_t)n * sizeof(double));
                memcpy(res_batch[b]->upperBound.repr, &res_combined[half + b * n], (size_t)n * sizeof(double));
            }
        }

        free(I_combined); free(J_combined); free(res_combined);

        if (gpu_status != 0) {
            ANCORA_ERROR(ANCORA_ERROR_GPU_LAUNCH, "ancora_interval_batched_minkowskiSum: GPU kernel launch failed.");
        }
    #else
        for (slong b = 0; b < B; b++) {
            for (slong i = 0; i < n; i++) {
                res_batch[b]->lowerBound.repr[i] = I_batch[b]->lowerBound.repr[i] + J_batch[b]->lowerBound.repr[i];
                res_batch[b]->upperBound.repr[i] = I_batch[b]->upperBound.repr[i] + J_batch[b]->upperBound.repr[i];
            }
        }
    #endif
#endif

    return ANCORA_OK;
}

ancora_status ancora_interval_affine(ancora_interval *res,
                                     const ancora_mat *A,
                                     const ancora_vec *c,
                                     const ancora_interval *I)
/* Affine map of an interval: res = A*I + c = { A*x + c | x in I }.
 * A is (n x m), I has dimension m, c has length n, and res has dimension n.
 *
 * This does NOT build a dense (m x m) diagonal matrix B = diag(radius) and
 * compute G = A*B via a full (n x m)*(m x m) matmul, as an earlier version
 * of this function did -- that costs O(n*m^2), even though the ONLY use
 * of G afterward was extracting each row's 1-norm, and
 * (A*diag(radius))_ij = A_ij*radius_j means that row 1-norm is exactly
 * sum_j |A_ij|*radius_j = (|A|*radius)_i -- a plain (n x m)*(m x 1)
 * matrix-vector product, O(n*m), with no diagonal matrix or G ever
 * needed. This is the same realization ancora_interval_batched_affine
 * already made (R = |A|*Rad); this function now matches it.
 *
 * INPUT:
 *      res             : Result interval, already initialized as dimension n
 *      A               : Linear map (n x m)
 *      c               : Translation vector (length n)
 *      I               : Interval to map (dimension m)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*m*ANCORA_DEFAULT_PREC): one entrywise abs of A, two (n x m)*
 *      (m x 1) mat-vec products (A*d and |A|*radius), versus
 *      O(n*m^2*ANCORA_DEFAULT_PREC) for the previous B/G-based version.
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-27
 * Author(s):     Adrian Kulmburg
 */
{
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_interval instance.");
    }
    if (A == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer A is NULL; it should point to a valid ancora_mat instance.");
    }
    if (c == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer c is NULL; it should point to a valid ancora_vec instance.");
    }
    if (I == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I is NULL; it should point to a valid ancora_interval instance.");
    }

    bool isVector;
    ANCORA_TRY(ancora_mat_isVector(c, &isVector));
    if (!isVector) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer c is not a vector; it should point to a valid ancora_vec instance.");
    }

    slong n = A->nrows;
    slong m = A->ncols;

    slong I_dimension;
    ANCORA_TRY(ancora_interval_dimension(I, &I_dimension));
    slong res_dimension;
    ANCORA_TRY(ancora_interval_dimension(res, &res_dimension));

    if (I_dimension != m) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Matrix A has %ld columns, interval I has dimension %ld; they need to be the same.",
                      (long)m, (long)I_dimension);
    }
    if (c->nrows != n) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Matrix A has %ld rows, translation vector c has length %ld; they need to be the same.",
                      (long)n, (long)c->nrows);
    }
    if (res_dimension != n) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Result should have dimension %ld, but res has dimension %ld.",
                      (long)n, (long)res_dimension);
    }
    if (res == I) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "res must not alias I for ancora_interval_affine.");
    }

    /* This whole ordeal only works if I is bounded; if not, the easiest
     * way around is to use polytopes I guess, but this will be
     * implemented later. */
#if ANCORA_MODE == ANCORA_MODE_SAFE
    for (slong j = 0; j < m; j++) {
        if (!arb_is_finite(arb_mat_entry(I->lowerBound.repr, j, 0)) ||
            !arb_is_finite(arb_mat_entry(I->upperBound.repr, j, 0))) {
            ANCORA_ERROR(ANCORA_ERROR_NOT_IMPLEMENTED,
                          "Unbounded interval bounds (component %ld) are not yet supported.",
                          (long)j);
        }
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
    for (slong j = 0; j < m; j++) {
        if (isinf(I->lowerBound.repr[j]) || isnan(I->lowerBound.repr[j]) ||
            isinf(I->upperBound.repr[j]) || isnan(I->upperBound.repr[j])) {
            ANCORA_ERROR(ANCORA_ERROR_NOT_IMPLEMENTED,
                          "Unbounded interval bounds (component %ld) are not yet supported.",
                          (long)j);
        }
    }
#endif

    ancora_vec d, radius, e, r;
    ancora_mat absA;
    ANCORA_TRY(ancora_vec_init(&d, m));
    ANCORA_TRY(ancora_vec_init(&radius, m));
    ANCORA_TRY(ancora_vec_init(&e, n));
    ANCORA_TRY(ancora_vec_init(&r, n));
    ANCORA_TRY(ancora_mat_init(&absA, n, m));

    // d = (lowerBound + upperBound) / 2 ; radius = (upperBound - lowerBound) / 2
    ANCORA_TRY(ancora_vec_add(&d, &I->lowerBound, &I->upperBound));
    // absA = |A|, computed once, needed for the radius mapping below.
#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_t half;
    arb_init(half);
    arb_set_d(half, 0.5);
    ANCORA_TRY(ancora_vec_scalarMul(&d, &d, half));
    ANCORA_TRY(ancora_vec_sub(&radius, &I->upperBound, &I->lowerBound));
    ANCORA_TRY(ancora_vec_scalarMul(&radius, &radius, half));
    arb_clear(half);
    for (slong i = 0; i < n; i++) {
        for (slong j = 0; j < m; j++) {
            arb_abs(arb_mat_entry(absA.repr, i, j), arb_mat_entry(A->repr, i, j));
        }
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
    ANCORA_TRY(ancora_vec_scalarMul(&d, &d, 0.5));
    ANCORA_TRY(ancora_vec_sub(&radius, &I->upperBound, &I->lowerBound));
    ANCORA_TRY(ancora_vec_scalarMul(&radius, &radius, 0.5));
    for (slong i = 0; i < n * m; i++) {
        absA.repr[i] = fabs(A->repr[i]);
    }
#endif

    // e = A*d + c (mat-vec, O(n*m))
    ANCORA_TRY(ancora_mat_mul(&e, A, &d));
    ANCORA_TRY(ancora_vec_add(&e, &e, c));
    // r = |A|*radius (mat-vec, O(n*m)) -- this replaces the old per-row
    // extraction of G plus ancora_vec_1norm calls entirely.
    ANCORA_TRY(ancora_mat_mul(&r, &absA, &radius));

    ANCORA_TRY(ancora_vec_sub(&res->lowerBound, &e, &r));
    ANCORA_TRY(ancora_vec_add(&res->upperBound, &e, &r));

    ANCORA_TRY(ancora_vec_free(&d));
    ANCORA_TRY(ancora_vec_free(&radius));
    ANCORA_TRY(ancora_vec_free(&e));
    ANCORA_TRY(ancora_vec_free(&r));
    ANCORA_TRY(ancora_mat_free(&absA));

    return ANCORA_OK;
}


ancora_status ancora_interval_batched_affine(
    ancora_interval **res_batch,
    const ancora_mat *A,
    const ancora_vec *c,
    const ancora_interval **I_batch,
    slong B)
/* Computes res_batch[b] = A*I_batch[b] + c for every b in [0, B). A (n x m)
 * and c (length n) are SHARED across the whole batch; only the intervals
 * differ. All I_batch[b] must have dimension m; every res_batch[b] must
 * have dimension n.
 *
 * This does NOT call ancora_interval_affine B times, which would build a
 * dense (m x m) diagonal matrix B_diag = diag(radius_b) and compute
 * G = A*B_diag via a full (n x m)*(m x m) matmul PER interval - O(n*m^2)
 * per pair, hence O(n*m^2*B) total, even though only O(n*m) of that
 * matmul's output is ever useful (row i of G is just row i of A, scaled
 * entrywise by radius_b; diag(radius_b) contributes nothing beyond that
 * scaling). Instead, since A does not vary across the batch:
 *   - |A| (entrywise absolute value) is computed ONCE, O(n*m), rather
 *     than being recomputed (implicitly, via the wasteful matmul above)
 *     for every pair.
 *   - Every interval's center is written as column b of an (m x B) matrix
 *     D, and E = A*D (n x B) is computed with ONE matmul, giving every
 *     pair's A*center_b at once, rather than B separate (n x m)*(m x 1)
 *     mat-vec products.
 *   - Likewise every interval's radius becomes column b of an (m x B)
 *     matrix Rad, and R = |A|*Rad (n x B) is computed with ONE matmul,
 *     giving every pair's row-wise-1-norm-equivalent scaling at once.
 *   - res_batch[b]'s bounds are then column b of E, plus c, plus/minus
 *     column b of R.
 * Total cost is O(n*m*B) (two (n x m)*(m x B) matmuls plus the one-time
 * O(n*m) abs), asymptotically better than the O(n*m^2*B) of B separate
 * ancora_interval_affine calls, and consolidates the batch's real work
 * into exactly two matrix multiplications rather than B small ones -
 * letting a single call into ancora_mat_mul (and, if GPU-backed, a single
 * kernel launch) do the work instead of B of each.
 *
 * INPUT:
 *      res_batch       : Array of B pointers, each already initialized
 *                        with dimension n; res_batch[b] receives
 *                        A*I_batch[b] + c
 *      A               : Linear map (n x m), shared across the batch
 *      c               : Translation vector (length n), shared across
 *                        the batch
 *      I_batch         : Array of B pointers to initialized
 *                        ancora_interval instances, all of dimension m
 *      B               : Number of intervals in the batch (>= 0)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*m*B*ANCORA_DEFAULT_PREC): two (n x m)*(m x B) matmuls plus one
 *      O(n*m) entrywise abs of A, versus O(n*m^2*B*ANCORA_DEFAULT_PREC)
 *      for B separate ancora_interval_affine calls (see above).
 *
 * Created:       2026-09-26
 * Last modified: 2026-09-26
 * Author(s):     Adrian Kulmburg
 */
{
    if (res_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res_batch is NULL; it should point to a valid array of ancora_interval pointers.");
    }
    if (A == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer A is NULL; it should point to a valid ancora_mat instance.");
    }
    if (c == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer c is NULL; it should point to a valid ancora_vec instance.");
    }
    if (I_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I_batch is NULL; it should point to a valid array of ancora_interval pointers.");
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

    for (slong b = 0; b < B; b++) {
        if (I_batch[b] == NULL) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I_batch[%ld] is NULL; it should point to a valid ancora_interval instance.", (long)b);
        }
        if (res_batch[b] == NULL) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res_batch[%ld] is NULL; it should point to a valid ancora_interval instance.", (long)b);
        }
        slong Ib_dim, resB_dim;
        ANCORA_TRY(ancora_interval_dimension(I_batch[b], &Ib_dim));
        ANCORA_TRY(ancora_interval_dimension(res_batch[b], &resB_dim));
        if (Ib_dim != m) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "Matrix A has %ld columns, I_batch[%ld] has dimension %ld; they need to be the same.",
                          (long)m, (long)b, (long)Ib_dim);
        }
        if (resB_dim != n) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "res_batch[%ld] should have dimension %ld, but has dimension %ld.",
                          (long)b, (long)n, (long)resB_dim);
        }
        if (res_batch[b] == I_batch[b]) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "res_batch[%ld] must not alias I_batch[%ld] for ancora_interval_batched_affine.", (long)b, (long)b);
        }

        /* Unboundedness check, same as ancora_interval_affine, per interval. */
#if ANCORA_MODE == ANCORA_MODE_SAFE
        for (slong j = 0; j < m; j++) {
            if (!arb_is_finite(arb_mat_entry(I_batch[b]->lowerBound.repr, j, 0)) ||
                !arb_is_finite(arb_mat_entry(I_batch[b]->upperBound.repr, j, 0))) {
                ANCORA_ERROR(ANCORA_ERROR_NOT_IMPLEMENTED,
                              "I_batch[%ld]: unbounded interval bounds (component %ld) are not yet supported.",
                              (long)b, (long)j);
            }
        }
#elif ANCORA_MODE == ANCORA_MODE_FAST
        for (slong j = 0; j < m; j++) {
            if (isinf(I_batch[b]->lowerBound.repr[j]) || isnan(I_batch[b]->lowerBound.repr[j]) ||
                isinf(I_batch[b]->upperBound.repr[j]) || isnan(I_batch[b]->upperBound.repr[j])) {
                ANCORA_ERROR(ANCORA_ERROR_NOT_IMPLEMENTED,
                              "I_batch[%ld]: unbounded interval bounds (component %ld) are not yet supported.",
                              (long)b, (long)j);
            }
        }
#endif
    }

    ancora_mat absA, D, Rad, E, R;
    ANCORA_TRY(ancora_mat_init(&absA, n, m));
    ANCORA_TRY(ancora_mat_init(&D, m, B));
    ANCORA_TRY(ancora_mat_init(&Rad, m, B));
    ANCORA_TRY(ancora_mat_init(&E, n, B));
    ANCORA_TRY(ancora_mat_init(&R, n, B));

    // absA = |A|, computed ONCE for the whole batch.
#if ANCORA_MODE == ANCORA_MODE_SAFE
    for (slong i = 0; i < n; i++) {
        for (slong j = 0; j < m; j++) {
            arb_abs(arb_mat_entry(absA.repr, i, j), arb_mat_entry(A->repr, i, j));
        }
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
    for (slong i = 0; i < n * m; i++) {
        absA.repr[i] = fabs(A->repr[i]);
    }
#endif

    // Fill D's and Rad's columns: center_b = (lb+ub)/2, radius_b = (ub-lb)/2.
    for (slong b = 0; b < B; b++) {
        for (slong j = 0; j < m; j++) {
#if ANCORA_MODE == ANCORA_MODE_SAFE
            arb_ptr lb = arb_mat_entry(I_batch[b]->lowerBound.repr, j, 0);
            arb_ptr ub = arb_mat_entry(I_batch[b]->upperBound.repr, j, 0);
            arb_ptr Dje = arb_mat_entry(D.repr, j, b);
            arb_ptr Rje = arb_mat_entry(Rad.repr, j, b);
            arb_add(Dje, lb, ub, ANCORA_DEFAULT_PREC);
            arb_mul_2exp_si(Dje, Dje, -1);
            arb_sub(Rje, ub, lb, ANCORA_DEFAULT_PREC);
            arb_mul_2exp_si(Rje, Rje, -1);
#elif ANCORA_MODE == ANCORA_MODE_FAST
            double lb = I_batch[b]->lowerBound.repr[j];
            double ub = I_batch[b]->upperBound.repr[j];
            D.repr[j * B + b] = (lb + ub) * 0.5;
            Rad.repr[j * B + b] = (ub - lb) * 0.5;
#endif
        }
    }

    // E = A*D (all centers mapped at once), R = |A|*Rad (all radii mapped
    // at once) - exactly two matmuls for the whole batch.
    ANCORA_TRY(ancora_mat_mul(&E, A, &D));
    ANCORA_TRY(ancora_mat_mul(&R, &absA, &Rad));

    // res_batch[b]->lowerBound = E[:,b] + c - R[:,b] ; upperBound = E[:,b] + c + R[:,b]
    for (slong b = 0; b < B; b++) {
        for (slong i = 0; i < n; i++) {
#if ANCORA_MODE == ANCORA_MODE_SAFE
            arb_ptr Eib = arb_mat_entry(E.repr, i, b);
            arb_ptr Rib = arb_mat_entry(R.repr, i, b);
            arb_ptr ci = arb_mat_entry(c->repr, i, 0);
            arb_ptr lo = arb_mat_entry(res_batch[b]->lowerBound.repr, i, 0);
            arb_ptr hi = arb_mat_entry(res_batch[b]->upperBound.repr, i, 0);
            arb_add(lo, Eib, ci, ANCORA_DEFAULT_PREC);
            arb_sub(lo, lo, Rib, ANCORA_DEFAULT_PREC);
            arb_add(hi, Eib, ci, ANCORA_DEFAULT_PREC);
            arb_add(hi, hi, Rib, ANCORA_DEFAULT_PREC);
#elif ANCORA_MODE == ANCORA_MODE_FAST
            double Eib = E.repr[i * B + b];
            double Rib = R.repr[i * B + b];
            double ci = c->repr[i];
            res_batch[b]->lowerBound.repr[i] = Eib + ci - Rib;
            res_batch[b]->upperBound.repr[i] = Eib + ci + Rib;
#endif
        }
    }

    ANCORA_TRY(ancora_mat_free(&absA));
    ANCORA_TRY(ancora_mat_free(&D));
    ANCORA_TRY(ancora_mat_free(&Rad));
    ANCORA_TRY(ancora_mat_free(&E));
    ANCORA_TRY(ancora_mat_free(&R));

    return ANCORA_OK;
}

ancora_status ancora_interval_matMul(ancora_interval *res,
                                     const ancora_mat *A,
                                     const ancora_interval *I)
/* Matrix map of an interval, without translation: res = A*I = { A*x | x in I }.
 * A is (n x m), I has dimension m, res has dimension n. Same computation
 * as ancora_interval_affine with c omitted entirely (not even a zero
 * vector is allocated).
 *
 * INPUT:
 *      res             : Result interval, already initialized as dimension n
 *      A               : Linear map (n x m)
 *      I               : Interval to map (dimension m)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*m*ANCORA_DEFAULT_PREC): one entrywise abs of A, two (n x m)*
 *      (m x 1) mat-vec products (see ancora_interval_affine).
 *
 * Created:       2026-09-27
 * Last modified: 2026-09-27
 * Author(s):     Adrian Kulmburg
 */
{
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_interval instance.");
    }
    if (A == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer A is NULL; it should point to a valid ancora_mat instance.");
    }
    if (I == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I is NULL; it should point to a valid ancora_interval instance.");
    }

    slong n = A->nrows;
    slong m = A->ncols;

    slong I_dimension;
    ANCORA_TRY(ancora_interval_dimension(I, &I_dimension));
    slong res_dimension;
    ANCORA_TRY(ancora_interval_dimension(res, &res_dimension));

    if (I_dimension != m) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Matrix A has %ld columns, interval I has dimension %ld; they need to be the same.",
                      (long)m, (long)I_dimension);
    }
    if (res_dimension != n) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Result should have dimension %ld, but res has dimension %ld.",
                      (long)n, (long)res_dimension);
    }
    if (res == I) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "res must not alias I for ancora_interval_matMul.");
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    for (slong j = 0; j < m; j++) {
        if (!arb_is_finite(arb_mat_entry(I->lowerBound.repr, j, 0)) ||
            !arb_is_finite(arb_mat_entry(I->upperBound.repr, j, 0))) {
            ANCORA_ERROR(ANCORA_ERROR_NOT_IMPLEMENTED,
                          "Unbounded interval bounds (component %ld) are not yet supported.",
                          (long)j);
        }
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
    for (slong j = 0; j < m; j++) {
        if (isinf(I->lowerBound.repr[j]) || isnan(I->lowerBound.repr[j]) ||
            isinf(I->upperBound.repr[j]) || isnan(I->upperBound.repr[j])) {
            ANCORA_ERROR(ANCORA_ERROR_NOT_IMPLEMENTED,
                          "Unbounded interval bounds (component %ld) are not yet supported.",
                          (long)j);
        }
    }
#endif

    ancora_vec d, radius, e, r;
    ancora_mat absA;
    ANCORA_TRY(ancora_vec_init(&d, m));
    ANCORA_TRY(ancora_vec_init(&radius, m));
    ANCORA_TRY(ancora_vec_init(&e, n));
    ANCORA_TRY(ancora_vec_init(&r, n));
    ANCORA_TRY(ancora_mat_init(&absA, n, m));

    ANCORA_TRY(ancora_vec_add(&d, &I->lowerBound, &I->upperBound));
#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_t half;
    arb_init(half);
    arb_set_d(half, 0.5);
    ANCORA_TRY(ancora_vec_scalarMul(&d, &d, half));
    ANCORA_TRY(ancora_vec_sub(&radius, &I->upperBound, &I->lowerBound));
    ANCORA_TRY(ancora_vec_scalarMul(&radius, &radius, half));
    arb_clear(half);
    for (slong i = 0; i < n; i++) {
        for (slong j = 0; j < m; j++) {
            arb_abs(arb_mat_entry(absA.repr, i, j), arb_mat_entry(A->repr, i, j));
        }
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
    ANCORA_TRY(ancora_vec_scalarMul(&d, &d, 0.5));
    ANCORA_TRY(ancora_vec_sub(&radius, &I->upperBound, &I->lowerBound));
    ANCORA_TRY(ancora_vec_scalarMul(&radius, &radius, 0.5));
    for (slong i = 0; i < n * m; i++) {
        absA.repr[i] = fabs(A->repr[i]);
    }
#endif

    // e = A*d (no c to add here)
    ANCORA_TRY(ancora_mat_mul(&e, A, &d));
    ANCORA_TRY(ancora_mat_mul(&r, &absA, &radius));

    ANCORA_TRY(ancora_vec_sub(&res->lowerBound, &e, &r));
    ANCORA_TRY(ancora_vec_add(&res->upperBound, &e, &r));

    ANCORA_TRY(ancora_vec_free(&d));
    ANCORA_TRY(ancora_vec_free(&radius));
    ANCORA_TRY(ancora_vec_free(&e));
    ANCORA_TRY(ancora_vec_free(&r));
    ANCORA_TRY(ancora_mat_free(&absA));

    return ANCORA_OK;
}

ancora_status ancora_interval_batched_matMul(
    ancora_interval **res_batch,
    const ancora_mat *A,
    const ancora_interval **I_batch,
    slong B)
/* Computes res_batch[b] = A*I_batch[b] for every b in [0, B) (no
 * translation). A (n x m) is SHARED across the batch; only the intervals
 * differ. Same as ancora_interval_batched_affine with c omitted.
 *
 * INPUT:
 *      res_batch       : Array of B pointers, each already initialized
 *                        with dimension n; res_batch[b] receives
 *                        A*I_batch[b]
 *      A               : Linear map (n x m), shared across the batch
 *      I_batch         : Array of B pointers to initialized
 *                        ancora_interval instances, all of dimension m
 *      B               : Number of intervals in the batch (>= 0)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*m*B*ANCORA_DEFAULT_PREC): two (n x m)*(m x B) matmuls plus one
 *      O(n*m) entrywise abs of A (see ancora_interval_batched_affine).
 *
 * Created:       2026-09-27
 * Last modified: 2026-09-27
 * Author(s):     Adrian Kulmburg
 */
{
    if (res_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res_batch is NULL; it should point to a valid array of ancora_interval pointers.");
    }
    if (A == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer A is NULL; it should point to a valid ancora_mat instance.");
    }
    if (I_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I_batch is NULL; it should point to a valid array of ancora_interval pointers.");
    }
    if (B < 0) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "B is negative (%ld); it should be nonnegative.", (long)B);
    }

    slong n = A->nrows;
    slong m = A->ncols;

    if (B == 0) {
        return ANCORA_OK; /* vacuously nothing to do */
    }

    for (slong b = 0; b < B; b++) {
        if (I_batch[b] == NULL) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer I_batch[%ld] is NULL; it should point to a valid ancora_interval instance.", (long)b);
        }
        if (res_batch[b] == NULL) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res_batch[%ld] is NULL; it should point to a valid ancora_interval instance.", (long)b);
        }
        slong Ib_dim, resB_dim;
        ANCORA_TRY(ancora_interval_dimension(I_batch[b], &Ib_dim));
        ANCORA_TRY(ancora_interval_dimension(res_batch[b], &resB_dim));
        if (Ib_dim != m) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "Matrix A has %ld columns, I_batch[%ld] has dimension %ld; they need to be the same.",
                          (long)m, (long)b, (long)Ib_dim);
        }
        if (resB_dim != n) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "res_batch[%ld] should have dimension %ld, but has dimension %ld.",
                          (long)b, (long)n, (long)resB_dim);
        }
        if (res_batch[b] == I_batch[b]) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "res_batch[%ld] must not alias I_batch[%ld] for ancora_interval_batched_matMul.", (long)b, (long)b);
        }

#if ANCORA_MODE == ANCORA_MODE_SAFE
        for (slong j = 0; j < m; j++) {
            if (!arb_is_finite(arb_mat_entry(I_batch[b]->lowerBound.repr, j, 0)) ||
                !arb_is_finite(arb_mat_entry(I_batch[b]->upperBound.repr, j, 0))) {
                ANCORA_ERROR(ANCORA_ERROR_NOT_IMPLEMENTED,
                              "I_batch[%ld]: unbounded interval bounds (component %ld) are not yet supported.",
                              (long)b, (long)j);
            }
        }
#elif ANCORA_MODE == ANCORA_MODE_FAST
        for (slong j = 0; j < m; j++) {
            if (isinf(I_batch[b]->lowerBound.repr[j]) || isnan(I_batch[b]->lowerBound.repr[j]) ||
                isinf(I_batch[b]->upperBound.repr[j]) || isnan(I_batch[b]->upperBound.repr[j])) {
                ANCORA_ERROR(ANCORA_ERROR_NOT_IMPLEMENTED,
                              "I_batch[%ld]: unbounded interval bounds (component %ld) are not yet supported.",
                              (long)b, (long)j);
            }
        }
#endif
    }

    ancora_mat absA, D, Rad, E, R;
    ANCORA_TRY(ancora_mat_init(&absA, n, m));
    ANCORA_TRY(ancora_mat_init(&D, m, B));
    ANCORA_TRY(ancora_mat_init(&Rad, m, B));
    ANCORA_TRY(ancora_mat_init(&E, n, B));
    ANCORA_TRY(ancora_mat_init(&R, n, B));

#if ANCORA_MODE == ANCORA_MODE_SAFE
    for (slong i = 0; i < n; i++) {
        for (slong j = 0; j < m; j++) {
            arb_abs(arb_mat_entry(absA.repr, i, j), arb_mat_entry(A->repr, i, j));
        }
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
    for (slong i = 0; i < n * m; i++) {
        absA.repr[i] = fabs(A->repr[i]);
    }
#endif

    for (slong b = 0; b < B; b++) {
        for (slong j = 0; j < m; j++) {
#if ANCORA_MODE == ANCORA_MODE_SAFE
            arb_ptr lb = arb_mat_entry(I_batch[b]->lowerBound.repr, j, 0);
            arb_ptr ub = arb_mat_entry(I_batch[b]->upperBound.repr, j, 0);
            arb_ptr Dje = arb_mat_entry(D.repr, j, b);
            arb_ptr Rje = arb_mat_entry(Rad.repr, j, b);
            arb_add(Dje, lb, ub, ANCORA_DEFAULT_PREC);
            arb_mul_2exp_si(Dje, Dje, -1);
            arb_sub(Rje, ub, lb, ANCORA_DEFAULT_PREC);
            arb_mul_2exp_si(Rje, Rje, -1);
#elif ANCORA_MODE == ANCORA_MODE_FAST
            double lb = I_batch[b]->lowerBound.repr[j];
            double ub = I_batch[b]->upperBound.repr[j];
            D.repr[j * B + b] = (lb + ub) * 0.5;
            Rad.repr[j * B + b] = (ub - lb) * 0.5;
#endif
        }
    }

    // No c to add: res_batch[b]->lowerBound = E[:,b] - R[:,b], upperBound = E[:,b] + R[:,b]
    ANCORA_TRY(ancora_mat_mul(&E, A, &D));
    ANCORA_TRY(ancora_mat_mul(&R, &absA, &Rad));

    for (slong b = 0; b < B; b++) {
        for (slong i = 0; i < n; i++) {
#if ANCORA_MODE == ANCORA_MODE_SAFE
            arb_ptr Eib = arb_mat_entry(E.repr, i, b);
            arb_ptr Rib = arb_mat_entry(R.repr, i, b);
            arb_ptr lo = arb_mat_entry(res_batch[b]->lowerBound.repr, i, 0);
            arb_ptr hi = arb_mat_entry(res_batch[b]->upperBound.repr, i, 0);
            arb_sub(lo, Eib, Rib, ANCORA_DEFAULT_PREC);
            arb_add(hi, Eib, Rib, ANCORA_DEFAULT_PREC);
#elif ANCORA_MODE == ANCORA_MODE_FAST
            double Eib = E.repr[i * B + b];
            double Rib = R.repr[i * B + b];
            res_batch[b]->lowerBound.repr[i] = Eib - Rib;
            res_batch[b]->upperBound.repr[i] = Eib + Rib;
#endif
        }
    }

    ANCORA_TRY(ancora_mat_free(&absA));
    ANCORA_TRY(ancora_mat_free(&D));
    ANCORA_TRY(ancora_mat_free(&Rad));
    ANCORA_TRY(ancora_mat_free(&E));
    ANCORA_TRY(ancora_mat_free(&R));

    return ANCORA_OK;
}
