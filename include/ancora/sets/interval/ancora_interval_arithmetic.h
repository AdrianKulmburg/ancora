/*
 * ancora_interval_arithmetic.h
 *
 * Description
 * -----------
 * Arithmetic operations for ancora_interval.
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

#ifndef ANCORA_INTERVAL_ARITHMETIC_H
#define ANCORA_INTERVAL_ARITHMETIC_H

#include <math.h>

#include "ancora/sets/interval/ancora_interval.h"

#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_interval_minkowskiSum(ancora_interval *res,
                                           const ancora_interval *I,
                                           const ancora_interval *J);
/* Minkowski sum of two intervals: res = I + J = { x + y | x in I, y in J }.
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
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_interval_batched_minkowskiSum(
    ancora_interval **res_batch,
    const ancora_interval **I_batch,
    const ancora_interval **J_batch,
    slong B);
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
 *      so there is no less work possible) -- the improvement is one
 *      validation pass instead of B, and, with ANCORA_USE_GPU, reuse of
 *      the existing elementwise-add kernel via an O(n*B) packing pass.
 *
 * Created:       2026-09-26
 * Last modified: 2026-09-26
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_interval_affine(ancora_interval *res,
                                     const ancora_mat *A,
                                     const ancora_vec *c,
                                     const ancora_interval *I);
/* Affine map of an interval: res = A*I + c = { A*x + c | x in I }.
 * A is (n x m), I has dimension m, c has length n, and res has dimension n.
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
 *      O(n*m*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_interval_batched_affine(
    ancora_interval **res_batch,
    const ancora_mat *A,
    const ancora_vec *c,
    const ancora_interval **I_batch,
    slong B);
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

#ifdef __cplusplus
}
#endif

#endif /* ANCORA_INTERVAL_ARITHMETIC_H */
