/*
 * ancora_zonotope_arithmetic.h
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

#ifndef ANCORA_ZONOTOPE_ARITHMETIC_H
#define ANCORA_ZONOTOPE_ARITHMETIC_H

#include "ancora/sets/zonotope/ancora_zonotope.h"

#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_zonotope_minkowskiSum(ancora_zonotope *res,
                                           const ancora_zonotope *Z1,
                                           const ancora_zonotope *Z2);
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

ancora_status ancora_zonotope_batched_minkowskiSum(
    ancora_zonotope **res_batch,
    const ancora_zonotope **Z1_batch,
    const ancora_zonotope **Z2_batch,
    slong B);
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

ancora_status ancora_zonotope_affine(ancora_zonotope *res,
                                     const ancora_mat *A,
                                     const ancora_vec *c,
                                     const ancora_zonotope *Z);
/* Affine map of a zonotope: res = A*Z + c = { A*x + c | x in Z }.
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
 *      O(n*m*p*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_zonotope_batched_affine(
    ancora_zonotope **res_batch,
    const ancora_mat *A,
    const ancora_vec *c,
    const ancora_zonotope **Z_batch,
    slong B);
/* Computes res_batch[b] = A*Z_batch[b] + c for every b in [0, B). A (n x m)
 * and c (length n) are SHARED across the whole batch; only the zonotopes
 * differ. All Z_batch[b] must have dimension m (but may each have a
 * DIFFERENT number of generators p_b); res_batch[b] must have dimension n
 * and p_b generators (matching Z_batch[b]'s own generator count, same
 * requirement as the single-instance ancora_zonotope_affine).
 *
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
 *      O(n*m*P*ANCORA_DEFAULT_PREC) where P = sum_b p_b - identical total
 *      arithmetic to B separate ancora_zonotope_affine calls (this is a
 *      genuine matmul, so there is no less total work possible); the
 *      packing/slicing passes add O(m*P + m*B + n*P + n*B), the same
 *      order as the work itself. The improvement is consolidating 2*B
 *      matmuls (and, with ANCORA_USE_GPU, 2*B GPU kernel launches) into
 *      exactly 2.
 *
 * Created:       2026-09-27
 * Last modified: 2026-09-27
 * Author(s):     Adrian Kulmburg
 */

#ifdef __cplusplus
}
#endif

#endif
