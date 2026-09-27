/*
 * ancora_zonotope_duality.h
 *
 * Description
 * -----------
 * Duality operations for zonotopes.
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

#ifndef ANCORA_ZONOTOPE_DUALITY_H
#define ANCORA_ZONOTOPE_DUALITY_H

#include "ancora/sets/zonotope/ancora_zonotope.h"

#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_zonotope_supportFunction(
#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_t res,
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double *res,
#endif
    const ancora_zonotope *Z,
    const ancora_vec *d);
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

ancora_status ancora_zonotope_batched_supportFunction(
#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_ptr res,
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double *res,
#endif
    const ancora_zonotope **Z_batch,
    const ancora_vec **d_batch,
    slong B);
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

#ifdef __cplusplus
}
#endif

#endif
