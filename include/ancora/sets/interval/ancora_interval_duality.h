/*
 * ancora_interval_duality.h
 *
 * Description
 * -----------
 * Duality operations for intervals.
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

#ifndef ANCORA_INTERVAL_DUALITY_H
#define ANCORA_INTERVAL_DUALITY_H

#include "ancora/sets/interval/ancora_interval.h"



#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_interval_supportFunction(
#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_t res,
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double *res,
#endif
    const ancora_interval *I,
    const ancora_vec *d);
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

ancora_status ancora_interval_batched_supportFunction(
#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_ptr res,
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double *res,
#endif
    const ancora_interval **I_batch,
    const ancora_vec **d_batch,
    slong B);
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

#ifdef __cplusplus
}
#endif

#endif
