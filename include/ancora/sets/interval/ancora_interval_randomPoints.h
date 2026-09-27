/*
 * ancora_interval_randomPoints.h
 *
 * Description
 * -----------
 * Random point generation for intervals.
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

#ifndef ANCORA_INTERVAL_RANDOMPOINTS_H
#define ANCORA_INTERVAL_RANDOMPOINTS_H

#include <math.h>

#include "ancora/ancora_config.h"
#include "ancora/ancora_error.h"

#include "ancora/ancora_random.h"

#include "ancora/linalg/matrices/ancora_mat.h"
#include "ancora/sets/interval/ancora_interval.h"

#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_interval_randomPoints_uniform(ancora_mat *P,
                                                   const ancora_interval *I,
                                                   slong N);
/* Fills P with N points drawn independently and uniformly at random from the
 * interval I. Each column of P is one point: P is an (n x N) matrix, where n
 * is the dimension of I. For each dimension i, the i-th coordinate of every
 * point is sampled uniformly from [lowerBound[i], upperBound[i]] via
 * ancora_random_uniform (one draw per coordinate per point).
 *
 * The interval must be bounded: uniform sampling is ill-defined on an
 * unbounded box, so an interval with any +-inf bound is rejected with
 * ANCORA_ERROR_NOT_IMPLEMENTED. An empty
 * interval (lowerBound[i] > upperBound[i] for some i) is rejected with
 * ANCORA_ERROR_INVALID_ARG.
 *
 * INPUT:
 *      P               : Result matrix, already initialized as (n x N), where
 *                        n is the dimension of I; column j holds the j-th
 *                        point
 *      I               : Pointer to an initialized ancora_interval instance
 *      N               : Number of points to generate (>= 0)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*N*ANCORA_DEFAULT_PREC)
 *      where n is the dimension of the interval.
 *
 * Created:       2026-09-26
 * Last modified: 2026-09-26
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_interval_batched_randomPoints_uniform(
    ancora_mat **P_batch,
    const ancora_interval **I_batch,
    slong B,
    slong N);
/* Fills P_batch[0..B-1] with N points each, drawn independently and
 * uniformly at random from I_batch[0..B-1] respectively (all intervals
 * must share the same dimension n). P_batch[b] is (n x N); column j of
 * P_batch[b] is the j-th random point from I_batch[b].
 *
 * Implementation: rather than looping B separate times over
 * ancora_interval_randomPoints_uniform, the B intervals' bounds are
 * vertically stacked (via ancora_mat_vcat, since sampling is coordinate-
 * independent) into one interval of dimension n*B, sampled ONCE for N
 * points, and the resulting (n*B x N) matrix is sliced back apart by row
 * block. This does NOT reduce the total number of random draws (still
 * exactly n*B*N, the inherent cost of one draw per coordinate per point
 * per interval) - it reduces this to one validation pass and one call
 * into the core sampling loop instead of B of each, and fills one
 * contiguous block of memory rather than B smaller ones.
 *
 * INPUT:
 *      P_batch         : Array of B pointers, each already initialized
 *                        as (n x N); P_batch[b] receives interval b's
 *                        points
 *      I_batch         : Array of B pointers to initialized
 *                        ancora_interval instances, all of the same
 *                        dimension n
 *      B               : Number of intervals in the batch (>= 0)
 *      N               : Number of points to generate per interval (>= 0)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*B*N*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-26
 * Last modified: 2026-09-26
 * Author(s):     Adrian Kulmburg
 */

#ifdef __cplusplus
}
#endif

#endif
