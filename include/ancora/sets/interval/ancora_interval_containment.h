/*
 * ancora_interval_containment.h
 *
 * Description
 * -----------
 * Containment checks for ancora_interval.
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

#ifndef ANCORA_INTERVAL_CONTAINMENT_H
#define ANCORA_INTERVAL_CONTAINMENT_H

#include "ancora/logic/ancora_logic.h"
#include "ancora/sets/interval/ancora_interval.h"

#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_interval_containsPoint(const ancora_interval *I,
                                       const ancora_vec *p,
                                       ancora_truth *result);
/* Checks whether the point p is contained in the interval I.
 *
 * INPUT:
 *      I               : Pointer to an initialized ancora_interval instance
 *      p               : Point to test, a column vector of length I->dimension
 *      result          : Output; set to true iff p is contained in I
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(dimension*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_interval_containsPoints(const ancora_interval *I,
                                              const ancora_mat *P,
                                              ancora_truth *result);
/* Checks whether every column of P is contained in the interval I. P is an
 * (n x k) matrix (where n is the dimension of I) whose j-th column is the
 * j-th point; the result is true iff all k points are contained in I.
 *
 * INPUT:
 *      I               : Pointer to an initialized ancora_interval instance
 *      P               : Matrix of points, (I->dimension x k), one point per
 *                        column
 *      result          : Output; set to true iff every column of P is
 *                        contained in I
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(dimension*k*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_interval_batched_containsPoints(
    const ancora_interval **I_batch,
    const ancora_mat **P_batch,
    slong B,
    ancora_truth *contained);
/* Checks whether, for every b in [0, B), every column of P_batch[b] is
 * contained in I_batch[b]. contained is set to true iff ALL points in
 * ALL pairs are contained in their respective interval.
 *
 * Unlike ancora_interval_batched_randomPoints_uniform (stacking) and
 * ancora_interval_batched_supportFunction (fused reduction), pair b's
 * work here does not depend on pair b' != b at all, so - unlike those
 * two - the intervals in the batch are NOT required to share a common
 * dimension; each I_batch[b]/P_batch[b] pair may have its own dimension
 * n_b and its own point count k_b.
 *
 * This fuses everything into one triple loop (pair b, then point j, then
 * dimension i) directly against I_batch[b]->lowerBound/upperBound.repr
 * and P_batch[b]->repr, rather than calling
 * ancora_interval_containsPoints (or containsPoint) once per pair. That
 * avoids two real costs the per-pair functions pay redundantly across a
 * batch: (1) ancora_interval_containsPoint re-validates NULL/dimension on
 * EVERY point - calling it per point across the whole batch means
 * n_b*k_b such re-checks per pair, all but the first of which are
 * wasted, since a pair's dimension does not change between its own
 * points; here validation happens once per pair, not once per point; and
 * (2) ancora_interval_containsPoints allocates one temporary ancora_vec
 * per call - calling it B times means B such allocations, all avoided
 * here since points are read directly from P_batch[b]->repr with no
 * intermediate vector at all.
 *
 * Short-circuits across the ENTIRE batch, not just within one pair: the
 * moment any point in any pair is found not contained, this returns
 * immediately without checking the rest of that pair's points or any
 * later pair at all.
 *
 * INPUT:
 *      I_batch         : Array of B pointers to initialized
 *                        ancora_interval instances (dimensions may differ
 *                        across pairs)
 *      P_batch         : Array of B pointers to point matrices; P_batch[b]
 *                        is (I_batch[b]'s dimension x k_b), one point per
 *                        column, k_b may differ across pairs
 *      B               : Number of (interval, points) pairs (>= 0)
 *      contained       : Output; set to true iff every point in every
 *                        pair is contained in its interval
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(sum_b n_b*k_b*ANCORA_DEFAULT_PREC) worst case (all points
 *      contained, so nothing short-circuits) -- identical total work to B
 *      separate calls to ancora_interval_containsPoints; the savings are
 *      the eliminated per-point re-validation and per-pair temporary
 *      allocation described above, plus batch-wide short-circuiting,
 *      which can make this substantially faster than B separate calls in
 *      practice whenever an early pair/point fails containment.
 *
 * Created:       2026-09-26
 * Last modified: 2026-09-26
 * Author(s):     Adrian Kulmburg
 */

#ifdef __cplusplus
}
#endif

#endif /* ANCORA_INTERVAL_CONTAINMENT_H */
