/*
 * ancora_zonotope_containment.h
 *
 * Description
 * -----------
 * Containment checks for ancora_zonotope: whether a point (or a set of points)
 * is contained in the zonotope.
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

#ifndef ANCORA_ZONOTOPE_CONTAINMENT_H
#define ANCORA_ZONOTOPE_CONTAINMENT_H

#include <stdlib.h>
#include <stdbool.h>

#include "ancora/sets/zonotope/ancora_zonotope.h"
#include "ancora/logic/ancora_logic.h"

#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_zonotope_containsPoint(const ancora_zonotope *Z,
                                       const ancora_vec *p,
                                       ancora_truth *contained);
/* Checks whether the point p is contained in the zonotope Z.
 * This is decided by solving a linear feasibility problem with HiGHS in
 * ANCORA_MODE_FAST mode; in ANCORA_MODE_SAFE mode it returns
 * ANCORA_ERROR_NOT_IMPLEMENTED at the moment.
 *
 * INPUT:
 *      Z               : Pointer to an initialized ancora_zonotope instance
 *      p               : Point to check
 *      contained       : Output; set to true iff p is contained in Z
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      TODO: Centrally, for optimization once it has been determined accurately
 *
 * Created:       2026-09-26
 * Last modified: 2026-09-26
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_zonotope_containsPoints(const ancora_zonotope *Z,
                                              const ancora_mat *P,
                                              ancora_truth *contained);
/* Checks whether every column of P is contained in the zonotope Z. P is a
 * matrix whose j-th column is the j-th point; contained is true iff all k
 * points are contained in Z.
 *
 * INPUT:
 *      Z               : Pointer to an initialized ancora_zonotope instance
 *      P               : Matrix of points, one point per
 *                        column
 *      contained       : Output; set to true iff every column of P is
 *                        contained in Z
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      TODO: Centrally, for optimization once it has been determined accurately
 *
 * Created:       2026-09-26
 * Last modified: 2026-09-26
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_zonotope_batched_containsPoints(
    const ancora_zonotope **Z_batch,
    const ancora_mat **P_batch,
    slong B,
    ancora_truth *contained);
/* Checks whether, for every b in [0, B), every column of P_batch[b] is
 * contained in Z_batch[b]. contained is set to true iff ALL points in ALL
 * pairs are contained in their respective zonotope. Every Z_batch[b] and
 * P_batch[b] must share the same dimension n; each Z_batch[b] may have its
 * OWN generator count p_b, and each P_batch[b] its OWN point count k_b.
 *
 * INPUT:
 *      Z_batch         : Array of B pointers to initialized
 *                        ancora_zonotope instances, all of dimension n
 *                        (generator counts p_b may differ)
 *      P_batch         : Array of B pointers to point matrices;
 *                        P_batch[b] is (n x k_b), one point per column
 *                        (k_b may differ)
 *      B               : Number of (zonotope, points) pairs (>= 0)
 *      contained       : Output; set to true iff every point in every
 *                        pair is contained in its zonotope
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      Worst case (nothing short-circuits), the sum over b of the cost of
 *      solving k_b LPs against a p_b-variable, n-constraint problem -
 *      identical total LP-solving work to B separate
 *      ancora_zonotope_containsPoints calls. The saving is avoiding B
 *      allocations of the n-sized arrays (allocated once instead) and,
 *      when generator counts repeat/are similar across the batch,
 *      avoiding repeated malloc/free of the p_b-sized arrays via reuse
 *      with growth - not a reduction in LP-solving work itself.
 *
 * Created:       2026-09-27
 * Last modified: 2026-09-27
 * Author(s):     Adrian Kulmburg
 */

#ifdef __cplusplus
}
#endif

#endif /* ANCORA_ZONOTOPE_CONTAINMENT_H */
