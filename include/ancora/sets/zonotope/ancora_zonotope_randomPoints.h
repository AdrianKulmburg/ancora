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

#ifndef ANCORA_ZONOTOPE_RANDOMPOINTS_H
#define ANCORA_ZONOTOPE_RANDOMPOINTS_H

#include "ancora/ancora_config.h"
#include "ancora/ancora_error.h"

#include "ancora/ancora_random.h"

#include "ancora/linalg/matrices/ancora_mat.h"
#include "ancora/sets/zonotope/ancora_zonotope.h"

#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_zonotope_randomPoints_standard(ancora_mat *P,
                                                    const ancora_zonotope *Z,
                                                    slong N);
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

ancora_status ancora_zonotope_batched_randomPoints_standard(
    ancora_mat **P_batch,
    const ancora_zonotope **Z_batch,
    slong B,
    slong N);
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

#ifdef __cplusplus
}
#endif

#endif
