/*
 * ancora_mat_concatenation.h
 *
 * Description
 * -----------
 * Concatenation operations for matrices: horizontal, vertical, and
 * block-diagonal.
 *
 * File Information
 * ----------------
 * Created:       2026-09-22
 * Last modified: 2026-09-22
 * Authors:       Adrian Kulmburg
 *
 * License
 * -------
 * Copyright (c) 2026 Adrian Kulmburg <adrian.kulmburg@kit.edu>
 * SPDX-License-Identifier: MIT
 */

#ifndef ANCORA_MAT_CONCATENATION_H
#define ANCORA_MAT_CONCATENATION_H

#include "ancora/ancora_config.h"
#include "ancora/ancora_error.h"

#include "ancora/linalg/matrices/ancora_mat.h"

#if ANCORA_MODE == ANCORA_MODE_SAFE
#include <flint/arb.h>
#include <flint/arb_mat.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_mat_hcat(ancora_mat *res, const ancora_mat *a, const ancora_mat *b);
/* Horizontal concatenation: res = [a b].
 * INPUT:
 *      res             : Result matrix, already initialized as
 *                        (n x (k1+k2)), where a is (n x k1), b is (n x k2)
 *      a               : Left block
 *      b               : Right block
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*(k1+k2))
 *      In ANCORA_MODE_SAFE mode, O(ANCORA_DEFAULT_PREC) worst case per
 *      entry copied, so O(n*(k1+k2)*ANCORA_DEFAULT_PREC) worst case overall.
 *
 * Created:       2026-09-22
 * Last modified: 2026-09-22
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_mat_vcat(ancora_mat *res, const ancora_mat *a, const ancora_mat *b);
/* Vertical concatenation: res = [a; b].
 * INPUT:
 *      res             : Result matrix, already initialized as
 *                        ((n1+n2) x m), where a is (n1 x m), b is (n2 x m)
 *      a               : Top block
 *      b               : Bottom block
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O((n1+n2)*m)
 *      In ANCORA_MODE_SAFE mode, O(ANCORA_DEFAULT_PREC) worst case per
 *      entry copied, so O((n1+n2)*m*ANCORA_DEFAULT_PREC) worst case overall.
 *
 * Created:       2026-09-22
 * Last modified: 2026-09-22
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_mat_diagcat(ancora_mat *res, const ancora_mat *a, const ancora_mat *b);
/* Block-diagonal concatenation: res = [a 0; 0 b].
 * INPUT:
 *      res             : Result matrix, already initialized as
 *                        ((n1+n2) x (m1+m2)), where a is (n1 x m1), b is
 *                        (n2 x m2)
 *      a               : Top-left block
 *      b               : Bottom-right block
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O((n1+n2)*(m1+m2))
 *      In ANCORA_MODE_SAFE mode, O(ANCORA_DEFAULT_PREC) worst case per
 *      entry touched, so O((n1+n2)*(m1+m2)*ANCORA_DEFAULT_PREC) worst case
 *      overall.
 *
 * Created:       2026-09-22
 * Last modified: 2026-09-22
 * Author(s):     Adrian Kulmburg
 */

#ifdef __cplusplus
}
#endif

#endif
