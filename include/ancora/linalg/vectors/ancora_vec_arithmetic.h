/*
 * ancora_vec_arithmetic.c
 *
 * Description
 * -----------
 * Elementary arithmetic operations for vectors.
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

#ifndef ANCORA_VEC_ARITHMETIC_H
#define ANCORA_VEC_ARITHMETIC_H

#include "ancora/linalg/vectors/ancora_vec.h"
#include "ancora/linalg/matrices/ancora_mat.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ancora_vec_add(res, a, b) ancora_mat_add((res), (a), (b))
/* Adds two vectors together.
 * INPUT:
 *      res             : Resulting vector
 *      a               : First summand
 *      b               : Second summand
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*ANCORA_DEFAULT_PREC)
 *      where n is the dimension of a (or b).
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */

#define ancora_vec_sub(res, a, b) ancora_mat_sub((res), (a), (b))
/* Elementwise subtraction of two vectors (res = a - b).
 * INPUT:
 *      res             : Resulting vector
 *      a               : Minuend
 *      b               : Subtrahend
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*ANCORA_DEFAULT_PREC)
 *      where n is the dimension of a (or b).
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */

#define ancora_vec_scalarMul(res, a, scalar) ancora_mat_scalarMul((res), (a), (scalar))
/* Elementwise scalar multiplication (res = scalar * a).
 * INPUT:
 *      res             : Resulting vector
 *      a               : Vector to scale
 *      scalar          : Scalar factor (arb_t in SAFE mode, double in FAST
 *                        mode)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*ANCORA_DEFAULT_PREC)
 *      where n is the dimension of a.
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */

#define ancora_vec_neg(res, v) ancora_mat_neg(res, v)
/* Computes the negative of a vector (res = -v).
 * INPUT:
 *      res             : Result vector, already initialized
 *      v               : Vector to negate
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n)
 *      where n is the dimension of v.
 *      In ANCORA_MODE_SAFE mode, negation is exact (no rounding), so this
 *      does not scale with ANCORA_DEFAULT_PREC.
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */

#define ancora_vec_transpose(res, v) ancora_mat_transpose(res, v)
/* Computes the transpose of a vector (res = a^T, res is now a matrix).
 * INPUT:
 *      res             : Result matrix, already initialized as
 *                        (1 x v->nrows)
 *      v               : Vector to transpose (nrows x 1)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(nrows*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-26
 * Last modified: 2026-09-26
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_vec_dot(
#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_t res,
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double *res,
#endif
    const ancora_vec *a,
    const ancora_vec *b);
/* Dot product res = a . b.
 * INPUT:
 *      res             : Output scalar (arb_t in SAFE mode, double* in FAST)
 *      a, b            : Vectors (ancora_vec)
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

#ifdef __cplusplus
}
#endif

#endif
