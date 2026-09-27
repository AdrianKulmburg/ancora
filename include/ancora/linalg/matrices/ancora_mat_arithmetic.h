/*
 * ancora_mat_arithmetic.h
 *
 * Description
 * -----------
 * Elementary arithmetic operations for matrices
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

#ifndef ANCORA_MAT_ARITHMETIC_H
#define ANCORA_MAT_ARITHMETIC_H

#include "ancora/ancora_config.h"
#include "ancora/ancora_error.h"

#include "ancora/linalg/matrices/ancora_mat.h"

#ifdef ANCORA_USE_GPU
#include "ancora/linalg/matrices/ancora_mat_arithmetic_gpu.hip.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_mat_add(ancora_mat *res,
                             const ancora_mat *a,
                             const ancora_mat *b);
/* Adds two matrices together.
 * INPUT:
 *      res             : Resulting matrix
 *      a               : First summand
 *      b               : Second summand
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*m*ANCORA_DEFAULT_PREC)
 *      where n is the number of rows of a (or b), and m the number of columns.
 *
 * Created:       2026-09-22
 * Last modified: 2026-09-22
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_mat_sub(ancora_mat *res,
                             const ancora_mat *a,
                             const ancora_mat *b);
/* Elementwise subtraction of two matrices (res = a - b).
 * INPUT:
 *      res             : Resulting matrix
 *      a               : Minuend
 *      b               : Subtrahend
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*m*ANCORA_DEFAULT_PREC)
 *      where n is the number of rows of a (or b), and m the number of columns.
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_mat_scalarMul(ancora_mat *res,
                                    const ancora_mat *a,
#if ANCORA_MODE == ANCORA_MODE_SAFE
                                    arb_t scalar
#elif ANCORA_MODE == ANCORA_MODE_FAST
                                    double scalar
#endif
                                   );
/* Elementwise scalar multiplication (res = scalar * a).
 * INPUT:
 *      res             : Resulting matrix
 *      a               : Matrix to scale
 *      scalar          : Scalar factor (arb_t in SAFE mode, double in FAST
 *                        mode)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*m*ANCORA_DEFAULT_PREC)
 *      where n is the number of rows of a and m the number of columns.
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_mat_mul(ancora_mat *res,
                             const ancora_mat *a,
                             const ancora_mat *b);
/* Multiplies two matrices together (res = a * b).
 * INPUT:
 *      res             : Resulting matrix
 *      a               : Left factor
 *      b               : Right factor
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*k*m*ANCORA_DEFAULT_PREC)
 *      where n is the number of rows of a, k its number of columns, m the
 *      number of columns of b.
 *
 * Created:       2026-09-22
 * Last modified: 2026-09-22
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_mat_transpose(ancora_mat *res, const ancora_mat *a);
/* Computes the transpose of a matrix (res = a^T).
 * INPUT:
 *      res             : Result matrix, already initialized as
 *                        (a->ncols x a->nrows)
 *      a               : Matrix to transpose (nrows x ncols)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(nrows*ncols*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-23
 * Last modified: 2026-09-23
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_mat_neg(ancora_mat *res, const ancora_mat *a);
/* Computes the negative of a matrix (res = -a).
 * INPUT:
 *      res             : Result matrix, already initialized
 *      a               : Matrix to negate
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(nrows*ncols)
 *      In ANCORA_MODE_SAFE mode, negation is exact (no rounding), so this
 *      does not scale with ANCORA_DEFAULT_PREC.
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */



#ifdef __cplusplus
}
#endif

#endif
