/*
 * ancora_mat_arithmetic.c
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

#if ANCORA_MODE == ANCORA_MODE_FAST
#include <limits.h>
#endif

#if ANCORA_MODE == ANCORA_MODE_FAST && !defined(ANCORA_USE_GPU)
#include <cblas.h>
#endif

#include "ancora/linalg/matrices/ancora_mat_arithmetic.h"

ancora_status ancora_mat_add(ancora_mat *res,
                             const ancora_mat *a,
                             const ancora_mat *b)
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
{
    // Check that all matrices are well-defined
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_mat instance.");
    }
    if (a == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer a is NULL; it should point to a valid ancora_mat instance.");
    }
    if (b == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer b is NULL; it should point to a valid ancora_mat instance.");
    }

    // Check that the number of rows/columns is consistent for a and b
    if (a->nrows != b->nrows || a->ncols != b->ncols)
    {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH, "Matrix a is (%ld x %ld), matrix b is (%ld x %ld); they need to be the same.", (long)a->nrows, (long)a->ncols, (long)b->nrows, (long)b->ncols);
    }
    if (res->nrows != a->nrows || res->ncols != a->ncols) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH, "Matrices a and b are (%ld x %ld), matrix res is (%ld x %ld); they need to be the same.", (long)a->nrows, (long)a->ncols, (long)res->nrows, (long)res->ncols);
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_mat_add(res->repr, a->repr, b->repr, ANCORA_DEFAULT_PREC);
#elif ANCORA_MODE == ANCORA_MODE_FAST
    size_t n = (size_t)(a->nrows * a->ncols);
    #ifdef ANCORA_USE_GPU
        if (ancora_mat_add_gpu(a->repr, b->repr, res->repr, n) != 0) {
            ANCORA_ERROR(ANCORA_ERROR_GPU_LAUNCH, "GPU launch failed.");
        }
    #else
        for (size_t k = 0; k < n; k++) {
            res->repr[k] = a->repr[k] + b->repr[k];
        }
    #endif
#endif
    return ANCORA_OK;
}

ancora_status ancora_mat_sub(ancora_mat *res,
                             const ancora_mat *a,
                             const ancora_mat *b)
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
{
    // Check that all matrices are well-defined
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_mat instance.");
    }
    if (a == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer a is NULL; it should point to a valid ancora_mat instance.");
    }
    if (b == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer b is NULL; it should point to a valid ancora_mat instance.");
    }

    // Check that the number of rows/columns is consistent for a and b
    if (a->nrows != b->nrows || a->ncols != b->ncols) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Matrix a is (%ld x %ld), matrix b is (%ld x %ld); they need to be the same.",
                      (long)a->nrows, (long)a->ncols, (long)b->nrows, (long)b->ncols);
    }
    if (res->nrows != a->nrows || res->ncols != a->ncols) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Matrices a and b are (%ld x %ld), matrix res is (%ld x %ld); they need to be the same.",
                      (long)a->nrows, (long)a->ncols, (long)res->nrows, (long)res->ncols);
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_mat_sub(res->repr, a->repr, b->repr, ANCORA_DEFAULT_PREC);
#elif ANCORA_MODE == ANCORA_MODE_FAST
    size_t n = (size_t)(a->nrows * a->ncols);
    #ifdef ANCORA_USE_GPU
        if (ancora_mat_sub_gpu(a->repr, b->repr, res->repr, n) != 0) {
            ANCORA_ERROR(ANCORA_ERROR_GPU_LAUNCH, "GPU launch failed.");
        }
    #else
        for (size_t k = 0; k < n; k++) {
            res->repr[k] = a->repr[k] - b->repr[k];
        }
    #endif
#endif
    return ANCORA_OK;
}

ancora_status ancora_mat_scalarMul(ancora_mat *res,
                                    const ancora_mat *a,
#if ANCORA_MODE == ANCORA_MODE_SAFE
                                    arb_t scalar
#elif ANCORA_MODE == ANCORA_MODE_FAST
                                    double scalar
#endif
                                   )
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
{
    // Check that all matrices are well-defined
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_mat instance.");
    }
    if (a == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer a is NULL; it should point to a valid ancora_mat instance.");
    }

    // Check that the number of rows/columns is consistent for res and a
    if (res->nrows != a->nrows || res->ncols != a->ncols) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Matrix a is (%ld x %ld), matrix res is (%ld x %ld); they need to be the same.",
                      (long)a->nrows, (long)a->ncols, (long)res->nrows, (long)res->ncols);
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_mat_scalar_mul_arb(res->repr, a->repr, scalar, ANCORA_DEFAULT_PREC);
#elif ANCORA_MODE == ANCORA_MODE_FAST
    #ifdef ANCORA_USE_GPU
        if (ancora_mat_scalarMul_gpu(a->repr, res->repr, scalar, (size_t)(a->nrows * a->ncols)) != 0) {
            ANCORA_ERROR(ANCORA_ERROR_GPU_LAUNCH, "GPU launch failed.");
        }
    #else
        for (slong i = 0; i < a->nrows; i++) {
            for (slong j = 0; j < a->ncols; j++) {
                res->repr[i * a->ncols + j] = scalar * a->repr[i * a->ncols + j];
            }
        }
    #endif
#endif
    return ANCORA_OK;
}

ancora_status ancora_mat_mul(ancora_mat *res,
                             const ancora_mat *a,
                             const ancora_mat *b)
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
{
    // Check that all matrices are well-defined
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_mat instance.");
    }
    if (a == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer a is NULL; it should point to a valid ancora_mat instance.");
    }
    if (b == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer b is NULL; it should point to a valid ancora_mat instance.");
    }

    // Check the inner dimension matches: a is (n x k), b is (k x m)
    if (a->ncols != b->nrows) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Matrix a is (%ld x %ld), matrix b is (%ld x %ld); a's column count must match b's row count.",
                      (long)a->nrows, (long)a->ncols, (long)b->nrows, (long)b->ncols);
    }
    // Check res is (n x m)
    if (res->nrows != a->nrows || res->ncols != b->ncols) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Result should be (%ld x %ld), but res is (%ld x %ld).",
                      (long)a->nrows, (long)b->ncols, (long)res->nrows, (long)res->ncols);
    }
    // res must be a distinct matrix from a and b: an in-place product isn't
    // a simple entrywise operation like addition, so aliasing would silently
    // corrupt intermediate sums. Reject it rather than producing a wrong
    // result.
    if (res == a || res == b) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG,
                      "res must not alias a or b for ancora_mat_mul.");
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_mat_mul(res->repr, a->repr, b->repr, ANCORA_DEFAULT_PREC);
#elif ANCORA_MODE == ANCORA_MODE_FAST
    slong n = a->nrows;
    slong k = a->ncols;
    slong m = b->ncols;
    #ifdef ANCORA_USE_GPU
        if (n > INT_MAX || k > INT_MAX || m > INT_MAX) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Matrix dimensions exceed BLAS's 32-bit integer limit.");
        }

        if (ancora_mat_mul_gpu(a->repr, b->repr, res->repr, n, k, m) != 0) {
            ANCORA_ERROR(ANCORA_ERROR_GPU_LAUNCH, "GPU launch failed.");
        }
    #else

        if (n > INT_MAX || k > INT_MAX || m > INT_MAX) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Matrix dimensions exceed BLAS's 32-bit integer limit.");
        }
        /* res = 1.0 * a * b + 0.0 * res, all row-major, no transposes.
         * Leading dimensions equal the row length of each matrix (k for
         * a, m for b and res), since there is no padding between rows. */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)n, (int)m, (int)k,
                    1.0, a->repr, (int)k,
                    b->repr, (int)m,
                    0.0, res->repr, (int)m);
    #endif
#endif
    return ANCORA_OK;
}

ancora_status ancora_mat_transpose(ancora_mat *res, const ancora_mat *a)
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
{
    // Check that all matrices are well-defined
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_mat instance.");
    }
    if (a == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer a is NULL; it should point to a valid ancora_mat instance.");
    }

    // Check that the number of rows/columns is consistent for res and a
    if (res->nrows != a->ncols || res->ncols != a->nrows) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Result should be (%ld x %ld), but res is (%ld x %ld).",
                      (long)a->ncols, (long)a->nrows, (long)res->nrows, (long)res->ncols);
    }
    /* res must be a distinct matrix from a: an in-place transpose is not a
     * simple entrywise operation, so aliasing would corrupt the result. */
    if (res == a) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "res must not alias a for ancora_mat_transpose.");
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    for (slong i = 0; i < a->nrows; i++) {
        for (slong j = 0; j < a->ncols; j++) {
            arb_set(arb_mat_entry(res->repr, j, i), arb_mat_entry(a->repr, i, j));
        }
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
    #ifdef ANCORA_USE_GPU
        if (ancora_mat_transpose_gpu(a->repr, res->repr, a->nrows, a->ncols) != 0) {
            ANCORA_ERROR(ANCORA_ERROR_GPU_LAUNCH, "GPU launch failed.");
        }
    #elif defined(ANCORA_HAVE_CBLAS_DOMATCOPY)
        /* cblas_domatcopy is a widely-shipped BLAS extension (OpenBLAS,
         * some other vendors), NOT part of the official BLAS standard, so
         * it isn't guaranteed to exist on every BLAS -- gated behind
         * ANCORA_HAVE_CBLAS_DOMATCOPY (see CMakeLists.txt), falling back
         * to the plain loop below when unavailable (e.g. reference BLAS). */
        cblas_domatcopy(CblasRowMajor, CblasTrans,
                        (size_t)a->nrows, (size_t)a->ncols,
                        1.0, a->repr, (size_t)a->ncols,
                        res->repr, (size_t)a->nrows);
    #else
        for (slong i = 0; i < a->nrows; i++) {
            for (slong j = 0; j < a->ncols; j++) {
                res->repr[j * a->nrows + i] = a->repr[i * a->ncols + j];
            }
        }
    #endif
#endif
    return ANCORA_OK;
}

ancora_status ancora_mat_neg(ancora_mat *res, const ancora_mat *a)
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
{
    // Check that res and a are well-defined
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_mat instance.");
    }
    if (a == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer a is NULL; it should point to a valid ancora_mat instance.");
    }

    // Check dimensions
    if (res->nrows != a->nrows || res->ncols != a->ncols) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Result should be (%ld x %ld), but res is (%ld x %ld).",
                      (long)a->nrows, (long)a->ncols, (long)res->nrows, (long)res->ncols);
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_mat_neg(res->repr, a->repr);
#elif ANCORA_MODE == ANCORA_MODE_FAST
    slong n = a->nrows * a->ncols;
    #ifdef ANCORA_USE_GPU
        if (ancora_mat_neg_gpu(a->repr, res->repr, (size_t)n) != 0) {
            ANCORA_ERROR(ANCORA_ERROR_GPU_LAUNCH, "GPU launch failed.");
        }
    #else
        for (slong k = 0; k < n; k++) {
            res->repr[k] = -a->repr[k];
        }
    #endif
#endif
    return ANCORA_OK;
}
