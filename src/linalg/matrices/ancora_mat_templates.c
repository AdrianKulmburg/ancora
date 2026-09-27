/*
 * ancora_mat_templates.c
 *
 * Description
 * -----------
 * Constructors for canonical matrices: identity, zeros, ones, and random
 * (uniform and standard Gaussian).
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

#include "ancora/linalg/matrices/ancora_mat_templates.h"

ancora_status ancora_mat_eye(ancora_mat *res)
/* Fills res with the (n x n) identity matrix.
 * INPUT:
 *      res             : Result matrix, already initialized as (n x n)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n^2)
 *      In ANCORA_MODE_SAFE mode, entries are exact (0 or 1), so this does
 *      not scale with ANCORA_DEFAULT_PREC.
 *
 * Created:       2026-09-22
 * Last modified: 2026-09-22
 * Author(s):     Adrian Kulmburg
 */
{
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_mat instance.");
    }
    if (res->ncols != res->nrows) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Result should be a square matrix, but res is (%ld x %ld).",
                      (long)res->nrows, (long)res->ncols);
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_mat_one(res->repr);
#elif ANCORA_MODE == ANCORA_MODE_FAST
    for (slong i = 0; i < res->nrows; i++) {
        for (slong j = 0; j < res->nrows; j++) {
            res->repr[i * res->nrows + j] = (i == j) ? 1.0 : 0.0;
        }
    }
#endif
    return ANCORA_OK;
}

ancora_status ancora_mat_zeros(ancora_mat *res)
/* Fills res with the (nrows x ncols) zero matrix.
 * INPUT:
 *      res             : Result matrix, already initialized as
 *                        (nrows x ncols)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(nrows*ncols) in ANCORA_MODE_FAST mode.
 *      In ANCORA_MODE_SAFE mode, entries are set to the exact value 0,
 *      which does not scale with ANCORA_DEFAULT_PREC.
 *
 * Created:       2026-09-22
 * Last modified: 2026-09-22
 * Author(s):     Adrian Kulmburg
 */
{
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_mat instance.");
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_mat_zero(res->repr);
#elif ANCORA_MODE == ANCORA_MODE_FAST
    if (res->nrows > 0 && res->ncols > 0) {
        memset(res->repr, 0, (size_t)(res->nrows * res->ncols) * sizeof(double));
    }
#endif
    return ANCORA_OK;
}

ancora_status ancora_mat_ones(ancora_mat *res)
/* Fills res with the (nrows x ncols) all-ones matrix.
 * INPUT:
 *      res             : Result matrix, already initialized as
 *                        (nrows x ncols)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(nrows*ncols)
 *      In ANCORA_MODE_SAFE mode, entries are exact (1), so this does not
 *      scale with ANCORA_DEFAULT_PREC.
 *
 * Created:       2026-09-22
 * Last modified: 2026-09-22
 * Author(s):     Adrian Kulmburg
 */
{
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_mat instance.");
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    for (slong i = 0; i < res->nrows; i++) {
        for (slong j = 0; j < res->ncols; j++) {
            arb_one(arb_mat_entry(res->repr, i, j));
        }
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
    for (slong k = 0; k < res->nrows * res->ncols; k++) {
        res->repr[k] = 1.0;
    }
#endif
    return ANCORA_OK;
}


ancora_status ancora_mat_randomUniform(ancora_mat *res)
/* Fills res with entries drawn independently, uniformly at random from
 * [-1, 1], via ancora_random_uniform / ancora_random_uniform_arb (one
 * draw per entry).
 * INPUT:
 *      res             : Result matrix, already initialized as
 *                        (nrows x ncols)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(nrows*ncols*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */
{
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_mat instance.");
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    for (slong i = 0; i < res->nrows; i++) {
        for (slong j = 0; j < res->ncols; j++) {
            ANCORA_TRY(ancora_random_uniform_arb(-1.0, 1.0, arb_mat_entry(res->repr, i, j)));
        }
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
    for (slong k = 0; k < res->nrows * res->ncols; k++) {
        ANCORA_TRY(ancora_random_uniform(-1.0, 1.0, &res->repr[k]));
    }
#endif
    return ANCORA_OK;
}

ancora_status ancora_mat_randomGaussian(ancora_mat *res)
/* Fills res with entries drawn independently from a standard Gaussian
 * (mean 0, variance 1).
 * INPUT:
 *      res             : Result matrix, already initialized as
 *                        (nrows x ncols)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(nrows*ncols*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */
{
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_mat instance.");
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    for (slong i = 0; i < res->nrows; i++) {
        for (slong j = 0; j < res->ncols; j++) {
            ANCORA_TRY(ancora_random_gaussian_arb(arb_mat_entry(res->repr, i, j)));
        }
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
    for (slong k = 0; k < res->nrows * res->ncols; k++) {
        ANCORA_TRY(ancora_random_gaussian(&res->repr[k]));
    }
#endif
    return ANCORA_OK;
}

ancora_status ancora_mat_vecDiag(ancora_mat *res, const ancora_vec *v)
/* Constructs the diagonal matrix that has the vector v on its diagonal.
 * INPUT:
 *      res             : Result matrix, already initialized as (n x n)
 *      v               : Initialized ancora_vec
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n^2*ANCORA_DEFAULT_PREC)
 *      where n is the dimension of the vector v.
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */
{
    // Check that res and v are well-defined
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_mat instance.");
    }
    if (v == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer v is NULL; it should point to a valid ancora_vec instance.");
    }

    // Verify that v is a vector
    bool isVector;
    ANCORA_TRY(ancora_mat_isVector(v, &isVector));
    if (!isVector)
    {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer v is not a vector; it should point to a valid ancora_vec instance.");
    }

    // Dimension checks
    if (res->nrows != res->ncols) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Matrix res has %ld rows and %ld columns; needs to be square.",
                      (long)res->nrows, (long)res->ncols);
    }
    if (res->nrows != v->nrows) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Matrix res has %ld rows and columns, v has length %ld; they need to be the same.",
                      (long)res->nrows, (long)v->nrows);
    }

    // Begin by setting res to zero
    ANCORA_TRY(ancora_mat_zeros(res));

    // Now, set the diagonal elemens
    for (slong i = 0; i < res->nrows; i++)
    {
#if ANCORA_MODE == ANCORA_MODE_SAFE
        ANCORA_TRY(ancora_mat_set(res, i, i, arb_mat_entry(v->repr, i, 0)));
#elif ANCORA_MODE == ANCORA_MODE_FAST
        ANCORA_TRY(ancora_mat_set(res, i, i, v->repr[i]));
#endif
    }

    return ANCORA_OK;
}
