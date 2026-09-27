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

#include "ancora/linalg/vectors/ancora_vec_arithmetic.h"

ancora_status ancora_vec_dot(
#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_t res,
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double *res,
#endif
    const ancora_vec *a,
    const ancora_vec *b)
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
{
    // Check that res, a, and b are well-defined
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid output scalar.");
    }
    if (a == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer a is NULL; it should point to a valid ancora_vec instance.");
    }
    if (b == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer b is NULL; it should point to a valid ancora_vec instance.");
    }

    // Check that a and b are vectors
    bool aIsVector;
    bool bIsVector;
    ANCORA_TRY(ancora_mat_isVector(a, &aIsVector));
    ANCORA_TRY(ancora_mat_isVector(b, &bIsVector));
    if (!aIsVector)
    {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer a is not a vector; it should point to a valid ancora_vec.");
    }
    if (!bIsVector)
    {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer b is not a vector; it should point to a valid ancora_vec.");
    }

    // Check that the dimensions match
    if (a->nrows != b->nrows) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Vector a has length %ld, vector b has length %ld; they need to be the same.",
                      (long)a->nrows, (long)b->nrows);
    }

    slong n = a->nrows;

#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_t acc, t;
    arb_init(acc);
    arb_init(t);

    arb_zero(acc);
    for (slong i = 0; i < n; i++) {
        arb_mul(t, arb_mat_entry(a->repr, i, 0), arb_mat_entry(b->repr, i, 0),
                ANCORA_DEFAULT_PREC);
        arb_add(acc, acc, t, ANCORA_DEFAULT_PREC);
    }
    arb_set(res, acc);

    arb_clear(acc);
    arb_clear(t);
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double sum = 0.0;
    for (slong i = 0; i < n; i++) {
        sum += a->repr[i] * b->repr[i];
    }
    *res = sum;
#endif
    return ANCORA_OK;
}
