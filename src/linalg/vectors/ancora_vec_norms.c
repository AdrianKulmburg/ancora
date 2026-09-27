/*
 * ancora_vec_norms.h
 *
 * Description
 * -----------
 * Norms for ancora_vec, with dedicated functions for the 1-norm, the
 * (Euclidean) 2-norm, and the inf norm.
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

#include "ancora/linalg/vectors/ancora_vec_norms.h"

ancora_status ancora_vec_1norm(
#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_t res,
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double *res,
#endif
    const ancora_vec *v)
/* 1-norm of a vector: ||v||_1 = sum_i |v_i|.
 * INPUT:
 *      res             : Output scalar (arb_t in SAFE mode, double* in FAST)
 *      v               : Vector
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
    // Check that both vectors are well-defined
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid output scalar.");
    }
    if (v == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer v is NULL; it should point to a valid ancora_vec.");
    }

    // Check that v is a vector
    bool isVector;
    ANCORA_TRY(ancora_mat_isVector(v, &isVector));
    if (!isVector)
    {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer v is not a vector; it should point to a valid ancora_vec.");
    }

    slong n = v->nrows;

#if ANCORA_MODE == ANCORA_MODE_SAFE
    if (n == 0) {
        arb_zero(res);
        return ANCORA_OK;
    }

    arb_t acc, t;
    arb_init(acc);
    arb_init(t);

    arb_zero(acc);
    for (slong i = 0; i < n; i++) {
        arb_abs(t, arb_mat_entry(v->repr, i, 0));
        arb_add(acc, acc, t, ANCORA_DEFAULT_PREC);
    }
    arb_set(res, acc);

    arb_clear(acc);
    arb_clear(t);

#elif ANCORA_MODE == ANCORA_MODE_FAST
    if (n == 0) {
        *res = 0.0;
        return ANCORA_OK;
    }


    double acc = 0.0;
    for (slong i = 0; i < n; i++) {
        acc += fabs(v->repr[i]);
    }
    *res = acc;
#endif
    return ANCORA_OK;
}
