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

#ifndef ANCORA_VEC_NORMS_H
#define ANCORA_VEC_NORMS_H

#include <math.h>

#include "ancora/linalg/matrices/ancora_mat.h"
#include "ancora/linalg/vectors/ancora_vec.h"

#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_vec_1norm(
#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_t res,
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double *res,
#endif
    const ancora_vec *v);
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

#ifdef __cplusplus
}
#endif

#endif /* ANCORA_VEC_NORMS_H */
