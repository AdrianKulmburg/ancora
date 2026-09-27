/*
 * ancora_vec_concatenation.h
 *
 * Description
 * -----------
 * Concatenation operations for vectors.
 *
 * File Information
 * ----------------
 * Created:       2026-09-22
 * Last modified: 2026-09-24
 * Authors:       Adrian Kulmburg
 *
 * License
 * -------
 * Copyright (c) 2026 Adrian Kulmburg <adrian.kulmburg@kit.edu>
 * SPDX-License-Identifier: MIT
 */

#ifndef ANCORA_VEC_CONCATENATION_H
#define ANCORA_VEC_CONCATENATION_H

#include "ancora/linalg/vectors/ancora_vec.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ancora_vec_cat(res, a, b) ancora_mat_vcat((res), (a), (b))
/* Concatenation: res = [a; b] (a followed by b). Delegates to ancora_mat_vcat.
 * INPUT:
 *      res             : Result vector, already initialized as
 *                        (a->nrows + b->nrows)
 *      a               : First block
 *      b               : Second block
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O((a->nrows + b->nrows)*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-22
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */

#define ancora_vec_hcat(res, a, b) ancora_mat_hcat((res), (a), (b))
/* Horizontal concatenation: res = [a b]. Delegates to ancora_mat_hcat.
 * For two column vectors this produces an (n x 2) matrix (not a vector), so
 * res must be initialized as (a->nrows x 2).
 * INPUT:
 *      res             : Result matrix, already initialized as
 *                        (a->nrows x (a->ncols + b->ncols))
 *      a               : Left block
 *      b               : Right block
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(a->nrows*(a->ncols+b->ncols)*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */

#ifdef __cplusplus
}
#endif

#endif
