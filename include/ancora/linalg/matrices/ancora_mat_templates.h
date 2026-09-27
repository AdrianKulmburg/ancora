/*
 * ancora_mat_templates.h
 *
 * Description
 * -----------
 * Constructors for canonical matrices: identity, zeros, ones, and random
 * (uniform and standard Gaussian).
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

#ifndef ANCORA_MAT_TEMPLATES_H
#define ANCORA_MAT_TEMPLATES_H

#include <string.h>

#include "ancora/ancora_config.h"
#include "ancora/ancora_error.h"
#include "ancora/ancora_random.h"

#include "ancora/linalg/matrices/ancora_mat.h"
#include "ancora/linalg/vectors/ancora_vec.h"

#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_mat_eye(ancora_mat *res);
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

ancora_status ancora_mat_zeros(ancora_mat *res);
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

ancora_status ancora_mat_ones(ancora_mat *res);
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

ancora_status ancora_mat_randomUniform(ancora_mat *res);
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

ancora_status ancora_mat_randomGaussian(ancora_mat *res);
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

ancora_status ancora_mat_vecDiag(ancora_mat *res, const ancora_vec *v);
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

#ifdef __cplusplus
}
#endif

#endif /* ANCORA_MAT_TEMPLATES_H */
