/*
 * ancora_vec_templates.h
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

#ifndef ANCORA_VEC_TEMPLATES_H
#define ANCORA_VEC_TEMPLATES_H

#include "ancora/linalg/vectors/ancora_vec.h"

#define ancora_vec_zeros(res) ancora_mat_zeros(res)
/* Fills res with the (nrows x 1) zero vector.
 * INPUT:
 *      res             : Result vector, already initialized as a
 *                        (nrows x 1) matrix
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(nrows) in ANCORA_MODE_FAST mode.
 *      In ANCORA_MODE_SAFE mode, entries are set to the exact value 0,
 *      which does not scale with ANCORA_DEFAULT_PREC.
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */

#define ancora_vec_ones(res) ancora_mat_ones(res)
/* Fills res with the (nrows x 1) all-ones vector.
 * INPUT:
 *      res             : Result vector, already initialized as a
 *                        (nrows x 1) matrix
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(nrows)
 *      In ANCORA_MODE_SAFE mode, entries are exact (1), so this does not
 *      scale with ANCORA_DEFAULT_PREC.
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */


#define ancora_vec_randomUniform(res) ancora_mat_randomUniform(res)
/* Fills res with entries drawn independently, uniformly at random from
 * [-1, 1], via ancora_random_uniform / ancora_random_uniform_arb (one
 * draw per entry).
 * INPUT:
 *      res             : Result vector, already initialized as a
 *                        (nrows x 1) matrix
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(nrows*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */

#define ancora_vec_randomGaussian(res) ancora_mat_randomGaussian(res)
/* Fills res with entries drawn independently from a standard Gaussian
 * (mean 0, variance 1).
 * INPUT:
 *      res             : Result vector, already initialized as a
 *                        (nrows x 1) matrix
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(nrows*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif /* ANCORA_VEC_TEMPLATES_H */
