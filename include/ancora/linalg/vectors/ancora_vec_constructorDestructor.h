/*
 * ancora_vec_constructorDestructor.h
 *
 * Description
 * -----------
 * Constructor, destructor, and copy for ancora_vec.
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

#ifndef ANCORA_VEC_CONSTRUCTORDESTRUCTOR_H
#define ANCORA_VEC_CONSTRUCTORDESTRUCTOR_H

#include "ancora/linalg/vectors/ancora_vec.h"

#define ancora_vec_init(v, length) ancora_mat_init((v), (length), 1)
/* Initializes a vector with a given length: a (length x 1) matrix.
 * INPUT:
 *      v               : Pointer to an (uninitialized) ancora vector, to be
 *                        initialized
 *      length          : Number of entries
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(length*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-22
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */

#define ancora_vec_free(v) ancora_mat_free(v)
/* Frees a previously initialized ancora vector.
 * INPUT:
 *      v               : Pointer to an initialized ancora_vec instance, to
 *                        be freed
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(1) in ANCORA_MODE_FAST mode.
 *      O(length) in ANCORA_MODE_SAFE mode (one arb_clear per entry;
 *      independent of ANCORA_DEFAULT_PREC).
 *
 * Created:       2026-09-22
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */

#define ancora_vec_copy(res, a) ancora_mat_copy((res), (a))
/* Copies the contents of a into res (res = a). Both must be column vectors of
 * the same length. Copying a vector into itself (res == a) is a safe no-op.
 * INPUT:
 *      res             : Result vector, already initialized as the same
 *                        length as a
 *      a               : Vector to copy
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(length*ANCORA_DEFAULT_PREC)
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

#endif
