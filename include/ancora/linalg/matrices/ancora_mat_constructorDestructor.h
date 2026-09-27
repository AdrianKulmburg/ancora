/*
 * ancora_mat_constructorDestructor.h
 *
 * Description
 * -----------
 * Constructor, destructor, and copy for ancora_mat.
 *
 * File Information
 * ----------------
 * Created:       2026-09-21
 * Last modified: 2026-09-21
 * Authors:       Adrian Kulmburg
 *
 * License
 * -------
 * Copyright (c) 2026 Adrian Kulmburg <adrian.kulmburg@kit.edu>
 * SPDX-License-Identifier: MIT
 */

#ifndef ANCORA_CONSTRUCTORDESTRUCTOR_H
#define ANCORA_CONSTRUCTORDESTRUCTOR_H

#include <string.h>

#include "ancora/linalg/matrices/ancora_mat.h"

#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_mat_init(ancora_mat *m, slong nrows, slong ncols);
/* Initializes a matrix with a set number of rows and columns.
 * INPUT:
 *      m               : Pointer to an (uninitialized) ancora matrix, to be
 *                        initialized
 *      nrows           : Number of rows
 *      ncols           : Number of columns
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(nrows*ncols*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-21
 * Last modified: 2026-09-21
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_mat_free(ancora_mat *m);
/* Frees a previously initialized ancora matrix.
 * INPUT:
 *      m               : Pointer to an initialized ancora_mat instance, to
 *                        be freed
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(1) in ANCORA_MODE_FAST mode.
 *      O(nrows*ncols) in ANCORA_MODE_SAFE mode (one arb_clear per entry;
 *      independent of ANCORA_DEFAULT_PREC).
 *
 * Created:       2026-09-21
 * Last modified: 2026-09-21
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_mat_copy(ancora_mat *res, const ancora_mat *a);
/* Copies the contents of a into res (res = a). Both must have the same
 * dimensions. Copying a matrix into itself (res == a) is a safe no-op.
 * INPUT:
 *      res             : Result matrix, already initialized as the same
 *                        dimensions as a
 *      a               : Matrix to copy
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(nrows*ncols*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):       Adrian Kulmburg
 */

#ifdef __cplusplus
}
#endif

#endif
