/*
 * ancora_mat_getSet.h
 *
 * Description
 * -----------
 * Elementary operations for ancora_mat (e.g., getters and setters of the
 * (i,j)-th entry).
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

#ifndef ANCORA_MAT_GETSET_H
#define ANCORA_MAT_GETSET_H

#include "ancora/linalg/matrices/ancora_mat.h"
#include "ancora/linalg/vectors/ancora_vec.h"

#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_mat_set(ancora_mat *m,
                             slong i,
                             slong j,
#if ANCORA_MODE == ANCORA_MODE_SAFE
                             arb_t in
#elif ANCORA_MODE == ANCORA_MODE_FAST
                             double in
#endif
                             );
/* Sets the (i,j) entry of an ancora matrix.
 * INPUT:
 *      m               : Pointer to an initialized ancora_mat instance
 *      i, j            : Row/column index (0-based)
 *      in              : Value to copy into (i,j)-th entry of m (arb_t or
 *                        double, depending on ANCORA_MODE)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-21
 * Last modified: 2026-09-21
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_mat_get(const ancora_mat *m,
                             const slong i,
                             const slong j,
#if ANCORA_MODE == ANCORA_MODE_SAFE
                             arb_t out
#elif ANCORA_MODE == ANCORA_MODE_FAST
                             double *out
#endif
                             );
/* Reads the (i,j)-th entry of an ancora matrix.
 *
 * INPUT:
 *      m               : Pointer to an initialized ancora_mat instance
 *      i, j            : Row/column index (0-based)
 *
 *      out             : Value to read into the (i,j)-th entry of m (arb_t
 *                        or *double depending on ANCORA_MODE)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-21
 * Last modified: 2026-09-21
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_mat_getColumn(const ancora_mat *m,
                             const slong i,
                             ancora_vec *out);
/* Reads the i-th column of an ancora matrix.
 *
 * INPUT:
 *      m               : Pointer to an initialized ancora_mat instance
 *      i               : Column index (0-based)
 *
 *      out             : Value to read into the i-th column of m
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*ANCORA_DEFAULT_PREC)
 *      where n is the number of rows of m
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */

#ifdef __cplusplus
}
#endif

#endif
