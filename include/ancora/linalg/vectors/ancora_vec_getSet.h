/*
 * ancora_vec_getSet.h
 *
 * Description
 * -----------
 * Elementary properties of an ancora_mat (e.g., getters and setters of the
 * (i,j)-th entry).
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

#ifndef ANCORA_VEC_GETSET_H
#define ANCORA_VEC_GETSET_H

#include "ancora/linalg/vectors/ancora_vec.h"

#define ancora_vec_set(v, i, in) ancora_mat_set(v, i, 0, in)
/* Sets the i-th entry of an ancora vector.
 * INPUT:
 *      v               : Pointer to an initialized ancora_vec instance
 *      i               : Index (0-based)
 *      in              : Value to copy into i-th entry of v (arb_t or
 *                        double, depending on ANCORA_MODE)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */

#define ancora_vec_get(v, i, out) ancora_mat_get(v, i, 0, out)
/* Reads the i-th entry of an ancora vector as a plain double.
 *
 * INPUT:
 *      v               : Pointer to an initialized ancora_vec instance
 *      i               : Index (0-based)
 *
 *      out             : Value to read into the i-th entry of v (arb_t
 *                        or *double depending on ANCORA_MODE)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(ANCORA_DEFAULT_PREC)
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
