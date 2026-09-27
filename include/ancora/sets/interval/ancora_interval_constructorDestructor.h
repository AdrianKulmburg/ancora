/*
 * ancora_interval_constructorDestructor.h
 *
 * Description
 * -----------
 * Constructor and destructor for ancora_interval.
 *
 * File Information
 * ----------------
 * Created:       2026-09-23
 * Last modified: 2026-09-23
 * Authors:       Adrian Kulmburg
 *
 * License
 * -------
 * Copyright (c) 2026 Adrian Kulmburg <adrian.kulmburg@kit.edu>
 * SPDX-License-Identifier: MIT
 */

#ifndef ANCORA_INTERVAL_CONSTRUCTORDESTRUCTOR_H
#define ANCORA_INTERVAL_CONSTRUCTORDESTRUCTOR_H

#include "ancora/sets/interval/ancora_interval.h"

#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_interval_init(ancora_interval *I,
                                   const ancora_vec *lowerBound,
                                   const ancora_vec *upperBound);
/* Initializes an interval from given lower and upper bound vectors.
 *
 * INPUT:
 *      I               : Pointer to an (uninitialized) ancora interval, to be
 *                        initialized
 *      lowerBound      : Lower bound vector (length n)
 *      upperBound      : Upper bound vector (length n)
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

ancora_status ancora_interval_free(ancora_interval *I);
/* Frees a previously initialized ancora interval.
 * INPUT:
 *      I               : Pointer to an initialized ancora_interval instance,
 *                        to be freed
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(1) in ANCORA_MODE_FAST mode.
 *      O(n) in ANCORA_MODE_SAFE mode (one arb_clear per bound entry;
 *      independent of ANCORA_DEFAULT_PREC), where n is the dimension of the
 *      interval.
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */

#ifdef __cplusplus
}
#endif

#endif
