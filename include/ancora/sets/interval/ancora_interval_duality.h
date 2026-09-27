/*
 * ancora_interval_duality.h
 *
 * Description
 * -----------
 * Duality operations for intervals.
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

#ifndef ANCORA_INTERVAL_DUALITY_H
#define ANCORA_INTERVAL_DUALITY_H

#include "ancora/sets/interval/ancora_interval.h"



#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_interval_supportFunction(
#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_t res,
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double *res,
#endif
    const ancora_interval *I,
    const ancora_vec *d);
/* Computes the support function of the interval I in the direction d, i.e.,
 * h_I(d) = sup { d^T x | x in I }.
 * INPUT:
 *      res             : Output scalar (arb_t in SAFE mode, double* in FAST
 *                        mode), set to the support value
 *      I               : Pointer to an initialized ancora_interval instance
 *      d               : Direction, a column vector of length I->lowerBound.nrows
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*ANCORA_DEFAULT_PREC)
 *      where n is the dimension of the interval.
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */

#ifdef __cplusplus
}
#endif

#endif
