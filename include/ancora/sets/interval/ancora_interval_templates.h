/*
 * ancora_interval_templates.h
 *
 * Description
 * -----------
 * Constructors for canonical and random intervals.
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

#ifndef ANCORA_INTERVAL_TEMPLATES_H
#define ANCORA_INTERVAL_TEMPLATES_H

#include "ancora/ancora_random.h"

#include "ancora/sets/interval/ancora_interval.h"

#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_interval_initRandom_uniform(ancora_interval *I);
/* Fills an already-initialized interval with random bounds.
 * For each dimension i, two samples are drawn from the uniform distribution on
 * [-1, 1]. The smallest number is assigned to lowerBound[i], the largest to
 * upperBound[i], ensuring that the interval is never empty.
 *
 * INPUT:
 *      I               : Pointer to an initialized ancora_interval instance
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
