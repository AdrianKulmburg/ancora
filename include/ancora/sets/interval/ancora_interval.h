/*
 * ancora_interval.h
 *
 * Description
 * -----------
 * The ancora interval set type: { x in R^n | lowerBound[i] <= x[i] <=
 * upperBound[i], for all i = 1,...,n }. The lower and upper bounds are stored
 * as ancora_vec. Bounds are allowed to be +inf or -inf, but never NaN.
 *
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

#ifndef ANCORA_INTERVAL_H
#define ANCORA_INTERVAL_H

#include "ancora/ancora_config.h"
#include "ancora/ancora_error.h"

#include "ancora/linalg/vectors/ancora_vec.h"
#include "ancora/linalg/matrices/ancora_mat.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The interval type. lowerBound and upperBound are ancora vectors holding the
 * lower and upper bounds, respectively.
 */
typedef struct {
    ancora_vec lowerBound;
    ancora_vec upperBound;
} ancora_interval;

#include "ancora/sets/interval/ancora_interval_constructorDestructor.h"
#include "ancora/sets/interval/ancora_interval_templates.h"
#include "ancora/sets/interval/ancora_interval_arithmetic.h"
#include "ancora/sets/interval/ancora_interval_containment.h"
#include "ancora/sets/interval/ancora_interval_properties.h"
#include "ancora/sets/interval/ancora_interval_duality.h"
#include "ancora/sets/interval/ancora_interval_randomPoints.h"

#ifdef __cplusplus
}
#endif

#endif
