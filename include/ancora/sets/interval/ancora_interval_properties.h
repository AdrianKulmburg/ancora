/*
 * ancora_interval_properties.h
 *
 * Description
 * -----------
 * Elementary properties for intervals.
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

#ifndef ANCORA_INTERVAL_PROPERTIES_H
#define ANCORA_INTERVAL_PROPERTIES_H

#include "ancora/ancora_config.h"
#include "ancora/ancora_error.h"

#include "ancora/linalg/vectors/ancora_vec.h"
#include "ancora/linalg/matrices/ancora_mat.h"

#include "ancora/sets/interval/ancora_interval.h"

#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_interval_dimension(const ancora_interval *I, slong *dimension);
/* Determines the dimension of an initialized interval.
 *
 * INPUT:
 *      I               : ancora_interval instance
 *      dimension       : Pointer to an slong that will be filled with the
 *                        dimension of I
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(1)
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):       Adrian Kulmburg
 */

#ifdef __cplusplus
}
#endif

#endif
