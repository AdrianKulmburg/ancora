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

#ifndef ANCORA_ZONOTOPE_PROPERTIES_H
#define ANCORA_ZONOTOPE_PROPERTIES_H

#include "ancora/ancora_config.h"
#include "ancora/ancora_error.h"

#include "ancora/linalg/vectors/ancora_vec.h"
#include "ancora/linalg/matrices/ancora_mat.h"

#include "ancora/sets/zonotope/ancora_zonotope.h"

#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_zonotope_dimension(const ancora_zonotope *Z, slong *dimension);
/* Determines the dimension of an initialized zonotope.
 *
 * INPUT:
 *      Z               : ancora_zonotope instance
 *      dimension       : Pointer to an slong that will be filled with the
 *                        dimension of Z
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(1)
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */

#ifdef __cplusplus
}
#endif

#endif
