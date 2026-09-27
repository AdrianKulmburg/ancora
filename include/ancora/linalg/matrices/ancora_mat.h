/*
 * ancora_mat.h
 *
 * Description
 * -----------
 * Matrices for ancora-internal computations, implemented both for the FAST and
 * SAFE versions of ancora.
 * IMPORTANT: Do **NOT** write an entry directly into a matrix, instead use
 * ancora_mat_get, otherwise some nasty errors might happen with the GPU.
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

#ifndef ANCORA_MAT_H
#define ANCORA_MAT_H

#include <stdlib.h>

#include "ancora/ancora_config.h"
#include "ancora/ancora_error.h"

/* FLINT/ARB headers are only pulled in when building in FLINT mode. */
#if ANCORA_MODE == ANCORA_MODE_SAFE
#include <flint/arb.h>
#include <flint/arb_mat.h>
#include <flint/flint.h>
#endif

typedef struct {
#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_mat_t repr;
#elif ANCORA_MODE == ANCORA_MODE_FAST
    double *repr; /* flat, row-major, length nrows*ncols */
#endif
    slong nrows;
    slong ncols;
} ancora_mat;

#include "ancora/linalg/matrices/ancora_mat_arithmetic.h"
#include "ancora/linalg/matrices/ancora_mat_concatenation.h"
#include "ancora/linalg/matrices/ancora_mat_constructorDestructor.h"
#include "ancora/linalg/matrices/ancora_mat_getSet.h"
#include "ancora/linalg/matrices/ancora_mat_properties.h"
#include "ancora/linalg/matrices/ancora_mat_templates.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif
