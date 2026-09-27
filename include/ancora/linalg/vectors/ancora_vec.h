/*
 * ancora_vec.h
 *
 * Description
 * -----------
 * Vectors for ancora-internal computations, implemented both for the FAST and
 * SAFE versions of ancora. Since vectors are just 1-column matrices, most
 * methods are just direct calls to the linalg/matrices methods.
 * IMPORTANT: Do **NOT** write an entry directly into a vector, instead use
 * ancora_vec_get, otherwise some nasty errors might happen with the GPU.
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

#ifndef ANCORA_VEC_H
#define ANCORA_VEC_H

// ancora_vec is just an alias for ancora_mat
#define ancora_vec ancora_mat

#include "ancora/ancora_config.h"
#include "ancora/ancora_error.h"

/* FLINT/ARB headers are only pulled in when building in FLINT mode. */
#if ANCORA_MODE == ANCORA_MODE_SAFE
#include <flint/arb.h>
#include <flint/arb_mat.h>
#include <flint/flint.h>
#endif

#include "ancora/linalg/matrices/ancora_mat.h"

#include "ancora/linalg/vectors/ancora_vec_arithmetic.h"
#include "ancora/linalg/vectors/ancora_vec_concatenation.h"
#include "ancora/linalg/vectors/ancora_vec_constructorDestructor.h"
#include "ancora/linalg/vectors/ancora_vec_getSet.h"
#include "ancora/linalg/vectors/ancora_vec_norms.h"
#include "ancora/linalg/vectors/ancora_vec_templates.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif
