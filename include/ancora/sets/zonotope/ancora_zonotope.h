/*
 * ancora_zonotope.h
 *
 * Description
 * -----------
 * The ancora zonotope set type: { G*x+c | x \in [-1, 1]^m}. The generator
 * matrix G is stored as an ancora_mat, the center vector c as an ancora_vec.
 * No entry of G or c is allowed to be +-inf or NaN.
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

#ifndef ANCORA_ZONOTOPE_H
#define ANCORA_ZONOTOPE_H

#include "ancora/ancora_config.h"
#include "ancora/ancora_error.h"

#include "ancora/linalg/vectors/ancora_vec.h"
#include "ancora/linalg/matrices/ancora_mat.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The zonotope type. G is an ancora matrix holding the genrators, c an ancora
 * vector holding the center vector.
 */
typedef struct {
    ancora_mat G;
    ancora_vec c;
} ancora_zonotope;

#include "ancora/sets/zonotope/ancora_zonotope_constructorDestructor.h"
#include "ancora/sets/zonotope/ancora_zonotope_templates.h"
#include "ancora/sets/zonotope/ancora_zonotope_arithmetic.h"
#include "ancora/sets/zonotope/ancora_zonotope_containment.h"
#include "ancora/sets/zonotope/ancora_zonotope_properties.h"
#include "ancora/sets/zonotope/ancora_zonotope_duality.h"
#include "ancora/sets/zonotope/ancora_zonotope_randomPoints.h"

#ifdef __cplusplus
}
#endif

#endif
