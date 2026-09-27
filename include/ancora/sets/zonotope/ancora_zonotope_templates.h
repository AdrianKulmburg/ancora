/*
 * ancora_zonotope_templates.h
 *
 * Description
 * -----------
 * Constructors for canonical and random zonotopes.
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

#ifndef ANCORA_ZONOTOPE_TEMPLATES_H
#define ANCORA_ZONOTOPE_TEMPLATES_H

#include "ancora/ancora_random.h"

#include "ancora/sets/zonotope/ancora_zonotope.h"

#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_zonotope_initRandom_uniform(ancora_zonotope *Z);
/* Fills an already-initialized zonotope with a random center and generators.
 * Each entry of the center vector and generator matrix is sampled from a
 * uniform distribution on [-1,1].
 *
 * INPUT:
 *      Z               : Pointer to an initialized ancora_zonotope instance
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*m*ANCORA_DEFAULT_PREC)
 *      where n is the dimension of the zonotope, m the number of generators.
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */

#ifdef __cplusplus
}
#endif

#endif /* ANCORA_ZONOTOPE_TEMPLATES_H */
