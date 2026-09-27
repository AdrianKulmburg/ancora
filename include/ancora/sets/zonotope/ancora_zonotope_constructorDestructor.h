/*
 * ancora_zonotope_constructorDestructor.h
 *
 * Description
 * -----------
 * Constructor and destructor for ancora_zonotope.
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

#ifndef ANCORA_ZONOTOPE_CONSTRUCTORDESTRUCTOR_H
#define ANCORA_ZONOTOPE_CONSTRUCTORDESTRUCTOR_H

#include "ancora/sets/zonotope/ancora_zonotope.h"

#ifdef __cplusplus
extern "C" {
#endif

ancora_status ancora_zonotope_init(ancora_zonotope *Z,
                                   const ancora_vec *c,
                                   const ancora_mat *G);
/* Initializes a zonotope from a given center vector c and generator matrix G.
 * INPUT:
 *      Z               : Pointer to an (uninitialized) ancora zonotope, to be
 *                        initialized
 *      c               : Center vector (length n)
 *      G               : Generator matrix (n x p)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*p*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_zonotope_free(ancora_zonotope *Z);
/* Frees a previously initialized ancora zonotope.
 * INPUT:
 *      Z               : Pointer to an initialized ancora_zonotope instance,
 *                        to be freed
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(dimension) in ANCORA_MODE_FAST mode.
 *      O(dimension*nrGenerators) in ANCORA_MODE_SAFE mode (one arb_clear per
 *      entry; independent of ANCORA_DEFAULT_PREC).
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):       Adrian Kulmburg
 */

#ifdef __cplusplus
}
#endif

#endif /* ANCORA_ZONOTOPE_CONSTRUCTORDESTRUCTOR_H */
