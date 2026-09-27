/*
 * ancora_random.h
 *
 * Description
 * -----------
 * Random number generation that works in both the SAFE and FAST modes of
 * ancora. A single seed (set via ancora_random_setSeed) deterministically
 * drives both the FAST-mode xorshift64* generator and, in SAFE mode, the
 * FLINT RNG state used by ARB's random sampling routines.
 *
 * File Information
 * ----------------
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Authors:       Adrian Kulmburg
 *
 * License
 * -------
 * Copyright (c) 2026 Adrian Kulmburg <adrian.kulmburg@kit.edu>
 * SPDX-License-Identifier: MIT
 */

#ifndef ANCORA_RANDOM_H
#define ANCORA_RANDOM_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#include "ancora/ancora_config.h"
#include "ancora/ancora_error.h"

#if ANCORA_MODE == ANCORA_MODE_SAFE
#include <flint/arb.h>
#include <flint/flint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* NOTE: the generator state itself (the xorshift64* state, and, in SAFE
 * mode, the flint_rand_t state) is intentionally NOT exposed here. It
 * lives as private (static) state inside ancora_random.c -- the only way
 * to influence it is ancora_random_setSeed, which keeps both generators
 * in a single, deterministically-derived, internally-consistent state
 * rather than letting external code poke at either one directly. */

void ancora_random_setSeed(unsigned int seed);
/* Deterministically (re-)seeds ancora's random number generation from a
 * single unsigned int seed. See ancora_random.h for the full contract
 * (both generators seeded together, marks ancora's RNG as seeded, not
 * suitable for cryptographic use).
 * INPUT:
 *      seed            : Seed value
 *
 * OUTPUT:
 *      NONE
 *
 * RUNTIME:
 *      O(1)
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */

void ancora_random_ensureRandSeeded(void);
/* Checks if ancora's random generator(s) have already been seeded; if
 * not, seeds them from the current wall-clock time via
 * ancora_random_setSeed. See ancora_random.h for the full contract.
 * INPUT:
 *      NONE
 *
 * OUTPUT:
 *      NONE
 *
 * RUNTIME:
 *      O(1)
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_random_uniform(double a, double b, double *res);
/* Generates a double, uniformly distributed in [a, b].
 * INPUT:
 *      a, b            : Bounds of the interval, a <= b required
 *      res             : Pointer to a value to be set to the random number
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(1)
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_random_gaussian(double *res);
/* Generates a double, drawn from a standard Gaussian (mean 0, variance 1),
 * via a Box-Muller transform.
 * INPUT:
 *      res             : Pointer to a value to be set to the random number
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(1)
 *
 * REFERENCES:
 *      G.E.P. Box, M.E. Muller, "A Note on the Generation of Random
 *      Normal Deviates," The Annals of Mathematical Statistics, 29(2),
 *      1958. (the transform used here: two independent uniform samples
 *      combined via sqrt(-2 ln u1) cos(2 pi u2) into a standard-normal
 *      sample; only the cosine branch is used, since a single normal
 *      value is needed per call)
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */

#if ANCORA_MODE == ANCORA_MODE_SAFE

ancora_status ancora_random_uniform_arb(double a, double b, arb_t res);
/* Generates an arb_t, uniformly distributed in [a, b].
 * INPUT:
 *      a, b            : Bounds of the interval, a <= b required
 *      res             : Value to be set to the random number. Must already be
 *                        initialized.
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */

ancora_status ancora_random_gaussian_arb(arb_t res);
/* Generates an arb_t, drawn from a standard Gaussian (mean 0, variance 1), via
 * a Box-Muller transform built from rigorously-tracked ARB transcendental
 * functions applied to ARB uniform samples.
 * INPUT:
 *      res             : Value to be set to the random number. Must already be
 *                        initialized.
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(ANCORA_DEFAULT_PREC)
 *
 * REFERENCES:
 *      G.E.P. Box, M.E. Muller, "A Note on the Generation of Random
 *      Normal Deviates," The Annals of Mathematical Statistics, 29(2),
 *      1958.
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */

#endif

#ifdef __cplusplus
}
#endif

#endif /* ANCORA_RANDOM_H */
