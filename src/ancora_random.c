/*
 * ancora_random.c
 *
 * Description
 * -----------
 * Random number generation that works in both the SAFE and FAST modes of
 * ancora. See ancora_random.h.
 *
 * File Information
 * ----------------
 * Created:       2026-09-24
 * Last modified: 2026-09-28
 * Authors:       Adrian Kulmburg
 *
 * License
 * -------
 * Copyright (c) 2026 Adrian Kulmburg <adrian.kulmburg@kit.edu>
 * SPDX-License-Identifier: MIT
 */

#include "ancora/ancora_random.h"

/*******************************************************************************
 * Auxiliary functions - Definitions
 ******************************************************************************/
/* --------------------------------------------------------------------
 * Private state. Deliberately NOT exposed in ancora_random.h -- the only
 * way to influence it is ancora_random_setSeed, so both generators stay
 * internally consistent (one seed -> both states), and nothing outside
 * this file can desync them by poking at one but not the other.
 */
static uint64_t ancora_xorshift_state;
static bool ancora_rand_seeded = false; /* covers BOTH generators */

#if ANCORA_MODE == ANCORA_MODE_SAFE
static flint_rand_t ancora_flint_rand_state;
#endif

static uint64_t splitmix64Next(uint64_t *state);
static uint64_t xorshiftNext(void);
/*******************************************************************************
 * Auxiliary functions - Definitions END
 ******************************************************************************/

double xorshiftUnit(void)
/* Draws a uniform double in [0, 1), from the top 53 bits of xorshiftNext's
 * output (the highest-quality bits of a xorshift64* draw - see
 * xorshiftNext's REFERENCES for why the top bits, not the bottom ones,
 * are used here). 53 bits matches a double's mantissa width exactly, so
 * this covers the full representable range of doubles in [0, 1) at full
 * precision, rather than leaving some representable doubles unreachable.
 * INPUT:
 *      NONE (operates on the private ancora_xorshift_state via
 *      xorshiftNext)
 *
 * OUTPUT:
 *      double          : A uniform sample in [0, 1)
 *
 * RUNTIME:
 *      O(1)
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */
{
    return (double)(xorshiftNext() >> 11) * (1.0 / 9007199254740992.0); /* 2^53 */
}

void ancora_random_setSeed(unsigned int seed)
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
{
    uint64_t sm_state = (uint64_t)seed;

    uint64_t xs = splitmix64Next(&sm_state);
    /* xorshift64* requires a nonzero state; SplitMix64 returning exactly
     * zero here is astronomically unlikely but not provably impossible,
     * so guard it rather than assume. */
    ancora_xorshift_state = (xs != 0) ? xs : 0x9E3779B97F4A7C15ULL;

#if ANCORA_MODE == ANCORA_MODE_SAFE
    uint64_t f1 = splitmix64Next(&sm_state);
    uint64_t f2 = splitmix64Next(&sm_state);
    /* FLINT 3.x renamed the old camelCase rand-state API
     * (flint_randinit/flint_randclear/flint_randseed, now deprecated) to
     * the flint_rand_* names below. */
    flint_rand_clear(ancora_flint_rand_state);
    flint_rand_init(ancora_flint_rand_state);
    flint_rand_set_seed(ancora_flint_rand_state, (ulong)f1, (ulong)f2);
#endif

    ancora_rand_seeded = true;
}

void ancora_random_ensureRandSeeded(void)
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
{
    if (!ancora_rand_seeded) {
        ancora_random_setSeed((unsigned int)time(NULL));
    }
}

// TODO: Implement a similar function that takes and outputs arb_t, and correct
// ancora_interval_randomPoints for SAFE mode
ancora_status ancora_random_uniform(double a, double b, double *res)
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
{
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid double.");
    }
    if (a > b) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Invalid interval: a (%g) must be <= b (%g).", a, b);
    }
    else if (a == b)
    {
        *res = a;
    }
    else
    {
        ancora_random_ensureRandSeeded();

        double u = xorshiftUnit();
        *res = a + u * (b - a);
    }

    return ANCORA_OK;
}

ancora_status ancora_random_gaussian(double *res)
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
{
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid double.");
    }

    ancora_random_ensureRandSeeded();

    // u1 must be in (0, 1], not [0, 1), to avoid log(0).
    double u1;
    do {
        u1 = xorshiftUnit();
    } while (u1 == 0.0);
    double u2 = xorshiftUnit();

    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    *res = r * cos(theta);
    return ANCORA_OK;
}

#if ANCORA_MODE == ANCORA_MODE_SAFE

ancora_status ancora_random_uniform_arb(double a, double b, arb_t res)
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
{
    if (a > b) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Invalid interval: a (%g) must be <= b (%g).", a, b);
    }

    ancora_random_ensureRandSeeded();

    // res <- uniform draw in [0, 1] at ANCORA_DEFAULT_PREC bits
    arb_urandom(res, ancora_flint_rand_state, ANCORA_DEFAULT_PREC);

    /* Rescale into [a, b]. a and b enter as untracked doubles at the
     * MATLAB/FAST boundary -- they do not carry rigorous interval
     * semantics themselves, only the sampled value's arithmetic is
     * rigorously precision-tracked from here on. */
    arb_t range, offset;
    arb_init(range);
    arb_init(offset);
    arb_set_d(range, b - a);
    arb_set_d(offset, a);

    arb_mul(res, res, range, ANCORA_DEFAULT_PREC);
    arb_add(res, res, offset, ANCORA_DEFAULT_PREC);

    arb_clear(range);
    arb_clear(offset);
    return ANCORA_OK;
}

ancora_status ancora_random_gaussian_arb(arb_t res)
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
{
    ancora_random_ensureRandSeeded();

    arb_t u1, u2, r, theta, pi;
    arb_init(u1);
    arb_init(u2);
    arb_init(r);
    arb_init(theta);
    arb_init(pi);

    /* u1 must be in (0, 1], not [0, 1), to avoid log(0); redraw on the
     * (measure-zero, but not impossible with a finite-precision PRNG)
     * exact-zero case. */
    do {
        arb_urandom(u1, ancora_flint_rand_state, ANCORA_DEFAULT_PREC);
    } while (arb_is_zero(u1));
    arb_urandom(u2, ancora_flint_rand_state, ANCORA_DEFAULT_PREC);

    arb_const_pi(pi, ANCORA_DEFAULT_PREC);

    // r = sqrt(-2 ln(u1))
    arb_log(r, u1, ANCORA_DEFAULT_PREC);
    arb_mul_si(r, r, -2, ANCORA_DEFAULT_PREC);
    arb_sqrt(r, r, ANCORA_DEFAULT_PREC);

    // theta = 2 pi u2
    arb_mul(theta, u2, pi, ANCORA_DEFAULT_PREC);
    arb_mul_si(theta, theta, 2, ANCORA_DEFAULT_PREC);

    // res = r cos(theta)
    arb_cos(res, theta, ANCORA_DEFAULT_PREC);
    arb_mul(res, res, r, ANCORA_DEFAULT_PREC);

    arb_clear(u1);
    arb_clear(u2);
    arb_clear(r);
    arb_clear(theta);
    arb_clear(pi);
    return ANCORA_OK;
}

#endif

/*******************************************************************************
 * Auxiliary functions - Source
 ******************************************************************************/

static uint64_t splitmix64Next(uint64_t *state)
/* One step of the SplitMix64 generator, used only to expand a single
 * unsigned int seed into several well-mixed 64-bit values (one for the
 * xorshift64* state, two more for the FLINT RNG state in SAFE mode) --
 * NOT used as ancora's actual random source, only as a seed spreader, so
 * that e.g., seed=1 and seed=2 don't produce suspiciously similar initial
 * xorshift64* /FLINT states.
 * INPUT:
 *      state           : Pointer to the SplitMix64 state, advanced in
 *                        place
 *
 * OUTPUT:
 *      uint64_t        : The next SplitMix64 output
 *
 * RUNTIME:
 *      O(1)
 *
 * REFERENCES:
 *      G. Steele, D. Lea, C. Flood, "Fast Splittable Pseudorandom Number
 *      Generators," OOPSLA 2014.
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */
{
    *state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = *state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static uint64_t xorshiftNext(void)
/* One step of the xorshift64* generator: advances the private xorshift
 * state and returns a scrambled 64-bit output derived from it. This is
 * ancora's FAST-mode (and mode-independent, double-producing) random
 * source. NOT suitable for cryptographic use.
 * INPUT:
 *      NONE (operates on the private ancora_xorshift_state)
 *
 * OUTPUT:
 *      uint64_t        : The next pseudorandom 64-bit value
 *
 * RUNTIME:
 *      O(1)
 *
 * REFERENCES:
 *      G. Marsaglia, "Xorshift RNGs," Journal of Statistical Software,
 *      8(14), 2003. (the base xorshift generator: the three shift/XOR
 *      steps)
 *      S. Vigna, "An experimental exploration of Marsaglia's xorshift
 *      generators, scrambled," ACM Transactions on Mathematical
 *      Software, 42(4), 2016 (also arXiv:1402.6246). (the "*" scrambling
 *      multiplication that fixes xorshift's weak low-order bits; the
 *      shift constants 12/25/27 and multiplier 0x2545F4914F6CDD1D used
 *      here are exactly the xorshift64* variant given in that paper)
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */
{
    ancora_xorshift_state ^= ancora_xorshift_state >> 12;
    ancora_xorshift_state ^= ancora_xorshift_state << 25;
    ancora_xorshift_state ^= ancora_xorshift_state >> 27;
    return ancora_xorshift_state * 0x2545F4914F6CDD1DULL;
}
