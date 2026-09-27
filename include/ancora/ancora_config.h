/*
 * ancora_config.h
 *
 * Description
 * -----------
 * Sets some of the core parameters of Ancora, e.g., the precision of floating
 * point operations.
 *
 * File Information
 * ----------------
 * Created:       2026-09-19
 * Last modified: 2026-09-19
 * Authors:       Adrian Kulmburg
 *
 * License
 * -------
 * Copyright (c) 2026 Adrian Kulmburg <adrian.kulmburg@kit.edu>
 * SPDX-License-Identifier: MIT
 */


#ifndef ANCORA_CONFIG_H
#define ANCORA_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

/* Ancora mode: Should it use (safe) FLINT matrices, that can track floating-
 * point errors, or regular matrices that can be processed faster?
 */

#define ANCORA_MODE_SAFE 0
#define ANCORA_MODE_FAST 1

/* Default bit precision for floating-point operations, assuming the SAFE mode
 * is enabled.
 */
#define ANCORA_DEFAULT_PREC 128

/* FLINT provides the `slong` type (a signed machine word). In FAST mode
 * FLINT headers are never included, so we define slong ourselves as a
 * plain `long`. In SAFE mode the FLINT-provided definition is used.
 */
#if ANCORA_MODE == ANCORA_MODE_FAST
#ifndef slong
#define slong long
#endif
#endif

/* If FLINT is not active, at some point we need to decide what a 'small enough'
 * error is. We set this here globally.
 */
#if ANCORA_MODE == ANCORA_MODE_FAST
#ifndef ANCORA_TOL
#define ANCORA_TOL 1e-6
// For a very large number, just take 1/ANCORA_TOL
#endif
#endif

#ifdef __cplusplus
}
#endif

#endif

