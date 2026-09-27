/*
 * ancora.h
 *
 * Description
 * -----------
 * Main header of ancora importing all other modules.
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


#ifndef ANCORA_H
#define ANCORA_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ancora/ancora_config.h"
#include "ancora/ancora_error.h"
#include "ancora/ancora_random.h"

#include "ancora/linalg/ancora_linalg.h"
#include "ancora/logic/ancora_logic.h"

#include "ancora/sets/interval/ancora_interval.h"
#include "ancora/sets/zonotope/ancora_zonotope.h"

#ifdef __cplusplus
}
#endif

#endif

