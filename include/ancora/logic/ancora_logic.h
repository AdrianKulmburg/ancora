/*
 * ancora_logic.h
 *
 * Description
 * -----------
 * Defines a new variable for truth (YES, NO, UNKNOWN) that also supports
 * probabilistic statements
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

#ifndef ANCORA_LOGIC_H
#define ANCORA_LOGIC_H

#include <stdio.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Definition of an ancora truth: a boolean (YES or NO), combined with a
 * certificate (CERTAIN or UNKNOWN), and probabilistic estimates (YES with
 * probability X, NO with probability X).
 */
typedef struct ancora_truth {
    bool verdict;
    bool certitude;

    double probability_yes;
    double probability_no;
} ancora_truth;

/* The verdict is meaningless if it is not certain; for ancora to give a clear
 * 'YES' or 'NO', there needs to hold certitude=true.
 */
// YES case
bool ancora_yes(ancora_truth truth);
// NO case
bool ancora_no(ancora_truth truth);

/* We also define some setters, that can make it a bit easier to instantiate
 * stuff.
 */
// YES case
void ancora_setYes(ancora_truth *truth);

// NO case
void ancora_setNo(ancora_truth *truth);

// UNKNOWN case
void ancora_setUnknown(ancora_truth *truth);

#ifdef __cplusplus
}
#endif

#endif

