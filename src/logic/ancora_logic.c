/*
 * ancora_logic.c
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

#include "ancora/logic/ancora_logic.h"

/* The verdict is meaningless if it is not certain; for ancora to give a clear
 * 'YES' or 'NO', there needs to hold certitude=true.
 */
// YES case
bool ancora_yes(const ancora_truth truth)
{
    if (truth.verdict && truth.certitude)
    {
        return true;
    }
    else
    {
        return false;
    }
}

// NO case
bool ancora_no(const ancora_truth truth)
{
    if (!truth.verdict && truth.certitude)
    {
        return true;
    }
    else
    {
        return false;
    }
}

/* We also define some setters, that can make it a bit easier to instantiate
 * stuff.
 */
// YES case
void ancora_setYes(ancora_truth *truth)
{
    truth->verdict = true;
    truth->certitude = true;

    truth->probability_yes = 1.0;
    truth->probability_no = 0.0;
}

// NO case
void ancora_setNo(ancora_truth *truth)
{
    truth->verdict = false;
    truth->certitude = true;

    truth->probability_yes = 0.0;
    truth->probability_no = 1.0;
}

// UNKNOWN case
void ancora_setUnknown(ancora_truth *truth)
{
    truth->verdict = true; // Doesn't matter here
    truth->certitude = false;

    truth->probability_yes = 0.0;
    truth->probability_no = 0.0;
}
