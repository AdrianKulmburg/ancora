/*
 * testingTools.h
 *
 * Description
 * -----------
 * Minimal test helpers. Meant to be included by test .c files.
 *
 * File Information
 * ----------------
 * Created:       2026-09-20
 * Last modified: 2026-09-20
 * Authors:       Adrian Kulmburg
 *
 * License
 * -------
 * Copyright (c) 2026 Adrian Kulmburg <adrian.kulmburg@kit.edu>
 * SPDX-License-Identifier: MIT
 */

#ifndef TESTINGTOOLS_H
#define TESTINGTOOLS_H

#include <stdio.h>
#include <string.h>

#include "ancora/ancora.h" // Needed for ANCORA_ERROR_TEST_FAILED

// Checks if a condition is true, otherwise return that the test failed
#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
            return ANCORA_ERROR_TEST_FAILED;                             \
        }                                                                \
    } while (0)

// Checks if a string is what it should be, otherwise return that the test
// failed
#define CHECK_STR_EQ(actual, expected)                                   \
    do {                                                                 \
        const char *a_ = (actual);                                       \
        const char *e_ = (expected);                                     \
        if (a_ == NULL || strcmp(a_, e_) != 0) {                         \
            printf("  FAIL %s:%d: expected \"%s\", got \"%s\"\n",        \
                   __FILE__, __LINE__, e_, a_ ? a_ : "(null)");          \
            return ANCORA_ERROR_TEST_FAILED;                             \
        }                                                                \
    } while (0)

// Checks that expr returns the expected ancora_status; failures print the
// status names instead of bare numbers, and returns that the test failed
#define CHECK_STATUS(expr, expected)                                      \
    do {                                                                  \
        ancora_status got_ = (expr);                                      \
        ancora_status exp_ = (expected);                                  \
        if (got_ != exp_) {                                               \
            printf("  FAIL %s:%d: %s returned \"%s\", expected \"%s\"\n", \
                   __FILE__, __LINE__, #expr,                             \
                   ancora_status_str(got_), ancora_status_str(exp_));     \
            return ANCORA_ERROR_TEST_FAILED;                              \
        }                                                                 \
    } while (0)

#endif
