/*
 * test_ancora_error.h
 *
 * Description
 * -----------
 * Tests basic functionality of ancora_error.c/.h.
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

#ifndef TEST_ANCORA_ERROR_H
#define TEST_ANCORA_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <string.h>

#include "ancora/ancora.h"
#include "testingTools.h"

// Testing whether the error gets passed to the higher level
static ancora_status test_ancora_errorPassing(void);

// Main function launching all the tests
ancora_status test_ancora_error(void);

#ifdef __cplusplus
}
#endif

#endif
