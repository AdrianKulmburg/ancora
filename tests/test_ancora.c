/*
 * test_ancora.c
 *
 * Description
 * -----------
 * Launches all tests for ancora.
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

#include <stdio.h>

#include "test_ancora_error.h"


int main(void)
{
    unsigned int failed = 0;
    ancora_status status = ANCORA_OK;

    if (test_ancora_error() != ANCORA_OK)
    {
        failed++;
    }

    printf("%d\n", failed);

    if (failed != 0)
    {
        printf("%d tests failed in %s.", failed, __func__);
        return ANCORA_ERROR_TEST_FAILED;
    }

    return 0;
}
