/*
 * test_ancora_error.c
 *
 * Description
 * -----------
 * Tests basic functionality of ancora_error.c/.h.
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

#include "test_ancora_error.h"

/*******************************************************************************
 * Auxiliary functions - Definitions
 ******************************************************************************/
static ancora_status produceError(void);
static ancora_status produceError_withValue(void);
static ancora_status produceError_withString(const char *msg);

static ancora_status passError(void);
static ancora_status passError_withValue(void);
static ancora_status passError_withString(const char *msg);
/*******************************************************************************
 * Auxiliary functions - Definitions END
 ******************************************************************************/

// Testing whether the error gets passed to the higher level
static ancora_status test_ancora_errorPassing(void)
{
    ancora_status status = ANCORA_OK;

    // If no error has been reported, no error trace should be present
    CHECK(ancora_get_last_errorCode() == ANCORA_OK);
    CHECK_STR_EQ(ancora_get_last_error(), "");

    // Check that a simple error message works
    // Check that ANCORA_TRY works, by calling upon the wrapper
    status = passError();
    CHECK(status == ANCORA_ERROR_ALLOC);
    // Also check that ancora_last_error and ancora_last_errorCode habe been set
    // properly
    CHECK(ancora_get_last_errorCode() == ANCORA_ERROR_ALLOC);
    CHECK_STR_EQ(ancora_get_last_error(), "Simple error message.");

    // Continue with an error message with a value
    // Check that ANCORA_TRY works, by calling upon the wrapper
    status = passError_withValue();
    CHECK(status == ANCORA_ERROR_INVALID_ARG);
    // Also check that ancora_last_error and ancora_last_errorCode habe been set
    // properly
    CHECK(ancora_get_last_errorCode() == ANCORA_ERROR_INVALID_ARG);
    CHECK_STR_EQ(ancora_get_last_error(), "Error message with a value 42.");

    // Finally, check with a custom message
    // Check that ANCORA_TRY works, by calling upon the wrapper
    status = passError_withString("test");
    CHECK(status == ANCORA_ERROR_INVALID_ARG);
    // Also check that ancora_last_error and ancora_last_errorCode habe been set
    // properly
    CHECK(ancora_get_last_errorCode() == ANCORA_ERROR_INVALID_ARG);
    CHECK_STR_EQ(ancora_get_last_error(), "Error message with a string test.");

    return ANCORA_OK;
}


// Main function launching all the tests
ancora_status test_ancora_error(void)
{
    unsigned int failed = 0;
    ancora_status status = ANCORA_OK;

    if (test_ancora_errorPassing() != ANCORA_OK)
    {
        failed++;
    }

    if (failed != 0)
    {
        printf("%d tests failed in %s.", failed, __func__);
        return ANCORA_ERROR_TEST_FAILED;
    }
    return ANCORA_OK;
}

/*******************************************************************************
 * Auxiliary functions - Source
 ******************************************************************************/


/* Code producing errors on purpose, to test that everything gets reported
 * right. They all should be rather self-explanatory.
 */
static ancora_status produceError(void)
{
    ANCORA_ERROR(ANCORA_ERROR_ALLOC, "Simple error message.");
}
static ancora_status produceError_withValue(void)
{
    ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Error message with a value %d.", 42);
}

static ancora_status produceError_withString(const char *msg)
{
    ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Error message with a string %s.", msg);
}

// Functions that pass the errors to the upper level using ANCORA_TRY
static ancora_status passError(void)
{
    ANCORA_TRY(produceError());
    return ANCORA_OK;
}

static ancora_status passError_withValue(void)
{
    ANCORA_TRY(produceError_withValue());
    return ANCORA_OK;
}

static ancora_status passError_withString(const char *msg)
{
    ANCORA_TRY(produceError_withString(msg));
    return ANCORA_OK;
}
