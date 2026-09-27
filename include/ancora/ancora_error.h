/*
 * ancora_error.h
 *
 * Description
 * -----------
 * Lists all the error codes of ancora, and provides some basic error handling
 * routines, including a traceback.
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

#ifndef ANCORA_ERROR_H
#define ANCORA_ERROR_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Let GCC/Clang check printf-style format strings of our error functions.
 * This only changes something if compiling with -Wall or -Wformat, and can
 * ensure that the right print routines are used.
 */
#if defined(__GNUC__) || defined(__clang__)
#define ANCORA_PRINTF(fmt_idx, first_arg) \
    __attribute__((format(printf, fmt_idx, first_arg)))
#else
#define ANCORA_PRINTF(fmt_idx, first_arg)
#endif

/* Status codes, used throughout ancora.
 * NOTE: Don't forget to update the .c file as well, if you add error codes
 * here!
 */
typedef enum ancora_status {
    ANCORA_OK = 0, // Everything went well
    ANCORA_ERROR_ALLOC, // Error allocating memory
    ANCORA_ERROR_DIM_MISMATCH, // Dimension mismatch between vectors/matrices/sets
    ANCORA_ERROR_INVALID_ARG, // Invalid argument
    ANCORA_ERROR_NOT_IMPLEMENTED, // Function is not yet implemented
    ANCORA_ERROR_UNSAFE, // When using a function would potentially be unsafe (particularly relevant in SAFE mode)
    ANCORA_ERROR_TEST_FAILED, // Test failed; usually not critical, but needs to be recorded
    ANCORA_ERROR_GPU_LAUNCH // Error trying to use the GPU - perhaps a missing driver/CUDA?
} ancora_status;

/* Returns a static description of a status code; never NULL.
 */
const char *ancora_status_str(ancora_status s);

/* Last-error message (thread-local).
 * The message is only meaningful right after a failure, and is valid until
 * the next error is set on the same thread.
 */
void ancora_set_last_error(const char *fmt, ...) ANCORA_PRINTF(1, 2);
const char *ancora_get_last_error(void);
ancora_status ancora_get_last_errorCode(void);
void ancora_set_last_errorCode(ancora_status status);

/* Set up traceback.
 */
#define ANCORA_TRACE_MAX 100 // Maximum size of the trace

// If an error is noticed -> call ANCORA_ERROR, which adds to the traceback and
// leaves the function
void ancora_error_begin(const char *file, int line, const char *func,
                        const char *message, ...) ANCORA_PRINTF(4, 5);
#define ANCORA_ERROR(status, ...)                                       \
    do {                                                                \
        ancora_error_begin(__FILE__, __LINE__, __func__, __VA_ARGS__);  \
        ancora_set_last_errorCode(status);                              \
        return status;                                                  \
    } while (0)

// Once an error has been reported, we need to go up the chain, always including
// the file names etc. for easier debugging
void ancora_error_push(const char *file, int line, const char *func);
#define ANCORA_TRY(expr)                                                \
    do {                                                                \
        ancora_status s_ = expr;                                        \
        if (s_ != ANCORA_OK) {                                          \
            ancora_error_push(__FILE__, __LINE__, __func__);            \
            return s_;                                                  \
        }                                                               \
    } while (0)

// Once we are at the top, we can print the error
void ancora_print_trace(FILE *out);


#ifdef __cplusplus
}
#endif

#endif
