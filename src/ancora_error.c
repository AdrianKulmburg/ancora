#include <stdarg.h>
#include <stdio.h>

#include "ancora/ancora_error.h"

#if defined(_MSC_VER)
#define ANCORA_THREAD_LOCAL __declspec(thread)
#else
#define ANCORA_THREAD_LOCAL _Thread_local
#endif

/* Status code descriptions.
 * No default: label, so the compiler warns (-Wswitch) if a new status is
 * added to the enum but not handled here.
 * NOTE: Don't forget to update the .h file as well, if you add error codes
 * here!
 */
const char *ancora_status_str(ancora_status s)
{
    switch (s) {
    case ANCORA_OK:                   return "Success";
    case ANCORA_ERROR_ALLOC:          return "Memory allocation failed";
    case ANCORA_ERROR_DIM_MISMATCH:   return "Dimension mismatch";
    case ANCORA_ERROR_INVALID_ARG:    return "Invalid input argument";
    case ANCORA_ERROR_NOT_IMPLEMENTED:return "Feature not yet implemented";
    case ANCORA_ERROR_UNSAFE:         return "Unsafe functionality";
    case ANCORA_ERROR_TEST_FAILED:    return "Test failed";
    case ANCORA_ERROR_GPU_LAUNCH:     return "GPU launch failed - perhaps a missing driver?";
    }
    return "Unknown error";
}

/* Set up traceback.
 */
// Last error message
static ANCORA_THREAD_LOCAL char ancora_last_error[1024] = {0};
// Last error code
static ANCORA_THREAD_LOCAL ancora_status ancora_last_errorCode = ANCORA_OK;

void ancora_set_last_error(const char *message, ...)
{
    va_list ap;
    va_start(ap, message);
    vsnprintf(ancora_last_error, sizeof(ancora_last_error), message, ap);
    va_end(ap);
}

void ancora_set_last_errorCode(ancora_status status)
{
    ancora_last_errorCode = status;
}

// Gives the last error message; not needed in the traceback, but could be
// useful for individual functions
const char *ancora_get_last_error(void)
{
    return ancora_last_error;
}

ancora_status ancora_get_last_errorCode(void)
{
    return ancora_last_errorCode;
}

// Struct collecting where the error originates from
typedef struct {
    const char *file;
    int line;
    const char *func;
} ancora_frame;

// The actual trace
static ANCORA_THREAD_LOCAL ancora_frame ancora_trace[ANCORA_TRACE_MAX];

// Lengths of the trace
static ANCORA_THREAD_LOCAL int ancora_trace_len = 0;      // stored frames
static ANCORA_THREAD_LOCAL int ancora_trace_total = 0;    // incl. dropped

// Function that adds frame to trace
void ancora_error_push(const char *file, int line, const char *func)
{
    if (ancora_trace_len < ANCORA_TRACE_MAX)
        ancora_trace[ancora_trace_len++] = (ancora_frame){file, line, func};
    ancora_trace_total++;
}

// Function that is called when using ANCORA_ERROR
void ancora_error_begin(const char *file, int line, const char *func,
                        const char *message, ...)
{
    va_list ap;
    va_start(ap, message);
    vsnprintf(ancora_last_error, sizeof(ancora_last_error), message, ap);
    va_end(ap);

    ancora_trace_len = 0;      // new error: discard the old trace
    ancora_trace_total = 0;
    ancora_error_push(file, line, func);
}

// Print the full trace
void ancora_print_trace(FILE *out)
{
    fprintf(out, "ancora error: %s\n", ancora_last_error);
    for (int i = 0; i < ancora_trace_len; i++)
        fprintf(out, "  #%d %s (%s:%d)\n", i, ancora_trace[i].func,
                ancora_trace[i].file, ancora_trace[i].line);
    if (ancora_trace_total > ancora_trace_len)
        fprintf(out, "  ... %d more\n", ancora_trace_total - ancora_trace_len);
}
