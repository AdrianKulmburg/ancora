/*
 * ancora_mat_getSet.c
 *
 * Description
 * -----------
 * Elementary operations for ancora_mat (e.g., getters and setters of the
 * (i,j)-th entry).
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

#include "ancora/linalg/matrices/ancora_mat_getSet.h"

ancora_status ancora_mat_set(ancora_mat *m,
                             slong i,
                             slong j,
#if ANCORA_MODE == ANCORA_MODE_SAFE
                             arb_t in
#elif ANCORA_MODE == ANCORA_MODE_FAST
                             double in
#endif
                             )
/* Sets the (i,j) entry of an ancora matrix.
 * INPUT:
 *      m               : Pointer to an initialized ancora_mat instance
 *      i, j            : Row/column index (0-based)
 *      in              : Value to copy into (i,j)-th entry of m (arb_t or
 *                        double, depending on ANCORA_MODE)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-21
 * Last modified: 2026-09-21
 * Author(s):     Adrian Kulmburg
 */
{
    // Check that the matrix is valid
    if (m == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer m is NULL; it should point to a valid ancora_mat instance.");
    }

    // Check that i and j make sense
    if (i < 0)
    {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Index i is negative; it should be nonnegative.");
    }
    else if (j < 0)
    {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Index j is negative; it should be nonnegative.");
    }
    else if (i>= m->nrows)
    {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Index i out of bounds; it is %ld, it should be <%ld.", (long)i, (long)m->nrows);
    }
    else if (j>= m->ncols)
    {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Index j out of bounds; it is %ld, it should be <%ld.", (long)j, (long)m->ncols);
    }

    // Do the actual writing operation
#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_set(arb_mat_entry(m->repr, i, j), in);
#elif ANCORA_MODE == ANCORA_MODE_FAST
    m->repr[i * m->ncols + j] = in;
#endif
    return ANCORA_OK;
}

ancora_status ancora_mat_get(const ancora_mat *m,
                             slong i,
                             slong j,
#if ANCORA_MODE == ANCORA_MODE_SAFE
                             arb_t out
#elif ANCORA_MODE == ANCORA_MODE_FAST
                             double *out
#endif
                             )
/* Reads the (i,j)-th entry of an ancora matrix.
 *
 * INPUT:
 *      m               : Pointer to an initialized ancora_mat instance
 *      i, j            : Row/column index (0-based)
 *
 *      out             : Value to read into the (i,j)-th entry of m (arb_t
 *                        or *double depending on ANCORA_MODE)
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-21
 * Last modified: 2026-09-21
 * Author(s):     Adrian Kulmburg
 */
{
    // Check that the matrix and output pointer are valid
    if (m == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer m is NULL; it should point to a valid ancora_mat instance.");
    }
    if (out == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer out is NULL; it should point to a valid double.");
    }

    // Check that i and j make sense
    if (i < 0) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Index i is negative; it should be nonnegative.");
    }
    if (j < 0) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Index j is negative; it should be nonnegative.");
    }
    if (i >= m->nrows) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Index i out of bounds; it is %ld, it should be <%ld.", (long)i, (long)m->nrows);
    }
    if (j >= m->ncols) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Index j out of bounds; it is %ld, it should be <%ld.", (long)j, (long)m->ncols);
    }

    // Do the actual reading operation
#if ANCORA_MODE == ANCORA_MODE_SAFE
    arb_set(out, arb_mat_entry(m->repr, i, j));
#elif ANCORA_MODE == ANCORA_MODE_FAST
    *out = m->repr[i * m->ncols + j];
#endif
    return ANCORA_OK;
}

ancora_status ancora_mat_getColumn(const ancora_mat *m,
                             const slong i,
                             ancora_vec *out)
/* Reads the i-th column of an ancora matrix.
 *
 * INPUT:
 *      m               : Pointer to an initialized ancora_mat instance
 *      i               : Column index (0-based)
 *
 *      out             : Value to read into the i-th column of m
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*ANCORA_DEFAULT_PREC)
 *      where n is the number of rows of m
 *
 * Created:       2026-09-25
 * Last modified: 2026-09-25
 * Author(s):     Adrian Kulmburg
 */
{
    // Check that the matrix and output pointer are valid
    if (m == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer m is NULL; it should point to a valid ancora_mat instance.");
    }
    if (out == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer out is NULL; it should point to a valid double.");
    }

    // Check that i makes sense
    if (i < 0) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Index i is negative; it should be nonnegative.");
    }
    if (i >= m->ncols) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Index i out of bounds; it is %ld, it should be <%ld.", (long)i, (long)m->nrows);
    }

    // Verify that out is a vector
    bool isVector;
    ANCORA_TRY(ancora_mat_isVector(out, &isVector));
    if (!isVector)
    {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer out is not a vector; it should point to a valid ancora_vec instance.");
    }

    // Dimension check
    if (m->nrows != out->nrows) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "out has dimension %ld, matrix m has %ld rows; they need to be the same.",
                      (long)out->nrows, (long)m->nrows);
    }

    // Do the actual reading operation
    for (slong j = 0; j < out->nrows; j++)
    {
#if ANCORA_MODE == ANCORA_MODE_SAFE
            arb_set(arb_mat_entry(out->repr, j, 0), arb_mat_entry(m->repr, j, i));
#elif ANCORA_MODE == ANCORA_MODE_FAST
            out->repr[j] = m->repr[j * m->ncols + i];
#endif
        }

    return ANCORA_OK;

}
