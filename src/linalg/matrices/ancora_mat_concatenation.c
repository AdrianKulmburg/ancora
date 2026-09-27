/*
 * ancora_mat_concatenation.c
 *
 * Description
 * -----------
 * Concatenation operations for matrices: horizontal, vertical, and
 * block-diagonal.
 *
 * File Information
 * ----------------
 * Created:       2026-09-22
 * Last modified: 2026-09-22
 * Authors:       Adrian Kulmburg
 *
 * License
 * -------
 * Copyright (c) 2026 Adrian Kulmburg <adrian.kulmburg@kit.edu>
 * SPDX-License-Identifier: MIT
 */

#include "ancora/linalg/matrices/ancora_mat_concatenation.h"

ancora_status ancora_mat_hcat(ancora_mat *res, const ancora_mat *a, const ancora_mat *b)
/* Horizontal concatenation: res = [a b].
 * INPUT:
 *      res             : Result matrix, already initialized as
 *                        (n x (k1+k2)), where a is (n x k1), b is (n x k2)
 *      a               : Left block
 *      b               : Right block
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(n*(k1+k2))
 *      In ANCORA_MODE_SAFE mode, O(ANCORA_DEFAULT_PREC) worst case per
 *      entry copied, so O(n*(k1+k2)*ANCORA_DEFAULT_PREC) worst case overall.
 *
 * Created:       2026-09-22
 * Last modified: 2026-09-22
 * Author(s):     Adrian Kulmburg
 */
{
    if (res == NULL)
    {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_mat instance.");
    }
    if (a == NULL)
    {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer a is NULL; it should point to a valid ancora_mat instance.");
    }
    if (b == NULL)
    {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer b is NULL; it should point to a valid ancora_mat instance.");
    }
    if (a->nrows != b->nrows)
    {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Matrix a has %ld rows, matrix b has %ld rows; they need to match for horizontal concatenation.",
                      (long)a->nrows, (long)b->nrows);
    }
    if (res->nrows != a->nrows || res->ncols != a->ncols + b->ncols)
    {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Result should be (%ld x %ld), but res is (%ld x %ld).",
                      (long)a->nrows, (long)(a->ncols + b->ncols), (long)res->nrows, (long)res->ncols);
    }
    if (res == a || res == b)
    {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "res must not alias a or b for ancora_mat_hcat.");
    }

    slong n = a->nrows;
    slong k1 = a->ncols;
    slong k2 = b->ncols;

#if ANCORA_MODE == ANCORA_MODE_SAFE
    for (slong i = 0; i < n; i++)
    {
        for (slong j = 0; j < k1; j++)
        {
            arb_set(arb_mat_entry(res->repr, i, j), arb_mat_entry(a->repr, i, j));
        }
        for (slong j = 0; j < k2; j++)
        {
            arb_set(arb_mat_entry(res->repr, i, k1 + j), arb_mat_entry(b->repr, i, j));
        }
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
    for (slong i = 0; i < n; i++) {
        memcpy(&res->repr[i * (k1 + k2)], &a->repr[i * k1], (size_t)k1 * sizeof(double));
        memcpy(&res->repr[i * (k1 + k2) + k1], &b->repr[i * k2], (size_t)k2 * sizeof(double));
    }
#endif
    return ANCORA_OK;
}

ancora_status ancora_mat_vcat(ancora_mat *res, const ancora_mat *a, const ancora_mat *b)
/* Vertical concatenation: res = [a; b].
 * INPUT:
 *      res             : Result matrix, already initialized as
 *                        ((n1+n2) x m), where a is (n1 x m), b is (n2 x m)
 *      a               : Top block
 *      b               : Bottom block
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O((n1+n2)*m)
 *      In ANCORA_MODE_SAFE mode, O(ANCORA_DEFAULT_PREC) worst case per
 *      entry copied, so O((n1+n2)*m*ANCORA_DEFAULT_PREC) worst case overall.
 *
 * Created:       2026-09-22
 * Last modified: 2026-09-22
 * Author(s):     Adrian Kulmburg
 */
{
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_mat instance.");
    }
    if (a == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer a is NULL; it should point to a valid ancora_mat instance.");
    }
    if (b == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer b is NULL; it should point to a valid ancora_mat instance.");
    }
    if (a->ncols != b->ncols) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Matrix a has %ld columns, matrix b has %ld columns; they need to match for vertical concatenation.",
                      (long)a->ncols, (long)b->ncols);
    }
    if (res->ncols != a->ncols || res->nrows != a->nrows + b->nrows) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Result should be (%ld x %ld), but res is (%ld x %ld).",
                      (long)(a->nrows + b->nrows), (long)a->ncols, (long)res->nrows, (long)res->ncols);
    }
    if (res == a || res == b) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "res must not alias a or b for ancora_mat_vcat.");
    }

    slong n1 = a->nrows;
    slong n2 = b->nrows;
    slong m = a->ncols;

#if ANCORA_MODE == ANCORA_MODE_SAFE
    for (slong i = 0; i < n1; i++) {
        for (slong j = 0; j < m; j++) {
            arb_set(arb_mat_entry(res->repr, i, j), arb_mat_entry(a->repr, i, j));
        }
    }
    for (slong i = 0; i < n2; i++) {
        for (slong j = 0; j < m; j++) {
            arb_set(arb_mat_entry(res->repr, n1 + i, j), arb_mat_entry(b->repr, i, j));
        }
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
    /* Both blocks are row-major and span the full row width m, so each
     * block is one contiguous run of memory - a single memcpy per block
     * rather than per row. */
    memcpy(res->repr, a->repr, (size_t)(n1 * m) * sizeof(double));
    memcpy(&res->repr[n1 * m], b->repr, (size_t)(n2 * m) * sizeof(double));
#endif
    return ANCORA_OK;
}

ancora_status ancora_mat_diagcat(ancora_mat *res, const ancora_mat *a, const ancora_mat *b)
/* Block-diagonal concatenation: res = [a 0; 0 b].
 * INPUT:
 *      res             : Result matrix, already initialized as
 *                        ((n1+n2) x (m1+m2)), where a is (n1 x m1), b is
 *                        (n2 x m2)
 *      a               : Top-left block
 *      b               : Bottom-right block
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O((n1+n2)*(m1+m2))
 *      In ANCORA_MODE_SAFE mode, O(ANCORA_DEFAULT_PREC) worst case per
 *      entry touched, so O((n1+n2)*(m1+m2)*ANCORA_DEFAULT_PREC) worst case
 *      overall.
 *
 * Created:       2026-09-22
 * Last modified: 2026-09-22
 * Author(s):     Adrian Kulmburg
 */
{
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_mat instance.");
    }
    if (a == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer a is NULL; it should point to a valid ancora_mat instance.");
    }
    if (b == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer b is NULL; it should point to a valid ancora_mat instance.");
    }
    if (res->nrows != a->nrows + b->nrows || res->ncols != a->ncols + b->ncols) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Result should be (%ld x %ld), but res is (%ld x %ld).",
                      (long)(a->nrows + b->nrows), (long)(a->ncols + b->ncols),
                      (long)res->nrows, (long)res->ncols);
    }
    if (res == a || res == b) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "res must not alias a or b for ancora_mat_diagcat.");
    }

    slong n1 = a->nrows, m1 = a->ncols;
    slong n2 = b->nrows, m2 = b->ncols;
    slong mres = m1 + m2;

#if ANCORA_MODE == ANCORA_MODE_SAFE
    // Top n1 rows: a in columns [0, m1), exact zero in columns [m1, mres).
    for (slong i = 0; i < n1; i++) {
        for (slong j = 0; j < m1; j++) {
            arb_set(arb_mat_entry(res->repr, i, j), arb_mat_entry(a->repr, i, j));
        }
        for (slong j = m1; j < mres; j++) {
            arb_zero(arb_mat_entry(res->repr, i, j));
        }
    }
    // Bottom n2 rows: exact zero in columns [0, m1), b in columns [m1, mres).
    for (slong i = 0; i < n2; i++) {
        for (slong j = 0; j < m1; j++) {
            arb_zero(arb_mat_entry(res->repr, n1 + i, j));
        }
        for (slong j = 0; j < m2; j++) {
            arb_set(arb_mat_entry(res->repr, n1 + i, m1 + j), arb_mat_entry(b->repr, i, j));
        }
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
    for (slong i = 0; i < n1; i++) {
        memcpy(&res->repr[i * mres], &a->repr[i * m1], (size_t)m1 * sizeof(double));
        memset(&res->repr[i * mres + m1], 0, (size_t)m2 * sizeof(double));
    }
    for (slong i = 0; i < n2; i++) {
        memset(&res->repr[(n1 + i) * mres], 0, (size_t)m1 * sizeof(double));
        memcpy(&res->repr[(n1 + i) * mres + m1], &b->repr[i * m2], (size_t)m2 * sizeof(double));
    }
#endif
    return ANCORA_OK;
}
