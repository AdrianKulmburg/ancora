/*
 * ancora_mat_constructorDestructor.c
 *
 * Description
 * -----------
 * Constructor and destructor for ancora_mat.
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

#include "ancora/linalg/matrices/ancora_mat_constructorDestructor.h"

ancora_status ancora_mat_init(ancora_mat *m, slong nrows, slong ncols)
/* Initializes a matrix with a given number of rows and columns.
 * INPUT:
 *      m               : Pointer to an (uninitialized) ancora matrix, to be
 *                        initialized
 *      nrows           : Number of rows
 *      ncols           : Number of columns
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(nrows*ncols*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-21
 * Last modified: 2026-09-21
 * Author(s):     Adrian Kulmburg
 */
{
    if (m == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer m is NULL; it should point to a valid ancora_mat instance.");
    }
    if (nrows < 0 || ncols < 0) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "nrows or ncols is negative; both should be nonnegative");
    }

    m->nrows = nrows;
    m->ncols = ncols;

#if ANCORA_MODE == ANCORA_MODE_SAFE
    /* NOTE: arb_mat_init/arb_mat_zero have no failure return path of their
     * own -- FLINT/GMP's allocator abort() the process outright on
     * out-of-memory rather than returning an error we could catch here.
     * There is nothing ANCORA_ERROR_ALLOC can do about that in FLINT mode;
     * it's a known limitation of building on top of FLINT/GMP as-is. */
    arb_mat_init(m->repr, nrows, ncols);
#elif ANCORA_MODE == ANCORA_MODE_FAST
    if (nrows > 0 && ncols > 0) {
        m->repr = (double *)calloc((size_t)(nrows * ncols), sizeof(double));
        if (m->repr == NULL) {
            ANCORA_ERROR(ANCORA_ERROR_ALLOC, "Error allocating memory.");
        }
    }
    else
    {
        m->repr = NULL;
    }
#endif

    return ANCORA_OK;
}

ancora_status ancora_mat_free(ancora_mat *m)
/* Frees a previously initialized ancora matrix.
 * INPUT:
 *      m               : Pointer to an initialized ancora_mat instance, to
 *                        be freed
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(1) in ANCORA_MODE_FAST mode.
 *      O(nrows*ncols) in ANCORA_MODE_SAFE mode (one arb_clear per entry;
 *      independent of ANCORA_DEFAULT_PREC).
 *
 * Created:       2026-09-21
 * Last modified: 2026-09-21
 * Author(s):     Adrian Kulmburg
 */
{
    if (m == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer m is NULL; it should point to a valid ancora_mat instance.");
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    /* arb_mat_clear has no failure path of its own; nothing to check here. */
    arb_mat_clear(m->repr);
#elif ANCORA_MODE == ANCORA_MODE_FAST
    /* free(NULL) is well-defined (a no-op), so this is safe even if m->repr
     * was never allocated (e.g. a 0x0 matrix, per ancora_mat_init). */
    free(m->repr);
    m->repr = NULL;
#endif

    m->nrows = 0;
    m->ncols = 0;

    return ANCORA_OK;
}

ancora_status ancora_mat_copy(ancora_mat *res, const ancora_mat *a)
/* Copies the contents of a into res (res = a). Both must have the same
 * dimensions. Copying a matrix into itself (res == a) is a safe no-op.
 * INPUT:
 *      res             : Result matrix, already initialized as the same
 *                        dimensions as a
 *      a               : Matrix to copy
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      O(nrows*ncols*ANCORA_DEFAULT_PREC)
 *
 * Created:       2026-09-24
 * Last modified: 2026-09-24
 * Author(s):     Adrian Kulmburg
 */
{
    // Check that all matrices are valid
    if (res == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer res is NULL; it should point to a valid ancora_mat instance.");
    }
    if (a == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer a is NULL; it should point to a valid ancora_mat instance.");
    }

    // Check that both matrices have the same size
    if (res->nrows != a->nrows || res->ncols != a->ncols) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Matrix a is (%ld x %ld), matrix res is (%ld x %ld); they need to be the same.",
                      (long)a->nrows, (long)a->ncols, (long)res->nrows, (long)res->ncols);
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    for (slong i = 0; i < a->nrows; i++) {
        for (slong j = 0; j < a->ncols; j++) {
            arb_set(arb_mat_entry(res->repr, i, j), arb_mat_entry(a->repr, i, j));
        }
    }
#elif ANCORA_MODE == ANCORA_MODE_FAST
    memcpy(res->repr, a->repr, (size_t)(a->nrows * a->ncols) * sizeof(double));
#endif
    return ANCORA_OK;
}


