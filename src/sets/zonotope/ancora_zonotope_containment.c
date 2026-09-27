/*
 * ancora_zonotope_containment.c
 *
 * Description
 * -----------
 * Containment checks for ancora_zonotope: whether a point (or a set of
 * points) is contained in the zonotope.
 *
 * Containment is decided by solving a linear feasibility problem with
 * HiGHS in ANCORA_MODE_FAST mode. In ANCORA_MODE_SAFE mode the LP is not
 * implemented and the functions return ANCORA_ERROR_NOT_IMPLEMENTED.
 *
 * ancora_zonotope_containsPoints builds the LP's constraint matrix (Z->G,
 * shared by every point) and creates the HiGHS instance ONCE, then, for
 * each point, only updates the row bounds (p - c changes per point, the
 * matrix does not) and re-solves the same instance -- rather than
 * rebuilding the matrix and creating/destroying a fresh HiGHS instance per
 * point, which is what the previous version of this file did by calling
 * ancora_zonotope_contains (a full standalone LP build+solve) once per
 * column. ancora_zonotope_contains is now a thin wrapper delegating to
 * ancora_zonotope_containsPoints with a single-column matrix.
 *
 * File Information
 * ----------------
 * Created:       2026-09-24
 * Last modified: 2026-09-26
 * Authors:       Adrian Kulmburg
 *
 * License
 * -------
 * Copyright (c) 2026 Adrian Kulmburg <adrian.kulmburg@kit.edu>
 * SPDX-License-Identifier: MIT
 */

#include "ancora/sets/zonotope/ancora_zonotope_containment.h"

#if ANCORA_MODE == ANCORA_MODE_FAST
#include <highs/interfaces/highs_c_api.h>
#endif

ancora_status ancora_zonotope_containsPoint(const ancora_zonotope *Z,
                                       const ancora_vec *p,
                                       ancora_truth *contained)
/* Checks whether the point p is contained in the zonotope Z.
 * This is decided by solving a linear feasibility problem with HiGHS in
 * ANCORA_MODE_FAST mode; in ANCORA_MODE_SAFE mode it returns
 * ANCORA_ERROR_NOT_IMPLEMENTED at the moment.
 *
 * INPUT:
 *      Z               : Pointer to an initialized ancora_zonotope instance
 *      p               : Point to check
 *      contained       : Output; set to true iff p is contained in Z
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      TODO: Centrally, for optimization once it has been determined accurately
 *
 * Created:       2026-09-26
 * Last modified: 2026-09-26
 * Author(s):     Adrian Kulmburg
 */
{
    if (p == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer p is NULL; it should point to a valid ancora_vec instance.");
    }

    bool isVector;
    ANCORA_TRY(ancora_mat_isVector(p, &isVector));
    if (!isVector) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer p is not a vector; it should be a column vector.");
    }

    // Send to the more general containsPoints
    return ancora_zonotope_containsPoints(Z, p, contained);
}

ancora_status ancora_zonotope_containsPoints(const ancora_zonotope *Z,
                                              const ancora_mat *P,
                                              ancora_truth *contained)
/* Checks whether every column of P is contained in the zonotope Z. P is a
 * matrix whose j-th column is the j-th point; contained is
 * true iff all k points are contained in Z.
 *
 * INPUT:
 *      Z               : Pointer to an initialized ancora_zonotope instance
 *      P               : Matrix of points, one point per
 *                        column
 *      contained       : Output; set to true iff every column of P is
 *                        contained in Z
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      TODO: Centrally, for optimization once it has been determined accurately
 *
 * Created:       2026-09-26
 * Last modified: 2026-09-26
 * Author(s):     Adrian Kulmburg
 */
{
    // Check that Z, P, and contained are well-defined
    if (Z == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z is NULL; it should point to a valid ancora_zonotope instance.");
    }
    if (P == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer P is NULL; it should point to a valid ancora_mat instance.");
    }
    if (contained == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer contained is NULL; it should point to a valid bool.");
    }

    // Dimension check
    if (P->nrows != Z->c.nrows) {
        ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                      "Zonotope Z has dimension %ld, each point (column of P) has length %ld; they need to be the same.",
                      (long)Z->c.nrows, (long)P->nrows);
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    ANCORA_ERROR(ANCORA_ERROR_NOT_IMPLEMENTED,
                 "Zonotope point containment via linear programming is only implemented in FAST mode; SAFE mode is not supported.");
#elif ANCORA_MODE == ANCORA_MODE_FAST
    slong n = Z->c.nrows;
    slong p_nr = Z->G.ncols;
    slong k = P->ncols;

    // Nothing to check: vacuously true, and building an LP for zero
    // points would be meaningless.
    if (k <= 0) {
        ancora_setYes(contained);
        return ANCORA_OK;
    }

    HighsInt numcol = (HighsInt)p_nr;
    HighsInt numrow = (HighsInt)n;
    HighsInt numnz = (HighsInt)(n * p_nr);

    HighsInt *astart = NULL, *aindex = NULL, *set = NULL;
    double *avalue = NULL, *colcost = NULL, *collower = NULL, *colupper = NULL;
    double *rowlower = NULL, *rowupper = NULL;
    void *highs = NULL;

    // For garbage collection
    // TODO: This is new, might want to implement ANCORA_ERROR globally that way
    ancora_status status = ANCORA_OK;

    astart = (HighsInt *)malloc((size_t)(numcol + 1) * sizeof(HighsInt));
    aindex = (HighsInt *)malloc((size_t)numnz * sizeof(HighsInt));
    avalue = (double *)malloc((size_t)numnz * sizeof(double));
    colcost = (double *)calloc((size_t)numcol, sizeof(double));
    collower = (double *)malloc((size_t)numcol * sizeof(double));
    colupper = (double *)malloc((size_t)numcol * sizeof(double));
    rowlower = (double *)malloc((size_t)numrow * sizeof(double));
    rowupper = (double *)malloc((size_t)numrow * sizeof(double));
    set = (HighsInt *)malloc((size_t)numrow * sizeof(HighsInt));

    if (astart == NULL || aindex == NULL || avalue == NULL || colcost == NULL ||
        collower == NULL || colupper == NULL || rowlower == NULL ||
        rowupper == NULL || set == NULL) {
        status = ANCORA_ERROR_ALLOC;
        goto cleanup;
    }

    /* Variable bounds: -1 <= beta <= 1 (shared by every point). */
    for (slong j = 0; j < p_nr; j++) {
        collower[j] = -1.0;
        colupper[j] = 1.0;
    }

    /* set = {0, 1, ..., n-1}: every row index, used to update ALL row
     * bounds at once for each point via Highs_changeRowsBoundsBySet. */
    for (slong i = 0; i < n; i++) {
        set[i] = (HighsInt)i;
    }

    /* Build the sparse constraint matrix A = G in column-wise format.
     * Z->G does not change across points, so this is done ONCE. */
    HighsInt nz = 0;
    for (slong j = 0; j < p_nr; j++) {
        astart[j] = nz;
        for (slong i = 0; i < n; i++) {
            aindex[nz] = (HighsInt)i;
            avalue[nz] = Z->G.repr[i * p_nr + j];
            nz++;
        }
    }
    astart[p_nr] = nz;

    /* Right-hand side for point 0: rowlower = rowupper = p_0 - c. */
    for (slong i = 0; i < n; i++) {
        rowlower[i] = P->repr[i * k + 0] - Z->c.repr[i];
        rowupper[i] = rowlower[i];
    }

    highs = Highs_create();
    if (highs == NULL) {
        status = ANCORA_ERROR_ALLOC;
        goto cleanup;
    }

    // Disable console logging
    Highs_setBoolOptionValue(highs, "log_to_console", false);

    // This LP is re-solved many times with only the right-hand side changing
    // (Highs_changeRowsBoundsBySet); presolve re-analyzes the whole problem
    // on every Highs_run call regardless, which for a problem this small
    // and cheap can dominate over the actual solve - disable it so re-solves
    // go straight to the solver. Simplex (not the default auto-chosen method,
    // which may select IPM) is forced because only simplex can warm-start
    // from the previous optimal basis after a bound change; IPM effectively
    // restarts from scratch on every re-solve, defeating the whole point of
    // reusing one HiGHS instance across points.
    Highs_setStringOptionValue(highs, "presolve", "off");
    Highs_setStringOptionValue(highs, "solver", "simplex");

    HighsInt pass_status = Highs_passLp(highs, numcol, numrow, numnz,
                                        kHighsMatrixFormatColwise,
                                        kHighsObjSenseMinimize, 0.0,
                                        colcost, collower, colupper,
                                        rowlower, rowupper,
                                        astart, aindex, avalue);
    if (pass_status != kHighsStatusOk) {
        status = ANCORA_ERROR_INVALID_ARG;
        goto cleanup;
    }

    ancora_setYes(contained);
    for (slong j = 0; j < k; j++) {
        if (j > 0) {
            /* Only the right-hand side changes between points; update it
             * in place and re-solve the SAME instance rather than
             * rebuilding the LP. */
            for (slong i = 0; i < n; i++) {
                rowlower[i] = P->repr[i * k + j] - Z->c.repr[i];
                rowupper[i] = rowlower[i];
            }
            HighsInt change_status = Highs_changeRowsBoundsBySet(
                highs, numrow, set, rowlower, rowupper);
            if (change_status != kHighsStatusOk) {
                status = ANCORA_ERROR_INVALID_ARG;
                goto cleanup;
            }
        }

        HighsInt run_status = Highs_run(highs);
        if (run_status != kHighsStatusOk) {
            status = ANCORA_ERROR_INVALID_ARG;
            goto cleanup;
        }

        HighsInt model_status = Highs_getModelStatus(highs);
        if (model_status != kHighsModelStatusOptimal) {
            ancora_setNo(contained);
            break; /* short-circuit: no need to check remaining points */
        }
    }

cleanup:
    if (highs != NULL) {
        Highs_destroy(highs);
    }
    free(astart);
    free(aindex);
    free(avalue);
    free(colcost);
    free(collower);
    free(colupper);
    free(rowlower);
    free(rowupper);
    free(set);

    if (status != ANCORA_OK) {
        ANCORA_ERROR(status, "Failed to pass the containment LP to HiGHS.");
    }
    return ANCORA_OK;
#endif
}

ancora_status ancora_zonotope_batched_containsPoints(
    const ancora_zonotope **Z_batch,
    const ancora_mat **P_batch,
    slong B,
    ancora_truth *contained)
/* Checks whether, for every b in [0, B), every column of P_batch[b] is
 * contained in Z_batch[b]. contained is set to true iff ALL points in ALL
 * pairs are contained in their respective zonotope. Every Z_batch[b] and
 * P_batch[b] must share the same dimension n; each Z_batch[b] may have its
 * OWN generator count p_b, and each P_batch[b] its OWN point count k_b.
 *
 * INPUT:
 *      Z_batch         : Array of B pointers to initialized
 *                        ancora_zonotope instances, all of dimension n
 *                        (generator counts p_b may differ)
 *      P_batch         : Array of B pointers to point matrices;
 *                        P_batch[b] is (n x k_b), one point per column
 *                        (k_b may differ)
 *      B               : Number of (zonotope, points) pairs (>= 0)
 *      contained       : Output; set to true iff every point in every
 *                        pair is contained in its zonotope
 *
 * OUTPUT:
 *      ancora_status   : Status (i.e., whether errors arose)
 *
 * RUNTIME:
 *      Worst case (nothing short-circuits), the sum over b of the cost of
 *      solving k_b LPs against a p_b-variable, n-constraint problem -
 *      identical total LP-solving work to B separate
 *      ancora_zonotope_containsPoints calls. The saving is avoiding B
 *      allocations of the n-sized arrays (allocated once instead) and,
 *      when generator counts repeat/are similar across the batch,
 *      avoiding repeated malloc/free of the p_b-sized arrays via reuse
 *      with growth - not a reduction in LP-solving work itself.
 *
 * Created:       2026-09-27
 * Last modified: 2026-09-27
 * Author(s):     Adrian Kulmburg
 */
{
    if (Z_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z_batch is NULL; it should point to a valid array of ancora_zonotope pointers.");
    }
    if (P_batch == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer P_batch is NULL; it should point to a valid array of ancora_mat pointers.");
    }
    if (contained == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer contained is NULL; it should point to a valid ancora_truth instance.");
    }
    if (B < 0) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "B is negative (%ld); it should be nonnegative.", (long)B);
    }

    if (B == 0) {
        ancora_setYes(contained);
        return ANCORA_OK; /* vacuously true */
    }

    if (Z_batch[0] == NULL) {
        ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z_batch[0] is NULL; it should point to a valid ancora_zonotope instance.");
    }
    slong n = Z_batch[0]->c.nrows;

    for (slong b = 0; b < B; b++) {
        if (Z_batch[b] == NULL) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer Z_batch[%ld] is NULL; it should point to a valid ancora_zonotope instance.", (long)b);
        }
        if (P_batch[b] == NULL) {
            ANCORA_ERROR(ANCORA_ERROR_INVALID_ARG, "Pointer P_batch[%ld] is NULL; it should point to a valid ancora_mat instance.", (long)b);
        }
        if (Z_batch[b]->c.nrows != n) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "Z_batch[0] has dimension %ld, Z_batch[%ld] has dimension %ld; every zonotope in the batch must share the same dimension.",
                          (long)n, (long)b, (long)Z_batch[b]->c.nrows);
        }
        if (P_batch[b]->nrows != n) {
            ANCORA_ERROR(ANCORA_ERROR_DIM_MISMATCH,
                          "Zonotope Z_batch[%ld] has dimension %ld, each point (column of P_batch[%ld]) has length %ld; they need to be the same.",
                          (long)b, (long)n, (long)b, (long)P_batch[b]->nrows);
        }
    }

#if ANCORA_MODE == ANCORA_MODE_SAFE
    ANCORA_ERROR(ANCORA_ERROR_NOT_IMPLEMENTED,
                 "Zonotope point containment via linear programming is only implemented in FAST mode; SAFE mode is not supported.");
#elif ANCORA_MODE == ANCORA_MODE_FAST
    HighsInt numrow = (HighsInt)n;

    // n is shared across the whole batch, so these are allocated ONCE.
    double *rowlower = (double *)malloc((size_t)numrow * sizeof(double));
    double *rowupper = (double *)malloc((size_t)numrow * sizeof(double));
    HighsInt *set = (HighsInt *)malloc((size_t)numrow * sizeof(HighsInt));
    if (rowlower == NULL || rowupper == NULL || set == NULL) {
        free(rowlower);
        free(rowupper);
        free(set);
        ANCORA_ERROR(ANCORA_ERROR_ALLOC, "ancora_zonotope_batched_containsPoints: failed to allocate shared row bookkeeping arrays.");
    }
    for (slong i = 0; i < n; i++) {
        set[i] = (HighsInt)i;
    }

    // p_b-sized arrays: grown via realloc as needed, reused across pairs.
    HighsInt *astart = NULL, *aindex = NULL;
    double *avalue = NULL, *colcost = NULL, *collower = NULL, *colupper = NULL;
    slong cap_p = 0, cap_nz = 0;

    ancora_status status = ANCORA_OK;
    ancora_setYes(contained);

    for (slong b = 0; b < B && status == ANCORA_OK; b++) {
        const ancora_zonotope *Z = Z_batch[b];
        const ancora_mat *P = P_batch[b];
        slong p_nr = Z->G.ncols;
        slong k = P->ncols;
        HighsInt numcol = (HighsInt)p_nr;
        HighsInt numnz = (HighsInt)(n * p_nr);

        if (k <= 0) {
            continue; /* vacuously satisfied for this pair; nothing to solve */
        }

        // Grow the p_b-sized arrays if this pair needs more room than the
        // largest pair seen so far; never shrink.
        if (p_nr > cap_p) {
            HighsInt *new_astart = (HighsInt *)realloc(astart, (size_t)(numcol + 1) * sizeof(HighsInt));
            double *new_colcost = (double *)realloc(colcost, (size_t)numcol * sizeof(double));
            double *new_collower = (double *)realloc(collower, (size_t)numcol * sizeof(double));
            double *new_colupper = (double *)realloc(colupper, (size_t)numcol * sizeof(double));
            if (new_astart == NULL || new_colcost == NULL || new_collower == NULL || new_colupper == NULL) {
                free(new_astart == NULL ? astart : new_astart);
                free(new_colcost == NULL ? colcost : new_colcost);
                free(new_collower == NULL ? collower : new_collower);
                free(new_colupper == NULL ? colupper : new_colupper);
                astart = NULL; colcost = NULL; collower = NULL; colupper = NULL;
                status = ANCORA_ERROR_ALLOC;
                break;
            }
            astart = new_astart;
            colcost = new_colcost;
            collower = new_collower;
            colupper = new_colupper;
            cap_p = p_nr;
        }
        if (numnz > cap_nz) {
            HighsInt *new_aindex = (HighsInt *)realloc(aindex, (size_t)numnz * sizeof(HighsInt));
            double *new_avalue = (double *)realloc(avalue, (size_t)numnz * sizeof(double));
            if (new_aindex == NULL || new_avalue == NULL) {
                free(new_aindex == NULL ? aindex : new_aindex);
                free(new_avalue == NULL ? avalue : new_avalue);
                aindex = NULL; avalue = NULL;
                status = ANCORA_ERROR_ALLOC;
                break;
            }
            aindex = new_aindex;
            avalue = new_avalue;
            cap_nz = numnz;
        }

        // Rebuild this pair's LP data (must be redone every pair: G_b differs).
        memset(colcost, 0, (size_t)numcol * sizeof(double));
        for (slong j = 0; j < p_nr; j++) {
            collower[j] = -1.0;
            colupper[j] = 1.0;
        }
        HighsInt nz = 0;
        for (slong j = 0; j < p_nr; j++) {
            astart[j] = nz;
            for (slong i = 0; i < n; i++) {
                aindex[nz] = (HighsInt)i;
                avalue[nz] = Z->G.repr[i * p_nr + j];
                nz++;
            }
        }
        astart[p_nr] = nz;

        for (slong i = 0; i < n; i++) {
            rowlower[i] = P->repr[i * k + 0] - Z->c.repr[i];
            rowupper[i] = rowlower[i];
        }

        void *highs = Highs_create();
        if (highs == NULL) {
            status = ANCORA_ERROR_ALLOC;
            break;
        }

        // Disable console logging
        Highs_setBoolOptionValue(highs, "log_to_console", false);

        // This LP is re-solved many times with only the right-hand side changing
        // (Highs_changeRowsBoundsBySet); presolve re-analyzes the whole problem
        // on every Highs_run call regardless, which for a problem this small
        // and cheap can dominate over the actual solve -- disable it so re-solves
        // go straight to the solver. Simplex (not the default auto-chosen method,
        // which may select IPM) is forced because only simplex can warm-start
        // from the previous optimal basis after a bound change; IPM effectively
        // restarts from scratch on every re-solve, defeating the whole point of
        // reusing one HiGHS instance across points.
        Highs_setStringOptionValue(highs, "presolve", "off");
        Highs_setStringOptionValue(highs, "solver", "simplex");

        HighsInt pass_status = Highs_passLp(highs, numcol, numrow, numnz,
                                            kHighsMatrixFormatColwise,
                                            kHighsObjSenseMinimize, 0.0,
                                            colcost, collower, colupper,
                                            rowlower, rowupper,
                                            astart, aindex, avalue);
        if (pass_status != kHighsStatusOk) {
            Highs_destroy(highs);
            status = ANCORA_ERROR_INVALID_ARG;
            break;
        }

        for (slong j = 0; j < k; j++) {
            if (j > 0) {
                for (slong i = 0; i < n; i++) {
                    rowlower[i] = P->repr[i * k + j] - Z->c.repr[i];
                    rowupper[i] = rowlower[i];
                }
                HighsInt change_status = Highs_changeRowsBoundsBySet(
                    highs, numrow, set, rowlower, rowupper);
                if (change_status != kHighsStatusOk) {
                    status = ANCORA_ERROR_INVALID_ARG;
                    break;
                }
            }

            HighsInt run_status = Highs_run(highs);
            if (run_status != kHighsStatusOk) {
                status = ANCORA_ERROR_INVALID_ARG;
                break;
            }

            HighsInt model_status = Highs_getModelStatus(highs);
            if (model_status != kHighsModelStatusOptimal) {
                ancora_setNo(contained);
                break; /* short-circuit: stop checking this pair's remaining points */
            }
        }

        Highs_destroy(highs);

        if (!ancora_yes(*contained)) {
            break; /* short-circuit: stop checking any further pairs too */
        }
    }

    free(rowlower);
    free(rowupper);
    free(set);
    free(astart);
    free(aindex);
    free(avalue);
    free(colcost);
    free(collower);
    free(colupper);

    if (status != ANCORA_OK) {
        ANCORA_ERROR(status, "ancora_zonotope_batched_containsPoints: LP setup/solve failed (see above).");
    }
    return ANCORA_OK;
#endif
}
