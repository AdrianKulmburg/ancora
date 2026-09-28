/*
 * ancora_zonotope_containment.c
 *
 * Description
 * -----------
 * Containment checks for ancora_zonotope: whether a point (or a set of
 * points) is contained in the zonotope.
 *
 * Containment is decided by solving a linear feasibility problem with
 * GLPK in ANCORA_MODE_FAST mode. In ANCORA_MODE_SAFE mode the LP is not
 * implemented and the functions return ANCORA_ERROR_NOT_IMPLEMENTED.
 *
 * ancora_zonotope_containsPoints builds the LP's constraint matrix (Z->G,
 * shared by every point) and creates the GLPK instance ONCE, then, for
 * each point, only updates the row bounds (p - c changes per point, the
 * matrix does not) and re-solves the same instance -- rather than
 * rebuilding the matrix and creating/destroying a fresh GLPK instance per
 * point, which is what the previous version of this file did by calling
 * ancora_zonotope_contains (a full standalone LP build+solve) once per
 * column. ancora_zonotope_contains is now a thin wrapper delegating to
 * ancora_zonotope_containsPoints with a single-column matrix.
 *
 * File Information
 * ----------------
 * Created:       2026-09-24
 * Last modified: 2026-09-28
 * Authors:       Adrian Kulmburg
 *
 * License
 * -------
 * Copyright (c) 2026 Adrian Kulmburg <adrian.kulmburg@kit.edu>
 * SPDX-License-Identifier: MIT
 */

#include "ancora/sets/zonotope/ancora_zonotope_containment.h"

#if ANCORA_MODE == ANCORA_MODE_FAST
#include <glpk.h>
#endif

ancora_status ancora_zonotope_containsPoint(const ancora_zonotope *Z,
                                       const ancora_vec *p,
                                       ancora_truth *contained)
/* Checks whether the point p is contained in the zonotope Z.
 * This is decided by solving a linear feasibility problem with GLPK in
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
 * Last modified: 2026-09-28
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

    int numcol = (int)p_nr;
    int numrow = (int)n;
    int numnz = (int)(n * p_nr);

    /* GLPK's sparse triplet arrays are 1-indexed; index 0 is unused, so
     * every array below is allocated with one extra slot. */
    int *ia = NULL, *ja = NULL;
    double *ar = NULL;
    glp_prob *lp = NULL;

    ancora_status status = ANCORA_OK;

    ia = (int *)malloc((size_t)(numnz + 1) * sizeof(int));
    ja = (int *)malloc((size_t)(numnz + 1) * sizeof(int));
    ar = (double *)malloc((size_t)(numnz + 1) * sizeof(double));

    if (ia == NULL || ja == NULL || ar == NULL) {
        status = ANCORA_ERROR_ALLOC;
        goto cleanup;
    }

    lp = glp_create_prob();
    if (lp == NULL) {
        status = ANCORA_ERROR_ALLOC;
        goto cleanup;
    }
    glp_set_obj_dir(lp, GLP_MIN);

    glp_add_rows(lp, numrow);
    glp_add_cols(lp, numcol);

    /* Variable bounds: -1 <= beta <= 1 (shared by every point). Objective
     * coefficients default to 0, which is exactly the feasibility problem
     * we want -- there's nothing to minimize, only a feasible point to
     * find. */
    for (slong j = 0; j < p_nr; j++) {
        glp_set_col_bnds(lp, (int)(j + 1), GLP_DB, -1.0, 1.0);
    }

    /* Build the sparse constraint matrix A = G. Z->G does not change
     * across points, so this is done ONCE. */
    int nz = 0;
    for (slong i = 0; i < n; i++) {
        for (slong j = 0; j < p_nr; j++) {
            nz++;
            ia[nz] = (int)(i + 1);
            ja[nz] = (int)(j + 1);
            ar[nz] = Z->G.repr[i * p_nr + j];
        }
    }
    glp_load_matrix(lp, nz, ia, ja, ar);

    /* Right-hand side for point 0: rowlower = rowupper = p_0 - c, i.e. an
     * equality (fixed) row. */
    for (slong i = 0; i < n; i++) {
        double rhs = P->repr[i * k + 0] - Z->c.repr[i];
        glp_set_row_bnds(lp, (int)(i + 1), GLP_FX, rhs, rhs);
    }

    glp_smcp parm;
    glp_init_smcp(&parm);
    parm.msg_lev = GLP_MSG_OFF;
    parm.presolve = GLP_OFF;
    parm.meth = GLP_PRIMAL;

    ancora_setYes(contained);
    for (slong j = 0; j < k; j++) {
        if (j > 0) {
            /* Only the right-hand side changes between points; update it
             * in place and re-solve the SAME glp_prob rather than
             * rebuilding the LP. glp_simplex always warm-starts from
             * whatever basis is already loaded, so re-solving after an
             * RHS-only change picks up from the previous optimal basis;
             * dual simplex is the natural method for recovering
             * optimality in that situation. */
            for (slong i = 0; i < n; i++) {
                double rhs = P->repr[i * k + j] - Z->c.repr[i];
                glp_set_row_bnds(lp, (int)(i + 1), GLP_FX, rhs, rhs);
            }
            parm.meth = GLP_DUAL;
        }

        int run_status = glp_simplex(lp, &parm);
        if (run_status != 0) {
            status = ANCORA_ERROR_INVALID_ARG;
            goto cleanup;
        }

        int model_status = glp_get_status(lp);
        if (model_status != GLP_OPT) {
            ancora_setNo(contained);
            break; /* short-circuit: no need to check remaining points */
        }
    }

cleanup:
    if (lp != NULL) {
        glp_delete_prob(lp);
    }
    free(ia);
    free(ja);
    free(ar);

    if (status != ANCORA_OK) {
        ANCORA_ERROR(status, "Failed to pass the containment LP to GLPK.");
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
 *      avoiding repeated malloc/free of the p_b-sized sparse-matrix
 *      arrays via reuse with growth - not a reduction in LP-solving work
 *      itself.
 *
 * Created:       2026-09-27
 * Last modified: 2026-09-28
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
    int numrow = (int)n;

    /* p_b-sized sparse-matrix arrays: grown via realloc as needed, reused
     * across pairs. GLPK's triplet arrays are 1-indexed, so allocate one
     * extra slot whenever they're (re)sized. */
    int *ia = NULL, *ja = NULL;
    double *ar = NULL;
    slong cap_nz = 0;

    ancora_status status = ANCORA_OK;
    ancora_setYes(contained);

    glp_smcp parm;
    glp_init_smcp(&parm);
    parm.msg_lev = GLP_MSG_OFF;
    parm.presolve = GLP_OFF;

    for (slong b = 0; b < B && status == ANCORA_OK; b++) {
        const ancora_zonotope *Z = Z_batch[b];
        const ancora_mat *P = P_batch[b];
        slong p_nr = Z->G.ncols;
        slong k = P->ncols;
        int numcol = (int)p_nr;
        int numnz = (int)(n * p_nr);

        if (k <= 0) {
            continue; /* vacuously satisfied for this pair; nothing to solve */
        }

        // Grow the nz-sized arrays if this pair needs more room than the
        // largest pair seen so far; never shrink.
        if (numnz > cap_nz) {
            int *new_ia = (int *)realloc(ia, (size_t)(numnz + 1) * sizeof(int));
            int *new_ja = (int *)realloc(ja, (size_t)(numnz + 1) * sizeof(int));
            double *new_ar = (double *)realloc(ar, (size_t)(numnz + 1) * sizeof(double));
            if (new_ia == NULL || new_ja == NULL || new_ar == NULL) {
                free(new_ia == NULL ? ia : new_ia);
                free(new_ja == NULL ? ja : new_ja);
                free(new_ar == NULL ? ar : new_ar);
                ia = NULL; ja = NULL; ar = NULL;
                status = ANCORA_ERROR_ALLOC;
                break;
            }
            ia = new_ia;
            ja = new_ja;
            ar = new_ar;
            cap_nz = numnz;
        }

        // Rebuild this pair's LP data (must be redone every pair: G_b
        // differs, and each pair gets its own glp_prob since numcol/numrow
        // can differ pair to pair).
        glp_prob *lp = glp_create_prob();
        if (lp == NULL) {
            status = ANCORA_ERROR_ALLOC;
            break;
        }
        glp_set_obj_dir(lp, GLP_MIN);
        glp_add_rows(lp, numrow);
        glp_add_cols(lp, numcol);

        for (slong j = 0; j < p_nr; j++) {
            glp_set_col_bnds(lp, (int)(j + 1), GLP_DB, -1.0, 1.0);
        }

        int nz = 0;
        for (slong i = 0; i < n; i++) {
            for (slong j = 0; j < p_nr; j++) {
                nz++;
                ia[nz] = (int)(i + 1);
                ja[nz] = (int)(j + 1);
                ar[nz] = Z->G.repr[i * p_nr + j];
            }
        }
        glp_load_matrix(lp, nz, ia, ja, ar);

        for (slong i = 0; i < n; i++) {
            double rhs = P->repr[i * k + 0] - Z->c.repr[i];
            glp_set_row_bnds(lp, (int)(i + 1), GLP_FX, rhs, rhs);
        }

        parm.meth = GLP_PRIMAL;

        for (slong j = 0; j < k; j++) {
            if (j > 0) {
                for (slong i = 0; i < n; i++) {
                    double rhs = P->repr[i * k + j] - Z->c.repr[i];
                    glp_set_row_bnds(lp, (int)(i + 1), GLP_FX, rhs, rhs);
                }
                parm.meth = GLP_DUAL;
            }

            int run_status = glp_simplex(lp, &parm);
            if (run_status != 0) {
                status = ANCORA_ERROR_INVALID_ARG;
                break;
            }

            int model_status = glp_get_status(lp);
            if (model_status != GLP_OPT) {
                ancora_setNo(contained);
                break; /* short-circuit: stop checking this pair's remaining points */
            }
        }

        glp_delete_prob(lp);

        if (!ancora_yes(*contained)) {
            break; /* short-circuit: stop checking any further pairs too */
        }
    }

    free(ia);
    free(ja);
    free(ar);

    if (status != ANCORA_OK) {
        ANCORA_ERROR(status, "ancora_zonotope_batched_containsPoints: LP setup/solve failed (see above).");
    }
    return ANCORA_OK;
#endif
}
