/*
 * ancora_zonotope_randomPoints_gpu.hip.h
 *
 * Description
 * -----------
 * GPU backend for ancora_zonotope_randomPoints when compiled in
 * ANCORA_MODE_FAST with ANCORA_USE_GPU enabled.
 *
 * File Information
 * ----------------
 * Created:       2026-09-27
 * Last modified: 2026-09-27
 * Authors:       Adrian Kulmburg
 *
 * License
 * -------
 * Copyright (c) 2026 Adrian Kulmburg <adrian.kulmburg@kit.edu>
 * SPDX-License-Identifier: MIT
 */

#ifndef ANCORA_ZONOTOPE_RANDOMPOINTS_GPU_HIP_H
#define ANCORA_ZONOTOPE_RANDOMPOINTS_GPU_HIP_H

#include <stddef.h>

#include "ancora/ancora_config.h"

#ifdef __cplusplus
extern "C" {
#endif

int ancora_zonotope_batched_supportFunction_gpu(
    const double *d_host,
    const double *G_host,
    const slong *owner_host,
    double *res_host, /* IN: dot(d_b,c_b) per b; OUT: full support values */
    slong n_in,
    slong P_in,
    slong B_in);

#ifdef __cplusplus
}
#endif

#endif
