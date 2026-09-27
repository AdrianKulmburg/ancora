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

int ancora_zonotope_batched_randomPoints_standard_gpu(
    const double *G_host,
    const double *Xbig_host,
    const double *c_host,
    const slong *offset_host,
    const slong *m_host,
    double *res_host,
    slong n_in, slong M_in, slong N_in, slong B_in);

#ifdef __cplusplus
}
#endif

#endif
