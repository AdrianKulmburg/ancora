/*
 * ancora_interval_duality_gpu.hip.h
 *
 * Description
 * -----------
 * GPU backend for ancora_interval_duality when compiled in
 * ANCORA_MODE_FAST with ANCORA_USE_GPU enabled.
 *
 * File Information
 * ----------------
 * Created:       2026-09-26
 * Last modified: 2026-09-26
 * Authors:       Adrian Kulmburg
 *
 * License
 * -------
 * Copyright (c) 2026 Adrian Kulmburg <adrian.kulmburg@kit.edu>
 * SPDX-License-Identifier: MIT
 */

#ifndef ANCORA_INTERVAL_DUALITY_GPU_HIP_H
#define ANCORA_INTERVAL_DUALITY_GPU_HIP_H

#include <stddef.h>

#include "ancora/ancora_config.h"

#ifdef __cplusplus
extern "C" {
#endif

int ancora_interval_batched_supportFunction_gpu(
    const double *lb, const double *ub, const double *d,
    double *res, slong n, slong B);

#ifdef __cplusplus
}
#endif

#endif
