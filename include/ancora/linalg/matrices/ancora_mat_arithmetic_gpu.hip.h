/*
 * ancora_mat_gpu.hip.h
 *
 * Description
 * -----------
 * Declarations for the GPU-backed matrix kernels (addition, multiplication).
 * Implemented once in HIP (ancora_mat_arithmetic_gpu.hip.cpp), compiled either
 * for the AMD/ROCm platform or, via HIP-over-CUDA, for NVIDIA hardware (see
 * the project's CMakeLists.txt (ANCORA_GPU_PLATFORM) for how the platform is
 * selected). Only meaningful when ANCORA_MODE == ANCORA_MODE_FAST and
 * ANCORA_USE_GPU is defined; there is no GPU-accelerated FLINT/ARB path.
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

#ifndef ANCORA_MAT_GPU_HIP_H
#define ANCORA_MAT_GPU_HIP_H

#include <stddef.h>

#include "ancora/ancora_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Elementwise addition: res[i] = a[i] + b[i] for i in [0, n).
 * a, b, res are flat, row-major double arrays of length n (n = nrows*ncols).
 */
int ancora_mat_add_gpu(const double *a,
                       const double *b,
                       double *res,
                       size_t n);

/* Elementwise subtraction: res[i] = a[i] - b[i] for i in [0, n).
 * a, b, res are flat, row-major double arrays of length n (n = nrows*ncols).
 */
int ancora_mat_sub_gpu(const double *a,
                       const double *b,
                       double *res,
                       size_t n);

/* Elementwise scalar multiplication: res[i] = scalar * a[i] for i in [0, n).
 * a, res are flat, row-major double arrays of length n (n = nrows*ncols).
 */
int ancora_mat_scalarMul_gpu(const double *a_host,
                             double *res_host,
                             double scalar,
                             size_t n);

/* Dense matrix multiplication: out = a * b, where a is (n x k) row-major,
 * b is (k x m) row-major, out is (n x m) row-major. res must not alias a
 * or b (same restriction as ancora_mat_mul itself).
 */
int ancora_mat_mul_gpu(const double *a,
                       const double *b,
                       double *res,
                       slong n,
                       slong k,
                       slong m);
/* Matrix transpose: res = a^T, where a is (nrows x ncols) row-major and
 * res is (ncols x nrows) row-major. res must not alias a.
 *
 * Uses a shared-memory tile so that both the global reads and the global
 * writes are coalesced. The +1 padding on the tile avoids shared-memory
 * bank conflicts. Each block is TILE_DIM x BLOCK_ROWS threads, and each
 * thread handles TILE_DIM / BLOCK_ROWS elements.
 */
#define ANCORA_TILE_DIM   32
#define ANCORA_BLOCK_ROWS 8
int ancora_mat_transpose_gpu(const double *a_host,
                             double *res_host,
                             slong nrows,
                             slong ncols);

/* Elementwise negation: out[i] = -a[i] for i in [0, n).
 * a, out are flat, row-major double arrays of length n (n = nrows*ncols).
 * Returns 0 on success, nonzero on any HIP/CUDA runtime failure -- the
 * caller (ancora_mat_neg) maps a nonzero return to ANCORA_ERROR_GPU_LAUNCH.
 * Implemented in ancora_mat_gpu.hip.cpp.
 */
int ancora_mat_neg_gpu(const double *a, double *out, size_t n);

#ifdef __cplusplus
}
#endif

#endif
