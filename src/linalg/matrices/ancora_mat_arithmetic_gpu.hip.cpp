/*
 * ancora_mat_gpu.hip.cpp
 *
 * Description
 * -----------
 * GPU backend for ancora_mat_arithmetic when compiled in
 * ANCORA_MODE_FAST with ANCORA_USE_GPU enabled.
 *
 * Written ONCE in HIP. Compiled with hipcc:
 *   - on the AMD/ROCm platform: runs natively on AMD GPUs
 *   - on the NVIDIA platform: HIP's headers wrap the CUDA runtime, and
 *     hipcc invokes nvcc underneath, producing real CUDA code from this
 *     same source file
 * See CMakeLists.txt (ANCORA_GPU_PLATFORM) for how the platform is chosen.
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

#include <hip/hip_runtime.h>

#include "ancora/linalg/matrices/ancora_mat_arithmetic_gpu.hip.h"

/* Elementwise addition: res[i] = a[i] + b[i] for i in [0, n).
 * a, b, res are flat, row-major double arrays of length n (n = nrows*ncols).
 */
__global__ void ancora_mat_add_kernel(const double *a,
                                      const double *b,
                                      double *res,
                                      size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        res[i] = a[i] + b[i];
    }
}

extern "C" int ancora_mat_add_gpu(const double *a_host,
                                  const double *b_host,
                                  double *res_host,
                                  size_t n)
{
    double *a_dev = NULL, *b_dev = NULL, *res_dev = NULL;
    size_t bytes = n * sizeof(double);
    hipError_t err;

    err = hipMalloc((void **)&a_dev, bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&b_dev, bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&res_dev, bytes);
    if (err != hipSuccess) goto fail;

    err = hipMemcpy(a_dev, a_host, bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;
    err = hipMemcpy(b_dev, b_host, bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;

    {
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        hipLaunchKernelGGL(ancora_mat_add_kernel, dim3(blocks), dim3(threads),
                           0, 0, a_dev, b_dev, res_dev, n);
    }
    err = hipGetLastError();
    if (err != hipSuccess) goto fail;
    err = hipDeviceSynchronize();
    if (err != hipSuccess) goto fail;

    err = hipMemcpy(res_host, res_dev, bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) goto fail;

    (void)hipFree(a_dev); (void)hipFree(b_dev); (void)hipFree(res_dev);
    return 0;

fail:
    if (a_dev) (void)hipFree(a_dev);
    if (b_dev) (void)hipFree(b_dev);
    if (res_dev) (void)hipFree(res_dev);
    return -1;
}

/* Elementwise subtraction: res[i] = a[i] - b[i] for i in [0, n).
 * a, b, res are flat, row-major double arrays of length n (n = nrows*ncols).
 */
__global__ void ancora_mat_sub_kernel(const double *a,
                                      const double *b,
                                      double *res,
                                      size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        res[i] = a[i] - b[i];
    }
}

extern "C" int ancora_mat_sub_gpu(const double *a_host,
                                  const double *b_host,
                                  double *res_host,
                                  size_t n)
{
    double *a_dev = NULL, *b_dev = NULL, *res_dev = NULL;
    size_t bytes = n * sizeof(double);
    hipError_t err;

    err = hipMalloc((void **)&a_dev, bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&b_dev, bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&res_dev, bytes);
    if (err != hipSuccess) goto fail;

    err = hipMemcpy(a_dev, a_host, bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;
    err = hipMemcpy(b_dev, b_host, bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;

    {
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        hipLaunchKernelGGL(ancora_mat_sub_kernel, dim3(blocks), dim3(threads),
                           0, 0, a_dev, b_dev, res_dev, n);
    }
    err = hipGetLastError();
    if (err != hipSuccess) goto fail;
    err = hipDeviceSynchronize();
    if (err != hipSuccess) goto fail;

    err = hipMemcpy(res_host, res_dev, bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) goto fail;

    (void)hipFree(a_dev); (void)hipFree(b_dev); (void)hipFree(res_dev);
    return 0;

fail:
    if (a_dev) (void)hipFree(a_dev);
    if (b_dev) (void)hipFree(b_dev);
    if (res_dev) (void)hipFree(res_dev);
    return -1;
}

/* Elementwise scalar multiplication: res[i] = scalar * a[i] for i in [0, n).
 * a, res are flat, row-major double arrays of length n (n = nrows*ncols).
 */
__global__ void ancora_mat_scalarMul_kernel(const double *a,
                                            double *res,
                                            double scalar,
                                            size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        res[i] = scalar * a[i];
    }
}

extern "C" int ancora_mat_scalarMul_gpu(const double *a_host,
                                        double *res_host,
                                        double scalar,
                                        size_t n)
{
    double *a_dev = NULL, *res_dev = NULL;
    size_t bytes = n * sizeof(double);
    hipError_t err;

    err = hipMalloc((void **)&a_dev, bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&res_dev, bytes);
    if (err != hipSuccess) goto fail;

    err = hipMemcpy(a_dev, a_host, bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;

    {
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        hipLaunchKernelGGL(ancora_mat_scalarMul_kernel, dim3(blocks), dim3(threads),
                           0, 0, a_dev, res_dev, scalar, n);
    }
    err = hipGetLastError();
    if (err != hipSuccess) goto fail;
    err = hipDeviceSynchronize();
    if (err != hipSuccess) goto fail;

    err = hipMemcpy(res_host, res_dev, bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) goto fail;

    (void)hipFree(a_dev); (void)hipFree(res_dev);
    return 0;

fail:
    if (a_dev) (void)hipFree(a_dev);
    if (res_dev) (void)hipFree(res_dev);
    return -1;
}

/* Dense matrix multiplication: out = a * b, where a is (n x k) row-major,
 * b is (k x m) row-major, out is (n x m) row-major. res must not alias a
 * or b (same restriction as ancora_mat_mul itself).
 */
__global__ void ancora_mat_mul_kernel(const double *a,
                                      const double *b,
                                      double *res,
                                      long n,
                                      long k,
                                      long m)
{
    long row = (long)blockIdx.y * blockDim.y + threadIdx.y;
    long col = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < m) {
        double sum = 0.0;
        for (long p = 0; p < k; p++) {
            sum += a[row * k + p] * b[p * m + col];
        }
        res[row * m + col] = sum;
    }
}

extern "C" int ancora_mat_mul_gpu(const double *a_host,
                                  const double *b_host,
                                  double *res_host,
                                  slong n_in,
                                  slong k_in,
                                  slong m_in)
{
    long n = (long)n_in, k = (long)k_in, m = (long)m_in;
    double *a_dev = NULL, *b_dev = NULL, *res_dev = NULL;
    size_t a_bytes = (size_t)(n * k) * sizeof(double);
    size_t b_bytes = (size_t)(k * m) * sizeof(double);
    size_t res_bytes = (size_t)(n * m) * sizeof(double);
    hipError_t err;

    err = hipMalloc((void **)&a_dev, a_bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&b_dev, b_bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&res_dev, res_bytes);
    if (err != hipSuccess) goto fail;

    err = hipMemcpy(a_dev, a_host, a_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;
    err = hipMemcpy(b_dev, b_host, b_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;

    {
        dim3 threads(16, 16);
        dim3 blocks((unsigned int)((m + threads.x - 1) / threads.x),
                    (unsigned int)((n + threads.y - 1) / threads.y));
        hipLaunchKernelGGL(ancora_mat_mul_kernel, blocks, threads, 0, 0,
                           a_dev, b_dev, res_dev, n, k, m);
    }
    err = hipGetLastError();
    if (err != hipSuccess) goto fail;
    err = hipDeviceSynchronize();
    if (err != hipSuccess) goto fail;

    err = hipMemcpy(res_host, res_dev, res_bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) goto fail;

    (void)hipFree(a_dev); (void)hipFree(b_dev); (void)hipFree(res_dev);
    return 0;

fail:
    if (a_dev) (void)hipFree(a_dev);
    if (b_dev) (void)hipFree(b_dev);
    if (res_dev) (void)hipFree(res_dev);
    return -1;
}

__global__ void ancora_mat_transpose_kernel(const double *a,
                                            double *res,
                                            long nrows,
                                            long ncols)
{
    __shared__ double tile[ANCORA_TILE_DIM][ANCORA_TILE_DIM + 1];

    // Read from a: x indexes columns of a, y indexes rows of a
    long x = (long)blockIdx.x * ANCORA_TILE_DIM + threadIdx.x;
    long y = (long)blockIdx.y * ANCORA_TILE_DIM + threadIdx.y;

    for (int j = 0; j < ANCORA_TILE_DIM; j += ANCORA_BLOCK_ROWS) {
        if (x < ncols && (y + j) < nrows) {
            tile[threadIdx.y + j][threadIdx.x] = a[(y + j) * ncols + x];
        }
    }

    __syncthreads();

    // Write to res: block coordinates are swapped, so x indexes columns of
    // res (= rows of a) and y indexes rows of res (= columns of a)
    x = (long)blockIdx.y * ANCORA_TILE_DIM + threadIdx.x;
    y = (long)blockIdx.x * ANCORA_TILE_DIM + threadIdx.y;

    for (int j = 0; j < ANCORA_TILE_DIM; j += ANCORA_BLOCK_ROWS) {
        if (x < nrows && (y + j) < ncols) {
            res[(y + j) * nrows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

extern "C" int ancora_mat_transpose_gpu(const double *a_host,
                                        double *res_host,
                                        slong nrows_in,
                                        slong ncols_in)
{
    long nrows = (long)nrows_in, ncols = (long)ncols_in;
    double *a_dev = NULL, *res_dev = NULL;
    size_t bytes = (size_t)(nrows * ncols) * sizeof(double);
    hipError_t err;

    err = hipMalloc((void **)&a_dev, bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&res_dev, bytes);
    if (err != hipSuccess) goto fail;

    err = hipMemcpy(a_dev, a_host, bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;

    {
        dim3 threads(ANCORA_TILE_DIM, ANCORA_BLOCK_ROWS);
        dim3 blocks((unsigned int)((ncols + ANCORA_TILE_DIM - 1) / ANCORA_TILE_DIM),
                    (unsigned int)((nrows + ANCORA_TILE_DIM - 1) / ANCORA_TILE_DIM));
        hipLaunchKernelGGL(ancora_mat_transpose_kernel, blocks, threads, 0, 0,
                           a_dev, res_dev, nrows, ncols);
    }
    err = hipGetLastError();
    if (err != hipSuccess) goto fail;
    err = hipDeviceSynchronize();
    if (err != hipSuccess) goto fail;

    err = hipMemcpy(res_host, res_dev, bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) goto fail;

    (void)hipFree(a_dev); (void)hipFree(res_dev);
    return 0;

fail:
    if (a_dev) (void)hipFree(a_dev);
    if (res_dev) (void)hipFree(res_dev);
    return -1;
}

__global__ void ancora_mat_neg_kernel(const double *a, double *out, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = -a[i];
    }
}

extern "C" int ancora_mat_neg_gpu(const double *a_host, double *out_host, size_t n)
{
    double *a_dev = NULL, *out_dev = NULL;
    size_t bytes = n * sizeof(double);
    hipError_t err;

    err = hipMalloc((void **)&a_dev, bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&out_dev, bytes);
    if (err != hipSuccess) goto fail;

    err = hipMemcpy(a_dev, a_host, bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;

    {
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        hipLaunchKernelGGL(ancora_mat_neg_kernel, dim3(blocks), dim3(threads),
                           0, 0, a_dev, out_dev, n);
    }
    err = hipGetLastError();
    if (err != hipSuccess) goto fail;
    err = hipDeviceSynchronize();
    if (err != hipSuccess) goto fail;

    err = hipMemcpy(out_host, out_dev, bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) goto fail;

    (void)hipFree(a_dev); (void)hipFree(out_dev);
    return 0;

fail:
    if (a_dev) (void)hipFree(a_dev);
    if (out_dev) (void)hipFree(out_dev);
    return -1;
}
