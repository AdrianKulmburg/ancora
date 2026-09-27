/*
 * ancora_zonotope_randomPoints_gpu.hip.cpp
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

#include <hip/hip_runtime.h>

#include "ancora/sets/zonotope/ancora_zonotope_randomPoints_gpu.hip.h"

/* One thread per (pair, output row, output col) triple  B*n*N threads
 * total. G_flat is (n x M) row-major, pair b's generators occupy columns
 * [offset[b], offset[b]+m[b]) of every row. Xbig is (M x N) row-major,
 * pair b's cube samples occupy rows [offset[b], offset[b]+m[b]). c_flat
 * is (B x n) row-major. res_flat is (B x n x N) row-major output. */
__global__ void ancora_zonotope_batched_randomPoints_standard_kernel(
    const double *G_flat,
    const double *Xbig,
    const double *c_flat,
    const long *offset,
    const long *m,
    double *res_flat,
    long n,
    long M,
    long N,
    long B)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = B * n * N;
    if (idx < total) {
        long b = idx / (n * N);
        long rem = idx % (n * N);
        long i = rem / N;
        long j = rem % N;

        long off = offset[b];
        long mb = m[b];

        double acc = c_flat[b * n + i];
        for (long k = 0; k < mb; k++) {
            acc += G_flat[i * M + off + k] * Xbig[(off + k) * N + j];
        }
        res_flat[idx] = acc;
    }
}

extern "C" int ancora_zonotope_batched_randomPoints_standard_gpu(
    const double *G_host,
    const double *Xbig_host,
    const double *c_host,
    const slong *offset_host,
    const slong *m_host,
    double *res_host,
    slong n_in, slong M_in, slong N_in, slong B_in)
{
    long n = (long)n_in, M = (long)M_in, N = (long)N_in, B = (long)B_in;
    double *G_dev = NULL, *Xbig_dev = NULL, *c_dev = NULL, *res_dev = NULL;
    long *offset_dev = NULL, *m_dev = NULL;
    hipError_t err;

    size_t G_bytes = (size_t)(n * M) * sizeof(double);
    size_t Xbig_bytes = (size_t)(M * N) * sizeof(double);
    size_t c_bytes = (size_t)(B * n) * sizeof(double);
    size_t res_bytes = (size_t)(B * n * N) * sizeof(double);
    size_t idx_bytes = (size_t)B * sizeof(long);

    err = hipMalloc((void **)&G_dev, G_bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&Xbig_dev, Xbig_bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&c_dev, c_bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&offset_dev, idx_bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&m_dev, idx_bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&res_dev, res_bytes);
    if (err != hipSuccess) goto fail;

    err = hipMemcpy(G_dev, G_host, G_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;
    err = hipMemcpy(Xbig_dev, Xbig_host, Xbig_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;
    err = hipMemcpy(c_dev, c_host, c_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;
    err = hipMemcpy(offset_dev, offset_host, idx_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;
    err = hipMemcpy(m_dev, m_host, idx_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;

    {
        long total = B * n * N;
        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        hipLaunchKernelGGL(ancora_zonotope_batched_randomPoints_standard_kernel,
                           dim3(blocks), dim3(threads), 0, 0,
                           G_dev, Xbig_dev, c_dev, offset_dev, m_dev, res_dev, n, M, N, B);
    }
    err = hipGetLastError();
    if (err != hipSuccess) goto fail;
    err = hipDeviceSynchronize();
    if (err != hipSuccess) goto fail;

    err = hipMemcpy(res_host, res_dev, res_bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) goto fail;

    (void)hipFree(G_dev); (void)hipFree(Xbig_dev); (void)hipFree(c_dev);
    (void)hipFree(offset_dev); (void)hipFree(m_dev); (void)hipFree(res_dev);
    return 0;

fail:
    if (G_dev) (void)hipFree(G_dev);
    if (Xbig_dev) (void)hipFree(Xbig_dev);
    if (c_dev) (void)hipFree(c_dev);
    if (offset_dev) (void)hipFree(offset_dev);
    if (m_dev) (void)hipFree(m_dev);
    if (res_dev) (void)hipFree(res_dev);
    return -1;
}
