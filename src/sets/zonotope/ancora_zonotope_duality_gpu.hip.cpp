/*
 * ancora_zonotope_duality_gpu.hip.cpp
 *
 * Description
 * -----------
 * GPU backend for ancora_zonotope_duality when compiled in
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

#include "ancora/sets/zonotope/ancora_zonotope_duality_gpu.hip.h"


/* One thread per (batch element, generator column) pair -- P threads
 * total. d_flat is (B*n)-length (pair b's direction at [b*n, (b+1)*n)),
 * G_flat is (n*P)-length row-major with pair b's generators occupying
 * columns [offset[b], offset[b]+p[b]) of every row, owner[k] gives which
 * batch index global column k belongs to. res must already hold
 * dot(d_b, c_b) as its initial value (computed on the host); this kernel
 * atomically ADDS |d_b . G_b[:,j]| for every column j of every pair. */
__global__ void ancora_zonotope_batched_supportFunction_kernel(
    const double *d_flat,
    const double *G_flat,
    const long *owner,
    double *res,
    long n,
    long P)
{
    long k = (long)blockIdx.x * blockDim.x + threadIdx.x; /* global column index */
    if (k < P) {
        long b = owner[k];
        double dot = 0.0;
        for (long i = 0; i < n; i++) {
            dot += d_flat[b * n + i] * G_flat[i * P + k];
        }
        atomicAdd(&res[b], fabs(dot));
    }
}

extern "C" int ancora_zonotope_batched_supportFunction_gpu(
    const double *d_host,
    const double *G_host,
    const slong *owner_host,
    double *res_host, /* IN: dot(d_b,c_b) per b; OUT: full support values */
    slong n_in,
    slong P_in,
    slong B_in)
{
    long n = (long)n_in, P = (long)P_in, B = (long)B_in;
    double *d_dev = NULL, *G_dev = NULL, *res_dev = NULL;
    long *owner_dev = NULL;
    hipError_t err;

    size_t d_bytes = (size_t)(B * n) * sizeof(double);
    size_t G_bytes = (size_t)(n * P) * sizeof(double);
    size_t res_bytes = (size_t)B * sizeof(double);
    size_t owner_bytes = (size_t)P * sizeof(long);

    err = hipMalloc((void **)&d_dev, d_bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&G_dev, G_bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&owner_dev, owner_bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&res_dev, res_bytes);
    if (err != hipSuccess) goto fail;

    err = hipMemcpy(d_dev, d_host, d_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;
    err = hipMemcpy(G_dev, G_host, G_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;
    /* owner_host is slong (project-wide, likely 64-bit long already);
     * copied directly into a `long` device buffer. */
    err = hipMemcpy(owner_dev, owner_host, owner_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;
    /* res_host already holds dot(d_b,c_b); copy it as the atomicAdd seed. */
    err = hipMemcpy(res_dev, res_host, res_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;

    {
        int threads = 256;
        int blocks = (int)((P + threads - 1) / threads);
        hipLaunchKernelGGL(ancora_zonotope_batched_supportFunction_kernel,
                           dim3(blocks), dim3(threads), 0, 0,
                           d_dev, G_dev, owner_dev, res_dev, n, P);
    }
    err = hipGetLastError();
    if (err != hipSuccess) goto fail;
    err = hipDeviceSynchronize();
    if (err != hipSuccess) goto fail;

    err = hipMemcpy(res_host, res_dev, res_bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) goto fail;

    (void)hipFree(d_dev); (void)hipFree(G_dev); (void)hipFree(owner_dev); (void)hipFree(res_dev);
    return 0;

fail:
    if (d_dev) (void)hipFree(d_dev);
    if (G_dev) (void)hipFree(G_dev);
    if (owner_dev) (void)hipFree(owner_dev);
    if (res_dev) (void)hipFree(res_dev);
    return -1;
}
