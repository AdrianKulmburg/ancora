/*
 * ancora_interval_duality_gpu.hip.cpp
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

#include <hip/hip_runtime.h>

#include "ancora/sets/interval/ancora_interval_duality_gpu.hip.h"


/* One thread per batch element b, looping over n internally (mirrors
 * ancora_mat_mul_kernel looping over its inner dimension k). lb, ub, d are
 * flat (B*n)-length row-major arrays: interval b's data occupies
 * [b*n, (b+1)*n). res is a flat B-length output array. */
__global__ void ancora_interval_batched_supportFunction_kernel(
    const double *lb,
    const double *ub,
    const double *d,
    double *res,
    long n,
    long B)
{
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        double dotAcc = 0.0;
        double normAcc = 0.0;
        for (long i = 0; i < n; i++) {
            double lbi = lb[b * n + i];
            double ubi = ub[b * n + i];
            double di = d[b * n + i];
            double centerVal = (lbi + ubi) * 0.5;
            double radiusVal = (ubi - lbi) * 0.5;
            dotAcc += centerVal * di;
            normAcc += fabs(radiusVal * di);
        }
        res[b] = dotAcc + normAcc;
    }
}

extern "C" int ancora_interval_batched_supportFunction_gpu(
    const double *lb_host,
    const double *ub_host,
    const double *d_host,
    double *res_host,
    slong n_in,
    slong B_in)
{
    long n = (long)n_in, B = (long)B_in;
    double *lb_dev = NULL, *ub_dev = NULL, *d_dev = NULL, *res_dev = NULL;
    size_t in_bytes = (size_t)(n * B) * sizeof(double);
    size_t res_bytes = (size_t)B * sizeof(double);
    hipError_t err;

    err = hipMalloc((void **)&lb_dev, in_bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&ub_dev, in_bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&d_dev, in_bytes);
    if (err != hipSuccess) goto fail;
    err = hipMalloc((void **)&res_dev, res_bytes);
    if (err != hipSuccess) goto fail;

    err = hipMemcpy(lb_dev, lb_host, in_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;
    err = hipMemcpy(ub_dev, ub_host, in_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;
    err = hipMemcpy(d_dev, d_host, in_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) goto fail;

    {
        int threads = 256;
        int blocks = (int)((B + threads - 1) / threads);
        hipLaunchKernelGGL(ancora_interval_batched_supportFunction_kernel,
                           dim3(blocks), dim3(threads), 0, 0,
                           lb_dev, ub_dev, d_dev, res_dev, n, B);
    }
    err = hipGetLastError();
    if (err != hipSuccess) goto fail;
    err = hipDeviceSynchronize();
    if (err != hipSuccess) goto fail;

    err = hipMemcpy(res_host, res_dev, res_bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) goto fail;

    (void)hipFree(lb_dev); (void)hipFree(ub_dev); (void)hipFree(d_dev); (void)hipFree(res_dev);
    return 0;

fail:
    if (lb_dev) (void)hipFree(lb_dev);
    if (ub_dev) (void)hipFree(ub_dev);
    if (d_dev) (void)hipFree(d_dev);
    if (res_dev) (void)hipFree(res_dev);
    return -1;
}
