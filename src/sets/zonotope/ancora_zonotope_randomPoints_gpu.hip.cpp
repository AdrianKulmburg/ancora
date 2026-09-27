/*
 * ancora_zonotope_randomPoints_gpu.hip.cpp
 *
 * Description
 * -----------
 * GPU backend for ancora_zonotope_randomPoints when compiled in
 * ANCORA_MODE_FAST with ANCORA_USE_GPU enabled.
 *
 * PERSISTENT STATE: the device buffers below are kept alive across calls
 * instead of a fresh hipMalloc/hipFree every time -- same rationale and
 * pattern as ancora_mat_gpu.hip.cpp's device buffer cache. Buffers only
 * GROW (never shrink) and are never freed until process exit. NOT
 * thread-safe (module-level static state, no locking); fine for a
 * single-threaded caller such as the benchmark driver.
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

/* ------------------------------------------------------------------ */
/* Persistent device buffer cache (see ancora_mat_gpu.hip.cpp for the   */
/* same pattern, documented in more detail there)                      */
/* ------------------------------------------------------------------ */

typedef struct {
    void *ptr;
    size_t capacity_bytes;
} ancora_zrp_gpu_buf;

static int ancora_zrp_gpu_buf_ensure(ancora_zrp_gpu_buf *buf, size_t needed_bytes)
{
    if (buf->capacity_bytes >= needed_bytes) {
        return 0;
    }
    if (buf->ptr) {
        (void)hipFree(buf->ptr);
        buf->ptr = NULL;
        buf->capacity_bytes = 0;
    }
    hipError_t err = hipMalloc(&buf->ptr, needed_bytes);
    if (err != hipSuccess) {
        return -1;
    }
    buf->capacity_bytes = needed_bytes;
    return 0;
}

static ancora_zrp_gpu_buf g_G = {0}, g_Xbig = {0}, g_c = {0};
static ancora_zrp_gpu_buf g_offset = {0}, g_m = {0}, g_res = {0};

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
    hipError_t err;

    size_t G_bytes = (size_t)(n * M) * sizeof(double);
    size_t Xbig_bytes = (size_t)(M * N) * sizeof(double);
    size_t c_bytes = (size_t)(B * n) * sizeof(double);
    size_t res_bytes = (size_t)(B * n * N) * sizeof(double);
    size_t idx_bytes = (size_t)B * sizeof(long);

    if (ancora_zrp_gpu_buf_ensure(&g_G, G_bytes) != 0) return -1;
    if (ancora_zrp_gpu_buf_ensure(&g_Xbig, Xbig_bytes) != 0) return -1;
    if (ancora_zrp_gpu_buf_ensure(&g_c, c_bytes) != 0) return -1;
    if (ancora_zrp_gpu_buf_ensure(&g_offset, idx_bytes) != 0) return -1;
    if (ancora_zrp_gpu_buf_ensure(&g_m, idx_bytes) != 0) return -1;
    if (ancora_zrp_gpu_buf_ensure(&g_res, res_bytes) != 0) return -1;

    double *G_dev = (double *)g_G.ptr;
    double *Xbig_dev = (double *)g_Xbig.ptr;
    double *c_dev = (double *)g_c.ptr;
    long *offset_dev = (long *)g_offset.ptr;
    long *m_dev = (long *)g_m.ptr;
    double *res_dev = (double *)g_res.ptr;

    err = hipMemcpy(G_dev, G_host, G_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) return -1;
    err = hipMemcpy(Xbig_dev, Xbig_host, Xbig_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) return -1;
    err = hipMemcpy(c_dev, c_host, c_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) return -1;
    err = hipMemcpy(offset_dev, offset_host, idx_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) return -1;
    err = hipMemcpy(m_dev, m_host, idx_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) return -1;

    {
        long total = B * n * N;
        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        hipLaunchKernelGGL(ancora_zonotope_batched_randomPoints_standard_kernel,
                           dim3(blocks), dim3(threads), 0, 0,
                           G_dev, Xbig_dev, c_dev, offset_dev, m_dev, res_dev, n, M, N, B);
    }
    err = hipGetLastError();
    if (err != hipSuccess) return -1;
    err = hipDeviceSynchronize();
    if (err != hipSuccess) return -1;

    err = hipMemcpy(res_host, res_dev, res_bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) return -1;

    return 0;
}
