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
 * PERSISTENT STATE: every function below keeps its device buffers (and,
 * for ancora_mat_mul_gpu, its cuBLAS/hipBLAS handle) alive across calls
 * instead of doing a fresh hipMalloc/hipFree (or handle create/destroy)
 * every time. This matters because the benchmark driver (and any real
 * caller doing repeated operations at the same problem size) calls these
 * functions many times in a loop with identical buffer shapes --
 * hipMalloc/hipFree and cublasCreate/cublasDestroy involve real driver-
 * level bookkeeping, so redoing them on every call wastes time that has
 * nothing to do with the actual computation being measured. Buffers only
 * GROW (never shrink) and are never freed until process exit; the handle
 * is created once and reused forever. This is NOT thread-safe (module-
 * level static state, no locking) -- fine for a single-threaded process
 * such as the benchmark driver, but would need per-thread state or
 * locking if this file's functions were ever called concurrently.
 *
 * File Information
 * ----------------
 * Created:       2026-09-22
 * Last modified: 2026-09-27
 * Authors:       Adrian Kulmburg
 *
 * License
 * -------
 * Copyright (c) 2026 Adrian Kulmburg <adrian.kulmburg@kit.edu>
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#if defined(__HIP_PLATFORM_NVIDIA__)
#include <cublas_v2.h>
#else
#include <hipblas/hipblas.h>
#endif

#include "ancora/linalg/matrices/ancora_mat_arithmetic_gpu.hip.h"

/* ------------------------------------------------------------------ */
/* Persistent device buffer cache                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    double *ptr;
    size_t capacity_bytes;
} ancora_gpu_buf;

/* Grows *buf (if needed) to hold at least needed_bytes; never shrinks,
 * never frees on success. Returns 0 on success, -1 on allocation
 * failure (in which case *buf is left empty, not partially valid). */
static int ancora_gpu_buf_ensure(ancora_gpu_buf *buf, size_t needed_bytes)
{
    if (buf->capacity_bytes >= needed_bytes) {
        return 0;
    }
    if (buf->ptr) {
        (void)hipFree(buf->ptr);
        buf->ptr = NULL;
        buf->capacity_bytes = 0;
    }
    hipError_t err = hipMalloc((void **)&buf->ptr, needed_bytes);
    if (err != hipSuccess) {
        return -1;
    }
    buf->capacity_bytes = needed_bytes;
    return 0;
}

/* One set of cache slots per function below. Kept separate per function
 * (rather than shared/generalized) since each function's buffers serve a
 * distinct role and this benchmark-driven use case only ever exercises
 * one operation shape at a time -- simplicity over generality here. */
static ancora_gpu_buf g_add_a = {0}, g_add_b = {0}, g_add_res = {0};
static ancora_gpu_buf g_sub_a = {0}, g_sub_b = {0}, g_sub_res = {0};
static ancora_gpu_buf g_scalarMul_a = {0}, g_scalarMul_res = {0};
static ancora_gpu_buf g_mul_a = {0}, g_mul_b = {0}, g_mul_res = {0};
static ancora_gpu_buf g_transpose_a = {0}, g_transpose_res = {0};
static ancora_gpu_buf g_neg_a = {0}, g_neg_res = {0};

/* ------------------------------------------------------------------ */
/* Addition                                                             */
/* ------------------------------------------------------------------ */

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
    size_t bytes = n * sizeof(double);
    hipError_t err;

    if (ancora_gpu_buf_ensure(&g_add_a, bytes) != 0) return -1;
    if (ancora_gpu_buf_ensure(&g_add_b, bytes) != 0) return -1;
    if (ancora_gpu_buf_ensure(&g_add_res, bytes) != 0) return -1;

    err = hipMemcpy(g_add_a.ptr, a_host, bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) return -1;
    err = hipMemcpy(g_add_b.ptr, b_host, bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) return -1;

    {
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        hipLaunchKernelGGL(ancora_mat_add_kernel, dim3(blocks), dim3(threads),
                           0, 0, g_add_a.ptr, g_add_b.ptr, g_add_res.ptr, n);
    }
    err = hipGetLastError();
    if (err != hipSuccess) return -1;
    err = hipDeviceSynchronize();
    if (err != hipSuccess) return -1;

    err = hipMemcpy(res_host, g_add_res.ptr, bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) return -1;

    return 0;
}

/* ------------------------------------------------------------------ */
/* Subtraction                                                          */
/* ------------------------------------------------------------------ */

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
    size_t bytes = n * sizeof(double);
    hipError_t err;

    if (ancora_gpu_buf_ensure(&g_sub_a, bytes) != 0) return -1;
    if (ancora_gpu_buf_ensure(&g_sub_b, bytes) != 0) return -1;
    if (ancora_gpu_buf_ensure(&g_sub_res, bytes) != 0) return -1;

    err = hipMemcpy(g_sub_a.ptr, a_host, bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) return -1;
    err = hipMemcpy(g_sub_b.ptr, b_host, bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) return -1;

    {
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        hipLaunchKernelGGL(ancora_mat_sub_kernel, dim3(blocks), dim3(threads),
                           0, 0, g_sub_a.ptr, g_sub_b.ptr, g_sub_res.ptr, n);
    }
    err = hipGetLastError();
    if (err != hipSuccess) return -1;
    err = hipDeviceSynchronize();
    if (err != hipSuccess) return -1;

    err = hipMemcpy(res_host, g_sub_res.ptr, bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) return -1;

    return 0;
}

/* ------------------------------------------------------------------ */
/* Scalar multiplication                                                */
/* ------------------------------------------------------------------ */

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
    size_t bytes = n * sizeof(double);
    hipError_t err;

    if (ancora_gpu_buf_ensure(&g_scalarMul_a, bytes) != 0) return -1;
    if (ancora_gpu_buf_ensure(&g_scalarMul_res, bytes) != 0) return -1;

    err = hipMemcpy(g_scalarMul_a.ptr, a_host, bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) return -1;

    {
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        hipLaunchKernelGGL(ancora_mat_scalarMul_kernel, dim3(blocks), dim3(threads),
                           0, 0, g_scalarMul_a.ptr, g_scalarMul_res.ptr, scalar, n);
    }
    err = hipGetLastError();
    if (err != hipSuccess) return -1;
    err = hipDeviceSynchronize();
    if (err != hipSuccess) return -1;

    err = hipMemcpy(res_host, g_scalarMul_res.ptr, bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) return -1;

    return 0;
}

/* ------------------------------------------------------------------ */
/* Matrix multiplication                                                */
/* ------------------------------------------------------------------ */

/* Dense matrix multiplication: res = a * b, where a is (n x k) row-major,
 * b is (k x m) row-major, res is (n x m) row-major. res must not alias a
 * or b (same restriction as ancora_mat_mul itself).
 *
 * The cuBLAS/hipBLAS handle is created ONCE (on first call) and kept for
 * the lifetime of the process, rather than created/destroyed on every
 * call: handle creation does real setup work (internal workspace
 * allocation, device property queries), and redoing it every call in a
 * repetition loop wastes time unrelated to the actual matmul being
 * measured.
 */
#if defined(__HIP_PLATFORM_NVIDIA__)
static cublasHandle_t g_mul_handle = NULL;
#else
static hipblasHandle_t g_mul_handle = NULL;
#endif
static bool g_mul_handle_ready = false;

extern "C" int ancora_mat_mul_gpu(const double *a_host,
                                  const double *b_host,
                                  double *res_host,
                                  slong n_in,
                                  slong k_in,
                                  slong m_in)
{
    long n = (long)n_in, k = (long)k_in, m = (long)m_in;
    size_t a_bytes = (size_t)(n * k) * sizeof(double);
    size_t b_bytes = (size_t)(k * m) * sizeof(double);
    size_t res_bytes = (size_t)(n * m) * sizeof(double);
    hipError_t err;

    if (ancora_gpu_buf_ensure(&g_mul_a, a_bytes) != 0) return -1;
    if (ancora_gpu_buf_ensure(&g_mul_b, b_bytes) != 0) return -1;
    if (ancora_gpu_buf_ensure(&g_mul_res, res_bytes) != 0) return -1;

    err = hipMemcpy(g_mul_a.ptr, a_host, a_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) return -1;
    err = hipMemcpy(g_mul_b.ptr, b_host, b_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) return -1;

    {
        const double alpha = 1.0, beta = 0.0;
        /* Row-major C = A*B via the standard BLAS row/column-major trick:
         * both cuBLAS and hipBLAS are column-major, so the same memory
         * reinterpreted as column-major gives C^T = B^T * A^T - swapping
         * operand order/dimensions produces the correct row-major result
         * with no actual data rearrangement. */
#if defined(__HIP_PLATFORM_NVIDIA__)
        /* ROCm's apt-packaged hipblas is AMD-backend-only (its CMake
         * config references a hip::host target that does not exist on
         * the NVIDIA/HIP-over-CUDA platform); rather than building
         * hipBLAS from source with -DUSE_CUDA=ON, call cuBLAS directly
         * here, since HIP-over-CUDA is already running on the CUDA stack
         * underneath. */
        if (!g_mul_handle_ready) {
            if (cublasCreate(&g_mul_handle) != CUBLAS_STATUS_SUCCESS) return -1;
            g_mul_handle_ready = true;
        }

        cublasStatus_t stat = cublasDgemm(g_mul_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                          (int)m, (int)n, (int)k,
                                          &alpha,
                                          g_mul_b.ptr, (int)m,
                                          g_mul_a.ptr, (int)k,
                                          &beta,
                                          g_mul_res.ptr, (int)m);
        if (stat != CUBLAS_STATUS_SUCCESS) return -1;
#else
        if (!g_mul_handle_ready) {
            if (hipblasCreate(&g_mul_handle) != HIPBLAS_STATUS_SUCCESS) return -1;
            g_mul_handle_ready = true;
        }

        hipblasStatus_t stat = hipblasDgemm(g_mul_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                            (int)m, (int)n, (int)k,
                                            &alpha,
                                            g_mul_b.ptr, (int)m,
                                            g_mul_a.ptr, (int)k,
                                            &beta,
                                            g_mul_res.ptr, (int)m);
        if (stat != HIPBLAS_STATUS_SUCCESS) return -1;
#endif
    }

    err = hipDeviceSynchronize();
    if (err != hipSuccess) return -1;

    err = hipMemcpy(res_host, g_mul_res.ptr, res_bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) return -1;

    return 0;
}

/* ------------------------------------------------------------------ */
/* Transpose                                                            */
/* ------------------------------------------------------------------ */

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
    size_t bytes = (size_t)(nrows * ncols) * sizeof(double);
    hipError_t err;

    if (ancora_gpu_buf_ensure(&g_transpose_a, bytes) != 0) return -1;
    if (ancora_gpu_buf_ensure(&g_transpose_res, bytes) != 0) return -1;

    err = hipMemcpy(g_transpose_a.ptr, a_host, bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) return -1;

    {
        dim3 threads(ANCORA_TILE_DIM, ANCORA_BLOCK_ROWS);
        dim3 blocks((unsigned int)((ncols + ANCORA_TILE_DIM - 1) / ANCORA_TILE_DIM),
                    (unsigned int)((nrows + ANCORA_TILE_DIM - 1) / ANCORA_TILE_DIM));
        hipLaunchKernelGGL(ancora_mat_transpose_kernel, blocks, threads, 0, 0,
                           g_transpose_a.ptr, g_transpose_res.ptr, nrows, ncols);
    }
    err = hipGetLastError();
    if (err != hipSuccess) return -1;
    err = hipDeviceSynchronize();
    if (err != hipSuccess) return -1;

    err = hipMemcpy(res_host, g_transpose_res.ptr, bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) return -1;

    return 0;
}

/* ------------------------------------------------------------------ */
/* Negation                                                             */
/* ------------------------------------------------------------------ */

__global__ void ancora_mat_neg_kernel(const double *a, double *out, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = -a[i];
    }
}

extern "C" int ancora_mat_neg_gpu(const double *a_host, double *out_host, size_t n)
{
    size_t bytes = n * sizeof(double);
    hipError_t err;

    if (ancora_gpu_buf_ensure(&g_neg_a, bytes) != 0) return -1;
    if (ancora_gpu_buf_ensure(&g_neg_res, bytes) != 0) return -1;

    err = hipMemcpy(g_neg_a.ptr, a_host, bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) return -1;

    {
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        hipLaunchKernelGGL(ancora_mat_neg_kernel, dim3(blocks), dim3(threads),
                           0, 0, g_neg_a.ptr, g_neg_res.ptr, n);
    }
    err = hipGetLastError();
    if (err != hipSuccess) return -1;
    err = hipDeviceSynchronize();
    if (err != hipSuccess) return -1;

    err = hipMemcpy(out_host, g_neg_res.ptr, bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) return -1;

    return 0;
}
