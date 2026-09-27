# ancora

`ancora` is a C library for set-based computations, modeled after the CORA
MATLAB toolbox (although **a**ncora is **n**ot **CORA**). The aim of ancora is
to provide an end-to-end safe framework, where floating-point errors and
optimization errors are all taken into account to provide accurate predictions
for safety-critical applications. Despite that, ancora is also built with a fast
(but potentially unsafe) alternative, suitable for instance for machine learning
or neural networks.

The library is designed around a single global **compute-mode** switch:

- **SAFE mode** (`ANCORA_MODE_SAFE=ON`): numeric payloads are stored as
  FLINT/ARB interval types (`arb_mat_t` / `arb_ptr`) and all arithmetic is
  performed with rigorous interval arithmetic at a configurable precision
  (`ANCORA_DEFAULT_PREC`, default 128 bits).
- **FAST mode** (`ANCORA_MODE_SAFE=OFF`, the default): numeric payloads are
  plain `double` arrays and all arithmetic is ordinary double-precision,
  structured as loop-based dense code that can also be dispatched to a GPU
  kernel (see the GPU section below).

Only one representation exists in any given build. Because the compute mode
changes the layout of the shared numeric types (`ancora_mat` / `ancora_vec`),
a SAFE-mode build is **ABI-incompatible** with a FAST-mode build. They must
never be mixed.

There are **three ways to build ancora**, each producing a differently-named
static library:

| Build configuration                          | Library name        |
|----------------------------------------------|---------------------|
| SAFE mode (`ANCORA_MODE_SAFE=ON`)            | `libancora`         |
| FAST mode, no GPU (default)                  | `libancora_fast`    |
| FAST mode, with GPU (`ANCORA_USE_GPU=ON`)    | `libancora_fast_gpu`|

The three are mutually exclusive configurations of a single build: pick one
and configure with it. The GPU variant is only meaningful in FAST mode (there
is no GPU-accelerated FLINT/ARB path).

## Requirements

- A C11 compiler (GCC, Clang, MSVC).
- CMake >= 3.21.
- **SAFE mode**: [FLINT](https://flintlib.org/) with ARB support (FLINT >= 2.9
  provides ARB), plus its dependencies GMP and MPFR.
- **FAST mode**: HIGHS (optimization solver), HIP/CUDA (GPU acceleration,
  optional).

---

## Installing the requirements on Ubuntu 24.04

These instructions assume a standard Ubuntu 24.04 (noble) installation.

### 1. Build tools and CMake

```sh
sudo apt update
sudo apt install -y build-essential cmake git pkg-config
```

### 2. FLINT (with GMP and MPFR)

Ubuntu 24.04 packages FLINT (with ARB) as `libflint-dev`, which pulls in GMP
and MPFR automatically:

```sh
sudo apt install -y libflint-dev
```

This installs the FLINT headers under `/usr/include/flint` and the library
`libflint` under `/usr/lib/x86_64-linux-gnu`, which is exactly what
`cmake/FindFLINT.cmake` searches for. You can verify the install with:

```sh
dpkg -l | grep flint
```

**Alternative: build FLINT from source.** If you need a newer FLINT than the
packaged one, build it yourself (FLINT >= 2.9 bundles ARB):

```sh
sudo apt install -y libgmp-dev libmpfr-dev
git clone https://github.com/flintlib/flint.git
cd flint
./bootstrap.sh
./configure --with-gmp=/usr --with-mpfr=/usr
make -j"$(nproc)"
sudo make install
sudo ldconfig
```

### 3. HIGHS (optimization solver)

HIGHS is the HiGHS linear/quadratic programming solver that ancora uses for
certain FAST optimization tasks. On Ubuntu 24.04 it is packaged as
`libhighs-dev`:

```sh
sudo apt install -y libhighs-dev
```

**Alternative: build HIGHS from source** (if the package is not available):

```sh
sudo apt install -y cmake build-essential
git clone https://github.com/ERGO-Code/HiGHS.git
cd HiGHS
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
sudo cmake --install build
sudo ldconfig
```

### 5. HIP and CUDA (optional, GPU acceleration)

The GPU path is **optional** and only applies to FAST mode (there is no
GPU-accelerated FLINT/ARB path). It is written once in HIP and can target
either AMD (ROCm) or NVIDIA (HIP-over-CUDA) hardware. GPU support is the most
hardware-dependent part of the setup; you only need it if you plan to run the
FAST-mode kernels on a GPU.

**NVIDIA (CUDA + HIP-over-CUDA):**

1. Install the CUDA toolkit from NVIDIA's repository (see
   https://developer.nvidia.com/cuda-downloads). For Ubuntu 24.04:
   ```sh
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt update
   sudo apt install -y cuda-toolkit
   ```
2. Install HIP so it can target CUDA. The ROCm HIP runtime can be built to
   wrap the CUDA runtime; the simplest route is to install the ROCm HIP
   packages and set `HIP_PLATFORM=nvidia` when configuring (the project's
   CMake does this for you when `ANCORA_GPU_PLATFORM=nvidia`). See
   https://rocm.docs.amd.com for the current ROCm install instructions.
3. Configure ancora with:
   ```sh
   cmake -S . -B build -DANCORA_MODE_SAFE=OFF -DANCORA_USE_GPU=ON \
       -DANCORA_GPU_PLATFORM=nvidia
   ```

**AMD (ROCm / HIP):**

1. Install ROCm (which includes HIP) following the official instructions at
   https://rocm.docs.amd.com (add the ROCm apt repository, then
   `sudo apt install -y rocm`).
2. Configure ancora with:
   ```sh
   cmake -S . -B build -DANCORA_MODE_SAFE=OFF -DANCORA_USE_GPU=ON \
       -DANCORA_GPU_PLATFORM=amd
   ```

> **Note on GPU in Docker:** GPU access inside a container requires extra
> runtime setup (the NVIDIA Container Toolkit with `--gpus all` for NVIDIA,
> or ROCm's container runtime with `/dev/kfd` and `/dev/dri` passthrough for
> AMD). The provided Docker image is CPU-only by default; see `docker/README.md`.

---

## Building the C library

### FAST mode, no GPU (default, no external deps)

```sh
cmake -S . -B build -DANCORA_MODE_SAFE=OFF
cmake --build build
```

This produces `libancora_fast`.

### FAST mode, with GPU

```sh
cmake -S . -B build -DANCORA_MODE_SAFE=OFF -DANCORA_USE_GPU=ON \
    -DANCORA_GPU_PLATFORM=amd   # or nvidia
cmake --build build
```

This produces `libancora_fast_gpu`.

### SAFE mode (requires FLINT)

```sh
cmake -S . -B build -DANCORA_MODE_SAFE=ON
cmake --build build
```

This produces `libancora`.

The compute mode is fixed for the entire build. Do not mix a SAFE-mode core
with a FAST-mode binding layer (or vice versa), and do not mix the GPU and
non-GPU FAST builds in one process.

### Installing

```sh
cmake --install build
```

This installs the headers under `<prefix>/include/ancora` and the static
library under `<prefix>/lib` (default prefix `/usr/local`). The library is
named `libancora.a` in SAFE mode, `libancora_fast.a` in FAST mode without
GPU, and `libancora_fast_gpu.a` in FAST mode with GPU.

## Building and running the tests

```sh
cmake -S . -B build -DANCORA_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Using ancora in your own code

After installing, compile against the library. For example, a program using
the vector layer in FAST mode (no GPU):

```sh
cc -I/usr/local/include myprog.c -L/usr/local/lib -lancora_fast -lm -o myprog
```

In FAST mode with GPU, link against the GPU variant (and any GPU runtime
libraries your build was configured with):

```sh
cc -I/usr/local/include myprog.c -L/usr/local/lib -lancora_fast_gpu -lm -o myprog
```

In SAFE mode, also link FLINT:

```sh
cc -I/usr/local/include myprog.c -L/usr/local/lib -lancora -lflint -lm -o myprog
```
