# Postmortem: gpu-burn `(DIED!)` on RTX 5090 (Blackwell, sm_120)

How the bare-metal `gpu_burn` path failed on RTX 5090 hosts and how it was fixed.
Two **independent** bugs hid behind a single symptom; fixing the first exposed
the second.

## Symptom

Running `gpu/run_sm.bash gpuburn` (or `gpu/gpu_burn` directly):

- `cuInit returned 0 (no error)` prints for every GPU, then each card instantly
  reports `(DIED!)` with `proc'd: -1`.
- The parent process then logs `read[0] error 0` → `No clients are alive! Aborting`.
- Workers die *before* doing any real compute.

The root cause class for both bugs is the same: **a userspace artifact built with
a newer CUDA toolchain running on top of an older NVIDIA driver.** The driver is
the user-space half of the kernel module and is version-locked to it; CUDA
userspace libraries and device code refuse to run on a driver older than the
toolkit they were built with.

Relevant driver / CUDA matrix for sm_120:

| Toolkit | First arch support | Min Linux driver |
|--------:|:-------------------|:-----------------|
| CUDA 12.8 | sm_120 (RTX 5090) | **570** |
| CUDA 13.0 | sm_120            | **580** |

CUDA **12.8** is the *lowest* toolkit that supports sm_120, needs only driver
≥ 570, and still runs on newer drivers — so one 12.8 artifact is portable across
the whole 5090 fleet. That is why the fix standardizes on 12.8.

---

## Bug 1 — cuBLAS linked to CUDA 13 vs a 12.8-max driver

**Root cause.** `gpu_burn` was linked against **CUDA 13.0 cuBLAS**
(`libcublas.so.13`), which requires driver ≥ 580. The failing host had driver
**570.148.08** (max CUDA 12.8).

**Why it shows as DIED.** `cuInit` is pure Driver API and succeeds (hence
`cuInit returned 0`). But each worker's first `cublasCreate` is rejected by the
CUDA 13 cuBLAS as "driver too old" → the worker exits → all GPUs `(DIED!)`.

Probe that pinned it (CUDA 13 cuBLAS + host driver):

```
cuDriverGetVersion        = 12080   # driver tops out at CUDA 12.8
cublasCreate_v2 status    = 1       # CUBLAS_STATUS_NOT_INITIALIZED
```

**Fix.** Rebuild against **CUDA 12.8**: `gpu_burn` now links `libcublas.so.12`,
and `gpu/lib/` vendors the CUDA 12.8 `libcublas.so.12` + `libcublasLt.so.12`.

---

## Bug 2 — `compare.fatbin` built with the CUDA 13 toolchain (the subtle one)

After Bug 1 was fixed, cuBLAS initialized fine (verified on the 570 host:
`cublasCreate` / `cublasSgemm` / `cuCtxSynchronize` all returned 0) — but workers
still `(DIED!)`. The real error, buried under the progress spam:

```
Couldn't init a GPU test: Error in load module (gpu_burn-drv.cpp:244):
    the provided PTX was compiled with an unsupported toolchain.
cuModuleLoad(compare.fatbin) -> 222   # CUDA_ERROR_UNSUPPORTED_PTX_VERSION
```

**Root cause.** `gpu_burn` loads a device kernel, `compare.fatbin`, at startup.
It had been built by **CUDA 13's `nvcc`** with `-arch=compute_120`. That flag
embeds **PTX only** (a *virtual* arch — intermediate code, no machine code). On
load the driver must JIT-compile the PTX, but the PTX ISA version emitted by
CUDA 13 is newer than the 570 / CUDA-12.8 driver's JIT accepts → error 222.

**The wrong assumption that caused it.** An earlier fix assumed
"`compare.fatbin` is sm_120 device code and is CUDA-major-agnostic, so reuse it."
**This is false.** A fatbin/cubin is **toolchain-version-locked**. Because of
that assumption the file was never recompiled — its bytes were identical to the
old CUDA-13 build (same sha256).

**Fix.** Recompile `compare.cu` with the **CUDA 12.8** toolchain, and emit
**`sm_120` SASS** (a real machine-code cubin) instead of PTX. With no PTX there
is nothing for an older driver to JIT, so error 222 is structurally impossible.
`cuModuleLoad` accepts a bare cubin, so it is written as `gpu/compare.fatbin`
(the filename `gpu_burn` expects). Verify the artifact is ELF, not PTX text:

```
$ file gpu/compare.fatbin
gpu/compare.fatbin: ELF 64-bit LSB executable, NVIDIA CUDA architecture ...
$ strings gpu/compare.fatbin | grep -c '\.version'   # PTX directives — must be 0
0
```

### Building the kernel without Docker or nvcc

The host had no Docker and no full `nvcc`. The pip `nvidia-cuda-nvcc-cu12` wheel
ships only `ptxas` (no `cicc`/`nvcc` frontend). The way through:
`nvidia-cuda-nvrtc-cu12` ships `libnvrtc.so.12`, which compiles CUDA C++ → cubin
**in-process**. `compare.cu` has zero `#include`s, so no header juggling is
needed. Driven from Python via `ctypes`:

```
nvrtcCreateProgram(compare.cu)
nvrtcCompileProgram(["--gpu-architecture=sm_120"])   # SASS, not PTX
nvrtcGetCUBIN()  -> write to gpu/compare.fatbin
```

`gpu/build_gpuburn.sh` now does this on every build.

---

## Hardening (so neither bug recurs silently)

1. **Driver preflight** in `gpu/run_sm.bash`: before running, compare the
   vendored cuBLAS major against `nvidia-smi`'s reported max CUDA version. On a
   mismatch it aborts with an actionable message (upgrade the driver, or rebuild
   against the driver's CUDA) instead of the cryptic `(DIED!)`.
2. **Never reuse the kernel.** `gpu/build_gpuburn.sh` regenerates
   `compare.fatbin` from `compare.cu` every build with the current toolchain, and
   asserts the output is an ELF cubin (not PTX). The "reuse / CUDA-major-agnostic"
   logic was removed.
3. **CUDA-major-agnostic scripts.** Nothing hardcodes `.so.12`; the lib check
   globs `libcublas.so.[0-9]*`, and `CUDA_VER` / `CUDA_MAJOR` / `COMPUTE` override
   the build to target a different driver/arch. `gpu/lib/.cuda_version` records
   what the libs were built against.

## Summary

| Bug | What | Symptom | Fix |
|----:|------|---------|-----|
| 1 | cuBLAS linked CUDA 13, driver maxed at 12.8 | `cublasCreate` rejected → DIED | Rebuild against CUDA 12.8 (`libcublas.so.12`) |
| 2 | `compare.fatbin` was CUDA-13 PTX | `cuModuleLoad` → 222, JIT rejects PTX → DIED | Recompile `compare.cu` → sm_120 SASS (no PTX) |

Both are "new toolchain artifact on an old driver," one in the **cuBLAS library**,
one in the **compare kernel**. Everything is now standardized on CUDA 12.8 and the
whole build is toolkit-free (g++ + pip wheels, no Docker/nvcc/sudo).

Verified on 8× RTX 5090: `gpu/run_sm.bash gpuburn` → 8× `OK`, 0 errors,
`===GPU Stress Test Success===`.
