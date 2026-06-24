# Driver ⇄ CUDA ⇄ artifact compatibility (RTX 5090 / sm_120)

Quick reference for **why a build runs on one host but `(DIED!)` on another**, and
the matching rule the runtime "catcher" (the driver preflight in
`gpu/run_sm.bash`) enforces before it lets gpu-burn run.

If you only remember one thing: **a CUDA userspace artifact never runs on a
driver older than the toolkit it was built with.** Both halves of gpu-burn — the
cuBLAS library *and* the `compare` kernel — obey this. See
[`POSTMORTEM-rtx5090-gpuburn.md`](POSTMORTEM-rtx5090-gpuburn.md) for the full story.

## The matrix

| NVIDIA driver | Max CUDA it supports | Runs CUDA 12.8 build? | Runs CUDA 13 build? |
|--------------:|:---------------------|:---------------------:|:-------------------:|
| **570.x**     | 12.8                 | ✅ yes                | ❌ no (needs ≥ 580) |
| **580.x**     | 13.0                 | ✅ yes                | ✅ yes              |
| **595.x**     | 13.2                 | ✅ yes                | ✅ yes              |

| CUDA toolkit | First arch support | Min Linux driver |
|-------------:|:-------------------|:-----------------|
| **12.8**     | sm_120 (RTX 5090)  | **570**          |
| **13.0**     | sm_120             | **580**          |

**Why this repo standardizes on CUDA 12.8:** it is the *lowest* toolkit that
supports sm_120 (so it covers the RTX 5090), needs only driver ≥ 570, and still
runs on newer drivers (580/595/…). One 12.8 artifact is therefore portable across
the entire 5090 fleet. A CUDA 13 build would exclude every host still on the 570
driver line.

`nvidia-smi`'s header shows the driver's **max** CUDA, e.g.
`CUDA Version: 12.8` — that is the ceiling, not what is installed.

## The two artifacts and their rules

gpu-burn has two CUDA pieces; **both** must satisfy the host driver:

| Artifact | File | Rule |
|----------|------|------|
| cuBLAS library | `gpu/lib/libcublas.so.<MAJOR>` (+ `libcublasLt`) | its CUDA **major** ≤ driver's max CUDA major |
| compare kernel | `gpu/compare.fatbin` | **`sm_120` SASS** (machine code, no PTX to JIT). If it is PTX instead, it must have been emitted by a toolkit ≤ the driver's CUDA, or the driver's JIT rejects it with error **222 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`** |

`gpu/lib/.cuda_version` records which CUDA the vendored libs were built against.

## What the preflight ("catcher") checks

Before launching gpu-burn, `gpuburn_driver_preflight()` in `gpu/run_sm.bash`:

1. Reads the vendored cuBLAS **major** from the filename `libcublas.so.<MAJOR>`
   (and the display string from `gpu/lib/.cuda_version`).
2. Reads the driver's **max CUDA** from `nvidia-smi` (`CUDA Version: X.Y`).
3. **If `driver_max_major < vendored_cublas_major` → abort** with an actionable
   message *before* any worker dies, instead of the cryptic `(DIED!)`.
   Otherwise it prints `[preflight] OK` and proceeds.

Example outputs:

```
[preflight] OK: driver supports CUDA 13.2; vendored cuBLAS is CUDA 12.8.
```

```
[preflight] FATAL driver/cuBLAS mismatch:
    vendored cuBLAS  : CUDA 13.x  (gpu/lib/libcublas.so.13)
    driver max CUDA  : 12.8  (570.148.08)
    CUDA 13 cuBLAS needs a driver supporting CUDA 13.x or newer; on this host
    every gpu_burn worker would die at cublasCreate (the cryptic '(DIED!)').
    Fix one of:
      - upgrade the NVIDIA driver to one supporting CUDA 13.x+, or
      - rebuild gpu-burn for this driver:  CUDA_VER=12.8 CUDA_MAJOR=12 gpu/build_gpuburn.sh
        then re-fetch matching libs:        CUDA_VER=12.8 CUDA_MAJOR=12 gpu/setup_libs.sh
```

> The preflight only guards the **cuBLAS** half (which it can detect cheaply from
> the driver's max CUDA). The **kernel** half is handled at build time:
> `gpu/build_gpuburn.sh` always recompiles `compare.cu` to `sm_120` SASS and
> asserts the result is an ELF cubin (not PTX), so the error-222 case cannot be
> reintroduced by reusing a stale kernel.

## Targeting a different driver / GPU

The scripts are CUDA-major-agnostic — nothing hardcodes `.so.12`. To build for a
different driver or arch, set the env overrides (defaults: `12.8` / `12` / `120`):

```bash
# e.g. a host pinned to an even older CUDA, or a different Blackwell arch
CUDA_VER=12.6 CUDA_MAJOR=12 COMPUTE=120 gpu/build_gpuburn.sh
CUDA_VER=12.6 CUDA_MAJOR=12              gpu/setup_libs.sh
```

| Override | Meaning | Default |
|----------|---------|---------|
| `CUDA_VER`   | CUDA version label (also written to `.cuda_version`) | `12.8` |
| `CUDA_MAJOR` | cuBLAS soname major (`libcublas.so.<MAJOR>`)         | `12`   |
| `COMPUTE`    | GPU arch for the compare kernel (`120` = sm_120)     | `120`  |

## Quick self-check on any host

```bash
nvidia-smi | grep "CUDA Version"                 # driver's max CUDA
cat gpu/lib/.cuda_version                         # what the libs were built for
file gpu/compare.fatbin                           # must say "ELF ... NVIDIA CUDA", not PTX text
strings gpu/compare.fatbin | grep -c '\.version'  # PTX directives — must be 0
```
Then `gpu/run_sm.bash gpuburn 30` → expect `[preflight] OK` and
`===GPU Stress Test Success===`.
