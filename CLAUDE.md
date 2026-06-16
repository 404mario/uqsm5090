# uqsm5090 — project notes

Fork of scitix `uqsm` v1.1 stress-test harness, with the GPU module able to use
**gpu-burn** instead of `dcgmi diag -r 4` so it works on RTX 5090 / Blackwell
(sm_120), where DCGM does not yet support the SKU.

## gpu-burn runs on bare metal (no Docker at runtime)

- The committed `gpu/gpu_burn` is a small g++ binary. Its only CUDA toolkit
  dependency is `libcublas.so.12` (+ transitive `libcublasLt.so.12`); both are
  vendored in `gpu/lib/` and loaded via `LD_LIBRARY_PATH` in `gpu/run_sm.bash`.
- `libcuda.so.1` (CUDA Driver API) is provided by the installed NVIDIA driver
  and must **never** be bundled or statically linked — it is the user-space half
  of the kernel module (`nvidia.ko`) and is version-locked to it.
- `gpu/lib/` is gitignored (libcublasLt.so.12 ≈ 717 MB > GitHub's 100 MB limit).
  Fetch it with `gpu/setup_libs.sh` (curl + pip-wheel extract, no Docker, no
  CUDA toolkit). The OSS release tarball already bundles these libs.
- `gpu/build_gpuburn.sh` *rebuilds* the binary with **g++ + pip wheels only** —
  no Docker, no nvcc, no sudo. Normal use never touches it.

## Why CUDA 12.8 (driver portability — the `(DIED!)` bug)

The vendored cuBLAS and the `gpu_burn` binary are built against **CUDA 12.8**,
deliberately, not the newest toolkit:

- 12.8 is the **first** CUDA toolkit that supports Blackwell / RTX 5090
  (`sm_120`), and it only requires NVIDIA driver **≥ 570** — while still running
  fine on every newer driver (e.g. 595 / CUDA 13.2). So **one 12.8 artifact is
  universal across the whole 5090 fleet.**
- A CUDA **13** build needs driver **≥ 580**. On a 570-era host, `cuInit`
  succeeds but the first `cublasCreate` in every worker fails, and gpu_burn
  reports only that each GPU `(DIED!)` — no hint why. This actually bit us on a
  570.148.08 host. Building against 12.8 avoids it entirely.
- `gpu/run_sm.bash` now runs a **driver preflight**: it compares the vendored
  cuBLAS major against `nvidia-smi`'s reported max CUDA version and aborts with
  an actionable message (rebuild against the driver's CUDA, or upgrade the
  driver) instead of letting the workers silently die.
- The scripts are **CUDA-major-agnostic**: nothing hardcodes `.so.12`. To target
  a different driver, rebuild/refetch with `CUDA_VER`/`CUDA_MAJOR` set, e.g.
  `CUDA_VER=12.6 CUDA_MAJOR=12 gpu/build_gpuburn.sh`. `gpu/lib/.cuda_version`
  records what the libs were built against.
- `gpu/compare.fatbin` is the device kernel. It is **NOT** CUDA-major-agnostic —
  a fatbin/cubin is toolchain-version-locked. `build_gpuburn.sh` **regenerates it
  every build** from `compare.cu` using `libnvrtc` (12.8), compiled straight to
  **`sm_120` SASS** (a raw cubin; `cuModuleLoad` accepts it). SASS has no PTX, so
  an older driver has nothing to JIT.
  - Pitfall that bit us: a CUDA-13-built `compare.fatbin` embeds `compute_120`
    **PTX** whose ISA version a 570 / CUDA-12.8 driver's JIT rejects →
    `cuModuleLoad` returns **222 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`** → every
    worker `(DIED!)`. Never reuse a fatbin built with a different toolchain.

## Publishing releases to OSS (scitix MinIO)

OSS is self-hosted **MinIO** (S3-compatible). Client is **`ossctl`** (Alibaba's
`mc`-style tool). Releases go to the public-read **`scitix-release`** bucket, the
same place `uqsm_v1.1.tar.gz` / `uqsm.tar.gz` live.

> **Credentials are NOT stored in this repo.** They live in `~/.ossctl/config.json`
> on the release host (and in the internal OSS ops doc). Use placeholders below.

```bash
# 1. Install ossctl (release host only)
curl -LO https://oss-ap-southeast.scitix.ai/scitix/packages/ossctl/latest/linux-amd64/ossctl
chmod +x ossctl && mv ossctl /usr/local/bin/    # or ~/.local/bin/

# 2. Configure aliases  (ossctl config add <name> <url> <AK> <SK>)
ossctl config add oss       https://oss-cn-shanghai.siflow.cn  <AK> <SK>   # Shanghai
ossctl config add oss-bench https://oss-ap-southeast.scitix.ai <AK> <SK>   # Malaysia

# 3. Upload   (cp <local> <alias>/<bucket>/<remote-path>)
ossctl cp uqsm5090.tar.gz oss/scitix-release/uqsm5090.tar.gz

# 4. List / inspect
ossctl ls   oss/scitix-release/
ossctl stat oss/scitix-release/uqsm5090.tar.gz
```

### Download (public, no client needed)

```bash
wget https://oss-cn-shanghai.siflow.cn/scitix-release/uqsm5090.tar.gz    # Shanghai
wget https://oss-ap-southeast.scitix.ai/scitix-release/uqsm5090.tar.gz   # Malaysia
```
