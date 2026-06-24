# uqsm5090

Fork of the scitix `uqsm` v1.1 stress-test harness with one change: the GPU
module can run with **gpu-burn** instead of **dcgmi diag -r 4**, so it works on
RTX 5090 / Blackwell hosts where DCGM does not yet support the SKU.

## Getting it — use the OSS tarball (recommended), not `git clone`

There are two sources and they are **not** equivalent:

| Source | gpu-burn CUDA libs included? | Ready to run? |
|--------|------------------------------|---------------|
| **OSS tarball (recommended)** | ✅ yes — `gpu/lib/*.so.12` are bundled | **Unzip and run, no extra step** |
| `git clone` (developers) | ❌ no — `libcublasLt.so.12` is ~717 MB, over GitHub's 100 MB/file limit | must run `gpu/setup_libs.sh` once to fetch the libs |

**Recommended (unzip-and-run):**
```bash
wget https://oss-cn-shanghai.siflow.cn/scitix-release/uqsm5090_v1.1.tar.gz   # Shanghai
#   or: https://oss-ap-southeast.scitix.ai/scitix-release/uqsm5090_v1.1.tar.gz  # Malaysia mirror
tar xzf uqsm5090_v1.1.tar.gz && cd uqsm5090
./gpu/run_sm.bash gpuburn 60        # runs immediately — libs are already in gpu/lib/
```

**From `git clone` (only if you are modifying the code):**
```bash
git clone https://github.com/404mario/uqsm5090 && cd uqsm5090
./gpu/setup_libs.sh                 # one-time: downloads the cuBLAS libs (~828 MB) that can't live in git
./gpu/run_sm.bash gpuburn 60
```

The big cuBLAS `.so` files physically cannot be committed to GitHub (single-file
100 MB limit), which is why a clone needs `setup_libs.sh` but the OSS tarball
does not. **If you just want to run the test, use the OSS tarball.**

> **Hit `(DIED!)`, or unsure which build matches your driver?** See
> [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md) — the driver ⇄ CUDA ⇄ artifact
> matching matrix and what the runtime preflight checks. Background:
> [`docs/POSTMORTEM-rtx5090-gpuburn.md`](docs/POSTMORTEM-rtx5090-gpuburn.md).

## Modules
- `cpu/`   stress-ng `--matrix 0 -t 10m` (unchanged)
- `mem/`   memtester via `memtester_loop.sh 88 32 65536` (unchanged, ~7h)
- `ib/`    `ubimonitor -t -q 10` (unchanged)
- `gpu/`   **either** `dcgmi diag -r 4` **or** `gpu_burn -d <sec>`

## Usage

```bash
./uqsm.bash                                    # all modules, auto GPU tool
./uqsm.bash -dt gpu                            # GPU only, auto tool
./uqsm.bash -dt gpu --gpu-tool gpuburn         # force gpu-burn
./uqsm.bash -dt gpu --gpu-tool dcgm            # force dcgmi diag -r 4
./uqsm.bash --gpu-tool gpuburn --gpu-duration 1800
```

Auto-detect rule: if `nvidia-smi -L` reports any RTX 50xx / Blackwell card the
default is `gpuburn`; otherwise it is `dcgm`. Override with `--gpu-tool`.

Results are written to `result/<module>_result.json` and rolled up into
`result/report.json`.

## Building gpu-burn for sm_120 (RTX 5090) — no Docker, no nvcc

The `gpu_burn` binary and `compare.fatbin` must live in `gpu/`. The build
wrapper needs only **g++, python3, and PyPI access** — no Docker, no CUDA
toolkit, no nvcc, no sudo:

```bash
~/uqsm5090/gpu/build_gpuburn.sh
```

It pulls the CUDA **12.8** headers, cuBLAS `.so`, and `libnvrtc` from NVIDIA's
pip wheels, compiles the g++ driver, links against `libcublas.so.12`, **and
regenerates `gpu/compare.fatbin` from `compare.cu` as `sm_120` SASS via nvrtc**
(no prebuilt kernel is reused — a fatbin is toolchain-version-locked, and a
CUDA-13 one fails on a 570 driver with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`).

**Why CUDA 12.8 and not the newest toolkit?** 12.8 is the first toolkit that
supports Blackwell `sm_120` and only requires NVIDIA driver **≥ 570**, yet it
also runs on every newer driver — so a single 12.8 build is portable across the
whole RTX 5090 fleet. A CUDA 13 build needs driver ≥ 580 and otherwise kills
every gpu_burn worker (`(DIED!)`) on a 570-era host. `run_sm.bash` runs a driver
preflight and the scripts are CUDA-major-agnostic (override `CUDA_VER` /
`CUDA_MAJOR` to target a different driver); see `CLAUDE.md` for details.

## Running gpu-burn (bare metal, no Docker)

gpu-burn runs **directly on the physical machine — Docker is not used at
runtime.** The only CUDA toolkit dependency is `libcublas.so.12` (which pulls
in `libcublasLt.so.12`); both are vendored under `gpu/lib/` and loaded via
`LD_LIBRARY_PATH`. The CUDA *driver* lib `libcuda.so.1` comes from the installed
NVIDIA driver — it is the user-space half of the kernel module and can never be
bundled or statically linked.

```bash
./uqsm.bash -dt gpu --gpu-tool gpuburn --gpu-duration 600   # no sudo needed
```

The libs (`gpu/lib/libcublas.so.12` + `libcublasLt.so.12`, ~828 MB) are too
large for GitHub's 100 MB limit, so they are **not** committed. They are already
bundled inside the OSS release tarball (below). After a bare `git clone`, fetch
them once — no Docker, no CUDA toolkit, just `curl`:

```bash
./gpu/setup_libs.sh        # downloads the cuBLAS pip wheel and extracts the two .so
```

`gpu/build_gpuburn.sh` is only needed to **rebuild** the `gpu_burn` binary
itself; the prebuilt binary is committed, so normal use never rebuilds anything.

## Using the dcgm backend (non-Blackwell hosts)

The DCGM `.deb` for CUDA 13 is 419 MB and is **not committed to git** (exceeds
GitHub's 100 MB single-file limit). Download manually and drop into `gpu/`
before running with `--gpu-tool dcgm`:

```bash
# DCGM 4.5.2 packages for Ubuntu 22.04 / amd64
cd ~/uqsm5090/gpu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/datacenter-gpu-manager-4-cuda13_4.5.2-1_amd64.deb
# (core .deb is small and already committed)
```

Or use any DCGM build matching your CUDA major version; `gpu/run_sm.bash`
globs `datacenter-gpu-manager-4-*_amd64.deb`.

## Smoke test result (8x RTX 5090, 600 s, gpu-burn)

```
Tested 8 GPUs:
    GPU 0..7: OK
===GPU Stress Test Success===
```
