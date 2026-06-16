# uqsm5090

Fork of the scitix `uqsm` v1.1 stress-test harness with one change: the GPU
module can run with **gpu-burn** instead of **dcgmi diag -r 4**, so it works on
RTX 5090 / Blackwell hosts where DCGM does not yet support the SKU.

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

## Building gpu-burn for sm_120 (RTX 5090)

The `gpu_burn` binary and `compare.fatbin` must live in `gpu/`. Use the
wrapper, which builds inside the CUDA 13 dev container:

```bash
sudo ~/uqsm5090/gpu/build_gpuburn.sh
```

`COMPUTE=120` targets Blackwell (sm_120). Upstream gpu-burn switched the
comparison kernel from `compare.ptx` to `compare.fatbin`; the wrapper copies
whichever is produced.

## Running gpu-burn (bare metal, no Docker)

gpu-burn runs **directly on the physical machine — Docker is not used at
runtime.** The only CUDA toolkit dependency is `libcublas.so.13` (which pulls
in `libcublasLt.so.13`); both are vendored under `gpu/lib/` and loaded via
`LD_LIBRARY_PATH`. The CUDA *driver* lib `libcuda.so.1` comes from the installed
NVIDIA driver — it is the user-space half of the kernel module and can never be
bundled or statically linked.

```bash
./uqsm.bash -dt gpu --gpu-tool gpuburn --gpu-duration 600   # no sudo needed
```

The libs (`gpu/lib/libcublas.so.13` + `libcublasLt.so.13`, ~565 MB) are too
large for GitHub's 100 MB limit, so they are **not** committed. They are already
bundled inside the OSS release tarball (below). After a bare `git clone`, fetch
them once — no Docker, no CUDA toolkit, just `curl`:

```bash
./gpu/setup_libs.sh        # downloads the cuBLAS pip wheel and extracts the two .so
```

`gpu/build_gpuburn.sh` (which *does* use a CUDA container) is only needed to
**rebuild** the `gpu_burn` binary itself; the prebuilt binary is committed, so
normal use never touches Docker.

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
