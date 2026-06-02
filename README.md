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

## Running gpu-burn

If the host has CUDA 13 user-space libs (`libcublas.so.13`, `libcudart.so.13`),
the binary runs directly. Otherwise the runner falls back to the CUDA 13
runtime container with `--gpus all`, so the driver is passed through but the
libs come from the image. This is the default on this host.

The runner uses `sudo docker` when the current user is not in the `docker`
group, so the whole harness should be invoked with `sudo` for the gpu-burn
path:

```bash
sudo ./uqsm.bash -dt gpu --gpu-tool gpuburn --gpu-duration 600
```

Override the image with the `GPUBURN_IMAGE` env var if you need a different
CUDA version.

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
