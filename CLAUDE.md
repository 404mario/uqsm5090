# uqsm5090 — project notes

Fork of scitix `uqsm` v1.1 stress-test harness, with the GPU module able to use
**gpu-burn** instead of `dcgmi diag -r 4` so it works on RTX 5090 / Blackwell
(sm_120), where DCGM does not yet support the SKU.

## gpu-burn runs on bare metal (no Docker at runtime)

- The committed `gpu/gpu_burn` is a small g++ binary. Its only CUDA toolkit
  dependency is `libcublas.so.13` (+ transitive `libcublasLt.so.13`); both are
  vendored in `gpu/lib/` and loaded via `LD_LIBRARY_PATH` in `gpu/run_sm.bash`.
- `libcuda.so.1` (CUDA Driver API) is provided by the installed NVIDIA driver
  and must **never** be bundled or statically linked — it is the user-space half
  of the kernel module (`nvidia.ko`) and is version-locked to it.
- `gpu/lib/` is gitignored (libcublasLt.so.13 ≈ 514 MB > GitHub's 100 MB limit).
  Fetch it with `gpu/setup_libs.sh` (curl + pip-wheel extract, no Docker, no
  CUDA toolkit). The OSS release tarball already bundles these libs.
- Docker is used **only** by `gpu/build_gpuburn.sh` to *rebuild* the binary
  (the host has no CUDA toolkit). Normal use never touches Docker.

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
