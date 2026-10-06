# h3_cutlass — vendored CUTLASS int8 GEMM extension

`_C.abi3.so` is the fused CUTLASS int8 GEMM extension used by the convrot
int8 forward/backward arms (`_get_cutlass_int8` in `toolkit/util/convrot_quant.py`).
This directory keeps one copy next to the loader so a fresh clone can use it
without installing the source package; if the file is absent the loader falls
back to the `comfy_kitchen` wheel unchanged.

## The shipped binary

Built from https://github.com/Comfy-Org/comfy-kitchen at commit
`be003b7c23c5b01328657955b8bc5d3f073d868e` (package version 0.2.37, Apache-2.0)
— stock sources, no out-of-tree kernels; the exported op set is identical to
this commit. Submodule pins: `third_party/cutlass`
@d4b4b494c3c51bf6507e7ab09fbafd1e9fa94f39 (what the int8 kernel compiles
against) and `third_party/flash-attention` @979702c87a8713a8e0a5e9fee122b90d2ef13be5
(feeds other ops in the same CMake target, not `cutlass_int8_dequant`).

Current file: **rebuilt here sm_120-only, stripped**, 9.2 MB,
`sha256 3fd92311e8d74945820fd5252492a0f93fbd12d5e09148622bcb5ba9f919f6ca`
(RUNPATH cleared with patchelf after link so the binary carries no
build-time paths; the int8 GEMM was re-verified bitwise after the strip).
Verified against the original build (38 MB wheel artifact,
`sha256 d19debb2…8f988`, kept in site-packages and rebuilt-candidate backups):
all 70 exported ops identical, and the int8 GEMM output is bitwise-equal
(full-range int8 harness, D = acc * xs[m] * ws[n], checksum 3800886322048)
and to `torch._int_mm`. The original shipped the Linux-default arch set
(sm_75-real/80-real/89(+PTX)/90a-real/100f/120f ≈ 33 MB of device code) plus a
HIP extension; neither is loaded on an sm_120 fleet.

The runtime package has no Python dependencies; `nvidia-cublas` is an
optional extra for other ops, not required by the int8 arm (the extension
links only glibc/libstdc++ and resolves CUDA via the torch install).

## Rebuild recipe (verified on this machine)

    git clone https://github.com/Comfy-Org/comfy-kitchen && cd comfy-kitchen
    git checkout be003b7c23c5b01328657955b8bc5d3f073d868e
    git submodule update --init third_party/cutlass third_party/flash-attention
    python -m venv venv && venv/bin/pip install "setuptools>=61" wheel "nanobind>=2.0" cmake "ninja"
    COMFY_CUDA_ARCHS="120-real" venv/bin/pip wheel --no-build-isolation -w dist .

Environment constraints found the hard way:

- Toolchain must be CUDA >= 12.8 for sm_120; this machine's CUDA 13.1 + gcc-15
  is broken (mathcalls.h exception-spec conflict in CMake's compiler id).
  CUDA 13.2 was assembled from NVIDIA redistributables
  (redistrib_13.2.0.json): `cuda_nvcc` 13.2.51, `libnvvm` 13.2.51 (nvcc needs
  `nvvm/bin/cicc`), `cuda_cudart` 13.2.51 (headers + libcudart_static.a),
  `cuda_crt` 13.2.51, `cuda_cccl` 13.2.27; merged flat into one root, plus
  `lib64 -> lib` symlink (CMake expects lib64; the redist ships lib) and
  `cublas*.h` headers (two ops include them; the cublas lib is dlopen'd at
  runtime, never linked — the headers were symlinked from the installed
  `nvidia/cu13` pip package instead of the 802 MB libcublas redist).
- Host compiler must be pinned: `CUDAHOSTCXX=/usr/bin/g++-15` (system gcc
  defaults to 16; nvcc 13.2 caps at 15).
- `COMFY_CUDA_ARCHS` is the setup.py env override consumed by CMake;
  `120-real` yields cubin-only, no PTX. HIP is off by default
  (`COMFY_KITCHEN_BUILD_HIP` unset), which drops the second `_C.abi3.so`
  the original wheel carried.
- After linking, clear the RUNPATH (`patchelf --remove-rpath _C.abi3.so`)
  so no staging path is embedded, and re-run the bitwise int8 check.

`cutlass_gemm_int8.cu`, `cutlass_gemm_common.cuh`, `dlpack_bindings.cpp` (and
its two local headers) are kept here for reference: they document the int8 NT
kernel and the `cutlass_int8_dequant` binding the trainer calls. They are not
compiled by the toolkit at runtime.
