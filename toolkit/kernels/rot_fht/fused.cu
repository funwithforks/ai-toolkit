#include <torch/extension.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

// One CTA per row. The row is staged in shared memory (fp32), the block
// regular-Hadamard rotation runs as 4 in-place butterfly stages (each thread
// owns whole (g, s) groups, so no race and no second buffer), the per-row
// amax comes from a block reduction, and the int8 write is the only global
// store. One read of x, one write of q, matching the 171 MB fused floor.

__global__ void rot_fht_quant_kernel(
    const __nv_bfloat16* __restrict__ x,   // (M, K)
    int8_t* __restrict__ q,                // (Mpad, K) — rows >= M left as-is
    float* __restrict__ scales,            // (Mpad,)
    int M,
    int K,
    int qmax)
{
    extern __shared__ float smem[];        // K floats row + blockDim.x reduction
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    float* rowbuf = smem;                  // K floats
    float* red = smem + K;                 // nt floats reduction scratch

    for (int i = tid; i < K; i += nt)
        rowbuf[i] = __bfloat162float(x[(long)row * K + i]);
    __syncthreads();

    // 4 butterfly stages; stage p combines groups of 4 elements at stride 4^p.
    // Each thread owns whole (g, s) groups: reads its group's 4 inputs into
    // registers, writes the 4 outputs — in-place safe.
    const float HALF = 0.5f;
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const long stride = 1L << (2 * p);          // 4^p
        const long groups = (long)K / 4;            // K/4 groups every stage
        for (long gid = tid; gid < groups; gid += nt) {
            const long g = gid / (1L << (2 * p));
            const long s = gid % (1L << (2 * p));
            const long base = g * (1L << (2 * p + 2)) + s;
            float in[4];
            #pragma unroll
            for (int c = 0; c < 4; ++c)
                in[c] = rowbuf[base + c * stride];
            #pragma unroll
            for (int a = 0; a < 4; ++a) {
                float acc = 0.f;
                #pragma unroll
                for (int c = 0; c < 4; ++c)
                    acc += ((a + c == 3) ? -1.f : 1.f) * in[c];
                rowbuf[base + a * stride] = acc * HALF;
            }
        }
        __syncthreads();
    }

    // per-row amax -> block reduction
    float local = 0.f;
    for (int i = tid; i < K; i += nt)
        local = fmaxf(local, fabsf(rowbuf[i]));
    red[tid] = local;
    __syncthreads();
    for (int off = nt / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
        __syncthreads();
    }
    float amax = red[0];
    float scale = (amax / qmax > 0.f) ? amax / qmax : 1.0f;
    if (tid == 0) scales[row] = scale;
    __syncthreads();

    for (int i = tid; i < K; i += nt) {
        float v = rowbuf[i];
        float qq = nearbyintf(v / scale);   // half-to-even, matches torch.round
        qq = fminf(fmaxf(qq, -(float)qmax), (float)qmax);
        q[(long)row * K + i] = (int8_t)qq;
    }
}

std::vector<at::Tensor> rot_fht_quant(
    at::Tensor x,       // (M, K) bf16, cuda
    at::Tensor q,       // (Mpad, K) int8 — pre-zeroed
    at::Tensor scales,  // (Mpad,) fp32 — pre-filled with 1.0
    int64_t qmax)
{
    const int M = x.size(0);
    const int K = x.size(1);
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bf16");
    TORCH_CHECK(K % 256 == 0, "K must be a multiple of 256");
    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    const size_t smem = (size_t)K * sizeof(float) + threads * sizeof(float);
    cudaFuncSetAttribute(rot_fht_quant_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    rot_fht_quant_kernel<<<M, threads, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        q.data_ptr<int8_t>(),
        scales.data_ptr<float>(),
        M, K, (int)qmax);
    return {q, scales};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rot_fht_quant", &rot_fht_quant, "fused rotate+hadamard+quant");
}
