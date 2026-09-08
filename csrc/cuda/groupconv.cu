#include "../common/groupconv.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <initializer_list>

namespace {
constexpr int TX = 16, TY = 8, THREADS = TX * TY;

__device__ float input(const float *x, gc_shape p, int64_t n, int64_t ci,
                       int64_t h, int64_t w) {
    return h >= 0 && h < p.h && w >= 0 && w < p.w
        ? x[((n * p.cin + ci) * p.h + h) * p.w + w] : 0.0f;
}

// General grouped cross-correlation. One reduction belongs to one thread.
__global__ void naive(gc_shape p, const float *x, const float *w, float *y,
                      int64_t ho, int64_t wo, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += int64_t(gridDim.x) * blockDim.x) {
        const int64_t ow = i % wo, oh = (i / wo) % ho;
        const int64_t co = (i / (wo * ho)) % p.cout;
        const int64_t n = i / (wo * ho * p.cout);
        const int64_t cpg = p.cin / p.groups;
        const int64_t first = (co / (p.cout / p.groups)) * cpg;
        float sum = 0.0f;
        for (int64_t ci = 0; ci < cpg; ++ci)
            for (int64_t r = 0; r < p.r; ++r)
                for (int64_t s = 0; s < p.s; ++s) {
                    const float a = input(x, p, n, first + ci,
                        oh * p.sh - p.ph + r * p.dh,
                        ow * p.sw - p.pw + s * p.dw);
                    sum = fmaf(a, w[((co * cpg + ci) * p.r + r) * p.s + s], sum);
                }
        y[i] = sum;
    }
}

// Same NCHW layout and arithmetic: each block maps a 16 x 8 spatial tile.
__global__ void spatial(gc_shape p, const float *x, const float *w, float *y,
                        int64_t tiles_x, int64_t tiles_y, int64_t tiles) {
    for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const int64_t ow = (tile % tiles_x) * TX + threadIdx.x;
        const int64_t oh = ((tile / tiles_x) % tiles_y) * TY + threadIdx.y;
        const int64_t plane = tile / (tiles_x * tiles_y);
        const int64_t co = plane % p.cout, n = plane / p.cout;
        if (ow >= p.w || oh >= p.h) continue;
        const int64_t cpg = p.cin / p.groups, first = (co / cpg) * cpg;
        float sum = 0.0f;
        for (int64_t ci = 0; ci < cpg; ++ci)
            for (int r = 0; r < 3; ++r)
                for (int s = 0; s < 3; ++s)
                    sum = fmaf(input(x, p, n, first + ci, oh + r - 1, ow + s - 1),
                               w[(co * cpg + ci) * 9 + r * 3 + s], sum);
        y[(plane * p.h + oh) * p.w + ow] = sum;
    }
}

// Both depthwise_m1 (CPG=1) and grouped_small_cpg (CPG=2,4) retain
// explicit group-local reductions. All threads load halo and hit barriers.
template<int CPG>
__global__ void shared_halo(gc_shape p, const float *x, const float *w, float *y,
                            int64_t tiles_x, int64_t tiles_y, int64_t tiles) {
    __shared__ float halo[CPG][TY + 2][TX + 2];
    const int tid = threadIdx.y * TX + threadIdx.x;
    constexpr int area = (TY + 2) * (TX + 2);
    for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const int64_t left = (tile % tiles_x) * TX;
        const int64_t top = ((tile / tiles_x) % tiles_y) * TY;
        const int64_t plane = tile / (tiles_x * tiles_y);
        const int64_t co = plane % p.cout, n = plane / p.cout;
        const int64_t first = (co / CPG) * CPG;
        for (int j = tid; j < CPG * area; j += THREADS) {
            const int ci = j / area, h = (j % area) / (TX + 2), col = j % (TX + 2);
            halo[ci][h][col] = input(x, p, n, first + ci, top + h - 1, left + col - 1);
        }
        __syncthreads();
        const int64_t oh = top + threadIdx.y, ow = left + threadIdx.x;
        if (oh < p.h && ow < p.w) {
            float sum = 0.0f;
            #pragma unroll
            for (int ci = 0; ci < CPG; ++ci)
                #pragma unroll
                for (int r = 0; r < 3; ++r)
                    #pragma unroll
                    for (int s = 0; s < 3; ++s)
                        sum = fmaf(halo[ci][threadIdx.y + r][threadIdx.x + s],
                                   w[(co * CPG + ci) * 9 + r * 3 + s], sum);
            y[(plane * p.h + oh) * p.w + ow] = sum;
        }
        // Prevent the next grid-stride tile's load racing with current readers.
        __syncthreads();
    }
}

// Independent of shared_halo: four outputs reuse each six-element input
// row and each weight in registers. No shared memory or synchronization.
__global__ void register_outputs(gc_shape p, const float *x, const float *w, float *y,
                                 int64_t chunks_x, int64_t chunks) {
    for (int64_t j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         j < chunks; j += int64_t(gridDim.x) * blockDim.x) {
        const int64_t ow = (j % chunks_x) * 4;
        const int64_t oh = (j / chunks_x) % p.h;
        const int64_t plane = j / (chunks_x * p.h);
        const int64_t co = plane % p.cout, n = plane / p.cout;
        const int64_t cpg = p.cin / p.groups, first = (co / cpg) * cpg;
        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int64_t ci = 0; ci < cpg; ++ci) {
            #pragma unroll
            for (int r = 0; r < 3; ++r) {
                float row[6];
                #pragma unroll
                for (int t = 0; t < 6; ++t)
                    row[t] = input(x, p, n, first + ci, oh + r - 1, ow + t - 1);
                #pragma unroll
                for (int s = 0; s < 3; ++s) {
                    const float weight = w[(co * cpg + ci) * 9 + r * 3 + s];
                    #pragma unroll
                    for (int t = 0; t < 4; ++t)
                        sums[t] = fmaf(row[t + s], weight, sums[t]);
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < 4; ++t)
            if (ow + t < p.w) y[(plane * p.h + oh) * p.w + ow + t] = sums[t];
    }
}

int check_pointer(const gc_tensor *tensor, int current) {
    cudaPointerAttributes attr{};
    const cudaError_t err = cudaPointerGetAttributes(&attr, tensor->data);
    if (err != cudaSuccess) return gc_error(GC_DEVICE_ERROR, cudaGetErrorString(err));
    // Deliberately exclude managed and mapped-host storage from this contract.
    if (attr.type != cudaMemoryTypeDevice || attr.device != tensor->device_index ||
        attr.device != current)
        return gc_error(GC_INVALID, "CUDA pointer must be device memory on descriptor/current device");
    return GC_OK;
}

bool contiguous(const gc_tensor *t) {
    int64_t stride = 1;
    for (int k = 3; k >= 0; --k) {
        if (t->strides[k] != stride) return false;
        stride *= t->dims[k]; // common validator already checks tensor size overflow
    }
    return true;
}

unsigned blocks(int64_t count, int threads) {
    // Modest 1-D grids also support very large plane counts without grid.y limits.
    return unsigned(std::min<int64_t>(1 + (count - 1) / threads, 65535));
}
} // namespace

extern "C" int gc_cuda_available(void) {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

extern "C" int gc_cuda_conv(const gc_shape *p, const gc_tensor *x,
                             const gc_tensor *w, gc_tensor *y, int implementation,
                             void *stream) {
    int status = gc_validate_tensors(p, x, w, y, GC_CUDA, GC_F32);
    if (status != GC_OK) return status;
    if (!contiguous(x) || !contiguous(w) || !contiguous(y))
        return gc_error(GC_UNSUPPORTED, "CUDA kernels require contiguous NCHW tensors");
    if (implementation < 0 || implementation > 3)
        return gc_error(GC_UNSUPPORTED, "Unknown CUDA implementation");
    const bool b = p->cin == p->cout && p->r == 3 && p->s == 3 &&
                   p->sh == 1 && p->sw == 1 && p->ph == 1 && p->pw == 1 &&
                   p->dh == 1 && p->dw == 1;
    if (implementation != 0 && !b)
        return gc_error(GC_UNSUPPORTED, "Optimized CUDA kernels require B geometry");
    const int64_t cpg = p->cin / p->groups;
    if (implementation == 2 && cpg != 1 && cpg != 2 && cpg != 4)
        return gc_error(GC_UNSUPPORTED, "CUDA shared halo supports cpg 1, 2, or 4");
    int current = -1;
    cudaError_t err = cudaGetDevice(&current);
    if (err != cudaSuccess) return gc_error(GC_DEVICE_ERROR, cudaGetErrorString(err));
    for (const gc_tensor *t : {x, w, static_cast<const gc_tensor *>(y)}) {
        status = check_pointer(t, current);
        if (status != GC_OK) return status;
    }
    // Querying stream flags is rejected during CUDA graph capture (CUDA 12.8).
    // Capture-status lookup validates the handle in both eager and captured use.
    // A cross-device stream is additionally rejected by CUDA at kernel launch.
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cudaStream_t caller_stream = reinterpret_cast<cudaStream_t>(stream);
    err = cudaStreamIsCapturing(caller_stream, &capture_status);
    if (err != cudaSuccess) return gc_error(GC_DEVICE_ERROR, cudaGetErrorString(err));
    int64_t ho = 0, wo = 0;
    status = gc_validate_shape(p, &ho, &wo);
    if (status != GC_OK) return status;
    const float *xp = static_cast<const float *>(x->data);
    const float *wp = static_cast<const float *>(w->data);
    float *yp = static_cast<float *>(y->data);
    if (implementation == 0) {
        const int64_t count = p->n * p->cout * ho * wo;
        naive<<<blocks(count, 256), 256, 0, caller_stream>>>(*p, xp, wp, yp, ho, wo, count);
    } else if (implementation == 3) {
        const int64_t chunks_x = 1 + (p->w - 1) / 4;
        const int64_t chunks = p->n * p->cout * p->h * chunks_x;
        register_outputs<<<blocks(chunks, 128), 128, 0, caller_stream>>>(*p, xp, wp, yp, chunks_x, chunks);
    } else {
        const int64_t nx = 1 + (p->w - 1) / TX, ny = 1 + (p->h - 1) / TY;
        const int64_t tiles = p->n * p->cout * nx * ny;
        const dim3 block(TX, TY);
        const unsigned grid = blocks(tiles, 1);
        if (implementation == 1)
            spatial<<<grid, block, 0, caller_stream>>>(*p, xp, wp, yp, nx, ny, tiles);
        else if (cpg == 1)
            shared_halo<1><<<grid, block, 0, caller_stream>>>(*p, xp, wp, yp, nx, ny, tiles);
        else if (cpg == 2)
            shared_halo<2><<<grid, block, 0, caller_stream>>>(*p, xp, wp, yp, nx, ny, tiles);
        else
            shared_halo<4><<<grid, block, 0, caller_stream>>>(*p, xp, wp, yp, nx, ny, tiles);
    }
    err = cudaGetLastError();
    return err == cudaSuccess ? gc_error(GC_OK, "")
                             : gc_error(GC_DEVICE_ERROR, cudaGetErrorString(err));
}
