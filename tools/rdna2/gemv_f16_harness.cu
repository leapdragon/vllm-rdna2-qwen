// Standalone harness: fp16 skinny GEMM (M<=8 tokens) for gfx1030, wave-per-output-row design
// (same shape as the validated int4 MoE GEMV): lanes stride K with 16-byte loads, 4-deep unroll
// for memory-level parallelism, v_dot2_f32_f16 accumulate, wave32 shfl reduce.
// y[M,N] = x[M,K] . w[N,K]^T (+ bias[N])
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)

template <int WAVES, int MT>
__global__ void __launch_bounds__(WAVES * 32)
gemv_f16_rdna2_(const half* __restrict__ x, const half* __restrict__ w,
                const half* __restrict__ bias, half* __restrict__ y,
                const int N, const int K) {
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = blockIdx.x * WAVES + wave;
  if (n >= N) return;
  const int K8 = K / 8;
  const uint4* __restrict__ wrow = reinterpret_cast<const uint4*>(w + (size_t)n * K);
  const uint4* __restrict__ xr = reinterpret_cast<const uint4*>(x);
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) acc[m] = 0.f;
  // 4-deep unrolled K loop: issue all weight loads first (memory-level parallelism).
  for (int i = lane; i < K8; i += 32 * 4) {
    uint4 wq[4];
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int idx = i + 32 * u;
      wq[u] = (idx < K8) ? wrow[idx] : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int idx = i + 32 * u;
      if (idx < K8) {
        const half2* wh = reinterpret_cast<const half2*>(&wq[u]);
#pragma unroll
        for (int m = 0; m < MT; m++) {
          const uint4 xq = xr[(size_t)m * K8 + idx];
          const half2* xh = reinterpret_cast<const half2*>(&xq);
          float a = acc[m];
          a = __builtin_amdgcn_fdot2(wh[0], xh[0], a, false);
          a = __builtin_amdgcn_fdot2(wh[1], xh[1], a, false);
          a = __builtin_amdgcn_fdot2(wh[2], xh[2], a, false);
          a = __builtin_amdgcn_fdot2(wh[3], xh[3], a, false);
          acc[m] = a;
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < MT; m++) {
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) acc[m] += __shfl_xor(acc[m], off);
  }
  if (lane == 0) {
    const float b = bias ? __half2float(bias[n]) : 0.f;
#pragma unroll
    for (int m = 0; m < MT; m++) y[(size_t)m * N + n] = __float2half(acc[m] + b);
  }
}

template <int MT>
static void launch_mt(const half* x, const half* w, const half* bias, half* y, int N, int K, hipStream_t s) {
  // Fewer waves per block for small N so the grid still covers all 72 CUs.
  if (N >= 1152) {
    gemv_f16_rdna2_<8, MT><<<(N + 7) / 8, 256, 0, s>>>(x, w, bias, y, N, K);
  } else if (N >= 576) {
    gemv_f16_rdna2_<4, MT><<<(N + 3) / 4, 128, 0, s>>>(x, w, bias, y, N, K);
  } else if (N >= 288) {
    gemv_f16_rdna2_<2, MT><<<(N + 1) / 2, 64, 0, s>>>(x, w, bias, y, N, K);
  } else {
    gemv_f16_rdna2_<1, MT><<<N, 32, 0, s>>>(x, w, bias, y, N, K);
  }
}
static void gemv_f16_rdna2(const half* x, const half* w, const half* bias, half* y, int M, int N, int K, hipStream_t s) {
  switch (M) {
    case 1: launch_mt<1>(x, w, bias, y, N, K, s); break;
    case 2: launch_mt<2>(x, w, bias, y, N, K, s); break;
    case 3: launch_mt<3>(x, w, bias, y, N, K, s); break;
    case 4: launch_mt<4>(x, w, bias, y, N, K, s); break;
    case 5: launch_mt<5>(x, w, bias, y, N, K, s); break;
    case 6: launch_mt<6>(x, w, bias, y, N, K, s); break;
    case 7: launch_mt<7>(x, w, bias, y, N, K, s); break;
    case 8: launch_mt<8>(x, w, bias, y, N, K, s); break;
    default: printf("M must be 1..8\n"); exit(1);
  }
}

static inline float h2f(uint16_t h) { __half x; memcpy(&x, &h, 2); return __half2float(x); }
static inline uint16_t f2h(float f) { __half x = __float2half(f); uint16_t h; memcpy(&h, &x, 2); return h; }

int main(int argc, char** argv) {
  struct S { const char* name; int N, K; };
  S shapes[] = {{"gdn.in_proj_qkv/rank", 2560, 2560}, {"gdn.in_proj_z/rank", 1536, 2560},
                {"gdn.out_proj/rank", 2560, 1536}, {"qsa.q_proj/rank", 3072, 2560},
                {"qsa.o_proj/rank", 2560, 1536}, {"router.gate", 512, 2560},
                {"shared.gate_up", 1280, 2560}, {"shared.down", 2560, 640},
                {"hc.down", 320, 10240}, {"hc.up", 10240, 320}, {"hc.inject", 4, 10240},
                {"lm_head/rank", 62080, 2560}};
  const int Ms[] = {1, 4, 8};
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> uf(-1.f, 1.f);
  int fails = 0;
  for (int M : Ms) {
    printf("--- M=%d ---\n", M);
    double sum_us = 0;
    for (auto& sh : shapes) {
      const int N = sh.N, K = sh.K;
      std::vector<uint16_t> x((size_t)M * K), w((size_t)N * K), b(N), y((size_t)M * N);
      for (auto& v : x) v = f2h(uf(rng));
      for (auto& v : w) v = f2h(uf(rng) * 0.05f);
      for (auto& v : b) v = f2h(uf(rng) * 0.1f);
      // CPU ref (fp32), using the half-rounded inputs
      std::vector<float> ref((size_t)M * N);
      for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
          float a = 0;
          const uint16_t* xr = &x[(size_t)m * K]; const uint16_t* wr = &w[(size_t)n * K];
          for (int k = 0; k < K; k++) a += h2f(xr[k]) * h2f(wr[k]);
          ref[(size_t)m * N + n] = a + h2f(b[n]);
        }
      half *dx, *dw, *db, *dy;
      CHECK(hipMalloc(&dx, x.size() * 2)); CHECK(hipMalloc(&dw, w.size() * 2));
      CHECK(hipMalloc(&db, b.size() * 2)); CHECK(hipMalloc(&dy, y.size() * 2));
      CHECK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
      CHECK(hipMemcpy(dw, w.data(), w.size() * 2, hipMemcpyHostToDevice));
      CHECK(hipMemcpy(db, b.data(), b.size() * 2, hipMemcpyHostToDevice));
      gemv_f16_rdna2(dx, dw, db, dy, M, N, K, 0);
      CHECK(hipDeviceSynchronize());
      CHECK(hipMemcpy(y.data(), dy, y.size() * 2, hipMemcpyDeviceToHost));
      double maxref = 0, maxerr = 0;
      for (size_t i = 0; i < y.size(); i++) {
        maxref = fmax(maxref, fabs(ref[i]));
        maxerr = fmax(maxerr, fabs(h2f(y[i]) - ref[i]));
      }
      const bool ok = maxerr <= 2e-2 * maxref + 1e-2;
      if (!ok) fails++;
      // timing (weights cold-ish per shape: repeated, so L2-warm for small shapes — same as serving)
      for (int i = 0; i < 20; i++) gemv_f16_rdna2(dx, dw, db, dy, M, N, K, 0);
      CHECK(hipDeviceSynchronize());
      const int iters = 200;
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; i++) gemv_f16_rdna2(dx, dw, db, dy, M, N, K, 0);
      CHECK(hipDeviceSynchronize());
      auto t1 = std::chrono::steady_clock::now();
      double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
      sum_us += us;
      double gb = (double)N * K * 2 / 1e9;
      printf("  %-22s [%6d x %5d] %6.1f MB  %7.1f us  %4.0f GB/s  relerr %.1e  %s\n", sh.name, N, K, gb * 1e3, us,
             gb / us * 1e6, maxerr / (maxref + 1e-9), ok ? "PASS" : "FAIL");
      hipFree(dx); hipFree(dw); hipFree(db); hipFree(dy);
    }
    printf("  SUM %.0f us\n", sum_us);
  }
  printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
  return fails ? 1 : 0;
}
