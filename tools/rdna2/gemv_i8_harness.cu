// int8 weight-only skinny GEMM for gfx1030 decode (M <= 8): y[M,N] = x[M,K] . (w_i8[N,K] * s[N])^T
// Same wave-per-row design as gemv_f16_rdna2, but each 16-byte load carries 16 weights.
// Per-output-channel symmetric scale (fp16). Compared against the fp16 kernel (bytes halve).
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

__device__ __forceinline__ half2 i8x2_to_h2(int lo, int hi) {
  return __floats2half2_rn((float)lo, (float)hi);
}

template <int WAVES, int MT>
__global__ void __launch_bounds__(WAVES * 32)
gemv_i8_rdna2_(const half* __restrict__ x, const int8_t* __restrict__ w, const half* __restrict__ scale,
               const half* __restrict__ bias, half* __restrict__ y, const int N, const int K) {
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = blockIdx.x * WAVES + wave;
  if (n >= N) return;
  const int K16 = K / 16;
  const uint4* __restrict__ wrow = reinterpret_cast<const uint4*>(w + (size_t)n * K);
  const uint4* __restrict__ xr = reinterpret_cast<const uint4*>(x);   // 8 halves per uint4
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) acc[m] = 0.f;
  for (int i = lane; i < K16; i += 32 * 4) {
    uint4 wq[4];
#pragma unroll
    for (int u = 0; u < 4; u++) { const int idx = i + 32 * u; wq[u] = (idx < K16) ? wrow[idx] : make_uint4(0, 0, 0, 0); }
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int idx = i + 32 * u;
      if (idx < K16) {
        // unpack 16 int8 -> 8 half2
        const int32_t* q = reinterpret_cast<const int32_t*>(&wq[u]);
        half2 wh[8];
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const int32_t v = q[j];
          wh[2 * j]     = i8x2_to_h2((int8_t)(v & 0xff), (int8_t)((v >> 8) & 0xff));
          wh[2 * j + 1] = i8x2_to_h2((int8_t)((v >> 16) & 0xff), (int8_t)((v >> 24) & 0xff));
        }
#pragma unroll
        for (int m = 0; m < MT; m++) {
          const uint4 xa = xr[((size_t)m * K16 + idx) * 2];
          const uint4 xb = xr[((size_t)m * K16 + idx) * 2 + 1];
          const half2* xh0 = reinterpret_cast<const half2*>(&xa);
          const half2* xh1 = reinterpret_cast<const half2*>(&xb);
          float a = acc[m];
          a = __builtin_amdgcn_fdot2(wh[0], xh0[0], a, false);
          a = __builtin_amdgcn_fdot2(wh[1], xh0[1], a, false);
          a = __builtin_amdgcn_fdot2(wh[2], xh0[2], a, false);
          a = __builtin_amdgcn_fdot2(wh[3], xh0[3], a, false);
          a = __builtin_amdgcn_fdot2(wh[4], xh1[0], a, false);
          a = __builtin_amdgcn_fdot2(wh[5], xh1[1], a, false);
          a = __builtin_amdgcn_fdot2(wh[6], xh1[2], a, false);
          a = __builtin_amdgcn_fdot2(wh[7], xh1[3], a, false);
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
    const float sc = __half2float(scale[n]);
    const float b = bias ? __half2float(bias[n]) : 0.f;
#pragma unroll
    for (int m = 0; m < MT; m++) y[(size_t)m * N + n] = __float2half(acc[m] * sc + b);
  }
}

template <int MT>
static void launch_i8(const half* x, const int8_t* w, const half* s, const half* bias, half* y, int N, int K, hipStream_t st) {
  if (N >= 1152) gemv_i8_rdna2_<8, MT><<<(N + 7) / 8, 256, 0, st>>>(x, w, s, bias, y, N, K);
  else if (N >= 576) gemv_i8_rdna2_<4, MT><<<(N + 3) / 4, 128, 0, st>>>(x, w, s, bias, y, N, K);
  else if (N >= 288) gemv_i8_rdna2_<2, MT><<<(N + 1) / 2, 64, 0, st>>>(x, w, s, bias, y, N, K);
  else gemv_i8_rdna2_<1, MT><<<N, 32, 0, st>>>(x, w, s, bias, y, N, K);
}
static void gemv_i8(const half* x, const int8_t* w, const half* s, const half* bias, half* y, int M, int N, int K, hipStream_t st) {
  switch (M) { case 1: launch_i8<1>(x,w,s,bias,y,N,K,st); break; case 4: launch_i8<4>(x,w,s,bias,y,N,K,st); break;
               case 8: launch_i8<8>(x,w,s,bias,y,N,K,st); break; default: printf("M\n"); exit(1); }
}
static inline float h2f(uint16_t h) { __half x; memcpy(&x, &h, 2); return __half2float(x); }
static inline uint16_t f2h(float f) { __half x = __float2half(f); uint16_t h; memcpy(&h, &x, 2); return h; }

int main() {
  struct S { const char* name; int N, K; };
  S shapes[] = {{"gdn.in_proj_qkv/rank", 2560, 2560}, {"gdn.in_proj_z/rank", 1536, 2560}, {"qsa.q_proj/rank", 3072, 2560},
                {"qsa.o_proj/rank", 2560, 1536}, {"router.gate", 512, 2560}, {"hc.down", 320, 10240}, {"hc.up", 10240, 320},
                {"lm_head/rank", 62080, 2560}};
  std::mt19937 rng(11); std::uniform_real_distribution<float> uf(-1.f, 1.f);
  int fails = 0;
  for (int M : {1, 4, 8}) {
    printf("--- M=%d ---\n", M); double sum = 0;
    for (auto& sh : shapes) {
      const int N = sh.N, K = sh.K;
      // fp16 "original" weights, then per-row symmetric int8 quantisation
      std::vector<float> wf((size_t)N * K); for (auto& v : wf) v = uf(rng) * 0.05f;
      std::vector<int8_t> w8((size_t)N * K); std::vector<uint16_t> sc(N);
      for (int n = 0; n < N; n++) { float mx = 0; for (int k = 0; k < K; k++) mx = fmaxf(mx, fabsf(wf[(size_t)n * K + k]));
        float s = mx / 127.f; sc[n] = f2h(s); float si = h2f(sc[n]);
        for (int k = 0; k < K; k++) { int q = (int)lrintf(wf[(size_t)n * K + k] / si); q = q > 127 ? 127 : (q < -127 ? -127 : q); w8[(size_t)n * K + k] = (int8_t)q; } }
      std::vector<uint16_t> x((size_t)M * K), y((size_t)M * N); for (auto& v : x) v = f2h(uf(rng));
      std::vector<float> ref((size_t)M * N);   // reference = dequantised int8 weights (kernel exactness), fp32 accumulate
      for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) { float a = 0; for (int k = 0; k < K; k++) a += h2f(x[(size_t)m * K + k]) * (float)w8[(size_t)n * K + k]; ref[(size_t)m * N + n] = a * h2f(sc[n]); }
      half *dx, *dsc, *dy; int8_t* dw;
      CHECK(hipMalloc(&dx, x.size() * 2)); CHECK(hipMalloc(&dw, w8.size())); CHECK(hipMalloc(&dsc, N * 2)); CHECK(hipMalloc(&dy, y.size() * 2));
      CHECK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice)); CHECK(hipMemcpy(dw, w8.data(), w8.size(), hipMemcpyHostToDevice));
      CHECK(hipMemcpy(dsc, sc.data(), N * 2, hipMemcpyHostToDevice));
      gemv_i8(dx, dw, dsc, nullptr, dy, M, N, K, 0); CHECK(hipDeviceSynchronize());
      CHECK(hipMemcpy(y.data(), dy, y.size() * 2, hipMemcpyDeviceToHost));
      double maxref = 0, maxerr = 0; for (size_t i = 0; i < y.size(); i++) { maxref = fmax(maxref, fabs(ref[i])); maxerr = fmax(maxerr, fabs(h2f(y[i]) - ref[i])); }
      bool ok = maxerr <= 2e-2 * maxref + 1e-2; if (!ok) fails++;
      for (int i = 0; i < 20; i++) gemv_i8(dx, dw, dsc, nullptr, dy, M, N, K, 0); CHECK(hipDeviceSynchronize());
      auto t0 = std::chrono::steady_clock::now(); for (int i = 0; i < 200; i++) gemv_i8(dx, dw, dsc, nullptr, dy, M, N, K, 0); CHECK(hipDeviceSynchronize());
      double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 200; sum += us;
      double gb = (double)N * K / 1e9;
      printf("  %-22s [%6d x %5d] %6.1f MB(i8) %7.1f us  %4.0f GB/s  relerr %.1e  %s\n", sh.name, N, K, gb * 1e3, us, gb / us * 1e6, maxerr / (maxref + 1e-9), ok ? "PASS" : "FAIL");
      hipFree(dx); hipFree(dw); hipFree(dsc); hipFree(dy);
    }
    printf("  SUM %.0f us\n", sum);
  }
  printf("%s\n", fails ? "SOME FAILED" : "ALL PASS"); return fails;
}
