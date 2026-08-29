// Standalone harness: EP-aware skinny int4 MoE GEMV (moe_w13_silu_gemv_ / moe_w2_gemv_)
// Validates the "-1 = non-local expert, skip" contract against a CPU reference and times it
// at Qwen3.8-Flash-Next decode shapes (K=2560 hidden, N=640 inter, topk=10, 128 local experts).
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

template <int WAVES>
__global__ void moe_w13_silu_gemv_(
    const half* __restrict__ input, const uint32_t* __restrict__ w13,
    const half* __restrict__ s13, const int32_t* __restrict__ topk_ids,
    half* __restrict__ act, const int K, const int N, const int topk,
    const int group_size) {
  const int m = blockIdx.z, s = blockIdx.y;
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = blockIdx.x * WAVES + wave;
  extern __shared__ half xs[];
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = input[m * K + i];
  __syncthreads();
  if (n >= N) return;
  const int expert = topk_ids[m * topk + s];
  // EP: expert < 0 means "not resident on this rank" — contribute nothing.
  if (expert < 0) {
    if (lane == 0) act[((uint64_t)m * topk + s) * N + n] = __float2half(0.f);
    return;
  }
  const int K8 = K / 8, KG = K / group_size;
  const uint64_t base = (uint64_t)expert * 2 * N;
  const uint32_t* wg = w13 + (base + n) * K8;
  const uint32_t* wu = w13 + (base + N + n) * K8;
  const half* sg = s13 + (base + n) * KG;
  const half* su = s13 + (base + N + n) * KG;
  float accg = 0.f, accu = 0.f;
  for (int i = lane; i < K8; i += 32) {
    const uint32_t qg = wg[i], qu = wu[i];
    const int k0 = i * 8;
    float pg = 0.f, pu = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const float xv = __half2float(xs[k0 + j]);
      pg += (float)((int)((qg >> (4 * j)) & 0xF) - 8) * xv;
      pu += (float)((int)((qu >> (4 * j)) & 0xF) - 8) * xv;
    }
    const int g = k0 / group_size;
    accg += pg * __half2float(sg[g]);
    accu += pu * __half2float(su[g]);
  }
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
    accg += __shfl_xor(accg, off);
    accu += __shfl_xor(accu, off);
  }
  if (lane == 0) {
    const float silu = accg / (1.f + __expf(-accg));
    act[((uint64_t)m * topk + s) * N + n] = __float2half(silu * accu);
  }
}

template <int WAVES>
__global__ void moe_w2_gemv_(
    const half* __restrict__ act, const uint32_t* __restrict__ w2,
    const half* __restrict__ s2, const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_w, half* __restrict__ out, const int N,
    const int H, const int topk, const int group_size) {
  const int m = blockIdx.z;
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int h = blockIdx.x * WAVES + wave;
  extern __shared__ half as[];
  for (int i = threadIdx.x; i < topk * N; i += blockDim.x)
    as[i] = act[(uint64_t)m * topk * N + i];
  __syncthreads();
  if (h >= H) return;
  const int N8 = N / 8, NG = N / group_size;
  float acc = 0.f;
  for (int s = 0; s < topk; s++) {
    const int expert = topk_ids[m * topk + s];
    if (expert < 0) continue;  // EP: non-local expert
    const uint32_t* wrow = w2 + ((uint64_t)expert * H + h) * N8;
    const half* srow = s2 + ((uint64_t)expert * H + h) * NG;
    const half* xrow = as + s * N;
    float sacc = 0.f;
    for (int i = lane; i < N8; i += 32) {
      const uint32_t q = wrow[i];
      const int k0 = i * 8;
      float p = 0.f;
#pragma unroll
      for (int j = 0; j < 8; j++)
        p += (float)((int)((q >> (4 * j)) & 0xF) - 8) * __half2float(xrow[k0 + j]);
      sacc += p * __half2float(srow[k0 / group_size]);
    }
    acc += sacc * topk_w[m * topk + s];
  }
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) acc += __shfl_xor(acc, off);
  if (lane == 0) out[(uint64_t)m * H + h] = __float2half(acc);
}

static inline float h2f(uint16_t h) { __half x; memcpy(&x, &h, 2); return __half2float(x); }
static inline uint16_t f2h(float f) { __half x = __float2half(f); uint16_t h; memcpy(&h, &x, 2); return h; }

int main(int argc, char** argv) {
  const int K = 2560, N = 640, topk = 10, E = 128, G = 128;
  const int M = argc > 1 ? atoi(argv[1]) : 4;
  const float local_frac = argc > 2 ? atof(argv[2]) : 0.25f;  // EP=4 → ~1/4 of experts local
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> nib(0, 15);
  std::uniform_real_distribution<float> uf(-1.f, 1.f);

  const size_t K8 = K / 8, N8 = N / 8, KG = K / G, NG = N / G;
  std::vector<uint32_t> w13((size_t)E * 2 * N * K8), w2((size_t)E * K * N8);
  std::vector<uint16_t> s13((size_t)E * 2 * N * KG), s2((size_t)E * K * NG), x((size_t)M * K);
  for (auto& v : w13) { v = 0; for (int j = 0; j < 8; j++) v |= (uint32_t)nib(rng) << (4 * j); }
  for (auto& v : w2)  { v = 0; for (int j = 0; j < 8; j++) v |= (uint32_t)nib(rng) << (4 * j); }
  for (auto& v : s13) v = f2h(uf(rng) * 0.02f);
  for (auto& v : s2)  v = f2h(uf(rng) * 0.02f);
  for (auto& v : x)   v = f2h(uf(rng));
  std::vector<int32_t> ids(M * topk); std::vector<float> tw(M * topk);
  std::uniform_real_distribution<float> u01(0.f, 1.f);
  std::uniform_int_distribution<int> eid(0, E - 1);
  int n_local = 0;
  for (int i = 0; i < M * topk; i++) {
    ids[i] = (u01(rng) < local_frac) ? eid(rng) : -1;
    if (ids[i] >= 0) n_local++;
    tw[i] = u01(rng);
  }
  // force one token to have zero local experts (all -1) to exercise the "write zeros" path
  if (M >= 2) for (int s = 0; s < topk; s++) { if (ids[1 * topk + s] >= 0) n_local--; ids[1 * topk + s] = -1; }

  // ---- CPU reference (float) ----
  std::vector<float> ref((size_t)M * K, 0.f);
  for (int m = 0; m < M; m++) {
    for (int s = 0; s < topk; s++) {
      const int e = ids[m * topk + s]; if (e < 0) continue;
      std::vector<float> a(N);
      for (int n = 0; n < N; n++) {
        float ag = 0, au = 0;
        const uint64_t base = (uint64_t)e * 2 * N;
        for (size_t i = 0; i < K8; i++) {
          uint32_t qg = w13[(base + n) * K8 + i], qu = w13[(base + N + n) * K8 + i];
          float pg = 0, pu = 0;
          for (int j = 0; j < 8; j++) {
            float xv = h2f(x[(size_t)m * K + i * 8 + j]);
            pg += (float)((int)((qg >> (4 * j)) & 0xF) - 8) * xv;
            pu += (float)((int)((qu >> (4 * j)) & 0xF) - 8) * xv;
          }
          size_t g = (i * 8) / G;
          ag += pg * h2f(s13[(base + n) * KG + g]);
          au += pu * h2f(s13[(base + N + n) * KG + g]);
        }
        float silu = ag / (1.f + expf(-ag));
        a[n] = h2f(f2h(silu * au));  // kernel stores act as half
      }
      for (int h = 0; h < K; h++) {
        float sacc = 0;
        for (size_t i = 0; i < N8; i++) {
          uint32_t q = w2[((uint64_t)e * K + h) * N8 + i];
          float p = 0;
          for (int j = 0; j < 8; j++)
            p += (float)((int)((q >> (4 * j)) & 0xF) - 8) * a[i * 8 + j];
          sacc += p * h2f(s2[((uint64_t)e * K + h) * NG + (i * 8) / G]);
        }
        ref[(size_t)m * K + h] += sacc * tw[m * topk + s];
      }
    }
  }

  // ---- GPU ----
  uint32_t *dw13, *dw2; half *ds13, *ds2, *dx, *dact, *dout; int32_t* dids; float* dtw;
  CHECK(hipMalloc(&dw13, w13.size() * 4)); CHECK(hipMalloc(&dw2, w2.size() * 4));
  CHECK(hipMalloc(&ds13, s13.size() * 2)); CHECK(hipMalloc(&ds2, s2.size() * 2));
  CHECK(hipMalloc(&dx, x.size() * 2)); CHECK(hipMalloc(&dact, (size_t)M * topk * N * 2));
  CHECK(hipMalloc(&dout, (size_t)M * K * 2)); CHECK(hipMalloc(&dids, ids.size() * 4));
  CHECK(hipMalloc(&dtw, tw.size() * 4));
  CHECK(hipMemcpy(dw13, w13.data(), w13.size() * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dw2, w2.data(), w2.size() * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(ds13, s13.data(), s13.size() * 2, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(ds2, s2.data(), s2.size() * 2, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dids, ids.data(), ids.size() * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dtw, tw.data(), tw.size() * 4, hipMemcpyHostToDevice));
  CHECK(hipMemset(dout, 0x7f, (size_t)M * K * 2));  // poison: kernel must overwrite every row

  constexpr int WAVES = 8;
  dim3 block(WAVES * 32);
  dim3 grid1((N + WAVES - 1) / WAVES, topk, M);
  dim3 grid2((K + WAVES - 1) / WAVES, 1, M);
  auto launch = [&]() {
    moe_w13_silu_gemv_<WAVES><<<grid1, block, K * 2>>>(dx, dw13, ds13, dids, dact, K, N, topk, G);
    moe_w2_gemv_<WAVES><<<grid2, block, topk * N * 2>>>(dact, dw2, ds2, dids, dtw, dout, N, K, topk, G);
  };
  launch(); CHECK(hipDeviceSynchronize());
  std::vector<uint16_t> out((size_t)M * K);
  CHECK(hipMemcpy(out.data(), dout, out.size() * 2, hipMemcpyDeviceToHost));
  double max_abs = 0, max_ref = 0; int bad = 0;
  for (size_t i = 0; i < out.size(); i++) {
    float o = h2f(out[i]), r = ref[i];
    double d = fabs(o - r); if (d > max_abs) max_abs = d; if (fabs(r) > max_ref) max_ref = fabs(r);
    if (d > 1e-2 + 2e-2 * fabs(r)) bad++;
  }
  printf("M=%d local_frac=%.2f local_slots=%d/%d  max|ref|=%.4f max_abs_err=%.5f mismatches=%d  -> %s\n",
         M, local_frac, n_local, M * topk, max_ref, max_abs, bad, bad ? "FAIL" : "PASS");
  // token 1 must be exactly zero
  bool z = true; for (int h = 0; h < K && M >= 2; h++) if (h2f(out[(size_t)1 * K + h]) != 0.f) { z = false; break; }
  printf("all-nonlocal token row is zero: %s\n", z ? "yes" : "NO");

  // timing
  const int iters = 500;
  for (int i = 0; i < 20; i++) launch();
  CHECK(hipDeviceSynchronize());
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; i++) launch();
  CHECK(hipDeviceSynchronize());
  auto t1 = std::chrono::steady_clock::now();
  double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  double bytes = (double)n_local * (2.0 * N * K / 2 + 2.0 * N * KG * 2 + (double)K * N / 2 + (double)K * NG * 2);
  printf("pair: %.1f us/call  (%d local expert-slots, %.1f MB weights → %.0f GB/s effective)\n",
         us, n_local, bytes / 1e6, bytes / us / 1e3);
  return bad ? 1 : 0;
}
