// T44 — push-based one-shot all-reduce for small messages on gfx1030, W ranks (2..8).
//
// Descendant of the TP=2 WS2 kernel (builds/shared/ws2-allreduce/src/allreduce.hip.h), whose
// findings it keeps:
//   * push, never pull        — peer STORE 14.3 GB/s vs peer LOAD 5.7 GB/s across PCIe;
//   * staging is UNCACHED     — hipDeviceMallocUncached; a peer's write to coarse-grained
//                               memory lands in DRAM while the owner keeps reading stale L2;
//   * flags live in each rank's OWN uncached device memory (2026-08-30; they were a
//     host-coherent page before): a peer announces by one posted P2P store into our flag
//     slot and we poll locally — waiting no longer generates PCIe read traffic. With the
//     host page, four GPUs hammered system memory with 32-bit reads across both root
//     complexes for the whole barrier; on this machine that coincided with the SAS HBA's
//     tape drive resetting and, twice, with cards dropping off the PCIe bus. Coarse-grained
//     device memory would not work (a peer's write is invisible to the owner's L2, T18);
//     uncached memory is exactly what the staging buffers already use for the same reason.
//   * the sequence number is derived on-device from a local counter, never passed as a kernel
//     argument (frozen at CUDA-graph capture, so every replay would fall through the wait);
//   * bounded spins that set a sticky abort flag instead of hanging the GPU -- and, since
//     T44b, record which phase/peer/sequence aborted in a host-visible word (see below).
// New here: W-way staging (one slot per source rank, x2 parities), pushes fanned out to all
// peers, a W-flag wait, and a FIXED-ORDER fp32 reduction (rank 0 .. W-1) so every rank produces
// bit-identical output — with TP, ranks that disagree by an ulp diverge from each other.
//
// Layout of a rank's staging buffer (uncached device memory), in elements of T:
//   stage[(parity * W + src) * max_elems + i]
// Peer j receives our contribution at its slot src = our rank.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define RDNA_AR_MAX_WORLD 8
#define RDNA_AR_FLAG_PAGE 4096  // bytes appended to each staging buffer for the flag slots
#define RDNA_AR_SPIN_CAP 2000000ull  // default polls before abort (~1 us each -> ~2 s); VLLM_RDNA_AR_SPIN_CAP

// T44b (2026-09-07) -- abort record. Before this, a collective that hit the spin cap set a bare
// sticky flag that only the boot self-test ever read: mid-serving, every later collective also
// spun to its cap and returned WITHOUT writing its output, so a wedged P2P path looked like
// "one or three GPUs pinned at 99 %, generation stopped" for the 300 s engine timeout (two
// boards reported it). Now the first block to hit the cap claims the device word `timeout`
// (atomicCAS, device scope) and writes one 64-bit code into `report`, a host-mapped mirror the
// Python side polls once per step with a plain load -- no device sync, no PCIe atomics:
//   bits 0-7   1 = aborted          bits 8-11  phase: 1 = own blocks' grid barrier,
//   bits 12-15 peer rank (phase 2)                    2 = a peer's flag never arrived
//   bits 16-31 spins / 1024 (~ms)   bits 32-63 sequence number of the collective
__device__ __forceinline__ void rdna_ar_abort(unsigned* timeout, unsigned long long* report,
                                              unsigned phase, unsigned peer, int seq,
                                              unsigned long long spins) {
  if (atomicCAS(timeout, 0u, 1u) == 0u) {
    unsigned long long code = 1ull | ((unsigned long long)(phase & 0xFu) << 8) |
                              ((unsigned long long)(peer & 0xFu) << 12) |
                              (((spins >> 10) & 0xFFFFull) << 16) |
                              ((unsigned long long)(unsigned)seq << 32);
    __hip_atomic_store(report, code, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ __forceinline__ float rdna_ar_to_f(float x) { return x; }
__device__ __forceinline__ float rdna_ar_to_f(__half x) { return __half2float(x); }
__device__ __forceinline__ void rdna_ar_from_f(float& d, float v) { d = v; }
__device__ __forceinline__ void rdna_ar_from_f(__half& d, float v) { d = __float2half(v); }

struct RdnaArPeers {
  void* stage[RDNA_AR_MAX_WORLD];  // peer j's staging buffer (IPC-mapped); [rank] = ours
  int* flags[RDNA_AR_MAX_WORLD];   // peer j's flag slots [W] (uncached, after its staging)
};

// Backoff between polls: s_sleep(n) idles the wave ~64*n clocks (~0.2 us at n=8) so a
// waiting rank does not saturate its memory path while spinning.
#define RDNA_AR_POLL_PAUSE() __builtin_amdgcn_s_sleep(8)

template <typename T>
__global__ void rdna_ar_oneshot(const T* __restrict__ in, T* __restrict__ out,
                                RdnaArPeers peers,               // stage + flags per rank
                                unsigned int* arrive,             // device, [2] per parity
                                int* seqbuf,                      // device, local seq mirror
                                unsigned* timeout,                // device, sticky abort claim
                                unsigned long long* report,       // host-mapped abort record (T44b)
                                int rank, int world, int n, long long max_elems,
                                int nblocks, int pace, unsigned long long spin_cap) {
  __shared__ int s_seq;
  __shared__ int s_abort;
  const int t = threadIdx.x, nt = blockDim.x, b = blockIdx.x;
  if (t == 0) {
    s_seq = __hip_atomic_load(seqbuf, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) + 1;
    s_abort = 0;
  }
  __syncthreads();
  const int seq = s_seq;
  const int p = seq & 1;
  const int gid = b * nt + t, gstride = nblocks * nt;

  // 1. push our slice into every peer's staging slot for us (posted PCIe writes).
  //    The peer order is staggered by rank, so at any instant each destination is being
  //    written by ONE source instead of all W-1 at once; `pace` idles the wave ~64 clocks per
  //    unit between strided stores to bound the burst rate (0 = off). Both are for the rest of
  //    the machine: these pushes land in the receiving GPU's root complex, where other devices'
  //    DMA completions queue behind them (2026-09-01: the SAS HBA on this box lives there).
  for (int k = 1; k < world; k++) {
    const int j = (rank + k) % world;
    T* dst = reinterpret_cast<T*>(peers.stage[j]) + ((long long)p * world + rank) * max_elems;
    for (int i = gid; i < n; i += gstride) {
      dst[i] = in[i];
      for (int q = 0; q < pace; q++) __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();

  // 2. grid barrier (all our blocks have pushed), payload-before-flag, announce, wait
  if (t == 0) {
    __threadfence_system();
    atomicAdd(&arrive[p], 1u);
    if (b == 0) {
      unsigned long long s = 0;
      while (__hip_atomic_load(&arrive[p], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) <
             (unsigned)nblocks) {
        RDNA_AR_POLL_PAUSE();
        if (++s > spin_cap) { rdna_ar_abort(timeout, report, 1u, (unsigned)rank, seq, s); s_abort = 1; break; }
      }
      if (!s_abort) {
        arrive[1 - p] = 0u;
        __hip_atomic_store(seqbuf, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        // announce: one posted P2P store into every peer's slot for us (and our own)
        for (int j = 0; j < world; j++)
          __hip_atomic_store(&peers.flags[j][rank], seq, __ATOMIC_RELEASE,
                             __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }
    if (!s_abort) {
      // wait: poll OUR flag slots (local uncached memory, no PCIe traffic)
      int* myflags = peers.flags[rank];
      for (int j = 0; j < world && !s_abort; j++) {
        if (j == rank) continue;
        unsigned long long s = 0;
        while (__hip_atomic_load(&myflags[j], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq) {
          RDNA_AR_POLL_PAUSE();
          if (++s > spin_cap) { rdna_ar_abort(timeout, report, 2u, (unsigned)j, seq, s); s_abort = 1; break; }
        }
      }
    }
  }
  __syncthreads();
  if (s_abort) return;

  // 3. fixed-order reduction: rank 0 .. W-1, fp32, identical on every rank
  const T* mine = reinterpret_cast<const T*>(peers.stage[rank]) + ((long long)p * world) * max_elems;
  for (int i = gid; i < n; i += gstride) {
    float v = 0.f;
    for (int j = 0; j < world; j++)
      v += (j == rank) ? rdna_ar_to_f(in[i]) : rdna_ar_to_f(mine[(long long)j * max_elems + i]);
    rdna_ar_from_f(out[i], v);
  }
}


// ---------------------------------------------------------------------------------------------
// Host-staged variant (VLLM_RDNA_AR_MODE=host, 2026-09-25). No GPU peer-to-peer traffic at all:
// every rank writes its contribution into ITS slot of one shared, pinned host buffer and reads
// the other ranks' slots from there. Motivation: on a 2-die X399 board (and on other users'
// boards) the p2p kernel above -- thousands of small posted writes per second straight into the
// peers' PCIe BAR windows -- is the one workload that knocks V620s off the bus; with RCCL instead
// (VLLM_RDNA_AR=0) the same load runs clean, at -26 % decode.
//
// Correctness does not depend on write ordering (posted writes to different destinations, or
// relaxed-ordered ones, can overtake each other -- the 2026-08-30 stale-element bug). Each 8-byte
// word carries 4 bytes of data and the collective's sequence number (LL-style, like RCCL's LL
// protocol): a reader accepts a word only when its tag matches, so a torn or stale word is never
// reduced. Waiting is cheap on the fabric: one thread per block polls ONE word per peer with
// backoff before the bulk reads; the bulk reads re-poll individually only if a word is still in
// flight. Two parities keep a fast rank's next collective off the slot a slow rank still reads
// (a rank can start collective s+2 only after every peer finished s, by stream order).
//
// Host layout, in 8-byte words: host[(parity * W + src) * slot_words + w],
//   word w = { low 32 bits: data (2 x fp16 or 1 x fp32), high 32 bits: sequence number }.
__device__ __forceinline__ unsigned long long rdna_ar_ll_load(const unsigned long long* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void rdna_ar_ll_store(unsigned long long* p, unsigned lo, int seq) {
  const unsigned long long v = ((unsigned long long)(unsigned)seq << 32) | lo;
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
template <typename T> struct RdnaLL;
template <> struct RdnaLL<__half> {   // two halves per word
  static constexpr int kPer = 2;
  __device__ static unsigned pack(const __half* in, int i, int n) {
    unsigned short a = __half_as_ushort(in[i]);
    unsigned short b = (i + 1 < n) ? __half_as_ushort(in[i + 1]) : (unsigned short)0;
    return (unsigned)a | ((unsigned)b << 16);
  }
  __device__ static void acc(float* v, unsigned lo) {
    v[0] += __half2float(__ushort_as_half((unsigned short)(lo & 0xFFFFu)));
    v[1] += __half2float(__ushort_as_half((unsigned short)(lo >> 16)));
  }
  __device__ static void own(float* v, const __half* in, int i, int n) {
    v[0] += __half2float(in[i]);
    if (i + 1 < n) v[1] += __half2float(in[i + 1]);
  }
  __device__ static void put(__half* out, int i, int n, const float* v) {
    out[i] = __float2half(v[0]);
    if (i + 1 < n) out[i + 1] = __float2half(v[1]);
  }
};
template <> struct RdnaLL<float> {    // one float per word
  static constexpr int kPer = 1;
  __device__ static unsigned pack(const float* in, int i, int) { return __float_as_uint(in[i]); }
  __device__ static void acc(float* v, unsigned lo) { v[0] += __uint_as_float(lo); }
  __device__ static void own(float* v, const float* in, int i, int) { v[0] += in[i]; }
  __device__ static void put(float* out, int i, int, const float* v) { out[i] = v[0]; }
};

template <typename T>
__global__ void rdna_ar_host_ll(const T* __restrict__ in, T* __restrict__ out,
                                unsigned long long* host,         // shared pinned host buffer
                                unsigned int* arrive, int* seqbuf,
                                unsigned* timeout, unsigned long long* report,
                                int rank, int world, int n, long long slot_words,
                                int nblocks, unsigned long long spin_cap) {
  __shared__ int s_seq;
  __shared__ int s_abort;
  const int t = threadIdx.x, nt = blockDim.x, b = blockIdx.x;
  if (t == 0) {
    s_seq = __hip_atomic_load(seqbuf, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) + 1;
    s_abort = 0;
  }
  __syncthreads();
  const int seq = s_seq;
  const int p = seq & 1;
  constexpr int kPer = RdnaLL<T>::kPer;
  const int nwords = (n + kPer - 1) / kPer;
  const int gid = b * nt + t, gstride = nblocks * nt;

  // 1. write our slot: posted writes to host memory, one tagged word per store
  unsigned long long* mine = host + ((long long)p * world + rank) * slot_words;
  for (int w = gid; w < nwords; w += gstride)
    rdna_ar_ll_store(mine + w, RdnaLL<T>::pack(in, w * kPer, n), seq);

  // 2. local grid barrier, only to advance the sequence counter after every block has read it
  __syncthreads();
  if (t == 0) {
    atomicAdd(&arrive[p], 1u);
    if (b == 0) {
      unsigned long long s = 0;
      while (__hip_atomic_load(&arrive[p], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) <
             (unsigned)nblocks) {
        RDNA_AR_POLL_PAUSE();
        if (++s > spin_cap) { rdna_ar_abort(timeout, report, 1u, (unsigned)rank, seq, s); s_abort = 1; break; }
      }
      if (!s_abort) {
        arrive[1 - p] = 0u;
        __hip_atomic_store(seqbuf, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    // 3. cheap wait: poll ONE word per peer (this block's first word) until its tag is ours
    const int w0 = b * nt < nwords ? b * nt : 0;
    for (int j = 0; j < world && !s_abort; j++) {
      if (j == rank) continue;
      const unsigned long long* peer = host + ((long long)p * world + j) * slot_words;
      unsigned long long s = 0;
      while ((int)(rdna_ar_ll_load(peer + w0) >> 32) != seq) {
        RDNA_AR_POLL_PAUSE();
        if (++s > spin_cap) { rdna_ar_abort(timeout, report, 2u, (unsigned)j, seq, s); s_abort = 1; break; }
      }
    }
  }
  __syncthreads();
  if (s_abort) return;

  // 4. read every peer's word, re-polling only a word that is still in flight; reduce in the
  //    fixed rank order 0 .. W-1 in fp32 so every rank produces bit-identical output
  for (int w = gid; w < nwords; w += gstride) {
    unsigned long long got[RDNA_AR_MAX_WORLD];
    for (int j = 0; j < world; j++)
      if (j != rank) got[j] = rdna_ar_ll_load(host + ((long long)p * world + j) * slot_words + w);
    float v[kPer];
    for (int k = 0; k < kPer; k++) v[k] = 0.f;
    for (int j = 0; j < world; j++) {
      if (j == rank) { RdnaLL<T>::own(v, in, w * kPer, n); continue; }
      const unsigned long long* src = host + ((long long)p * world + j) * slot_words + w;
      unsigned long long s = 0;
      while ((int)(got[j] >> 32) != seq) {
        RDNA_AR_POLL_PAUSE();
        if (++s > spin_cap) { rdna_ar_abort(timeout, report, 2u, (unsigned)j, seq, s); return; }
        got[j] = rdna_ar_ll_load(src);
      }
      RdnaLL<T>::acc(v, (unsigned)(got[j] & 0xFFFFFFFFull));
    }
    RdnaLL<T>::put(out, w * kPer, n, v);
  }
}
