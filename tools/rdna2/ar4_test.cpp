// Cross-process test of rdna_ar_oneshot for W ranks: correctness (vs exact fp32 sum),
// bit-identical outputs across ranks, amortized latency, graph replay, skew soak.
// Usage: ar4_test W n iters dev0 dev1 dev2 dev3   (devs may repeat: same-device IPC test)
#include "rdna_allreduce.cuh"
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <chrono>

#define CHK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "rank? HIP %s @%d\n", hipGetErrorString(e), __LINE__); _exit(3);} } while (0)

struct Shared {
  hipIpcMemHandle_t handle[RDNA_AR_MAX_WORLD];
  volatile int handle_ready[RDNA_AR_MAX_WORLD];
  int flags[16];                       // host-coherent all-reduce flags (one cache line region)
  volatile int result[RDNA_AR_MAX_WORLD];
  double us[RDNA_AR_MAX_WORLD];
  unsigned crc[RDNA_AR_MAX_WORLD];
};

static unsigned crc32(const void* d, size_t n) {
  unsigned c = 0xffffffffu; const unsigned char* p = (const unsigned char*)d;
  for (size_t i = 0; i < n; i++) { c ^= p[i]; for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u))); }
  return ~c;
}

int main(int argc, char** argv) {
  const int W = argc > 1 ? atoi(argv[1]) : 4;
  const int n = argc > 2 ? atoi(argv[2]) : 10240;      // 20 KB of fp16 = the model's [4 x 2560]
  const int IT = argc > 3 ? atoi(argv[3]) : 2000;
  int devs[RDNA_AR_MAX_WORLD] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int r = 0; r < W; r++) devs[r] = argc > 4 + r ? atoi(argv[4 + r]) : 0;
  const int THREADS = 256;
  const long long max_elems = 262144;                  // 512 KB fp16 per slot
  Shared* sh = (Shared*)mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  memset((void*)sh, 0, sizeof(Shared));
  pid_t pids[RDNA_AR_MAX_WORLD];
  for (int r = 0; r < W; r++) {
    pids[r] = fork();
    if (pids[r] != 0) continue;
    CHK(hipSetDevice(devs[r]));
    for (int j = 0; j < W; j++) if (devs[j] != devs[r]) (void)hipDeviceEnablePeerAccess(devs[j], 0);
    CHK(hipHostRegister((void*)sh->flags, 64, hipHostRegisterMapped | hipHostRegisterPortable));
    int* dflags = nullptr; CHK(hipHostGetDevicePointer((void**)&dflags, (void*)sh->flags, 0));
    __half *in, *out; void* stage;
    CHK(hipMalloc(&in, n * 2)); CHK(hipMalloc(&out, n * 2));
    const size_t stage_bytes = 2ull * W * max_elems * sizeof(__half);
    CHK(hipExtMallocWithFlags(&stage, stage_bytes, hipDeviceMallocUncached));
    CHK(hipMemset(stage, 0, stage_bytes));
    unsigned int* arrive; int* seqbuf; unsigned* timeout;
    CHK(hipMalloc(&arrive, 8)); CHK(hipMemset(arrive, 0, 8));
    CHK(hipMalloc(&seqbuf, 4)); CHK(hipMemset(seqbuf, 0, 4));
    CHK(hipMalloc(&timeout, 4)); CHK(hipMemset(timeout, 0, 4));
    std::vector<__half> h(n);
    for (int i = 0; i < n; i++) h[i] = __float2half((float)((r + 1) * 7 + (i % 97) * 0.125f));
    CHK(hipMemcpy(in, h.data(), n * 2, hipMemcpyHostToDevice));
    CHK(hipIpcGetMemHandle(&sh->handle[r], stage));
    __sync_synchronize(); sh->handle_ready[r] = 1;
    for (int j = 0; j < W; j++) while (!sh->handle_ready[j]) usleep(200);
    RdnaArPeers peers;
    for (int j = 0; j < W; j++) {
      if (j == r) { peers.stage[j] = stage; continue; }
      CHK(hipIpcOpenMemHandle(&peers.stage[j], sh->handle[j], hipIpcMemLazyEnablePeerAccess));
    }
    hipStream_t stream; CHK(hipStreamCreate(&stream));
    const int nblocks = argc > 4 + W ? atoi(argv[4 + W]) : ((n * 2 <= 16384) ? 1 : 4);
    auto fire = [&] {
      hipLaunchKernelGGL(rdna_ar_oneshot<__half>, dim3(nblocks), dim3(THREADS), 0, stream,
                         in, out, peers, dflags, arrive, seqbuf, timeout, r, W, n, max_elems, nblocks);
    };
    for (int i = 0; i < 20; i++) fire();
    CHK(hipStreamSynchronize(stream));
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < IT; i++) fire();
    CHK(hipStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    sh->us[r] = std::chrono::duration<double, std::micro>(t1 - t0).count() / IT;
    // graph capture + replay (the serving shape)
    hipGraph_t g; hipGraphExec_t ge;
    CHK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    fire(); fire(); fire();
    CHK(hipStreamEndCapture(stream, &g));
    CHK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    for (int i = 0; i < 300; i++) CHK(hipGraphLaunch(ge, stream));
    CHK(hipStreamSynchronize(stream));
    // skew soak: rank r sleeps a different amount between launches
    for (int i = 0; i < 200; i++) { fire(); if (i % (r + 2) == 0) usleep(50 * (r + 1)); }
    CHK(hipStreamSynchronize(stream));
    std::vector<__half> ho(n);
    CHK(hipMemcpy(ho.data(), out, n * 2, hipMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; i++) {
      float want = 0.f;
      for (int j = 0; j < W; j++) want += __half2float(__float2half((float)((j + 1) * 7 + (i % 97) * 0.125f)));
      if (__half2float(ho[i]) != __half2float(__float2half(want))) bad++;
    }
    unsigned to = 0; CHK(hipMemcpy(&to, timeout, 4, hipMemcpyDeviceToHost));
    sh->result[r] = to ? -1 : bad;
    sh->crc[r] = crc32(ho.data(), n * 2);
    for (int j = 0; j < W; j++) if (j != r) CHK(hipIpcCloseMemHandle(peers.stage[j]));
    _exit(0);
  }
  int died = 0, status = 0;
  for (int r = 0; r < W; r++) { waitpid(pids[r], &status, 0); if (!WIFEXITED(status) || WEXITSTATUS(status)) died++; }
  bool ok = !died, same = true;
  for (int r = 0; r < W; r++) { if (sh->result[r] != 0) ok = false; if (sh->crc[r] != sh->crc[0]) same = false; }
  printf("W=%d n=%d (%d B) devs", W, n, n * 2); for (int r = 0; r < W; r++) printf(" %d", devs[r]);
  printf(":  us/op"); for (int r = 0; r < W; r++) printf(" %.2f", sh->us[r]);
  printf("   result"); for (int r = 0; r < W; r++) printf(" %d", sh->result[r]);
  printf("   %s, outputs %s across ranks%s\n", ok ? "CORRECT" : (died ? "CHILD DIED" : "WRONG/TIMEOUT"),
         same ? "bit-identical" : "DIFFER", (ok && same) ? "  -> PASS" : "  -> FAIL");
  return (ok && same) ? 0 : 1;
}
