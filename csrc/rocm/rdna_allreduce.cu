// T44 — torch ops around rdna_ar_oneshot: a push-based one-shot all-reduce for small
// tensor-parallel messages on gfx1030 (2..8 ranks, one process per rank).
//
//   rdna_ar_init(rank, world, device_ids, max_bytes, shm_name) -> uint8[64] IPC handle
//   rdna_ar_connect(handles uint8[world,64])                    (opens every peer's staging)
//   rdna_ar_can(t) -> bool, rdna_ar_all_reduce(t) -> Tensor, rdna_ar_timed_out() -> bool
//
// Why not vLLM's own custom all-reduce: its barrier spins on coarse-grained device memory,
// which a peer's write never makes visible on this platform (T18); and it is gated to
// XGMI-connected gfx9. See rdna_allreduce.cuh for the protocol.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <string>

#include "rdna_allreduce.cuh"

#define RDNA_AR_CHK(x)                                                        \
  do {                                                                        \
    hipError_t e_ = (x);                                                      \
    TORCH_CHECK(e_ == hipSuccess, "rdna_ar HIP: ", hipGetErrorString(e_));    \
  } while (0)

namespace {
struct RdnaArState {
  bool ready = false;
  int rank = -1, world = 0;
  int64_t max_bytes = 0;
  void* stage = nullptr;         // ours: 2 * world * max_bytes, uncached
  RdnaArPeers peers{};           // [j] = peer j's staging (IPC), [rank] = ours
  bool opened[RDNA_AR_MAX_WORLD] = {};
  void* shm = nullptr;           // 4096 B host-coherent page holding the flags
  int* dflags = nullptr;         // device pointer to shm
  unsigned int* arrive = nullptr;
  int* seqbuf = nullptr;
  unsigned* timeout = nullptr;
  int64_t fast_calls = 0;
};
// One instance per process group (vLLM builds several GroupCoordinators over the same
// ranks: world, TP, EP ...). Addressed by the handle rdna_ar_init returns.
constexpr int kMaxInstances = 8;
RdnaArState g_inst[kMaxInstances];
int g_ninst = 0;
RdnaArState& inst(int64_t h) {
  TORCH_CHECK(h >= 0 && h < g_ninst, "rdna_ar: bad handle ", h);
  return g_inst[h];
}
}  // namespace

// Returns [handle (int64), 64 bytes of IPC handle] packed in a uint8 tensor of 72 bytes.
at::Tensor rdna_ar_init(int64_t rank, int64_t world, const at::Tensor& device_ids,
                        int64_t max_bytes, const std::string& shm_name) {
  TORCH_CHECK(g_ninst < kMaxInstances, "rdna_ar: too many instances");
  RdnaArState& g = g_inst[g_ninst];
  const int64_t handle_id = g_ninst;
  TORCH_CHECK(world >= 2 && world <= RDNA_AR_MAX_WORLD, "rdna_ar: world must be 2..8");
  TORCH_CHECK(rank >= 0 && rank < world, "rdna_ar: bad rank");
  TORCH_CHECK(device_ids.numel() == world && device_ids.scalar_type() == at::kLong,
              "rdna_ar: device_ids must be int64[world]");
  g.rank = (int)rank;
  g.world = (int)world;
  g.max_bytes = max_bytes;
  const int64_t* dev = device_ids.data_ptr<int64_t>();
  int mydev = -1;
  RDNA_AR_CHK(hipGetDevice(&mydev));
  for (int j = 0; j < world; j++) {
    if (j == rank || dev[j] == mydev) continue;
    hipError_t pe = hipDeviceEnablePeerAccess((int)dev[j], 0);
    TORCH_CHECK(pe == hipSuccess || pe == hipErrorPeerAccessAlreadyEnabled,
                "rdna_ar: peer access to device ", dev[j], ": ", hipGetErrorString(pe));
    // "already enabled" (second instance in this process) is sticky in the HIP
    // last-error slot and would surface at torch's next launch check -- clear it.
    (void)hipGetLastError();
  }
  const size_t stage_bytes = 2ull * world * (size_t)max_bytes;
  RDNA_AR_CHK(hipExtMallocWithFlags(&g.stage, stage_bytes, hipDeviceMallocUncached));
  RDNA_AR_CHK(hipMemset(g.stage, 0, stage_bytes));
  RDNA_AR_CHK(hipMalloc((void**)&g.arrive, 8));
  RDNA_AR_CHK(hipMemset(g.arrive, 0, 8));
  RDNA_AR_CHK(hipMalloc((void**)&g.seqbuf, 4));
  RDNA_AR_CHK(hipMemset(g.seqbuf, 0, 4));
  RDNA_AR_CHK(hipMalloc((void**)&g.timeout, 4));
  RDNA_AR_CHK(hipMemset(g.timeout, 0, 4));
  // host-coherent flag page shared by every rank of the group (rank 0 owns creation;
  // the Python side orders init rank by rank with barriers)
  if (rank == 0) shm_unlink(shm_name.c_str());
  int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0600);
  TORCH_CHECK(fd >= 0, "rdna_ar: shm_open failed: ", strerror(errno));
  TORCH_CHECK(ftruncate(fd, 4096) == 0, "rdna_ar: ftruncate failed");
  g.shm = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  TORCH_CHECK(g.shm != MAP_FAILED, "rdna_ar: mmap failed");
  if (rank == 0) std::memset(g.shm, 0, 4096);
  RDNA_AR_CHK(hipHostRegister(g.shm, 4096, hipHostRegisterMapped | hipHostRegisterPortable));
  RDNA_AR_CHK(hipHostGetDevicePointer((void**)&g.dflags, g.shm, 0));
  for (int j = 0; j < RDNA_AR_MAX_WORLD; j++) g.peers.stage[j] = nullptr;
  g.peers.stage[rank] = g.stage;
  hipIpcMemHandle_t h;
  RDNA_AR_CHK(hipIpcGetMemHandle(&h, g.stage));
  g_ninst++;
  auto out = at::empty({(int64_t)(8 + sizeof(h))}, at::TensorOptions().dtype(at::kByte));
  std::memcpy(out.mutable_data_ptr(), &handle_id, 8);
  std::memcpy(static_cast<uint8_t*>(out.mutable_data_ptr()) + 8, &h, sizeof(h));
  return out;
}

void rdna_ar_connect(int64_t handle, const at::Tensor& handles) {
  RdnaArState& g = inst(handle);
  TORCH_CHECK(g.stage != nullptr && !g.ready, "rdna_ar: init first / already connected");
  TORCH_CHECK(handles.dim() == 2 && handles.size(0) == g.world &&
                  handles.size(1) == (int64_t)sizeof(hipIpcMemHandle_t) &&
                  handles.scalar_type() == at::kByte && handles.is_contiguous(),
              "rdna_ar: handles must be contiguous uint8[world, 64]");
  const uint8_t* p = handles.data_ptr<uint8_t>();
  for (int j = 0; j < g.world; j++) {
    if (j == g.rank) continue;
    hipIpcMemHandle_t h;
    std::memcpy(&h, p + (size_t)j * sizeof(h), sizeof(h));
    RDNA_AR_CHK(hipIpcOpenMemHandle(&g.peers.stage[j], h, hipIpcMemLazyEnablePeerAccess));
    g.opened[j] = true;
  }
  g.ready = true;
}

bool rdna_ar_can(int64_t handle, const at::Tensor& t) {
  const RdnaArState& g = inst(handle);
  return g.ready && t.is_cuda() && t.is_contiguous() &&
         (t.scalar_type() == at::kHalf || t.scalar_type() == at::kFloat) &&
         t.numel() * t.element_size() <= g.max_bytes && t.numel() > 0;
}

at::Tensor rdna_ar_all_reduce(int64_t handle, const at::Tensor& in) {
  TORCH_CHECK(rdna_ar_can(handle, in), "rdna_ar: tensor not eligible");
  RdnaArState& g = inst(handle);
  const at::cuda::OptionalCUDAGuard guard(in.device());
  auto out = at::empty_like(in);
  const int n = (int)in.numel();
  const int64_t bytes = (int64_t)n * in.element_size();
  // Measured on 4x V620 (T44): 20 KB 1/4/16 blocks = 76/36/33 us; 5 KB 1/4/8 = 27/15/18 us;
  // 80 KB 4/16/32 = 115/91/85 us. More blocks = more concurrent PCIe pushes.
  const int nblocks = bytes <= 8192 ? 4 : (bytes <= 32768 ? 16 : 32);
  const int threads = 256;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (in.scalar_type() == at::kHalf) {
    const long long max_elems = g.max_bytes / 2;
    rdna_ar_oneshot<__half><<<nblocks, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(in.const_data_ptr()),
        reinterpret_cast<__half*>(out.mutable_data_ptr()), g.peers, g.dflags, g.arrive,
        g.seqbuf, g.timeout, g.rank, g.world, n, max_elems, nblocks);
  } else {
    const long long max_elems = g.max_bytes / 4;
    rdna_ar_oneshot<float><<<nblocks, threads, 0, stream>>>(
        reinterpret_cast<const float*>(in.const_data_ptr()),
        reinterpret_cast<float*>(out.mutable_data_ptr()), g.peers, g.dflags, g.arrive,
        g.seqbuf, g.timeout, g.rank, g.world, n, max_elems, nblocks);
  }
  g.fast_calls++;
  return out;
}

bool rdna_ar_timed_out(int64_t handle) {
  const RdnaArState& g = inst(handle);
  if (!g.ready) return false;
  unsigned t = 0;
  RDNA_AR_CHK(hipMemcpy(&t, g.timeout, 4, hipMemcpyDeviceToHost));
  return t != 0;
}

int64_t rdna_ar_fast_calls(int64_t handle) { return inst(handle).fast_calls; }
