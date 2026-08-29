import torch, time
import vllm
from vllm import envs
from vllm.platforms.rocm import on_gfx1x, on_gfx9, on_gfx10x
print("on_gfx1x", on_gfx1x(), "on_gfx10x", on_gfx10x(), "SKINNY env", envs.VLLM_ROCM_USE_SKINNY_GEMM)
import vllm.model_executor.layers.utils as _u
route=torch.ops.vllm.rocm_unquantized_gemm
dev='cuda'
def t(fn, iters=200):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/iters*1e6
shapes = {"gdn.in_proj_qkv/rank":(2560,2560), "gdn.in_proj_z/rank":(1536,2560), "gdn.out_proj/rank":(2560,1536),
          "qsa.q_proj/rank":(3072,2560), "qsa.o_proj/rank":(2560,1536), "router.gate":(512,2560),
          "shared.gate_up":(1280,2560), "shared.down":(2560,640), "hc.down":(320,10240), "hc.up":(10240,320),
          "hc.inject":(4,10240), "lm_head/rank":(62080,2560)}
from torch.profiler import profile, ProfilerActivity
for M in (1,4,5,8):
    print(f"--- M={M} tokens ---")
    tv=tt=0
    for name,(N,K) in shapes.items():
        x=torch.randn(M,K,dtype=torch.float16,device=dev); w=torch.randn(N,K,dtype=torch.float16,device=dev)*0.02
        ref=torch.nn.functional.linear(x,w); out=route(x,w,None)
        err=((out.float()-ref.float()).abs().max()/ref.float().abs().max()).item()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            route(x,w,None); torch.cuda.synchronize()
        kn=[e.key for e in p.key_averages() if e.device_time_total>0 and not e.key.startswith("hip") and "Memcpy" not in e.key and "rocm_unquantized" not in e.key]
        us_v=t(lambda: route(x,w,None)); us_t=t(lambda: torch.nn.functional.linear(x,w)); tv+=us_v; tt+=us_t
        gb=N*K*2/1e9
        print(f"  {name:20s} [{N:6d}x{K:5d}] {gb*1e3:6.1f}MB  route {us_v:6.1f}us ({gb/us_v*1e6:4.0f}GB/s)  rocBLAS {us_t:6.1f}us ({gb/us_t*1e6:4.0f}GB/s)  x{us_t/us_v:4.1f}  relerr {err:.1e}  {','.join(k[:22] for k in kn)[:40]}")
    print(f"  SUM route {tv:.0f}us vs rocBLAS {tt:.0f}us")
