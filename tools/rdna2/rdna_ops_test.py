import torch, torch.nn.functional as F
import vllm, vllm._rocm_C  # noqa
import vllm.model_executor.layers.rdna_ops  # noqa: registers the ops
from vllm.models.qwen4_exp.amd.ops.hc import hc_gate_mix, hc_silu
torch.manual_seed(0); dev="cuda"
def i8(w):
    amax=w.abs().amax(dim=1).float().clamp_min(1e-8); s=amax/127.0
    q=torch.round(w.float()/s[:,None]).clamp_(-127,127).to(torch.int8); return q.contiguous(), s.half().contiguous()
def rel(a,b): return ((a.float()-b.float()).abs().max()/b.float().abs().max().clamp_min(1e-6)).item()
ok=True
for M in (4, 64):
    x=torch.randn(M,2560,device=dev,dtype=torch.float16); W=torch.randn(512,2560,device=dev,dtype=torch.float16)*0.02; q,s=i8(W)
    y=torch.ops.vllm.rdna_dense_gemm(x,W,q,s,None); ref=F.linear(x,W); e=rel(y,ref); ok&=e<2e-2
    print(f"M={M} dense_gemm relerr {e:.1e}")
    xn=torch.randn(M,10240,device=dev,dtype=torch.float16); Wd=torch.randn(336,10240,device=dev,dtype=torch.float16)*0.02; Wu=torch.randn(10240,320,device=dev,dtype=torch.float16)*0.05
    qd,sd=i8(Wd); qu,su=i8(Wu)
    bi,dai=torch.ops.vllm.rdna_hc_mix(xn,Wd,qd,sd,Wu,qu,su,320,4)
    dai_r=F.linear(xn,Wd); lora=hc_silu(dai_r[:,:320].contiguous(),4); gate=F.linear(lora,Wu); bi_r=hc_gate_mix(xn,gate,4)
    e1=rel(bi,bi_r); e2=rel(dai[:,320:324],dai_r[:,320:324]); ok&=e1<3e-2 and e2<3e-2
    print(f"M={M} hc_mix relerr block_input {e1:.1e} injection {e2:.1e}")
    W1=torch.randn(320,2560,device=dev,dtype=torch.float16)*0.02; W2=torch.randn(2560,160,device=dev,dtype=torch.float16)*0.05; wg=torch.randn(2560,device=dev,dtype=torch.float16)*0.02
    q1,s1=i8(W1); q2,s2=i8(W2)
    out=torch.ops.vllm.rdna_shared_expert(x,W1,q1,s1,W2,q2,s2,wg)
    gu=F.linear(x,W1); act=F.silu(gu[:,:160])*gu[:,160:]; ref=torch.sigmoid(F.linear(x,wg.reshape(1,-1)))*F.linear(act,W2)
    e=rel(out,ref); ok&=e<3e-2; print(f"M={M} shared_expert relerr {e:.1e}")
print("PASS" if ok else "FAIL")
