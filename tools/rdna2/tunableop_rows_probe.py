"""Run every GemmTunableOp_Half_{TN,NN} row of a TunableOp CSV as a torch matmul (lookup mode) and report rows whose
recorded solution the runtime rejects (TORCH_CHECK iter != ops_.end()). usage: tunable_rows_probe.py <csv>"""
import re, sys, torch
rows = [l.strip().split(",") for l in open(sys.argv[1]) if l.startswith("GemmTunableOp_Half_")]
bad, ok, skipped = [], 0, 0
for op, sig, sol, t in rows:
    m = re.match(r"(t|n)(t|n)_(\d+)_(\d+)_(\d+)_ld_(\d+)_(\d+)_(\d+)$", sig)
    if not m: skipped += 1; continue
    ta, tb, M, N, K, lda, ldb, ldc = m.group(1), m.group(2), *map(int, m.groups()[2:])
    try:
        if ta == "t" and tb == "n":      # torch: X[N,K] @ W[M,K].T  -> C[N,M]
            X = torch.randn(N, K, dtype=torch.half, device="cuda"); W = torch.randn(M, K, dtype=torch.half, device="cuda"); C = X @ W.t()
        elif ta == "n" and tb == "n":    # torch: X[N,K] @ W[K,M]    -> C[N,M]
            X = torch.randn(N, K, dtype=torch.half, device="cuda"); W = torch.randn(K, M, dtype=torch.half, device="cuda"); C = X @ W
        else:
            skipped += 1; continue
        torch.cuda.synchronize(); ok += 1
    except Exception as e:
        bad.append((sig, sol, str(e).splitlines()[0][:80]))
print(f"{sys.argv[1].split('/')[-1]}: ok {ok}, failed {len(bad)}, skipped {skipped}")
for sig, sol, err in bad[:6]: print(f"   FAIL {sig} -> {sol}: {err}")
