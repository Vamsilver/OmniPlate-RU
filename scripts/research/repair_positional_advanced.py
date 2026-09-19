#!/usr/bin/env python3
"""Repair script: eval по типам + ONNX export для advanced checkpoint."""
import sys, os, json, time, torch
import numpy as np
sys.path.insert(0, r"D:\AIProjects\VolgaIT")

from src.pipeline.positional_net import (
    PositionalPlateNet, export_positional_onnx,
    decode_positional, decode_positional_constrained, encode_plate_positional,
)
from scripts.train_ocr_v3 import PlateCropDataset
from torch.utils.data import DataLoader, Dataset

LAT_TO_CYR = {"A":"А","B":"В","E":"Е","K":"К","M":"М","H":"Н","O":"О","P":"Р","C":"С","T":"Т","Y":"У","X":"Х"}
def lat_to_cyr(t): return "".join(LAT_TO_CYR.get(c,c) for c in t.upper())

def lev(s1, s2):
    if len(s1) < len(s2): return lev(s2, s1)
    if not s2: return len(s1)
    p = list(range(len(s2)+1))
    for c1 in s1:
        c = [p[0]+1]
        for j,c2 in enumerate(s2):
            c.append(min(p[j+1]+1, c[j]+1, p[j]+(c1!=c2)))
        p = c
    return p[-1]

# 1. Загружаем checkpoint на CPU чисто
pt_path = r"D:\AIProjects\VolgaIT\models\positional_net_advanced.pt"
model = PositionalPlateNet(fused_channels=448, dropout_rate=0.3)
ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"[+] Checkpoint: best_epoch={ckpt['epoch']}, saved_seq_acc={ckpt['seq_acc']:.2f}%")

# 2. Val dataset
class PDS(Dataset):
    def __init__(self, base):
        self.base = base
        self.idx = []
        for i in range(len(base)):
            try:
                item = base[i]
                if item is None: continue
                _, t, pt = item
                tgt = encode_plate_positional(lat_to_cyr(t))
                if tgt is not None:
                    self.idx.append((i, pt))
            except: pass
        print(f"  Val filtered: {len(self.idx)} / {len(base)}")

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        ri, pt = self.idx[i]
        img, t, _ = self.base[ri]
        tgt = encode_plate_positional(lat_to_cyr(t))
        return img, tgt, lat_to_cyr(t), pt

base = PlateCropDataset(
    r"D:\AIProjects\VolgaIT\dataset\meta.csv",
    r"D:\AIProjects\VolgaIT\dataset",
    is_train=False, val_split=0.15, cache_in_ram=True,
)
val_ds  = PDS(base)

def collate(batch):
    imgs, tgts, gts, pts = zip(*batch)
    return torch.stack(imgs), torch.stack(tgts), list(gts), list(pts)

loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate)

# 3. Eval: constrained vs unconstrained, per type
stats = {
    "constrained":   {"ex":0, "ed":0, "tc":0, "tot":0},
    "unconstrained": {"ex":0, "ed":0, "tc":0, "tot":0},
}
type_stats_c = {}
with torch.no_grad():
    for imgs, tgts, gts, ptypes in loader:
        logits = model(imgs)
        pred_c = decode_positional_constrained(logits)
        pred_u = decode_positional(logits)
        for pc, pu, gt, pt in zip(pred_c, pred_u, gts, ptypes):
            for k, pr in [("constrained", pc), ("unconstrained", pu)]:
                stats[k]["tot"] += 1
                if pr == gt: stats[k]["ex"] += 1
                stats[k]["ed"] += lev(pr, gt)
                stats[k]["tc"] += len(gt)
            if pt not in type_stats_c:
                type_stats_c[pt] = {"ex":0, "tot":0}
            type_stats_c[pt]["tot"] += 1
            if pc == gt:
                type_stats_c[pt]["ex"] += 1

print("\n=== EVAL RESULTS ===")
for k, v in stats.items():
    acc = v["ex"] / max(1,v["tot"]) * 100
    cer = v["ed"] / max(1,v["tc"]) * 100
    print(f"  {k:>14}: SeqAcc {acc:.2f}%  CER {cer:.3f}%  ({v['ex']}/{v['tot']})")
print("\n  По типам (constrained):")
for pt, v in sorted(type_stats_c.items()):
    acc = v["ex"] / max(1,v["tot"]) * 100
    print(f"    {pt}: {v['ex']}/{v['tot']} = {acc:.1f}%")

# 4. ONNX export (модель уже на CPU)
onnx_path = r"D:\AIProjects\VolgaIT\models\positional_net_advanced.onnx"
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
export_positional_onnx(model, onnx_path, opset_version=17)
onnx_kb = round(os.path.getsize(onnx_path)/1024, 1)
print(f"\n[+] ONNX: {onnx_kb} KB -> {onnx_path}")

# 5. CPU latency
dummy = torch.randn(1, 3, 36, 160)
for _ in range(10): model(dummy)
lats = []
for _ in range(100):
    t0 = time.perf_counter(); model(dummy)
    lats.append((time.perf_counter()-t0)*1000)
lats.sort()
lat_cpu = {"mean_ms": round(float(np.mean(lats)),3),
           "p50_ms":  round(float(np.percentile(lats,50)),3),
           "p95_ms":  round(float(np.percentile(lats,95)),3)}
print(f"  CPU Latency: mean={lat_cpu['mean_ms']} ms  p50={lat_cpu['p50_ms']} ms")

print("\nDONE")
