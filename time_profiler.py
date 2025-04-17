import time
import torch

ticmap = {}
ticamap = {}

def tic(st=""):
  global ticmap
  torch.cuda.synchronize()
  ticmap[st] = time.time()

def toc(st=""):
  global ticmap
  torch.cuda.synchronize()
  elapsed = time.time() - ticmap[st]
  print(f"{st}: {elapsed:.2f} s")
  return elapsed

def toca(st=""):
  global ticmap, ticamap
  torch.cuda.synchronize()
  t = time.time()
  if st in ticmap:
    if st in ticamap:
      # print("accum", st)
      ticamap[st] += t - ticmap[st]
    else:
      ticamap[st] = t - ticmap[st]

def tocaList():
    global ticamap
    total = sum(ticamap.values())
    sorted_items = sorted(ticamap.items(), key=lambda item: item[1], reverse=True)
    for k, v in sorted_items:
        pct = (v / total) * 100 if total > 0 else 0
        print(f"{k}: {v:.2f} s ({pct:.1f}%)")
