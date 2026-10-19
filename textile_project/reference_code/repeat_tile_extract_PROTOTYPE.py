# PROTOTYPE: user ke saath tested aur approved logic. Hardcoded paths/thresholds hain.
# ROADMAP ke hisaab se module me refactor karo; logic mat badlo bina test ke.
import numpy as np, cv2
ds=cv2.imread('deshear.png').astype(np.float32); valid=np.load('valid.npy')
o=56
from seam_quilt_lib import make
cands=[]
for W in range(496,514,4):
  for H in range(308,326,4):
    for y0 in range(0,ds.shape[0]-H-o,12):
      for x0 in range(0,ds.shape[1]-W-o,12):
        if not valid[y0:y0+H+o,x0:x0+W+o].all(): continue
        a=np.mean((ds[y0:y0+o,x0:x0+W]-ds[y0+H:y0+H+o,x0:x0+W])**2)
        b=np.mean((ds[y0:y0+H,x0:x0+o]-ds[y0:y0+H,x0+W:x0+W+o])**2)
        cands.append((a+b,W,H,x0,y0))
cands.sort(); print(len(cands),cands[:5])
best=None
for c in cands[:25]:
    _,W,H,x0,y0=c; t,cost=make(ds,x0,y0,W,H,o)
    if best is None or cost<best[1]: best=(t,cost,c)
t,cost,c=best; print('chosen',c,'seam',round(cost,1))
cv2.imwrite('hd_tile.png',np.clip(t,0,255).astype(np.uint8))
