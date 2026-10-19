# PROTOTYPE: user ke saath tested aur approved logic. Hardcoded paths/thresholds hain.
# ROADMAP ke hisaab se module me refactor karo; logic mat badlo bina test ke.
import numpy as np, cv2
from scipy import ndimage
from collections import deque
S=3535
line=cv2.imread('/mnt/user-data/uploads/23551.png',0)
g=cv2.resize(line,(S,S),interpolation=cv2.INTER_LANCZOS4); g=cv2.GaussianBlur(g,(3,3),0)
lines=g<150
L,n=ndimage.label(~lines); L=L.astype(np.int32)
print('regions',n)
dirs=[(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
K=14
def shift(a,dy,dx):
    out=np.zeros_like(a)
    ys=slice(max(dy,0),S+min(dy,0)); yd=slice(max(-dy,0),S+min(-dy,0))
    xs=slice(max(dx,0),S+min(dx,0)); xd=slice(max(-dx,0),S+min(-dx,0))
    out[yd,xd]=a[ys,xs]; return out
Fs=[]; edges=set()
for dy,dx in dirs:
    F=np.zeros_like(L)
    for k in range(K,0,-1):
        sk=shift(L,dy*k,dx*k); F=np.where(sk!=0,sk,F)
    Fs.append(F)
    nxt=shift(L,dy,dx)
    m=(L!=0)&(nxt==0)&(F!=0)&(F!=L)
    a=L[m]; b=F[m]
    pr=np.unique(np.stack([np.minimum(a,b),np.maximum(a,b)],1),axis=0)
    edges.update(map(tuple,pr))
adj=[[] for _ in range(n+1)]
for a,b in edges: adj[a].append(b); adj[b].append(a)
border=set(np.unique(np.concatenate([L[0],L[-1],L[:,0],L[:,-1]])))-{0}
ref=cv2.imread('/mnt/user-data/uploads/23534.png',0)
refb=cv2.resize(ref,(S,S),interpolation=cv2.INTER_AREA)<110
area=np.bincount(L.ravel(),minlength=n+1)
blk=np.bincount(L.ravel(),weights=refb.ravel(),minlength=n+1)/np.maximum(area,1)
cx=ndimage.center_of_mass(np.ones_like(L),L,range(1,n+1)); cx=np.array([0]+[c[1] for c in cx])
side=(cx<0.27*S)|(cx>0.73*S)
big=[r for r in range(1,n+1) if (area[r]>6000 and blk[r]>0.8) or (side[r] and area[r]>9000 and blk[r]>0.55)]
print('extra black seeds',len(big))
border|=set(big)
depth=np.full(n+1,-1); q=deque()
for r in border: depth[r]=0; q.append(r)
while q:
    r=q.popleft()
    for t in adj[r]:
        if depth[t]<0: depth[t]=depth[r]+1; q.append(t)
print('unreached',int((depth[1:]<0).sum()),'depth hist',np.bincount(depth[1:][depth[1:]>=0]))
depth[depth<0]=1
cream_r=(depth%2==1); cream_r[0]=False
creamR=cream_r[L]                       # region pixels cream?
# line pixels: black only if every neighbour region found is cream
allc=np.ones((S,S),bool); anyf=np.zeros((S,S),bool)
for F in Fs:
    f=F!=0; anyf|=f
    allc&=np.where(f,cream_r[F],True)
lineCream=lines & ~(allc&anyf)
out=np.where(lines,lineCream,creamR)
np.save('cream.npy',out)
INK=np.array([16,16,15],np.uint8); CR=np.array([232,223,210],np.uint8)
img=np.where(out[...,None],CR,INK)
cv2.imwrite('depth.png',img[:,:,::-1])
cv2.imwrite('dp.png',cv2.resize(img[:,:,::-1],(900,900),interpolation=cv2.INTER_AREA))
