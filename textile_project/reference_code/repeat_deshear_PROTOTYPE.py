# PROTOTYPE: user ke saath tested aur approved logic. Hardcoded paths/thresholds hain.
# ROADMAP ke hisaab se module me refactor karo; logic mat badlo bina test ke.
import numpy as np, cv2
im=cv2.imread('/mnt/user-data/uploads/23527.png'); H0,W0=im.shape[:2]
s=29.0/315.8
# deshear: x' = x - s*y + s*H0  (seedha karna)
M=np.float32([[1,-s,s*H0],[0,1,0]])
Wd=int(W0+s*H0)+1
ds=cv2.warpAffine(im,M,(Wd,H0),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
valid=cv2.warpAffine(np.ones((H0,W0),np.uint8),M,(Wd,H0),flags=cv2.INTER_NEAREST)>0
# refine periods
g=cv2.cvtColor(ds,cv2.COLOR_BGR2GRAY)
def best_shift(dy0,dx0,rng=8):
    t=g[400:900,int(s*H0)+60:int(s*H0)+60+420] if True else None
    y,x=400,int(s*H0)+60
    r=cv2.matchTemplate(g,g[y:y+300,x:x+300],cv2.TM_CCOEFF_NORMED)
    sub=r[y+dy0-rng:y+dy0+rng+1, x+dx0-rng:x+dx0+rng+1]
    yy,xx=np.unravel_index(sub.argmax(),sub.shape); return dy0-rng+yy,dx0-rng+xx,sub.max()
print('vert',best_shift(316,0),'horiz',best_shift(0,505),'halfdrop',best_shift(158,253,12))
cv2.imwrite('deshear.png',ds); np.save('valid.npy',valid)
