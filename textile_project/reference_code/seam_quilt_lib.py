import numpy as np
def seam_path(cost):
    R,C=cost.shape; E=cost.copy(); back=np.zeros((R,C),int)
    for r in range(1,R):
        prev=E[r-1]
        l=np.r_[np.inf,prev[:-1]]; m=prev; rr=np.r_[prev[1:],np.inf]
        st=np.vstack([l,m,rr]); k=st.argmin(0)
        back[r]=np.arange(C)+k-1; E[r]+=st.min(0)
    p=np.zeros(R,int); p[-1]=E[-1].argmin()
    for r in range(R-1,0,-1): p[r-1]=back[r,p[r]]
    return p,E[-1].min()/R
def make(img,x0,y0,W,H,o):
    C=img[y0:y0+H+o,x0:x0+W+o].copy()
    A,B=C[:,:o],C[:,W:W+o]; p,ch=seam_path(((A-B)**2).sum(-1))
    T=C[:,:W].copy()
    for r in range(H+o): T[r,:p[r]]=B[r,:p[r]]
    A,B=T[:o],T[H:H+o]; q,cv=seam_path(((A-B)**2).sum(-1).T)
    out=T[:H].copy()
    for c in range(W): out[:q[c],c]=B[:q[c],c]
    return out,ch+cv
