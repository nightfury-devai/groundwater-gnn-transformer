# Run from the project root: python output/make_fig_pe.py
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
pe=np.load("encoding/pe_table.npy")
L,d=pe.shape
# independent recompute
pos=np.arange(L)[:,None]; i=np.arange(d//2)[None,:]
a=pos/np.power(10000.0,2*i/d); chk=np.zeros((L,d)); chk[:,0::2]=np.sin(a); chk[:,1::2]=np.cos(a)
assert np.abs(chk-pe).max()<1e-12
D=np.linalg.norm(pe[:,None,:]-pe[None,:,:],axis=2)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9})
fig,ax=plt.subplots(1,2,figsize=(10,3.6),gridspec_kw={"width_ratios":[2.2,1]})
im=ax[0].imshow(pe,aspect="auto",cmap="RdBu_r",vmin=-1,vmax=1,interpolation="nearest")
ax[0].set_xlabel("Embedding dimension (0 to 63)"); ax[0].set_ylabel("Position in window (0 = oldest, 7 = latest)")
ax[0].set_yticks(range(L)); ax[0].set_title("(a) Positional encoding table (8 x 64)",loc="left",fontsize=9)
cb=fig.colorbar(im,ax=ax[0],fraction=0.025,pad=0.02); cb.set_label("value")
im2=ax[1].imshow(D,cmap="viridis")
for r in range(L):
    for c in range(L):
        ax[1].text(c,r,f"{D[r,c]:.1f}",ha="center",va="center",fontsize=6.5,color="white" if D[r,c]<2.6 else "black")
ax[1].set_xticks(range(L)); ax[1].set_yticks(range(L)); ax[1].set_xlabel("Position"); ax[1].set_ylabel("Position")
ax[1].set_title("(b) Distance between position vectors",loc="left",fontsize=9)
fig.tight_layout(); fig.savefig("figures/fig4_positional_encoding.png",dpi=300); fig.savefig("figures/fig4_positional_encoding.pdf")
print("ok",D[0,1],D[3,4],D[0,7],D[0,2],D[0,3])
