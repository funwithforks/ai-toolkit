"""Exact real-number beta2=0 no-cap log-risk mean factorization.

Block weighting is 1/N as in risk_keyvar_threshold, including final tail block.
KC is the existing BF16 block mean; QB and KV are existing FP32 summaries.
R epsilon/tail sizes are already encoded in C=log2(n*max(R,1e-6)).
The factorized reduction can round differently from TF32x3 dot products.
"""
import torch,triton
import triton.language as tl
CACHE={}
@triton.jit
def keyvar_mean(KV,KM,N:tl.constexpr,D:tl.constexpr,BK:tl.constexpr=32):
 bh=tl.program_id(0);ds=tl.arange(0,D);js0=tl.arange(0,BK);acc=tl.zeros((D,),tl.float32)
 for start in range(0,N,BK):
  js=start+js0;value=tl.load(KV+(bh*N+js[:,None])*D+ds[None,:],(js<N)[:,None],0);acc+=tl.sum(value,0)
 tl.store(KM+bh*D+ds,acc/N)
@triton.jit
def mean_threshold(QB,M,KM,TH2,N:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,ALPHA:tl.constexpr,BQ:tl.constexpr=16):
 tile,bh=tl.program_id(0),tl.program_id(1);qs=tile*BQ+tl.arange(0,BQ);ds=tl.arange(0,D)
 q=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0)
 mk=tl.load(M+bh*(2*D+2)+ds);mc=tl.load(M+bh*(2*D+2)+2*D);kv=tl.load(KM+bh*D+ds)
 threshold=tl.sum(q*mk[None,:],1)*SCALE*1.4426950408889634+mc+.5*SCALE*SCALE*ALPHA*tl.sum(q*q*kv[None,:],1)*1.4426950408889634
 tl.store(TH2+bh*N+qs,threshold,qs<N)
def threshold(qb,mom,kv,th2,*,n,d,scale,beta2,risk_cap,risk_alpha=1.,risk_contrast=False):
 assert float(beta2)==0. and float(risk_cap)==float('inf') and not risk_contrast,'Linear log-risk mean shortcut requires beta2=0, risk cap infinite and no Value contrast'
 key=(tuple(kv.shape),kv.device)
 if key not in CACHE:CACHE[key]=torch.empty((kv.shape[0],d),device=kv.device,dtype=torch.float32)
 km=CACHE[key];keyvar_mean[(kv.shape[0],)](kv,km,n,d,num_warps=4,num_stages=1);mean_threshold[(triton.cdiv(n,16),kv.shape[0])](qb,mom,km,th2,n,d,scale,float(risk_alpha),num_warps=4,num_stages=1)
