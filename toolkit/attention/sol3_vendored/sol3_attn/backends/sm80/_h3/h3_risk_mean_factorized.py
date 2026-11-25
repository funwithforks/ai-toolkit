import triton
import triton.language as tl
@triton.jit
def keyvar_mean(KV,MEAN,N:tl.constexpr,D:tl.constexpr):
 bh=tl.program_id(0);js=tl.arange(0,32);ds=tl.arange(0,D);sm=tl.zeros((D,),tl.float32)
 for start in range(0,N,32):
  j=start+js;kv=tl.load(KV+(bh*N+j[:,None])*D+ds[None,:],(j<N)[:,None],0)
  sm+=tl.sum(kv,0)
 tl.store(MEAN+bh*D+ds,sm/N)
@triton.jit
def factorized_mean_threshold(QB,M,KM,TH,N:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,ALPHA:tl.constexpr,BQ:tl.constexpr):
 qi,bh=tl.program_id(0),tl.program_id(1);qs=qi*BQ+tl.arange(0,BQ);ds=tl.arange(0,D)
 q=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0)
 mk=tl.load(M+bh*(2*D+2)+ds);mc=tl.load(M+bh*(2*D+2)+2*D);kv=tl.load(KM+bh*D+ds)
 # Applicable only to beta2=0, uncapped KeyVar risk, no Value contrast.
 mu=tl.sum(q*mk[None,:],1)*SCALE*1.4426950408889634
 delta=.5*SCALE*SCALE*tl.sum(q*q*kv[None,:],1)
 threshold=mu+mc+ALPHA*delta*1.4426950408889634
 tl.store(TH+bh*N+qs,threshold,qs<N)
MEAN_CACHE={}
def factorized_threshold(qb,mom,kv,th2,n,d,scale,alpha):
 import torch,triton
 key=(tuple(qb.shape),qb.device)
 if key not in MEAN_CACHE:MEAN_CACHE[key]=torch.empty((qb.shape[0],d),device=qb.device,dtype=torch.float32)
 mean=MEAN_CACHE[key]
 keyvar_mean[(qb.shape[0],)](kv,mean,n,d,num_warps=4,num_stages=1)
 factorized_mean_threshold[(triton.cdiv(n,32),qb.shape[0])](qb,mom,mean,th2,n,d,scale,alpha,32,num_warps=4,num_stages=1)
