import torch,triton
import triton.language as tl
CACHE={}
@triton.jit
def covariance_partials(K,V,P,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,SPLITS:tl.constexpr,BLOCKS:tl.constexpr,SINK_BLOCKS:tl.constexpr):
 tile,split,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2)
 ri=tile//2*64+tl.arange(0,64);ci=tile%2*64+tl.arange(0,64);ts=tl.arange(0,64)
 b=bh//H;h=bh%H;acc=tl.zeros((64,64),tl.float32)
 for j0 in range(0,BLOCKS):
  j=split*BLOCKS+j0
  if j<N and j>=SINK_BLOCKS:
   kt=j*64+ts;live=kt<T
   ki=((b*T+kt[:,None]).to(tl.int64)*H+h)*D+ri[None,:]
   vi=((b*T+kt[:,None]).to(tl.int64)*H+h)*D+ci[None,:]
   kk=tl.load(K+ki,live[:,None],0).to(tl.float32);vv=tl.load(V+vi,live[:,None],0)
   nn=tl.minimum(64,T-j*64);km=tl.sum(kk,0)/nn
   centered=tl.where(live[:,None],kk-km[None,:],0.).to(tl.bfloat16)
   acc=tl.dot(tl.trans(centered),vv,acc)
 dest=((bh*SPLITS+split)*D+ri[:,None])*D+ci[None,:]
 tl.store(P+dest,acc)
@triton.jit
def reduce_covariance(P,OUT,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,SPLITS:tl.constexpr,SS:tl.constexpr,BLOCK:tl.constexpr):
 tile,bh=tl.program_id(0),tl.program_id(1);x=tile*BLOCK+tl.arange(0,BLOCK);sp=tl.arange(0,SS)
 val=tl.load(P+((bh*SPLITS+sp[:,None])*D*D+x[None,:]).to(tl.int64),(sp<SPLITS)[:,None]&(x<D*D)[None,:],0)
 out=tl.sum(val,0)/T;tl.store(OUT+bh*D*D+x,out,x<D*D)
def global_covariance(k,v,prefix,blocks=32):
 b,t,h,d=k.shape;n=triton.cdiv(t,64);splits=triton.cdiv(n,blocks);assert d==128
 key=(tuple(k.shape),k.device,blocks)
 if key not in CACHE:CACHE[key]=(torch.empty((b*h,splits,d,d),device=k.device,dtype=torch.float32),torch.empty((b*h,d,d),device=k.device,dtype=torch.bfloat16))
 parts,out=CACHE[key]
 covariance_partials[(4,splits,b*h)](k,v,parts,t,n,h,d,splits,blocks,triton.cdiv(prefix,64),num_warps=4,num_stages=1)
 reduce_covariance[(triton.cdiv(d*d,256),b*h)](parts,out,t,h,d,splits,triton.next_power_of_2(splits),256,num_warps=4,num_stages=1)
 return out
