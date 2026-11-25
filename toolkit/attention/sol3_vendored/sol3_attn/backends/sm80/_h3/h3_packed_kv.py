import torch,triton
import triton.language as tl
CACHE={}
@triton.jit
def stage_kv(K,V,KK,VV,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,TOTAL:tl.constexpr,BLOCK:tl.constexpr):
 x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=x<TOTAL
 ds=x%D;token=(x//D)%T;bh=x//(D*T);b=bh//H;h=bh%H
 src=((b*T+token).to(tl.int64)*H+h)*D+ds
 tl.store(KK+x,tl.load(K+src,valid,0),valid);tl.store(VV+x,tl.load(V+src,valid,0),valid)
def packed_kv(k,v,block=2048):
 key=(tuple(k.shape),k.device)
 if key not in CACHE:
  b,t,h,d=k.shape;CACHE[key]=(torch.empty((b,h,t,d),device=k.device,dtype=k.dtype),torch.empty((b,h,t,d),device=v.device,dtype=v.dtype))
 kk,vv=CACHE[key];total=k.numel();b,t,h,d=k.shape
 stage_kv[(triton.cdiv(total,block),)](k,v,kk,vv,t,h,d,total,block,num_warps=4)
 return kk,vv
