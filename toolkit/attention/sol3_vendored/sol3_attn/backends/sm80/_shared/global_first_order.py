"""Compute the global first moment of block-centered keys and values."""
import torch,triton
import triton.language as tl
CACHE={}
@triton.jit
def centered_kv(K,V,KC,HM,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BT:tl.constexpr=128,BK:tl.constexpr=32,BV:tl.constexpr=32):
 tile,bh=tl.program_id(0),tl.program_id(1);kd=(tile//triton.cdiv(D,BV))*BK+tl.arange(0,BK);vd=(tile%triton.cdiv(D,BV))*BV+tl.arange(0,BV);ts=tl.arange(0,BT);acc=tl.zeros((BK,BV),tl.float32)
 for start in range(0,T,BT):
  kt=start+ts;valid=kt<T;group=kt//64
  key=tl.load(K+(kt[:,None].to(tl.int64)*H+bh)*D+kd[None,:],valid[:,None],0).to(tl.float32)
  center=tl.load(KC+(bh*N+group[:,None])*D+kd[None,:],valid[:,None],0).to(tl.float32)
  value=tl.load(V+(kt[:,None].to(tl.int64)*H+bh)*D+vd[None,:],valid[:,None],0)
  residual=(key-center).to(value.dtype);acc+=tl.dot(tl.trans(residual),value)
 tl.store(HM+(bh*D+kd[:,None])*D+vd[None,:],acc/T)
def global_first_order(k,v,kc):
 b,t,h,d=k.shape;assert b==1;n=triton.cdiv(t,64);key=(tuple(k.shape),k.device)
 if key not in CACHE:CACHE[key]=torch.empty((h,d,d),device=k.device,dtype=torch.bfloat16)
 hm=CACHE[key];centered_kv[(triton.cdiv(d,32)*triton.cdiv(d,32),h)](k,v,kc,hm,t,n,h,d,num_warps=4,num_stages=3);return hm
