from sol3_attn.backends.sm80.paged import run_paged
from sol3_attn.backends.sm80.representatives import launch_merge
from .sol3_semantics import require_cap0
from .representative_merge import rep_merge
import math,pathlib,torch,triton,flashinfer
import triton.language as tl
from . import block_selector as selector
from .block_routing import route_only
from .pair_routing import route_only as route_pair
from .flashinfer_plan_vectorized import vectorize_plan
CACHE={};LAST={}
def plan_info_scalars(info):return [int(x) for x in info]
@triton.jit
def csr_lengths(FLAGS,LENGTHS,T:tl.constexpr,N:tl.constexpr,ACTIVE:tl.constexpr,SKIP:tl.constexpr,NN:tl.constexpr):
 row=tl.program_id(0);bh=row//ACTIVE;qi=row%ACTIVE+SKIP;js=tl.arange(0,NN)
 flag=tl.load(FLAGS+(bh*N+qi)*N+js,js<N,-2);sizes=tl.minimum(64,tl.maximum(T-js*64,0))
 exact=tl.where(flag==-1,sizes,tl.maximum(flag,0));tl.store(LENGTHS+row,tl.sum(tl.where(js<N,exact,0),0))
@triton.jit
def csr_indices(FLAGS,BITS,PTR,INDEX,T:tl.constexpr,N:tl.constexpr,ACTIVE:tl.constexpr,SKIP:tl.constexpr,H:tl.constexpr,CHUNK:tl.constexpr):
 row=tl.program_id(0);bh=row//ACTIVE;qi=row%ACTIVE+SKIP;startptr=tl.load(PTR+row);kk=tl.arange(0,CHUNK);ts=tl.arange(0,64);carry=tl.full((),0,tl.int32)
 for start in range(0,N,CHUNK):
  js=start+kk;flag=tl.load(FLAGS+(bh*N+qi)*N+js,js<N,-2);word=tl.load(BITS+(bh*N+qi)*N+js,js<N,0)
  keys=js[:,None]*64+ts[None,:];selected=(((word[:,None]>>ts[None,:].to(tl.uint64))&1)!=0)&(flag>=0)[:,None]
  selected=(selected|(flag==-1)[:,None])&(js<N)[:,None]&(keys<T);cnt=tl.sum(selected.to(tl.int32),1);amount=tl.sum(cnt,0)
  if amount>0:
   rank=tl.cumsum(selected.to(tl.int32),1)-1;off=tl.cumsum(cnt,0)-cnt+carry;dest=startptr+off[:,None]+rank
   tl.store(INDEX+dest,keys*H+bh%H,selected)
  carry+=amount
@triton.jit
def merge(UR,LR,MR,OS,LS,O,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,SKIPTOK:tl.constexpr,PREFIX:tl.constexpr,TOTAL:tl.constexpr,LSE_BASE2:tl.constexpr):
 rs=tl.program_id(0)*16+tl.arange(0,16);ds=tl.arange(0,D);valid=(rs<TOTAL);h=rs%H;b=rs//(T*H);t=(rs//H)%T;live=valid&(t>=PREFIX)
 r=tl.load(LR+rs,live,0.);m=tl.load(MR+rs,live,0.);u=tl.load(UR+rs[:,None]*D+ds[None,:],live[:,None],0.)
 packed=(b*H+h)*(T-SKIPTOK)+(t-SKIPTOK);os=tl.load(OS+packed[:,None]*D+ds[None,:],live[:,None],0.);ls=tl.load(LS+packed,live,-float('inf'))
 if not LSE_BASE2:ls*=1.4426950408889634
 lr=tl.where(r>0,m+tl.log2(tl.maximum(r,1.e-30)),-float('inf'));mx=tl.maximum(lr,ls);wr=tl.exp2(lr-mx);ws=tl.exp2(ls-mx)
 out=(wr[:,None]*(u/tl.maximum(r,1.e-30)[:,None])+ws[:,None]*os)/(wr+ws)[:,None]
 tl.store(O+rs[:,None]*D+ds[None,:],out,live[:,None])
def calibrate(wrapper,device,d):
 q=torch.zeros((16,1,d),device=device,dtype=torch.bfloat16);k=torch.zeros((7,1,1,d),device=device,dtype=q.dtype);v=torch.zeros_like(k)
 wrapper.plan(torch.tensor([0,16],dtype=torch.int32),torch.tensor([0,7],dtype=torch.int32),torch.arange(7,device=device,dtype=torch.int32),torch.ones(1,dtype=torch.int32),1,1,d,1,q_data_type=q.dtype,kv_data_type=k.dtype,causal=False)
 out,lse=wrapper.run(q,(k,v),return_lse=True);number=float(lse[0,0]);assert torch.isfinite(out).all()
 if abs(number-math.log(7))<1.e-3:return False
 if abs(number-math.log2(7))<1.e-3:return True
 raise RuntimeError('Unrecognized FlashInfer LSE base: '+str(number))
def workspace(q,skip):
 b,t,h,d=q.shape;assert b==1;active=triton.cdiv(t,64)-skip;ta=t-skip*64;key=(tuple(q.shape),q.device,skip)
 if key not in CACHE:
  groups=b*h*active;max_indices=groups*t;assert max_indices<2**31
  lengths=torch.empty(groups,device=q.device,dtype=torch.int32);ptr=torch.empty(groups+1,device=q.device,dtype=torch.int32);pages=torch.empty(max_indices,device=q.device,dtype=torch.int32)
  qp=torch.empty((b*h,ta,d),device=q.device,dtype=q.dtype)
  os=torch.empty((b*h*ta,1,d),device=q.device,dtype=q.dtype);lse=torch.empty((b*h*ta,1),device=q.device)
  qo=((torch.arange(b*h,dtype=torch.int32)[:,None]*ta)+torch.arange(active,dtype=torch.int32)[None,:]*64).flatten();qo=torch.cat([qo,torch.tensor([b*h*ta],dtype=torch.int32)])
  fws=torch.empty(128*1024*1024,device=q.device,dtype=torch.uint8);wrapper=vectorize_plan(flashinfer.BatchPrefillWithPagedKVCacheWrapper(fws,kv_layout='NHD',backend='fa2'));base2=calibrate(wrapper,q.device,d)
  CACHE[key]=dict(lengths=lengths,ptr=ptr,pages=pages,qp=qp,os=os,lse=lse,qo=qo,last_cpu=torch.ones(groups,dtype=torch.int32),float_workspace=fws,wrapper=wrapper,lse_base2=base2)
 w=CACHE[key]
 if 'static_plan' not in w:
  rows=b*h*active;dense_ptr=torch.arange(rows+1,dtype=torch.int32)*t
  w['wrapper'].plan(w['qo'],dense_ptr,w['pages'],w['last_cpu'],1,1,d,1,causal=False,q_data_type=q.dtype,kv_data_type=q.dtype,use_fp16_qk_reduction=False)
  info=plan_info_scalars(w['wrapper']._plan_info);w['static_plan']=(len(info)==15 and info[14]==0)
  if w['static_plan']:
   w['wrapper']._paged_kv_indptr_buf=w['ptr'];w['wrapper']._paged_kv_indices_buf=w['pages']
 return w
def fast(q,k,v,*,cfg,**kw):
 require_cap0(cfg)
 if kw.get('tau',1.)<=-100:
  from sol3_attn.backends.workspace import native_attention as sol_attn
  return sol_attn(q,k,v,**kw)
 assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous();route=(route_pair if cfg.get('route_pair',False) else route_only)(q,k,v,cfg=dict(cfg,capacity=1,organization='fused',compute_prefix=False),**kw)
 b,t,h,d,n,s=[route[x] for x in ['b','t','h','d','n','s']];prefix=route['qskip'];skip=prefix//64;active=n-skip;w=workspace(q,skip);flags,bits=route['flags'],route['bits'];groups=b*h*active
 csr_lengths[(groups,)](flags,w['lengths'],t,n,active,skip,triton.next_power_of_2(n),num_warps=4);w['ptr'][0]=0;w['ptr'][1:]=torch.cumsum(w['lengths'],0,dtype=torch.int32)
 ptr_cpu=None;total=None
 if not w['static_plan']:
  ptr_cpu=w['ptr'].cpu();total=int(ptr_cpu[-1]);assert 0<total<=w['pages'].numel();assert bool((ptr_cpu[1:]>ptr_cpu[:-1]).all())
 csr_indices[(groups,)](flags,bits,w['ptr'],w['pages'],t,n,active,skip,h,16,num_warps=4)
 w['qp'].view(b,h,t-skip*64,d).copy_(q[:,skip*64:].permute(0,2,1,3))
 if w['static_plan']:
  w['wrapper']._kv_lens_buffer[:groups].copy_(w['lengths']);w['wrapper']._sm_scale=s
 else:w['wrapper'].plan(w['qo'],ptr_cpu,w['pages'][:total],w['last_cpu'],1,1,d,1,causal=False,sm_scale=s,q_data_type=q.dtype,kv_data_type=k.dtype,use_fp16_qk_reduction=False)
 run_paged(w['wrapper'], w['qp'].view(-1,1,d),(k.view(-1,1,1,d),v.view(-1,1,1,d)),out=w['os'],lse=w['lse'],return_lse=True)
 # Normalize representative and exact-token contributions together.
 out=torch.empty_like(q)
 launch_merge(rep_merge, ((n,b*h)), q,route['kc'],route['vm'],flags,w['os'],w['lse'],out,s,t,n,h,d,True,prefix,cfg.get('rep_bk',64),True,True,skip*64,w['lse_base2'],num_warps=4,num_stages=1)
 if prefix and cfg.get('compute_prefix',True):
  from sol3_attn.backends.workspace import dense_prefix as flash_attn_func
  out[:,:prefix]=flash_attn_func(q[:,:prefix],k,v,softmax_scale=s)
 LAST.clear();LAST.update(route=route,workspace=w,ptr_cpu=ptr_cpu,total_indices=total,skip_groups=skip,active_groups=active)
 return out
