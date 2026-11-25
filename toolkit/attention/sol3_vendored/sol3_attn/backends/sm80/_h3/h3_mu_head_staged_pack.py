from sol3_attn.backends.sm80.routing import launch_h3_scores
from sol3_attn.backends.workspace import active_call
"""Equivalent H3 complete Mixed-token moments with bounded head staging.

The BF16 32x256 score tiles and reduction across the same 256-Key tiles are
unchanged. Only the independent query/head launch organization changes.
"""
import torch,triton
import triton.language as tl
CACHE={};LAST={};HEAD_CHUNK=4;PACK_KEYS=False;ORIGINAL_ROUTE=None;FALLBACKS=0
@triton.jit
def pack_keys(K,PACK,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,AMOUNT:tl.constexpr,HEAD0,BLOCK:tl.constexpr):
 ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=ix<AMOUNT*T*D
 local=ix//(T*D);token=ix//D%T;d=ix%D;head=local+HEAD0
 src=((head//H*T+token).to(tl.int64)*H+head%H)*D+d
 tl.store(PACK+ix,tl.load(K+src,valid,0),valid)
@triton.jit
def score_moments(QB,K,FLAGS,RAW,P,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,QSKIP:tl.constexpr,BQ:tl.constexpr,BK:tl.constexpr,NT:tl.constexpr,PAD:tl.constexpr,HEAD0,PACKED:tl.constexpr):
 ktile,qtile,local=tl.program_id(0),tl.program_id(1),tl.program_id(2)
 bh=HEAD0+local;b,h=bh//H,bh%H
 qs=qtile*BQ+tl.arange(0,BQ);ds=tl.arange(0,D);kt=ktile*BK+tl.arange(0,BK)
 flag=tl.load(FLAGS+(bh*N+qs[:,None])*N+kt[None,:]//64,(qs<N)[:,None]&(kt<T)[None,:],-2)
 eligible=(qs<N)[:,None]&((qs+1)*64>QSKIP)[:,None]&(kt<T)[None,:]&(flag>=0)
 count=tl.sum(eligible.to(tl.float32),1);mean=tl.zeros((BQ,),tl.float32);m2=tl.zeros((BQ,),tl.float32)
 if tl.sum(count,0)>0:
  qb=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0).to(tl.bfloat16)
  if PACKED:ki=(local*T+kt[:,None]).to(tl.int64)*D+ds[None,:]
  else:ki=((b*T+kt[:,None]).to(tl.int64)*H+h)*D+ds[None,:]
  k=tl.load(K+ki,(kt<T)[:,None],0)
  scores=tl.dot(qb,tl.trans(k))*SCALE*1.4426950408889634
  mean=tl.sum(tl.where(eligible,scores,0.),1)/tl.maximum(count,1.)
  m2=tl.sum(tl.where(eligible,(scores-mean[:,None])*(scores-mean[:,None]),0.),1)
  tl.store(RAW+(local*N+qs[:,None]).to(tl.int64)*PAD+kt[None,:],scores,(qs<N)[:,None]&(kt<T)[None,:])
 ix=((local*N+qs)*NT+ktile)*3
 tl.store(P+ix,count,qs<N);tl.store(P+ix+1,mean,qs<N);tl.store(P+ix+2,m2,qs<N)
@triton.jit
def reduce_moments(P,CUT,MU,SIGMA,N:tl.constexpr,NT:tl.constexpr,NN:tl.constexpr,BETA:tl.constexpr,HEAD0):
 qi,local=tl.program_id(0),tl.program_id(1);js=tl.arange(0,NN);ix=((local*N+qi)*NT+js)*3
 count=tl.load(P+ix,js<NT,0.);mean=tl.load(P+ix+1,js<NT,0.);m2=tl.load(P+ix+2,js<NT,0.)
 amount=tl.sum(count,0);mu=tl.sum(count*mean,0)/tl.maximum(amount,1.)
 var=tl.sum(m2+count*(mean-mu)*(mean-mu),0)/tl.maximum(amount,1.)
 sigma=tl.sqrt(tl.maximum(var,0.)+1.e-6);cut=tl.where(amount>0,mu+BETA*sigma,float('inf'))
 target=(HEAD0+local)*N+qi;tl.store(CUT+target,cut);tl.store(MU+target,mu);tl.store(SIGMA+target,sigma)
@triton.jit
def select_cached(FLAGS,BITS,RAW,CUT,T:tl.constexpr,N:tl.constexpr,QSKIP:tl.constexpr,BQ:tl.constexpr,PAD:tl.constexpr,HEAD0):
 j,qtile,local=tl.program_id(0),tl.program_id(1),tl.program_id(2);bh=HEAD0+local
 qs=qtile*BQ+tl.arange(0,BQ);ts=tl.arange(0,64);kt=j*64+ts
 ix=(bh*N+qs)*N+j;flag=tl.load(FLAGS+ix,qs<N,-2)
 mixed=(qs<N)&((qs+1)*64>QSKIP)&(flag>=0)
 scores=tl.load(RAW+(local*N+qs[:,None]).to(tl.int64)*PAD+kt[None,:],mixed[:,None]&(kt<T)[None,:],0.)
 cut=tl.load(CUT+bh*N+qs,qs<N,float('inf'))
 chosen=mixed[:,None]&(kt<T)[None,:]&(scores>cut[:,None])
 count=tl.sum(chosen.to(tl.int32),1);tokenbit=tl.full((64,),1,tl.uint64)<<ts.to(tl.uint64)
 bits=tl.sum(tl.where(chosen,tokenbit[None,:],0),1)
 tl.store(FLAGS+ix,tl.where(flag>=0,count,flag),qs<N);tl.store(BITS+ix,bits,qs<N)
def route(qb,k,flags,bits,cut,*,t,n,h,d,scale,qskip,beta,bq=32,bk=256):
 global FALLBACKS
 head_limit=active_call().h3_score_heads
 heads=k.shape[0]*h;pad=n*64;nt=triton.cdiv(t,bk)
 if bq!=32 or bk!=256:
  if ORIGINAL_ROUTE is None:raise RuntimeError('Original route must be bound before using noncanonical arithmetic tiles')
  FALLBACKS+=1
  LAST.clear();LAST.update(fallback_original_for_arithmetic_tile=True,fallback_count=FALLBACKS,requested_tile=[bq,bk])
  return ORIGINAL_ROUTE(qb,k,flags,bits,cut,t=t,n=n,h=h,d=d,scale=scale,qskip=qskip,beta=beta,bq=bq,bk=bk)
 # Bounded worst-case allocation; capacity follows the actual input shape.
 per_head=n*pad*4+n*nt*3*4+(t*d*k.element_size() if PACK_KEYS else 0)
 if per_head>1536*1024**2 and ORIGINAL_ROUTE is not None:
  FALLBACKS+=1
  LAST.clear();LAST.update(fallback_original_for_workspace_bound=True,fallback_count=FALLBACKS)
  return ORIGINAL_ROUTE(qb,k,flags,bits,cut,t=t,n=n,h=h,d=d,scale=scale,qskip=qskip,beta=beta,bq=bq,bk=bk)
 shape_key=(tuple(k.shape),k.device,bq,bk,head_limit,PACK_KEYS)
 if shape_key in CACHE:chunk=CACHE[shape_key][0].shape[0]
 else:
  CACHE.clear();LAST.clear()
  free,_=torch.cuda.mem_get_info(k.device)
  reusable=torch.cuda.memory_reserved(k.device)-torch.cuda.memory_allocated(k.device)
  available=max(0,free+reusable-512*1024**2)
  chunk=min(heads,max(1,min(head_limit,(1536*1024**2)//per_head,available//per_head)))
  if available<per_head and ORIGINAL_ROUTE is not None:
   FALLBACKS+=1
   LAST.update(fallback_original_for_memory=True,fallback_count=FALLBACKS)
   return ORIGINAL_ROUTE(qb,k,flags,bits,cut,t=t,n=n,h=h,d=d,scale=scale,qskip=qskip,beta=beta,bq=bq,bk=bk)
 key=shape_key
 if key not in CACHE:
  CACHE.clear()
  CACHE[key]=(torch.empty((chunk,n,pad),device=k.device,dtype=torch.float32),torch.empty((chunk,n,nt,3),device=k.device,dtype=torch.float32),torch.empty((heads,n),device=k.device,dtype=torch.float32),torch.empty((heads,n),device=k.device,dtype=torch.float32),torch.empty((chunk,t,d),device=k.device,dtype=k.dtype) if PACK_KEYS else None)
 raw,p,mu,sigma,packed=CACHE[key]
 for head0 in range(0,heads,chunk):
  amount=min(chunk,heads-head0)
  if PACK_KEYS:pack_keys[(triton.cdiv(amount*t*d,2048),)](k,packed,t,h,d,amount,head0,2048,num_warps=4)
  launch_h3_scores(score_moments, ((nt,triton.cdiv(n,bq),amount)), qb,packed if PACK_KEYS else k,flags,raw,p,t,n,h,d,scale,qskip,bq,bk,nt,pad,head0,PACK_KEYS,num_warps=4,num_stages=2)
  reduce_moments[(n,amount)](p,cut,mu,sigma,n,nt,triton.next_power_of_2(nt),beta,head0,num_warps=4)
  select_cached[(n,triton.cdiv(n,bq),amount)](flags,bits,raw,cut,t,n,qskip,bq,pad,head0,num_warps=4,num_stages=1)
 LAST.clear();LAST.update(mu=mu,sigma=sigma,cut=cut,partial=p,raw=raw,head_chunk=chunk,packed_keys=PACK_KEYS,raw_fresh_each_head_chunk=True,all_eligible_keys_each_query=True,bounded_raw_bytes=raw.numel()*raw.element_size(),same_arithmetic_tile=[bq,bk])
