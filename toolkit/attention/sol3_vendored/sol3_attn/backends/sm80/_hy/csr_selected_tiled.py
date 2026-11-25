"""Specialized CSR/selected-token kernels; same routes and mass cap.
The accumulation order differs from FlashInfer.
"""
import triton
import triton.language as tl

@triton.jit
def csr_parallel(FLAGS,BITS,PTR,INDEX,T:tl.constexpr,N:tl.constexpr,
                 ACTIVE:tl.constexpr,SKIP:tl.constexpr,H:tl.constexpr,
                 HEAD_BASE:tl.constexpr,CK:tl.constexpr,NN:tl.constexpr):
 row=tl.program_id(0);tile=tl.program_id(1);lh=row//ACTIVE;qi=row%ACTIVE+SKIP
 allj=tl.arange(0,NN)
 f=tl.load(FLAGS+(lh*N+qi)*N+allj,allj<N,-2)
 sizes=tl.minimum(64,tl.maximum(T-allj*64,0))
 counts=tl.where(f==-1,sizes,tl.maximum(f,0))
 carry=tl.sum(tl.where((allj<N)&(allj<tile*CK),counts,0),0)
 js=tile*CK+tl.arange(0,CK);ts=tl.arange(0,64)
 flag=tl.load(FLAGS+(lh*N+qi)*N+js,js<N,-2)
 word=tl.load(BITS+(lh*N+qi)*N+js,js<N,0)
 keys=js[:,None]*64+ts[None,:]
 chosen=(((word[:,None]>>ts[None,:].to(tl.uint64))&1)!=0)&(flag>=0)[:,None]
 chosen=(chosen|(flag==-1)[:,None])&(js<N)[:,None]&(keys<T)
 amount=tl.sum(chosen.to(tl.int32),1)
 rank=tl.cumsum(chosen.to(tl.int32),1)-1
 offset=tl.cumsum(amount,0)-amount+carry+tl.load(PTR+row)
 tl.store(INDEX+offset[:,None]+rank,keys*H+lh+HEAD_BASE,chosen)

@triton.jit
def selected_attention(QP,K,V,PTR,INDEX,OS,LS,T:tl.constexpr,TA:tl.constexpr,
                       ACTIVE:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,
                       QT:tl.constexpr,BK:tl.constexpr,LSE_BASE2:tl.constexpr):
 row=tl.program_id(0);qtile=tl.program_id(1)
 qs=(row//ACTIVE)*TA+(row%ACTIVE)*64+qtile*QT+tl.arange(0,QT)
 valid=(row%ACTIVE)*64+qtile*QT+tl.arange(0,QT)<TA
 ds=tl.arange(0,D);js=tl.arange(0,BK)
 q=tl.load(QP+qs[:,None].to(tl.int64)*D+ds[None,:],valid[:,None],0)
 begin=tl.load(PTR+row);end=tl.load(PTR+row+1)
 mx=tl.full((QT,),-float('inf'),tl.float32)
 den=tl.zeros((QT,),tl.float32);acc=tl.zeros((QT,D),tl.float32)
 for start in range(begin,end,BK):
  ids=start+js;ok=ids<end
  ix=tl.load(INDEX+ids,ok,0)
  k=tl.load(K+ix[:,None].to(tl.int64)*D+ds[None,:],ok[:,None],0)
  logits=tl.dot(q,tl.trans(k))*SCALE*1.4426950408889634
  logits=tl.where(ok[None,:],logits,-float('inf'))
  nextmx=tl.maximum(mx,tl.max(logits,1));a=tl.exp2(mx-nextmx)
  p=tl.exp2(logits-nextmx[:,None])
  v=tl.load(V+ix[:,None].to(tl.int64)*D+ds[None,:],ok[:,None],0)
  acc=tl.dot(p.to(q.dtype),v,acc*a[:,None])
  den=den*a+tl.sum(p,1);mx=nextmx
 tl.store(OS+qs[:,None].to(tl.int64)*D+ds[None,:],acc/den[:,None],valid[:,None])
 lse=mx+tl.log2(den)
 if not LSE_BASE2:lse*=0.6931471805599453
 tl.store(LS+qs,lse,valid)
