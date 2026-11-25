"""Same FP32 current raw-Key SCORE stored immediately after dot; all moments identical."""
import triton
import triton.language as tl
@triton.jit
def partial_scores_early(QB,K,FLAGS,P,SCORE,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,QSKIP:tl.constexpr,HEAD0:tl.constexpr,BQ:tl.constexpr,BK:tl.constexpr):
 tile,jtile,local=tl.program_id(0),tl.program_id(1),tl.program_id(2);bh=local+HEAD0;b,h=bh//H,bh%H;NB:tl.constexpr=BK//64
 qs=tile*BQ+tl.arange(0,BQ);js=jtile*NB+tl.arange(0,NB);ds=tl.arange(0,D);kt=jtile*BK+tl.arange(0,BK)
 flag=tl.load(FLAGS+(bh*N+qs[:,None])*N+js[None,:],(qs<N)[:,None]&(js<N)[None,:],-2)
 mix=(qs<N)[:,None]&((qs+1)*64>QSKIP)[:,None]&(js<N)[None,:]&(flag>=0)
 count=tl.zeros((BQ,NB),tl.float32);mean=tl.zeros((BQ,NB),tl.float32);m2=tl.zeros((BQ,NB),tl.float32)
 if tl.sum(tl.sum(mix.to(tl.int32),1),0)>0:
  q=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0).to(tl.bfloat16)
  key=tl.load(K+((b*T+kt[:,None]).to(tl.int64)*H+h)*D+ds[None,:],(kt<T)[:,None],0)
  scores=tl.dot(q,tl.trans(key))*SCALE*1.4426950408889634
  active=tl.reshape(tl.broadcast_to(mix[:,:,None],(BQ,NB,64)),(BQ,BK))&(kt<T)[None,:]
  tl.store(SCORE+(local*N+qs[:,None]).to(tl.int64)*T+kt[None,:],scores,active)
  chunks=tl.reshape(scores,(BQ,NB,64));valid=tl.reshape(kt<T,(NB,64));ns=tl.maximum(tl.minimum(64,T-js*64),1).to(tl.float32)
  av=tl.sum(tl.where(valid[None,:,:],chunks,0.),2)/ns[None,:];vv=tl.sum(tl.where(valid[None,:,:],(chunks-av[:,:,None])*(chunks-av[:,:,None]),0.),2)
  count=tl.where(mix,ns[None,:],0.);mean=tl.where(mix,av,0.);m2=tl.where(mix,vv,0.)
 ix=((local*N+qs[:,None])*N+js[None,:])*3;mask=(qs<N)[:,None]&(js<N)[None,:]
 tl.store(P+ix,count,mask);tl.store(P+ix+1,mean,mask);tl.store(P+ix+2,m2,mask)
