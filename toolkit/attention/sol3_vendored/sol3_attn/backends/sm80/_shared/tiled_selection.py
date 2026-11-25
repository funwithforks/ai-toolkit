import triton
import triton.language as tl
@triton.jit
def classify_tiled(QB,KC,C,M,KV,VC,TH2,FLAGS,SCALE:tl.constexpr,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,SINK0:tl.constexpr,SINK1:tl.constexpr,USE_TH2:tl.constexpr,QSKIP:tl.constexpr,QBATCH:tl.constexpr,JBATCH:tl.constexpr,RISK_CAP:tl.constexpr,RISK_ALPHA:tl.constexpr,RISK_CONTRAST:tl.constexpr):
 qi,jtile,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2);qs=qi*QBATCH+tl.arange(0,QBATCH);js=jtile*JBATCH+tl.arange(0,JBATCH);ds=tl.arange(0,D);active=(qs<N)&((qs+1)*64>QSKIP)
 qb=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0);kc=tl.load(KC+(bh*N+js[:,None])*D+ds[None,:],(js<N)[:,None],0).to(tl.float32)
 mk=tl.load(M+bh*(2*D+2)+ds);vk=tl.load(M+bh*(2*D+2)+D+ds);mc=tl.load(M+bh*(2*D+2)+2*D);vc=tl.load(M+bh*(2*D+2)+2*D+1);c=tl.load(C+bh*N+js,js<N,0);s=SCALE*1.4426950408889634
 mu=tl.sum(qb*mk[None,:],1)*s;var=tl.sum(qb*qb*vk[None,:],1)*s*s;th1=mu+B1*tl.sqrt(tl.maximum(var,0)+1.e-6);th2=mu+mc+B2*tl.sqrt(tl.maximum(var+vc,0)+1.e-6)
 if USE_TH2:th2=tl.load(TH2+bh*N+qs,qs<N,0)
 proxy=tl.sum(qb[:,None,:]*kc[None,:,:],2)*s;ex=active[:,None]&((proxy>th1[:,None])|(tl.abs(qs[:,None]-js[None,:])<=1)|((js>=SINK0)&(js<SINK1))[None,:]);risk=proxy+c[None,:]
 if RISK_CAP>0:
  kv=tl.load(KV+(bh*N+js[:,None])*D+ds[None,:],(js<N)[:,None],0);delta=.5*SCALE*SCALE*tl.sum((qb*qb)[:,None,:]*kv[None,:,:],2);clipped=tl.minimum(delta,RISK_CAP)
  if RISK_CONTRAST:
   contrast=tl.load(VC+bh*N+js,js<N,0);ns=tl.minimum(64,tl.maximum(T-js*64,1)).to(tl.float32);rms=tl.exp2(c)/ns;gamma=tl.exp2(clipped*1.4426950408889634)-1.;risk+=tl.log2(tl.maximum(rms[None,:]+RISK_ALPHA*gamma*contrast[None,:],1.e-6)/rms[None,:])
  else:risk+=RISK_ALPHA*clipped*1.4426950408889634
 pure=active[:,None]&~ex&(risk<th2[:,None]);flag=tl.where(ex,-1,tl.where(pure,-2,0));ix=(bh*N+qs[:,None])*N+js[None,:];tl.store(FLAGS+ix,flag,(qs<N)[:,None]&(js<N)[None,:])
@triton.jit
def global_route_pair(QB,K,FLAGS,BITS,CUT,SCALE:tl.constexpr,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BF16:tl.constexpr,QSKIP:tl.constexpr,QBATCH:tl.constexpr):
 tile,jtile,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2);b,h=bh//H,bh%H;qs=tile*QBATCH+tl.arange(0,QBATCH);js=jtile*2+tl.arange(0,2);ts=tl.arange(0,64);ds=tl.arange(0,D);kt=jtile*128+tl.arange(0,128)
 ix=(bh*N+qs[:,None])*N+js[None,:];valid=(qs<N)[:,None]&(js<N)[None,:];flag=tl.load(FLAGS+ix,valid,-2);cut=tl.load(CUT+bh*N+qs,qs<N,float('inf'));mix=(qs<N)[:,None]&((qs+1)*64>QSKIP)[:,None]&(flag>=0)&(cut<float('inf'))[:,None]&valid
 count=tl.zeros((QBATCH,2),tl.int32);bits=tl.zeros((QBATCH,2),tl.uint64)
 if tl.sum(tl.sum(mix.to(tl.int32),1),0)>0:
  q=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0);k=tl.load(K+((b*T+kt[:,None]).to(tl.int64)*H+h)*D+ds[None,:],(kt<T)[:,None],0)
  if BF16:a=tl.dot(q.to(tl.bfloat16),tl.trans(k))*SCALE*1.4426950408889634
  else:a=tl.dot(q,tl.trans(k.to(tl.float32)),input_precision='tf32x3')*SCALE*1.4426950408889634
  score=tl.reshape(a,(QBATCH,2,64));tokens=js[:,None]*64+ts[None,:];selected=mix[:,:,None]&(tokens<T)[None,:,:]&(score>cut[:,None,None]);count=tl.sum(selected.to(tl.int32),2);bit=tl.full((64,),1,tl.uint64)<<ts.to(tl.uint64);bits=tl.sum(tl.where(selected,bit[None,None,:],0),2)
 tl.store(FLAGS+ix,tl.where(flag>=0,count,flag),valid);tl.store(BITS+ix,bits,valid)
