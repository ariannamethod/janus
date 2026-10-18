/*
 * resonance-janus-bpe-probe.c — backward probes for resonance-janus-bpe.c
 *
 *   cc resonance-janus-bpe-probe.c -O2 -lm -o probe                                    (this tree's file)
 *   cc resonance-janus-bpe-probe.c -O2 -lm -DRJ_SRC='"/abs/path/old/resonance-janus-bpe.c"' -o probe_old
 *   ./probe fwd                              forward fingerprint on fixed inputs (must match across revisions)
 *   ./probe grad [depth] [T] [eps] [n]       gradcheck: f64-reference central differences, n params per class over all blocks
 *                                            (T=0 -> cfg_from_depth's T)
 *   ./probe step data.txt [depth]            one real train_model() step, max|dw| per class of the stepped matrix A
 *   ./probe train data.txt N [lr] [depth]    fixed seed, fixed batch stream, 90/10 split: step|train|val
 *
 * Every probe runs with a populated Dario field: df_overlay adds a weight-independent bias to the logits,
 * the same vector on every row once DF.res sits at its fixed point (df_bias checks this).
 */
#ifndef RJ_SRC
#define RJ_SRC "resonance-janus-bpe.c"
#endif
#define main resonance_janus_bpe_main
#include RJ_SRC
#undef main

#define NCLS 17
#define REF_V 2048   /* DarioField.embeds is [2048]: V above it is out of the file's range */
static const char*CN[NCLS]={"tok_emb","pos_emb","rms1","wq","wk","wv","wr","wvr","wj","gate",
                            "wo","rms2","w_gate","w_up","w_down","rms_f","out_w"};
static int cls_blk(int k){return k>=2&&k<=14;}
static float*cls_ptr(Ptrs*p,int k,int b){
    switch(k){case 0:return p->tok_emb;case 1:return p->pos_emb;case 2:return p->rms1[b];
    case 3:return p->wq[b];case 4:return p->wk[b];case 5:return p->wv[b];case 6:return p->wr[b];
    case 7:return p->wvr[b];case 8:return p->wj[b];case 9:return p->gate[b];case 10:return p->wo[b];
    case 11:return p->rms2[b];case 12:return p->w_gate[b];case 13:return p->w_up[b];case 14:return p->w_down[b];
    case 15:return p->rms_f;default:return p->out_w;}
}
static int cls_n(int k,const Cfg*c){
    int E=c->E,H=c->H,D=c->D,T=c->T,M=c->M,V=c->V;
    switch(k){case 0:return V*E;case 1:return T*E;case 2:case 11:case 15:return E;
    case 3:case 4:case 5:case 7:return H*E*D;case 6:return H*E*T;
    case 8:case 10:return E*E;case 9:return H*3;case 12:case 13:return E*M;
    case 14:return M*E;default:return E*V;}
}

/* fixed Dario field with repeats (co-occurrence, prophecy, destiny all non-zero); returns the overlay bias */
static float*df_bias(int V){
    df_init();for(int i=0;i<48;i++)df_ingest((((i*5)%11)*97+3)%V);
    float*b0=calloc(V,4),*b1=calloc(V,4);
    df_overlay(b0,V);memset(b0,0,(size_t)V*4);df_overlay(b0,V);df_overlay(b1,V);
    if(memcmp(b0,b1,(size_t)V*4)){fprintf(stderr,"df_overlay is not stationary\n");exit(4);}
    free(b1);return b0;
}

/* CE of the forward's logits, reduced in double (fwd with tgt=NULL leaves raw logits + overlay in a->lg) */
static double loss_d(SingleModel*m,Acts*a,int*tok,int*tgt,Cfg*c){
    fwd(&m->w,a,tok,NULL,c);int T=c->T,V=c->V;double L=0;
    for(int t=0;t<T;t++){const float*l=a->lg+(size_t)t*V;double mx=l[0],s=0;
    for(int v=1;v<V;v++)if(l[v]>mx)mx=l[v];
    for(int v=0;v<V;v++)s+=exp((double)l[v]-mx);L+=mx+log(s)-l[tgt[t]];}
    return L/T;
}

/* Double transcription of fwd(): same masks, sigmoid gate with the same +-20 clamp, wr_h as E x T (row stride T),
   same f32 constants (scale, cd, dt, 1e-5f, 1e-6f), same overlay bias. probe_grad refuses to run if it disagrees
   with fwd() beyond f32 precision. */
static void rmm_w(double*C,const double*A,const float*B,int m,int k,int n){
    for(int i=0;i<m;i++)for(int j=0;j<n;j++){double s=0;for(int p=0;p<k;p++)s+=A[i*k+p]*B[p*n+j];C[i*n+j]=s;}
}
static void rmm(double*C,const double*A,const double*B,int m,int k,int n){
    for(int i=0;i<m;i++)for(int j=0;j<n;j++){double s=0;for(int p=0;p<k;p++)s+=A[i*k+p]*B[p*n+j];C[i*n+j]=s;}
}
static void rrms(double*o,const double*x,const float*g,int T,int E){
    for(int t=0;t<T;t++){double ss=0;for(int e=0;e<E;e++)ss+=x[t*E+e]*x[t*E+e];
    double inv=1/sqrt(ss/E+1e-5f);for(int e=0;e<E;e++)o[t*E+e]=g[e]*x[t*E+e]*inv;}
}
static void rcsm(double*a,int T){
    for(int i=0;i<T;i++){double mx=a[i*T],s=0;for(int j=1;j<=i;j++)if(a[i*T+j]>mx)mx=a[i*T+j];
    for(int j=0;j<T;j++){a[i*T+j]=j<=i?exp(a[i*T+j]-mx):0;s+=a[i*T+j];}for(int j=0;j<=i;j++)a[i*T+j]/=s;}
}
static double rsig(float x){return x>20.0f?1:x<-20.0f?0:1/(1+exp(-(double)x));}
static double ref_loss(Ptrs*w,const int*tok,const int*tgt,const Cfg*c,const float*bias){
    const int E=c->E,T=c->T,H=c->H,D=c->D,M=c->M,V=c->V;
    static double x[MAX_T*MAX_DIM],rm[MAX_T*MAX_DIM],cat[MAX_T*MAX_DIM],y[MAX_T*MAX_DIM],je[MAX_T*MAX_DIM],
        q[MAX_T*MAX_DIM],k[MAX_T*MAX_DIM],v[MAX_T*MAX_DIM],o[MAX_T*MAX_DIM],at[MAX_T*MAX_T],ja[MAX_T*MAX_T],
        mg[MAX_T*2*MAX_DIM],mu[MAX_T*2*MAX_DIM],js[MAX_T],lg[REF_V];
    const double hs=1.0f/sqrtf((float)D),cd=1+0.5f*calendar_dissonance(calendar_days_since_epoch()),dt=1+AML.prophecy_debt;
    for(int t=0;t<T;t++)for(int e=0;e<E;e++)x[t*E+e]=(double)w->tok_emb[tok[t]*E+e]+w->pos_emb[t*E+e];
    for(int b=0;b<c->B;b++){
        rrms(rm,x,w->rms1[b],T,E);memset(cat,0,sizeof cat);
        for(int h=0;h<H;h++){
            double gl[3];for(int i=0;i<3;i++)gl[i]=rsig(w->gate[b][h*3+i]);
            rmm_w(q,rm,w->wq[b]+h*E*D,T,E,D);rmm_w(k,rm,w->wk[b]+h*E*D,T,E,D);rmm_w(v,rm,w->wv[b]+h*E*D,T,E,D);
            for(int i=0;i<T;i++)for(int j=0;j<=i;j++){double s=0;for(int d=0;d<D;d++)s+=q[i*D+d]*k[j*D+d];at[i*T+j]=s*hs;}
            rcsm(at,T);rmm(o,at,v,T,T,D);
            for(int t=0;t<T;t++)for(int d=0;d<D;d++)cat[t*E+h*D+d]+=gl[0]*o[t*D+d];
            rmm_w(v,rm,w->wvr[b]+h*E*D,T,E,D);
            rmm_w(at,rm,w->wr[b]+h*E*T,T,E,T);for(int i=0;i<T*T;i++)at[i]*=hs;
            rcsm(at,T);rmm(o,at,v,T,T,D);
            for(int t=0;t<T;t++)for(int d=0;d<D;d++)cat[t*E+h*D+d]+=gl[1]*o[t*D+d];
            if(h==0){const float*wj=w->wj[b];
                for(int t=0;t<T;t++){const double*xt=rm+t*E;double*pr=je+t*E,nm=0,sv=0;
                for(int i=0;i<E;i++){double s=0;for(int j=0;j<E;j++)s+=wj[i*E+j]*xt[j];pr[i]=s;nm+=s*s;}
                for(int i=0;i<E;i++){double s=0;for(int j=0;j<E;j++)s+=wj[j*E+i]*pr[j];sv+=xt[i]*s;}
                js[t]=sv/(sqrt(nm)+1e-6f)*cd;}
                for(int i=0;i<T;i++)for(int j=0;j<=i;j++)ja[i*T+j]=js[i]*js[j]/dt;rcsm(ja,T);}
            for(int t=0;t<T;t++)for(int d=0;d<D;d++){double s=0;
                for(int j=0;j<=t;j++)s+=ja[t*T+j]*je[j*E+h*D+d];cat[t*E+h*D+d]+=gl[2]*s;}
        }
        rmm_w(y,cat,w->wo[b],T,E,E);for(int i=0;i<T*E;i++)x[i]+=y[i];
        rrms(rm,x,w->rms2[b],T,E);rmm_w(mg,rm,w->w_gate[b],T,E,M);rmm_w(mu,rm,w->w_up[b],T,E,M);
        for(int i=0;i<T*M;i++)mg[i]=(mg[i]>-20?mg[i]/(1+exp(-mg[i])):0)*mu[i];
        rmm_w(y,mg,w->w_down[b],T,M,E);for(int i=0;i<T*E;i++)x[i]+=y[i];
    }
    rrms(rm,x,w->rms_f,T,E);double L=0;
    for(int t=0;t<T;t++){rmm_w(lg,rm+t*E,w->out_w,1,E,V);for(int v2=0;v2<V;v2++)lg[v2]+=bias[v2];
    double mx=lg[0],s=0;for(int v2=1;v2<V;v2++)if(lg[v2]>mx)mx=lg[v2];
    for(int v2=0;v2<V;v2++)s+=exp(lg[v2]-mx);L+=mx+log(s)-lg[tgt[t]];}
    return L/T;
}

static Cfg probe_cfg(int depth,int T){
    Cfg c=cfg_from_depth(depth,BPE_VOCAB);if(T>0)c.T=T;
    if(depth<1||depth>MAX_BLK||c.E>MAX_DIM||c.T<2||c.T>MAX_T||c.V>REF_V){fprintf(stderr,"bad cfg depth=%d T=%d\n",depth,c.T);exit(2);}
    return c;
}

static int probe_fwd(void){
    const int cfgs[3][2]={{2,0},{4,16},{12,0}};
    for(int r=0;r<3;r++){Cfg c=probe_cfg(cfgs[r][0],cfgs[r][1]);int T=c.T,V=c.V,tok[MAX_T],tgt[MAX_T];
    float*bias=df_bias(V);double bm=0;for(int i=0;i<V;i++)if(fabs(bias[i])>bm)bm=fabs(bias[i]);
    srand(1);SingleModel m;single_init(&m,&c);Acts a;acts_alloc(&a,&c);
    srand(7);for(int t=0;t<T;t++){tok[t]=rand()%V;tgt[t]=rand()%V;}
    float L=fwd(&m.w,&a,tok,tgt,&c);uint64_t hsh=1469598103934665603ULL;
    const unsigned char*p=(const unsigned char*)a.lg;
    for(size_t i=0;i<(size_t)T*V*4;i++){hsh^=p[i];hsh*=1099511628211ULL;}
    printf("fwd depth=%d E=%d H=%d B=%d T=%d max|overlay|=%.4f loss=%.9g probs_fnv64=%016llx\n",
           cfgs[r][0],c.E,c.H,c.B,T,bm,L,(unsigned long long)hsh);
    acts_free(&a);single_free(&m);free(bias);}
    return 0;
}

/* Gradcheck point: generic weights — at init scale 0.02*sqrt(2/E) every softmax is ~uniform and gates sit at 1/2,
   so their backward terms are barely exercised. wj at 1.28/E keeps |wj x| ~ 1.3, janus logits sc_i*sc_j ~ O(1). */
static void gc_point(SingleModel*m,const Cfg*c){
    for(int i=0;i<m->n_params;i++)m->data[i]=randn()*0.1f;
    for(int b=0;b<c->B;b++){for(int i=0;i<c->E*c->E;i++)m->w.wj[b][i]*=12.8f/c->E;
    for(int i=0;i<c->H*3;i++)m->w.gate[b][i]=randn()*0.5f;
    for(int e=0;e<c->E;e++){m->w.rms1[b][e]=1+0.1f*randn();m->w.rms2[b][e]=1+0.1f*randn();}}
    for(int e=0;e<c->E;e++)m->w.rms_f[e]=1+0.1f*randn();
}

static double relerr(double a,double b){double d=fmax(fabs(a),fabs(b));return d>0?fabs(a-b)/d:0;}

static int probe_grad(int depth,int Tov,float eps,int n){
    Cfg c=probe_cfg(depth,Tov);const int T=c.T,V=c.V,E=c.E;float*bias=df_bias(V);
    srand(1);SingleModel m;single_init(&m,&c);Acts a;acts_alloc(&a,&c);
    int tok[MAX_T],tgt[MAX_T];for(int t=0;t<T;t++){tok[t]=rand()%V;tgt[t]=rand()%V;}
    double R0=ref_loss(&m.w,tok,tgt,&c,bias),F0=loss_d(&m,&a,tok,tgt,&c);gc_point(&m,&c);
    memset(m.grad,0,m.n_params*sizeof(float));
    clock_t t0=clock();float L=fwd(&m.w,&a,tok,tgt,&c);clock_t t1=clock();bwd(&m.w,&m.g,&a,tok,tgt,&c);clock_t t2=clock();
    double R=ref_loss(&m.w,tok,tgt,&c,bias),F=loss_d(&m,&a,tok,tgt,&c);
    printf("gradcheck depth=%d E=%d H=%d D=%d B=%d M=%d T=%d V=%d n=%d/class  gate: f64-ref central diff eps=%g | context: f32 fwd() central diff eps=1e-2\n",
           depth,E,c.H,c.D,c.B,c.M,T,V,n,eps);
    printf("fwd %.3fs bwd %.3fs (bwd/fwd %.2f)\n",(double)(t1-t0)/CLOCKS_PER_SEC,(double)(t2-t1)/CLOCKS_PER_SEC,
           (double)(t2-t1)/(double)(t1-t0>0?t1-t0:1));
    printf("ref check  init point: fwd=%.9f ref=%.9f rel=%.2e | gc point: fwd=%.9f ref=%.9f rel=%.2e (fwd ret %.6f)\n",
           F0,R0,relerr(F0,R0),F,R,relerr(F,R),L);
    if(relerr(F0,R0)>1e-5||relerr(F,R)>1e-5){printf("REF MISMATCH: f64 transcription disagrees with fwd()\n");return 3;}
    printf("%-7s %3s %7s %14s %14s %10s %14s %10s\n","class","blk","idx","analytic","num_f64","rel","num_f32","rel_f32");
    int fails=0;
    struct{double mx,nmin,nmax;int zero;}S[NCLS];
    for(int k=0;k<NCLS;k++){S[k].mx=0;S[k].nmin=1e30;S[k].nmax=0;S[k].zero=0;
    for(int i=0;i<n;i++){int b=cls_blk(k)?i%c.B:0,idx;
        if(k==0)idx=tok[rand()%T]*E+rand()%E;
        else if(k==16)idx=(rand()%E)*V+tgt[rand()%T];
        else idx=rand()%cls_n(k,&c);
        float*w=cls_ptr(&m.w,k,b)+idx;double an=cls_ptr(&m.g,k,b)[idx];float w0=*w;
        *w=w0+eps;float wp=*w;double Rp=ref_loss(&m.w,tok,tgt,&c,bias);
        *w=w0-eps;float wm=*w;double Rm=ref_loss(&m.w,tok,tgt,&c,bias);
        *w=w0+1e-2f;float fp=*w;double Fp=loss_d(&m,&a,tok,tgt,&c);
        *w=w0-1e-2f;float fm=*w;double Fm=loss_d(&m,&a,tok,tgt,&c);*w=w0;
        double nu=(Rp-Rm)/((double)wp-(double)wm),nf=(Fp-Fm)/((double)fp-(double)fm),rel=relerr(an,nu);
        if(an==0&&fabs(nu)>0)S[k].zero++;
        if(rel>S[k].mx)S[k].mx=rel;
        if(fabs(nu)<S[k].nmin)S[k].nmin=fabs(nu);
        if(fabs(nu)>S[k].nmax)S[k].nmax=fabs(nu);
        printf("%-7s %3d %7d %14.6e %14.6e %10.3e %14.6e %10.3e%s\n",CN[k],b,idx,an,nu,rel,nf,relerr(an,nf),rel<1e-2?"":"  <-- FAIL");
        fflush(stdout);}
    }
    printf("\n%-7s %10s %12s %12s %8s %s\n","class","max_rel","min|num|","max|num|","an==0","verdict");
    for(int k=0;k<NCLS;k++){int ok=S[k].mx<1e-2;fails+=!ok;
    printf("%-7s %10.3e %12.3e %12.3e %5d/%-2d %s\n",CN[k],S[k].mx,S[k].nmin,S[k].nmax,S[k].zero,n,ok?"PASS":"FAIL");}
    printf("%s: %d/%d classes fail (criterion max rel < 1e-2)\n",fails?"GRADCHECK FAIL":"GRADCHECK PASS",fails,NCLS);
    acts_free(&a);single_free(&m);free(bias);return fails?1:0;
}

static int probe_step(const char*path,int depth){
    Cfg c=probe_cfg(depth,0);df_init();srand(1);DualModel dm;dual_init(&dm,&c);size_t nb=(size_t)dm.A.n_params*sizeof(float);
    float*w0=malloc(nb),*wb=malloc(nb);memcpy(w0,dm.A.data,nb);memcpy(wb,dm.B.data,nb);
    train_model(&dm,path,1,3e-4f);
    Ptrs W0;assign_ptrs(&W0,w0,&c);int dead=0;
    printf("\nmatrix A (stepped at step 1)\n%-7s %12s %10s %10s\n","class","max|dw|","moved","total");
    for(int k=0;k<NCLS;k++){double mx=0;long mv=0,tot=0;
    for(int b=0;b<(cls_blk(k)?c.B:1);b++){float*p=cls_ptr(&dm.A.w,k,b),*q=cls_ptr(&W0,k,b);
    for(int i=0;i<cls_n(k,&c);i++){double d=fabs((double)p[i]-q[i]);if(d>mx)mx=d;mv+=d>0;tot++;}}
    dead+=mx==0;printf("%-7s %12.4e %10ld %10ld%s\n",CN[k],mx,mv,tot,mx==0?"  <-- DEAD":"");}
    printf("%d/%d classes did not move; matrix B untouched by step 1: %s\n",dead,NCLS,memcmp(wb,dm.B.data,nb)?"NO":"yes");
    free(w0);free(wb);dual_free(&dm);return 0;
}

static uint32_t xs=0x9E3779B9u;
static uint32_t xs32(void){xs^=xs<<13;xs^=xs>>17;xs^=xs<<5;return xs;}

static double val_loss(Ptrs*w,Acts*a,const int*bd,int ntr,int bl,Cfg*c,int NV){
    int T=c->T,V=c->V,tok[MAX_T],tgt[MAX_T];double vl=0;
    for(int k=0;k<NV;k++){int off=ntr+k*((bl-ntr-T-1)/NV);
    for(int t=0;t<T;t++){tok[t]=bd[off+t]%V;tgt[t]=bd[off+t+1]%V;}
    vl+=fwd(w,a,tok,tgt,c);}
    return vl/NV;
}

/* train_model() with a fixed batch stream and a held-out tail: A/B alternate per step, df_ingest per micro-batch,
   val on the blended weights inference uses (dual_blend) and on A and B alone */
static int probe_train(const char*path,int steps,float lr,int depth){
    Cfg c=probe_cfg(depth,0);const int T=c.T,V=c.V,NV=16;
    FILE*f=fopen(path,"r");if(!f){fprintf(stderr,"cannot open %s\n",path);return 1;}
    fseek(f,0,SEEK_END);long fsz=ftell(f);fseek(f,0,SEEK_SET);
    unsigned char*raw=malloc(fsz);if(fread(raw,1,fsz,f)!=(size_t)fsz){fclose(f);return 1;}fclose(f);
    int nm=V-256;if(nm>MAX_MERGES)nm=MAX_MERGES;bpe_learn_merges(raw,fsz,nm);
    int*bd=malloc(fsz*sizeof(int));int bl=bpe_encode(raw,fsz,bd,fsz);free(raw);
    int ntr=bl*9/10,tok[MAX_T],tgt[MAX_T];
    if(ntr-T-1<1||(bl-ntr-T-1)/NV<1){fprintf(stderr,"data too short: %d tokens\n",bl);return 1;}
    printf("[probe] depth=%d T=%d %d tokens, train [0,%d) val [%d,%d), %d val windows, GRAD_ACCUM=%d lr=%g\n",
           depth,T,bl,ntr,ntr,bl,NV,GRAD_ACCUM,lr);
    df_init();srand(1);DualModel dm;dual_init(&dm,&c);Acts a;acts_alloc(&a,&c);
    double acc=0;int nacc=0;clock_t t0=clock();
    for(int step=1;step<=steps;step++){
        SingleModel*act=(step%2)?&dm.A:&dm.B;
        memset(act->grad,0,act->n_params*sizeof(float));float sl=0;
        for(int ga=0;ga<GRAD_ACCUM;ga++){int off=xs32()%(uint32_t)(ntr-T-1);
            for(int t=0;t<T;t++){tok[t]=bd[off+t]%V;tgt[t]=bd[off+t+1]%V;}
            sl+=fwd(&act->w,&a,tok,tgt,&c);bwd(&act->w,&act->g,&a,tok,tgt,&c);
            for(int t=0;t<T;t++)df_ingest(tok[t]);}
        sl/=GRAD_ACCUM;for(int i=0;i<act->n_params;i++)act->grad[i]*=1.0f/GRAD_ACCUM;
        chuck_observe(sl);chuck_update(act->data,act->grad,act->cm,act->cv,act->n_params,lr);
        acc+=sl;nacc++;
        if(step==1||step%25==0){dual_blend(&dm);
            double vb=val_loss(&dm.bw,&a,bd,ntr,bl,&c,NV),va=val_loss(&dm.A.w,&a,bd,ntr,bl,&c,NV),vB=val_loss(&dm.B.w,&a,bd,ntr,bl,&c,NV);
            printf("step %5d | train %.4f | val %.4f | val_A %.4f | val_B %.4f | alpha %.3f | %.1fs\n",
                   step,acc/nacc,vb,va,vB,dm.blend_alpha,(double)(clock()-t0)/CLOCKS_PER_SEC);
            fflush(stdout);acc=0;nacc=0;}
    }
    acts_free(&a);dual_free(&dm);free(bd);return 0;
}

int main(int argc,char**argv){
    calendar_init();metajanus_init();
    if(argc>=2&&!strcmp(argv[1],"fwd"))return probe_fwd();
    if(argc>=2&&!strcmp(argv[1],"grad"))
        return probe_grad(argc>2?atoi(argv[2]):2,argc>3?atoi(argv[3]):0,argc>4?(float)atof(argv[4]):1e-4f,argc>5?atoi(argv[5]):12);
    if(argc>=3&&!strcmp(argv[1],"step"))return probe_step(argv[2],argc>3?atoi(argv[3]):12);
    if(argc>=4&&!strcmp(argv[1],"train"))
        return probe_train(argv[2],atoi(argv[3]),argc>4?(float)atof(argv[4]):3e-4f,argc>5?atoi(argv[5]):12);
    fprintf(stderr,"usage: %s fwd | grad [depth] [T] [eps] [n] | step data.txt [depth] | train data.txt N [lr] [depth]\n",argv[0]);
    return 2;
}
