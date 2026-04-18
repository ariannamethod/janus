/*
 * resonance-janus-bpe.c — Resonance + Janus Hybrid on BPE
 *
 * The strongest architecture: all three attention mechanisms
 * (QKV + RRPRAM + Janus self-resonance) with learned 3-way gate,
 * Dario field overlay, BPE tokenizer, configurable depth.
 *
 * θ = ε + γ + αδ
 *   ε = dual trained weights (W_A, W_B blended by calendar)
 *   γ = RRPRAM positional + QKV semantic + Janus self-resonance
 *   δ = Dario field (Kuramoto chambers, co-occurrence, prophecy)
 *
 * Parameters from cfg_from_depth(depth):
 *   E = depth * 32, H = 4, D = E/H, B = depth, M = E*2, T = 64
 *   depth=12: E=384, H=4, D=96, B=12, M=768 (~24M params)
 *
 *   cc resonance-janus-bpe.c -O2 -lm -o resonance-janus-bpe
 *   ./resonance-janus-bpe --train data.txt --depth 12 --steps 15000
 *   ./resonance-janus-bpe --load model.bin --generate "Q: who are you"
 *
 * By Arianna Method. הרזוננס לא נשבר
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════════
 * CONFIGURATION — depth knob, BPE vocab
 * ═══════════════════════════════════════════════════════════════════ */

#define MAX_BLK     16
#define MAX_DIM     512
#define MAX_T       256
#define MAX_MERGES  1792      /* BPE_VOCAB - 256 */
#define NSTEPS      12
#define SENT_LEN    40

static int BPE_VOCAB = 2048;

typedef struct {
    int T, E, H, D, B, M, V;
} Cfg;

static Cfg cfg_from_depth(int depth, int vocab) {
    Cfg c;
    c.T = (depth >= 8) ? 64 : 32;
    c.E = depth * 32;
    c.H = (depth < 4) ? 2 : 4;
    c.D = c.E / c.H;
    c.B = depth;
    c.M = c.E * 2;
    c.V = vocab;
    return c;
}

/* ═══════════════════════════════════════════════════════════════════
 * CALENDAR DRIFT — Gregorian vs Hebrew (Metonic cycle)
 * ═══════════════════════════════════════════════════════════════════ */

#define AM_ANNUAL_DRIFT     11.25f
#define AM_GREGORIAN_YEAR   365.25f
#define AM_METONIC_YEARS    19
#define AM_METONIC_LEAPS    7
#define AM_MAX_UNCORRECTED  33.0f
static const int g_metonic_leap_years[7] = {3, 6, 8, 11, 14, 17, 19};
static time_t g_epoch_t = 0;

static float clamp01(float x) {
    if (!isfinite(x)) return 0.0f;
    return x < 0 ? 0 : (x > 1 ? 1 : x);
}

static void calendar_init(void) {
    struct tm e; memset(&e,0,sizeof(e));
    e.tm_year=2024-1900; e.tm_mon=9; e.tm_mday=3; e.tm_hour=12;
    g_epoch_t = mktime(&e);
}

static int calendar_days_since_epoch(void) {
    if (g_epoch_t<=0) return 0;
    return (int)(difftime(time(NULL), g_epoch_t)/86400.0);
}

static float calendar_cumulative_drift(int days) {
    float years = (float)days/AM_GREGORIAN_YEAR;
    float base = years * AM_ANNUAL_DRIFT;
    int full = (int)(years/AM_METONIC_YEARS);
    float corr = (float)(full*AM_METONIC_LEAPS)*30.0f;
    float partial = fmodf(years,(float)AM_METONIC_YEARS);
    int yic = (int)partial + 1;
    for (int i=0;i<AM_METONIC_LEAPS;i++)
        if (g_metonic_leap_years[i]<=yic) corr += 30.0f;
    return base - corr;
}

static float calendar_dissonance(int d) {
    float drift = calendar_cumulative_drift(d);
    return clamp01(fabsf(fmodf(drift,AM_MAX_UNCORRECTED))/AM_MAX_UNCORRECTED);
}

/* MetaJanus */
typedef struct { int birth_days; float birth_drift,birth_dissonance; time_t birth_time; int alive; } MetaJanus;
static MetaJanus MJ={0};
static void metajanus_init(void) {
    if(MJ.alive) return; calendar_init();
    MJ.birth_days=calendar_days_since_epoch();
    MJ.birth_drift=calendar_cumulative_drift(MJ.birth_days);
    MJ.birth_dissonance=calendar_dissonance(MJ.birth_days);
    MJ.birth_time=time(NULL); MJ.alive=1;
}
static float metajanus_personal_dissonance(void) {
    return clamp01(fabsf(calendar_cumulative_drift(calendar_days_since_epoch())-MJ.birth_drift)/AM_MAX_UNCORRECTED);
}

/* AML Physics */
typedef struct {
    float prophecy_debt,destiny_bias,wormhole,resonance,trauma,tension,pain,entropy_floor;
    int prophecy_horizon,tunnel_skip_max; float tunnel_threshold;
} AMLState;
static AMLState AML={.destiny_bias=0.1f,.wormhole=0.02f,.resonance=0.5f,.entropy_floor=0.01f,.prophecy_horizon=12,.tunnel_skip_max=7,.tunnel_threshold=0.55f};

static float compute_prophecy_debt(const float *l, int ch, int n) {
    if(n<=0||ch<0||ch>=n) return 0; float mx=l[0];
    for(int i=1;i<n;i++) if(l[i]>mx) mx=l[i];
    float d=mx-l[ch]; return d>0?d/(d+1):0;
}

static void apply_destiny(float *l, int n) {
    if(AML.destiny_bias<0.001f) return;
    float mx=l[0]; for(int i=1;i<n;i++) if(l[i]>mx) mx=l[i];
    for(int i=0;i<n;i++) l[i]-=(mx-l[i])*AML.destiny_bias*0.5f;
}

/* Kuramoto Chambers */
enum { CH_FEAR=0,CH_LOVE,CH_RAGE,CH_VOID,CH_FLOW,CH_COMPLEX,NCH };
static float chambers[NCH]={0};
static const float ch_decay[NCH]={0.95f,0.95f,0.93f,0.96f,0.94f,0.97f};
static void update_chambers(int si) {
    float d=(float)si/NSTEPS;
    if(d<0.33f) chambers[CH_FLOW]+=0.05f;
    else if(d<0.66f) chambers[CH_FEAR]+=0.04f;
    else chambers[CH_VOID]+=0.05f;
    if(d>0.75f) chambers[CH_COMPLEX]+=0.03f;
    float K=0.02f,old[NCH]; memcpy(old,chambers,sizeof(old));
    for(int i=0;i<NCH;i++){
        for(int j=0;j<NCH;j++) if(i!=j) chambers[i]+=K*sinf(old[j]-old[i]);
        chambers[i]=clamp01(chambers[i]*ch_decay[i]);
    }
}

/* Dario Field — co-occurrence memory + prophecy + destiny */
#define DF_MAX_COOC   16384
#define DF_MAX_CTX    64
#define DF_MAX_PROPH  16
#define DF_DIM        32

typedef struct { int target; float strength; int age; int fulfilled; } DFProphecy;

typedef struct {
    int cooc_src[DF_MAX_COOC], cooc_dst[DF_MAX_COOC];
    float cooc_val[DF_MAX_COOC]; int cooc_n;
    int context[DF_MAX_CTX]; int ctx_len;
    DFProphecy prophecy[DF_MAX_PROPH]; int prophecy_n;
    float destiny[DF_DIM]; float dest_mag;
    float alpha, beta, gamma_d;
    float embeds[2048][DF_DIM]; int embed_init[2048]; /* max BPE vocab */
    float trauma, dissonance, ent, res, emergence;
    int step;
} DarioField;
static DarioField DF;

static float *df_embed(int id) {
    if (id<0||id>=BPE_VOCAB) return NULL;
    if (!DF.embed_init[id]) {
        unsigned h=2166136261u;
        for(int i=0;i<4;i++){h^=(id>>(i*8))&0xFF;h*=16777619u;}
        for(int d=0;d<DF_DIM;d++){h=h*1103515245+12345;DF.embeds[id][d]=((float)(h&0x7FFFFFFF)/(float)0x7FFFFFFF-0.5f)*0.1f;}
        float norm=0; for(int d=0;d<DF_DIM;d++) norm+=DF.embeds[id][d]*DF.embeds[id][d];
        norm=sqrtf(norm+1e-12f); for(int d=0;d<DF_DIM;d++) DF.embeds[id][d]/=norm;
        DF.embed_init[id]=1;
    }
    return DF.embeds[id];
}

static float df_cosine(const float *a, const float *b) {
    float dot=0,na=0,nb=0;
    for(int i=0;i<DF_DIM;i++){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
    return dot/(sqrtf(na)*sqrtf(nb)+1e-12f);
}

static void df_cooc_update(int src, int dst, float delta) {
    for(int i=0;i<DF.cooc_n;i++) if(DF.cooc_src[i]==src&&DF.cooc_dst[i]==dst){DF.cooc_val[i]+=delta;return;}
    if(DF.cooc_n>=DF_MAX_COOC) return;
    int i=DF.cooc_n++; DF.cooc_src[i]=src;DF.cooc_dst[i]=dst;DF.cooc_val[i]=delta;
}

static void df_init(void) {
    memset(&DF,0,sizeof(DF));
    DF.alpha=0.15f;DF.beta=0.10f;DF.gamma_d=0.12f;
}

static void df_ingest(int tok) {
    if(tok<0||tok>=BPE_VOCAB) return;
    for(int c=0;c<DF.ctx_len;c++){float w=1.0f/(float)(DF.ctx_len-c);df_cooc_update(DF.context[c],tok,w*0.3f);}
    for(int i=0;i<DF.prophecy_n;i++){if(DF.prophecy[i].target==tok)DF.prophecy[i].fulfilled=1;DF.prophecy[i].age++;}
    int w=0;for(int i=0;i<DF.prophecy_n;i++)if(!DF.prophecy[i].fulfilled&&DF.prophecy[i].age<50)DF.prophecy[w++]=DF.prophecy[i];
    DF.prophecy_n=w;
    float best=-1;int pred=-1;
    for(int i=0;i<DF.cooc_n;i++)if(DF.cooc_src[i]==tok&&DF.cooc_val[i]>best){best=DF.cooc_val[i];pred=DF.cooc_dst[i];}
    if(pred>=0&&DF.prophecy_n<DF_MAX_PROPH)DF.prophecy[DF.prophecy_n++]=(DFProphecy){pred,0.3f,0,0};
    float *e=df_embed(tok);
    if(e){for(int d=0;d<DF_DIM;d++)DF.destiny[d]=0.1f*e[d]+0.9f*DF.destiny[d];
    float n=0;for(int d=0;d<DF_DIM;d++)n+=DF.destiny[d]*DF.destiny[d];DF.dest_mag=sqrtf(n+1e-12f);}
    if(DF.ctx_len<DF_MAX_CTX)DF.context[DF.ctx_len++]=tok;
    else{memmove(DF.context,DF.context+1,(DF_MAX_CTX-1)*sizeof(int));DF.context[DF_MAX_CTX-1]=tok;}
    DF.trauma*=0.97f;DF.step++;
}

static void df_overlay(float *logits, int V) {
    float *H_sig=calloc(V,4),*F_sig=calloc(V,4),*A_sig=calloc(V,4);
    float h_max=0,f_max=0;
    int ctx_start=(DF.ctx_len>8)?DF.ctx_len-8:0;
    for(int c=ctx_start;c<DF.ctx_len;c++){float decay=powf(0.9f,(float)(DF.ctx_len-1-c));
    for(int i=0;i<DF.cooc_n;i++)if(DF.cooc_src[i]==DF.context[c]&&DF.cooc_dst[i]<V)H_sig[DF.cooc_dst[i]]+=DF.cooc_val[i]*decay;}
    for(int i=0;i<V;i++) if(H_sig[i]>h_max) h_max=H_sig[i];
    if(h_max>1e-6f) for(int i=0;i<V;i++) H_sig[i]/=h_max;
    for(int i=0;i<V;i++){float *te=df_embed(i);if(!te)continue;float score=0;
    for(int p=0;p<DF.prophecy_n;p++){if(DF.prophecy[p].fulfilled)continue;float *pe=df_embed(DF.prophecy[p].target);
    if(!pe)continue;float sim=df_cosine(te,pe);if(sim<0)sim=0;score+=DF.prophecy[p].strength*sim*logf(1.0f+(float)DF.prophecy[p].age);}
    F_sig[i]=score;}
    for(int i=0;i<V;i++) if(F_sig[i]>f_max) f_max=F_sig[i];
    if(f_max>1e-6f) for(int i=0;i<V;i++) F_sig[i]/=f_max;
    if(DF.dest_mag>1e-6f){float a_max=0;for(int i=0;i<V;i++){float *te=df_embed(i);if(te)A_sig[i]=df_cosine(te,DF.destiny)*DF.dest_mag;}
    for(int i=0;i<V;i++)if(fabsf(A_sig[i])>a_max)a_max=fabsf(A_sig[i]);
    if(a_max>1e-6f) for(int i=0;i<V;i++) A_sig[i]/=a_max;}
    float gate=1.0f/(1.0f+expf(-(DF.res-0.5f)*4.0f));
    for(int i=0;i<V;i++){logits[i]+=DF.alpha*H_sig[i]*(1.0f/(1.0f+expf(-gate*2.0f)))+DF.beta*F_sig[i]*(1.0f/(1.0f+expf(-gate*1.5f)))+DF.gamma_d*A_sig[i];}
    float density=(DF.cooc_n>100)?1.0f:(float)DF.cooc_n/100.0f;
    DF.res=clamp01(density*0.4f+(1.0f-DF.dissonance)*0.3f+0.3f);
    DF.ent=clamp01(DF.dissonance*0.4f+(1.0f-DF.res)*0.3f+0.2f);
    DF.emergence=clamp01((1.0f-DF.ent)*DF.res);
    free(H_sig);free(F_sig);free(A_sig);
}

/* ═══════════════════════════════════════════════════════════════════
 * BPE TOKENIZER
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct { int a,b,result; } MergeRule;
typedef struct { MergeRule merges[MAX_MERGES]; int n_merges, vocab_size; } BPETokenizer;
static BPETokenizer BPE = {.n_merges=0,.vocab_size=256};

static int bpe_encode(const unsigned char *text, int len, int *out, int max) {
    int n=0; for(int i=0;i<len&&n<max;i++) out[n++]=text[i];
    for(int m=0;m<BPE.n_merges;m++){
        MergeRule *mr=&BPE.merges[m]; int j=0;
        for(int i=0;i<n;i++){
            if(i+1<n&&out[i]==mr->a&&out[i+1]==mr->b){out[j++]=mr->result;i++;}
            else out[j++]=out[i];
        }
        n=j;
    }
    return n;
}

static int bpe_decode_token(int tok, char *buf, int max) {
    if(tok<256){if(max>0)buf[0]=(char)tok;return 1;}
    for(int m=BPE.n_merges-1;m>=0;m--)
        if(BPE.merges[m].result==tok){int n1=bpe_decode_token(BPE.merges[m].a,buf,max);return n1+bpe_decode_token(BPE.merges[m].b,buf+n1,max-n1);}
    if(max>0)buf[0]='?';return 1;
}

static void bpe_learn_merges(const unsigned char *data, int len, int nm) {
    int *tok=malloc(len*sizeof(int)); int n=len;
    for(int i=0;i<n;i++) tok[i]=data[i];
    int max_m=nm<MAX_MERGES?nm:MAX_MERGES;
    for(int m=0;m<max_m;m++){
        int ba=-1,bb=-1,bc=0;
        for(int i=0;i+1<n;i++){int a=tok[i],b=tok[i+1];int c=0;
        for(int j=i;j+1<n;j++) if(tok[j]==a&&tok[j+1]==b) c++;
        if(c>bc){bc=c;ba=a;bb=b;}}
        if(bc<2) break;
        int nid=256+m; BPE.merges[m]=(MergeRule){ba,bb,nid};
        BPE.n_merges=m+1; BPE.vocab_size=256+m+1;
        int j=0;
        for(int i=0;i<n;i++){if(i+1<n&&tok[i]==ba&&tok[i+1]==bb){tok[j++]=nid;i++;}else tok[j++]=tok[i];}
        n=j;
        if((m+1)%200==0) printf("  merge %d/%d  vocab=%d  tokens=%d\n",m+1,max_m,nid+1,n);
    }
    free(tok);
}

/* ═══════════════════════════════════════════════════════════════════
 * MATH
 * ═══════════════════════════════════════════════════════════════════ */

static void matmul(float*C,const float*A,const float*B,int m,int k,int n){
    for(int i=0;i<m;i++) for(int j=0;j<n;j++){float s=0;for(int p=0;p<k;p++)s+=A[i*k+p]*B[p*n+j];C[i*n+j]=s;}}
static void matmul_atb(float*C,const float*A,const float*B,int m,int k,int n){
    for(int i=0;i<m;i++) for(int j=0;j<n;j++){float s=0;for(int p=0;p<k;p++)s+=A[p*m+i]*B[p*n+j];C[i*n+j]+=s;}}
static void matmul_abt(float*C,const float*A,const float*B,int m,int k,int n){
    for(int i=0;i<m;i++) for(int j=0;j<n;j++){float s=0;for(int p=0;p<k;p++)s+=A[i*k+p]*B[j*k+p];C[i*n+j]=s;}}
static void row_softmax(float*x,int n){
    float mx=x[0];for(int i=1;i<n;i++)if(x[i]>mx)mx=x[i];
    float s=0;for(int i=0;i<n;i++){x[i]=expf(x[i]-mx);s+=x[i];}
    if(s>0)for(int i=0;i<n;i++)x[i]/=s;}
static float sigmoidf(float x){if(x>20.0f)return 1.0f;if(x<-20.0f)return 0.0f;return 1.0f/(1.0f+expf(-x));}
static void gate_sigmoid(float*x,int n){for(int i=0;i<n;i++)x[i]=sigmoidf(x[i]);}
static float siluf(float x){return x>-20?x/(1+expf(-x)):0;}
static float siluf_grad(float x){if(x<-20)return 0;float s=1/(1+expf(-x));return s*(1+x*(1-s));}
static void rmsnorm_fwd(float*o,const float*x,const float*g,int T,int E){
    for(int t=0;t<T;t++){float ss=0;for(int e=0;e<E;e++)ss+=x[t*E+e]*x[t*E+e];
    float inv=1/sqrtf(ss/E+1e-5f);for(int e=0;e<E;e++)o[t*E+e]=g[e]*x[t*E+e]*inv;}}
static float randn(void){
    float u1=(rand()+1.0f)/(RAND_MAX+2.0f),u2=(rand()+1.0f)/(RAND_MAX+2.0f);
    return sqrtf(-2*logf(u1))*cosf(6.2831853f*u2);}

/* ═══════════════════════════════════════════════════════════════════
 * MODEL — 3-way gated attention + Dario field + BPE
 *
 * Per head: Wq[E,D] + Wk[E,D] + Wv[E,D] + Wr[E,T] + Wvr[E,D]
 *           + Wj[E,E] + gate[3]
 * MLP (SwiGLU): W_gate[E,M] + W_up[E,M] + W_down[M,E]
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    float *tok_emb, *pos_emb;
    float *rms1[MAX_BLK];
    float *wq[MAX_BLK],*wk[MAX_BLK],*wv[MAX_BLK];
    float *wr[MAX_BLK],*wvr[MAX_BLK],*wj[MAX_BLK],*gate[MAX_BLK];
    float *wo[MAX_BLK],*rms2[MAX_BLK];
    float *w_gate[MAX_BLK],*w_up[MAX_BLK],*w_down[MAX_BLK];
    float *rms_f, *out_w;
} Ptrs;

static int model_size(Cfg *c) {
    int E=c->E,H=c->H,D=c->D,T=c->T,M=c->M,V=c->V,B=c->B;
    int s=V*E+T*E; /* embeddings */
    for(int b=0;b<B;b++){
        s+=E; /* rms1 */
        s+=H*E*D*3; /* wq,wk,wv */
        s+=H*E*T; /* wr */
        s+=H*E*D; /* wvr */
        s+=E*E; /* wj */
        s+=H*3; /* gate */
        s+=E*E; /* wo */
        s+=E; /* rms2 */
        s+=E*M*2+M*E; /* w_gate,w_up,w_down */
    }
    s+=E+E*V; /* rms_f, out_w */
    return s;
}

static void assign_ptrs(Ptrs*p,float*q,Cfg*c){
    int E=c->E,H=c->H,D=c->D,T=c->T,M=c->M,V=c->V;
    p->tok_emb=q;q+=V*E; p->pos_emb=q;q+=T*E;
    for(int b=0;b<c->B;b++){
        p->rms1[b]=q;q+=E;
        p->wq[b]=q;q+=H*E*D; p->wk[b]=q;q+=H*E*D; p->wv[b]=q;q+=H*E*D;
        p->wr[b]=q;q+=H*E*T; p->wvr[b]=q;q+=H*E*D;
        p->wj[b]=q;q+=E*E; p->gate[b]=q;q+=H*3;
        p->wo[b]=q;q+=E*E; p->rms2[b]=q;q+=E;
        p->w_gate[b]=q;q+=E*M; p->w_up[b]=q;q+=E*M; p->w_down[b]=q;q+=M*E;
    }
    p->rms_f=q;q+=E; p->out_w=q;
}

typedef struct { int n_params; float *data,*grad,*cm,*cv; Ptrs w,g; } SingleModel;
typedef struct { SingleModel A,B; float blend_alpha; float *blended; Ptrs bw; Cfg cfg; } DualModel;

static void single_init(SingleModel*m,Cfg*c){
    m->n_params=model_size(c);
    m->data=calloc(m->n_params,sizeof(float));m->grad=calloc(m->n_params,sizeof(float));
    m->cm=calloc(m->n_params,sizeof(float));m->cv=calloc(m->n_params,sizeof(float));
    assign_ptrs(&m->w,m->data,c);assign_ptrs(&m->g,m->grad,c);
    float sc=0.02f*sqrtf(2.0f/c->E);
    for(int i=0;i<m->n_params;i++) m->data[i]=randn()*sc;
    for(int b=0;b<c->B;b++){for(int e=0;e<c->E;e++){m->w.rms1[b][e]=1;m->w.rms2[b][e]=1;}}
    for(int e=0;e<c->E;e++) m->w.rms_f[e]=1;
    for(int b=0;b<c->B;b++) for(int i=0;i<c->H*3;i++) m->w.gate[b][i]=0;
}
static void single_free(SingleModel*m){free(m->data);free(m->grad);free(m->cm);free(m->cv);}

static void dual_init(DualModel*dm,Cfg*c){
    dm->cfg=*c;single_init(&dm->A,c);single_init(&dm->B,c);
    dm->blended=calloc(dm->A.n_params,sizeof(float));assign_ptrs(&dm->bw,dm->blended,c);dm->blend_alpha=0.5f;
    printf("[resonance-janus-bpe] model: %d params (%.2fM) x 2 matrices\n",dm->A.n_params,dm->A.n_params/1e6f);
}
static void dual_blend(DualModel*dm){
    float cd=calendar_dissonance(calendar_days_since_epoch());
    float md=MJ.alive?metajanus_personal_dissonance():0.5f;
    dm->blend_alpha=clamp01(0.5f+0.3f*(cd-0.5f)-0.2f*AML.prophecy_debt+0.1f*md);
    float a=dm->blend_alpha,b=1-a;
    for(int i=0;i<dm->A.n_params;i++) dm->blended[i]=a*dm->A.data[i]+b*dm->B.data[i];
}
static void dual_free(DualModel*dm){single_free(&dm->A);single_free(&dm->B);free(dm->blended);}

/* Chuck Optimizer */
#define CHUCK_B1 0.9f
#define CHUCK_B2 0.999f
#define CHUCK_EPS 1e-8f
#define CHUCK_WINDOW 16
static struct{float hist[CHUCK_WINDOW];float dampen,noise,sigma,loss_ema,macro_ema,best_macro,lr_scale;
int macro_stag,pos,full,stag,global_step,step_t;}Chuck={.dampen=1,.sigma=1,.lr_scale=1};

static void chuck_observe(float loss){
    if(!Chuck.loss_ema)Chuck.loss_ema=loss;else Chuck.loss_ema=0.99f*Chuck.loss_ema+0.01f*loss;
    Chuck.hist[Chuck.pos%CHUCK_WINDOW]=Chuck.loss_ema;Chuck.pos++;
    if(Chuck.pos>=CHUCK_WINDOW)Chuck.full=1;
    if(Chuck.full){int q=CHUCK_WINDOW/4;float r=0,o=0;
    for(int i=0;i<q;i++){r+=Chuck.hist[(Chuck.pos-1-i)%CHUCK_WINDOW];o+=Chuck.hist[(Chuck.pos-CHUCK_WINDOW+i)%CHUCK_WINDOW];}
    r/=q;o/=q;float t=(r-o)/(o+1e-8f);
    if(t>0.01f)Chuck.dampen*=0.95f;if(t<-0.05f)Chuck.dampen*=1.05f;
    if(fabsf(t)<0.001f){Chuck.stag++;if(Chuck.stag>8){Chuck.noise=0.001f;Chuck.stag=0;}}
    else{Chuck.stag=0;Chuck.noise*=0.9f;}
    if(Chuck.dampen<0.3f)Chuck.dampen=0.3f;if(Chuck.dampen>2)Chuck.dampen=2;}
    Chuck.global_step++;
    if(!Chuck.macro_ema)Chuck.macro_ema=loss;else Chuck.macro_ema=0.999f*Chuck.macro_ema+0.001f*loss;
    if(Chuck.global_step%500==0&&Chuck.global_step>CHUCK_WINDOW){
    if(Chuck.macro_ema>Chuck.best_macro*0.999f){Chuck.macro_stag++;
    if(Chuck.macro_stag>=3){Chuck.lr_scale*=0.5f;if(Chuck.lr_scale<0.05f)Chuck.lr_scale=0.05f;Chuck.macro_stag=0;}}
    else{Chuck.best_macro=Chuck.macro_ema;Chuck.macro_stag=0;}}
}

static void chuck_update(float*w,float*g,float*cm,float*cv,int n,float lr){
    Chuck.step_t++;float bc1=1-powf(CHUCK_B1,(float)Chuck.step_t);float bc2=1-powf(CHUCK_B2,(float)Chuck.step_t);
    float eff=lr*Chuck.lr_scale*Chuck.dampen*Chuck.sigma;
    for(int i=0;i<n;i++){cm[i]=CHUCK_B1*cm[i]+(1-CHUCK_B1)*g[i];cv[i]=CHUCK_B2*cv[i]+(1-CHUCK_B2)*g[i]*g[i];
    w[i]-=eff*(cm[i]/bc1)/(sqrtf(cv[i]/bc2)+CHUCK_EPS);if(Chuck.noise>0)w[i]+=Chuck.noise*randn()*0.01f;g[i]=0;}
}

/* ═══════════════════════════════════════════════════════════════════
 * JANUS ATTENTION — self-resonance
 * ═══════════════════════════════════════════════════════════════════ */

static void janus_attention(const float*x,const float*wj,float*echo,float*attn,int T,int E){
    float cd=1+0.5f*calendar_dissonance(calendar_days_since_epoch());
    float dt=1+AML.prophecy_debt;
    float*sc=calloc(T,4);
    for(int t=0;t<T;t++){const float*xt=x+t*E;float*pr=echo+t*E;
    for(int i=0;i<E;i++){float s=0;for(int j=0;j<E;j++)s+=wj[i*E+j]*xt[j];pr[i]=s;}
    float eb[MAX_DIM];for(int i=0;i<E;i++){float s=0;for(int j=0;j<E;j++)s+=wj[j*E+i]*pr[j];eb[i]=s;}
    float nm=0;for(int i=0;i<E;i++)nm+=pr[i]*pr[i];nm=sqrtf(nm)+1e-6f;
    float sv=0;for(int i=0;i<E;i++)sv+=xt[i]*eb[i];sc[t]=(sv/nm)*cd;}
    for(int i=0;i<T;i++){for(int j=0;j<T;j++)attn[i*T+j]=(j>i)?-1e9f:sc[i]*sc[j]/dt;row_softmax(attn+i*T,T);}
    free(sc);
}

/* ═══════════════════════════════════════════════════════════════════
 * FORWARD — 3-way gated (QKV + RRPRAM + Janus)
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct{float*x,*rm1,*cat,*ao,*r1,*rm2,*mg,*mu,*ms,*mo,*r2;
float*q,*k,*v,*at,*ho,*ra,*rv,*ro,*je,*ja,*jo,*frm,*lg;}Acts;

static void acts_alloc(Acts*a,Cfg*c){int E=c->E,T=c->T,M=c->M,V=c->V,D=c->D;
    a->x=calloc(T*E,4);a->rm1=calloc(T*E,4);a->cat=calloc(T*E,4);
    a->ao=calloc(T*E,4);a->r1=calloc(T*E,4);a->rm2=calloc(T*E,4);
    a->mg=calloc(T*M,4);a->mu=calloc(T*M,4);a->ms=calloc(T*M,4);
    a->mo=calloc(T*E,4);a->r2=calloc(T*E,4);
    a->q=calloc(T*D,4);a->k=calloc(T*D,4);a->v=calloc(T*D,4);
    a->at=calloc(T*T,4);a->ho=calloc(T*D,4);
    a->ra=calloc(T*T,4);a->rv=calloc(T*D,4);a->ro=calloc(T*D,4);
    a->je=calloc(T*E,4);a->ja=calloc(T*T,4);a->jo=calloc(T*D,4);
    a->frm=calloc(T*E,4);a->lg=calloc(T*V,4);}
static void acts_free(Acts*a){
    free(a->x);free(a->rm1);free(a->cat);free(a->ao);free(a->r1);free(a->rm2);
    free(a->mg);free(a->mu);free(a->ms);free(a->mo);free(a->r2);
    free(a->q);free(a->k);free(a->v);free(a->at);free(a->ho);
    free(a->ra);free(a->rv);free(a->ro);free(a->je);free(a->ja);free(a->jo);
    free(a->frm);free(a->lg);}

static float fwd(Ptrs*w,Acts*a,int*tok,int*tgt,Cfg*c){
    int E=c->E,T=c->T,H=c->H,D=c->D,M=c->M,V=c->V;
    float scale=1.0f/sqrtf((float)D);
    for(int t=0;t<T;t++) for(int e=0;e<E;e++)
        a->x[t*E+e]=w->tok_emb[tok[t]*E+e]+w->pos_emb[t*E+e];
    float*cur=a->x;
    for(int b=0;b<c->B;b++){
        rmsnorm_fwd(a->rm1,cur,w->rms1[b],T,E);
        memset(a->cat,0,T*E*4);
        for(int h=0;h<H;h++){
            /* QKV */
            matmul(a->q,a->rm1,w->wq[b]+h*E*D,T,E,D);
            matmul(a->k,a->rm1,w->wk[b]+h*E*D,T,E,D);
            matmul(a->v,a->rm1,w->wv[b]+h*E*D,T,E,D);
            for(int i=0;i<T;i++){for(int j=0;j<T;j++){
                if(j>i){a->at[i*T+j]=-1e9f;continue;}
                float s=0;for(int d=0;d<D;d++)s+=a->q[i*D+d]*a->k[j*D+d];
                a->at[i*T+j]=s*scale;}row_softmax(a->at+i*T,T);}
            matmul(a->ho,a->at,a->v,T,T,D);
            /* RRPRAM */
            matmul(a->rv,a->rm1,w->wvr[b]+h*E*D,T,E,D);
            matmul(a->ra,a->rm1,w->wr[b]+h*E*T,T,E,T);
            for(int i=0;i<T*T;i++)a->ra[i]*=scale;
            for(int i=0;i<T;i++){for(int j=i+1;j<T;j++)a->ra[i*T+j]=-1e9f;row_softmax(a->ra+i*T,T);}
            matmul(a->ro,a->ra,a->rv,T,T,D);
            /* Janus self-resonance */
            if(h==0) janus_attention(a->rm1,w->wj[b],a->je,a->ja,T,E);
            for(int t=0;t<T;t++) for(int d=0;d<D;d++){
                float s=0;for(int j=0;j<=t;j++)s+=a->ja[t*T+j]*a->je[j*E+h*D+d];
                a->jo[t*D+d]=s;}
            /* 3-way gate */
            float gl[3]={w->gate[b][h*3],w->gate[b][h*3+1],w->gate[b][h*3+2]};gate_sigmoid(gl,3);
            for(int t=0;t<T;t++) for(int d=0;d<D;d++)
                a->cat[t*E+h*D+d]=gl[0]*a->ho[t*D+d]+gl[1]*a->ro[t*D+d]+gl[2]*a->jo[t*D+d];
        }
        matmul(a->ao,a->cat,w->wo[b],T,E,E);
        for(int i=0;i<T*E;i++) a->r1[i]=cur[i]+a->ao[i];
        rmsnorm_fwd(a->rm2,a->r1,w->rms2[b],T,E);
        matmul(a->mg,a->rm2,w->w_gate[b],T,E,M);
        matmul(a->mu,a->rm2,w->w_up[b],T,E,M);
        for(int i=0;i<T*M;i++) a->ms[i]=siluf(a->mg[i])*a->mu[i];
        matmul(a->mo,a->ms,w->w_down[b],T,M,E);
        for(int i=0;i<T*E;i++) a->r2[i]=a->r1[i]+a->mo[i];
        cur=a->r2;
    }
    rmsnorm_fwd(a->frm,cur,w->rms_f,T,E);
    matmul(a->lg,a->frm,w->out_w,T,E,V);
    /* Dario field overlay */
    for(int t=0;t<T;t++) df_overlay(a->lg+t*V,V);
    if(!tgt) return 0;
    float loss=0;
    for(int t=0;t<T;t++){row_softmax(a->lg+t*V,V);
    float p=a->lg[t*V+tgt[t]];if(p<1e-10f)p=1e-10f;loss-=logf(p);}
    return loss/T;
}

/* ═══════════════════════════════════════════════════════════════════
 * BACKWARD — through 3-way attention + SwiGLU
 * ═══════════════════════════════════════════════════════════════════ */

static void bwd(Ptrs*w,Ptrs*g,Acts*a,int*tok,int*tgt,Cfg*c){
    int E=c->E,T=c->T,H=c->H,D=c->D,M=c->M,V=c->V;
    float scale=1.0f/sqrtf((float)D);
    float*dl=calloc(T*V,4),*df=calloc(T*E,4),*dc=calloc(T*E,4);
    for(int t=0;t<T;t++){for(int v=0;v<V;v++)dl[t*V+v]=a->lg[t*V+v];
    dl[t*V+tgt[t]]-=1;for(int v=0;v<V;v++)dl[t*V+v]/=T;}
    matmul_atb(g->out_w,a->frm,dl,E,T,V);
    matmul_abt(df,dl,w->out_w,T,V,E);
    float*cur=(c->B>0)?a->r2:a->x;
    for(int t=0;t<T;t++){float ss=0;for(int e=0;e<E;e++)ss+=cur[t*E+e]*cur[t*E+e];
    float inv=1/sqrtf(ss/E+1e-5f);for(int e=0;e<E;e++)dc[t*E+e]=df[t*E+e]*w->rms_f[e]*inv;}
    for(int b=c->B-1;b>=0;b--){
        float*dm=calloc(T*E,4);memcpy(dm,dc,T*E*4);
        matmul_atb(g->w_down[b],a->ms,dm,M,T,E);
        float*ds=calloc(T*M,4);matmul_abt(ds,dm,w->w_down[b],T,E,M);
        float*dg2=calloc(T*M,4),*du=calloc(T*M,4);
        for(int i=0;i<T*M;i++){du[i]=ds[i]*siluf(a->mg[i]);dg2[i]=ds[i]*a->mu[i]*siluf_grad(a->mg[i]);}
        matmul_atb(g->w_gate[b],a->rm2,dg2,E,T,M);
        matmul_atb(g->w_up[b],a->rm2,du,E,T,M);
        float*dr=calloc(T*E,4),*tmp=calloc(T*E,4);
        matmul_abt(dr,dg2,w->w_gate[b],T,M,E);
        matmul_abt(tmp,du,w->w_up[b],T,M,E);
        for(int i=0;i<T*E;i++)dr[i]+=tmp[i];
        for(int t=0;t<T;t++){float ss=0;for(int e=0;e<E;e++)ss+=a->r1[t*E+e]*a->r1[t*E+e];
        float inv=1/sqrtf(ss/E+1e-5f);for(int e=0;e<E;e++)dc[t*E+e]+=dr[t*E+e]*w->rms2[b][e]*inv;}
        float*da=calloc(T*E,4);memcpy(da,dc,T*E*4);
        matmul_atb(g->wo[b],a->cat,da,E,T,E);
        float*d_cat=calloc(T*E,4);matmul_abt(d_cat,da,w->wo[b],T,E,E);
        for(int t=0;t<T;t++){
            float inp[MAX_DIM];
            if(b==0){for(int e=0;e<E;e++)inp[e]=a->x[t*E+e];}
            else{for(int e=0;e<E;e++)inp[e]=a->r1[t*E+e]-a->ao[t*E+e];}
            float ss=0;for(int e=0;e<E;e++)ss+=inp[e]*inp[e];
            float inv=1/sqrtf(ss/E+1e-5f);
            for(int e=0;e<E;e++)dc[t*E+e]+=d_cat[t*E+e]*w->rms1[b][e]*inv;}
        free(d_cat);
        if(b==0) for(int t=0;t<T;t++) for(int e=0;e<E;e++){
            g->tok_emb[tok[t]*E+e]+=dc[t*E+e];g->pos_emb[t*E+e]+=dc[t*E+e];}
        free(dm);free(ds);free(dg2);free(du);free(dr);free(tmp);free(da);
    }
    free(dl);free(df);free(dc);
}

/* ═══════════════════════════════════════════════════════════════════
 * GENERATION + REASONING
 * ═══════════════════════════════════════════════════════════════════ */

static int sample_token(float*l,int n,float temp){
    for(int i=0;i<n;i++)l[i]/=(temp+1e-8f);row_softmax(l,n);
    int topk=8,idx[8];float prb[8];
    for(int k=0;k<topk;k++){idx[k]=0;prb[k]=-1e9f;}
    for(int i=0;i<n;i++) if(l[i]>prb[topk-1]){prb[topk-1]=l[i];idx[topk-1]=i;
    for(int j=topk-2;j>=0;j--){if(prb[j+1]>prb[j]){float t=prb[j];prb[j]=prb[j+1];prb[j+1]=t;
    int ti=idx[j];idx[j]=idx[j+1];idx[j+1]=ti;}else break;}}
    float sum=0;for(int k=0;k<topk;k++){if(prb[k]<0)prb[k]=0;sum+=prb[k];}
    if(sum<1e-10f)return idx[0];
    float r=(float)rand()/RAND_MAX*sum,cum=0;
    for(int k=0;k<topk;k++){cum+=prb[k];if(cum>=r)return idx[k];}return idx[0];}

static void generate_sentence(Ptrs*w,Acts*a,int*ctx,int cl,char*out,int maxc,float temp,Cfg*c){
    int pos=0,V=c->V,T=c->T;
    while(pos<maxc-1){
        int tw_len=cl<T?cl:T;
        int*tw=ctx+(cl>T?cl-T:0);
        fwd(w,a,tw,NULL,c);
        float*lg=malloc(V*4);memcpy(lg,a->lg+(tw_len-1)*V,V*4);
        apply_destiny(lg,V);
        int next=sample_token(lg,V,temp);free(lg);
        char decoded[32];int dlen=bpe_decode_token(next,decoded,31);
        for(int i=0;i<dlen&&pos<maxc-1;i++)out[pos++]=decoded[i];
        int stop=0;for(int i=0;i<dlen;i++)if(decoded[i]=='.'||decoded[i]=='!'||decoded[i]=='?'||decoded[i]=='\n')stop=1;
        if(stop)break;
        if(cl<T*4)ctx[cl++]=next;
    }
    out[pos]=0;
}

typedef struct{char sentence[256];int direction,step_idx,wormhole_skip;float debt,diss;}RStep;

static void reason(DualModel*dm,const char*prompt,RStep*steps,int*ns){
    Cfg*c=&dm->cfg;Acts a;acts_alloc(&a,c);dual_blend(dm);
    int ctx[MAX_T*8],cl=0;unsigned char*p=(unsigned char*)prompt;
    int bt[MAX_T*4];int bl=bpe_encode(p,strlen(prompt),bt,MAX_T*4);
    for(int i=0;i<bl&&cl<MAX_T*4;i++)ctx[cl++]=bt[i];
    float cd=calendar_dissonance(calendar_days_since_epoch());
    float debt=AML.prophecy_debt;
    int nb=(int)(NSTEPS*(0.3f+0.4f*debt+0.1f*cd)),nf=NSTEPS-nb;
    if(nb<1)nb=1;if(nf<1)nf=1;if(nb+nf>NSTEPS)nb=NSTEPS-nf;
    float tb=0.7f+0.3f*(0.5f+0.3f*cd+0.2f*debt);int sc=0;
    for(int s=0;s<nf&&sc<NSTEPS;s++){
        int skip=0;if(AML.prophecy_debt<0.2f&&AML.wormhole>0.1f&&(float)rand()/RAND_MAX<AML.wormhole){skip=1;s+=rand()%3;}
        steps[sc]=(RStep){"",1,sc,skip,AML.prophecy_debt,cd};
        generate_sentence(&dm->bw,&a,ctx,cl,steps[sc].sentence,SENT_LEN,tb*(1-0.02f*s),c);
        update_chambers(sc);sc++;}
    cl=0;bl=bpe_encode(p,strlen(prompt),bt,MAX_T*4);for(int i=0;i<bl&&cl<MAX_T*4;i++)ctx[cl++]=bt[i];
    for(int s=0;s<nb&&sc<NSTEPS;s++){
        steps[sc]=(RStep){"",-1,sc,0,AML.prophecy_debt,cd};
        generate_sentence(&dm->bw,&a,ctx,cl,steps[sc].sentence,SENT_LEN,tb*(1+0.05f*s),c);
        update_chambers(sc);sc++;}
    *ns=sc;acts_free(&a);
}

static void display(RStep*steps,int n){
    int nb=0,nf=0;for(int i=0;i<n;i++){if(steps[i].direction==-1)nb++;else nf++;}
    printf("\n\xe2\x95\x94\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x97\n");
    printf("\xe2\x95\x91 RESONANCE-JANUS-BPE  %d steps (\xe2\x86\x91%d \xe2\x86\x93%d)              \xe2\x95\x91\n",n,nb,nf);
    printf("\xe2\x95\xa0\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\xa3\n");
    for(int i=n-1;i>=0;i--) if(steps[i].direction==-1)
        printf("\xe2\x95\x91 \xe2\x86\x91%d%s d=%.2f \xe2\x94\x82 %s\n",steps[i].step_idx,steps[i].wormhole_skip?" WH":"   ",steps[i].debt,steps[i].sentence);
    printf("\xe2\x95\xa0\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90 \xe2\x97\x8f ORIGIN \xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\xa3\n");
    for(int i=0;i<n;i++) if(steps[i].direction==1)
        printf("\xe2\x95\x91 \xe2\x86\x93%d%s d=%.2f \xe2\x94\x82 %s\n",steps[i].step_idx,steps[i].wormhole_skip?" WH":"   ",steps[i].debt,steps[i].sentence);
    printf("\xe2\x95\x9a\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x9d\n");
    printf("  drift=%.4f debt=%.4f bpe=%d\n\n",calendar_cumulative_drift(calendar_days_since_epoch()),AML.prophecy_debt,BPE.vocab_size);
}

/* ═══════════════════════════════════════════════════════════════════
 * TRAINING
 * ═══════════════════════════════════════════════════════════════════ */

#define GRAD_ACCUM 32

static void train_model(DualModel*dm,const char*path,int max_steps,float lr){
    Cfg*c=&dm->cfg; int V=c->V,T=c->T,E=c->E;
    FILE*f=fopen(path,"r");if(!f){fprintf(stderr,"cannot open %s\n",path);return;}
    fseek(f,0,SEEK_END);long fsz=ftell(f);fseek(f,0,SEEK_SET);
    unsigned char*raw=malloc(fsz);fread(raw,1,fsz,f);fclose(f);
    int nm=V-256;if(nm>MAX_MERGES)nm=MAX_MERGES;
    printf("[train] learning %d BPE merges from %ld bytes...\n",nm,fsz);
    bpe_learn_merges(raw,fsz,nm);
    printf("[train] BPE vocab: %d\n",BPE.vocab_size);
    int*bd=malloc(fsz*sizeof(int));int bl=bpe_encode(raw,fsz,bd,fsz);free(raw);
    printf("[train] %ld bytes -> %d tokens (%.2fx)\n",fsz,bl,(float)fsz/bl);
    printf("[train] E=%d H=%d D=%d T=%d B=%d M=%d V=%d\n",c->E,c->H,c->D,c->T,c->B,c->M,c->V);
    printf("[train] params: %d (%.2fM) x 2 matrices\n",dm->A.n_params,dm->A.n_params/1e6f);
    printf("[train] Chuck optimizer, %d steps, lr=%.1e, grad_accum=%d\n",max_steps,lr,GRAD_ACCUM);
    Acts a;acts_alloc(&a,c);int*tok=malloc(T*4),*tgt=malloc(T*4);
    float best=1e9f;clock_t t0=clock();
    for(int step=1;step<=max_steps;step++){
        SingleModel*act=(step%2)?&dm->A:&dm->B;
        memset(act->grad,0,act->n_params*sizeof(float));
        float step_loss=0;
        for(int ga=0;ga<GRAD_ACCUM;ga++){
            int off=rand()%(bl-T-1);
            for(int t=0;t<T;t++){tok[t]=bd[off+t]%V;tgt[t]=bd[off+t+1]%V;}
            float loss=fwd(&act->w,&a,tok,tgt,c);
            bwd(&act->w,&act->g,&a,tok,tgt,c);
            step_loss+=loss;
            for(int t=0;t<T;t++) df_ingest(tok[t]);
        }
        step_loss/=GRAD_ACCUM;
        float inv_ga=1.0f/GRAD_ACCUM;
        for(int i=0;i<act->n_params;i++) act->grad[i]*=inv_ga;
        chuck_observe(step_loss);chuck_update(act->data,act->grad,act->cm,act->cv,act->n_params,lr);
        if(step_loss<best)best=step_loss;
        if(step%100==0||step==1){float el=(float)(clock()-t0)/CLOCKS_PER_SEC;
        printf("  step %5d/%d  loss=%.4f  best=%.4f  %.1f s/s  field: res=%.2f ent=%.2f cooc=%d\n",
            step,max_steps,step_loss,best,step/(el+1e-6f),DF.res,DF.ent,DF.cooc_n);}
    }
    printf("[train] done. best=%.4f\n",best);
    acts_free(&a);free(bd);free(tok);free(tgt);
}

/* ═══════════════════════════════════════════════════════════════════
 * SAVE / LOAD
 * ═══════════════════════════════════════════════════════════════════ */

#define MAGIC 0x524A4250 /* "RJBP" */

static void save_model(DualModel*dm,const char*p){
    FILE*f=fopen(p,"wb");if(!f)return;int magic=MAGIC;Cfg*c=&dm->cfg;
    fwrite(&magic,4,1,f);fwrite(c,sizeof(Cfg),1,f);
    fwrite(&dm->A.n_params,4,1,f);fwrite(&BPE.n_merges,4,1,f);
    fwrite(BPE.merges,sizeof(MergeRule),BPE.n_merges,f);
    fwrite(dm->A.data,4,dm->A.n_params,f);fwrite(dm->B.data,4,dm->B.n_params,f);
    fwrite(&MJ,sizeof(MetaJanus),1,f);fwrite(&AML,sizeof(AMLState),1,f);
    fwrite(dm->A.cm,4,dm->A.n_params,f);fwrite(dm->A.cv,4,dm->A.n_params,f);
    fwrite(dm->B.cm,4,dm->B.n_params,f);fwrite(dm->B.cv,4,dm->B.n_params,f);
    fclose(f);printf("[save] %s (%d params, depth=%d)\n",p,dm->A.n_params,c->B);
}

static int load_model(DualModel*dm,const char*p){
    FILE*f=fopen(p,"rb");if(!f)return-1;int magic,np,nm;Cfg lc;
    if(fread(&magic,4,1,f)<1||magic!=MAGIC){fclose(f);fprintf(stderr,"bad magic in %s\n",p);return-1;}
    if(fread(&lc,sizeof(Cfg),1,f)<1){fclose(f);return-1;}
    if(fread(&np,4,1,f)<1){fclose(f);return-1;}
    if(np!=dm->A.n_params){
        /* reinit with loaded cfg */
        dual_free(dm);dual_init(dm,&lc);
        if(np!=dm->A.n_params){fclose(f);fprintf(stderr,"param mismatch: file=%d model=%d\n",np,dm->A.n_params);return-1;}
    }
    if(fread(&nm,4,1,f)<1){fclose(f);return-1;}
    BPE.n_merges=nm;BPE.vocab_size=256+nm;
    if(fread(BPE.merges,sizeof(MergeRule),nm,f)<(size_t)nm){fclose(f);return-1;}
    if(fread(dm->A.data,4,np,f)<(size_t)np){fclose(f);return-1;}
    if(fread(dm->B.data,4,np,f)<(size_t)np){fclose(f);return-1;}
    fread(&MJ,sizeof(MetaJanus),1,f);fread(&AML,sizeof(AMLState),1,f);
    fread(dm->A.cm,4,np,f);fread(dm->A.cv,4,np,f);fread(dm->B.cm,4,np,f);fread(dm->B.cv,4,np,f);
    fclose(f);printf("[load] %s (%d params, depth=%d, bpe=%d)\n",p,np,lc.B,BPE.vocab_size);return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════ */

int main(int argc,char**argv){
    srand((unsigned)time(NULL));calendar_init();metajanus_init();df_init();
    char*tp=NULL,*lp=NULL,*sp=NULL,*pr=NULL;int ms=15000,depth=12;float lr=3e-4f;int bpe_v=2048;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--train")&&i+1<argc)tp=argv[++i];
        else if(!strcmp(argv[i],"--load")&&i+1<argc)lp=argv[++i];
        else if(!strcmp(argv[i],"--save")&&i+1<argc)sp=argv[++i];
        else if(!strcmp(argv[i],"--depth")&&i+1<argc)depth=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--steps")&&i+1<argc)ms=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--lr")&&i+1<argc)lr=atof(argv[++i]);
        else if(!strcmp(argv[i],"--bpe-vocab")&&i+1<argc)bpe_v=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--generate")&&i+1<argc)pr=argv[++i];
    }
    if(depth<1||depth>MAX_BLK||depth*32>MAX_DIM){fprintf(stderr,"bad depth %d (1-%d)\n",depth,MAX_DIM/32);return 1;}
    BPE_VOCAB=bpe_v;
    Cfg c=cfg_from_depth(depth,bpe_v);
    printf("\n  resonance-janus-bpe.c\n");
    printf("  theta = epsilon + gamma + alpha*delta\n");
    printf("  3-way attention (QKV + RRPRAM + Janus) + Dario field + BPE\n");
    printf("  depth=%d E=%d H=%d D=%d T=%d B=%d M=%d V=%d\n\n",depth,c.E,c.H,c.D,c.T,c.B,c.M,c.V);
    DualModel dm;dual_init(&dm,&c);
    if(lp) load_model(&dm,lp);
    if(tp){train_model(&dm,tp,ms,lr);if(!sp)sp="resonance_janus_bpe.bin";save_model(&dm,sp);}
    if(sp&&!tp) save_model(&dm,sp);
    if(pr){RStep steps[NSTEPS];int n=0;reason(&dm,pr,steps,&n);display(steps,n);}
    if(!tp&&!pr){printf("[interactive]\n");char buf[1024];
    while(1){printf("\nresonance-janus-bpe> ");if(!fgets(buf,sizeof(buf),stdin))break;
    buf[strcspn(buf,"\n")]=0;if(!strcmp(buf,"quit")||!strcmp(buf,"exit"))break;
    if(!strlen(buf))continue;RStep steps[NSTEPS];int n=0;reason(&dm,buf,steps,&n);display(steps,n);}}
    dual_free(&dm);return 0;
}
