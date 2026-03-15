/*
 * janus-bpe.c — Janus Pure BPE Post-Transformer
 *
 * Pure BPE version: BPE in, BPE out. No char-level pressure.
 * Same hybrid attention (QKV + RRPRAM + Janus self-resonance).
 * Calendar Drift, AML physics, Chuck optimizer, dual weight matrices.
 * 12 bi-directional associative reasoning steps.
 *
 *   cc janus-bpe.c -O2 -lm -o janus-bpe
 *   ./janus-bpe --train data.txt --steps 5000
 *   ./janus-bpe --generate "To be or not"
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
 * CONFIGURATION
 * ═══════════════════════════════════════════════════════════════════ */

#define BPE_VOCAB   512
#define MAX_T       256
#define DIM         288
#define HEADS       6
#define HEAD_DIM    (DIM/HEADS)
#define BLOCKS      6
#define MLP_DIM     768
#define MAX_BLK     16
#define NSTEPS      12
#define MAX_MERGES  256

/* ═══════════════════════════════════════════════════════════════════
 * CALENDAR DRIFT — exact port from ariannamethod.c
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

/* Kuramoto */
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

/* BPE decode: expand token back to bytes */
static int bpe_decode_token(int tok, char *buf, int max) {
    if (tok < 256) { if (max > 0) buf[0] = (char)tok; return 1; }
    /* Find merge rule that created this token */
    for (int m = BPE.n_merges - 1; m >= 0; m--) {
        if (BPE.merges[m].result == tok) {
            int n1 = bpe_decode_token(BPE.merges[m].a, buf, max);
            int n2 = bpe_decode_token(BPE.merges[m].b, buf + n1, max - n1);
            return n1 + n2;
        }
    }
    if (max > 0) buf[0] = '?';
    return 1;
}

static void bpe_learn_merges(const unsigned char *data, int len, int nm) {
    int *tok=malloc(len*sizeof(int)); int n=len;
    for(int i=0;i<n;i++) tok[i]=data[i];
    for(int m=0;m<nm&&m<MAX_MERGES;m++){
        int ba=-1,bb=-1,bc=0;
        for(int i=0;i+1<n;i++){
            int a=tok[i],b=tok[i+1]; int c=0;
            for(int j=i;j+1<n;j++) if(tok[j]==a&&tok[j+1]==b) c++;
            if(c>bc){bc=c;ba=a;bb=b;}
        }
        if(bc<2) break;
        int nid=256+m; BPE.merges[m]=(MergeRule){ba,bb,nid};
        BPE.n_merges=m+1; BPE.vocab_size=256+m+1;
        int j=0;
        for(int i=0;i<n;i++){
            if(i+1<n&&tok[i]==ba&&tok[i+1]==bb){tok[j++]=nid;i++;}
            else tok[j++]=tok[i];
        }
        n=j;
    }
    free(tok);
}

/* ═══════════════════════════════════════════════════════════════════
 * MATH
 * ═══════════════════════════════════════════════════════════════════ */

static void matmul(float*C,const float*A,const float*B,int m,int k,int n){
    for(int i=0;i<m;i++) for(int j=0;j<n;j++){
        float s=0; for(int p=0;p<k;p++) s+=A[i*k+p]*B[p*n+j]; C[i*n+j]=s;
    }
}
static void matmul_atb(float*C,const float*A,const float*B,int m,int k,int n){
    for(int i=0;i<m;i++) for(int j=0;j<n;j++){
        float s=0; for(int p=0;p<k;p++) s+=A[p*m+i]*B[p*n+j]; C[i*n+j]=s;
    }
}
static void matmul_abt(float*C,const float*A,const float*B,int m,int k,int n){
    for(int i=0;i<m;i++) for(int j=0;j<n;j++){
        float s=0; for(int p=0;p<k;p++) s+=A[i*k+p]*B[j*k+p]; C[i*n+j]=s;
    }
}
static void row_softmax(float*x,int n){
    float mx=x[0]; for(int i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-mx);s+=x[i];}
    if(s>0) for(int i=0;i<n;i++) x[i]/=s;
}
static float siluf(float x){return x>-20?x/(1+expf(-x)):0;}
static float siluf_grad(float x){if(x<-20)return 0;float s=1/(1+expf(-x));return s*(1+x*(1-s));}
static void rmsnorm_fwd(float*o,const float*x,const float*g,int T,int E){
    for(int t=0;t<T;t++){float ss=0;for(int e=0;e<E;e++) ss+=x[t*E+e]*x[t*E+e];
    float inv=1/sqrtf(ss/E+1e-5f);for(int e=0;e<E;e++) o[t*E+e]=g[e]*x[t*E+e]*inv;}
}
static float randn(void){
    float u1=(rand()+1.0f)/(RAND_MAX+2.0f),u2=(rand()+1.0f)/(RAND_MAX+2.0f);
    return sqrtf(-2*logf(u1))*cosf(6.2831853f*u2);
}

/* ═══════════════════════════════════════════════════════════════════
 * MODEL — pure BPE
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

typedef struct {
    int n_params; float *data,*grad,*cm,*cv; Ptrs w,g;
} Model;

static int model_size(void){
    int s=BPE_VOCAB*DIM+MAX_T*DIM;
    for(int b=0;b<BLOCKS;b++){
        s+=DIM; s+=HEADS*DIM*HEAD_DIM*3; s+=HEADS*DIM*MAX_T;
        s+=HEADS*DIM*HEAD_DIM; s+=DIM*DIM; s+=HEADS*3;
        s+=DIM*DIM; s+=DIM; s+=DIM*MLP_DIM*2+MLP_DIM*DIM;
    }
    s+=DIM+DIM*BPE_VOCAB; return s;
}

static void assign_ptrs(Ptrs*p,float*q){
    p->tok_emb=q;q+=BPE_VOCAB*DIM; p->pos_emb=q;q+=MAX_T*DIM;
    for(int b=0;b<BLOCKS;b++){
        p->rms1[b]=q;q+=DIM;
        p->wq[b]=q;q+=HEADS*DIM*HEAD_DIM; p->wk[b]=q;q+=HEADS*DIM*HEAD_DIM;
        p->wv[b]=q;q+=HEADS*DIM*HEAD_DIM; p->wr[b]=q;q+=HEADS*DIM*MAX_T;
        p->wvr[b]=q;q+=HEADS*DIM*HEAD_DIM; p->wj[b]=q;q+=DIM*DIM;
        p->gate[b]=q;q+=HEADS*3; p->wo[b]=q;q+=DIM*DIM;
        p->rms2[b]=q;q+=DIM;
        p->w_gate[b]=q;q+=DIM*MLP_DIM; p->w_up[b]=q;q+=DIM*MLP_DIM;
        p->w_down[b]=q;q+=MLP_DIM*DIM;
    }
    p->rms_f=q;q+=DIM; p->out_w=q;
}

static void model_init(Model*m){
    m->n_params=model_size();
    m->data=calloc(m->n_params,sizeof(float)); m->grad=calloc(m->n_params,sizeof(float));
    m->cm=calloc(m->n_params,sizeof(float)); m->cv=calloc(m->n_params,sizeof(float));
    assign_ptrs(&m->w,m->data); assign_ptrs(&m->g,m->grad);
    float sc=0.02f*sqrtf(2.0f/DIM);
    for(int i=0;i<m->n_params;i++) m->data[i]=randn()*sc;
    for(int b=0;b<BLOCKS;b++){for(int e=0;e<DIM;e++){m->w.rms1[b][e]=1;m->w.rms2[b][e]=1;}}
    for(int e=0;e<DIM;e++) m->w.rms_f[e]=1;
    for(int b=0;b<BLOCKS;b++) for(int i=0;i<HEADS*3;i++) m->w.gate[b][i]=0;
    printf("[janus-bpe] model: %d params (%.2fMB)\n",m->n_params,m->n_params*4.0f/1e6f);
}
static void model_free(Model*m){free(m->data);free(m->grad);free(m->cm);free(m->cv);}

typedef struct{Model A,B;float blend_alpha;float*blended;Ptrs bw;}DualModel;
static void dual_init(DualModel*dm){model_init(&dm->A);model_init(&dm->B);
    dm->blended=calloc(dm->A.n_params,sizeof(float));assign_ptrs(&dm->bw,dm->blended);dm->blend_alpha=0.5f;}
static void dual_blend(DualModel*dm){
    float cd=calendar_dissonance(calendar_days_since_epoch());
    float md=MJ.alive?metajanus_personal_dissonance():0.5f;
    dm->blend_alpha=clamp01(0.5f+0.3f*(cd-0.5f)-0.2f*AML.prophecy_debt+0.1f*md);
    float a=dm->blend_alpha,b=1-a;
    for(int i=0;i<dm->A.n_params;i++) dm->blended[i]=a*dm->A.data[i]+b*dm->B.data[i];
}
static void dual_free(DualModel*dm){model_free(&dm->A);model_free(&dm->B);free(dm->blended);}

/* Chuck */
#define CHUCK_B1 0.9f
#define CHUCK_B2 0.999f
#define CHUCK_EPS 1e-8f
#define CHUCK_WINDOW 16
static struct{float hist[CHUCK_WINDOW];float dampen,noise,sigma,loss_ema,macro_ema,best_macro,lr_scale;
int macro_stag,pos,full,stag,global_step,step_t;}Chuck={.dampen=1,.sigma=1,.lr_scale=1};

static void chuck_observe(float loss){
    if(!Chuck.loss_ema) Chuck.loss_ema=loss; else Chuck.loss_ema=0.99f*Chuck.loss_ema+0.01f*loss;
    Chuck.hist[Chuck.pos%CHUCK_WINDOW]=Chuck.loss_ema; Chuck.pos++;
    if(Chuck.pos>=CHUCK_WINDOW) Chuck.full=1;
    if(Chuck.full){int q=CHUCK_WINDOW/4;float r=0,o=0;
    for(int i=0;i<q;i++){r+=Chuck.hist[(Chuck.pos-1-i)%CHUCK_WINDOW];o+=Chuck.hist[(Chuck.pos-CHUCK_WINDOW+i)%CHUCK_WINDOW];}
    r/=q;o/=q;float t=(r-o)/(o+1e-8f);
    if(t>0.01f) Chuck.dampen*=0.95f; if(t<-0.05f) Chuck.dampen*=1.05f;
    if(fabsf(t)<0.001f){Chuck.stag++;if(Chuck.stag>8){Chuck.noise=0.001f;Chuck.stag=0;}}
    else{Chuck.stag=0;Chuck.noise*=0.9f;}
    if(Chuck.dampen<0.3f)Chuck.dampen=0.3f;if(Chuck.dampen>2)Chuck.dampen=2;}
    Chuck.global_step++;
    if(!Chuck.macro_ema) Chuck.macro_ema=loss; else Chuck.macro_ema=0.999f*Chuck.macro_ema+0.001f*loss;
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
 * FORWARD + BACKWARD
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct{float*x,*rm1,*cat,*ao,*r1,*rm2,*mg,*mu,*ms,*mo,*r2;
float*q,*k,*v,*at,*ho,*ra,*rv,*ro,*je,*ja,*jo,*frm,*lg;}Acts;

static void acts_alloc(Acts*a){
    a->x=calloc(MAX_T*DIM,4);a->rm1=calloc(MAX_T*DIM,4);a->cat=calloc(MAX_T*DIM,4);
    a->ao=calloc(MAX_T*DIM,4);a->r1=calloc(MAX_T*DIM,4);a->rm2=calloc(MAX_T*DIM,4);
    a->mg=calloc(MAX_T*MLP_DIM,4);a->mu=calloc(MAX_T*MLP_DIM,4);a->ms=calloc(MAX_T*MLP_DIM,4);
    a->mo=calloc(MAX_T*DIM,4);a->r2=calloc(MAX_T*DIM,4);
    a->q=calloc(MAX_T*HEAD_DIM,4);a->k=calloc(MAX_T*HEAD_DIM,4);a->v=calloc(MAX_T*HEAD_DIM,4);
    a->at=calloc(MAX_T*MAX_T,4);a->ho=calloc(MAX_T*HEAD_DIM,4);
    a->ra=calloc(MAX_T*MAX_T,4);a->rv=calloc(MAX_T*HEAD_DIM,4);a->ro=calloc(MAX_T*HEAD_DIM,4);
    a->je=calloc(MAX_T*DIM,4);a->ja=calloc(MAX_T*MAX_T,4);a->jo=calloc(MAX_T*HEAD_DIM,4);
    a->frm=calloc(MAX_T*DIM,4);a->lg=calloc(MAX_T*BPE_VOCAB,4);
}
static void acts_free(Acts*a){
    free(a->x);free(a->rm1);free(a->cat);free(a->ao);free(a->r1);free(a->rm2);
    free(a->mg);free(a->mu);free(a->ms);free(a->mo);free(a->r2);
    free(a->q);free(a->k);free(a->v);free(a->at);free(a->ho);
    free(a->ra);free(a->rv);free(a->ro);free(a->je);free(a->ja);free(a->jo);
    free(a->frm);free(a->lg);
}

static void janus_attention(const float*x,const float*wj,float*echo,float*attn,int T){
    float cd=1+0.5f*calendar_dissonance(calendar_days_since_epoch());
    float dt=1+AML.prophecy_debt;
    float*sc=calloc(T,4);
    for(int t=0;t<T;t++){const float*xt=x+t*DIM;float*pr=echo+t*DIM;
    for(int i=0;i<DIM;i++){float s=0;for(int j=0;j<DIM;j++)s+=wj[i*DIM+j]*xt[j];pr[i]=s;}
    float eb[DIM];for(int i=0;i<DIM;i++){float s=0;for(int j=0;j<DIM;j++)s+=wj[j*DIM+i]*pr[j];eb[i]=s;}
    float nm=0;for(int i=0;i<DIM;i++)nm+=pr[i]*pr[i];nm=sqrtf(nm)+1e-6f;
    float sv=0;for(int i=0;i<DIM;i++)sv+=xt[i]*eb[i];sc[t]=(sv/nm)*cd;}
    for(int i=0;i<T;i++){for(int j=0;j<T;j++)attn[i*T+j]=(j>i)?-1e9f:sc[i]*sc[j]/dt;row_softmax(attn+i*T,T);}
    free(sc);
}

static float forward(Ptrs*w,Acts*a,int*tok,int*tgt,int T){
    for(int t=0;t<T;t++) for(int e=0;e<DIM;e++)
        a->x[t*DIM+e]=w->tok_emb[tok[t]*DIM+e]+w->pos_emb[t*DIM+e];
    float*cur=a->x;
    for(int b=0;b<BLOCKS;b++){
        rmsnorm_fwd(a->rm1,cur,w->rms1[b],T,DIM);
        memset(a->cat,0,T*DIM*4);
        for(int h=0;h<HEADS;h++){
            matmul(a->q,a->rm1,w->wq[b]+h*DIM*HEAD_DIM,T,DIM,HEAD_DIM);
            matmul(a->k,a->rm1,w->wk[b]+h*DIM*HEAD_DIM,T,DIM,HEAD_DIM);
            matmul(a->v,a->rm1,w->wv[b]+h*DIM*HEAD_DIM,T,DIM,HEAD_DIM);
            float sc=1/sqrtf((float)HEAD_DIM);
            for(int i=0;i<T;i++){for(int j=0;j<T;j++){
                if(j>i){a->at[i*T+j]=-1e9f;continue;}
                float s=0;for(int d=0;d<HEAD_DIM;d++)s+=a->q[i*HEAD_DIM+d]*a->k[j*HEAD_DIM+d];
                a->at[i*T+j]=s*sc;}row_softmax(a->at+i*T,T);}
            matmul(a->ho,a->at,a->v,T,T,HEAD_DIM);
            matmul(a->rv,a->rm1,w->wvr[b]+h*DIM*HEAD_DIM,T,DIM,HEAD_DIM);
            matmul(a->ra,a->rm1,w->wr[b]+h*DIM*MAX_T,T,DIM,T);
            for(int i=0;i<T;i++){for(int j=i+1;j<T;j++)a->ra[i*T+j]=-1e9f;row_softmax(a->ra+i*T,T);}
            matmul(a->ro,a->ra,a->rv,T,T,HEAD_DIM);
            if(h==0) janus_attention(a->rm1,w->wj[b],a->je,a->ja,T);
            for(int t=0;t<T;t++) for(int d=0;d<HEAD_DIM;d++){
                float s=0;for(int j=0;j<=t;j++)s+=a->ja[t*T+j]*a->je[j*DIM+h*HEAD_DIM+d];
                a->jo[t*HEAD_DIM+d]=s;}
            float gl[3]={w->gate[b][h*3],w->gate[b][h*3+1],w->gate[b][h*3+2]};row_softmax(gl,3);
            for(int t=0;t<T;t++) for(int d=0;d<HEAD_DIM;d++)
                a->cat[t*DIM+h*HEAD_DIM+d]=gl[0]*a->ho[t*HEAD_DIM+d]+gl[1]*a->ro[t*HEAD_DIM+d]+gl[2]*a->jo[t*HEAD_DIM+d];
        }
        matmul(a->ao,a->cat,w->wo[b],T,DIM,DIM);
        for(int i=0;i<T*DIM;i++) a->r1[i]=cur[i]+a->ao[i];
        rmsnorm_fwd(a->rm2,a->r1,w->rms2[b],T,DIM);
        matmul(a->mg,a->rm2,w->w_gate[b],T,DIM,MLP_DIM);
        matmul(a->mu,a->rm2,w->w_up[b],T,DIM,MLP_DIM);
        for(int i=0;i<T*MLP_DIM;i++) a->ms[i]=siluf(a->mg[i])*a->mu[i];
        matmul(a->mo,a->ms,w->w_down[b],T,MLP_DIM,DIM);
        for(int i=0;i<T*DIM;i++) a->r2[i]=a->r1[i]+a->mo[i];
        cur=a->r2;
    }
    rmsnorm_fwd(a->frm,cur,w->rms_f,T,DIM);
    matmul(a->lg,a->frm,w->out_w,T,DIM,BPE_VOCAB);
    if(!tgt) return 0;
    float loss=0;
    for(int t=0;t<T;t++){row_softmax(a->lg+t*BPE_VOCAB,BPE_VOCAB);
    float p=a->lg[t*BPE_VOCAB+tgt[t]];if(p<1e-10f)p=1e-10f;loss-=logf(p);}
    return loss/T;
}

static void backward(Ptrs*w,Ptrs*g,Acts*a,int*tok,int*tgt,int T){
    float*dl=calloc(T*BPE_VOCAB,4),*df=calloc(T*DIM,4),*dc=calloc(T*DIM,4);
    for(int t=0;t<T;t++){for(int v=0;v<BPE_VOCAB;v++)dl[t*BPE_VOCAB+v]=a->lg[t*BPE_VOCAB+v];
    dl[t*BPE_VOCAB+tgt[t]]-=1;for(int v=0;v<BPE_VOCAB;v++)dl[t*BPE_VOCAB+v]/=T;}
    matmul_atb(g->out_w,a->frm,dl,DIM,T,BPE_VOCAB);
    matmul_abt(df,dl,w->out_w,T,BPE_VOCAB,DIM);
    float*cur=(BLOCKS>0)?a->r2:a->x;
    for(int t=0;t<T;t++){float ss=0;for(int e=0;e<DIM;e++)ss+=cur[t*DIM+e]*cur[t*DIM+e];
    float inv=1/sqrtf(ss/DIM+1e-5f);for(int e=0;e<DIM;e++)dc[t*DIM+e]=df[t*DIM+e]*w->rms_f[e]*inv;}
    for(int b=BLOCKS-1;b>=0;b--){
        float*dm=calloc(T*DIM,4);memcpy(dm,dc,T*DIM*4);
        matmul_atb(g->w_down[b],a->ms,dm,MLP_DIM,T,DIM);
        float*ds=calloc(T*MLP_DIM,4);matmul_abt(ds,dm,w->w_down[b],T,DIM,MLP_DIM);
        float*dg2=calloc(T*MLP_DIM,4),*du=calloc(T*MLP_DIM,4);
        for(int i=0;i<T*MLP_DIM;i++){du[i]=ds[i]*siluf(a->mg[i]);dg2[i]=ds[i]*a->mu[i]*siluf_grad(a->mg[i]);}
        matmul_atb(g->w_gate[b],a->rm2,dg2,DIM,T,MLP_DIM);
        matmul_atb(g->w_up[b],a->rm2,du,DIM,T,MLP_DIM);
        float*dr=calloc(T*DIM,4),*tmp=calloc(T*DIM,4);
        matmul_abt(dr,dg2,w->w_gate[b],T,MLP_DIM,DIM);
        matmul_abt(tmp,du,w->w_up[b],T,MLP_DIM,DIM);
        for(int i=0;i<T*DIM;i++)dr[i]+=tmp[i];
        for(int t=0;t<T;t++){float ss=0;for(int e=0;e<DIM;e++)ss+=a->r1[t*DIM+e]*a->r1[t*DIM+e];
        float inv=1/sqrtf(ss/DIM+1e-5f);for(int e=0;e<DIM;e++)dc[t*DIM+e]+=dr[t*DIM+e]*w->rms2[b][e]*inv;}
        float*da=calloc(T*DIM,4);memcpy(da,dc,T*DIM*4);
        matmul_atb(g->wo[b],a->cat,da,DIM,T,DIM);
        if(b==0) for(int t=0;t<T;t++) for(int e=0;e<DIM;e++){
            g->tok_emb[tok[t]*DIM+e]+=dc[t*DIM+e]; g->pos_emb[t*DIM+e]+=dc[t*DIM+e];}
        free(dm);free(ds);free(dg2);free(du);free(dr);free(tmp);free(da);
    }
    free(dl);free(df);free(dc);
}

/* ═══════════════════════════════════════════════════════════════════
 * GENERATION — pure BPE in/out
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
    for(int k=0;k<topk;k++){cum+=prb[k];if(cum>=r)return idx[k];}return idx[0];
}

static void generate_sentence(Ptrs*w,Acts*a,int*ctx,int cl,char*out,int maxc,float temp){
    int pos=0;
    while(pos<maxc-1){
        int T=cl<MAX_T?cl:MAX_T;
        int*tw=ctx+(cl>MAX_T?cl-MAX_T:0);
        forward(w,a,tw,NULL,T);
        float lg[BPE_VOCAB];memcpy(lg,a->lg+(T-1)*BPE_VOCAB,BPE_VOCAB*4);
        apply_destiny(lg,BPE_VOCAB);
        int next=sample_token(lg,BPE_VOCAB,temp);
        /* Decode BPE token to chars */
        char decoded[16]; int dlen = bpe_decode_token(next, decoded, 15);
        for (int i = 0; i < dlen && pos < maxc-1; i++) out[pos++] = decoded[i];
        int stop=0;
        for(int i=0;i<dlen;i++) if(decoded[i]=='.'||decoded[i]=='!'||decoded[i]=='?'||decoded[i]=='\n') stop=1;
        if(stop) break;
        if(cl<MAX_T*4) ctx[cl++]=next;
    }
    out[pos]=0;
}

/* Reasoning */
typedef struct{char sentence[256];int direction,step_idx,wormhole_skip;float debt,diss;}RStep;

static void reason(DualModel*dm,const char*prompt,RStep*steps,int*ns){
    Acts a;acts_alloc(&a);dual_blend(dm);
    int ctx[MAX_T*8],cl=0;unsigned char*p=(unsigned char*)prompt;
    int bt[MAX_T*4];int bl=bpe_encode(p,strlen(prompt),bt,MAX_T*4);
    for(int i=0;i<bl&&cl<MAX_T*4;i++) ctx[cl++]=bt[i];
    float cd=calendar_dissonance(calendar_days_since_epoch());
    float debt=AML.prophecy_debt;
    int nb=(int)(NSTEPS*(0.3f+0.4f*debt+0.1f*cd)),nf=NSTEPS-nb;
    if(nb<1)nb=1;if(nf<1)nf=1;if(nb+nf>NSTEPS)nb=NSTEPS-nf;
    float tb=0.7f+0.3f*(0.5f+0.3f*cd+0.2f*debt);int sc=0;
    for(int s=0;s<nf&&sc<NSTEPS;s++){
        int skip=0;if(AML.prophecy_debt<0.2f&&AML.wormhole>0.1f&&(float)rand()/RAND_MAX<AML.wormhole){skip=1;s+=rand()%3;}
        steps[sc]=(RStep){"",1,sc,skip,AML.prophecy_debt,cd};
        generate_sentence(&dm->bw,&a,ctx,cl,steps[sc].sentence,40,tb*(1-0.02f*s));
        update_chambers(sc);sc++;
    }
    cl=0;bl=bpe_encode(p,strlen(prompt),bt,MAX_T*4);for(int i=0;i<bl&&cl<MAX_T*4;i++)ctx[cl++]=bt[i];
    for(int s=0;s<nb&&sc<NSTEPS;s++){
        steps[sc]=(RStep){"",-1,sc,0,AML.prophecy_debt,cd};
        generate_sentence(&dm->bw,&a,ctx,cl,steps[sc].sentence,40,tb*(1+0.05f*s));
        update_chambers(sc);sc++;
    }
    *ns=sc;acts_free(&a);
}

static void display(RStep*steps,int n){
    int nb=0,nf=0;for(int i=0;i<n;i++){if(steps[i].direction==-1)nb++;else nf++;}
    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║ JANUS-BPE  %d steps (↑%d ↓%d)                ║\n",n,nb,nf);
    printf("╠══════════════════════════════════════════════╣\n");
    for(int i=n-1;i>=0;i--) if(steps[i].direction==-1)
        printf("║ ↑%d%s d=%.2f │ %s\n",steps[i].step_idx,steps[i].wormhole_skip?" WH":"   ",steps[i].debt,steps[i].sentence);
    printf("╠═════════════ ● ORIGIN ═══════════════════════╣\n");
    for(int i=0;i<n;i++) if(steps[i].direction==1)
        printf("║ ↓%d%s d=%.2f │ %s\n",steps[i].step_idx,steps[i].wormhole_skip?" WH":"   ",steps[i].debt,steps[i].sentence);
    printf("╚══════════════════════════════════════════════╝\n");
    printf("  drift=%.4f debt=%.4f bpe=%d\n\n",calendar_cumulative_drift(calendar_days_since_epoch()),AML.prophecy_debt,BPE.vocab_size);
}

/* Training */
static void train_bpe(DualModel*dm,const char*path,int max_steps,float lr){
    FILE*f=fopen(path,"r");if(!f){fprintf(stderr,"cannot open %s\n",path);return;}
    fseek(f,0,SEEK_END);long fsz=ftell(f);fseek(f,0,SEEK_SET);
    unsigned char*raw=malloc(fsz);fread(raw,1,fsz,f);fclose(f);
    int nm=BPE_VOCAB-256;if(nm>MAX_MERGES)nm=MAX_MERGES;
    printf("[janus-bpe] learning %d BPE merges...\n",nm);
    bpe_learn_merges(raw,fsz,nm);
    printf("[janus-bpe] BPE vocab: %d\n",BPE.vocab_size);
    int*bd=malloc(fsz*sizeof(int));int bl=bpe_encode(raw,fsz,bd,fsz);free(raw);
    Acts a;acts_alloc(&a);int T=MAX_T;int*tok=malloc(T*4),*tgt=malloc(T*4);
    printf("[janus-bpe] %ld bytes → %d tokens\n",fsz,bl);
    printf("[janus-bpe] params: %d (%.2fMB) × 2 matrices\n",dm->A.n_params,dm->A.n_params*4.0f/1e6f);
    printf("[janus-bpe] Chuck optimizer, %d steps, lr=%.1e\n",max_steps,lr);
    float best=1e9f;clock_t t0=clock();
    for(int step=1;step<=max_steps;step++){
        Model*act=(step%2)?&dm->A:&dm->B;
        int off=rand()%(bl-T-1);
        for(int t=0;t<T;t++){tok[t]=bd[off+t]%BPE_VOCAB;tgt[t]=bd[off+t+1]%BPE_VOCAB;}
        float loss=forward(&act->w,&a,tok,tgt,T);
        backward(&act->w,&act->g,&a,tok,tgt,T);
        chuck_observe(loss);chuck_update(act->data,act->grad,act->cm,act->cv,act->n_params,lr);
        if(loss<best)best=loss;
        if(step%100==0||step==1){float el=(float)(clock()-t0)/CLOCKS_PER_SEC;
        printf("  step %5d/%d  loss=%.4f  best=%.4f  %.1f s/s\n",step,max_steps,loss,best,step/(el+1e-6f));}
    }
    printf("[janus-bpe] done. best=%.4f\n",best);
    acts_free(&a);free(bd);free(tok);free(tgt);
}

/* Save/Load */
static void save_model(DualModel*dm,const char*p){
    FILE*f=fopen(p,"wb");if(!f)return;int magic=0x4A425045;
    fwrite(&magic,4,1,f);fwrite(&dm->A.n_params,4,1,f);fwrite(&BPE.n_merges,4,1,f);
    fwrite(BPE.merges,sizeof(MergeRule),BPE.n_merges,f);
    fwrite(dm->A.data,4,dm->A.n_params,f);fwrite(dm->B.data,4,dm->B.n_params,f);
    fwrite(&MJ,sizeof(MetaJanus),1,f);fwrite(&AML,sizeof(AMLState),1,f);
    fwrite(dm->A.cm,4,dm->A.n_params,f);fwrite(dm->A.cv,4,dm->A.n_params,f);
    fwrite(dm->B.cm,4,dm->B.n_params,f);fwrite(dm->B.cv,4,dm->B.n_params,f);
    fclose(f);printf("[janus-bpe] saved to %s\n",p);
}

static int load_model(DualModel*dm,const char*p){
    FILE*f=fopen(p,"rb");if(!f)return-1;int magic,np,nm;
    if(fread(&magic,4,1,f)<1||magic!=0x4A425045){fclose(f);return-1;}
    if(fread(&np,4,1,f)<1||np!=dm->A.n_params){fclose(f);return-1;}
    if(fread(&nm,4,1,f)<1){fclose(f);return-1;}
    BPE.n_merges=nm;BPE.vocab_size=256+nm;
    if(fread(BPE.merges,sizeof(MergeRule),nm,f)<(size_t)nm){fclose(f);return-1;}
    if(fread(dm->A.data,4,np,f)<(size_t)np){fclose(f);return-1;}
    if(fread(dm->B.data,4,np,f)<(size_t)np){fclose(f);return-1;}
    fread(&MJ,sizeof(MetaJanus),1,f);fread(&AML,sizeof(AMLState),1,f);
    fread(dm->A.cm,4,np,f);fread(dm->A.cv,4,np,f);fread(dm->B.cm,4,np,f);fread(dm->B.cv,4,np,f);
    fclose(f);printf("[janus-bpe] loaded from %s\n",p);return 0;
}

int main(int argc,char**argv){
    srand((unsigned)time(NULL));calendar_init();metajanus_init();
    char*tp=NULL,*lp=NULL,*sp=NULL,*pr=NULL;int ms=5000;float lr=3e-4f;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--train")&&i+1<argc)tp=argv[++i];
        else if(!strcmp(argv[i],"--load")&&i+1<argc)lp=argv[++i];
        else if(!strcmp(argv[i],"--save")&&i+1<argc)sp=argv[++i];
        else if(!strcmp(argv[i],"--steps")&&i+1<argc)ms=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--lr")&&i+1<argc)lr=atof(argv[++i]);
        else if(!strcmp(argv[i],"--generate")&&i+1<argc)pr=argv[++i];
    }
    DualModel dm;dual_init(&dm);
    if(lp) load_model(&dm,lp);
    if(tp) train_bpe(&dm,tp,ms,lr);
    if(sp) save_model(&dm,sp);
    if(pr){RStep steps[NSTEPS];int n=0;reason(&dm,pr,steps,&n);display(steps,n);}
    if(!tp&&!pr){printf("\n[janus-bpe] interactive mode\n");char buf[1024];
    while(1){printf("\njanus-bpe> ");if(!fgets(buf,sizeof(buf),stdin))break;
    buf[strcspn(buf,"\n")]=0;if(!strcmp(buf,"quit")||!strcmp(buf,"exit"))break;
    if(!strlen(buf))continue;RStep steps[NSTEPS];int n=0;reason(&dm,buf,steps,&n);display(steps,n);}}
    dual_free(&dm);return 0;
}
