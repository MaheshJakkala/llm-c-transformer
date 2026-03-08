#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/wait.h>

#define H 256
#define FH 512
#define V 4096
#define NC 9
#define BENCH 200
#define WARM 20

static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
static unsigned long long g=42;
static float rnd(void){g=g*6364136223846793005ULL+1442695040888963407ULL;return (float)(g>>33)/(float)(1ULL<<31)*0.04f-0.02f;}
static float*fw(int n){float*w=malloc(n*4);for(int i=0;i<n;i++)w[i]=rnd();return w;}
static void mm(const float*A,const float*B,float*C,int M,int K,int N){
    memset(C,0,M*N*4);
    for(int i=0;i<M;i++)for(int k=0;k<K;k++){float a=A[i*K+k];for(int j=0;j<N;j++)C[i*N+j]+=a*B[k*N+j];}
}
static void ln(float*x,int R,int C){for(int r=0;r<R;r++){float m=0,v=0;
    for(int c=0;c<C;c++)m+=x[r*C+c];m/=C;for(int c=0;c<C;c++){float d=x[r*C+c]-m;v+=d*d;}v/=C;
    float s=1/sqrtf(v+1e-5f);for(int c=0;c<C;c++)x[r*C+c]=(x[r*C+c]-m)*s;}
}
static void gelu(float*x,int n){for(int i=0;i<n;i++)x[i]=0.5f*x[i]*(1+tanhf(0.7978845608f*(x[i]+0.044715f*x[i]*x[i]*x[i])));}
static void sa(int fd,const void*b,size_t n){const char*p=b;while(n>0){ssize_t r=write(fd,p,n);if(r<=0){perror("w");exit(1);}p+=r;n-=r;}}
static void ra(int fd,void*b,size_t n){char*p=b;while(n>0){ssize_t r=read(fd,p,n);if(r<=0){perror("r");exit(1);}p+=r;n-=r;}}

int main(void){
    int seq=16;
    int sv[2]; if(socketpair(AF_UNIX,SOCK_STREAM,0,sv)){perror("sp");return 1;}
    pid_t pid=fork(); if(pid<0){perror("fork");return 1;}
    
    if(pid==0){
        /* NODE 1: FFN + Head */
        close(sv[0]);
        g=42; free(fw(V*H));free(fw(H*H));free(fw(H*H));free(fw(H*H));free(fw(H*H));
        float*W1=fw(H*FH),*b1=calloc(FH,4),*W2=fw(FH*H),*b2=calloc(H,4),*Wc=fw(H*NC),*bc=calloc(NC,4);
        float*x=malloc(seq*H*4),*hf=malloc(seq*FH*4),*out=malloc(seq*H*4),*lg=malloc(seq*NC*4);
        for(int it=0;it<WARM+BENCH;it++){
            ra(sv[1],x,seq*H*4);
            mm(x,W1,hf,seq,H,FH);for(int i=0;i<seq;i++)for(int j=0;j<FH;j++)hf[i*FH+j]+=b1[j];
            gelu(hf,seq*FH);
            mm(hf,W2,out,seq,FH,H);for(int i=0;i<seq;i++)for(int j=0;j<H;j++)out[i*H+j]+=b2[j]+x[i*H+j];
            ln(out,seq,H);
            mm(out,Wc,lg,seq,H,NC);for(int i=0;i<seq;i++)for(int j=0;j<NC;j++)lg[i*NC+j]+=bc[j];
            sa(sv[1],lg,seq*NC*4);
        }
        free(W1);free(b1);free(W2);free(b2);free(Wc);free(bc);free(x);free(hf);free(out);free(lg);
        close(sv[1]); exit(0);
    }
    
    /* NODE 0: Embedding + Attention */
    close(sv[1]);
    g=42;
    float*emb=fw(V*H),*Wq=fw(H*H),*Wk=fw(H*H),*Wv=fw(H*H),*Wo=fw(H*H);
    float*x=malloc(seq*H*4),*Q=malloc(seq*H*4),*K_=malloc(seq*H*4),*V_=malloc(seq*H*4);
    float*sc=malloc(seq*seq*4),*ao=malloc(seq*H*4),*out=malloc(seq*H*4),*lg=malloc(seq*NC*4);
    int ids[64]; for(int i=0;i<seq;i++)ids[i]=(i*37+13)%V;
    double tc=0,tk=0;
    for(int it=0;it<WARM+BENCH;it++){
        double t0=ms();
        for(int i=0;i<seq;i++) memcpy(x+i*H,emb+ids[i]*H,H*4);
        mm(x,Wq,Q,seq,H,H); mm(x,Wk,K_,seq,H,H); mm(x,Wv,V_,seq,H,H);
        float sf=1/sqrtf((float)H);
        for(int i=0;i<seq;i++)for(int j=0;j<seq;j++){float s=0;for(int k=0;k<H;k++)s+=Q[i*H+k]*K_[j*H+k];sc[i*seq+j]=s*sf;}
        for(int i=0;i<seq;i++){float mx=-1e30f;for(int j=0;j<seq;j++)if(sc[i*seq+j]>mx)mx=sc[i*seq+j];
            float sm=0;for(int j=0;j<seq;j++){sc[i*seq+j]=expf(sc[i*seq+j]-mx);sm+=sc[i*seq+j];}
            for(int j=0;j<seq;j++)sc[i*seq+j]/=sm;}
        for(int i=0;i<seq;i++)for(int j=0;j<H;j++){float s=0;for(int k=0;k<seq;k++)s+=sc[i*seq+k]*V_[k*H+j];ao[i*H+j]=s;}
        mm(ao,Wo,out,seq,H,H);
        for(int i=0;i<seq*H;i++)out[i]+=x[i]; ln(out,seq,H);
        double t1=ms();
        sa(sv[0],out,seq*H*4);
        ra(sv[0],lg,seq*NC*4);
        double t2=ms();
        if(it>=WARM){tc+=t1-t0;tk+=t2-t1;}
    }
    int st; waitpid(pid,&st,0);
    
    printf("=============================================================\n");
    printf(" Distributed Inference — 2-Process Pipeline Parallelism\n");
    printf("=============================================================\n\n");
    printf("  Protocol      : %d warm-up + %d timed, averaged\n",WARM,BENCH);
    printf("  seq=%d  hidden=%d  tensor per transfer=%zu bytes (%.1f KB)\n",
           seq,H,(size_t)seq*H*4,(float)seq*H*4/1024);
    printf("\n  NODE 0 compute   : %.3f ms  (embedding + attention)\n",tc/BENCH);
    printf("  Comm roundtrip   : %.3f ms  (send hidden + recv logits)\n",tk/BENCH);
    printf("  Total distributed: %.3f ms\n",(tc+tk)/BENCH);
    printf("  Comm overhead    : %.1f%%\n",100.0*tk/(tc+tk));
    double bw=((double)seq*H*4*8)/((tk/BENCH/2)/1e3)/1e9;
    printf("  Socket bandwidth : %.2f Gbps\n",bw);
    printf("  1 Gbps LAN cost  : +%.2f ms/call (vs %.3f ms socket)\n",
           (seq*H*4*8.0/1e9)*1e3,tk/BENCH/2);

    FILE*f=fopen("results/metrics/distributed_inference.csv","w");
    if(f){fprintf(f,"metric,value\n");
        fprintf(f,"seq_len,%d\n",seq);
        fprintf(f,"node0_compute_ms,%.4f\n",tc/BENCH);
        fprintf(f,"comm_roundtrip_ms,%.4f\n",tk/BENCH);
        fprintf(f,"total_ms,%.4f\n",(tc+tk)/BENCH);
        fprintf(f,"comm_overhead_pct,%.2f\n",100.0*tk/(tc+tk));
        fprintf(f,"tensor_bytes,%d\n",seq*H*4);
        fprintf(f,"socket_bw_gbps,%.4f\n",bw);
        fclose(f); printf("\nSaved: results/metrics/distributed_inference.csv\n");}
    
    free(emb);free(Wq);free(Wk);free(Wv);free(Wo);free(x);free(Q);free(K_);free(V_);free(sc);free(ao);free(out);free(lg);
    close(sv[0]); return 0;
}
