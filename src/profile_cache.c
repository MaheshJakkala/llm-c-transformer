#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <immintrin.h>

#define REPS 500
#define WARM  50

static double ms(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e3+t.tv_nsec/1e6;
}

/* ── BEFORE: naive row x col — column access strides through memory ─────── */
void matmul_naive(const float*A, const float*B, float*C, int N){
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            float s=0;
            for(int k=0;k<N;k++) s+=A[i*N+k]*B[k*N+j]; /* B[k][j]: stride=N */
            C[i*N+j]=s;
        }
}

/* ── AFTER v1: transpose B first, then row x row ───────────────────────── */
void matmul_transposed(const float*A, const float*B, float*C, int N){
    float *BT = malloc(N*N*4);
    /* transpose: BT[j][k] = B[k][j] */
    for(int k=0;k<N;k++) for(int j=0;j<N;j++) BT[j*N+k]=B[k*N+j];
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            float s=0;
            for(int k=0;k<N;k++) s+=A[i*N+k]*BT[j*N+k]; /* sequential! */
            C[i*N+j]=s;
        }
    free(BT);
}

/* ── AFTER v2: cache-tiled — blocks fit in L1d (48KB) ──────────────────── */
/* Tile size: 32×32 floats = 4KB — fits comfortably in 48KB L1d           */
#define TILE 32
void matmul_tiled(const float*A, const float*B, float*C, int N){
    memset(C,0,N*N*4);
    for(int i=0;i<N;i+=TILE)
    for(int j=0;j<N;j+=TILE)
    for(int k=0;k<N;k+=TILE){
        int imax=i+TILE<N?i+TILE:N;
        int jmax=j+TILE<N?j+TILE:N;
        int kmax=k+TILE<N?k+TILE:N;
        for(int ii=i;ii<imax;ii++)
        for(int kk=k;kk<kmax;kk++){
            float a=A[ii*N+kk];
            for(int jj=j;jj<jmax;jj++) C[ii*N+jj]+=a*B[kk*N+jj];
        }
    }
}

/* ── AFTER v3: tiled + AVX2 FMA ─────────────────────────────────────────── */
void matmul_tiled_avx2(const float*A, const float*B, float*C, int N){
    memset(C,0,N*N*4);
    for(int i=0;i<N;i+=TILE)
    for(int k=0;k<N;k+=TILE)
    for(int j=0;j<N;j+=TILE){
        int imax=i+TILE<N?i+TILE:N;
        int kmax=k+TILE<N?k+TILE:N;
        int jmax=j+TILE<N?j+TILE:N;
        for(int ii=i;ii<imax;ii++)
        for(int kk=k;kk<kmax;kk++){
            __m256 a_v=_mm256_set1_ps(A[ii*N+kk]);
            for(int jj=j;jj<jmax;jj+=8){
                __m256 c_v=_mm256_loadu_ps(C+ii*N+jj);
                __m256 b_v=_mm256_loadu_ps(B+kk*N+jj);
                c_v=_mm256_fmadd_ps(a_v,b_v,c_v);
                _mm256_storeu_ps(C+ii*N+jj,c_v);
            }
        }
    }
}

static double bench(void(*fn)(const float*,const float*,float*,int),
                    const float*A,const float*B,float*C,int N){
    for(int i=0;i<WARM;i++) fn(A,B,C,N);
    double t0=ms();
    for(int i=0;i<REPS;i++) fn(A,B,C,N);
    return (ms()-t0)/REPS;
}

int main(void){
    /* Test at 3 matrix sizes to show effect at different cache levels */
    int sizes[]={64,128,256,0};
    
    printf("=======================================================================\n");
    printf(" Cache-Line Optimization: Before vs After — Measured with CLOCK_MONOTONIC\n");
    printf(" CPU L1d=48KB  L2=2MB  (getconf LEVEL1_DCACHE_SIZE / LEVEL2_CACHE_SIZE)\n");
    printf("=======================================================================\n\n");
    printf(" Tile size: %dx%d = %d KB  (fits in 48KB L1d with A/B/C tiles)\n\n",
           TILE,TILE,TILE*TILE*4*3/1024);
    printf("%-6s | %-10s | %-12s | %-12s | %-12s | %-10s | %-10s\n",
           "N","Matrix KB","Naive (ms)","Transposed","Tiled","Tiled+AVX2","vs Naive");
    printf("--------|------------|--------------|--------------|--------------|------------|----------\n");

    FILE *fcsv=fopen("/tmp/cache_profile.csv","w");
    fprintf(fcsv,"n,matrix_kb,naive_ms,transposed_ms,tiled_ms,tiled_avx2_ms,"
                 "transposed_speedup,tiled_speedup,avx2_speedup\n");

    for(int si=0;sizes[si];si++){
        int N=sizes[si];
        float *A=malloc(N*N*4),*B=malloc(N*N*4),*C=malloc(N*N*4);
        for(int i=0;i<N*N;i++){A[i]=(float)i*0.001f; B[i]=(float)(i+1)*0.001f;}

        double t_naive=bench(matmul_naive,A,B,C,N);
        double t_trans=bench(matmul_transposed,A,B,C,N);
        double t_tiled=bench(matmul_tiled,A,B,C,N);
        double t_avx2 =bench(matmul_tiled_avx2,A,B,C,N);

        int kb=N*N*4/1024;
        printf("%-6d | %-10d | %-12.4f | %-12.4f | %-12.4f | %-12.4f | %.2fx\n",
               N,kb,t_naive,t_trans,t_tiled,t_avx2,t_naive/t_avx2);
        fprintf(fcsv,"%d,%d,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f\n",
                N,kb,t_naive,t_trans,t_tiled,t_avx2,
                t_naive/t_trans,t_naive/t_tiled,t_naive/t_avx2);
        free(A);free(B);free(C);
    }
    fclose(fcsv);
    printf("\nSaved: /tmp/cache_profile.csv\n");
    return 0;
}
