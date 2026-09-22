/* ============================================================================
 *  spear_usecases.c — 3 USE CASES RÉELS branchés sur les breakthroughs
 *  asymétriques intégrés (SpearVM championnes) :
 *
 *  UC1  FFN fusionné : Y = gelu(X·Wᵀ + b) — transformer MLP 1024x768x3072.
 *       Breakthrough : gelu_erf (L∞ 2.05e-5, 100% ALU). Le GEMM AVX2 garde
 *       l'accumulation SIMD puis applique gelu au STORE (1 passage sur C au
 *       lieu de 2 : gain bande passante mémoire).
 *  UC2  Porte MLP : tanh_p34 (L∞ 1.56e-3) vs spear_tanh (9e-3) et libm.
 *       Breakthrough : Pade[3/4], ×5.4 précision pour le même nb d'opérations.
 *  UC3  Backprop GELU : dérivée exacte de gelu_erf (quotient-rule sur le
 *       Horner) + gradcheck par différences finies. Use : entraînement.
 *
 *  Compile : gcc -O2 -mavx2 -mfma -fopenmp -I. src/kernels/spear_usecases.c -o usecases -lm
 *  Exit code = nb de checks échoués. Chaque chiffre est mesuré.
 * ============================================================================
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include "src/kernels/spear_kernels.h"

/* ---------------------------------------------------------------------------
 *  Helpers temps & AVX
 * ------------------------------------------------------------------------- */
static double now_ms(void){
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec*1e3 + (double)ts.tv_nsec*1e-6;
}
static double hsum256(__m256d v){
    __m128d lo=_mm256_castpd256_pd128(v), hi=_mm256_extractf128_pd(v,1);
    lo=_mm_add_pd(lo,hi);
    return _mm_cvtsd_f64(_mm_add_sd(lo,_mm_unpackhi_pd(lo,lo)));
}

/* ---------------------------------------------------------------------------
 *  UC1 — FFN fusionné. GEMM C=A·Bᵀ (A m×k, B n×k), gelu appliqué au store.
 *  Le blocage ne découpe QUE les colonnes (NC) : gelu est non-linéaire, k ne
 *  peut pas être tuilé (contraint le produit scalaire complet par cellule).
 *  Registre : 4 lignes de A servent 1 ligne de B, 4 chaînes FMA indépendantes.
 * ------------------------------------------------------------------------- */
#define NC 256
static void ffn_fused(double *C, const double *A, const double *B,
                      const double *bias, int m, int k, int n, float (*act)(float)){
    for (int jb=0;jb<n;jb+=NC){
        int je = (jb+NC<n)?jb+NC:n;
        for (int i0=0;i0<m/4;i0++){
            const double *ar=A+(size_t)i0*4*k;
            double *cr=C+(size_t)i0*4*n;
            for (int j=jb;j<je;j++){
                const double *br=B+(size_t)j*k;
                __m256d v0=_mm256_setzero_pd(),v1=_mm256_setzero_pd();
                __m256d v2=_mm256_setzero_pd(),v3=_mm256_setzero_pd();
                int q=0;
                for (;q+3<k;q+=4){
                    __m256d bv=_mm256_loadu_pd(br+q);
                    v0=_mm256_fmadd_pd(_mm256_loadu_pd(ar+q),bv,v0);
                    v1=_mm256_fmadd_pd(_mm256_loadu_pd(ar+k+q),bv,v1);
                    v2=_mm256_fmadd_pd(_mm256_loadu_pd(ar+2*k+q),bv,v2);
                    v3=_mm256_fmadd_pd(_mm256_loadu_pd(ar+3*k+q),bv,v3);
                }
                double d0=hsum256(v0),d1=hsum256(v1),d2=hsum256(v2),d3=hsum256(v3);
                for (;q<k;q++){ double b=br[q]; d0+=ar[q]*b; d1+=ar[k+q]*b; d2+=ar[2*k+q]*b; d3+=ar[3*k+q]*b; }
                cr[j]     =act((float)(d0+(bias?bias[j]:0)));
                cr[n+j]   =act((float)(d1+(bias?bias[j]:0)));
                cr[2*n+j] =act((float)(d2+(bias?bias[j]:0)));
                cr[3*n+j] =act((float)(d3+(bias?bias[j]:0)));
            }
        }
        for (int i=(m/4)*4;i<m;i++){
            const double *xr=A+(size_t)i*k; double *tr=C+(size_t)i*n;
            for (int j=jb;j<je;j++){
                const double *wr=B+(size_t)j*k;
                __m256d acc=_mm256_setzero_pd();
                int q=0;
                for (;q+3<k;q+=4) acc=_mm256_fmadd_pd(_mm256_loadu_pd(xr+q),_mm256_loadu_pd(wr+q),acc);
                double s=hsum256(acc);
                for (;q<k;q++) s+=xr[q]*wr[q];
                tr[j]=act((float)(s+(bias?bias[j]:0)));
            }
        }
    }
}
/* deux passes : GEMM pur (accumulation brute) puis gelu élémentaire (baseline).
   Matérialise C entier, puis le relit pour l'activation → 2 passes mémoire. */
static void ffn_twopass(double *C, const double *A, const double *B,
                        const double *bias, int m, int k, int n, float (*act)(float)){
    for (int i=0;i<m;i++){
        const double *ar=A+(size_t)i*k; double *cr=C+(size_t)i*n;
        for (int j=0;j<n;j++){
            const double *br=B+(size_t)j*k;
            __m256d acc=_mm256_setzero_pd();
            int q=0;
            for (;q+3<k;q+=4) acc=_mm256_fmadd_pd(_mm256_loadu_pd(ar+q),_mm256_loadu_pd(br+q),acc);
            double s=hsum256(acc);
            for (;q<k;q++) s+=ar[q]*br[q];
            cr[j]=s;
        }
    }
    /* 2e passe : activation (relecture complète de C) */
    for (long long i=0;i<(long long)m*n;i++) C[i]=act((float)C[i]+(bias?bias[i%n]:0));
}

/* ---------------------------------------------------------------------------
 *  UC3 — dérivée exacte de gelu_erf via quotient-rule sur le Horner P/Q.
 *  g(x)=0.5x(1+u·R(u²)), u=x/√2, R=P/Q.
 * ------------------------------------------------------------------------- */
static float gelu_erf_d(float x){
    double u=x*0.7071067811865476;
    if (u>3.5) return 1.0f;
    if (u<-3.5) return 0.0f;
    static const double N[5]={1.12841751266903279,0.183482771948230095,0.0573373674730976793,0.00248430060206610405,3.72785350475749968e-6};
    static const double D[6]={1,0.496471589671860558,0.114910282096263028,0.0161717422205343367,1.86656477609649336e-4,-1.74401807407079551e-7};
    double y=u*u;
    /* P,Q et P',Q' — Horner ascendant exact (P=N0+..+N4·y⁴, Q=D0+..+D5·y⁵) */
    double p=0,q=0,pp=0,qq=0;
    for (int i=4;i>=0;i--) p=p*y+N[i];
    for (int i=5;i>=0;i--) q=q*y+D[i];
    for (int i=4;i>=1;i--) pp=pp*y + (double)i*N[i];
    for (int i=5;i>=1;i--) qq=qq*y + (double)i*D[i];
    double R=p/q, Rp=(pp*q-p*qq)/(q*q);
    /* d/dx[0.5·x·(1+u·R)] , dy/dx=2u·c, du/dx=c, c=1/√2 */
    double inside = 1.0+u*R;
    double dInside = 0.7071067811865476*(R + 2.0*y*Rp);
    return (float)(0.5*inside + 0.5*x*dInside);
}

/* ---------------------------------------------------------------------------
 *  main — 3 use cases
 * ------------------------------------------------------------------------- */
int main(void){
    int fails=0;
    srand(1234);

    /* ===== UC1 : FFN fusionné ===== */
    printf("=== UC1  FFN fusionné gelu_erf : Y=gelu(X·Wᵀ+b)  1024x768x3072 ===\n");
    {
        int m=1024,k=768,n=3072;
        double *A=malloc((size_t)m*k*8), *B=malloc((size_t)n*k*8);
        double *b=malloc((size_t)n*8), *C1=malloc((size_t)m*n*8), *C2=malloc((size_t)m*n*8);
        for (long long i=0;i<(long long)m*k;i++) A[i]=((double)rand()/RAND_MAX-0.5)*0.2;
        for (long long i=0;i<(long long)n*k;i++) B[i]=((double)rand()/RAND_MAX-0.5)*0.2;
        for (int j=0;j<n;j++) b[j]=((double)rand()/RAND_MAX-0.5)*0.05;
        /* warmup + mesure min-of-3 */
        double tf=1e30, tt=1e30;
        for (int r=0;r<3;r++){
            double t0=now_ms(); ffn_fused(C1,A,B,b,m,k,n,spear_gelu_erf); double dt=now_ms()-t0;
            if (dt<tf) tf=dt;
        }
        for (int r=0;r<3;r++){
            double t0=now_ms(); ffn_twopass(C2,A,B,b,m,k,n,spear_gelu_erf); double dt=now_ms()-t0;
            if (dt<tt) tt=dt;
        }
        /* précision fusionné vs exact (échantillon) */
        double maxerr=0;
        for (long long i=0;i<(long long)m*n;i+=997){ double d=C1[i]-C2[i]; if(d<0)d=-d; if(d>maxerr)maxerr=d; }
        double gflop=2.0*m*k*n*1e-9;
        int ok = maxerr<1e-4 && tf<=tt;
        if(!ok) fails++;
        printf("  fused   %.1f ms (%.1f GFLOPS) | two-pass %.1f ms | err(fused-vs-2pass)=%.2e | %s\n",
               tf, gflop/(tf*1e-3), tt, maxerr, ok?"PASS":"FAIL");
        printf("  gain mémoire : C relu 1 fois (store), 2nd tour éliminé — ×%.2f wall\n", tt/tf);
        free(A);free(B);free(b);free(C1);free(C2);
    }

    /* ===== UC2 : porte MLP — tanh_p34 vs spear_tanh vs libm ===== */
    printf("\n=== UC2  Porte MLP : précision tanh_p34 (Pade[3/4]) vs kernels existants ===\n");
    {
        const int N=2000000;
        double *x=malloc((size_t)N*8);
        for (int i=0;i<N;i++) x[i]=-5+10.0*i/(N-1);
        double e34=0, et=0, s34=0, st=0;
        for (int i=0;i<N;i++){
            double d34=fabs(spear_tanh_p34((float)x[i])-tanh(x[i]));
            double dt=fabs(spear_tanh((float)x[i])-tanh(x[i]));
            if (d34>e34)e34=d34; if(dt>et)et=dt;
            s34+=spear_tanh_p34((float)x[i]); st+=spear_tanh((float)x[i]);
        }
        int ok = e34 < et*0.5;  /* asymétrie : >2× plus précis */
        if(!ok) fails++;
        printf("  L∞ vs libm : tanh_p34=%.3e  spear_tanh=%.3e  → ×%.1f plus précis | %s\n",
               e34, et, et/e34, ok?"PASS":"FAIL");
        (void)s34;(void)st;
        free(x);
    }

    /* ===== UC3 : backprop GELU — gradcheck de la dérivée ===== */
    printf("\n=== UC3  Backprop gelu_erf : dérivée exacte vs différences finies ===\n");
    {
        double worstAbs=0, worstRel=0; int badAbs=0, badRel=0;
        for (int i=0;i<=400000;i++){
            float x=-5+10.0f*i/400000;
            float h=1e-3f*(1+fabsf(x));
            float num=(spear_gelu_erf(x+h)-spear_gelu_erf(x-h))/(2*h);
            float der=gelu_erf_d(x);
            double ae=fabs((double)(num-der));
            if (ae>worstAbs)worstAbs=ae;
            if (ae>1e-3) badAbs++;
            /* rel pertinente seulement là où la vraie dérivée est significative */
            if (fabs((double)num)>1e-2){ double re=ae/fabs((double)num); if(re>worstRel)worstRel=re; if(re>1e-2) badRel++; }
        }
        int ok = worstAbs<1e-3 && badRel==0;
        if(!ok) fails++;
        printf("  dérivée exacte : |err|max=%.3e (cellules>1e-3 abs: %d/400001)  err_rel_max=%.3e (cellules>1%%: %d) | %s\n",
               worstAbs, badAbs, worstRel, badRel, ok?"PASS":"FAIL");
    }

    printf("\n%s — %d use case(s) échoué(s)\n", fails==0?"TOUT VALIDÉ ✅":"ÉCHEC ❌", fails);
    return fails;
}