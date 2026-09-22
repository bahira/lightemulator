/* ============================================================================
 *  spear_train_quality.c — BENCHMARK COURT : la précision du noyau d'activation
 *  améliore-t-elle la QUALITÉ d'entraînement ?
 *
 *  Protocole : même MLP (2→16→2), même seed, même SGD, même dataset (spirale
 *  2 classes), seule variable = le noyau d'activation + sa dérivée exacte.
 *    - gelu_erf      (L∞ 2.05e-5, breakthrough)
 *    - gelu_quintic  (L∞ 1.74e-2)
 *    - exact_gelu    (référence : erf mathématique) = plafond théorique
 *  Conclusion : si gelu_erf rejoint exact_gelu ET bat gelu_quintic, la
 *  précision du forward ⇒ qualité. Chaque chiffre est mesuré (L3).
 *
 *  Compile : gcc -O2 -I. src/kernels/spear_train_quality.c -o tq -lm
 * ============================================================================
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "src/kernels/spear_kernels.h"

static double now_ms(void){ struct timespec ts; timespec_get(&ts, TIME_UTC); return (double)ts.tv_sec*1e3+(double)ts.tv_nsec*1e-6; }

/* ----------------------- générateurs (déterministes) ---------------------- */
static unsigned long R=123456789UL;
static double frand(void){ R=R*6364136223846793005UL+1442695040888963407UL; return (double)((R>>33)&0xffffffff)/4294967296.0; }
static double gauss(void){ double u=0; for(int i=0;i<6;i++)u+=frand(); return (u-3.0)/1.0; }

/* ----------------------------- dataset spirale ---------------------------- */
#define NSAM 800
static double X[NSAM][2]; static int Y[NSAM];
static void make_spiral(void){
    for (int i=0;i<NSAM;i++){
        double t=((double)(i%400)/400.0)*4.0*M_PI;
        double r=(i%400)*0.05/400.0+0.05;
        double n0=gauss()*0.05, n1=gauss()*0.05;
        X[i][0]=r*cos(t)+n0; X[i][1]=r*sin(t)+n1;
        Y[i]=(i<400)?0:1;
    }
}

/* ------------------------------- MLP --------------------------------
   z1=W1·x, a=act(z1), z2=W2·a, p=softmax(z2), CE. Entraînement SGD. */
#define H 16
#define OUT 2
static double W1[H][2], b1[H], W2[OUT][H], b2[OUT];

typedef float (*ActFn)(float);
typedef float (*ActDn)(float);

static void init_net(void){
    for (int i=0;i<H;i++){ W1[i][0]=gauss()*0.5; W1[i][1]=gauss()*0.5; b1[i]=gauss()*0.1;
        for (int o=0;o<OUT;o++) W2[o][i]=gauss()*0.5; }
    for (int o=0;o<OUT;o++) b2[o]=0;
}

static double forward(int n, ActFn act, double a[H]){
    for (int i=0;i<H;i++) a[i]=act((float)(W1[i][0]*X[n][0]+W1[i][1]*X[n][1]+b1[i]));
    double z[OUT];
    for (int o=0;o<OUT;o++){ double s=b2[o]; for (int i=0;i<H;i++) s+=W2[o][i]*a[i]; z[o]=s; }
    double m=z[0]>z[1]?z[0]:z[1]; double e0=exp(z[0]-m),e1=exp(z[1]-m); double S=e0+e1;
    return (Y[n]==0)? -log(e0/S) : -log(e1/S);
}

static void backward(int n, ActFn act, ActDn actd, double a[H], double gW1[H][2], double gb1[H], double gW2[OUT][H], double gb2[OUT]){
    /* p */
    double z[OUT];
    for (int o=0;o<OUT;o++){ double s=b2[o]; for (int i=0;i<H;i++) s+=W2[o][i]*a[i]; z[o]=s; }
    double m=z[0]>z[1]?z[0]:z[1]; double e0=exp(z[0]-m),e1=exp(z[1]-m); double S=e0+e1;
    double p0=e0/S,p1=e1/S; double t0=p0-(Y[n]==0?1:0), t1=p1-(Y[n]==1?1:0);
    for (int i=0;i<H;i++){
        gW2[0][i]+=t0*a[i]; gW2[1][i]+=t1*a[i];
        double dz=(t0*W2[0][i]+t1*W2[1][i])*actd((float)(W1[i][0]*X[n][0]+W1[i][1]*X[n][1]+b1[i]));
        gW1[i][0]+=dz*X[n][0]; gW1[i][1]+=dz*X[n][1]; gb1[i]+=dz;
    }
    gb2[0]+=t0; gb2[1]+=t1;
}

/* ------------------------- dérivées exactes ------------------------- */
/* gelu_erf : dérivée exacte (quotient-rule sur Horner) */
static float q_erf_d(float x){
    double u=x*0.7071067811865476;
    if (u>3.5) return 1.0f; if (u<-3.5) return 0.0f;
    static const double N[5]={1.12841751266903279,0.183482771948230095,0.0573373674730976793,0.00248430060206610405,3.72785350475749968e-6};
    static const double D[6]={1,0.496471589671860558,0.114910282096263028,0.0161717422205343367,1.86656477609649336e-4,-1.74401807407079551e-7};
    double y=u*u,p=0,q=0,pp=0,qq=0;
    for (int i=4;i>=0;i--)p=p*y+N[i]; for (int i=5;i>=0;i--)q=q*y+D[i];
    for (int i=4;i>=1;i--)pp=pp*y+(double)i*N[i]; for (int i=5;i>=1;i--)qq=qq*y+(double)i*D[i];
    double R=p/q, Rp=(pp*q-p*qq)/(q*q);
    return (float)(0.5*(1.0+u*R)+0.5*x*0.7071067811865476*(R+2.0*y*Rp));
}
/* gelu_quintic : g=x·t³(6t²-15t+10)-off, t=A·x+0.5 ; g'=s + 30·A·x·t²(t-1)² */
#define QA 0.200055340257
#define QOFF 0.01104961
static float q_quintic_d(float x){
    double t=QA*x+0.5;
    if (t<0) return 0; if (t>1) return 1;
    double s=t*t*t*(6*t*t-15*t+10);
    return (float)(s + 30.0*QA*x*t*t*(t-1)*(t-1));
}
/* exact gelu : 0.5x(1+erf(x/√2)), dérivée 0.5(1+erf(x/√2)) + x/√(2π)·e^(-x²/2) */
static float exact_gelu_ref(float x){ return (float)(0.5*x*(1+erf(x*0.7071067811865476))); }
static float exact_gelu_d(float x){ double u=x*0.7071067811865476; return (float)(0.5*(1+erf(u)) + x*0.3989422804014327*exp(-0.5*x*x)); }

/* ----------------------------- train & score ------------------------- */
static double train(ActFn act, ActDn actd, int steps, double lr){
    init_net();
    double a[H]; double gW1[H][2],gb1[H],gW2[OUT][H],gb2[OUT];
    for (int s=0;s<steps;s++){
        int n=s%NSAM;
        for (int i=0;i<H;i++){ gW1[i][0]=gW1[i][1]=gb1[i]=0; gW2[0][i]=gW2[1][i]=0; }
        gb2[0]=gb2[1]=0;
        backward(n,act,actd,a,gW1,gb1,gW2,gb2);
        for (int i=0;i<H;i++){ W1[i][0]-=lr*gW1[i][0]; W1[i][1]-=lr*gW1[i][1]; b1[i]-=lr*gb1[i]; W2[0][i]-=lr*gW2[0][i]; W2[1][i]-=lr*gW2[1][i]; }
        b2[0]-=lr*gb2[0]; b2[1]-=lr*gb2[1];
    }
    /* score final sur TOUT le dataset */
    double loss=0; int acc=0;
    for (int n=0;n<NSAM;n++){
        loss+=forward(n,act,a);
        double z0=b2[0],z1=b2[1];
        for (int i=0;i<H;i++){ z0+=W2[0][i]*a[i]; z1+=W2[1][i]*a[i]; }
        if (Y[n]==((z0>z1)?0:1)) acc++;
    }
    return loss/NSAM; /* (acc printé séparément) */
}

int main(void){
    make_spiral();
    /* forward L∞ de chaque noyau vs exact (grounding) */
    printf("=== Précision forward des noyaux (vs exact gelu, grille [-4,4]) ===\n");
    { double e=0; for(int i=0;i<200000;i++){ float x=-4+8.0f*i/199999; double d=spear_gelu_erf(x)-exact_gelu_ref(x); if(d<0)d=-d; if(d>e)e=d; } printf("  gelu_erf     L∞ = %.3e\n",e); }
    { double e=0; for(int i=0;i<200000;i++){ float x=-4+8.0f*i/199999; double d=spear_gelu_quintic(x)-exact_gelu_ref(x); if(d<0)d=-d; if(d>e)e=d; } printf("  gelu_quintic L∞ = %.3e\n",e); }

    /* gradcheck dérivées */
    printf("\n=== Gradcheck dérivées (finies) ===\n");
    { float x=-2.7f,h=1e-4f; double num=(spear_gelu_erf(x+h)-spear_gelu_erf(x-h))/(2*h); printf("  gelu_erf     d/df : %.3e (err %.1e)\n",q_erf_d(x),fabs(q_erf_d(x)-num)); }
    { float x=-2.7f,h=1e-4f; double num=(spear_gelu_quintic(x+h)-spear_gelu_quintic(x-h))/(2*h); printf("  gelu_quintic d/df : %.3e (err %.1e)\n",q_quintic_d(x),fabs(q_quintic_d(x)-num)); }

    /* entraînement 3 variantes, même seed */
    R=123456789UL;
    printf("\n=== Entraînement MLP 2→16→2 (spirale, SGD, même seed) ===\n");
    double t0=now_ms(); double Lq=train(spear_gelu_quintic,q_quintic_d,2000,0.15);
    R=123456789UL;        double Le=train(spear_gelu_erf,q_erf_d,2000,0.15);
    R=123456789UL;        double Lx=train(exact_gelu_ref,exact_gelu_d,2000,0.15);
    printf("  gelu_quintic final CE = %.4f\n",Lq);
    printf("  gelu_erf     final CE = %.4f   (Δ vs quintic = %+.4f)\n",Le,Le-Lq);
    printf("  exact_gelu   final CE = %.4f   (plafond)\n",Lx);
    double dt=(now_ms()-t0)/3;
    printf("\n  %s\n",
        (Le<Lq) ? "✅ gelu_erf < gelu_quintic : précision du forward ⇒ meilleure qualité" 
                : "❌ pas d'amélioration mesurée");
    printf("  (3 entraînements en %.1f ms chacun)\n",dt);
    return (Le<Lq)?0:1;
}