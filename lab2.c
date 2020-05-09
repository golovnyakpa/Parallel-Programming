#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define  Max(a,b) ((a)>(b)?(a):(b))
#define  N (300)
double   maxeps = 0.1e-7;
int itmax = 13;
int i,j,k;
double eps;
double A [N][N][N],  B[N][N][N];

const int THREAD_NUM = 2;

void relax();
void resid();
void init();
void verify();
void wtime();

int main(int an, char **av) {
	int it;
	double time0, time1;
	init();
	time0 = omp_get_wtime ();
	for(it=1; it<=itmax; it++){
		eps = 0.;
			{relax();
			resid();}
		printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}
	time1 = omp_get_wtime ();
	printf("Time in seconds=%gs\t",time1-time0);
	verify();
	return 0;
}

void init()
{
    #pragma omp parallel for collapse(3)
	for(i=0; i<=N-1; i++)
		for(j=0; j<=N-1; j++)
			for(k=0; k<=N-1; k++)
			{
				if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)
					A[i][j][k]= 0.;
				else
					A[i][j][k]= ( 4. + i + j + k) ;
			}
}

void relax()
{
	#pragma omp parallel for collapse(3)
		for(i=1; i<=N-2; i++)
			for(j=1; j<=N-2; j++)
				for(k=1; k<=N-2; k++)
					B[i][j][k]=(A[i-1][j][k]+A[i+1][j][k]+A[i][j-1][k]+A[i][j+1][k]+A[i][j][k-1]+A[i][j][k+1])/6.;
}

void resid()
{
    #pragma omp parallel num_threads(THREAD_NUM)
    #pragma omp  for collapse(3)reduction(max:eps) //private(eps)
		for(i=1; i<=N-2; i++)
			for(j=1; j<=N-2; j++)
				for(k=1; k<=N-2; k++) {
					double e;
					e = fabs(A[i][j][k] - B[i][j][k]);
					A[i][j][k] = B[i][j][k];
						eps = Max(eps,e);
				}
}

void verify()
{
	double s;
	s=0.;
    #pragma omp parallel num_threads(THREAD_NUM)
	#pragma omp parallel for collapse(3) reduction (+:s)
	for(i=0; i<=N-1; i++)
		for(j=0; j<=N-1; j++)
			for(k=0; k<=N-1; k++)
			{
				s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
			}
	printf("  S = %f\n",s);
}
