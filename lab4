#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

#define Max(a,b) ((a)>(b)?(a):(b))

#define N 131

double maxeps = 0.1e-7;

int itmax = 13;

int i,j,k;

double eps;

double A [N][N][N], B[N][N][N] ,C[N][N][N];

MPI_Status status;

void relax();

void resid();

void init();

void verify();

int it,size=0,rank=0,count;

int main()
{
	double time0, time1;
	MPI_Init(0,0);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	init();
	if(rank==0)time0 = MPI_Wtime();
	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		resid();
		if(rank==0)printf( "it=%4i eps=%f\n", it,eps);
		if (eps < maxeps) break;
		MPI_Barrier(MPI_COMM_WORLD);
	}
	if(rank==0)time1 = MPI_Wtime();
	if(rank==0)printf("Time in seconds=%gs\t",time1-time0);
	if(rank==0)verify();
	MPI_Finalize();
	return 0;
}

void init()
{
	for(i=0; i<=N-1; i++)
		for(j=0; j<=N-1; j++)
			for(k=0; k<=N-1; k++)
			{
				if (i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1) A[i][j][k]= 0.;
				else A[i][j][k] = ( 4. + i + j + k) ;
			}
}

void relax()
{
	for(i=1+rank; i<=N-2; i+=size)
		for(j=1; j<=N-2; j++)
			for(k=1; k<=N-2; k++)
				B[i][j][k]=(A[i-1][j][k]+A[i+1][j][k]+A[i][j-1][k]+A[i][j+1][k]+A[i][j][k-1]+A[i][j][k+1])/6.; 
}

void resid()
{
	for(i=1+rank; i<=N-2; i+=size)
		for(j=1; j<=N-2; j++)
			for(k=1; k<=N-2; k++)
			{
				double e;   
				e = fabs(A[i][j][k] - B[i][j][k]);
				A[i][j][k] = B[i][j][k];
				eps = Max(eps,e);
			}
	MPI_Allreduce(&eps,&eps,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	for(i=0;i<N-2;i++)
	{
		if(rank==(i % size) && (i % size != 0))
			MPI_Send(A[i+1],N*N, MPI_DOUBLE, 0, i, MPI_COMM_WORLD); 
	}
	for(i=0;i<N-2;i++)
		if(rank==0 && (i % size != 0))
			MPI_Recv(A[i+1],N*N, MPI_DOUBLE, (i % size), i, MPI_COMM_WORLD,&status);
	/*if(rank!= 0){
	for(i=0;i<N-2;i++)
			{MPI_Send(A[i+1],N*N, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);printf("essage sent from proc % d with tag %d, rank, i) }
		}
	if(rank==0){
	for(i=0;i<N-2;i++)
			MPI_Recv(A[i+1],N*N, MPI_DOUBLE, (i % size), i, MPI_COMM_WORLD,&status);
	}*/
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(A,N*N*N, MPI_DOUBLE,0, MPI_COMM_WORLD);
}

void verify()
{
	double s;
	s=0.;
	for(i=0; i<=N-1; i++)
		for(j=0; j<=N-1; j++)
			for(k=0; k<=N-1; k++)
			{
				s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
			}
	printf(" S = %f\n",s);
} 
