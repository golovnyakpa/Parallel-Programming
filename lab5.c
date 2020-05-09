#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fstream>

using namespace std;

int main()
{
	MPI_Init(0,0);
	ofstream fout;
	MPI_Status status;
	fout.open("output0.txt");
	int low_limit, high_limit, size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	double time1;
	if (rank==0)
	{
		cout << "Введите верхнюю границу диапазона" << endl;
		cin >> high_limit;
		cout << "Введите нижнюю границу диапазона" << endl;
		cin >> low_limit;
		time1=MPI_Wtime();
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&high_limit, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&low_limit, 1, MPI_INT, 0, MPI_COMM_WORLD);
	int a[high_limit];
	for (int i=0; i<high_limit; i++)
		a[i]=0;
	for (int i = rank + low_limit ; i < high_limit; i+=rank)
	{
		int flag = 0;
		for (int k = 2; k < sqrt(i); k++)
		{
			if (i % k == 0) flag = 1;
		}
		if (flag == 0) a[i]=1;
		else flag=0;
	}
	MPI_Barrier(MPI_COMM_WORLD);
	for (int i=low_limit+rank; i<high_limit; i+=rank)
	{
		if ((i%size)!=0) MPI_Send(&a[i], 1, MPI_INT, 0, i, MPI_COMM_WORLD);
		if ((i%size)==0) MPI_Recv(&a[i], 1, MPI_INT, (i%size), i, MPI_COMM_WORLD, &status);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank==0)
	{
		for (int i=low_limit; i<high_limit; i++)
			if (a[i] == 1) fout << i << endl;
		cout << "Time in seconds = " << MPI_Wtime()-time1 << endl;
	}
	MPI_Finalize();
	return 0;
}
