#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <omp.h>
#include <windows.h>
#define n 4
#define eps 0.00001
double det=1;

void print(double **A,int N){
	int i,j;
	for(i=0;i<N;++i){
		for(j=0;j<N;++j)
            std::cout << A[i][j] << "  ";
			std::cout << std::endl;
		}
}

void swapstrings(double **A, int N, int k)
{
    int num=k;
    double temp;
    for (int i=k+1; i<N; i++)
        if (fabs(A[i][k]>0)) num=i;
    for (int i=k; i<N; i++)
    {
        temp=A[num][i];
        A[num][i]= A[k][i];
        A[k][i]=temp;
    }
}


void clear(double **A, int N){
	for (int i = 0; i < N; i++)
        delete A[i];
    delete A;
}

double **init(double **A,int N){
	A = new double *[N];
    for (int i = 0; i < N; i++)
        A[i] = new double [N];
    return A;
}

double **inversion(double **A, int N)
{
    double temp;
    double **E;
    E=init(E,N);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++){
            E[i][j] = 0.0;
           // std::cout << omp_get_thread_num() << std::endl;
            if (i == j)
                E[i][j] = 1.0;
        }
    for (int k = 0; k < N; k++)
    {
        if (fabs(A[k][k])<eps)
        {
            swapstrings(A, N, k);
            swapstrings(E, N, k);
            det*=-1;
        }
        if (det==0) return NULL;
        temp=A[k][k];
        det*=temp;
        #pragma omp parallel
        {
		#pragma omp for
        for (int j = 0; j < N; j++)
        {
            A[k][j] /= temp;
            E[k][j] /= temp;
            //std::cout << omp_get_thread_num() << std::endl;
        }
      #pragma omp for private(temp)
        for (int i = k + 1; i < N; i++)
        {
            temp = A[i][k];
            std::cout << omp_get_thread_num() << std::endl;
            for (int j = 0; j < N; j++)
            {
                A[i][j] -= A[k][j] * temp;
                E[i][j] -= E[k][j] * temp;
                //std::cout << omp_get_thread_num() << std::endl;
            }
        }
	}
    }

    for (int k = N - 1; k > 0; k--)
    {
		#pragma omp parallel for private(temp)
        for (int i = k - 1; i >= 0; i--)
        {
            temp = A[i][k];

            for (int j = 0; j < N; j++)
            {
                A[i][j] -= A[k][j] * temp;
                E[i][j] -= E[k][j] * temp;
            }
        }
    }

    clear(A,N);
    return E;
}

void TestInit(double **A, int N)
{
    A[0][0]=3;
    A[0][1]=4;
    A[0][2]=-2;
    A[1][0]=-2;
    A[1][1]=1;
    A[1][2]=0;
    A[2][0]=2;
    A[2][1]=3;
    A[2][2]=0;
}

void TestInit2(double **A, int N)
{
    A[0][0]=2;
    A[0][1]=5;
    A[0][2]=7;
    A[1][0]=6;
    A[1][1]=3;
    A[1][2]=4;
    A[2][0]=5;
    A[2][1]=-2;
    A[2][2]=-3;
}

int main()
{
    //std::cout << "Введите количество потоков"  <<
    /*int n;
    std::cin >> n;*/
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);
    int N;
    double time;
    omp_set_num_threads(n);
    std::cout << "Введите размерность матрицы N: ";
    std::cin >> N;
    double **matrix = new double *[N];
    for (int i = 0; i < N; i++)
        matrix[i] = new double [N];
    //TestInit2(matrix, N);
	for(int i=0;i<N;++i)
		for(int j=0;j<N;++j)
			matrix[i][j]=rand()%10;
	/*for(int i=0;i<N;++i)
		for(int j=0;j<N;++j)
			std::cin >> matrix[i][j];*/
	time=omp_get_wtime();
    matrix=inversion(matrix, N);
    if (matrix==NULL) {std::cout << "Ошибка! Строки лиенйно зависимы"; return -1;}
    #pragma omp parallel for
    for(int i=0;i<N;++i)
		for(int j=0;j<N;++j)
        {
			matrix[i][j]*=det;
			//std::cout << omp_get_thread_num() << std::endl;
        }
    std::cout << "Время нахождения матрицы алгебраических дополнений: " << (omp_get_wtime()-time) << " секунд\n";
	for(int i=0;i<N;++i){
		for(int j=0;j<N;++j)
			std::cout << matrix[j][i] << " ";
			std::cout << "\n";
		}
    clear(matrix,N);
    return 0;
}
