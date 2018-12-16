//Задания нормального уровня сложности #3
//6. Решение систем линейных уравнений методом сопряженных градиентов

#include "mpi.h"
#include "stdio.h"
#include <stdlib.h> 
#include <ctime>

#define EPSILON 1.0E-20
#define MAX_ITERATIONS 100000

void СalculationNEwX(double *x, double *d, double s, int size) {
	int i;

	for (i = 0; i < size; i++)
		x[i] = x[i] + s*d[i];
}
double Vectorsmultiplication(double *Vec1, double *Vec2, int size) {
	int i;
	double res;

	for (i = 0, res = 0; i < size; i++)
		res += Vec1[i] * Vec2[i];
	
	return res;
}
void BlocVectormultiplication(double *Bloc, double *Vec, double *VecR, int sizeB, int sizeV) {
	int i, j, sizeR;

	sizeR = sizeB / sizeV;

	for (int i = 0; i < sizeR; i++)
		VecR[i] = 0;

	for (i = 0; i < sizeR; i++)
		for (j = 0; j < sizeV; j++)
			VecR[i] += Bloc[i*sizeV + j] * Vec[j];
}
void CreateTask(int size, double** A, double *_a, double* B, double *g_count, double *tmpr) {
	int i, j, k;

	//**/srand(time(NULL));

	for (i = 0; i < size; i++)
		for (j = 0; j <= i; j++) {
			A[i][j] = (double)(rand() % 10);
			A[j][i] = A[i][j];
		}

	k = 0;
	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++) {
			_a[k] = A[i][j];
			k++;
		}

	for (i = 0; i < size; i++) {
		B[i] = (double)(rand() % 10);
		g_count[i] = 0;
		tmpr[i] = 0;
	}
}

int main(int argc, char *argv[]) {

	int ProcSize, ProcRank, Root;
	int task_size, size_block_elem, size_block_row, residue;
	int i, iter;
	double StartTime, EndTime;
	double division;
	double eps;
	
	double **MatrA, *A, *A_bloc, *B, *X;
	double s, *d_prev, *g_prev, *g_count;
	double *tmpv, *tmpres;
	int *sendcounts, *rowcounts, *displs, *displsrow;		

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

	Root = 0;
	task_size = atoi(argv[1]);
	residue = task_size % ProcSize;

	if (ProcRank < residue) {
		size_block_row = task_size / ProcSize + 1;
		size_block_elem = size_block_row * task_size;
	}
	else {
		size_block_row = task_size / ProcSize;
		size_block_elem = size_block_row * task_size;
	}

	A_bloc = (double*)malloc(size_block_elem * sizeof(double*));
	X = (double*)malloc(task_size * sizeof(double*));
	d_prev = (double*)malloc(task_size * sizeof(double*));
	tmpv = (double*)malloc(size_block_row * sizeof(double*));

	if (ProcRank == Root) {
		MatrA = (double**)malloc(task_size * sizeof(double*));
		for (int i = 0; i < task_size; i++)
			MatrA[i] = (double*)malloc(task_size * sizeof(double*));
		A = (double*)malloc(task_size*task_size * sizeof(double*));
		B = (double*)malloc(task_size * sizeof(double*));
		g_prev = (double*)malloc(task_size * sizeof(double*));
		g_count = (double*)malloc(task_size * sizeof(double*));
		tmpres = (double*)malloc(task_size * sizeof(double*));

		CreateTask(task_size, MatrA, A, B, g_count, tmpres);

		sendcounts = (int*)malloc(ProcSize * sizeof(int));
		rowcounts = (int*)malloc(ProcSize * sizeof(int));
		displsrow = (int*)malloc(ProcSize * sizeof(int));
		displs = (int*)malloc(ProcSize * sizeof(int));

		for (i = 0; i < ProcSize; i++) {
			if (i < residue) {
				sendcounts[i] = (task_size / ProcSize + 1)*task_size;
				rowcounts[i] = task_size / ProcSize + 1;
				displs[i] = i*(task_size / ProcSize + 1)*task_size;
				displsrow[i] = i*(task_size / ProcSize + 1);
			}
			else {
				sendcounts[i] = (task_size / ProcSize)*task_size;
				rowcounts[i] = task_size / ProcSize;
				displs[i] = ((i - residue)*(task_size / ProcSize) + residue*(task_size / ProcSize + 1))*task_size;
				displsrow[i] = (i - residue)*(task_size / ProcSize) + residue*(task_size / ProcSize + 1);

			}
		}
	}

	for (i = 0; i < size_block_row; i++)
		tmpv[i] = 0.0;
	for (i = 0; i < task_size; i++)
		X[i] = 0.0;

	MPI_Barrier(MPI_COMM_WORLD);
	StartTime = MPI_Wtime();

	/************************************************************************/
	/***************************** [ Старт ] ********************************/
	/************************************************************************/

	MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, A_bloc, size_block_elem, MPI_DOUBLE, Root, MPI_COMM_WORLD);

//-------------------------------------------------------------------------//
//-----------------------------[ g0 = b - A*x ]----------------------------//
//-------------------------------------------------------------------------//

	if (ProcRank == 0)
		for (i = 0; i < task_size; i++) {
			g_prev[i] = B[i];
			d_prev[i] = g_prev[i];
		}

	iter = 0;

	do {

		//-------------------------------------------------------------------------//
		//--------------[ s^k = <g^k-1,g^k-1> / (d^k-1 * A * d^k-1) ]--------------//
		//-------------------------------------------------------------------------//

		MPI_Bcast(d_prev, task_size, MPI_DOUBLE, Root, MPI_COMM_WORLD);

		BlocVectormultiplication(A_bloc, d_prev, tmpv, size_block_elem, task_size);

		MPI_Gatherv(tmpv, size_block_row, MPI_DOUBLE, tmpres, rowcounts, displsrow, MPI_DOUBLE, Root, MPI_COMM_WORLD);

		if (ProcRank == Root) {
			s = Vectorsmultiplication(g_prev, g_prev, task_size) / Vectorsmultiplication(tmpres, d_prev, task_size);

			//-------------------------------------------------------------------------//
			//---------------------[ x^k = x^(k-1) + s^k * d^k-1 ]---------------------//
			//-------------------------------------------------------------------------//

			СalculationNEwX(X, d_prev, s, task_size);

			//-------------------------------------------------------------------------//
			//----------------------[ g^k = g^k-1  -  s*A*d^k-1 ]----------------------//
			//-------------------------------------------------------------------------//

			for (i = 0; i < task_size; i++)
				g_count[i] = g_prev[i] - s * tmpres[i];

			eps = Vectorsmultiplication(g_count, g_count, task_size);

			//-------------------------------------------------------------------------//
			//----------[ d^k = g^k + <g^k,g^k>/<g^(k-1),g^(k-1)> * d^(k-1) ]----------//
			//-------------------------------------------------------------------------//

			division = Vectorsmultiplication(g_count, g_count, task_size) / Vectorsmultiplication(g_prev, g_prev, task_size);

			for (i = 0; i < task_size; i++) {
				d_prev[i] = g_count[i] + division*d_prev[i];
				g_prev[i] = g_count[i];
			}
		}

		MPI_Bcast(&eps, 1, MPI_DOUBLE, Root, MPI_COMM_WORLD);

		iter++;

	} while (iter < MAX_ITERATIONS && eps > EPSILON);

	/************************************************************************/
	/***************************** [ Конец ] ********************************/
	/************************************************************************/

	MPI_Barrier(MPI_COMM_WORLD);
	EndTime = MPI_Wtime();

	if (ProcRank == Root) {

		printf("\nNum of Iteration: %d.\nTime: %lf.\n", iter, EndTime - StartTime);

		if (atoi(argv[2]) == 1) {
			printf("\nMatrix:\n");

			for (int i = 0; i < task_size; i++) {
				for (int j = 0; j < task_size; j++)
					printf("%.3lf ", MatrA[i][j]);
				printf("\n");
			}

			printf("\nB:\n");

			for (int i = 0; i < task_size; i++)
				printf("%.3lf ", B[i]);

			printf("\n\nX is:\n");

			for (int i = 0; i < task_size; i++)
				printf("%.6lf ", X[i]);

			printf("\n");
		}
	}

	free(A_bloc);
	free(X);
	free(d_prev);
	free(tmpv);

	if (ProcRank == 0) {
		free(MatrA);
		free(A);
		free(B);
		free(g_prev);
		free(g_count);
		free(tmpres);
		free(sendcounts);
		free(rowcounts);
		free(displsrow);
		free(displs);
	}

	MPI_Finalize();

	return 0;
}