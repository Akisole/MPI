//Задания нормального уровня сложности #3
//6. Решение систем линейных уравнений методом сопряженных градиентов

#include "mpi.h"
#include "stdio.h"
#include <stdlib.h> 
#include <iostream>
#include <math.h>
#include <ctime>

#define EPSILON 1.0E-20
#define MAX_ITERATIONS 10000

void GetPreconditionMatrix(double **Bloc_Precond_Matrix, int NoofRows_Bloc, int NoofCols)
{

	/*... Preconditional Martix is identity matrix .......*/

	int Bloc_MatrixSize;

	int irow, icol, index;

	double *Precond_Matrix;
	Bloc_MatrixSize = NoofRows_Bloc*NoofCols;
	Precond_Matrix = (double *)malloc(Bloc_MatrixSize * sizeof(double));
	index = 0;

	for (irow = 0; irow<NoofRows_Bloc; irow++) {

		for (icol = 0; icol<NoofCols; icol++) {

			Precond_Matrix[index++] = 1.0;

		}

	}

	*Bloc_Precond_Matrix = Precond_Matrix;

}
double ComputeVectorDotProduct(double *Vector1, double *Vector2, int VectorSize)
{

	int index;

	double Product;
	Product = 0.0;

	for (index = 0; index<VectorSize; index++)

		Product += Vector1[index] * Vector2[index];
	return(Product);

}
void CalculateResidueVector(double *Bloc_Residue_Vector, double *Bloc_Matrix_A, double *Input_B, double *Vector_X, int NoofRows_Bloc, int VectorSize, int MyRank)
{

	/*... Computes residue = AX - b .......*/

	int irow, index, GlobalVectorIndex;

	double value;
	GlobalVectorIndex = MyRank * NoofRows_Bloc;

	for (irow = 0; irow<NoofRows_Bloc; irow++) {

		index = irow * VectorSize;

		value = ComputeVectorDotProduct(&Bloc_Matrix_A[index], Vector_X,

			VectorSize);

		Bloc_Residue_Vector[irow] = value - Input_B[GlobalVectorIndex++];

	}

}
void SolvePrecondMatrix(double *Bloc_Precond_Matrix, double *HVector, double *Bloc_Residue_Vector, int Bloc_VectorSize)
{

	/*...HVector = Bloc_Precond_Matrix inverse * Bloc_Residue_Vector.......*/

	int index;
	for (index = 0; index<Bloc_VectorSize; index++) {

		HVector[index] = Bloc_Residue_Vector[index] / 1.0;

	}

}


void СalculationNEwX(double *x, double *d, double s, int size) {
	int i;
	/*	printf("----------------------\n");
	printf("\n----------calculationNEwX------------\n");
	printf("\n----------Old_x------------\n");
	for (i = 0; i < size; i++)
	printf("%.3lf ", x[i]);
	*/

	for (i = 0; i < size; i++)
		x[i] = x[i] + s*d[i];
	/*
	printf("\n----------Nex_x------------\n");
	for (i = 0; i < size; i++)
	printf("%.3lf ", x[i]);
	printf("\n----------------------\n");*/
}
double Vectorsmultiplication(double *Vec1, double *Vec2, int size) {
	int i;
	double res;


	for (i = 0, res = 0; i < size; i++)
		res += Vec1[i] * Vec2[i];
	/*
	printf("----------------------\n");
	printf("\n----------Vectorsmultiplication-----------\n");
	printf("\n----------vec1------------\n");
	for (i = 0; i < size; i++)
	printf("%.3lf ", Vec1[i]);
	printf("\n----------vec2------------\n");
	for (i = 0; i < size; i++)
	printf("%.3lf ", Vec2[i]);
	printf("\n----------Result------------\n");
	printf("%.3lf ", res);
	printf("\n----------------------\n");
	*/
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

	/*printf("----------------------\n");
	printf("\n----------Bloc------------\n");
	for (i = 0; i < sizeB; i++)
	printf("%.3lf ", Bloc[i]);
	printf("\n----------Vec------------\n");
	for (i = 0; i < sizeV; i++)
	printf("%.3lf ", Vec[i]);
	printf("\n----------Multi------------\n");
	for (i = 0; i < sizeR; i++)
	printf("%.3lf ", VecR[i]);
	printf("\n----------------------\n");
	*/
}
void CreateTask(int size, double** A, double *_a, double* B, double *g_count, double *tmpr) {
	int i, j, k;

	/*	if (size == 0) {
	_a[0] = _a[3] = A[0][0] = A[1][1] = 3;
	_a[1] = _a[2] = A[1][0] = A[0][1] = -1;

	B[0] = 3; B[1] = 7;

	g_prev[0] = -3;
	g_prev[1] = -7;
	g_count[0] = g_count[1] = 0;
	X[0] = X[1] = 0;
	d[0] = d[1] = 0;
	tmpr[0] = tmpr[1] = 0;
	}
	else { */

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
	//	}
}

int main(int argc, char *argv[]) {

	MPI_Status status;
	int ProcSize, ProcRank, Root = 0;
	int k = atoi(argv[2]);

	if (k == 8) {
		int task_size, Max_iter, iter;
		int size_block_elem, size_block_row;
		int i;

		double **MatrA, *A, *A_bloc, *B, *X;
		double s, *d_prev, *g_prev, *g_count;
		double *tmpv, *tmpres;
		double division;
		double eps;
		double StartTime, EndTime;

		task_size = atoi(argv[1]);

		MPI_Init(&argc, &argv);
		MPI_Comm_size(MPI_COMM_WORLD, &ProcSize);
		MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

		if (task_size < ProcSize) {
			MPI_Finalize();
			if (ProcRank == Root)
				printf("Error : The number of processes is large.");
			return -1;
		}
		if (task_size % ProcSize != 0) {
			MPI_Finalize();
			if (ProcRank == Root)
				printf("Error : The matrix is not divided into processes.");
			return -1;
		}

		size_block_row = task_size / ProcSize;
		size_block_elem = size_block_row * task_size;

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

		MPI_Scatter(A, size_block_elem, MPI_DOUBLE, A_bloc, size_block_elem, MPI_DOUBLE, Root, MPI_COMM_WORLD);

		//-------------------------------------------------------------------------//
		//-----------------------[ g0 = b - A*x (tmp = A*x) ]----------------------//
		//-------------------------------------------------------------------------//

		BlocVectormultiplication(A_bloc, X, tmpv, size_block_elem, task_size);
		MPI_Gather(tmpv, size_block_row, MPI_DOUBLE, tmpres, size_block_row, MPI_DOUBLE, Root, MPI_COMM_WORLD);

		if (ProcRank == 0)
			for (i = 0; i < task_size; i++) {
				g_prev[i] = B[i] - tmpres[i];
				d_prev[i] = g_prev[i];
			}

		iter = 0;
		do {

			//-------------------------------------------------------------------------//
			//--------------[ s^k = <g^k-1,g^k-1> / (d^k-1 * A * d^k-1) ]--------------//
			//-------------------------------------------------------------------------//

			MPI_Bcast(d_prev, task_size, MPI_DOUBLE, Root, MPI_COMM_WORLD);
			BlocVectormultiplication(A_bloc, d_prev, tmpv, size_block_elem, task_size);
			MPI_Gather(tmpv, size_block_row, MPI_DOUBLE, tmpres, size_block_row, MPI_DOUBLE, Root, MPI_COMM_WORLD);

			if (ProcRank == Root) {
				s = Vectorsmultiplication(g_prev, g_prev, task_size) / Vectorsmultiplication(tmpres, d_prev, task_size);

				//-------------------------------------------------------------------------//
				//---------------------[ x^k = x^(k-1) + s^k * d^k-1 ]---------------------//
				//-------------------------------------------------------------------------//

				СalculationNEwX(X, d_prev, s, task_size);
			}

			MPI_Bcast(X, task_size, MPI_DOUBLE, Root, MPI_COMM_WORLD);

			//-------------------------------------------------------------------------//
			//----------------------[ g^k = g^k-1  -  s*A*d^k-1 ]----------------------//
			//-------------------------------------------------------------------------//

			BlocVectormultiplication(A_bloc, d_prev, tmpv, size_block_elem, task_size);
			MPI_Gather(tmpv, size_block_row, MPI_DOUBLE, tmpres, size_block_row, MPI_DOUBLE, Root, MPI_COMM_WORLD);

			if (ProcRank == Root) {
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
			printf("\nHi from %d. The number of processors is %d. Num of Iteration: %d.\nTime: %lf.", ProcRank, ProcSize, iter, EndTime - StartTime);
			if (atoi(argv[3]) == 1) {
				printf("\nX is:\n");
				for (int i = 0; i < task_size; i++)
					printf("%.6lf ", X[i]);
			}
			printf("\n");
		}

		MPI_Finalize();

	}


	if (k == 9) {

		int NumProcs, MyRank;

		int NoofRows, NoofCols, VectorSize; //task_size

		int n_size; //task_size

		int NoofRows_Bloc;	//size_block_row
		int Bloc_MatrixSize; //size_block_elem
		int Bloc_VectorSize; //size_block_row

		int Iteration = 0, irow, icol, index, CorrectResult;

		double **Matrix_A; //MatrA
		double *Input_A;	//A
		double *Input_B;	//B
		double  *Vector_X;	//X
		double  *Bloc_Vector_X;

		double *Bloc_Matrix_A;			//A_bloc
		double *Bloc_Precond_Matrix;	//
		double *Buffer;

		double *Bloc_Residue_Vector, *Bloc_HVector, *Bloc_Gradient_Vector;

		double *Direction_Vector, *Bloc_Direction_Vector;

		double Delta0, Delta1, Bloc_Delta0, Bloc_Delta1;

		double Tau, val, temp, Beta;
		double *AMatDirect_local, *XVector_local;

		double *ResidueVector_local, *DirectionVector_local;

		double StartTime, EndTime;

		//	MPI_Status status;

		//	FILE *fp;
		/*...Initialising MPI .......*/

		MPI_Init(&argc, &argv);
		MPI_Comm_size(MPI_COMM_WORLD, &NumProcs);
		MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);

		VectorSize = NoofRows = NoofCols = n_size = atoi(argv[1]);

		/* .......Read the Input file ......*/
		//Matrix_A Input_B Input_A(матрица в виде вектора)
		if (MyRank == Root) {

			/* ...Allocate memory and read data .....*/

			Matrix_A = (double **)malloc(n_size * sizeof(double *));
			for (int i = 0; i < n_size; i++)
				Matrix_A[i] = (double *)malloc(n_size * sizeof(double));
			Input_B = (double *)malloc(n_size * sizeof(double));
			Input_A = (double *)malloc(n_size*n_size * sizeof(double));

			for (int i = 0; i < n_size; i++)
				for (int j = 0; j <= i; j++) {
					Matrix_A[i][j] = (double)(rand() % 10);
					Matrix_A[j][i] = Matrix_A[i][j];
				}
			int k = 0;
			for (int i = 0; i < n_size; i++)
				for (int j = 0; j < n_size; j++) {
					Input_A[k] = Matrix_A[i][j];
					k++;
				}
			for (int i = 0; i < n_size; i++) {
				Input_B[i] = (double)(rand() % 10);
			}
		}


		MPI_Barrier(MPI_COMM_WORLD);


		/*...Broadcast Matrix and Vector size and perform input validation tests...*/

		MPI_Bcast(&NoofRows, 1, MPI_INT, Root, MPI_COMM_WORLD);

		MPI_Bcast(&NoofCols, 1, MPI_INT, Root, MPI_COMM_WORLD);

		MPI_Bcast(&VectorSize, 1, MPI_INT, Root, MPI_COMM_WORLD);

		//проверка ошибок
		/*
		if (NoofRows != NoofCols) {
		MPI_Finalize();
		if (MyRank == Root)
		printf("Error : Coefficient Matrix Should be square matrix");
		exit(-1);
		}
		if (NoofRows != VectorSize) {

		MPI_Finalize();

		if (MyRank == Root)

		printf("Error : Matrix Size should be equal to VectorSize");

		exit(-1);

		}
		*/
		//ету оставить
		if (NoofRows % NumProcs != 0) {
			MPI_Finalize();
			if (MyRank == Root)
				printf("Error : Matrix cannot be evenly striped among processes");
			exit(-1);
		}

		/*...Allocate memory for Input_B and BroadCast Input_B.......*/

		if (MyRank != Root)
			Input_B = (double *)malloc(VectorSize * sizeof(double));

		MPI_Bcast(Input_B, VectorSize, MPI_DOUBLE, Root, MPI_COMM_WORLD);

		/*...Allocate memory for Block Matrix A and Scatter Input_A .......*/

		NoofRows_Bloc = NoofRows / NumProcs;

		Bloc_VectorSize = NoofRows_Bloc;

		Bloc_MatrixSize = NoofRows_Bloc * NoofCols;

		Bloc_Matrix_A = (double *)malloc(Bloc_MatrixSize * sizeof(double));

		MPI_Scatter(Input_A, Bloc_MatrixSize, MPI_DOUBLE, Bloc_Matrix_A, Bloc_MatrixSize, MPI_DOUBLE, Root, MPI_COMM_WORLD);


		/*... Allocates memory for solution vector and intialise it to zero.......*/

		Vector_X = (double *)malloc(VectorSize * sizeof(double));

		for (index = 0; index < VectorSize; index++)

			Vector_X[index] = 0.0;
		/*...Calculate RESIDUE = AX - b .......*/
		StartTime = MPI_Wtime();
		Bloc_Residue_Vector = (double *)malloc(NoofRows_Bloc * sizeof(double));

		CalculateResidueVector(Bloc_Residue_Vector, Bloc_Matrix_A, Input_B, Vector_X, NoofRows_Bloc, VectorSize, MyRank);
		/*... Precondition Matrix is identity matrix ......*/

		GetPreconditionMatrix(&Bloc_Precond_Matrix, NoofRows_Bloc, NoofCols);
		/*...Bloc_HVector = Bloc_Precond_Matrix inverse * Bloc_Residue_Vector......*/

		Bloc_HVector = (double *)malloc(Bloc_VectorSize * sizeof(double));

		SolvePrecondMatrix(Bloc_Precond_Matrix, Bloc_HVector, Bloc_Residue_Vector, Bloc_VectorSize);


		/*...Initailise Bloc Direction Vector = -(Bloc_HVector).......*/

		Bloc_Direction_Vector = (double *)malloc(Bloc_VectorSize * sizeof(double));

		for (index = 0; index < Bloc_VectorSize; index++)

			Bloc_Direction_Vector[index] = 0 - Bloc_HVector[index];
		/*...Calculate Delta0 and check for convergence .......*/

		Bloc_Delta0 = ComputeVectorDotProduct(Bloc_Residue_Vector, Bloc_HVector, Bloc_VectorSize);

		MPI_Allreduce(&Bloc_Delta0, &Delta0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		if (Delta0 < EPSILON) {

			MPI_Finalize();

			exit(0);

		}
		/*...Allocate memory for Direction Vector.......*/

		Direction_Vector = (double *)malloc(VectorSize * sizeof(double));
		/*...Allocate temporary buffer to store Bloc_Matrix_A*Direction_Vector...*/

		Buffer = (double *)malloc(Bloc_VectorSize * sizeof(double));
		/*...Allocate memory for Bloc_Vector_X .......*/

		Bloc_Vector_X = (double *)malloc(Bloc_VectorSize * sizeof(double));
		Iteration = 0;

		do {
			Iteration++;

			/*

			if(MyRank == Root)

			printf("Iteration : %d\n",Iteration);

			*/
			/*...Gather Direction Vector on all processes.......*/

			MPI_Allgather(Bloc_Direction_Vector, Bloc_VectorSize, MPI_DOUBLE,

				Direction_Vector, Bloc_VectorSize, MPI_DOUBLE, MPI_COMM_WORLD);
			/*...Compute Tau = Delta0 / (DirVector Transpose*Matrix_A*DirVector)...*/

			for (index = 0; index < NoofRows_Bloc; index++) {

				Buffer[index] = ComputeVectorDotProduct(&Bloc_Matrix_A[index*NoofCols],

					Direction_Vector, VectorSize);

			}

			temp = ComputeVectorDotProduct(Bloc_Direction_Vector, Buffer, Bloc_VectorSize);
			MPI_Allreduce(&temp, &val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

			Tau = Delta0 / val;
			/*Compute new vector Xnew = Xold + Tau*Direction........................*/

			/*Compute BlocResidueVec = BlocResidueVect + Tau*Bloc_MatA*DirVector...*/

			for (index = 0; index < Bloc_VectorSize; index++) {

				Bloc_Vector_X[index] = Vector_X[MyRank*Bloc_VectorSize + index] + Tau*Bloc_Direction_Vector[index];

				Bloc_Residue_Vector[index] = Bloc_Residue_Vector[index] + Tau*Buffer[index];

			}

			/*...Gather New Vector X at all processes......*/

			MPI_Allgather(Bloc_Vector_X, Bloc_VectorSize, MPI_DOUBLE, Vector_X,

				Bloc_VectorSize, MPI_DOUBLE, MPI_COMM_WORLD);
			SolvePrecondMatrix(Bloc_Precond_Matrix, Bloc_HVector, Bloc_Residue_Vector, Bloc_VectorSize);

			Bloc_Delta1 = ComputeVectorDotProduct(Bloc_Residue_Vector, Bloc_HVector, Bloc_VectorSize);

			MPI_Allreduce(&Bloc_Delta1, &Delta1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			if (Delta1 < EPSILON)

				break;
			Beta = Delta1 / Delta0;

			Delta0 = Delta1;

			for (index = 0; index < Bloc_VectorSize; index++) {

				Bloc_Direction_Vector[index] = -Bloc_HVector[index] + Beta*Bloc_Direction_Vector[index];

			}

		} while (Delta0 > EPSILON && Iteration < MAX_ITERATIONS);
		MPI_Barrier(MPI_COMM_WORLD);

		EndTime = MPI_Wtime();
		if (MyRank == 0) {
			printf("\n");
			printf("Results on processor %d: ", MyRank);
			printf("Number of iterations = %d. \nTime: %lf", Iteration, EndTime - StartTime);

			if (atoi(argv[3]) == 1) {
				printf("\nSolution vector \n");
				for (irow = 0; irow < n_size; irow++)
					printf("%.6lf ", Vector_X[irow]);
			}
			printf("\n");
		}

		MPI_Finalize();

	}


	return 0;
}