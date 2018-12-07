//ТОПОЛОГИИ СЕТЕЙ ПЕРЕДАЧИ ДАННЫХ
//11. Кольцо
//Одностороннее кольцо

#include "mpi.h"
#include "stdio.h"
#include <stdlib.h> 
#include <iostream>

/*
void data_transfer(MPI_Comm comm, int Comm_size, int Comm_rank, void* buf, int count, MPI_Datatype datatype, MPI_Status *status,
	int sourse, int prev_neighbor, int next_neighbor, int dest) {
	int middle;

	middle = Comm_size / 2;


	if (Comm_rank == sourse) {	//если источник
		if (sourse > dest) {
			if (sourse - dest <= middle) {
				MPI_Send(buf, count, datatype, prev_neighbor, 10, comm);
				printf("I %d, send to %d.\n", Comm_rank, prev_neighbor);
			}
			else if (sourse - dest > middle) {
				MPI_Send(buf, count, datatype, next_neighbor, 10, comm);
				printf("I %d, send to %d.\n", Comm_rank, next_neighbor);
			}
			else printf("Err\n");
		}
		else if (sourse < dest) {
			if (dest - sourse <= middle) {
				MPI_Send(buf, count, datatype, next_neighbor, 10, comm);
				printf("I %d, send to %d.\n", Comm_rank, next_neighbor);
			}
			else if (dest - sourse > middle) {
				MPI_Send(buf, count, datatype, prev_neighbor, 10, comm);
				printf("I %d, send to %d.\n", Comm_rank, prev_neighbor);
			}
			else printf("Err\n");
		}
		else printf("Err\n");
	}
	/*	else if (Comm_rank == dest) {	//если приемник
	if (sourse > dest) {
	if (sourse - dest <= middle) {
	MPI_Recv(buf, count, datatype, next_neighbor, 10, comm, status);
	printf("I %d, get from %d.\n", Comm_rank, next_neighbor);
	}
	else if (sourse - dest > middle) {
	MPI_Recv(buf, count, datatype, prev_neighbor, 10, comm, status);
	printf("I %d, get from %d.\n", Comm_rank, prev_neighbor);
	}
	else printf("Err\n");
	}
	else if (sourse < dest) {
	if (dest - sourse <= middle) {
	MPI_Recv(buf, count, datatype, prev_neighbor, 10, comm, status);
	printf("I %d, get from %d.\n", Comm_rank, prev_neighbor);
	}
	else if (dest - sourse > middle) {
	MPI_Recv(buf, count, datatype, next_neighbor, 10, comm, status);
	printf("I %d, get from %d.\n", Comm_rank, next_neighbor);
	}
	else printf("Err\n");
	}
	else printf("Err\n");
	}
	else if (Comm_rank != sourse && Comm_rank != dest) { //если промежуточный
	if (sourse > dest) {
	if (sourse - dest <= middle) {
	MPI_Recv(buf, count, datatype, next_neighbor, 10, comm, status);
	MPI_Send(buf, count, datatype, prev_neighbor, 10, comm);
	printf("I %d, get from %d and send to %d.\n", Comm_rank, next_neighbor, prev_neighbor);
	}
	else if (sourse - dest > middle) {
	MPI_Recv(buf, count, datatype, prev_neighbor, 10, comm, status);
	MPI_Send(buf, count, datatype, next_neighbor, 10, comm);
	printf("I %d, get from %d and send to %d.\n", Comm_rank, prev_neighbor, next_neighbor);
	}
	else printf("Err\n");
	}
	else if (sourse < dest) {
	if (dest - sourse <= middle) {
	MPI_Recv(buf, count, datatype, prev_neighbor, 10, comm, status);
	MPI_Send(buf, count, datatype, next_neighbor, 10, comm);
	printf("I %d, get from %d and send to %d.\n", Comm_rank, prev_neighbor, next_neighbor);
	}
	else if (dest - sourse > middle) {
	MPI_Recv(buf, count, datatype, next_neighbor, 10, comm, status);
	MPI_Send(buf, count, datatype, prev_neighbor, 10, comm);
	printf("I %d, get from %d and send to %d.\n", Comm_rank, next_neighbor, prev_neighbor);
	}
	else printf("Err\n");
	}
	else printf("Err\n");
	}
	*//*
}
*/

/*
void data_transfer(MPI_Comm comm, int Comm_size, int Comm_rank, void* buf, int count, MPI_Datatype datatype, MPI_Status *status,
	int sourse, int prev_neighbor, int next_neighbor, int dest) {

	int middle = (sourse + Comm_rank / 2) % Comm_rank;

	if (Comm_rank == sourse)
	{
		MPI_Send(buf, count, datatype, prev_neighbor, 12, comm);
		MPI_Send(buf, count, datatype, next_neighbor, 12, comm);
	}
	else
		if (Comm_rank == middle)
		{
			MPI_Recv(buf, count, datatype, prev_neighbor, MPI_ANY_TAG, comm, status);
		}
		else
		{
			if ((sourse < Comm_rank && Comm_rank < middle) || ((middle < sourse) && (sourse < Comm_rank || Comm_rank < middle)))
			{
				MPI_Recv(buf, count, datatype, prev_neighbor, MPI_ANY_TAG, comm, status);
				MPI_Send(buf, count, datatype, next_neighbor, 12, comm);
			}
			else
			{
				MPI_Recv(buf, count, datatype, next_neighbor, MPI_ANY_TAG, comm, status);
				if (prev_neighbor != middle)
				{
					MPI_Send(buf, count, datatype, prev_neighbor, 12, comm);
				}
			}
		}
}
*/

/*
void data_transfer(MPI_Comm comm, int Comm_size, int Comm_rank, void* buf, int count, MPI_Datatype datatype, MPI_Status *status,
	int sourse, int prev_neighbor, int next_neighbor, int dest) {
	int middle;

	middle = Comm_size / 2;
	// -n 6 +++ 2 0 +++ m=3

	if (Comm_rank == sourse) {	//если источник
		if (sourse > dest) {
			if (sourse <= middle) {
				MPI_Send(buf, count, datatype, prev_neighbor, 10, comm);
				printf("I %d, send to %d.\n", Comm_rank, prev_neighbor);
			}
			else if (sourse > middle) {
				if (sourse - dest <= middle) {
					MPI_Send(buf, count, datatype, prev_neighbor, 10, comm);
					printf("I %d, send to %d.\n", Comm_rank, prev_neighbor);
				}
				else if (sourse - dest > middle) {
					MPI_Send(buf, count, datatype, next_neighbor, 10, comm);
					printf("I %d, send to %d.\n", Comm_rank, next_neighbor);
				}
			}
			else if (sourse < dest) {
				if (sourse > middle) {
					MPI_Send(buf, count, datatype, next_neighbor, 10, comm);
					printf("I %d, send to %d.\n", Comm_rank, next_neighbor);
				}
				else if (sourse <= middle) {
					if (dest - sourse <= middle) {
						MPI_Send(buf, count, datatype, next_neighbor, 10, comm);
						printf("I %d, send to %d.\n", Comm_rank, next_neighbor);
					}
					else if (dest - sourse > middle) {
						MPI_Send(buf, count, datatype, prev_neighbor, 10, comm);
						printf("I %d, send to %d.\n", Comm_rank, prev_neighbor);
					}

				}
			}

		}
	}
	/*	else if (Comm_rank == dest) {	//если приемник
	if (sourse > dest) {
	if (sourse - dest <= middle) {
	MPI_Recv(buf, count, datatype, next_neighbor, 10, comm, status);
	printf("I %d, get from %d.\n", Comm_rank, next_neighbor);
	}
	else if (sourse - dest > middle) {
	MPI_Recv(buf, count, datatype, prev_neighbor, 10, comm, status);
	printf("I %d, get from %d.\n", Comm_rank, prev_neighbor);
	}
	else printf("Err\n");
	}
	else if (sourse < dest) {
	if (dest - sourse <= middle) {
	MPI_Recv(buf, count, datatype, prev_neighbor, 10, comm, status);
	printf("I %d, get from %d.\n", Comm_rank, prev_neighbor);
	}
	else if (dest - sourse > middle) {
	MPI_Recv(buf, count, datatype, next_neighbor, 10, comm, status);
	printf("I %d, get from %d.\n", Comm_rank, next_neighbor);
	}
	else printf("Err\n");
	}
	else printf("Err\n");
	}
	else if (Comm_rank != sourse && Comm_rank != dest) { //если промежуточный
	if (sourse > dest) {
	if (sourse - dest <= middle) {
	MPI_Recv(buf, count, datatype, next_neighbor, 10, comm, status);
	MPI_Send(buf, count, datatype, prev_neighbor, 10, comm);
	printf("I %d, get from %d and send to %d.\n", Comm_rank, next_neighbor, prev_neighbor);
	}
	else if (sourse - dest > middle) {
	MPI_Recv(buf, count, datatype, prev_neighbor, 10, comm, status);
	MPI_Send(buf, count, datatype, next_neighbor, 10, comm);
	printf("I %d, get from %d and send to %d.\n", Comm_rank, prev_neighbor, next_neighbor);
	}
	else printf("Err\n");
	}
	else if (sourse < dest) {
	if (dest - sourse <= middle) {
	MPI_Recv(buf, count, datatype, prev_neighbor, 10, comm, status);
	MPI_Send(buf, count, datatype, next_neighbor, 10, comm);
	printf("I %d, get from %d and send to %d.\n", Comm_rank, prev_neighbor, next_neighbor);
	}
	else if (dest - sourse > middle) {
	MPI_Recv(buf, count, datatype, next_neighbor, 10, comm, status);
	MPI_Send(buf, count, datatype, prev_neighbor, 10, comm);
	printf("I %d, get from %d and send to %d.\n", Comm_rank, next_neighbor, prev_neighbor);
	}
	else printf("Err\n");
	}
	else printf("Err\n");
	}
	*//*
}
*/

void SSend(double& buf, int count, MPI_Datatype datatype, MPI_Comm com, int sourse, int destination, int i) {

	//std::cout << buf;

	MPI_Status status;
	int ProcSize, ProcRank;// nnod, nedg;

	MPI_Comm_size(com, &ProcSize);
	MPI_Comm_rank(com, &ProcRank);

//	MPI_Graphdims_get(com, &nnod, &nedg);



	if (ProcRank == sourse) {
		if (ProcRank != ProcSize - 1) {
			buf = MPI_Wtime();
		//	printf("Hi-, I'm sourse %d and i send to %d\n", ProcRank, ProcRank + 1);
			MPI_Send(&buf, count, datatype, ProcRank + 1, 10, com);
		}
		else {
			buf = MPI_Wtime();
		//	printf("Hi-, I'm sourse %d and i send to %d\n", ProcRank, 0);
			MPI_Send(&buf, count, datatype, 0, 10, com);
		}
	}
	else if (ProcRank == destination) {
		MPI_Recv(&buf, count, datatype, MPI_ANY_SOURCE, 10, com, &status);
	//	printf("Hi-, I'm destination %d and i get %d\n", ProcRank, buf);
	//	printf("Hi-, I'm destination %d\n", ProcRank);
		if (i == 1)
			std::cout << "Time from Graph_Create.\n  " << MPI_Wtime() - buf << std::endl;
		if (i == 2)
			std::cout << "Time.\n  " << MPI_Wtime() - buf << std::endl;
		if (i == 3)
			std::cout << "Time2.\n  " << MPI_Wtime() - buf << std::endl;
	}
	else {
		if (sourse < destination - 1 && sourse < ProcRank && ProcRank < destination) {

			MPI_Recv(&buf, count, datatype, MPI_ANY_SOURCE, 10, com, &status);
			MPI_Send(&buf, count, datatype, ProcRank + 1, 10, com);
		//	printf("Hi-, I'm %d and i send %d to %d\n", ProcRank, buf, ProcRank);
		//	printf("Hi-, I'm %d and i send to %d\n", ProcRank, ProcRank+1);
		}
		else if (sourse > destination && (sourse < ProcRank || destination > ProcRank)) {
			if (ProcRank != ProcSize - 1) {
				MPI_Recv(&buf, count, datatype, MPI_ANY_SOURCE, 10, com, &status);
				MPI_Send(&buf, count, datatype, ProcRank + 1, 10, com);
			//	printf("Hi-, I'm %d and i send %d to %d\n", ProcRank, buf, ProcRank + 1);
			//  printf("Hi-, I'm %d and i send to %d\n", ProcRank, ProcRank + 1);
			}
			else {
				MPI_Recv(&buf, count, datatype, MPI_ANY_SOURCE, 10, com, &status);
				MPI_Send(&buf, count, datatype, 0, 10, com);
			//	printf("Hi-, I'm %d and i send %d to %d\n", ProcRank, buf, 0);
			//	printf("Hi-, I'm %d and i send to %d\n", ProcRank, 0);
			}
		}
		else
			;
		//	printf("Hi-, I'm %d and i nothing to do\n", ProcRank);
	}
}


/*
void main(int argc, char *argv[]) {
	
	MPI_Comm comm;
	MPI_Status status;
	int ProcSize, ProcRank, source, destination, buffer;
	int nnodes, n_index, n_edges, nneighbors;
	int *index, *edges, *neighbors;
	int tmp;
	//-------------------------------------------------------------
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

	nnodes = ProcSize;								//количество узлов
	index = (int*)malloc(nnodes * sizeof(int));		//список чисел дуг, идущих от каждой вершины !!!!!!!!!!!!!!!!!!!!!!!
	edges = (int*)malloc(2 * nnodes * sizeof(int));	//последовательный список дуг
	neighbors = (int*)malloc(2 * sizeof(int));

	//заполнение списка дуг и чисел дуг вершин, n_index=числу вершин n_edges = числу дуг
	n_index = n_edges = 0;
	for (n_index; n_index < nnodes; n_index++) {
		index[n_index] = 2 * (n_index + 1);
		if (n_index != 0 && n_index != nnodes - 1) {
			edges[n_edges++] = n_index - 1;
			edges[n_edges++] = n_index + 1;
		}
		else if (n_index == 0) {
			edges[n_edges++] = nnodes - 1;
			edges[n_edges++] = n_index + 1;
		}
		else if (n_index == nnodes - 1) {
			edges[n_edges++] = n_index - 1;
			edges[n_edges++] = 0;
		}
	}

	//проверка списка дуг и чисел дуг вершин
	if (0) {
		tmp = 0;
		printf("index: ");
		for (tmp; tmp < n_index; tmp++)
			printf("%d ", index[tmp]);
		tmp = 0;
		printf(". edges: ");
		for (tmp; tmp < n_edges; tmp++)
			printf("%d ", edges[tmp]);
		printf("\n");
	}

	MPI_Graph_create(MPI_COMM_WORLD, nnodes, index, edges, 0, &comm);
	MPI_Graph_neighbors(comm, ProcRank, 2, neighbors);

	source = atoi(argv[1]);				//передатчик
	destination = atoi(argv[2]);		//приемник

//	printf("I am %d of %d. My neighbours is %d and %d\n", ProcRank, ProcSize, neighbors[0], neighbors[1]);
	data_transfer(comm, ProcSize, ProcRank, &buffer, 1, MPI_INT, &status, source, neighbors[0], neighbors[1], destination);


	//	printf("I am %d of %d. My neighbours is %d and %d\n", ProcRank, ProcSize, neighbors[0], neighbors[1]);
	//	printf("Sourse #%d, recipient #%d.\n", source, recipient);


	free(index);
	free(edges);
	free(neighbors);
	MPI_Finalize();

}
*/

//Одностороннее кольцо

void main(int argc, char *argv[]) {
	MPI_Comm comm;
	MPI_Status *status = new MPI_Status();
	int ProcSize, ProcRank, source, destination, buf;
	int nnodes, n_index, n_edges, nneighbors;
	int *index, *edges, *neighbors;

	//Параметры передачи
	source = atoi(argv[1]);
	destination = atoi(argv[2]);
	double t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;
//	buf = ProcRank;
	int count = 1;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

	//Формирование графовой топологии типа "кольцо"
	nnodes = ProcSize;								
	index = (int*)malloc(nnodes * sizeof(int));		
	edges = (int*)malloc(nnodes * sizeof(int));		
	neighbors = (int*)malloc(sizeof(int));

	n_index = n_edges = 0;
	for (n_index; n_index < nnodes; n_index++) {
		index[n_index] = n_index + 1;
		if (n_index != 0 && n_index != nnodes - 1) {
			edges[n_edges++] = n_index + 1;
		}
		else if (n_index == 0) {
			edges[n_edges++] = n_index + 1;
		}
		else if (n_index == nnodes - 1) {
			edges[n_edges++] = 0;
		}
	}

	MPI_Graph_create(MPI_COMM_WORLD, nnodes, index, edges, 0, &comm);
	MPI_Graph_neighbors(comm, ProcRank, 1, neighbors);



//	std::cout << buf;

	//Передача сообщения
/*	if (ProcRank == source) {
		printf("I'm sourse %d and i send to %d\n", ProcRank, neighbors[0]);
		MPI_Send(&buf, 1, MPI_INT, neighbors[0], 10, comm);
	}
	else if (ProcRank == destination) {
		MPI_Recv(&buf, 1, MPI_INT, MPI_ANY_SOURCE, 10, comm, &status);
		printf("I'm destination %d and i get %d\n", ProcRank, buf);
	}
	else {
		if (source < destination -1 && source < ProcRank && ProcRank < destination) {
			MPI_Recv(&buf, 1, MPI_INT, MPI_ANY_SOURCE, 10, comm, &status);
			MPI_Send(&buf, 1, MPI_INT, neighbors[0], 10, comm);
			printf("I'm %d and i send %d to %d\n", ProcRank, buf, neighbors[0]);
		}
		else if (source > destination &&  (source < ProcRank || destination > ProcRank)) {
			MPI_Recv(&buf, 1, MPI_INT, MPI_ANY_SOURCE, 10, comm, &status);
			MPI_Send(&buf, 1, MPI_INT, neighbors[0], 10, comm);
			printf("I'm %d and i send %d to %d\n", ProcRank, buf, neighbors[0]);
		}
		else 
			printf("I'm %d and i nothing to do\n", ProcRank);
	}
*/

	t1 = 0.0;
	SSend(t1, count, MPI_DOUBLE, comm, source, destination, 1);

	/********************************************/

	t2 = 0.0;
	SSend(t2, count, MPI_DOUBLE, MPI_COMM_WORLD, source, destination, 2);

	/********************************************/
/*
	MPI_Comm comm1;
	int dims[1];
	int period[1] = { 1};
	dims[0] = ProcSize;
	MPI_Cart_create(MPI_COMM_WORLD, 1, dims, period, 0, &comm1);
	t3 = 0.0;

	if (ProcRank == source)
	{
		t3 = MPI_Wtime();
		MPI_Send(&t3, 1, MPI_DOUBLE, destination, NULL, comm1);
	}
	if (ProcRank == destination)
	{
		MPI_Recv(&t3, 1, MPI_DOUBLE, MPI_ANY_SOURCE, NULL, comm1, status);
		std::cout << "Time from Cart_create.\n " << MPI_Wtime() - t3 << std::endl;
	}
*/
	/********************************************/
/*	MPI_Comm comm2;
//	int dims[1];
//	int period[1] = { 1 };
//	dims[0] = ProcSize;
	MPI_Cart_create(MPI_COMM_WORLD, 1, dims, period, 0, &comm2);
	t4 = 0.0;
	SSend(t4, count, MPI_DOUBLE, comm2, source, destination, 3);
*/


	free(index);
	free(edges);
	free(neighbors);

	MPI_Finalize();
}
