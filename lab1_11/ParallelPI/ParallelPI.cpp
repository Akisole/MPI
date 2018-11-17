//ТОПОЛОГИИ СЕТЕЙ ПЕРЕДАЧИ ДАННЫХ
//11. Кольцо
//Одностороннее кольцо

#include "mpi.h"
#include "stdio.h"
#include <stdlib.h> 

void main(int argc, char *argv[]) {
	MPI_Comm comm;
	MPI_Status status;
	int ProcSize, ProcRank, source, destination, buf;
	int nnodes, n_index, n_edges, nneighbors;
	int *index, *edges, *neighbors;

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

	//Параметры передачи
	source = atoi(argv[1]);							
	destination = atoi(argv[2]);					
	buf = ProcRank;

	//Передача сообщения
	if (ProcRank == source) {
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

	free(index);
	free(edges);
	free(neighbors);

	MPI_Finalize();
}