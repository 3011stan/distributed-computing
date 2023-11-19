#include <mpi.h>

int world_rank, world_size;
MPI_Comm custom_comm1, custom_comm2, custom_comm3, tmp;

void splitting() {
  int color;
  MPI_Comm new_comm;
  
  // 1- First splitting here.
  // With only one call to MPI_Comm_split you should be able to split processes 0-3 in custom_comm1
  // and processes 4-6 in custom_comm2
  if (world_rank < 4) { // 0, 1, 2, 3 processors (custom_comm1)
    color = 0;
  } else if (world_rank < 7) { // 4, 5, 6 processors (custom_comm2)
    color = 1;
  } else {
    color = MPI_UNDEFINED;
  }

  // split / criar comunicadores de acordo com a cor especificada
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &new_comm);

  // atribui o novo comunicador ao comunicador global
  if (color == 0) {
    custom_comm1 = new_comm;
  } else if (color == 1) {
    custom_comm2 = new_comm;
  }

  // 2- Second splitting here

	// add os processos 0 e 4 no custom_comm3
  if (world_rank == 0 || world_rank == 4) {
    color = 0;
  } else {
    color = MPI_UNDEFINED;
  }
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &new_comm);
  if (color == 0) {
    custom_comm3 = new_comm;
  }
}