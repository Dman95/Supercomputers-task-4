#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "lib/matrix_lib.h"
#include <string.h>
#include "lib/sparse_lib.h"

int main(int argc, char **argv)
{
    char A_matrixname[100];
    char B_matrixname[100];
    char resultname[100];
    
    MPI_Init(&argc, &argv);
    double start = -MPI_Wtime();

    sprintf(A_matrixname, "data/%s", argv[1]);
    sprintf(B_matrixname, "data/%s", argv[2]);
    sprintf(resultname, "results/mul%son%snproc%sfox", argv[1], argv[2], argv[3]);

    if (!strcmp(argv[4], "--fox")) {
        mpi_fox_multiply_and_save_matrix(A_matrixname, B_matrixname, resultname);
    } else {
        mpi_cannon_multiply_and_save_matrix(A_matrixname, B_matrixname, resultname);
    }

    double time_elapsed = start + MPI_Wtime();
    printf("Time: %f\n", time_elapsed);

    MPI_Finalize();
    return 0;
}
