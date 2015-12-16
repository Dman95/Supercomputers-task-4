#ifndef MATRIX_LIB_H
#define MATRIX_LIB_H

#include "mpi.h"

//vector operations
typedef struct vector
{
    long long size;
    double *values;
} vector;

vector *vector_new(long long size);
vector *vector_new_with_memory(long long size, void *memory);
vector *vector_gen(long long size);
vector *vector_load(char *filename);
vector *vector_load_fp(MPI_File *f);
vector *vector_load_part_fp_with_memory(MPI_File *f, long long which, long long from_how_much, void *memory);
void vector_save_part_fp(vector *v, MPI_File *f, long long which, long long from_how_much, long long size);
void vector_save(vector *v, char *filename);
void vector_save_fp(vector *v, MPI_File *f);
void vector_print(vector *v);
void vector_delete(vector *v);


//matrix operations
typedef struct matrix
{
    long long row_count;
    long long column_count;
    vector **rows;
} matrix;

matrix *matrix_new(long long row_count, long long column_count);
matrix *matrix_new_without_vector_init(long long row_count, long long column_count);
matrix *matrix_gen(long long row_count, long long column_count);
matrix *matrix_load(char *filename);
matrix *matrix_load_part(char *filename, long long which, long long from_how_much);
matrix *matrix_load_block_with_memory(char *filename, long long i, long long j, long long i_dim_n_cuts, long long j_dim_n_cuts, void *memory);
void matrix_save_block(matrix *m, char *filename, long long i, long long j, long long row_count, long long column_count);
void matrix_get_size(char *filename, long long *row_count, long long *column_count);
MPI_File matrix_get_file_started_from_part(char *filename, long long which, long long from_how_much, long long *needed_to_read_row_count, long long *column_count);
MPI_File matrix_get_file_started_from_part_world(char *filename, long long which, long long from_how_much, long long *needed_to_read_row_count, long long *column_count);
void matrix_save(matrix *m, char *filename);
void matrix_gen_and_save(long long row_count, long long column_count, char *filename);
void mpi_matrix_gen_and_save(long long row_count, long long column_count, char *filename);
void matrix_print(matrix *m);
void matrix_delete(matrix *m);
void matrix_delete_with_memory(matrix *m, void *memory);

//multiplication operations
double dot(vector *v1, vector *v2);
vector *multiply(matrix *m, vector *v);
vector *mpi_multiply(char *matrixname, char *vectorname);
long long count_part(long long which, long long from_how_much, long long from_what);
void mpi_multiply_and_save_matrix(char *A_matrixname, char *B_matrixname, char *resultname);
MPI_File get_file_with_offset_for_write(char *filename, long long offset);
void sleep(int sec);
void mpi_cannon_multiply_and_save_matrix(char *A_matrixname, char *B_matrixname, char *resultname);
void multiply_matrix_with_addition(matrix *A, matrix *B, matrix *C);

typedef struct { 
    int grid_one_dim; 
    int my_row;
    int my_col;
    MPI_Comm row_comm;
    MPI_Comm col_comm;
    int uprank;
    int downrank;
} grid_info;
void setup_grid(grid_info *grid); 
void mpi_fox_multiply_and_save_matrix(char *A_matrixname, char *B_matrixname, char *resultname);

#endif

