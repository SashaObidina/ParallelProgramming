#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stack>
#include <stdio.h>
#include <stdlib.h>

int size; //размер матрицы
int threads_count; //количество потоков
FILE* fp1;
FILE* fp2;

double* a;
double* b;
double* x;

int main(int argc, char* argv[]) {
    int k, l;
    int i, j;
    size = 10;
    float time_use = 0;
    int row_n, num, row_i, row_j;
    double scaling;
    int barrier = 1;

    threads_count = strtol(argv[1], NULL, 10);

    a = (double*)malloc(sizeof(*a) * size * size); // Матрица коэффициентов
    b = (double*)malloc(sizeof(*b) * size); // Столбец свободных членов
    x = (double*)malloc(sizeof(*x) * size); // Неизвестные

    fp1 = fopen("m.txt", "r");
    for (k = 0; k < size; k++)
        for (l = 0; l < size + 1; l++)
        {
            if (l != size)
                fscanf(fp1, "%lf", &a[k * size + l]);
            else
                fscanf(fp1, "%lf", &b[k]);
        }
    fclose(fp1);

    //Заполняем вектор x
    for (k = 0; k < size; k++) {
        x[k] = 0;
    }

	omp_set_num_threads(threads_count);
    double time = omp_get_wtime();

    // Elimination 
    for (row_n = 0; row_n < size - 1; row_n++) {
	#pragma omp parallel for shared(a, b) private(i, j, scaling)
        for (i = row_n + 1; i < size; i++) {
            scaling = a[i * size + row_n] / a[row_n * size + row_n];
            for (j = row_n; j < size; j++) {
                a[i * size + j] -= a[row_n * size + j] * scaling;
            }
            b[i] -= scaling * b[row_n];
        }
        
    }

    // Back substitution (with diagonal' elements normalization)
    for (row_i = size - 1; row_i >= 0; row_i--) {
        barrier = 0;
		#pragma omp barrier
		//#pragma omp parallel for shared(a, b, x) private(row_i, row_j)
        x[row_i] = b[row_i];
		//#pragma omp parallel for shared(a, b, x) private(row_i, row_j)
        for (row_j = size - 1; row_j > row_i; row_j--) {
            x[row_i] -= a[row_i * size + row_j] * x[row_j];
        }
        x[row_i] /= a[row_i * size + row_i];
    }


    time = omp_get_wtime() - time;

    //запись результатов в файл
    fp2 = fopen("x_openmp.txt", "w");
    for (i = 0; i < size; i++) {
        fprintf(fp2, "%lf ", x[i]);
    }
    fclose(fp2);

	printf("Time: %.6fs\n", time);
    printf("OpenMP/C: size of matrice %d, threads %d\n", size, threads_count);

    free(a);
    free(b);
    free(x);
    return 0;
}