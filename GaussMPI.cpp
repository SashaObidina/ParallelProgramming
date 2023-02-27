#define _CRT_SECURE_NO_WARNINGS
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int size; //размер матрицы
FILE* fp1;
FILE* fp2;

int main(int argc, char* argv[])
{
	int size = 10; 
	int rank, numprocs;
	double* m = (double*)malloc(sizeof(*m) * (size) * (size + 1));
	int i, j;

	fp1 = fopen("m.txt", "r");
	for (i = 0; i < size; i++) {
		for (j = 0; j < size + 1; j++)
		{
			fscanf(fp1, "%lf", &m[i * (size + 1) + j]);
			//printf("%lf", m[i * (size + 1) + j]);
		}
		//printf("\n");
	}
	fclose(fp1);

	MPI_Init(NULL, NULL);
	double time = MPI_Wtime();
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); //уникальный номер - ранг процесса
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs); //общее количество доступных процессов

	//количество строк для процесса
	int tmp = size / numprocs;
	if (size % numprocs) 
		tmp++; 
	int rest = numprocs * tmp - size;
	int numrows = tmp;
	if (rank >= numprocs - rest) //случай последнего процесса, которому достается остаток строк
		numrows = tmp - 1;

	int* rows_number = (int*)malloc(sizeof(*rows_number) * numrows); //номера строк, которые должен передать оставшимся процессам конкретный процесс

	double* a = (double*)malloc(sizeof(*a) * numrows * (size + 1)); //коэффициенты со столбцом свободных членов
	double* x = (double*)malloc(sizeof(*x) * size); //решения системы
	double* t = (double*)malloc(sizeof(*t) * (size + 1));

	for (i = 0; i < numrows; i++) {
		rows_number[i] = rank + numprocs * i; //заносим номера строк матрицы
		//printf("num: %d; ", rows_number[i]);
		for (j = 0; j < size + 1; j++) {
			a[i * (size + 1) + j] = m[rows_number[i] * (size + 1) + j];
			//printf("%lf", a[rows_number[i] * (size + 1) + j]);
		}
		//a[i * (size + 1) + size] = m[rows_number[i] * (size + 1) + (size)];
		//printf("%lf", a[rows_number[i] * (size + 1) + size]);
		//printf("\n");
	}

	//Elimination
	int r = 0;
	for (i = 0; i < size - 1; i++) {
		// Убираем из последующих уравнений xi
		if (i == rows_number[r]) {
			// Посылаем в последующие уравнения строку i 
			MPI_Bcast(&a[r * (size + 1)], size + 1, MPI_DOUBLE, rank, MPI_COMM_WORLD);
			for (j = 0; j <= size; j++)
				t[j] = a[r * (size + 1) + j];
			r++;
		}
		else {
			MPI_Bcast(t, size + 1, MPI_DOUBLE, i % numprocs, MPI_COMM_WORLD);
		}
		// Принятая строка вычитается из тех уравнений, за которые отвечает данный процесс
		for (j = r; j < numrows; j++) {
			double scaling = a[j * (size + 1) + i] / t[i];
			for (int k = i; k < size + 1; k++)
				a[j * (size + 1) + k] -= scaling * t[k];
		}
	}

	//Заполняем вектор x
	r = 0;
	for (i = 0; i < size; i++) {
		x[i] = 0;
		if (i == rows_number[r]) {
			x[i] = a[r * (size + 1) + size];
			r++;
		}
	}
	//Back substitution
	r = numrows - 1;
	for (i = size - 1; i > 0; i--) {
		if (r >= 0) {
			if (i == rows_number[r]) {
				x[i] /= a[r * (size + 1) + i];
				// Передаем найденное x_i
				MPI_Bcast(&x[i], 1, MPI_DOUBLE, rank, MPI_COMM_WORLD);
				r--;
			}
			else
				MPI_Bcast(&x[i], 1, MPI_DOUBLE, i % numprocs, MPI_COMM_WORLD);
		}
		else
			MPI_Bcast(&x[i], 1, MPI_DOUBLE, i % numprocs, MPI_COMM_WORLD);
		for (int j = 0; j <= r; j++)
			x[rows_number[j]] -= a[j * (size + 1) + i] * x[i];
	}
	if (rank == 0)
		x[0] /= a[r * (size + 1)];
	MPI_Bcast(x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); //все процессы содержат вектор решений
	free(t);
	free(rows_number);
	free(a);
	time = MPI_Wtime() - time;
	
	//запись результатов в файл
	fp2 = fopen("x_mpi_c.txt", "w");
	for (i = 0; i < size; i++) {
		fprintf(fp2, "%lf ", x[i]);
	}
	fclose(fp2);

	if (rank == 0) {
		printf("Time: %.6fs\n", time);
		printf("MPI/C: size of matrice %d, processes %d\n", size, numprocs);
	}

	free(x);
	MPI_Finalize();
	return 0;
}