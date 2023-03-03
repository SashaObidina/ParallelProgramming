#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/time.h>
#include <pthread.h>

int size; //ðàçìåð ìàòðèöû
int threads_count; //êîëè÷åñòâî ïîòîêîâ
FILE* fp1;
FILE* fp2;

double* a;
double* b;
double* x;

struct threads_data {
    int num;
    int row_n;
};

void* pthread_func(void* td) {
    int i, j;
    double scaling;
    struct threads_data* data = (struct threads_data* ) td;
    int num = data->num;
    int row_n = data->row_n;
    for (i = row_n + num + 1; i < size; i = i + threads_count) {
        scaling = a[i*size + row_n] / a[row_n *size + row_n];
        for (j = row_n; j < size; j++) {
            a[i*size + j] -= a[row_n *size + j] * scaling;
        }
        b[i] -= scaling* b[row_n];
    }
}

int main(int argc, char* argv[]) {
    int i, j;
    long thread;
    size = 10;
    float time_use = 0;
    struct timeval start_time, end_time;
    int row_n, num, row_i, row_j;
    double scaling;

    a = (double*)malloc(sizeof(*a) * size * size); // Ìàòðèöà êîýôôèöèåíòîâ
    b = (double*)malloc(sizeof(*b) * size); // Ñòîëáåö ñâîáîäíûõ ÷ëåíîâ
    x = (double*)malloc(sizeof(*x) * size); // Íåèçâåñòíûå

    fp1 = fopen("m.txt", "r");
    for (i = 0; i < size; i++)
        for (j = 0; j < size + 1; j++)
        {
            if (j != size)
                fscanf(fp1, "%lf", &a[i * size + j]);
            else
                fscanf(fp1, "%lf", &b[i]);
        }
    fclose(fp1);

    //Çàïîëíÿåì âåêòîð x
    for (i = 0; i < size; i++) {
        x[i] = 0;
    }

    gettimeofday(&start_time, NULL);

    threads_count = strtol(argv[1], NULL, 10);
    pthread_t* thread_handles;
    thread_handles = (pthread_t*) malloc(threads_count * sizeof(pthread_t));
    
    // Elimination 
    for (row_n = 0; row_n < size - 1; row_n++) {
        struct threads_data* tdata = (struct threads_data*) malloc(threads_count * sizeof(struct threads_data));

        for (thread = 0; thread < threads_count; thread++) {
            tdata[thread].row_n = row_n;
            tdata[thread].num = thread;
            pthread_create(&thread_handles[thread], NULL, pthread_func, (void*)&tdata[thread]);
        }

        for (thread = 0; thread < threads_count; thread++) 
            pthread_join(thread_handles[thread], NULL);

        free(tdata);
    }

    // Back substitution (with diagonal' elements normalization)
    for (row_i = size - 1; row_i >= 0; row_i--) {
        x[row_i] = b[row_i];
        for (row_j = size - 1; row_j > row_i; row_j--) {
            x[row_i] -= a[row_i*size + row_j] * x[row_j];
        }
        x[row_i] /= a[row_i*size + row_i];
    }
   
    gettimeofday(&end_time, NULL);

    //çàïèñü ðåçóëüòàòîâ â ôàéë
    fp2 = fopen("x_pthreads.txt", "w");
    for (i = 0; i < size; i++) {
        fprintf(fp2, "%lf ", x[i]);
    }
    fclose(fp2);
    
    time_use = (end_time.tv_sec - start_time.tv_sec) * 1000000 + (end_time.tv_usec - start_time.tv_usec);

    printf("Time:%lfs\n", time_use / 1000000);

    printf("Linux pthreads: size of matrice %d, threads %d \n", size, threads_count);

    free(a);
    free(b);
    free(x);
    return 0;
}
