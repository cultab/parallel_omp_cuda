#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#define LIMIT 4
#define SIZE 100

static int * GLOBAL_array;
static int * GLOBAL_space;

static unsigned int seed;

void print_array(int *, int);

void ARRAY();
void SPACE();

void sort(int *, int *);

void merge(int *, const int *, int *, const int *, int *);
void multisort(int *, int *, size_t);

void quicksort(int *array, size_t low, size_t high);
size_t partition(int *array, size_t low, size_t high);

// print @array of @size
void print_array(int *array, int size)
{
    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }
    puts("");
}

void ARRAY()
{
    printf("\tARRAY: ");
    print_array(GLOBAL_array, SIZE);
}

void SPACE()
{
    printf("\tSPACE: ");
    print_array(GLOBAL_space, SIZE);
}

// sort arrays of length 2 :^)
void sort(int *start, int *end)
{
    if (start[0] > start[1]) {
        int tmp  = start[0];
        start[0] = start[1];
        start[1] = tmp;
    }
}

/*
 * Take elements from x and y,
 * place them in space,
 * until x and y both reach their end.
 */
void merge(int *x, const int *x_end, int *y, const int *y_end, int *space)
{
    // until x or y reach their end
    while (x != x_end && y != y_end) {
        if (*x < *y) {
            *(space++) = *(x++);
        } else {
            *(space++) = *(y++);
        }
    }

    // copy remaining from x
    while (x != x_end) {
        *(space++) = *(x++);
    }

    // copy remaining from y
    while (y != y_end) {
        *(space++) = *(y++);
    }
}

void multisort(int *start, int *space, size_t size)
{
    /* printf("multisort size = %d\n", size); */
    if (size <= LIMIT) {
        /* qsort does NOT exist! */
        quicksort(start, 0, size - 1);
        return;
    }
    int quarter = size / 4;

    int *startA = start;
    int *spaceA = space;

    int *startB = startA + quarter;
    int *spaceB = spaceA + quarter;

    int *startC = startB + quarter;
    int *spaceC = spaceB + quarter;

    int *startD = startC + quarter;
    int *spaceD = spaceC + quarter;

    #pragma omp task
    multisort(startA, spaceA, quarter);

    #pragma omp task
    multisort(startB, spaceB, quarter);

    #pragma omp task
    multisort(startC, spaceC, quarter);

    #pragma omp task
    multisort(startD, spaceD, size - 3 * quarter);

    #pragma omp taskwait

    #pragma omp task
    merge(startA, startA + quarter, startB, startB + quarter, spaceA);

    #pragma omp task
    merge(startC, startC + quarter, startD, start + size, spaceC);

    #pragma omp taskwait

    merge(spaceA, spaceC, spaceC, spaceA + size, startA);
}

/*
 * swap values pointed to by a and b
 */
void swap(int *a, int *b)
{
    int tmp;
    tmp = *a;
    *a = *b;
    *b = tmp;
}

size_t partition(int *array, size_t low, size_t high)
{
    size_t i = low - 1;
    size_t j = high + 1;

    int pivot = array[(j + i) / 2]; // The value in the middle of the array

    do {
        do i++; while(array[i] < pivot); // until we find an element smaller than pivot
        do j--; while(array[j] > pivot); // until we find an element greater than pivot

        if (i >= j) return j; // if the indexes passed by eachother we are done, so return the pivot which is now j

        swap(&array[i], &array[j]); // swap the smaller than pivot element with the greater than pivot element
    } while(true);

    // unreached code :/
}

void quicksort(int *array, size_t low, size_t high)
{
    if (low < high) {
        size_t pivot = partition(array, low, high);
        quicksort(array, low, pivot);
        quicksort(array, pivot + 1, high);
    }
}

int main(void)
{
    // init a seed value for rand_r()
    seed = rand();

    // allocate memory for array and space
    int *array = malloc(SIZE * sizeof(int));
    int *space = malloc(SIZE * sizeof(int));

    // copy pointers to globals for easy debuging
    GLOBAL_array = array;
    GLOBAL_space = space;

    // init space and array
    #pragma omp parallel for schedule(auto)
    for (int i = 0; i < SIZE; i++) {
        array[i] = rand_r(&seed) % 100;
        space[i] = 0;
    }
    puts("Original: ");
    print_array(array, SIZE);

    // is this necessary?
    #pragma omp parallel
    {
        #pragma omp single
        multisort(array, space, SIZE);
    }

    puts("Result: ");
    print_array(array, SIZE);
}
