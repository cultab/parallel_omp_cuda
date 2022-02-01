#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#define SANITY_CHECK 0
#define PRINT 0

// 8 optimal
#define BLOCK_DIM 8
#define BLOCK_NUM 100

// matrix size
#define SIZE 15000

// defines for the elem type
typedef double elem;
#define MAXELEM DBL_MAX
#define MINELEM DBL_MIN

#ifdef __CUDA_ARCH__
#define syncthreads() __syncthreads()
#else
#define syncthreads()
#endif

inline void cudaPrintError(cudaError_t cudaerr, const char *file, int line)
{
    if (cudaerr != cudaSuccess) {
        fprintf(stderr, "CUDA error: \"%s\" in file %s at line %d.\n", cudaGetErrorString(cudaerr), file, line);
        exit(cudaerr);
    }
}

#define cudaErr(ans)                                                                                                   \
    do {                                                                                                               \
        cudaPrintError((ans), __FILE__, __LINE__);                                                                     \
    } while (0)

#define cudaLastErr()                                                                                                  \
    do {                                                                                                               \
        cudaError_t cudaerr = cudaDeviceSynchronize();                                                                 \
        cudaPrintError(cudaerr, __FILE__, __LINE__);                                                                   \
    } while (0)

__device__ double atomicMax(double* address, double val)
{
    // return atomicMax((long long*)address, __double_as_longlong(val));  // NOTE: also works
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long  old = *addr_as_ull;
    unsigned long long  assumed;
    do {
        assumed = old;
        double temp = __longlong_as_double(assumed);  //  HACK: Casting magic
        if (val > temp) {
            old = atomicCAS(addr_as_ull, assumed, *(unsigned long long*)&val);
        } else {
            break;
        }
    } while(assumed != old);
    return *((double*)&old);
}

/* convolution in host code, just to check if the cuda result is correct */
void Convolution(double* A, double* B)/*{{{*/
{
	int i, j;
	double c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;


	for (i = 1; i < SIZE - 1; ++i) {
		for (j = 1; j < SIZE - 1; ++j) {
			B[i*SIZE + j] = c11 * A[(i - 1)*SIZE + (j - 1)]  +  c12 * A[(i + 0)*SIZE + (j - 1)]  +  c13 * A[(i + 1)*SIZE + (j - 1)]
				    + c21 * A[(i - 1)*SIZE + (j + 0)]  +  c22 * A[(i + 0)*SIZE + (j + 0)]  +  c23 * A[(i + 1)*SIZE + (j + 0)] 
				    + c31 * A[(i - 1)*SIZE + (j + 1)]  +  c32 * A[(i + 0)*SIZE + (j + 1)]  +  c33 * A[(i + 1)*SIZE + (j + 1)];
		}
	}
}/*}}}*/

__global__ void convolution_max(elem *d_A,  elem *d_B, size_t height, size_t width, elem *d_max)
{
    // get starting position of thread
    int start_y = blockIdx.y * blockDim.y + threadIdx.y;
    int start_x = blockIdx.x * blockDim.x + threadIdx.x;

    // get size of each step
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    // use shared memory for a block-local minimum
    __shared__ elem local_max;

    // the thread in each block with id (0, 0) initializes the local min to whatever the minimum is
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        local_max = MINELEM;
    }
    // printf("(%dx%d): stride at %d,%d\n", start_y, start_x, stride_y, stride_x);

    // sync after initializing block local maximum
    syncthreads();

	double c11 = +0.2,  c21 = +0.5,  c31 = -0.8;
	double c12 = -0.3,  c22 = +0.6,  c32 = -0.9;
	double c13 = +0.4,  c23 = +0.7,  c33 = +0.10;

	for (int i = start_y + 1; i < height - 1; i += stride_y) {
		for (int j = start_x + 1; j < width - 1; j += stride_x) {
			d_B[i*width + j] = c11 * d_A[(i - 1)*width + (j - 1)]  +  c12 * d_A[(i + 0)*width + (j - 1)]  +  c13 * d_A[(i + 1)*width + (j - 1)]
				             + c21 * d_A[(i - 1)*width + (j + 0)]  +  c22 * d_A[(i + 0)*width + (j + 0)]  +  c23 * d_A[(i + 1)*width + (j + 0)]
				             + c31 * d_A[(i - 1)*width + (j + 1)]  +  c32 * d_A[(i + 0)*width + (j + 1)]  +  c33 * d_A[(i + 1)*width + (j + 1)];
            // if we are at the diagonal, check for a new maximum
            if (i == j) {
                atomicMax(&local_max, d_B[i*width + j]);
            }
		}
	}

    // sync after finding block local maximum
    syncthreads();

    // find the global maximum
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicMax(d_max, local_max);
    }
}

void print_mat(elem *mat, size_t height, size_t width) {
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            printf("%6.2f ", mat[i * width + j]);
        }
        printf("\n");
    }
}

int main(void)
{
    elem *d_A = NULL;
    elem *d_B = NULL;
    elem *d_max = NULL;

    size_t size = SIZE;

    // print memory used per matrix (assume double type for elem)
    long long mem = ((SIZE * SIZE * (64L / 8L)) / 1024L) / 1024L; // use long long literal
    printf("\n\tAttemping to use %lld MiB per %dx%d matrix!\n\n", mem, SIZE, SIZE);

    // allocate host memory
    elem *A = (elem*)malloc(sizeof(elem) * size * size);
    if (A == NULL) {
        fprintf(stderr, "Failed to allocate memory at line %d\n", __LINE__);
        exit(-1);
    }
    elem *B = (elem*)malloc(sizeof(elem) * size * size);
    if (B == NULL) {
        fprintf(stderr, "Failed to allocate memory at line %d\n", __LINE__);
        exit(-1);
    }
    elem *C = (elem*)malloc(sizeof(elem) * size * size);
    if (C == NULL) {
        fprintf(stderr, "Failed to allocate memory at line %d\n", __LINE__);
        exit(-1);
    }
    elem *max = (elem*)malloc(sizeof(elem));
    if (max == NULL) {
        fprintf(stderr, "Failed to allocate memory at line %d\n", __LINE__);
        exit(-1);
    }

    unsigned int seed = rand();

    // initialize matricies
    // using omp because we can
    #pragma omp parallel for schedule(auto) collapse(2)
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            A[i * size + j] = rand_r(&seed) % 100;
            // A[i * size + j] = rand() % 100;
            // initialize B to all zeros;
            B[i * size + j] = 0;
            C[i * size + j] = 0;
        }
    }

    #if PRINT+0
    puts("Matrix A");
    print_mat(A, size, size);
    #endif

    // dim3 DimGrid(256, 1, 1);
    dim3 DimBlock(BLOCK_DIM, BLOCK_DIM, 1);
    cudaLastErr();  //check for error

    // allocate device memory for matricies
    cudaErr(cudaMalloc(&d_A, size * size * sizeof(elem)));
    cudaErr(cudaMalloc(&d_B, size * size * sizeof(elem)));
    cudaErr(cudaMalloc(&d_max, sizeof(elem)));

    // copy matricies to device (yes B as well)
    cudaErr(cudaMemcpy(d_A, A, size * size * sizeof(elem), cudaMemcpyHostToDevice));
    cudaErr(cudaMemcpy(d_B, B, size * size * sizeof(elem), cudaMemcpyHostToDevice));

    convolution_max<<<BLOCK_NUM, DimBlock>>>(d_A, d_B, size, size, d_max);
    cudaLastErr();  // check for errors during kernel run

    cudaErr(cudaMemcpy(B,   d_B, size * size * sizeof(elem), cudaMemcpyDeviceToHost));
    cudaErr(cudaMemcpy(max, d_max,             sizeof(elem), cudaMemcpyDeviceToHost));

    #if PRINT+0
    puts("Matrix B");
    print_mat(B, size, size);
    #endif

    #if SANITY_CHECK/*{{{*/
    elem max2 = MINELEM;
    #pragma omp parallel for reduction(max:max2)
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            if (i == j) {
                if (max2 < B[i * size + j]) {
                    max2 = B[i * size + j];
                }
            }
        }
    }

    // check convolution using host code
    Convolution(A, C);
    puts("Matrix C");
    print_mat(C, size, size);
    printf("Diagonal's sanity max = %f\n", max2);
    #endif/*}}}*/

    printf("Diagonal's max = %f\n", *max);


    free(A);
    free(B);
    free(C);

    cudaErr(cudaFree(d_A));
    cudaErr(cudaFree(d_B));
    cudaErr(cudaFree(d_max));

    return 0;
}
