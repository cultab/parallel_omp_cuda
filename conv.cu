#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

// check for diagnoal's max in host code
#define SANITY_CHECK

// 8 optimal
static int BLOCK_DIM = 4;

static int SIZE = 200;

typedef double elem;
#define MAXELEM MAXFLOAT

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
        double temp = __longlong_as_double(assumed);  //  HACK: WHY IS THIS NECESSARY
        if (val > temp) {
            old = atomicCAS(addr_as_ull, assumed, *(unsigned long long*)&val);
        } else {
            break;
        }
    } while(assumed != old);
    return *((double*)&old);
}

__global__ void convolution_max(elem *d_A,  elem *d_B, size_t height, size_t width, elem *d_max)
{
    int start_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_y = blockDim.y * gridDim.y;

    int start_x = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;

    __shared__ elem local_max;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        local_max = -MAXELEM;
    }
    // printf("(%dx%d): stride at %d,%d\n", start_y, start_x, stride_y, stride_x);

	double c11 = +0.2,  c21 = +0.5,  c31 = -0.8;

	double c12 = -0.3,  c22 = +0.6,  c32 = -0.9;
	double c13 = +0.4,  c23 = +0.7,  c33 = +0.10;

	for (int i = start_y + 1; i < height - 1; i += stride_y) {
		for (int j = start_x + 1; j < width - 1; j += stride_x) {
			d_B[i*width + j] = c11 * d_A[(i - 1)*width + (j - 1)]  +  c12 * d_A[(i + 0)*width + (j - 1)]  +  c13 * d_A[(i + 1)*width + (j - 1)]
				              + c21 * d_A[(i - 1)*width + (j + 0)]  +  c22 * d_A[(i + 0)*width + (j + 0)]  +  c23 * d_A[(i + 1)*width + (j + 0)]
				              + c31 * d_A[(i - 1)*width + (j + 1)]  +  c32 * d_A[(i + 0)*width + (j + 1)]  +  c33 * d_A[(i + 1)*width + (j + 1)];
            // printf("(%dx%d): d_B[%d * %lu + %d] = d_B[%lu] modified;\n", start_y, start_x, i, width, j, i * width + j);
            // printf("(%dx%d): element at %d,%d\n", start_y, start_x, i, j);
            // d_B[i * width + j] = 37;
            if (i == j) {
                // atomicMaxDouble(&local_max, d_B[i*width + j]);
                atomicMax(&local_max, d_B[i*width + j]);
            }
		}
	}

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // atomicMaxDouble(d_max, local_max);
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
    elem *A = (elem*)malloc(sizeof(elem) * size * size);
    elem *B = (elem*)malloc(sizeof(elem) * size * size);
    elem *max = (elem*)malloc(sizeof(elem));

    unsigned int seed = rand();

    // because we can
    // #pragma omp parallel for schedule(auto) collapse(2)
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            // A[i * size + j] = rand_r(&seed) % 100;
            A[i * size + j] = rand() % 100;
            // initialize B to all zeros;
            B[i * size + j] = 0;
        }
    }

    // puts("Matrix A");
    // print_mat(A, size, size);
    // puts("Matrix B");
    // print_mat(B, size, size);

    // dim3 DimGrid(256, 1, 1);
    dim3 DimBlock(BLOCK_DIM, BLOCK_DIM, 1);
    cudaLastErr();  //check for error

    // allocate device memory for matricies
    cudaErr(cudaMalloc((void**)&d_A, size * size * sizeof(elem)));
    cudaErr(cudaMalloc((void**)&d_B, size * size * sizeof(elem)));
    cudaErr(cudaMalloc((void**)&d_max, sizeof(elem)));

    // copy matricies to device (yes B as well)
    cudaErr(cudaMemcpy(d_A, A, size * size * sizeof(elem), cudaMemcpyHostToDevice));
    cudaErr(cudaMemcpy(d_B, B, size * size * sizeof(elem), cudaMemcpyHostToDevice));

    convolution_max<<<20, DimBlock>>>(d_A, d_B, size, size, d_max);
    cudaLastErr();  // check for errors during kernel run

    cudaErr(cudaMemcpy(B,   d_B, size * size * sizeof(elem), cudaMemcpyDeviceToHost));
    cudaErr(cudaMemcpy(max, d_max,             sizeof(elem), cudaMemcpyDeviceToHost));

    puts("Matrix B");
    // print_mat(B, size, size);

    #ifdef SANITY_CHECK
    elem max2 = -MAXELEM;
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
    printf("Diagonal's sanity max = %f\n", max2);
    #endif


    printf("Diagonal's max = %f\n", *max);
}
