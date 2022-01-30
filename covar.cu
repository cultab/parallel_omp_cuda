#include <cooperative_groups.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#define PRINT 0
#define SIMPLE 0

// matrix size
#define SIZE_Y 2000
#define SIZE_X 2000


#define BLOCK_NUM 128
#define THREADS_NUM 128

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

__host__ __device__ void print_mat(elem *mat, size_t height, size_t width, const char* name) {
    printf("%s (%ldx%ld):\n", name, height ,width);
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            printf("%12.2f ", mat[i * width + j]);
        }
        printf("\n");
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

__global__ void col_average_distance_matrix(elem *d_A, size_t size_x, size_t size_y)
{
    elem col_average;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("Thread(%d)\n", tid);

    int stride = gridDim.x * blockDim.x;

    // if (tid == 0) {
    //     printf("gridDim %d\n", gridDim.x);
    // }

    // printf("Thread(%d): \n", threadId);

    for (int i = tid; i < size_x; i += stride) {
        col_average = d_A[0 * size_x + i];

        for (int j = 1; j < size_y; j++) {
            col_average += d_A[j * size_x + i];
        }

        // printf("Thread(%d): local [%d] = %f\n", thread_id, i, col_average);

        col_average = col_average / (elem)size_y;

        for (int j = 0; j < size_y; j++) {
            d_A[j * size_x + i] -= col_average;
        }
    }

}

__global__ void something(elem* d_A, elem* d_B, elem* d_Res, size_t size_y, size_t size_x)
{
    int start_x = threadIdx.x + blockIdx.x * blockDim.x;
    int start_y = threadIdx.y + blockIdx.y * blockDim.y;

    d_Res[start_y * size_x + start_x] = -35;
}


/*
  ##################
  # REALLY SLOW :( #
  ##################
*/
__global__ void matrix_mul(elem* d_A, elem* d_B, elem* d_Res, size_t row_A, size_t col_B, size_t com_col_A)
{
    int start_x = threadIdx.x + blockIdx.x * blockDim.x; 
    int start_y = threadIdx.y + blockIdx.y * blockDim.y; 

    int stride_x = gridDim.x;
    int stride_y = gridDim.y;

    // printf("tid(%d,%d): hre!\n", start_x, start_y);
    elem Pvalue = 0;
    // each thread computes several elements of the output matrix
    for (int y = start_y; y < row_A; y += stride_y) {
        for (int x = start_x; x < col_B; x += stride_x) {
            Pvalue = 0;

            for (int k = 0; k < com_col_A; ++k) {
                // printf("Read from  A[%d][%d]\n"
                //        "Read from At[%d][%d]\n", y, k, k , y);
                Pvalue += d_A[y * com_col_A + k] * d_B[k * col_B + x];
            }

            // write back to the global memory
            d_Res[y* col_B + x] = Pvalue;
            // printf("Wrote %f to B[%d][%d]\n", Pvalue , y, x);
        }
    }
}
    // if (x == 0 && y == 0)
    // printf("%6.2f += %6.2f * %6.2f\n", Pvalue, d_A[y * com_col_A + k], d_B[k * col_B + x]);
    // printf("--------------------\n");

__global__ void matrix_transpose(elem* d_mat, elem* d_result, size_t size_y, size_t size_x)
{
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;

    int row_stride = gridDim.x;
    int col_stride = blockDim.x;

    // printf("Block(%d),Thread(%d): here!\n", block_id, thread_id);

    for (int i = block_id; i < size_y; i += row_stride) {
        for (int j = thread_id; j < size_x; j += col_stride) {
            d_result[j * size_y + i] = d_mat[i * size_x + j];
        }
    }
}

int main(void)
{
    size_t size_y = SIZE_Y;
    size_t size_x = SIZE_X;

    elem *A;
    elem *d_A;

    elem *At;
    elem *d_At;

    elem *B;
    elem *d_B;

    A = (elem*)malloc(size_y * size_x * sizeof(elem));
    if (A == NULL) {
        fprintf(stderr, "Failed to allocate memory at line %d\n", __LINE__);
        exit(-1);
    }
    At = (elem*)malloc(size_x * size_y * sizeof(elem));
    if (At == NULL) {
        fprintf(stderr, "Failed to allocate memory at line %d\n", __LINE__);
        exit(-1);
    }
    B = (elem*)malloc(size_y * size_y * sizeof(elem));
    if (B == NULL) {
        fprintf(stderr, "Failed to allocate memory at line %d\n", __LINE__);
        exit(-1);
    }

    cudaEvent_t start, stop;
    float   elapsedTime;

    cudaErr(cudaEventCreate(&start));
    cudaErr(cudaEventCreate(&stop));


    cudaErr(cudaMalloc(&d_A, size_y * size_x * sizeof(elem)));
    cudaErr(cudaMalloc(&d_At, size_x * size_y * sizeof(elem)));
    cudaErr(cudaMalloc(&d_B, size_y * size_y * sizeof(elem)));

    // initialize matrix A
    for (size_t i = 0; i < size_y; i++) {
        for (size_t j = 0; j < size_x; j++) {
            #if SIMPLE+0
                // A[i * size_x + j] = 10 * i + j;
                A[i * size_x + j] = i * j;
            #else
                A[i * size_x + j] = rand() % 10;
            #endif
            // A[i * size_x + j] = i * j;
        }
    }

    #if PRINT+0
    print_mat(A, size_y, size_x, "A");
    #endif

    cudaErr(cudaMemcpy(d_A, A, size_y * size_x * sizeof(elem), cudaMemcpyHostToDevice));
    cudaErr(cudaMemcpy(d_At, A, size_y * size_x * sizeof(elem), cudaMemcpyHostToDevice));

    // start clock
    cudaErr(cudaEventRecord(start, 0));

    col_average_distance_matrix<<<BLOCK_NUM, THREADS_NUM>>>(d_A, size_x, size_y);
    cudaLastErr();

    // stop clock
    cudaErr(cudaEventRecord(stop, 0));
    cudaErr(cudaEventSynchronize(stop));

    cudaErr(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time of first kernel:  %3.1f ms\n", elapsedTime);

    #if PRINT+0
    // copy A back to host (only to print)
    cudaErr(cudaMemcpy(A, d_A, size_y * size_x * sizeof(elem), cudaMemcpyDeviceToHost));
    #endif

    matrix_transpose<<<BLOCK_NUM, THREADS_NUM>>>(d_A, d_At, size_y, size_x);
    cudaLastErr();

    //   32x32 =  1024  threads per block is the max
    // 256x256 = 65536 blocks per grid is the max
    // dim3 dimBlock(32, 32);
    // dim3 dimGrid(256, 256);
    // dim3 dimBlock(1, 1);
    // dim3 dimGrid(1, 1);
    dim3 dimBlock(16, 16);
    dim3 dimGrid(256, 256);

    // start clock
    cudaErr(cudaEventRecord(start, 0));

    matrix_mul<<<dimGrid, dimBlock>>>(d_A, d_At, d_B, size_y, size_y, size_x);
    cudaLastErr();

    // stop clock
    cudaErr(cudaEventRecord(stop, 0));
    cudaErr(cudaEventSynchronize(stop));

    cudaErr(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time of second kernel:  %3.1f ms\n", elapsedTime);

    // puts("Start copying");
    cudaErr(cudaMemcpy(B, d_B, size_y * size_y * sizeof(elem), cudaMemcpyDeviceToHost));
    // puts("End copying");

    #if PRINT+0
    // copy At back to host (only to print)
    cudaErr(cudaMemcpy(At, d_At, size_y * size_x * sizeof(elem), cudaMemcpyDeviceToHost));

    print_mat(A, size_y, size_x, "A - average of column");
    print_mat(At, size_x, size_y, "Transpose");
    print_mat(B, size_y, size_y, "Result");
    #endif


    return 0;
}



// __global__ void MatrixMulKernel(elem* Md, elem* Nd, elem* Pd, size_t Width){{{
// {
//     printf("jere\n");
//     // declare cache in the shared memory
//     __shared__ elem Mds[BLOCK_DIM][BLOCK_DIM];
//     __shared__ elem Nds[BLOCK_DIM][BLOCK_DIM];
//
//     // keep track of column index of the Pd element using thread index
//     int x = threadIdx.x + blockIdx.x * blockDim.x; // x is column
//     // keep track of row index of the Pd element using thread index
//     int y = threadIdx.y + blockIdx.y * blockDim.y; // y is row
//
//     printf("Thread(%d,%d): here!\n", y, x);
//
//     // optimization ???
//     if (x < y) {
//         return;
//     }
//
//     elem Pvalue = 0;
//     // Loop over the Md and Nd block dimension required to compute the Pd element
//     for (int m = 0; m < Width/BLOCK_DIM; m++) {
//
//         // collaboratively loading of Md and Nd blocks into shared memory	 
//         Mds[threadIdx.y][threadIdx.x] = Md[y * Width + (m * BLOCK_DIM + threadIdx.x)];
//         Nds[threadIdx.y][threadIdx.x] = Md[(m * BLOCK_DIM + threadIdx.y) * Width + x];
//         syncthreads();
//
//         // keep track of the running sum
//         for (int k = 0; k < BLOCK_DIM; k++) {
//             Pvalue += Mds[threadIdx.y][k] * Nds[k][threadIdx.x];
//         }
//         syncthreads();
//     }
//
//     // write back to the global memory
//     printf("Thread(%d,%d): Pvalue = %f\n", y, x, Pvalue);
//     Pd[y * Width + x] = Pvalue;
// }}}}




/*

    */
