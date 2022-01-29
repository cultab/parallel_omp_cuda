#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#define PRINT 0
#define SIMPLE 0

// matrix size
#define SIZE_Y 10000
#define SIZE_X 10000
#define BLOCK_NUM 128
#define THREADS_NUM 32

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
            printf("%6.2f ", mat[i * width + j]);
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


// each block calculates several rows of the result,
// a row is only ever calculated by a single block
__global__ void col_average_distance_matrix(elem *d_A, elem *d_At, size_t size_x, size_t size_y)
{
    __shared__ elem col_average;

    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;

    int col_stride = gridDim.x;
    int row_stride = blockDim.x;

    // printf("Block(%d),Thread(%d): \n", blockId, threadId);

    for (int i = block_id; i < size_x; i += col_stride) {

        elem local_col_sum = 0;

        if (thread_id == 0) {
            col_average = 0;
        }

        for (int j = thread_id; j < size_y; j += row_stride) {
            local_col_sum += d_At[i * size_y + j];
        }

        // printf("Block(%d),Thread(%d): local [%d] = %f\n", block_id, thread_id, i, local_col_sum);

        local_col_sum = local_col_sum / (elem)size_y;

        atomicAdd(&col_average, local_col_sum);

        syncthreads();

        // if (thread_id == 0) {
        //     printf("Block(%d),Thread(%d): avg[%d] = %f\n", block_id, thread_id, i, col_average);
        // }


        for (int j = thread_id; j < size_y; j += row_stride) {
            // printf("Block(%d),Thread(%d): A[%d][%d](%6.2f) -= %6.2f = %6.2f in [%d][%d]\n",
            //     block_id,
            //     thread_id,
            //     i, j,
            //     d_At[i * size_y + j],
            //     col_average,
            //     d_At[i * size_y + j] - col_average,
            //     j, i
            // );

            // d_A[j * size_x + i] = d_At[i * size_y + j] - col_average;
            d_A[i * size_y + j] = j * i;
        }
    }

}

__global__ void FindAverage (elem* A)
{
    int blockId = gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + blockDim.x + threadIdx.x;

    elem sum;

    for(int i = threadId; i < SIZE_X; i+= blockDim.x * gridDim.x)
    {
        sum = 0;
        for(int j = 0; j<SIZE_Y;j++)
        {
            sum += A[j * SIZE_X + i];
        }

        for(int j = 0; j<SIZE_Y;j++)
        {
            A[j * SIZE_X + i] -= sum/SIZE_Y; 
        }

    }
}

__global__ void matrix_mul(elem* d_A, elem* d_B, elem* d_Res, size_t size_y, size_t size_x)
{
    __shared__ elem block_result;

    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;

    int row_stride = gridDim.x;
    int col_stride = blockDim.x;

    for (int i = block_id; i < size_y; i += row_stride) {

        elem thread_result = 0;

        if (thread_id == 0) {
            block_result = 0;
        }

        // printf("Block(%d),Thread(%d): here\n", blockId, threadId);
        for (int j = thread_id; j < size_x; j += col_stride) {
                // printf("Block(%d),Thread(%d): vec[%d] mat[%d][%d]\n", blockId, threadId, j, i, j);
                thread_result += d_B[j] * d_A[i * size_x + j];
                // printf("Block(%d),Thread(%d): local_res += %f \n", blockId, threadId, tmp);
        }

        // wait for all threads to finish calculating their result
        syncthreads();

        // all threads add their local results
        atomicAdd(&block_result, thread_result);

        // wait for all threads to add their local result
        syncthreads();

        // one thread adds they the block's result to global memory
        // non-atomically since each element gets calculated by exactly one block
        if (thread_id == 0) {
            // printf("Block(%d): d_res[%d] += %f\n", blockId, i, local_res);
            d_Res[i] = block_result;
        }
    }
}

__global__ void matrix_trasnpose(elem* d_mat, elem* d_result, size_t size_y, size_t size_x)
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
    cudaErr(cudaEventCreate(&start));
    cudaErr(cudaEventCreate(&stop));


    cudaErr(cudaMalloc(&d_A, size_y * size_x * sizeof(elem)));
    cudaErr(cudaMalloc(&d_At, size_x * size_y * sizeof(elem)));
    // cudaErr(cudaMalloc(&d_B, size_y * size_y * sizeof(elem)));

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

    // transpose A
    matrix_trasnpose<<<BLOCK_NUM, THREADS_NUM>>>(d_A, d_At, size_y, size_x);
    cudaLastErr();

    cudaErr(cudaEventRecord(start, 0));
    col_average_distance_matrix<<<BLOCK_NUM, THREADS_NUM>>>(d_A, d_At, size_x, size_y);
    cudaLastErr();

    cudaErr(cudaEventRecord(stop, 0));
    cudaErr(cudaEventSynchronize(stop));

    // FindAverage<<<BLOCK_NUM, THREADS_NUM>>>(d_A);
    // cudaLastErr();

    float   elapsedTime;
    cudaErr(cudaEventElapsedTime(&elapsedTime, start, stop ));
    printf( "Time to average:  %3.1f ms\n", elapsedTime );

    // copy A back to host
    cudaErr(cudaMemcpy(A, d_A, size_y * size_x * sizeof(elem), cudaMemcpyDeviceToHost));
    // copy At back to host
    cudaErr(cudaMemcpy(At, d_At, size_y * size_x * sizeof(elem), cudaMemcpyDeviceToHost));

    // cudaErr(cudaMemcpy(B, d_B, size_y * size_y * sizeof(elem), cudaMemcpyDeviceToHost));

    #if PRINT+0
    print_mat(A, size_y, size_x, "col distance");
    // print_mat(B, size_y, size_y, "B");
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
