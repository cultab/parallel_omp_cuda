#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#define PRINT 0
#define SIMPLE 0

#define BLOCK_NUM 128
#define THREADS_NUM 128

// matrix size
#define SIZE_Y 10000
#define SIZE_X 20000
// #define SIZE_Y 3
// #define SIZE_X 2

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

__host__ __device__ void print_mat(elem *mat, size_t height, size_t width, const char* name) {
    printf("%s (%ldx%ld):\n", name, height ,width);
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            printf("%6.2f ", mat[i * width + j]);
        }
        printf("\n");
    }
}


// each block computes several elements of the final vector;
__global__ void matrix_vector_mul(elem* d_mat, elem* d_vec, elem* d_res, size_t size_y, size_t size_x)
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
                thread_result += d_vec[j] * d_mat[i * size_x + j];
                // printf("Block(%d),Thread(%d): local_res += %f \n", blockId, threadId, tmp);
        }

        // all threads add their local results
        atomicAdd(&block_result, thread_result);

        // wait for all threads to add their local result
        syncthreads();

        // one thread adds they the block's result to global memory
        // non-atomically since each element gets calculated by exactly one block
        if (thread_id == 0) {
            // printf("Block(%d): d_res[%d] += %f\n", blockId, i, local_res);
            d_res[i] = block_result;
        }
    }
}

/* 
 * Transpose a matrix.
 *
 * @d_mat is the input matrix.
 * @d_result is the output matrix.
 *
 */

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
    elem *A;
    elem *d_A; // matrix A

    elem *At;
    elem *d_At; // A transpose

    elem *x;
    elem *d_x; // vector x, also store the final result

    elem *A_mul_x;
    elem *d_A_mul_x; // stores the result of A * x

    size_t size_x = SIZE_X;
    size_t size_y = SIZE_Y;

    /* Allocate host memory */

    A = (elem*)malloc(size_y * size_x * sizeof(elem));
    if (A == NULL) {
        fprintf(stderr, "Failed to allocate memory at line %d\n", __LINE__);
        exit(-1);
    }
    At = (elem*)malloc(size_y * size_x * sizeof(elem));
    if (At == NULL) {
        fprintf(stderr, "Failed to allocate memory at line %d\n", __LINE__);
        exit(-1);
    }
    x = (elem*)malloc(size_x * sizeof(elem));
    if (x == NULL) {
        fprintf(stderr, "Failed to allocate memory at line %d\n", __LINE__);
        exit(-1);
    }
    A_mul_x = (elem*)malloc(size_y * sizeof(elem));
    if (A_mul_x == NULL) {
        fprintf(stderr, "Failed to allocate memory at line %d\n", __LINE__);
        exit(-1);
    }

    // initialize matrix A
    for (size_t i = 0; i < size_y; i++) {
        for (size_t j = 0; j < size_x; j++) {
            #if SIMPLE+0
                A[i * size_x + j] = i;
            #else
                A[i * size_x + j] = rand() % 10;
            #endif
            // A[i * size_x + j] = i * j;
        }
    }

    // initialize vector x
    for (size_t i = 0; i < size_x; i++) {
        #if SIMPLE+0
            x[i] = i;
        #else
            x[i] = rand() % 10;
        #endif
    }

    /* Allocate memory */

    // matricies
    cudaErr(cudaMalloc(&d_A, size_y * size_x * sizeof(elem)));
    cudaErr(cudaMalloc(&d_At, size_x * size_y * sizeof(elem)));

    // vectors
    cudaErr(cudaMalloc(&d_x, size_x * sizeof(elem)));
    cudaErr(cudaMalloc(&d_A_mul_x, size_y * sizeof(elem)));
    // cudaErr(cudaMalloc(&d_final_res, size_x * sizeof(elem)));

    /* Copy data to device */

    cudaErr(cudaMemcpy(d_A, A, size_y * size_x * sizeof(elem), cudaMemcpyHostToDevice));
    cudaErr(cudaMemcpy(d_x, x, size_x * sizeof(elem), cudaMemcpyHostToDevice));

    // A * x
    matrix_vector_mul<<<BLOCK_NUM, THREADS_NUM>>>(d_A, d_x, d_A_mul_x, size_y, size_x);
    cudaLastErr();

    // copy back result
    cudaErr(cudaMemcpy(A_mul_x, d_A_mul_x, size_y * sizeof(elem), cudaMemcpyDeviceToHost));

    // get At by transposing A
    matrix_transpose<<<BLOCK_NUM, THREADS_NUM>>>(d_A, d_At, size_y, size_x);
    cudaLastErr();

    // free A from device memory
    cudaErr(cudaFree(d_A));

    #if PRINT+0
    // copy At to host (only to print)
    cudaErr(cudaMemcpy(At, d_At, size_x * size_y * sizeof(elem), cudaMemcpyDeviceToHost));
    #endif

    // At * (A * x)
    matrix_vector_mul<<<BLOCK_NUM, THREADS_NUM>>>(d_At, d_A_mul_x, d_x, size_x, size_y);
    cudaLastErr();

    // free At and A_mul_x from device memory
    cudaErr(cudaFree(d_At));
    cudaErr(cudaFree(d_A_mul_x));

    // copy final result to host
    cudaErr(cudaMemcpy(x, d_x, size_x * sizeof(elem), cudaMemcpyDeviceToHost));

    // free x from device memory
    cudaErr(cudaFree(d_x));

    #if PRINT+0
    print_mat(A, size_y, size_x, "A");
    print_mat(x, size_x, 1, "x");
    print_mat(A_mul_x, 1, size_y, "result of A * x");

    print_mat(At, size_x, size_y, "At");
    print_mat(x, 1, size_x, "final result of At * A * x");
    #endif

    free(A);
    free(At);
    free(x);

    return 0;
}

// int chunk = 5000; // less than 48KBs of elems (6000 is about right)
// int chunk = 4; // less than 48KBs of elems (6000 is about right)
// matrix_vector_mul_shared<<<BLOCK_NUM, THREADS_NUM, chunk * sizeof(elem)>>>(d_A, d_x, d_temp2, size_y, size_x, chunk);
// __device__ void copy_to_shared(elem* d_vec, size_t d_vec_size, elem* block_local_vec, size_t start, size_t size){{{
// {
//     int threadId = threadIdx.x;
//     int stride = blockDim.x;
//
//     int j;
//     size_t i;
//     for (j = threadId, i = threadId + start; i < d_vec_size && j < size; i += stride, j += stride) {
//         // printf("Block(%d),Thread(%u): start=%ld loc_vec[%d] glob_vec[%ld]\n", blockIdx.x, threadId, start, j, i);
//         block_local_vec[j] = d_vec[i];
//     }
//
// }
//
// __global__ void matrix_vector_mul_shared(elem* d_mat, elem* d_vec, elem* d_res, size_t size_y, size_t size_x, size_t chunk)
// {
//     __shared__ elem local_res;
//     extern __shared__ elem local_vec_chunk[];
//
//     int blockId = blockIdx.x;
//     int threadId = threadIdx.x;
//
//     int row_stride = gridDim.x;
//     int col_stride = blockDim.x;
//
//     for (int k = 0; k <= size_x / chunk; k++) {
//
//         if (threadIdx.x == 0) {
//             printf("Block(%d)): run %d/%ld with chunk=%ld\n", blockId, k, size_x/chunk, chunk);
//             local_res = 0;
//         }
//
//         copy_to_shared(d_vec, size_x, local_vec_chunk, k * chunk, chunk);
//
//         // if (threadIdx.x == 0) {
//         //     print_mat(local_vec_chunk, 1, chunk, "chunk");
//         // }
//
//         syncthreads();
//
//         // result (1x5):
//         //   0.00  30.00  60.00  90.00 120.00
//         #define IDX i * size_x + v
//         for (int i = blockId; i < size_y; i += row_stride) {
//             int j;
//             int v;
//          // for (j = threadId, i = threadId + start; i < d_vec_size && j < size; i += stride, j += stride) {
//             for (j = threadId, v = (threadId + 1) * (chunk * k); v <= size_x && j < chunk; j += col_stride, v += col_stride + (chunk * k)) {
//                 printf("Block(%d),Thread(%d): vec[%d] mat[%d][%d] mat[%ld]\n", blockId, threadId, j, i, v, IDX);
//                 atomicAdd(&local_res, local_vec_chunk[j] * d_mat[IDX]);
//                 syncthreads();
//             }
//
//             syncthreads();
//
//             if (threadIdx.x == 0) {
//                 // printf("Block(%d): d_res[%d] += %f\n", blockId, i, local_res);
//                 atomicAdd(&d_res[i], local_res);
//             }
//         }
//     }
//
// }}}}
