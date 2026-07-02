#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <time.h>

#include "acx_cuda.cu"

#ifndef ACX_MIXED_PRECISION_CUDA_DECLS
#define ACX_MIXED_PRECISION_CUDA_DECLS

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n;
    int k;
    int tf32_iterations;
    int fp64_iterations;
    int tf32_f_calls;
    int fp64_f_calls;
    int tf32_converged;
    int fp64_converged;
    double tf32_time_sec;
    double fp64_time_sec;
    double total_time_sec;
    double finalization_time_sec;
    double tf32_fixed_point_residual;
    double fp64_fixed_point_residual;
    double fp64_max_residual;
    double *d_vectors;
    double *d_values;
} ACXMixedTF32CudaResult;

void acx_mixed_precision_tf32_cuda_reset_result(
    ACXMixedTF32CudaResult *result);
void acx_mixed_precision_tf32_cuda_free_result(
    ACXMixedTF32CudaResult *result);
int acx_mixed_precision_tf32_cuda_device_tol(
    const double *d_matrix_col_major, int n, int k, double tf32_tol,
    int tf32_maxiter, double fp64_tol, int fp64_maxiter,
    cublasHandle_t handle, ACXMixedTF32CudaResult *result);

#ifdef __cplusplus
}
#endif

#endif

static double acx_mixed_now_seconds(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

extern "C" void acx_mixed_precision_tf32_cuda_reset_result(
    ACXMixedTF32CudaResult *result)
{
    if (result == NULL) {
        return;
    }

    result->n = 0;
    result->k = 0;
    result->tf32_iterations = 0;
    result->fp64_iterations = 0;
    result->tf32_f_calls = 0;
    result->fp64_f_calls = 0;
    result->tf32_converged = 0;
    result->fp64_converged = 0;
    result->tf32_time_sec = NAN;
    result->fp64_time_sec = NAN;
    result->total_time_sec = NAN;
    result->finalization_time_sec = NAN;
    result->tf32_fixed_point_residual = NAN;
    result->fp64_fixed_point_residual = NAN;
    result->fp64_max_residual = NAN;
    result->d_vectors = NULL;
    result->d_values = NULL;
}

extern "C" void acx_mixed_precision_tf32_cuda_free_result(
    ACXMixedTF32CudaResult *result)
{
    if (result == NULL) {
        return;
    }

    cudaFree(result->d_vectors);
    cudaFree(result->d_values);
    acx_mixed_precision_tf32_cuda_reset_result(result);
}

__global__ static void
acx_mixed_double_to_float_offdiag_kernel(const double *input, float *output,
                                         int n, size_t total)
{
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + threadIdx.x;

    if (idx < total) {
        int row = (int)(idx % (size_t)n);
        int col = (int)(idx / (size_t)n);

        output[idx] = (row == col) ? 0.0f : (float)input[idx];
    }
}

__global__ static void
acx_mixed_extract_diagonal_float_kernel(const double *matrix,
                                        float *diagonal, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        diagonal[idx] = (float)matrix[idx + (size_t)idx * (size_t)n];
    }
}

__global__ static void acx_mixed_set_identity_float_kernel(float *x, int n,
                                                          int k)
{
    size_t total = (size_t)n * (size_t)k;
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + threadIdx.x;

    if (idx < total) {
        int row = (int)(idx % (size_t)n);
        int col = (int)(idx / (size_t)n);

        x[idx] = (row == col) ? 1.0f : 0.0f;
    }
}

__global__ static void acx_mixed_float_to_double_kernel(const float *input,
                                                       double *output,
                                                       size_t total)
{
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + threadIdx.x;

    if (idx < total) {
        output[idx] = (double)input[idx];
    }
}

__global__ static void acx_mixed_fixed_point_delta_float_kernel(
    float *delta, const float *next, const float *current, size_t total)
{
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + threadIdx.x;

    if (idx < total) {
        delta[idx] = next[idx] - current[idx];
    }
}

__global__ static void
acx_mixed_tf32_metadata_kernel(const float *delta_x, float *update_diag, int n,
                               int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < k) {
        update_diag[col] = delta_x[col + (size_t)col * (size_t)n];
    }
}

__global__ static void acx_mixed_tf32_finalize_kernel(
    float *delta_x, const float *x, const float *diagonal,
    const float *update_diag, int *invalid_flag, int n, int k)
{
    size_t total = (size_t)n * (size_t)k;
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + threadIdx.x;

    if (idx < total) {
        int row = (int)(idx % (size_t)n);
        int col = (int)(idx / (size_t)n);

        if (row == col) {
            delta_x[idx] = 1.0f;
        } else {
            float denom = diagonal[col] - diagonal[row];

            if (fabsf(denom) <= FLT_EPSILON) {
                *invalid_flag = 1;
                delta_x[idx] = 0.0f;
            } else {
                delta_x[idx] =
                    (delta_x[idx] - x[idx] * update_diag[col]) / denom;
            }
        }
    }
}

__global__ static void acx_mixed_update_order2_float_kernel(
    float *x, const float *f1, const float *f2, int n, int k)
{
    extern __shared__ unsigned char acx_mixed_shared_bytes[];
    float *num_shared = (float *)acx_mixed_shared_bytes;
    float *den_shared = num_shared + blockDim.x;
    int col = blockIdx.x;
    int tid = threadIdx.x;
    float numerator = 0.0f;
    float denominator = 0.0f;

    if (col >= k) {
        return;
    }

    for (int row = tid; row < n; row += blockDim.x) {
        size_t idx = (size_t)row + (size_t)col * (size_t)n;
        float d1 = f1[idx] - x[idx];
        float d2 = f2[idx] - 2.0f * f1[idx] + x[idx];

        numerator += d2 * d1;
        denominator += d2 * d2;
    }

    num_shared[tid] = numerator;
    den_shared[tid] = denominator;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            num_shared[tid] += num_shared[tid + stride];
            den_shared[tid] += den_shared[tid + stride];
        }
        __syncthreads();
    }

    float sigma = 0.0f;
    if (den_shared[0] > 0.0f) {
        sigma = fabsf(num_shared[0] / den_shared[0]);
    }
    float sigma2 = sigma * sigma;

    for (int row = tid; row < n; row += blockDim.x) {
        size_t idx = (size_t)row + (size_t)col * (size_t)n;
        float d1 = f1[idx] - x[idx];
        float d2 = f2[idx] - 2.0f * f1[idx] + x[idx];

        x[idx] += 2.0f * sigma * d1 + sigma2 * d2;
    }
}

__global__ static void acx_mixed_update_order3_float_kernel(
    float *x, const float *f1, const float *f2, const float *f3, int n, int k)
{
    extern __shared__ unsigned char acx_mixed_shared_bytes[];
    float *num_shared = (float *)acx_mixed_shared_bytes;
    float *den_shared = num_shared + blockDim.x;
    int col = blockIdx.x;
    int tid = threadIdx.x;
    float numerator = 0.0f;
    float denominator = 0.0f;

    if (col >= k) {
        return;
    }

    for (int row = tid; row < n; row += blockDim.x) {
        size_t idx = (size_t)row + (size_t)col * (size_t)n;
        float d2 = f2[idx] - 2.0f * f1[idx] + x[idx];
        float d3 = f3[idx] - 3.0f * f2[idx] + 3.0f * f1[idx] - x[idx];

        numerator += d3 * d2;
        denominator += d3 * d3;
    }

    num_shared[tid] = numerator;
    den_shared[tid] = denominator;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            num_shared[tid] += num_shared[tid + stride];
            den_shared[tid] += den_shared[tid + stride];
        }
        __syncthreads();
    }

    float sigma = 0.0f;
    if (den_shared[0] > 0.0f) {
        sigma = fabsf(num_shared[0] / den_shared[0]);
    }
    float sigma2 = sigma * sigma;
    float sigma3 = sigma2 * sigma;

    for (int row = tid; row < n; row += blockDim.x) {
        size_t idx = (size_t)row + (size_t)col * (size_t)n;
        float d1 = f1[idx] - x[idx];
        float d2 = f2[idx] - 2.0f * f1[idx] + x[idx];
        float d3 = f3[idx] - 3.0f * f2[idx] + 3.0f * f1[idx] - x[idx];

        x[idx] += 3.0f * sigma * d1 + 3.0f * sigma2 * d2 +
                  sigma3 * d3;
    }
}

#define ACX_MIXED_CUDA_CHECK(call)                                             \
    do {                                                                       \
        cudaError_t mixed_cuda_status = (call);                                \
        if (mixed_cuda_status != cudaSuccess) {                                \
            status = ACX_CUDA_CUDA_ERROR;                                      \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

#define ACX_MIXED_CUBLAS_CHECK(call)                                           \
    do {                                                                       \
        cublasStatus_t mixed_cublas_status = (call);                           \
        if (mixed_cublas_status != CUBLAS_STATUS_SUCCESS) {                    \
            status = ACX_CUDA_CUBLAS_ERROR;                                    \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

static int acx_tf32_apply_f_device(cublasHandle_t handle,
                                   const float *d_offdiag_matrix,
                                   const float *d_x, float *d_y,
                                   const float *d_diagonal,
                                   float *d_update_diag,
                                   int *d_invalid_flag, int n, int k)
{
    int status = ACX_CUDA_SUCCESS;
    const float one = 1.0f;
    const float zero = 0.0f;
    const int block = 256;
    int grid_cols = (k + block - 1) / block;
    int grid_elements =
        (int)(((size_t)n * (size_t)k + (size_t)block - 1) / (size_t)block);
    int invalid_host = 0;

    ACX_MIXED_CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &one, d_offdiag_matrix,
        CUDA_R_32F, n, d_x, CUDA_R_32F, n, &zero, d_y, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT));

    ACX_MIXED_CUDA_CHECK(cudaMemset(d_invalid_flag, 0, sizeof(int)));

    acx_mixed_tf32_metadata_kernel<<<grid_cols, block>>>(d_y, d_update_diag,
                                                         n, k);
    ACX_MIXED_CUDA_CHECK(cudaGetLastError());

    acx_mixed_tf32_finalize_kernel<<<grid_elements, block>>>(
        d_y, d_x, d_diagonal, d_update_diag, d_invalid_flag, n, k);
    ACX_MIXED_CUDA_CHECK(cudaGetLastError());

    ACX_MIXED_CUDA_CHECK(cudaMemcpy(&invalid_host, d_invalid_flag,
                                    sizeof(int), cudaMemcpyDeviceToHost));
    return invalid_host ? ACX_CUDA_ZERO_DENOMINATOR : ACX_CUDA_SUCCESS;

cleanup:
    return status;
}

static int acx_tf32_initial_device_tol(
    const double *d_matrix_col_major, int n, int k, double tol, int maxiter,
    cublasHandle_t handle, double **d_initial_vectors_out,
    int *iterations_done_out, int *f_calls_done_out, int *converged_out,
    double *fixed_point_residual_out, double *residual_check_time_out)
{
    int status = ACX_CUDA_SUCCESS;
    int iterations_done = 0;
    int f_calls_done = 0;
    int converged = 0;
    size_t matrix_elements = 0;
    size_t vector_elements = 0;
    float *d_offdiag_matrix = NULL;
    float *d_x = NULL;
    float *d_f1 = NULL;
    float *d_f2 = NULL;
    float *d_f3 = NULL;
    float *d_diagonal = NULL;
    float *d_update_diag = NULL;
    float *d_delta = NULL;
    float *d_final = NULL;
    double *d_initial = NULL;
    int *d_invalid_flag = NULL;
    double fixed_point_residual = NAN;
    double residual_check_time = 0.0;
    const int block = 256;
    int grid_matrix = 0;
    int grid_elements = 0;
    int grid_diag = 0;

    if (d_initial_vectors_out != NULL) {
        *d_initial_vectors_out = NULL;
    }
    if (iterations_done_out != NULL) {
        *iterations_done_out = 0;
    }
    if (f_calls_done_out != NULL) {
        *f_calls_done_out = 0;
    }
    if (converged_out != NULL) {
        *converged_out = 0;
    }
    if (fixed_point_residual_out != NULL) {
        *fixed_point_residual_out = NAN;
    }
    if (residual_check_time_out != NULL) {
        *residual_check_time_out = 0.0;
    }

    if (d_matrix_col_major == NULL || handle == NULL ||
        d_initial_vectors_out == NULL || n <= 0 || k <= 0 || k > n ||
        tol <= 0.0 || maxiter <= 0) {
        return ACX_CUDA_INVALID_ARGUMENT;
    }

    matrix_elements = (size_t)n * (size_t)n;
    vector_elements = (size_t)n * (size_t)k;
    if (matrix_elements > (size_t)INT_MAX ||
        vector_elements > (size_t)INT_MAX) {
        return ACX_CUDA_INVALID_ARGUMENT;
    }

    grid_matrix =
        (int)((matrix_elements + (size_t)block - 1) / (size_t)block);
    grid_elements =
        (int)((vector_elements + (size_t)block - 1) / (size_t)block);
    grid_diag = (n + block - 1) / block;

    ACX_MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_offdiag_matrix,
                   matrix_elements * sizeof(float)));
    ACX_MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_x, vector_elements * sizeof(float)));
    ACX_MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_f1, vector_elements * sizeof(float)));
    ACX_MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_f2, vector_elements * sizeof(float)));
    ACX_MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_f3, vector_elements * sizeof(float)));
    ACX_MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_diagonal, (size_t)n * sizeof(float)));
    ACX_MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_update_diag, (size_t)k * sizeof(float)));
    ACX_MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_delta, vector_elements * sizeof(float)));
    ACX_MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_initial, vector_elements * sizeof(double)));
    ACX_MIXED_CUDA_CHECK(cudaMalloc((void **)&d_invalid_flag, sizeof(int)));

    acx_mixed_double_to_float_offdiag_kernel<<<grid_matrix, block>>>(
        d_matrix_col_major, d_offdiag_matrix, n, matrix_elements);
    ACX_MIXED_CUDA_CHECK(cudaGetLastError());

    acx_mixed_extract_diagonal_float_kernel<<<grid_diag, block>>>(
        d_matrix_col_major, d_diagonal, n);
    ACX_MIXED_CUDA_CHECK(cudaGetLastError());

    acx_mixed_set_identity_float_kernel<<<grid_elements, block>>>(d_x, n, k);
    ACX_MIXED_CUDA_CHECK(cudaGetLastError());
    ACX_MIXED_CUDA_CHECK(cudaDeviceSynchronize());
    d_final = d_x;

    for (int iter = 1; iter <= maxiter; ++iter) {
        int order = (iter % 2 == 1) ? 2 : 3;
        float delta_norm = 0.0f;
        float current_norm = 0.0f;
        double residual_start = 0.0;

        iterations_done = iter;
        status = acx_tf32_apply_f_device(handle, d_offdiag_matrix, d_x, d_f1,
                                         d_diagonal, d_update_diag,
                                         d_invalid_flag, n, k);
        if (status != ACX_CUDA_SUCCESS) {
            goto cleanup;
        }
        f_calls_done += 1;

        ACX_MIXED_CUDA_CHECK(cudaDeviceSynchronize());
        residual_start = acx_mixed_now_seconds();
        acx_mixed_fixed_point_delta_float_kernel<<<grid_elements, block>>>(
            d_delta, d_f1, d_x, vector_elements);
        ACX_MIXED_CUDA_CHECK(cudaGetLastError());
        ACX_MIXED_CUBLAS_CHECK(
            cublasSnrm2(handle, (int)vector_elements, d_delta, 1,
                        &delta_norm));
        ACX_MIXED_CUBLAS_CHECK(
            cublasSnrm2(handle, (int)vector_elements, d_x, 1, &current_norm));
        residual_check_time += acx_mixed_now_seconds() - residual_start;

        if (current_norm <= 0.0f || !isfinite((double)current_norm)) {
            status = ACX_CUDA_INVALID_ARGUMENT;
            goto cleanup;
        }

        fixed_point_residual = (double)delta_norm / (double)current_norm;
        if (fixed_point_residual <= tol) {
            converged = 1;
            d_final = d_f1;
            break;
        }

        status = acx_tf32_apply_f_device(handle, d_offdiag_matrix, d_f1, d_f2,
                                         d_diagonal, d_update_diag,
                                         d_invalid_flag, n, k);
        if (status != ACX_CUDA_SUCCESS) {
            goto cleanup;
        }
        f_calls_done += 1;

        if (order == 2) {
            acx_mixed_update_order2_float_kernel<<<
                k, block, 2 * block * sizeof(float)>>>(d_x, d_f1, d_f2, n,
                                                        k);
            ACX_MIXED_CUDA_CHECK(cudaGetLastError());
        } else {
            status = acx_tf32_apply_f_device(
                handle, d_offdiag_matrix, d_f2, d_f3, d_diagonal,
                d_update_diag, d_invalid_flag, n, k);
            if (status != ACX_CUDA_SUCCESS) {
                goto cleanup;
            }
            f_calls_done += 1;

            acx_mixed_update_order3_float_kernel<<<
                k, block, 2 * block * sizeof(float)>>>(d_x, d_f1, d_f2, d_f3,
                                                        n, k);
            ACX_MIXED_CUDA_CHECK(cudaGetLastError());
        }

        ACX_MIXED_CUDA_CHECK(cudaDeviceSynchronize());
        d_final = d_x;
    }

    acx_mixed_float_to_double_kernel<<<grid_elements, block>>>(
        d_final, d_initial, vector_elements);
    ACX_MIXED_CUDA_CHECK(cudaGetLastError());

    *d_initial_vectors_out = d_initial;
    if (iterations_done_out != NULL) {
        *iterations_done_out = iterations_done;
    }
    if (f_calls_done_out != NULL) {
        *f_calls_done_out = f_calls_done;
    }
    if (converged_out != NULL) {
        *converged_out = converged;
    }
    if (fixed_point_residual_out != NULL) {
        *fixed_point_residual_out = fixed_point_residual;
    }
    d_initial = NULL;

cleanup:
    if (residual_check_time_out != NULL) {
        *residual_check_time_out = residual_check_time;
    }

    cudaFree(d_offdiag_matrix);
    cudaFree(d_x);
    cudaFree(d_f1);
    cudaFree(d_f2);
    cudaFree(d_f3);
    cudaFree(d_diagonal);
    cudaFree(d_update_diag);
    cudaFree(d_delta);
    cudaFree(d_initial);
    cudaFree(d_invalid_flag);

    if (status != ACX_CUDA_SUCCESS && d_initial_vectors_out != NULL) {
        *d_initial_vectors_out = NULL;
    }

    return status;
}

static int acx_cuda_device_with_initial_tol_mixed(
    const double *d_matrix_col_major, const double *d_initial_vectors_col_major,
    int n, int k, double tol, int maxiter, cublasHandle_t handle,
    ACXCudaResult *result)
{
    int status = ACX_CUDA_SUCCESS;
    size_t elements = 0;
    double *d_x = NULL;
    double *d_f1 = NULL;
    double *d_f2 = NULL;
    double *d_f3 = NULL;
    double *d_final = NULL;
    double *d_diagonal = NULL;
    double *d_mx_diag = NULL;
    double *d_update_diag = NULL;
    double *d_residual_sq = NULL;
    double *d_residuals = NULL;
    double *d_delta = NULL;
    double *h_residuals = NULL;
    double *d_values = NULL;
    int *d_invalid_flag = NULL;
    int block = 256;
    int grid_elements = 0;
    int grid_diag = 0;
    double stop_tol = 0.0;
    double total_start = 0.0;
    double iteration_start = 0.0;
    double finalization_start = 0.0;

    if (result == NULL) {
        return ACX_CUDA_INVALID_ARGUMENT;
    }
    acx_cuda_reset_result(result);
    total_start = acx_mixed_now_seconds();

    if (d_matrix_col_major == NULL || d_initial_vectors_col_major == NULL ||
        handle == NULL || n <= 0 || k <= 0 || k > n || tol <= 0.0 ||
        maxiter <= 0) {
        return ACX_CUDA_INVALID_ARGUMENT;
    }

    elements = (size_t)n * (size_t)k;
    if (elements > (size_t)INT_MAX) {
        return ACX_CUDA_INVALID_ARGUMENT;
    }

    grid_elements =
        (int)((elements + (size_t)block - 1) / (size_t)block);
    grid_diag = (n + block - 1) / block;
    stop_tol = tol < ACX_FIXED_POINT_RTOL ? tol : ACX_FIXED_POINT_RTOL;

    if (cudaMalloc((void **)&d_x, elements * sizeof(double)) != cudaSuccess ||
        cudaMalloc((void **)&d_f1, elements * sizeof(double)) != cudaSuccess ||
        cudaMalloc((void **)&d_f2, elements * sizeof(double)) != cudaSuccess ||
        cudaMalloc((void **)&d_f3, elements * sizeof(double)) != cudaSuccess ||
        cudaMalloc((void **)&d_diagonal, (size_t)n * sizeof(double)) !=
            cudaSuccess ||
        cudaMalloc((void **)&d_mx_diag, (size_t)k * sizeof(double)) !=
            cudaSuccess ||
        cudaMalloc((void **)&d_update_diag, (size_t)k * sizeof(double)) !=
            cudaSuccess ||
        cudaMalloc((void **)&d_residual_sq, (size_t)k * sizeof(double)) !=
            cudaSuccess ||
        cudaMalloc((void **)&d_residuals, (size_t)k * sizeof(double)) !=
            cudaSuccess ||
        cudaMalloc((void **)&d_delta, elements * sizeof(double)) !=
            cudaSuccess ||
        cudaMalloc((void **)&d_values, (size_t)k * sizeof(double)) !=
            cudaSuccess ||
        cudaMalloc((void **)&d_invalid_flag, sizeof(int)) != cudaSuccess) {
        status = ACX_CUDA_ALLOCATION_FAILED;
        goto cleanup;
    }

    h_residuals = (double *)malloc((size_t)k * sizeof(double));
    if (h_residuals == NULL) {
        status = ACX_CUDA_ALLOCATION_FAILED;
        goto cleanup;
    }

    acx_extract_diagonal_kernel<<<grid_diag, block>>>(d_matrix_col_major,
                                                       d_diagonal, n);
    if (cudaGetLastError() != cudaSuccess) {
        status = ACX_CUDA_CUDA_ERROR;
        goto cleanup;
    }
    if (cudaMemcpy(d_x, d_initial_vectors_col_major,
                   elements * sizeof(double),
                   cudaMemcpyDeviceToDevice) != cudaSuccess) {
        status = ACX_CUDA_CUDA_ERROR;
        goto cleanup;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        status = ACX_CUDA_CUDA_ERROR;
        goto cleanup;
    }
    d_final = d_x;

    iteration_start = acx_mixed_now_seconds();
    for (int iter = 1; iter <= maxiter; ++iter) {
        int order = (iter % 2 == 1) ? 2 : 3;
        double max_residual = 0.0;
        double delta_norm = 0.0;
        double current_norm = 0.0;
        double fixed_point_residual = NAN;

        result->iterations = iter;
        status = acx_apply_f_device(handle, d_matrix_col_major, d_x, d_f1,
                                    d_diagonal, d_mx_diag, d_update_diag,
                                    d_residual_sq, d_residuals,
                                    d_invalid_flag, n, k);
        if (status != ACX_CUDA_SUCCESS) {
            goto cleanup;
        }
        result->f_calls += 1;

        if (cudaMemcpy(h_residuals, d_residuals, (size_t)k * sizeof(double),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            status = ACX_CUDA_CUDA_ERROR;
            goto cleanup;
        }
        for (int col = 0; col < k; ++col) {
            if (h_residuals[col] > max_residual) {
                max_residual = h_residuals[col];
            }
        }
        result->max_residual = max_residual;

        acx_fixed_point_delta_kernel<<<grid_elements, block>>>(d_delta, d_f1,
                                                               d_x, elements);
        if (cudaGetLastError() != cudaSuccess) {
            status = ACX_CUDA_CUDA_ERROR;
            goto cleanup;
        }
        if (cublasDnrm2(handle, (int)elements, d_delta, 1, &delta_norm) !=
                CUBLAS_STATUS_SUCCESS ||
            cublasDnrm2(handle, (int)elements, d_x, 1, &current_norm) !=
                CUBLAS_STATUS_SUCCESS) {
            status = ACX_CUDA_CUBLAS_ERROR;
            goto cleanup;
        }
        if (current_norm <= 0.0 || !isfinite(current_norm)) {
            status = ACX_CUDA_INVALID_ARGUMENT;
            goto cleanup;
        }
        fixed_point_residual = delta_norm / current_norm;
        result->fixed_point_residual = fixed_point_residual;

        if (fixed_point_residual < stop_tol) {
            result->converged = 1;
            d_final = d_f1;
            break;
        }

        status = acx_apply_f_device(handle, d_matrix_col_major, d_f1, d_f2,
                                    d_diagonal, d_mx_diag, d_update_diag,
                                    d_residual_sq, d_residuals,
                                    d_invalid_flag, n, k);
        if (status != ACX_CUDA_SUCCESS) {
            goto cleanup;
        }
        result->f_calls += 1;

        if (order == 2) {
            acx_update_order2_kernel<<<k, block, 2 * block * sizeof(double)>>>(
                d_x, d_f1, d_f2, n, k);
            if (cudaGetLastError() != cudaSuccess) {
                status = ACX_CUDA_CUDA_ERROR;
                goto cleanup;
            }
        } else {
            status = acx_apply_f_device(handle, d_matrix_col_major, d_f2,
                                        d_f3, d_diagonal, d_mx_diag,
                                        d_update_diag, d_residual_sq,
                                        d_residuals, d_invalid_flag, n, k);
            if (status != ACX_CUDA_SUCCESS) {
                goto cleanup;
            }
            result->f_calls += 1;
            acx_update_order3_kernel<<<k, block, 2 * block * sizeof(double)>>>(
                d_x, d_f1, d_f2, d_f3, n, k);
            if (cudaGetLastError() != cudaSuccess) {
                status = ACX_CUDA_CUDA_ERROR;
                goto cleanup;
            }
        }

        if (cudaDeviceSynchronize() != cudaSuccess) {
            status = ACX_CUDA_CUDA_ERROR;
            goto cleanup;
        }
        d_final = d_x;
    }
    result->iteration_time_sec = acx_mixed_now_seconds() - iteration_start;

    finalization_start = acx_mixed_now_seconds();
    status = acx_finalize_values(handle, d_matrix_col_major, d_final, d_f2,
                                 d_values, n, k);
    if (status != ACX_CUDA_SUCCESS) {
        goto cleanup;
    }
    result->finalization_time_sec =
        acx_mixed_now_seconds() - finalization_start;

    result->n = n;
    result->k = k;
    result->time_sec = acx_mixed_now_seconds() - total_start;
    result->d_vectors = d_final;
    result->d_values = d_values;
    if (d_final == d_x) {
        d_x = NULL;
    } else if (d_final == d_f1) {
        d_f1 = NULL;
    }
    d_values = NULL;

cleanup:
    cudaFree(d_x);
    cudaFree(d_f1);
    cudaFree(d_f2);
    cudaFree(d_f3);
    cudaFree(d_diagonal);
    cudaFree(d_mx_diag);
    cudaFree(d_update_diag);
    cudaFree(d_residual_sq);
    cudaFree(d_residuals);
    cudaFree(d_delta);
    cudaFree(d_values);
    cudaFree(d_invalid_flag);
    free(h_residuals);
    if (status != ACX_CUDA_SUCCESS) {
        acx_cuda_free_result(result);
    }
    return status;
}

extern "C" int acx_mixed_precision_tf32_cuda_device_tol(
    const double *d_matrix_col_major, int n, int k, double tf32_tol,
    int tf32_maxiter, double fp64_tol, int fp64_maxiter,
    cublasHandle_t handle, ACXMixedTF32CudaResult *result)
{
    int status = ACX_CUDA_SUCCESS;
    double *d_initial = NULL;
    ACXCudaResult fp64_result;
    double total_start = 0.0;
    double stage_start = 0.0;
    double tf32_residual_check_time = 0.0;
    double raw_tf32_time = 0.0;

    if (result == NULL) {
        return ACX_CUDA_INVALID_ARGUMENT;
    }
    acx_mixed_precision_tf32_cuda_reset_result(result);
    acx_cuda_reset_result(&fp64_result);

    if (d_matrix_col_major == NULL || handle == NULL || n <= 0 || k <= 0 ||
        k > n || tf32_tol <= 0.0 || fp64_tol <= 0.0 ||
        tf32_maxiter <= 0 || fp64_maxiter <= 0) {
        return ACX_CUDA_INVALID_ARGUMENT;
    }

    total_start = acx_mixed_now_seconds();

    stage_start = acx_mixed_now_seconds();
    status = acx_tf32_initial_device_tol(
        d_matrix_col_major, n, k, tf32_tol, tf32_maxiter, handle, &d_initial,
        &result->tf32_iterations, &result->tf32_f_calls,
        &result->tf32_converged, &result->tf32_fixed_point_residual,
        &tf32_residual_check_time);
    if (status != ACX_CUDA_SUCCESS) {
        goto cleanup;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        status = ACX_CUDA_CUDA_ERROR;
        goto cleanup;
    }
    raw_tf32_time = acx_mixed_now_seconds() - stage_start;
    result->tf32_time_sec = raw_tf32_time - tf32_residual_check_time;
    if (result->tf32_time_sec < 0.0) {
        result->tf32_time_sec = 0.0;
    }

    stage_start = acx_mixed_now_seconds();
    status = acx_cuda_device_with_initial_tol_mixed(
        d_matrix_col_major, d_initial, n, k, fp64_tol, fp64_maxiter, handle,
        &fp64_result);
    if (status != ACX_CUDA_SUCCESS) {
        goto cleanup;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        status = ACX_CUDA_CUDA_ERROR;
        goto cleanup;
    }
    result->fp64_time_sec = acx_mixed_now_seconds() - stage_start;
    result->total_time_sec =
        acx_mixed_now_seconds() - total_start - tf32_residual_check_time;
    if (result->total_time_sec <
        result->tf32_time_sec + result->fp64_time_sec) {
        result->total_time_sec =
            result->tf32_time_sec + result->fp64_time_sec;
    }

    result->n = n;
    result->k = k;
    result->fp64_iterations = fp64_result.iterations;
    result->fp64_f_calls = fp64_result.f_calls;
    result->fp64_converged = fp64_result.converged;
    result->finalization_time_sec = fp64_result.finalization_time_sec;
    result->fp64_fixed_point_residual = fp64_result.fixed_point_residual;
    result->fp64_max_residual = fp64_result.max_residual;
    result->d_vectors = fp64_result.d_vectors;
    result->d_values = fp64_result.d_values;
    fp64_result.d_vectors = NULL;
    fp64_result.d_values = NULL;

cleanup:
    cudaFree(d_initial);
    acx_cuda_free_result(&fp64_result);
    if (status != ACX_CUDA_SUCCESS) {
        acx_mixed_precision_tf32_cuda_free_result(result);
    }
    return status;
}
