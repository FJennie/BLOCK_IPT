#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <time.h>

#include "../ipt_cuda.cu"

#ifndef IPT_MIXED_PRECISION_CUDA_DECLS
#define IPT_MIXED_PRECISION_CUDA_DECLS

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n;
    int k;
    int fp16_iterations;
    int fp64_iterations;
    double fp16_time_sec;
    double fp64_time_sec;
    double total_time_sec;
    double fp16_fixed_point_residual;
    double fp64_fixed_point_residual;
    double *d_vectors;
    double *d_values;
} IPTMixedCudaDeviceResult;

typedef struct {
    int n;
    int k;
    int fp32_iterations;
    int fp64_iterations;
    double fp32_time_sec;
    double fp64_time_sec;
    double total_time_sec;
    double fp32_fixed_point_residual;
    double fp64_fixed_point_residual;
    double *d_vectors;
    double *d_values;
} IPTMixedFP32CudaDeviceResult;

typedef struct {
    int n;
    int k;
    int tf32_iterations;
    int fp64_iterations;
    double tf32_time_sec;
    double fp64_time_sec;
    double total_time_sec;
    double tf32_fixed_point_residual;
    double fp64_fixed_point_residual;
    double *d_vectors;
    double *d_values;
} IPTMixedTF32CudaDeviceResult;

void ipt_mixed_precision_cuda_free_device_result(
    IPTMixedCudaDeviceResult *result);
int ipt_mixed_precision_cuda_device_tol(
    const double *d_matrix_col_major, int n, int k, double fp16_tol,
    int fp16_maxiter, double fp64_tol, int fp64_maxiter,
    cublasHandle_t handle, IPTMixedCudaDeviceResult *result);
void ipt_mixed_precision_fp32_cuda_free_device_result(
    IPTMixedFP32CudaDeviceResult *result);
int ipt_mixed_precision_fp32_cuda_device_tol(
    const double *d_matrix_col_major, int n, int k, double fp32_tol,
    int fp32_maxiter, double fp64_tol, int fp64_maxiter,
    cublasHandle_t handle, IPTMixedFP32CudaDeviceResult *result);
void ipt_mixed_precision_tf32_cuda_free_device_result(
    IPTMixedTF32CudaDeviceResult *result);
int ipt_mixed_precision_tf32_cuda_device_tol(
    const double *d_matrix_col_major, int n, int k, double tf32_tol,
    int tf32_maxiter, double fp64_tol, int fp64_maxiter,
    cublasHandle_t handle, IPTMixedTF32CudaDeviceResult *result);

#ifdef __cplusplus
}
#endif

#endif

static double ipt_mixed_now_seconds(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void ipt_mixed_reset_device_result(IPTMixedCudaDeviceResult *result)
{
    if (result == NULL) {
        return;
    }

    result->n = 0;
    result->k = 0;
    result->fp16_iterations = 0;
    result->fp64_iterations = 0;
    result->fp16_time_sec = NAN;
    result->fp64_time_sec = NAN;
    result->total_time_sec = NAN;
    result->fp16_fixed_point_residual = NAN;
    result->fp64_fixed_point_residual = NAN;
    result->d_vectors = NULL;
    result->d_values = NULL;
}

static void
ipt_mixed_fp32_reset_device_result(IPTMixedFP32CudaDeviceResult *result)
{
    if (result == NULL) {
        return;
    }

    result->n = 0;
    result->k = 0;
    result->fp32_iterations = 0;
    result->fp64_iterations = 0;
    result->fp32_time_sec = NAN;
    result->fp64_time_sec = NAN;
    result->total_time_sec = NAN;
    result->fp32_fixed_point_residual = NAN;
    result->fp64_fixed_point_residual = NAN;
    result->d_vectors = NULL;
    result->d_values = NULL;
}

static void
ipt_mixed_tf32_reset_device_result(IPTMixedTF32CudaDeviceResult *result)
{
    if (result == NULL) {
        return;
    }

    result->n = 0;
    result->k = 0;
    result->tf32_iterations = 0;
    result->fp64_iterations = 0;
    result->tf32_time_sec = NAN;
    result->fp64_time_sec = NAN;
    result->total_time_sec = NAN;
    result->tf32_fixed_point_residual = NAN;
    result->fp64_fixed_point_residual = NAN;
    result->d_vectors = NULL;
    result->d_values = NULL;
}

extern "C" void ipt_mixed_precision_cuda_free_device_result(
    IPTMixedCudaDeviceResult *result)
{
    if (result == NULL) {
        return;
    }

    cudaFree(result->d_vectors);
    cudaFree(result->d_values);
    ipt_mixed_reset_device_result(result);
}

extern "C" void ipt_mixed_precision_fp32_cuda_free_device_result(
    IPTMixedFP32CudaDeviceResult *result)
{
    if (result == NULL) {
        return;
    }

    cudaFree(result->d_vectors);
    cudaFree(result->d_values);
    ipt_mixed_fp32_reset_device_result(result);
}

extern "C" void ipt_mixed_precision_tf32_cuda_free_device_result(
    IPTMixedTF32CudaDeviceResult *result)
{
    if (result == NULL) {
        return;
    }

    cudaFree(result->d_vectors);
    cudaFree(result->d_values);
    ipt_mixed_tf32_reset_device_result(result);
}

__global__ static void double_to_half_offdiag_kernel(const double *input,
                                                     __half *output, int n,
                                                     int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        int row = idx % n;
        int col = idx / n;

        output[idx] =
            __float2half_rn(row == col ? 0.0f : (float)input[idx]);
    }
}

__global__ static void double_to_float_offdiag_kernel(const double *input,
                                                      float *output, int n,
                                                      int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        int row = idx % n;
        int col = idx / n;

        output[idx] = row == col ? 0.0f : (float)input[idx];
    }
}

__global__ static void set_identity_half_kernel(__half *x, int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;

    if (idx >= total) {
        return;
    }

    {
        int row = idx % n;
        int col = idx / n;
        x[idx] = __float2half_rn(row == col ? 1.0f : 0.0f);
    }
}

__global__ static void set_identity_float_kernel(float *x, int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;

    if (idx >= total) {
        return;
    }

    {
        int row = idx % n;
        int col = idx / n;
        x[idx] = (row == col) ? 1.0f : 0.0f;
    }
}

__global__ static void build_g_float_from_double_diag_kernel(
    const double *diagonal, float *g, int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;

    if (idx >= total) {
        return;
    }

    {
        int row = idx % n;
        int col = idx / n;

        if (row == col) {
            g[idx] = 0.0f;
        } else {
            g[idx] = (float)(1.0 / (diagonal[col] - diagonal[row]));
        }
    }
}

__global__ static void column_diagonal_after_e_float_kernel(
    const float *y, float *column_diagonal, int n, int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < k) {
        int diag_idx = col + col * n;

        column_diagonal[col] = y[diag_idx];
    }
}

__global__ static void column_diagonal_after_e_half_kernel(
    const __half *y, float *column_diagonal, int n, int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < k) {
        int diag_idx = col + col * n;

        column_diagonal[col] = __half2float(y[diag_idx]);
    }
}

__global__ static void ipt_update_float_kernel(
    float *y, const float *x, const float *g,
    const float *column_diagonal, int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;

    if (idx >= total) {
        return;
    }

    {
        int row = idx % n;
        int col = idx / n;

        if (row == col) {
            y[idx] = 1.0f;
            return;
        }

        y[idx] = (y[idx] - x[idx] * column_diagonal[col]) * g[idx];
    }
}

__global__ static void ipt_update_half_kernel(
    __half *y, const __half *x, const float *g,
    const float *column_diagonal, int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;

    if (idx >= total) {
        return;
    }

    {
        int row = idx % n;
        int col = idx / n;

        if (row == col) {
            y[idx] = __float2half_rn(1.0f);
            return;
        }

        {
            float x_value = __half2float(x[idx]);
            float y_value =
                (__half2float(y[idx]) - x_value * column_diagonal[col]) *
                g[idx];
            y[idx] = __float2half_rn(y_value);
        }
    }
}

__global__ static void float_delta_kernel(float *delta, const float *next,
                                          const float *prev, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        delta[idx] = next[idx] - prev[idx];
    }
}

__global__ static void half_delta_and_current_float_kernel(
    float *delta, float *current, const __half *next, const __half *prev,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        float next_value = __half2float(next[idx]);
        float prev_value = __half2float(prev[idx]);

        delta[idx] = next_value - prev_value;
        current[idx] = prev_value;
    }
}

__global__ static void float_to_double_kernel(const float *input,
                                              double *output, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        output[idx] = (double)input[idx];
    }
}

__global__ static void half_to_double_kernel(const __half *input,
                                             double *output, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        output[idx] = (double)__half2float(input[idx]);
    }
}

#define MIXED_CUDA_CHECK(call)                                                 \
    do {                                                                       \
        cudaError_t mixed_cuda_status = (call);                                \
        if (mixed_cuda_status != cudaSuccess) {                                \
            status = IPT_CUDA_CUDA_ERROR;                                      \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

#define MIXED_CUBLAS_CHECK(call)                                               \
    do {                                                                       \
        cublasStatus_t mixed_cublas_status = (call);                           \
        if (mixed_cublas_status != CUBLAS_STATUS_SUCCESS) {                    \
            status = IPT_CUDA_CUBLAS_ERROR;                                    \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

static int ipt_fp16_initial_device_tol(const double *d_matrix_col_major, int n,
                                       int k, double tol, int maxiter,
                                       cublasHandle_t handle,
                                       double **d_initial_vectors_out,
                                       int *iterations_done_out,
                                       double *fixed_point_residual_out,
                                       double *residual_check_time_out)
{
    int status = IPT_CUDA_SUCCESS;
    int iterations_done = 0;
    size_t matrix_elements = 0;
    size_t vector_elements = 0;
    __half *d_matrix_half = NULL;
    __half *d_x_a = NULL;
    __half *d_x_b = NULL;
    __half *d_x = NULL;
    __half *d_y = NULL;
    double *d_diagonal = NULL;
    double *d_initial = NULL;
    float *d_g = NULL;
    float *d_column_diagonal = NULL;
    float *d_delta = NULL;
    float *d_current = NULL;
    double fixed_point_residual = NAN;
    double residual_check_time = 0.0;
    const float one = 1.0f;
    const float zero = 0.0f;
    const int block_size = 256;

    if (d_initial_vectors_out != NULL) {
        *d_initial_vectors_out = NULL;
    }
    if (iterations_done_out != NULL) {
        *iterations_done_out = 0;
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
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    matrix_elements = (size_t)n * (size_t)n;
    vector_elements = (size_t)n * (size_t)k;
    if (matrix_elements > (size_t)INT_MAX ||
        vector_elements > (size_t)INT_MAX) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_matrix_half, matrix_elements * sizeof(__half)));
    MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_x_a, vector_elements * sizeof(__half)));
    MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_x_b, vector_elements * sizeof(__half)));
    MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_diagonal, (size_t)n * sizeof(double)));
    MIXED_CUDA_CHECK(cudaMalloc((void **)&d_g, vector_elements * sizeof(float)));
    MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_column_diagonal, (size_t)k * sizeof(float)));
    MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_delta, vector_elements * sizeof(float)));
    MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_current, vector_elements * sizeof(float)));
    MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_initial, vector_elements * sizeof(double)));

    {
        int matrix_blocks =
            ((int)matrix_elements + block_size - 1) / block_size;
        int vector_blocks =
            ((int)vector_elements + block_size - 1) / block_size;
        int diag_blocks = (n + block_size - 1) / block_size;

        double_to_half_offdiag_kernel<<<matrix_blocks, block_size>>>(
            d_matrix_col_major, d_matrix_half, n, (int)matrix_elements);
        MIXED_CUDA_CHECK(cudaGetLastError());

        set_identity_half_kernel<<<vector_blocks, block_size>>>(d_x_a, n, k);
        MIXED_CUDA_CHECK(cudaGetLastError());

        extract_diagonal_kernel<<<diag_blocks, block_size>>>(
            d_matrix_col_major, d_diagonal, n);
        MIXED_CUDA_CHECK(cudaGetLastError());

        build_g_float_from_double_diag_kernel<<<vector_blocks, block_size>>>(
            d_diagonal, d_g, n, k);
        MIXED_CUDA_CHECK(cudaGetLastError());
    }

    d_x = d_x_a;
    d_y = d_x_b;

    for (int iter = 0; iter < maxiter; ++iter) {
        int col_blocks = (k + block_size - 1) / block_size;
        int vector_blocks =
            ((int)vector_elements + block_size - 1) / block_size;
        __half *tmp = NULL;
        float delta_norm = 0.0f;
        float current_norm = 0.0f;

        MIXED_CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &one, d_matrix_half,
            CUDA_R_16F, n, d_x, CUDA_R_16F, n, &zero, d_y, CUDA_R_16F, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        column_diagonal_after_e_half_kernel<<<col_blocks, block_size>>>(
            d_y, d_column_diagonal, n, k);
        MIXED_CUDA_CHECK(cudaGetLastError());

        ipt_update_half_kernel<<<vector_blocks, block_size>>>(
            d_y, d_x, d_g, d_column_diagonal, n, k);
        MIXED_CUDA_CHECK(cudaGetLastError());

        iterations_done = iter + 1;

        {
            double residual_start = 0.0;

            MIXED_CUDA_CHECK(cudaDeviceSynchronize());
            residual_start = ipt_mixed_now_seconds();

            half_delta_and_current_float_kernel<<<vector_blocks, block_size>>>(
                d_delta, d_current, d_y, d_x, (int)vector_elements);
            MIXED_CUDA_CHECK(cudaGetLastError());

            MIXED_CUBLAS_CHECK(cublasSnrm2(handle, (int)vector_elements,
                                           d_delta, 1, &delta_norm));
            MIXED_CUBLAS_CHECK(cublasSnrm2(handle, (int)vector_elements,
                                           d_current, 1, &current_norm));
            residual_check_time += ipt_mixed_now_seconds() - residual_start;
        }

        if (current_norm <= 0.0f || !isfinite((double)current_norm)) {
            status = IPT_CUDA_INVALID_ARGUMENT;
            goto cleanup;
        }

        fixed_point_residual = (double)delta_norm / (double)current_norm;
        if (fixed_point_residual <= tol) {
            tmp = d_x;
            d_x = d_y;
            d_y = tmp;
            break;
        }

        tmp = d_x;
        d_x = d_y;
        d_y = tmp;
    }

    {
        int vector_blocks =
            ((int)vector_elements + block_size - 1) / block_size;

        half_to_double_kernel<<<vector_blocks, block_size>>>(
            d_x, d_initial, (int)vector_elements);
        MIXED_CUDA_CHECK(cudaGetLastError());
    }

    *d_initial_vectors_out = d_initial;
    if (iterations_done_out != NULL) {
        *iterations_done_out = iterations_done;
    }
    if (fixed_point_residual_out != NULL) {
        *fixed_point_residual_out = fixed_point_residual;
    }
    d_initial = NULL;

cleanup:
    if (residual_check_time_out != NULL) {
        *residual_check_time_out = residual_check_time;
    }

    cudaFree(d_matrix_half);
    cudaFree(d_x_a);
    cudaFree(d_x_b);
    cudaFree(d_diagonal);
    cudaFree(d_g);
    cudaFree(d_column_diagonal);
    cudaFree(d_delta);
    cudaFree(d_current);
    cudaFree(d_initial);

    if (status != IPT_CUDA_SUCCESS && d_initial_vectors_out != NULL) {
        *d_initial_vectors_out = NULL;
    }

    return status;
}

static int ipt_float_initial_device_tol(
    const double *d_matrix_col_major, int n, int k, double tol, int maxiter,
    cublasHandle_t handle, cublasComputeType_t compute_type,
    cublasGemmAlgo_t gemm_algo, double **d_initial_vectors_out,
    int *iterations_done_out, double *fixed_point_residual_out,
    double *residual_check_time_out)
{
    int status = IPT_CUDA_SUCCESS;
    int iterations_done = 0;
    size_t matrix_elements = 0;
    size_t vector_elements = 0;
    float *d_matrix_float = NULL;
    float *d_x_a = NULL;
    float *d_x_b = NULL;
    float *d_x = NULL;
    float *d_y = NULL;
    double *d_diagonal = NULL;
    double *d_initial = NULL;
    float *d_g = NULL;
    float *d_column_diagonal = NULL;
    float *d_delta = NULL;
    double fixed_point_residual = NAN;
    double residual_check_time = 0.0;
    const float one = 1.0f;
    const float zero = 0.0f;
    const int block_size = 256;

    if (d_initial_vectors_out != NULL) {
        *d_initial_vectors_out = NULL;
    }
    if (iterations_done_out != NULL) {
        *iterations_done_out = 0;
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
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    matrix_elements = (size_t)n * (size_t)n;
    vector_elements = (size_t)n * (size_t)k;
    if (matrix_elements > (size_t)INT_MAX ||
        vector_elements > (size_t)INT_MAX) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_matrix_float, matrix_elements * sizeof(float)));
    MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_x_a, vector_elements * sizeof(float)));
    MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_x_b, vector_elements * sizeof(float)));
    MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_diagonal, (size_t)n * sizeof(double)));
    MIXED_CUDA_CHECK(cudaMalloc((void **)&d_g, vector_elements * sizeof(float)));
    MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_column_diagonal, (size_t)k * sizeof(float)));
    MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_delta, vector_elements * sizeof(float)));
    MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_initial, vector_elements * sizeof(double)));

    {
        int matrix_blocks =
            ((int)matrix_elements + block_size - 1) / block_size;
        int vector_blocks =
            ((int)vector_elements + block_size - 1) / block_size;
        int diag_blocks = (n + block_size - 1) / block_size;

        double_to_float_offdiag_kernel<<<matrix_blocks, block_size>>>(
            d_matrix_col_major, d_matrix_float, n, (int)matrix_elements);
        MIXED_CUDA_CHECK(cudaGetLastError());

        set_identity_float_kernel<<<vector_blocks, block_size>>>(d_x_a, n, k);
        MIXED_CUDA_CHECK(cudaGetLastError());

        extract_diagonal_kernel<<<diag_blocks, block_size>>>(
            d_matrix_col_major, d_diagonal, n);
        MIXED_CUDA_CHECK(cudaGetLastError());

        build_g_float_from_double_diag_kernel<<<vector_blocks, block_size>>>(
            d_diagonal, d_g, n, k);
        MIXED_CUDA_CHECK(cudaGetLastError());
    }

    d_x = d_x_a;
    d_y = d_x_b;

    for (int iter = 0; iter < maxiter; ++iter) {
        int col_blocks = (k + block_size - 1) / block_size;
        int vector_blocks =
            ((int)vector_elements + block_size - 1) / block_size;
        float *tmp = NULL;
        float delta_norm = 0.0f;
        float current_norm = 0.0f;

        MIXED_CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &one, d_matrix_float,
            CUDA_R_32F, n, d_x, CUDA_R_32F, n, &zero, d_y, CUDA_R_32F, n,
            compute_type, gemm_algo));

        column_diagonal_after_e_float_kernel<<<col_blocks, block_size>>>(
            d_y, d_column_diagonal, n, k);
        MIXED_CUDA_CHECK(cudaGetLastError());

        ipt_update_float_kernel<<<vector_blocks, block_size>>>(
            d_y, d_x, d_g, d_column_diagonal, n, k);
        MIXED_CUDA_CHECK(cudaGetLastError());

        iterations_done = iter + 1;

        {
            double residual_start = 0.0;

            MIXED_CUDA_CHECK(cudaDeviceSynchronize());
            residual_start = ipt_mixed_now_seconds();

            float_delta_kernel<<<vector_blocks, block_size>>>(
                d_delta, d_y, d_x, (int)vector_elements);
            MIXED_CUDA_CHECK(cudaGetLastError());

            MIXED_CUBLAS_CHECK(cublasSnrm2(handle, (int)vector_elements,
                                           d_delta, 1, &delta_norm));
            MIXED_CUBLAS_CHECK(cublasSnrm2(handle, (int)vector_elements, d_x,
                                           1, &current_norm));
            residual_check_time += ipt_mixed_now_seconds() - residual_start;
        }

        if (current_norm <= 0.0f || !isfinite((double)current_norm)) {
            status = IPT_CUDA_INVALID_ARGUMENT;
            goto cleanup;
        }

        fixed_point_residual = (double)delta_norm / (double)current_norm;
        if (fixed_point_residual <= tol) {
            tmp = d_x;
            d_x = d_y;
            d_y = tmp;
            break;
        }

        tmp = d_x;
        d_x = d_y;
        d_y = tmp;
    }

    {
        int vector_blocks =
            ((int)vector_elements + block_size - 1) / block_size;

        float_to_double_kernel<<<vector_blocks, block_size>>>(
            d_x, d_initial, (int)vector_elements);
        MIXED_CUDA_CHECK(cudaGetLastError());
    }

    *d_initial_vectors_out = d_initial;
    if (iterations_done_out != NULL) {
        *iterations_done_out = iterations_done;
    }
    if (fixed_point_residual_out != NULL) {
        *fixed_point_residual_out = fixed_point_residual;
    }
    d_initial = NULL;

cleanup:
    if (residual_check_time_out != NULL) {
        *residual_check_time_out = residual_check_time;
    }

    cudaFree(d_matrix_float);
    cudaFree(d_x_a);
    cudaFree(d_x_b);
    cudaFree(d_diagonal);
    cudaFree(d_g);
    cudaFree(d_column_diagonal);
    cudaFree(d_delta);
    cudaFree(d_initial);

    if (status != IPT_CUDA_SUCCESS && d_initial_vectors_out != NULL) {
        *d_initial_vectors_out = NULL;
    }

    return status;
}

static int ipt_fp32_initial_device_tol(const double *d_matrix_col_major, int n,
                                       int k, double tol, int maxiter,
                                       cublasHandle_t handle,
                                       double **d_initial_vectors_out,
                                       int *iterations_done_out,
                                       double *fixed_point_residual_out,
                                       double *residual_check_time_out)
{
    return ipt_float_initial_device_tol(
        d_matrix_col_major, n, k, tol, maxiter, handle,
        CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT,
        d_initial_vectors_out, iterations_done_out,
        fixed_point_residual_out, residual_check_time_out);
}

static int ipt_tf32_initial_device_tol(const double *d_matrix_col_major, int n,
                                       int k, double tol, int maxiter,
                                       cublasHandle_t handle,
                                       double **d_initial_vectors_out,
                                       int *iterations_done_out,
                                       double *fixed_point_residual_out,
                                       double *residual_check_time_out)
{
    return ipt_float_initial_device_tol(
        d_matrix_col_major, n, k, tol, maxiter, handle,
        CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT,
        d_initial_vectors_out, iterations_done_out,
        fixed_point_residual_out, residual_check_time_out);
}

static int ipt_cuda_device_with_initial_tol_mixed(
    const double *d_matrix_col_major, const double *d_initial_vectors_col_major,
    int n, int k, double tol, int maxiter, cublasHandle_t handle,
    double **d_vectors_out, double **d_values_out, int *iterations_done_out,
    double *fixed_point_residual_out)
{
    int status = IPT_CUDA_SUCCESS;
    int iterations_done = 0;
    size_t vector_elements = 0;
    size_t vector_bytes = 0;
    double *d_x_a = NULL;
    double *d_x_b = NULL;
    double *d_diagonal = NULL;
    double *d_g = NULL;
    double *d_column_diagonal = NULL;
    double *d_delta = NULL;
    double *d_values = NULL;
    double *d_x = NULL;
    double *d_y = NULL;
    double fixed_point_residual = NAN;
    const double one = 1.0;
    const double zero = 0.0;
    const int block_size = 256;

    if (d_vectors_out != NULL) {
        *d_vectors_out = NULL;
    }
    if (d_values_out != NULL) {
        *d_values_out = NULL;
    }
    if (iterations_done_out != NULL) {
        *iterations_done_out = 0;
    }
    if (fixed_point_residual_out != NULL) {
        *fixed_point_residual_out = NAN;
    }

    if (d_matrix_col_major == NULL || d_initial_vectors_col_major == NULL ||
        handle == NULL || d_vectors_out == NULL || d_values_out == NULL ||
        n <= 0 || k <= 0 || k > n || tol <= 0.0 || maxiter <= 0) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    vector_elements = (size_t)n * (size_t)k;
    if (vector_elements > (size_t)INT_MAX) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    vector_bytes = vector_elements * sizeof(double);

    MIXED_CUDA_CHECK(cudaMalloc((void **)&d_x_a, vector_bytes));
    MIXED_CUDA_CHECK(cudaMalloc((void **)&d_x_b, vector_bytes));
    MIXED_CUDA_CHECK(cudaMalloc((void **)&d_diagonal, (size_t)n * sizeof(double)));
    MIXED_CUDA_CHECK(cudaMalloc((void **)&d_g, vector_bytes));
    MIXED_CUDA_CHECK(
        cudaMalloc((void **)&d_column_diagonal, (size_t)k * sizeof(double)));
    MIXED_CUDA_CHECK(cudaMalloc((void **)&d_delta, vector_bytes));
    MIXED_CUDA_CHECK(cudaMalloc((void **)&d_values, (size_t)k * sizeof(double)));
    MIXED_CUDA_CHECK(cudaMemcpy(d_x_a, d_initial_vectors_col_major,
                                vector_bytes, cudaMemcpyDeviceToDevice));

    {
        int vector_blocks =
            ((int)vector_elements + block_size - 1) / block_size;
        int diag_blocks = (n + block_size - 1) / block_size;

        extract_diagonal_kernel<<<diag_blocks, block_size>>>(
            d_matrix_col_major, d_diagonal, n);
        MIXED_CUDA_CHECK(cudaGetLastError());

        build_g_kernel<<<vector_blocks, block_size>>>(d_diagonal, d_g, n, k);
        MIXED_CUDA_CHECK(cudaGetLastError());
    }

    d_x = d_x_a;
    d_y = d_x_b;

    for (int iter = 0; iter < maxiter; ++iter) {
        int col_blocks = (k + block_size - 1) / block_size;
        int vector_blocks =
            ((int)vector_elements + block_size - 1) / block_size;
        double *tmp = NULL;
        double delta_norm = 0.0;
        double current_norm = 0.0;

        MIXED_CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k,
                                       n, &one, d_matrix_col_major, n, d_x, n,
                                       &zero, d_y, n));

        column_diagonal_after_d_kernel<<<col_blocks, block_size>>>(
            d_y, d_x, d_diagonal, d_column_diagonal, n, k);
        MIXED_CUDA_CHECK(cudaGetLastError());

        ipt_update_kernel<<<vector_blocks, block_size>>>(
            d_y, d_x, d_diagonal, d_g, d_column_diagonal, n, k);
        MIXED_CUDA_CHECK(cudaGetLastError());

        iterations_done = iter + 1;

        fixed_point_delta_kernel<<<vector_blocks, block_size>>>(
            d_delta, d_y, d_x, (int)vector_elements);
        MIXED_CUDA_CHECK(cudaGetLastError());

        MIXED_CUBLAS_CHECK(cublasDnrm2(handle, (int)vector_elements, d_delta,
                                       1, &delta_norm));
        MIXED_CUBLAS_CHECK(cublasDnrm2(handle, (int)vector_elements, d_x, 1,
                                       &current_norm));

        if (current_norm <= 0.0 || !isfinite(current_norm)) {
            status = IPT_CUDA_INVALID_ARGUMENT;
            goto cleanup;
        }

        fixed_point_residual = delta_norm / current_norm;
        if (fixed_point_residual <= tol) {
            tmp = d_x;
            d_x = d_y;
            d_y = tmp;
            break;
        }

        tmp = d_x;
        d_x = d_y;
        d_y = tmp;
    }

    MIXED_CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n,
                                   &one, d_matrix_col_major, n, d_x, n, &zero,
                                   d_y, n));

    {
        int col_blocks = (k + block_size - 1) / block_size;

        gather_values_kernel<<<col_blocks, block_size>>>(d_y, d_values, n, k);
        MIXED_CUDA_CHECK(cudaGetLastError());
    }

    *d_vectors_out = d_x;
    *d_values_out = d_values;
    if (iterations_done_out != NULL) {
        *iterations_done_out = iterations_done;
    }
    if (fixed_point_residual_out != NULL) {
        *fixed_point_residual_out = fixed_point_residual;
    }
    if (d_x == d_x_a) {
        d_x_a = NULL;
    } else if (d_x == d_x_b) {
        d_x_b = NULL;
    }
    d_values = NULL;

cleanup:
    cudaFree(d_x_a);
    cudaFree(d_x_b);
    cudaFree(d_diagonal);
    cudaFree(d_g);
    cudaFree(d_column_diagonal);
    cudaFree(d_delta);
    cudaFree(d_values);

    if (status != IPT_CUDA_SUCCESS) {
        if (d_vectors_out != NULL) {
            *d_vectors_out = NULL;
        }
        if (d_values_out != NULL) {
            *d_values_out = NULL;
        }
    }

    return status;
}

extern "C" int ipt_mixed_precision_cuda_device_tol(
    const double *d_matrix_col_major, int n, int k, double fp16_tol,
    int fp16_maxiter, double fp64_tol, int fp64_maxiter,
    cublasHandle_t handle, IPTMixedCudaDeviceResult *result)
{
    int status = IPT_CUDA_SUCCESS;
    double *d_initial = NULL;
    double *d_vectors = NULL;
    double *d_values = NULL;
    double total_start = 0.0;
    double stage_start = 0.0;
    double fp16_residual_check_time = 0.0;
    double raw_fp16_time = 0.0;

    ipt_mixed_reset_device_result(result);

    if (d_matrix_col_major == NULL || handle == NULL || result == NULL ||
        n <= 0 || k <= 0 || k > n || fp16_tol <= 0.0 || fp64_tol <= 0.0 ||
        fp16_maxiter <= 0 || fp64_maxiter <= 0) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    total_start = ipt_mixed_now_seconds();

    stage_start = ipt_mixed_now_seconds();
    status = ipt_fp16_initial_device_tol(
        d_matrix_col_major, n, k, fp16_tol, fp16_maxiter, handle, &d_initial,
        &result->fp16_iterations, &result->fp16_fixed_point_residual,
        &fp16_residual_check_time);
    if (status != IPT_CUDA_SUCCESS) {
        goto cleanup;
    }
    MIXED_CUDA_CHECK(cudaDeviceSynchronize());
    raw_fp16_time = ipt_mixed_now_seconds() - stage_start;
    result->fp16_time_sec = raw_fp16_time - fp16_residual_check_time;
    if (result->fp16_time_sec < 0.0) {
        result->fp16_time_sec = 0.0;
    }

    stage_start = ipt_mixed_now_seconds();
    status = ipt_cuda_device_with_initial_tol_mixed(
        d_matrix_col_major, d_initial, n, k, fp64_tol, fp64_maxiter, handle,
        &d_vectors, &d_values, &result->fp64_iterations,
        &result->fp64_fixed_point_residual);
    if (status != IPT_CUDA_SUCCESS) {
        goto cleanup;
    }
    MIXED_CUDA_CHECK(cudaDeviceSynchronize());
    result->fp64_time_sec = ipt_mixed_now_seconds() - stage_start;
    result->total_time_sec =
        ipt_mixed_now_seconds() - total_start - fp16_residual_check_time;
    if (result->total_time_sec < result->fp64_time_sec) {
        result->total_time_sec = result->fp16_time_sec + result->fp64_time_sec;
    }

    result->n = n;
    result->k = k;
    result->d_vectors = d_vectors;
    result->d_values = d_values;
    d_vectors = NULL;
    d_values = NULL;

cleanup:
    cudaFree(d_initial);
    cudaFree(d_vectors);
    cudaFree(d_values);

    if (status != IPT_CUDA_SUCCESS) {
        ipt_mixed_precision_cuda_free_device_result(result);
    }

    return status;
}

extern "C" int ipt_mixed_precision_fp32_cuda_device_tol(
    const double *d_matrix_col_major, int n, int k, double fp32_tol,
    int fp32_maxiter, double fp64_tol, int fp64_maxiter,
    cublasHandle_t handle, IPTMixedFP32CudaDeviceResult *result)
{
    int status = IPT_CUDA_SUCCESS;
    double *d_initial = NULL;
    double *d_vectors = NULL;
    double *d_values = NULL;
    double total_start = 0.0;
    double stage_start = 0.0;
    double fp32_residual_check_time = 0.0;
    double raw_fp32_time = 0.0;

    ipt_mixed_fp32_reset_device_result(result);

    if (d_matrix_col_major == NULL || handle == NULL || result == NULL ||
        n <= 0 || k <= 0 || k > n || fp32_tol <= 0.0 || fp64_tol <= 0.0 ||
        fp32_maxiter <= 0 || fp64_maxiter <= 0) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    total_start = ipt_mixed_now_seconds();

    stage_start = ipt_mixed_now_seconds();
    status = ipt_fp32_initial_device_tol(
        d_matrix_col_major, n, k, fp32_tol, fp32_maxiter, handle, &d_initial,
        &result->fp32_iterations, &result->fp32_fixed_point_residual,
        &fp32_residual_check_time);
    if (status != IPT_CUDA_SUCCESS) {
        goto cleanup;
    }
    MIXED_CUDA_CHECK(cudaDeviceSynchronize());
    raw_fp32_time = ipt_mixed_now_seconds() - stage_start;
    result->fp32_time_sec = raw_fp32_time - fp32_residual_check_time;
    if (result->fp32_time_sec < 0.0) {
        result->fp32_time_sec = 0.0;
    }

    stage_start = ipt_mixed_now_seconds();
    status = ipt_cuda_device_with_initial_tol_mixed(
        d_matrix_col_major, d_initial, n, k, fp64_tol, fp64_maxiter, handle,
        &d_vectors, &d_values, &result->fp64_iterations,
        &result->fp64_fixed_point_residual);
    if (status != IPT_CUDA_SUCCESS) {
        goto cleanup;
    }
    MIXED_CUDA_CHECK(cudaDeviceSynchronize());
    result->fp64_time_sec = ipt_mixed_now_seconds() - stage_start;
    result->total_time_sec =
        ipt_mixed_now_seconds() - total_start - fp32_residual_check_time;
    if (result->total_time_sec < result->fp64_time_sec) {
        result->total_time_sec = result->fp32_time_sec + result->fp64_time_sec;
    }

    result->n = n;
    result->k = k;
    result->d_vectors = d_vectors;
    result->d_values = d_values;
    d_vectors = NULL;
    d_values = NULL;

cleanup:
    cudaFree(d_initial);
    cudaFree(d_vectors);
    cudaFree(d_values);

    if (status != IPT_CUDA_SUCCESS) {
        ipt_mixed_precision_fp32_cuda_free_device_result(result);
    }

    return status;
}

extern "C" int ipt_mixed_precision_tf32_cuda_device_tol(
    const double *d_matrix_col_major, int n, int k, double tf32_tol,
    int tf32_maxiter, double fp64_tol, int fp64_maxiter,
    cublasHandle_t handle, IPTMixedTF32CudaDeviceResult *result)
{
    int status = IPT_CUDA_SUCCESS;
    double *d_initial = NULL;
    double *d_vectors = NULL;
    double *d_values = NULL;
    double total_start = 0.0;
    double stage_start = 0.0;
    double tf32_residual_check_time = 0.0;
    double raw_tf32_time = 0.0;

    ipt_mixed_tf32_reset_device_result(result);

    if (d_matrix_col_major == NULL || handle == NULL || result == NULL ||
        n <= 0 || k <= 0 || k > n || tf32_tol <= 0.0 || fp64_tol <= 0.0 ||
        tf32_maxiter <= 0 || fp64_maxiter <= 0) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    total_start = ipt_mixed_now_seconds();

    stage_start = ipt_mixed_now_seconds();
    status = ipt_tf32_initial_device_tol(
        d_matrix_col_major, n, k, tf32_tol, tf32_maxiter, handle, &d_initial,
        &result->tf32_iterations, &result->tf32_fixed_point_residual,
        &tf32_residual_check_time);
    if (status != IPT_CUDA_SUCCESS) {
        goto cleanup;
    }
    MIXED_CUDA_CHECK(cudaDeviceSynchronize());
    raw_tf32_time = ipt_mixed_now_seconds() - stage_start;
    result->tf32_time_sec = raw_tf32_time - tf32_residual_check_time;
    if (result->tf32_time_sec < 0.0) {
        result->tf32_time_sec = 0.0;
    }

    stage_start = ipt_mixed_now_seconds();
    status = ipt_cuda_device_with_initial_tol_mixed(
        d_matrix_col_major, d_initial, n, k, fp64_tol, fp64_maxiter, handle,
        &d_vectors, &d_values, &result->fp64_iterations,
        &result->fp64_fixed_point_residual);
    if (status != IPT_CUDA_SUCCESS) {
        goto cleanup;
    }
    MIXED_CUDA_CHECK(cudaDeviceSynchronize());
    result->fp64_time_sec = ipt_mixed_now_seconds() - stage_start;
    result->total_time_sec =
        ipt_mixed_now_seconds() - total_start - tf32_residual_check_time;
    if (result->total_time_sec < result->fp64_time_sec) {
        result->total_time_sec = result->tf32_time_sec + result->fp64_time_sec;
    }

    result->n = n;
    result->k = k;
    result->d_vectors = d_vectors;
    result->d_values = d_values;
    d_vectors = NULL;
    d_values = NULL;

cleanup:
    cudaFree(d_initial);
    cudaFree(d_vectors);
    cudaFree(d_values);

    if (status != IPT_CUDA_SUCCESS) {
        ipt_mixed_precision_tf32_cuda_free_device_result(result);
    }

    return status;
}

#undef MIXED_CUDA_CHECK
#undef MIXED_CUBLAS_CHECK
