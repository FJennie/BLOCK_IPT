#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ACX_CUDA_SUCCESS 0
#define ACX_CUDA_INVALID_ARGUMENT 1
#define ACX_CUDA_ALLOCATION_FAILED 2
#define ACX_CUDA_CUDA_ERROR 3
#define ACX_CUDA_CUBLAS_ERROR 4
#define ACX_CUDA_ZERO_DENOMINATOR 5
#define ACX_FIXED_POINT_RTOL 1.0e-12

typedef struct {
    int n;
    int k;
    int iterations;
    int f_calls;
    int converged;
    double time_sec;
    double iteration_time_sec;
    double finalization_time_sec;
    double max_residual;
    double fixed_point_residual;
    double *d_vectors;
    double *d_values;
} ACXCudaResult;

static double acx_now_seconds(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

extern "C" const char *acx_cuda_status_string(int status)
{
    switch (status) {
    case ACX_CUDA_SUCCESS:
        return "success";
    case ACX_CUDA_INVALID_ARGUMENT:
        return "invalid argument";
    case ACX_CUDA_ALLOCATION_FAILED:
        return "allocation failed";
    case ACX_CUDA_CUDA_ERROR:
        return "cuda error";
    case ACX_CUDA_CUBLAS_ERROR:
        return "cublas error";
    case ACX_CUDA_ZERO_DENOMINATOR:
        return "zero diagonal denominator";
    default:
        return "unknown error";
    }
}

extern "C" void acx_cuda_reset_result(ACXCudaResult *result)
{
    if (result == NULL) {
        return;
    }
    memset(result, 0, sizeof(*result));
    result->max_residual = NAN;
    result->fixed_point_residual = NAN;
}

extern "C" void acx_cuda_free_result(ACXCudaResult *result)
{
    if (result == NULL) {
        return;
    }
    cudaFree(result->d_vectors);
    cudaFree(result->d_values);
    acx_cuda_reset_result(result);
}

__global__ static void acx_set_identity_kernel(double *x, int n, int k)
{
    size_t total = (size_t)n * (size_t)k;
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + threadIdx.x;

    if (idx < total) {
        int row = (int)(idx % (size_t)n);
        int col = (int)(idx / (size_t)n);

        x[idx] = (row == col) ? 1.0 : 0.0;
    }
}

__global__ static void acx_extract_diagonal_kernel(const double *matrix,
                                                   double *diagonal, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        diagonal[idx] = matrix[idx + (size_t)idx * (size_t)n];
    }
}

__global__ static void acx_apply_metadata_kernel(const double *y,
                                                 const double *x,
                                                 const double *diagonal,
                                                 double *mx_diag,
                                                 double *update_diag, int n,
                                                 int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < k) {
        size_t idx = (size_t)col + (size_t)col * (size_t)n;
        double mx = y[idx];

        mx_diag[col] = mx;
        update_diag[col] = mx - diagonal[col] * x[idx];
    }
}

__global__ static void acx_apply_finalize_kernel(
    double *y, const double *x, const double *diagonal, const double *mx_diag,
    const double *update_diag, double *residual_sq, int *invalid_flag, int n,
    int k)
{
    size_t total = (size_t)n * (size_t)k;
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + threadIdx.x;

    if (idx < total) {
        int row = (int)(idx % (size_t)n);
        int col = (int)(idx / (size_t)n);
        double y_value = y[idx];
        double x_value = x[idx];
        double mx = mx_diag[col];
        double r = y_value - x_value * mx;

        atomicAdd(residual_sq + col, r * r);
        if (row == col) {
            y[idx] = 1.0;
        } else {
            double denom = diagonal[col] - diagonal[row];

            if (fabs(denom) <= DBL_EPSILON) {
                *invalid_flag = 1;
                y[idx] = 0.0;
            } else {
                y[idx] = (y_value - diagonal[row] * x_value -
                          x_value * update_diag[col]) /
                         denom;
            }
        }
    }
}

__global__ static void acx_sqrt_kernel(double *residuals,
                                       const double *residual_sq, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < k) {
        residuals[idx] = sqrt(residual_sq[idx]);
    }
}

__global__ static void acx_extract_values_kernel(const double *work,
                                                 double *values, int n, int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < k) {
        values[col] = work[col + (size_t)col * (size_t)n];
    }
}

__global__ static void acx_fixed_point_delta_kernel(double *delta,
                                                    const double *next,
                                                    const double *current,
                                                    size_t total)
{
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + threadIdx.x;

    if (idx < total) {
        delta[idx] = next[idx] - current[idx];
    }
}

__global__ static void acx_update_order2_kernel(double *x, const double *f1,
                                                const double *f2, int n,
                                                int k)
{
    extern __shared__ double shared[];
    double *num_shared = shared;
    double *den_shared = shared + blockDim.x;
    int col = blockIdx.x;
    int tid = threadIdx.x;
    double numerator = 0.0;
    double denominator = 0.0;

    if (col >= k) {
        return;
    }

    for (int row = tid; row < n; row += blockDim.x) {
        size_t idx = (size_t)row + (size_t)col * (size_t)n;
        double d1 = f1[idx] - x[idx];
        double d2 = f2[idx] - 2.0 * f1[idx] + x[idx];

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

    double sigma = 0.0;
    if (den_shared[0] > 0.0) {
        sigma = fabs(num_shared[0] / den_shared[0]);
    }
    double sigma2 = sigma * sigma;

    for (int row = tid; row < n; row += blockDim.x) {
        size_t idx = (size_t)row + (size_t)col * (size_t)n;
        double d1 = f1[idx] - x[idx];
        double d2 = f2[idx] - 2.0 * f1[idx] + x[idx];

        x[idx] += 2.0 * sigma * d1 + sigma2 * d2;
    }
}

__global__ static void acx_update_order3_kernel(double *x, const double *f1,
                                                const double *f2,
                                                const double *f3, int n,
                                                int k)
{
    extern __shared__ double shared[];
    double *num_shared = shared;
    double *den_shared = shared + blockDim.x;
    int col = blockIdx.x;
    int tid = threadIdx.x;
    double numerator = 0.0;
    double denominator = 0.0;

    if (col >= k) {
        return;
    }

    for (int row = tid; row < n; row += blockDim.x) {
        size_t idx = (size_t)row + (size_t)col * (size_t)n;
        double d2 = f2[idx] - 2.0 * f1[idx] + x[idx];
        double d3 = f3[idx] - 3.0 * f2[idx] + 3.0 * f1[idx] - x[idx];

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

    double sigma = 0.0;
    if (den_shared[0] > 0.0) {
        sigma = fabs(num_shared[0] / den_shared[0]);
    }
    double sigma2 = sigma * sigma;
    double sigma3 = sigma2 * sigma;

    for (int row = tid; row < n; row += blockDim.x) {
        size_t idx = (size_t)row + (size_t)col * (size_t)n;
        double d1 = f1[idx] - x[idx];
        double d2 = f2[idx] - 2.0 * f1[idx] + x[idx];
        double d3 = f3[idx] - 3.0 * f2[idx] + 3.0 * f1[idx] - x[idx];

        x[idx] +=
            3.0 * sigma * d1 + 3.0 * sigma2 * d2 + sigma3 * d3;
    }
}

static int acx_apply_f_device(cublasHandle_t handle, const double *d_matrix,
                              const double *d_x, double *d_y,
                              const double *d_diagonal, double *d_mx_diag,
                              double *d_update_diag, double *d_residual_sq,
                              double *d_residuals, int *d_invalid_flag, int n,
                              int k)
{
    const double one = 1.0;
    const double zero = 0.0;
    cublasStatus_t cublas_status;
    cudaError_t cuda_status;
    int block = 256;
    int grid_cols = (k + block - 1) / block;
    int grid_elements =
        (int)(((size_t)n * (size_t)k + (size_t)block - 1) / (size_t)block);
    int invalid_host = 0;

    cublas_status =
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &one, d_matrix,
                    n, d_x, n, &zero, d_y, n);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        return ACX_CUDA_CUBLAS_ERROR;
    }

    cuda_status = cudaMemset(d_residual_sq, 0, (size_t)k * sizeof(double));
    if (cuda_status != cudaSuccess) {
        return ACX_CUDA_CUDA_ERROR;
    }
    cuda_status = cudaMemset(d_invalid_flag, 0, sizeof(int));
    if (cuda_status != cudaSuccess) {
        return ACX_CUDA_CUDA_ERROR;
    }

    acx_apply_metadata_kernel<<<grid_cols, block>>>(
        d_y, d_x, d_diagonal, d_mx_diag, d_update_diag, n, k);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        return ACX_CUDA_CUDA_ERROR;
    }

    acx_apply_finalize_kernel<<<grid_elements, block>>>(
        d_y, d_x, d_diagonal, d_mx_diag, d_update_diag, d_residual_sq,
        d_invalid_flag, n, k);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        return ACX_CUDA_CUDA_ERROR;
    }

    acx_sqrt_kernel<<<grid_cols, block>>>(d_residuals, d_residual_sq, k);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        return ACX_CUDA_CUDA_ERROR;
    }

    cuda_status = cudaMemcpy(&invalid_host, d_invalid_flag, sizeof(int),
                             cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        return ACX_CUDA_CUDA_ERROR;
    }
    return invalid_host ? ACX_CUDA_ZERO_DENOMINATOR : ACX_CUDA_SUCCESS;
}

static int acx_finalize_values(cublasHandle_t handle, const double *d_matrix,
                               const double *d_vectors, double *d_work,
                               double *d_values, int n, int k)
{
    const double one = 1.0;
    const double zero = 0.0;
    int block = 256;
    int grid = (k + block - 1) / block;
    cublasStatus_t cublas_status;
    cudaError_t cuda_status;

    cublas_status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n,
                                &one, d_matrix, n, d_vectors, n, &zero,
                                d_work, n);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        return ACX_CUDA_CUBLAS_ERROR;
    }

    acx_extract_values_kernel<<<grid, block>>>(d_work, d_values, n, k);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        return ACX_CUDA_CUDA_ERROR;
    }
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        return ACX_CUDA_CUDA_ERROR;
    }
    return ACX_CUDA_SUCCESS;
}

extern "C" int acx_cuda_device_tol(const double *d_matrix_col_major, int n,
                                   int k, double tol, int maxiter,
                                   cublasHandle_t handle,
                                   ACXCudaResult *result)
{
    int status = ACX_CUDA_SUCCESS;
    int own_handle = 0;
    size_t elements = 0;
    double *d_x = NULL;
    double *d_f1 = NULL;
    double *d_f2 = NULL;
    double *d_f3 = NULL;
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
    total_start = acx_now_seconds();

    if (d_matrix_col_major == NULL || n <= 0 || k <= 0 || k > n ||
        tol <= 0.0 || maxiter <= 0) {
        return ACX_CUDA_INVALID_ARGUMENT;
    }

    if (handle == NULL) {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            return ACX_CUDA_CUBLAS_ERROR;
        }
        own_handle = 1;
    }

    elements = (size_t)n * (size_t)k;
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
    acx_set_identity_kernel<<<grid_elements, block>>>(d_x, n, k);
    if (cudaGetLastError() != cudaSuccess) {
        status = ACX_CUDA_CUDA_ERROR;
        goto cleanup;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        status = ACX_CUDA_CUDA_ERROR;
        goto cleanup;
    }

    iteration_start = acx_now_seconds();
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
    }
    result->iteration_time_sec = acx_now_seconds() - iteration_start;

    finalization_start = acx_now_seconds();
    status = acx_finalize_values(handle, d_matrix_col_major, d_f1, d_f2,
                                 d_values, n, k);
    if (status != ACX_CUDA_SUCCESS) {
        goto cleanup;
    }
    result->finalization_time_sec = acx_now_seconds() - finalization_start;

    result->n = n;
    result->k = k;
    result->time_sec = acx_now_seconds() - total_start;
    result->d_vectors = d_f1;
    result->d_values = d_values;
    d_f1 = NULL;
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
    if (own_handle) {
        cublasDestroy(handle);
    }
    if (status != ACX_CUDA_SUCCESS) {
        acx_cuda_free_result(result);
    }
    return status;
}
