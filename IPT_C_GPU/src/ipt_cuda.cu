#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse_v2.h>

#include <chrono>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "ipt_preparation.cu"

static double ipt_cuda_env_double(const char *name, double default_value)
{
    const char *raw = getenv(name);
    char *end = NULL;
    double value = 0.0;

    if (raw == NULL || raw[0] == '\0') {
        return default_value;
    }
    value = strtod(raw, &end);
    if (end == raw || value < 0.0 || !isfinite(value)) {
        return default_value;
    }
    return value;
}

static int ipt_cuda_env_int(const char *name, int default_value)
{
    const char *raw = getenv(name);
    char *end = NULL;
    long value = 0;

    if (raw == NULL || raw[0] == '\0') {
        return default_value;
    }
    value = strtol(raw, &end, 10);
    if (end == raw || value < 0 || value > INT_MAX) {
        return default_value;
    }
    return (int)value;
}

static int ipt_cuda_env_flag(const char *name)
{
    const char *raw = getenv(name);

    if (raw == NULL || raw[0] == '\0') {
        return 0;
    }
    if (strcmp(raw, "0") == 0 || strcmp(raw, "false") == 0 ||
        strcmp(raw, "FALSE") == 0 || strcmp(raw, "no") == 0 ||
        strcmp(raw, "NO") == 0) {
        return 0;
    }
    return 1;
}

#ifndef IPT_CUDA_DECLS
#define IPT_CUDA_DECLS

#ifdef __cplusplus
extern "C" {
#endif

enum {
    IPT_CUDA_SUCCESS = 0,
    IPT_CUDA_INVALID_ARGUMENT = 1,
    IPT_CUDA_ALLOCATION_FAILED = 2,
    IPT_CUDA_CUDA_ERROR = 3,
    IPT_CUDA_CUBLAS_ERROR = 4,
    IPT_CUDA_CUSPARSE_ERROR = 5,
    IPT_CUDA_CUSOLVER_ERROR = 6
};

typedef struct {
    int n;
    int k;
    int iterations;
    double *vectors; /* Column-major n-by-k eigenvector/eigenmatrix estimate. */
    double *values;  /* Length-k diagonal of M * vectors. */
    int basis_cols;  /* Rayleigh-Ritz/subspace columns used to produce k pairs. */
    long long matvecs;
    double preparation_time_sec;
    double transfer_setup_time_sec;
    double iteration_time_sec;
    double rayleigh_ritz_time_sec;
    double solve_time_sec;
    double fixed_point_residual;
    int oversample;
    int rayleigh_ritz_used_qr;
    double basis_orthogonality_frobenius_error;
    double basis_orthogonality_max_abs_error;
    double ritz_vectors_orthogonality_frobenius_error;
    double ritz_vectors_orthogonality_max_abs_error;
    int basis_has_nan_or_inf;
    int ritz_vectors_has_nan_or_inf;
    int davidson_attempted;
    int davidson_accepted;
    int davidson_target_index;
    double davidson_residual_before;
    double davidson_residual_after;
    double davidson_denom_clip;
    int adaptive_block_enabled;
    double adaptive_coupling_tau;
    int adaptive_limit_hit;
    int adaptive_target_block_start;
    int adaptive_target_block_end;
    char adaptive_added_indices[256];
} IPTCudaResult;

const char *ipt_cuda_status_string(int status);
void ipt_cuda_free_result(IPTCudaResult *result);
void ipt_cuda_free_device_result(double *d_vectors, double *d_values);
int ipt_cuda(const double *matrix_col_major, int n, int k, int iterations,
             IPTCudaResult *result);
int ipt_cuda_tol(const double *matrix_col_major, int n, int k, double tol,
                 int maxiter, IPTCudaResult *result);
int ipt_cuda_with_initial(const double *matrix_col_major,
                          const double *initial_vectors_col_major, int n,
                          int k, int iterations, IPTCudaResult *result);
int ipt_cuda_with_initial_tol(const double *matrix_col_major,
                              const double *initial_vectors_col_major, int n,
                              int k, double tol, int maxiter,
                              IPTCudaResult *result);
int ipt_cuda_device(const double *d_matrix_col_major, int n, int k,
                    int iterations, cublasHandle_t handle,
                    double **d_vectors_out, double **d_values_out);
int ipt_cuda_device_tol(const double *d_matrix_col_major, int n, int k,
                        double tol, int maxiter, cublasHandle_t handle,
                        double **d_vectors_out, double **d_values_out,
                        int *iterations_done_out,
                        double *fixed_point_residual_out);
int ipt_cuda_sparse_csc(const int *col_ptr, const int *row_ind,
                        const double *matrix_values, int n, int k, int nnz,
                        int iterations, IPTCudaResult *result);
int ipt_cuda_sparse_csc_tol(const int *col_ptr, const int *row_ind,
                            const double *matrix_values, int n, int k,
                            int nnz, double tol, int maxiter,
                            IPTCudaResult *result);
int ipt_cuda_sparse_csc_with_initial(
    const int *col_ptr, const int *row_ind, const double *matrix_values,
    const double *initial_vectors_col_major, int n, int k, int nnz,
    int iterations, IPTCudaResult *result);
int ipt_cuda_sparse_csc_with_initial_tol(
    const int *col_ptr, const int *row_ind, const double *matrix_values,
    const double *initial_vectors_col_major, int n, int k, int nnz,
    double tol, int maxiter, IPTCudaResult *result);
int ipt_cuda_sparse_csc_device(
    const int *d_col_ptr, const int *d_row_ind,
    const double *d_matrix_values, int n, int k, int nnz, int iterations,
    cublasHandle_t handle, double **d_vectors_out, double **d_values_out);
int ipt_cuda_sparse_csc_device_tol(
    const int *d_col_ptr, const int *d_row_ind,
    const double *d_matrix_values, int n, int k, int nnz, double tol,
    int maxiter, cublasHandle_t handle, double **d_vectors_out,
    double **d_values_out, int *iterations_done_out,
    double *fixed_point_residual_out);

#ifdef __cplusplus
}
#endif

#endif

static const char *cublas_status_name(cublasStatus_t status)
{
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
        return "CUBLAS_STATUS_UNKNOWN";
    }
}

static const char *cusparse_status_name(cusparseStatus_t status)
{
    switch (status) {
    case CUSPARSE_STATUS_SUCCESS:
        return "CUSPARSE_STATUS_SUCCESS";
    case CUSPARSE_STATUS_NOT_INITIALIZED:
        return "CUSPARSE_STATUS_NOT_INITIALIZED";
    case CUSPARSE_STATUS_ALLOC_FAILED:
        return "CUSPARSE_STATUS_ALLOC_FAILED";
    case CUSPARSE_STATUS_INVALID_VALUE:
        return "CUSPARSE_STATUS_INVALID_VALUE";
    case CUSPARSE_STATUS_ARCH_MISMATCH:
        return "CUSPARSE_STATUS_ARCH_MISMATCH";
    case CUSPARSE_STATUS_MAPPING_ERROR:
        return "CUSPARSE_STATUS_MAPPING_ERROR";
    case CUSPARSE_STATUS_EXECUTION_FAILED:
        return "CUSPARSE_STATUS_EXECUTION_FAILED";
    case CUSPARSE_STATUS_INTERNAL_ERROR:
        return "CUSPARSE_STATUS_INTERNAL_ERROR";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSPARSE_STATUS_NOT_SUPPORTED:
        return "CUSPARSE_STATUS_NOT_SUPPORTED";
    case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:
        return "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
    default:
        return "CUSPARSE_STATUS_UNKNOWN";
    }
}

// 错误码转字符串
extern "C" const char *ipt_cuda_status_string(int status)
{
    switch (status) {
    case IPT_CUDA_SUCCESS:
        return "success";
    case IPT_CUDA_INVALID_ARGUMENT:
        return "invalid argument";
    case IPT_CUDA_ALLOCATION_FAILED:
        return "allocation failed";
    case IPT_CUDA_CUDA_ERROR:
        return "CUDA runtime error";
    case IPT_CUDA_CUBLAS_ERROR:
        return "cuBLAS error";
    case IPT_CUDA_CUSPARSE_ERROR:
        return "cuSPARSE error";
    case IPT_CUDA_CUSOLVER_ERROR:
        return "cuSOLVER error";
    default:
        return "unknown error";
    }
}

// 释放 host 侧 IPTCudaResult
extern "C" void ipt_cuda_free_result(IPTCudaResult *result)
{
    if (result == NULL) {
        return;
    }

    free(result->vectors);
    free(result->values);

    result->n = 0;
    result->k = 0;
    result->iterations = 0;
    result->vectors = NULL;
    result->values = NULL;
    result->basis_cols = 0;
    result->matvecs = 0;
    result->preparation_time_sec = 0.0;
    result->transfer_setup_time_sec = 0.0;
    result->iteration_time_sec = 0.0;
    result->rayleigh_ritz_time_sec = 0.0;
    result->solve_time_sec = 0.0;
    result->fixed_point_residual = NAN;
    result->oversample = 0;
    result->rayleigh_ritz_used_qr = 0;
    result->basis_orthogonality_frobenius_error = NAN;
    result->basis_orthogonality_max_abs_error = NAN;
    result->ritz_vectors_orthogonality_frobenius_error = NAN;
    result->ritz_vectors_orthogonality_max_abs_error = NAN;
    result->basis_has_nan_or_inf = 0;
    result->ritz_vectors_has_nan_or_inf = 0;
    result->davidson_attempted = 0;
    result->davidson_accepted = 0;
    result->davidson_target_index = -1;
    result->davidson_residual_before = NAN;
    result->davidson_residual_after = NAN;
    result->davidson_denom_clip = NAN;
    result->adaptive_block_enabled = 0;
    result->adaptive_coupling_tau = NAN;
    result->adaptive_limit_hit = 0;
    result->adaptive_target_block_start = -1;
    result->adaptive_target_block_end = -1;
    result->adaptive_added_indices[0] = '\0';
}

// 释放 device 侧 d_vectors/d_values
extern "C" void ipt_cuda_free_device_result(double *d_vectors,
                                             double *d_values)
{
    cudaFree(d_vectors);
    cudaFree(d_values);
}

// 生成初始 n x k 单位列向量
__global__ static void set_identity_kernel(double *x, int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;

    if (idx >= total) {
        return;
    }

    int row = idx % n;
    int col = idx / n;
    x[idx] = (row == col) ? 1.0 : 0.0;
}

// 提取矩阵对角线
__global__ static void extract_diagonal_kernel(const double *matrix,
                                               double *diagonal, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        diagonal[row] = matrix[row + row * n];
    }
}

// 构造 G 更新因子：g[row, col] = 1 / (diagonal[col] - diagonal[row])
__global__ static void build_g_kernel(const double *diagonal, double *g, int n,
                                      int k, double min_gap_allowed,
                                      int *bad_gap)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;

    if (idx >= total) {
        return;
    }

    int row = idx % n;
    int col = idx / n;

    if (row == col) {
        g[idx] = 0.0;
    } else {
        double gap = diagonal[col] - diagonal[row];
        if (!isfinite(gap) || fabs(gap) <= min_gap_allowed) {
            if (bad_gap != NULL) {
                atomicExch(bad_gap, 1);
            }
            g[idx] = NAN;
        } else {
            g[idx] = 1.0 / gap;
        }
    }
}

// 为公式里的 D(ΔZ) 准备每一列的对角修正量。
// y[diag_idx] = (M z_i)_i
// x[diag_idx] = (z_i)_i
// diagonal[col] = D_ii
// 所以这一步是在计算(M z_i)_i - D_ii (z_i)_i = ((M - D) z_i)_i = (Δ z_i)_i
__global__ static void column_diagonal_after_d_kernel(const double *y,
                                                      const double *x,
                                                      const double *diagonal,
                                                      double *column_diagonal,
                                                      int n, int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < k) {
        int diag_idx = col + col * n;
        // column_diagonal[col] 存 D(ΔZ) 的第 col 个对角元素，用它更新非对角元素
        column_diagonal[col] = y[diag_idx] - diagonal[col] * x[diag_idx];
    }
}

// 执行 IPT 迭代更新
__global__ static void ipt_update_kernel(double *y, const double *x,
                                         const double *diagonal,
                                         const double *g,
                                         const double *column_diagonal, int n,
                                         int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;

    if (idx >= total) {
        return;
    }

    int row = idx % n;
    int col = idx / n;

    if (row == col) {
        y[idx] = 1.0;
        return;
    }
    // column_diagonal[col] 存 D(ΔZ) 的第 col 个对角元素
    y[idx] =
        (y[idx] - diagonal[row] * x[idx] - x[idx] * column_diagonal[col]) *
        g[idx];
}

// 从 M * vectors 的对角位置取近似特征值
__global__ static void gather_values_kernel(const double *mx, double *values,
                                            int n, int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < k) {
        values[col] = mx[col + col * n];
    }
}

// 形成 Algorithm 3.2 的固定点残差向量 F(Z) - Z
__global__ static void fixed_point_delta_kernel(double *delta,
                                                const double *next,
                                                const double *current,
                                                int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        delta[idx] = next[idx] - current[idx];
    }
}

__global__ static void extract_csc_diagonal_kernel(const int *col_ptr,
                                                   const int *row_ind,
                                                   const double *values,
                                                   double *diagonal, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= n) {
        return;
    }

    double diag = 0.0;
    for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
        if (row_ind[p] == col) {
            diag = values[p];
            break;
        }
    }

    diagonal[col] = diag;
}

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t cuda_check_status = (call);                                \
        if (cuda_check_status != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(cuda_check_status));                    \
            status = IPT_CUDA_CUDA_ERROR;                                      \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t cublas_check_status = (call);                           \
        if (cublas_check_status != CUBLAS_STATUS_SUCCESS) {                    \
            fprintf(stderr, "cuBLAS error at %s:%d: %s\n", __FILE__,          \
                    __LINE__, cublas_status_name(cublas_check_status));        \
            status = IPT_CUDA_CUBLAS_ERROR;                                    \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

#define CUSPARSE_CHECK(call)                                                   \
    do {                                                                       \
        cusparseStatus_t cusparse_check_status = (call);                       \
        if (cusparse_check_status != CUSPARSE_STATUS_SUCCESS) {                \
            fprintf(stderr, "cuSPARSE error at %s:%d: %s\n", __FILE__,        \
                    __LINE__, cusparse_status_name(cusparse_check_status));    \
            status = IPT_CUDA_CUSPARSE_ERROR;                                  \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

#define CUSOLVER_CHECK(call)                                                   \
    do {                                                                       \
        cusolverStatus_t cusolver_check_status = (call);                       \
        if (cusolver_check_status != CUSOLVER_STATUS_SUCCESS) {                \
            fprintf(stderr, "cuSOLVER error at %s:%d: status=%d\n",           \
                    __FILE__, __LINE__, (int)cusolver_check_status);            \
            status = IPT_CUDA_CUSOLVER_ERROR;                                  \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

static cusparseStatus_t ipt_cusparse_spmm(cusparseHandle_t handle,
                                          cusparseSpMatDescr_t matrix,
                                          cusparseDnMatDescr_t x_desc,
                                          cusparseDnMatDescr_t y_desc,
                                          double *d_x, double *d_y,
                                          void *buffer)
{
    const double one = 1.0;
    const double zero = 0.0;
    cusparseStatus_t status = cusparseDnMatSetValues(x_desc, d_x);

    if (status != CUSPARSE_STATUS_SUCCESS) {
        return status;
    }

    status = cusparseDnMatSetValues(y_desc, d_y);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return status;
    }

    return cusparseSpMM(handle, CUSPARSE_OPERATION_TRANSPOSE,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matrix,
                        x_desc, &zero, y_desc, CUDA_R_64F,
                        CUSPARSE_SPMM_ALG_DEFAULT, buffer);
}

// d_matrix 已经在 GPU -> 调 GPU kernel/cuBLAS -> 结果留在 GPU
static int ipt_cuda_device_impl(const double *d_matrix_col_major, int n, int k,
                                int maxiter, double tol, int use_tolerance,
                                cublasHandle_t handle, double **d_vectors_out,
                                double **d_values_out,
                                int *iterations_done_out,
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

    if (d_matrix_col_major == NULL || handle == NULL || d_vectors_out == NULL ||
        d_values_out == NULL || n <= 0 || k <= 0 || k > n ||
        maxiter < 0 || (use_tolerance && (tol <= 0.0 || maxiter <= 0))) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    vector_elements = (size_t)n * (size_t)k;

    if (vector_elements > (size_t)INT_MAX) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    vector_bytes = vector_elements * sizeof(double);

    CUDA_CHECK(cudaMalloc((void **)&d_x_a, vector_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_x_b, vector_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_diagonal, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_g, vector_bytes));
    CUDA_CHECK(
        cudaMalloc((void **)&d_column_diagonal, (size_t)k * sizeof(double)));
    if (use_tolerance) {
        CUDA_CHECK(cudaMalloc((void **)&d_delta, vector_bytes));
    }
    CUDA_CHECK(cudaMalloc((void **)&d_values, (size_t)k * sizeof(double)));

    {
        int blocks = ((int)vector_elements + block_size - 1) / block_size;
        int diag_blocks = (n + block_size - 1) / block_size;

        set_identity_kernel<<<blocks, block_size>>>(d_x_a, n, k);
        CUDA_CHECK(cudaGetLastError());

        extract_diagonal_kernel<<<diag_blocks, block_size>>>(
            d_matrix_col_major, d_diagonal, n);
        CUDA_CHECK(cudaGetLastError());

        build_g_kernel<<<blocks, block_size>>>(d_diagonal, d_g, n, k, 0.0,
                                               NULL);
        CUDA_CHECK(cudaGetLastError());
    }

    d_x = d_x_a;
    d_y = d_x_b;

    for (int iter = 0; iter < maxiter; ++iter) {
        int col_blocks = (k + block_size - 1) / block_size;
        int vector_blocks = ((int)vector_elements + block_size - 1) / block_size;
        double *tmp = NULL;

        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n,
                                 &one, d_matrix_col_major, n, d_x, n, &zero,
                                 d_y, n));

        column_diagonal_after_d_kernel<<<col_blocks, block_size>>>(
            d_y, d_x, d_diagonal, d_column_diagonal, n, k);
        CUDA_CHECK(cudaGetLastError());

        ipt_update_kernel<<<vector_blocks, block_size>>>(
            d_y, d_x, d_diagonal, d_g, d_column_diagonal, n, k);
        CUDA_CHECK(cudaGetLastError());

        iterations_done = iter + 1;

        if (use_tolerance) {
            double delta_norm = 0.0;
            double current_norm = 0.0;

            fixed_point_delta_kernel<<<vector_blocks, block_size>>>(
                d_delta, d_y, d_x, (int)vector_elements);
            CUDA_CHECK(cudaGetLastError());

            CUBLAS_CHECK(cublasDnrm2(handle, (int)vector_elements, d_delta, 1,
                                     &delta_norm));
            CUBLAS_CHECK(cublasDnrm2(handle, (int)vector_elements, d_x, 1,
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
        }

        tmp = d_x;
        d_x = d_y;
        d_y = tmp;
    }

    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &one,
                             d_matrix_col_major, n, d_x, n, &zero, d_y, n));

    {
        int col_blocks = (k + block_size - 1) / block_size;

        gather_values_kernel<<<col_blocks, block_size>>>(d_y, d_values, n, k);
        CUDA_CHECK(cudaGetLastError());
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

extern "C" int ipt_cuda_device(const double *d_matrix_col_major, int n, int k,
                                int iterations, cublasHandle_t handle,
                                double **d_vectors_out, double **d_values_out)
{
    return ipt_cuda_device_impl(d_matrix_col_major, n, k, iterations, 0.0, 0,
                                handle, d_vectors_out, d_values_out, NULL,
                                NULL);
}

extern "C" int ipt_cuda_device_tol(const double *d_matrix_col_major, int n,
                                    int k, double tol, int maxiter,
                                    cublasHandle_t handle,
                                    double **d_vectors_out,
                                    double **d_values_out,
                                    int *iterations_done_out,
                                    double *fixed_point_residual_out)
{
    return ipt_cuda_device_impl(d_matrix_col_major, n, k, maxiter, tol, 1,
                                handle, d_vectors_out, d_values_out,
                                iterations_done_out,
                                fixed_point_residual_out);
}

// host matrix -> 拷到 GPU -> 调 GPU kernel/cuBLAS -> 拷回 host
extern "C" int ipt_cuda(const double *matrix_col_major, int n, int k,
                         int iterations, IPTCudaResult *result)
{
    return ipt_cuda_with_initial(matrix_col_major, NULL, n, k, iterations,
                                 result);
}

extern "C" int ipt_cuda_tol(const double *matrix_col_major, int n, int k,
                             double tol, int maxiter, IPTCudaResult *result)
{
    return ipt_cuda_with_initial_tol(matrix_col_major, NULL, n, k, tol,
                                     maxiter, result);
}

static int ipt_cuda_with_initial_impl(const double *matrix_col_major,
                                      const double *initial_vectors_col_major,
                                      int n, int k, int maxiter, double tol,
                                      int use_tolerance,
                                      IPTCudaResult *result)
{
    int status = IPT_CUDA_SUCCESS;
    int iterations_done = 0;
    size_t matrix_elements = 0;
    size_t vector_elements = 0;
    size_t matrix_bytes = 0;
    size_t vector_bytes = 0;
    cublasHandle_t handle = NULL;
    double *d_matrix = NULL;
    double *d_x_a = NULL;
    double *d_x_b = NULL;
    double *d_diagonal = NULL;
    double *d_g = NULL;
    double *d_column_diagonal = NULL;
    double *d_delta = NULL;
    double *d_values = NULL;
    double *d_x = NULL;
    double *d_y = NULL;
    const double one = 1.0;
    const double zero = 0.0;
    const int block_size = 256;

    if (result != NULL) {
        result->n = 0;
        result->k = 0;
        result->iterations = 0;
        result->vectors = NULL;
        result->values = NULL;
        result->basis_cols = 0;
        result->fixed_point_residual = NAN;
    }

    if (matrix_col_major == NULL || result == NULL || n <= 0 || k <= 0 ||
        k > n || maxiter < 0 ||
        (use_tolerance && (tol <= 0.0 || maxiter <= 0))) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    matrix_elements = (size_t)n * (size_t)n;
    vector_elements = (size_t)n * (size_t)k;

    if (matrix_elements > (size_t)INT_MAX ||
        vector_elements > (size_t)INT_MAX) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    matrix_bytes = matrix_elements * sizeof(double);
    vector_bytes = vector_elements * sizeof(double);

    result->vectors = (double *)calloc(vector_elements, sizeof(double));
    result->values = (double *)calloc((size_t)k, sizeof(double));

    if (result->vectors == NULL || result->values == NULL) {
        status = IPT_CUDA_ALLOCATION_FAILED;
        goto cleanup;
    }

    CUDA_CHECK(cudaMalloc((void **)&d_matrix, matrix_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_x_a, vector_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_x_b, vector_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_diagonal, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_g, vector_bytes));
    CUDA_CHECK(
        cudaMalloc((void **)&d_column_diagonal, (size_t)k * sizeof(double)));
    if (use_tolerance) {
        CUDA_CHECK(cudaMalloc((void **)&d_delta, vector_bytes));
    }
    CUDA_CHECK(cudaMalloc((void **)&d_values, (size_t)k * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_matrix, matrix_col_major, matrix_bytes,
                          cudaMemcpyHostToDevice));

    if (initial_vectors_col_major == NULL) {
        int blocks = ((int)vector_elements + block_size - 1) / block_size;
        set_identity_kernel<<<blocks, block_size>>>(d_x_a, n, k);
        CUDA_CHECK(cudaGetLastError());
    } else {
        CUDA_CHECK(cudaMemcpy(d_x_a, initial_vectors_col_major, vector_bytes,
                              cudaMemcpyHostToDevice));
    }

    {
        int diag_blocks = (n + block_size - 1) / block_size;
        int vector_blocks = ((int)vector_elements + block_size - 1) / block_size;

        extract_diagonal_kernel<<<diag_blocks, block_size>>>(d_matrix,
                                                             d_diagonal, n);
        CUDA_CHECK(cudaGetLastError());

        build_g_kernel<<<vector_blocks, block_size>>>(d_diagonal, d_g, n, k,
                                                      0.0, NULL);
        CUDA_CHECK(cudaGetLastError());
    }

    CUBLAS_CHECK(cublasCreate(&handle));

    d_x = d_x_a;
    d_y = d_x_b;

    for (int iter = 0; iter < maxiter; ++iter) {
        int col_blocks = (k + block_size - 1) / block_size;
        int vector_blocks = ((int)vector_elements + block_size - 1) / block_size;
        double *tmp = NULL;

        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n,
                                 &one, d_matrix, n, d_x, n, &zero, d_y, n));

        column_diagonal_after_d_kernel<<<col_blocks, block_size>>>(
            d_y, d_x, d_diagonal, d_column_diagonal, n, k);
        CUDA_CHECK(cudaGetLastError());

        ipt_update_kernel<<<vector_blocks, block_size>>>(
            d_y, d_x, d_diagonal, d_g, d_column_diagonal, n, k);
        CUDA_CHECK(cudaGetLastError());

        iterations_done = iter + 1;

        if (use_tolerance) {
            double delta_norm = 0.0;
            double current_norm = 0.0;
            double fixed_point_residual = 0.0;

            fixed_point_delta_kernel<<<vector_blocks, block_size>>>(
                d_delta, d_y, d_x, (int)vector_elements);
            CUDA_CHECK(cudaGetLastError());

            CUBLAS_CHECK(cublasDnrm2(handle, (int)vector_elements, d_delta, 1,
                                     &delta_norm));
            CUBLAS_CHECK(cublasDnrm2(handle, (int)vector_elements, d_x, 1,
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
        }

        tmp = d_x;
        d_x = d_y;
        d_y = tmp;
    }

    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &one,
                             d_matrix, n, d_x, n, &zero, d_y, n));

    {
        int col_blocks = (k + block_size - 1) / block_size;

        gather_values_kernel<<<col_blocks, block_size>>>(d_y, d_values, n, k);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(result->vectors, d_x, vector_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result->values, d_values, (size_t)k * sizeof(double),
                          cudaMemcpyDeviceToHost));

    result->n = n;
    result->k = k;
    result->iterations = iterations_done;
    result->basis_cols = k;

cleanup:
    if (handle != NULL) {
        cublasDestroy(handle);
    }

    cudaFree(d_matrix);
    cudaFree(d_x_a);
    cudaFree(d_x_b);
    cudaFree(d_diagonal);
    cudaFree(d_g);
    cudaFree(d_column_diagonal);
    cudaFree(d_delta);
    cudaFree(d_values);

    if (status != IPT_CUDA_SUCCESS) {
        ipt_cuda_free_result(result);
    }

    return status;
}

extern "C" int ipt_cuda_with_initial(const double *matrix_col_major,
                                      const double *initial_vectors_col_major,
                                      int n, int k, int iterations,
                                      IPTCudaResult *result)
{
    return ipt_cuda_with_initial_impl(matrix_col_major,
                                      initial_vectors_col_major, n, k,
                                      iterations, 0.0, 0, result);
}

extern "C" int ipt_cuda_with_initial_tol(
    const double *matrix_col_major, const double *initial_vectors_col_major,
    int n, int k, double tol, int maxiter, IPTCudaResult *result)
{
    return ipt_cuda_with_initial_impl(matrix_col_major,
                                      initial_vectors_col_major, n, k,
                                      maxiter, tol, 1, result);
}

static int ipt_cuda_sparse_csc_device_impl(
    const int *d_col_ptr, const int *d_row_ind,
    const double *d_matrix_values, int n, int k, int nnz, int maxiter,
    double tol, int use_tolerance, cublasHandle_t handle,
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
    cusparseHandle_t sparse_handle = NULL;
    cusparseSpMatDescr_t sparse_matrix = NULL;
    cusparseDnMatDescr_t dense_x = NULL;
    cusparseDnMatDescr_t dense_y = NULL;
    void *d_spmm_buffer = NULL;
    size_t spmm_buffer_size = 0;
    double fixed_point_residual = NAN;
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

    if (d_col_ptr == NULL || d_row_ind == NULL || d_matrix_values == NULL ||
        handle == NULL || d_vectors_out == NULL || d_values_out == NULL ||
        n <= 0 || k <= 0 || k > n || nnz < 0 || maxiter < 0 ||
        (use_tolerance && (tol <= 0.0 || maxiter <= 0))) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    vector_elements = (size_t)n * (size_t)k;

    if (vector_elements > (size_t)INT_MAX) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    vector_bytes = vector_elements * sizeof(double);

    CUDA_CHECK(cudaMalloc((void **)&d_x_a, vector_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_x_b, vector_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_diagonal, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_g, vector_bytes));
    CUDA_CHECK(
        cudaMalloc((void **)&d_column_diagonal, (size_t)k * sizeof(double)));
    if (use_tolerance) {
        CUDA_CHECK(cudaMalloc((void **)&d_delta, vector_bytes));
    }
    CUDA_CHECK(cudaMalloc((void **)&d_values, (size_t)k * sizeof(double)));

    CUSPARSE_CHECK(cusparseCreate(&sparse_handle));
    CUSPARSE_CHECK(cusparseCreateCsr(
        &sparse_matrix, n, n, nnz, (void *)d_col_ptr, (void *)d_row_ind,
        (void *)d_matrix_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_x, n, k, n, d_x_a, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_y, n, k, n, d_x_b, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    {
        const double one = 1.0;
        const double zero = 0.0;

        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            sparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &one, sparse_matrix, dense_x,
            &zero, dense_y, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
            &spmm_buffer_size));
        if (spmm_buffer_size > 0) {
            CUDA_CHECK(cudaMalloc(&d_spmm_buffer, spmm_buffer_size));
        }
    }

    {
        int vector_blocks = ((int)vector_elements + block_size - 1) / block_size;
        int diag_blocks = (n + block_size - 1) / block_size;

        set_identity_kernel<<<vector_blocks, block_size>>>(d_x_a, n, k);
        CUDA_CHECK(cudaGetLastError());

        extract_csc_diagonal_kernel<<<diag_blocks, block_size>>>(
            d_col_ptr, d_row_ind, d_matrix_values, d_diagonal, n);
        CUDA_CHECK(cudaGetLastError());

        build_g_kernel<<<vector_blocks, block_size>>>(d_diagonal, d_g, n, k,
                                                      0.0, NULL);
        CUDA_CHECK(cudaGetLastError());
    }

    d_x = d_x_a;
    d_y = d_x_b;

    for (int iter = 0; iter < maxiter; ++iter) {
        int col_blocks = (k + block_size - 1) / block_size;
        int vector_blocks = ((int)vector_elements + block_size - 1) / block_size;
        double *tmp = NULL;

        CUSPARSE_CHECK(ipt_cusparse_spmm(sparse_handle, sparse_matrix,
                                         dense_x, dense_y, d_x, d_y,
                                         d_spmm_buffer));

        column_diagonal_after_d_kernel<<<col_blocks, block_size>>>(
            d_y, d_x, d_diagonal, d_column_diagonal, n, k);
        CUDA_CHECK(cudaGetLastError());

        ipt_update_kernel<<<vector_blocks, block_size>>>(
            d_y, d_x, d_diagonal, d_g, d_column_diagonal, n, k);
        CUDA_CHECK(cudaGetLastError());

        iterations_done = iter + 1;

        if (use_tolerance) {
            double delta_norm = 0.0;
            double current_norm = 0.0;

            fixed_point_delta_kernel<<<vector_blocks, block_size>>>(
                d_delta, d_y, d_x, (int)vector_elements);
            CUDA_CHECK(cudaGetLastError());

            CUBLAS_CHECK(cublasDnrm2(handle, (int)vector_elements, d_delta, 1,
                                     &delta_norm));
            CUBLAS_CHECK(cublasDnrm2(handle, (int)vector_elements, d_x, 1,
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
        }

        tmp = d_x;
        d_x = d_y;
        d_y = tmp;
    }

    CUSPARSE_CHECK(ipt_cusparse_spmm(sparse_handle, sparse_matrix, dense_x,
                                     dense_y, d_x, d_y, d_spmm_buffer));

    {
        int col_blocks = (k + block_size - 1) / block_size;

        gather_values_kernel<<<col_blocks, block_size>>>(d_y, d_values, n, k);
        CUDA_CHECK(cudaGetLastError());
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
    cudaFree(d_spmm_buffer);
    if (dense_x != NULL) {
        cusparseDestroyDnMat(dense_x);
    }
    if (dense_y != NULL) {
        cusparseDestroyDnMat(dense_y);
    }
    if (sparse_matrix != NULL) {
        cusparseDestroySpMat(sparse_matrix);
    }
    if (sparse_handle != NULL) {
        cusparseDestroy(sparse_handle);
    }

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

extern "C" int ipt_cuda_sparse_csc_device(
    const int *d_col_ptr, const int *d_row_ind,
    const double *d_matrix_values, int n, int k, int nnz, int iterations,
    cublasHandle_t handle, double **d_vectors_out, double **d_values_out)
{
    return ipt_cuda_sparse_csc_device_impl(
        d_col_ptr, d_row_ind, d_matrix_values, n, k, nnz, iterations, 0.0, 0,
        handle, d_vectors_out, d_values_out, NULL, NULL);
}

extern "C" int ipt_cuda_sparse_csc_device_tol(
    const int *d_col_ptr, const int *d_row_ind,
    const double *d_matrix_values, int n, int k, int nnz, double tol,
    int maxiter, cublasHandle_t handle, double **d_vectors_out,
    double **d_values_out, int *iterations_done_out,
    double *fixed_point_residual_out)
{
    return ipt_cuda_sparse_csc_device_impl(
        d_col_ptr, d_row_ind, d_matrix_values, n, k, nnz, maxiter, tol, 1,
        handle, d_vectors_out, d_values_out, iterations_done_out,
        fixed_point_residual_out);
}

extern "C" int ipt_cuda_sparse_csc(
    const int *col_ptr, const int *row_ind, const double *matrix_values, int n,
    int k, int nnz, int iterations, IPTCudaResult *result)
{
    return ipt_cuda_sparse_csc_with_initial(col_ptr, row_ind, matrix_values,
                                            NULL, n, k, nnz, iterations,
                                            result);
}

extern "C" int ipt_cuda_sparse_csc_tol(
    const int *col_ptr, const int *row_ind, const double *matrix_values, int n,
    int k, int nnz, double tol, int maxiter, IPTCudaResult *result)
{
    return ipt_cuda_sparse_csc_with_initial_tol(
        col_ptr, row_ind, matrix_values, NULL, n, k, nnz, tol, maxiter,
        result);
}

__global__ static void ipt_block_cluster_identity_kernel(
    double *basis, int n, int m, int first)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * m;

    if (idx < total) {
        int row = idx % n;
        int col = idx / n;
        basis[idx] = row == first + col ? 1.0 : 0.0;
    }
}

__global__ static void ipt_block_cluster_gather_h_kernel(
    const double *y, double *h, int n, int m, int first)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < m * m) {
        int row = idx % m;
        int col = idx / m;
        h[idx] = y[first + row + col * n];
    }
}

__global__ static void ipt_block_cluster_update_kernel(
    const double *basis, const double *y, const double *diagonal,
    const double *h, double *next_basis, int n, int m, int first, int last,
    double damping, int *singular)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ unsigned char shared_raw[];
    double *aug = reinterpret_cast<double *>(shared_raw);
    int *meta = reinterpret_cast<int *>(aug + (size_t)m * (size_t)(m + 1));

    if (row >= n) {
        return;
    }
    if (row >= first && row <= last) {
        if (tid < m) {
            next_basis[row + tid * n] = row == first + tid ? 1.0 : 0.0;
        }
        return;
    }

    for (int equation = tid; equation < m; equation += blockDim.x) {
        for (int unknown = 0; unknown < m; ++unknown) {
            aug[equation * (m + 1) + unknown] =
                (equation == unknown ? diagonal[row] : 0.0) -
                h[unknown + equation * m];
        }
        aug[equation * (m + 1) + m] =
            diagonal[row] * basis[row + equation * n] -
            y[row + equation * n];
    }
    if (tid == 0) {
        meta[0] = 0;
        meta[1] = 0;
    }
    __syncthreads();

    for (int pivot_col = 0; pivot_col < m; ++pivot_col) {
        if (tid == 0) {
            int pivot_row = pivot_col;
            double pivot_abs =
                fabs(aug[pivot_col * (m + 1) + pivot_col]);

            for (int candidate = pivot_col + 1; candidate < m; ++candidate) {
                double candidate_abs =
                    fabs(aug[candidate * (m + 1) + pivot_col]);
                if (candidate_abs > pivot_abs) {
                    pivot_abs = candidate_abs;
                    pivot_row = candidate;
                }
            }
            meta[0] = pivot_row;
            if (!isfinite(pivot_abs) || pivot_abs <= 64.0 * DBL_EPSILON) {
                meta[1] = 1;
                atomicExch(singular, 1);
            }
        }
        __syncthreads();
        if (meta[1]) {
            return;
        }

        if (meta[0] != pivot_col) {
            for (int col = tid; col <= m; col += blockDim.x) {
                double temp = aug[pivot_col * (m + 1) + col];
                aug[pivot_col * (m + 1) + col] =
                    aug[meta[0] * (m + 1) + col];
                aug[meta[0] * (m + 1) + col] = temp;
            }
        }
        __syncthreads();

        {
            double pivot = aug[pivot_col * (m + 1) + pivot_col];
            for (int col = tid; col <= m; col += blockDim.x) {
                aug[pivot_col * (m + 1) + col] /= pivot;
            }
        }
        __syncthreads();

        for (int equation = tid; equation < m; equation += blockDim.x) {
            if (equation != pivot_col) {
                double factor = aug[equation * (m + 1) + pivot_col];
                for (int col = pivot_col; col <= m; ++col) {
                    aug[equation * (m + 1) + col] -=
                        factor * aug[pivot_col * (m + 1) + col];
                }
            }
        }
        __syncthreads();
    }

    if (tid < m) {
        double value = aug[tid * (m + 1) + m];
        if (damping < 1.0) {
            value = (1.0 - damping) * basis[row + tid * n] +
                    damping * value;
        }
        if (!isfinite(value)) {
            atomicExch(singular, 1);
            value = 0.0;
        }
        next_basis[row + tid * n] = value;
    }
}

__global__ static void ipt_block_cluster_unpermute_kernel(
    const double *sorted_vectors, double *vectors, const int *perm, int n,
    int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n * k) {
        int sorted_row = idx % n;
        int col = idx / n;
        vectors[perm[sorted_row] + col * n] = sorted_vectors[idx];
    }
}

static double ipt_block_elapsed_seconds(
    const std::chrono::steady_clock::time_point &start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
        .count();
}

static void ipt_block_orthogonality_stats(
    const std::vector<double> &vectors, int n, int cols,
    double *frobenius_error, double *max_abs_error, int *has_nan_or_inf)
{
    double frobenius_sq = 0.0;
    double max_error = 0.0;
    int invalid = 0;

    for (size_t i = 0; i < vectors.size(); ++i) {
        if (!isfinite(vectors[i])) {
            invalid = 1;
            break;
        }
    }
    if (!invalid) {
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < cols; ++row) {
                double dot = 0.0;

                for (int i = 0; i < n; ++i) {
                    dot += vectors[(size_t)i + (size_t)row * (size_t)n] *
                           vectors[(size_t)i + (size_t)col * (size_t)n];
                }
                {
                    double error = dot - (row == col ? 1.0 : 0.0);

                    frobenius_sq += error * error;
                    max_error = std::max(max_error, fabs(error));
                }
            }
        }
    }
    *has_nan_or_inf = invalid;
    *frobenius_error = invalid ? NAN : sqrt(frobenius_sq);
    *max_abs_error = invalid ? NAN : max_error;
}

static double ipt_block_host_matrix_inf_norm(
    const int *col_ptr, const int *row_ind, const double *values, int n)
{
    std::vector<double> row_sum((size_t)n, 0.0);
    double norm = 0.0;

    for (int col = 0; col < n; ++col) {
        for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
            row_sum[(size_t)row_ind[p]] += fabs(values[p]);
        }
    }
    for (int row = 0; row < n; ++row) {
        norm = std::max(norm, row_sum[(size_t)row]);
    }
    return norm;
}

static double ipt_block_host_pair_residual(
    const int *col_ptr, const int *row_ind, const double *values, int n,
    const double *vector, double eigenvalue, double matrix_norm,
    std::vector<double> *residual_vector)
{
    std::vector<double> av((size_t)n, 0.0);
    double residual_norm_sq = 0.0;
    double vector_norm_sq = 0.0;

    for (int col = 0; col < n; ++col) {
        double x = vector[col];

        for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
            av[(size_t)row_ind[p]] += values[p] * x;
        }
    }
    if (residual_vector != NULL) {
        residual_vector->assign((size_t)n, 0.0);
    }
    for (int row = 0; row < n; ++row) {
        double residual = av[(size_t)row] - eigenvalue * vector[row];

        if (residual_vector != NULL) {
            (*residual_vector)[(size_t)row] = residual;
        }
        residual_norm_sq += residual * residual;
        vector_norm_sq += vector[row] * vector[row];
    }
    return sqrt(residual_norm_sq) /
           (std::max(1.0, matrix_norm) *
            std::max(1.0, sqrt(vector_norm_sq)));
}

static double ipt_block_host_max_residual(
    const int *col_ptr, const int *row_ind, const double *values, int n,
    int k, const std::vector<double> &ritz_values,
    const std::vector<double> &ritz_vectors, double matrix_norm,
    int *max_index)
{
    double maximum = NAN;

    *max_index = -1;
    for (int col = 0; col < k; ++col) {
        double residual = ipt_block_host_pair_residual(
            col_ptr, row_ind, values, n,
            ritz_vectors.data() + (size_t)col * (size_t)n,
            ritz_values[(size_t)col], matrix_norm, NULL);

        if (!isfinite(maximum) || residual > maximum) {
            maximum = residual;
            *max_index = col;
        }
    }
    return maximum;
}

static int ipt_block_build_davidson_correction(
    const int *col_ptr, const int *row_ind, const double *values, int n,
    int k, const std::vector<double> &diagonal,
    const std::vector<double> &ritz_values,
    const std::vector<double> &ritz_vectors, int target_index,
    double denom_clip, std::vector<double> *correction)
{
    std::vector<double> residual;
    const double *target =
        ritz_vectors.data() + (size_t)target_index * (size_t)n;
    double norm_sq = 0.0;

    ipt_block_host_pair_residual(
        col_ptr, row_ind, values, n, target,
        ritz_values[(size_t)target_index], 1.0, &residual);
    correction->assign((size_t)n, 0.0);
    for (int row = 0; row < n; ++row) {
        double denominator =
            diagonal[(size_t)row] - ritz_values[(size_t)target_index];

        if (fabs(denominator) < denom_clip) {
            denominator = denominator < 0.0 ? -denom_clip : denom_clip;
        }
        (*correction)[(size_t)row] =
            residual[(size_t)row] / denominator;
    }
    for (int pass = 0; pass < 2; ++pass) {
        for (int col = 0; col < k; ++col) {
            const double *vector =
                ritz_vectors.data() + (size_t)col * (size_t)n;
            double dot = 0.0;

            for (int row = 0; row < n; ++row) {
                dot += vector[row] * (*correction)[(size_t)row];
            }
            for (int row = 0; row < n; ++row) {
                (*correction)[(size_t)row] -= dot * vector[row];
            }
        }
    }
    for (int row = 0; row < n; ++row) {
        if (!isfinite((*correction)[(size_t)row])) {
            return 0;
        }
        norm_sq += (*correction)[(size_t)row] *
                   (*correction)[(size_t)row];
    }
    if (!isfinite(norm_sq) || norm_sq <= DBL_EPSILON) {
        return 0;
    }
    {
        double inverse_norm = 1.0 / sqrt(norm_sq);

        for (int row = 0; row < n; ++row) {
            (*correction)[(size_t)row] *= inverse_norm;
        }
    }
    return 1;
}

static int ipt_block_cluster_solve_basis_gpu(
    cusparseHandle_t sparse_handle, cusparseSpMatDescr_t sparse_matrix,
    cublasHandle_t cublas_handle, const double *d_diagonal, int n, int m,
    int first, int last, int maxiter, double tol, double damping,
    double **d_basis_out, int *iterations_done, double *iteration_time_sec,
    double *fixed_point_residual_out)
{
    int status = IPT_CUDA_SUCCESS;
    size_t elements = (size_t)n * (size_t)m;
    size_t bytes = elements * sizeof(double);
    double *d_a = NULL;
    double *d_b = NULL;
    double *d_y = NULL;
    double *d_h = NULL;
    double *d_delta = NULL;
    double *d_current = NULL;
    double *d_next = NULL;
    int *d_singular = NULL;
    void *d_spmm_buffer = NULL;
    size_t spmm_buffer_size = 0;
    cusparseDnMatDescr_t dense_x = NULL;
    cusparseDnMatDescr_t dense_y = NULL;
    int done = 0;
    const int threads = 256;
    double fixed_point_residual = NAN;

    *d_basis_out = NULL;
    *iterations_done = 0;
    *iteration_time_sec = 0.0;
    if (fixed_point_residual_out != NULL) {
        *fixed_point_residual_out = NAN;
    }

    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_y, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_h, (size_t)m * (size_t)m *
                                         sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_delta, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_singular, sizeof(int)));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_x, n, m, n, d_a, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_y, n, m, n, d_y, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    {
        const double one = 1.0;
        const double zero = 0.0;
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            sparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &one, sparse_matrix, dense_x,
            &zero, dense_y, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
            &spmm_buffer_size));
    }
    if (spmm_buffer_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_spmm_buffer, spmm_buffer_size));
    }

    ipt_block_cluster_identity_kernel<<<
        (int)((elements + threads - 1) / threads), threads>>>(d_a, n, m,
                                                               first);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    d_current = d_a;
    d_next = d_b;

    {
        std::chrono::steady_clock::time_point start =
            std::chrono::steady_clock::now();

        for (int iter = 0; iter < maxiter; ++iter) {
            double delta_norm = 0.0;
            double basis_norm = 0.0;
            int singular = 0;
            size_t shared_bytes =
                (size_t)m * (size_t)(m + 1) * sizeof(double) +
                2U * sizeof(int);

            CUSPARSE_CHECK(ipt_cusparse_spmm(
                sparse_handle, sparse_matrix, dense_x, dense_y, d_current,
                d_y, d_spmm_buffer));
            ipt_block_cluster_gather_h_kernel<<<
                (m * m + threads - 1) / threads, threads>>>(
                d_y, d_h, n, m, first);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemset(d_singular, 0, sizeof(int)));
            ipt_block_cluster_update_kernel<<<n, m, shared_bytes>>>(
                d_current, d_y, d_diagonal, d_h, d_next, n, m, first, last,
                damping, d_singular);
            CUDA_CHECK(cudaGetLastError());

            fixed_point_delta_kernel<<<
                (int)((elements + threads - 1) / threads), threads>>>(
                d_delta, d_next, d_current, (int)elements);
            CUDA_CHECK(cudaGetLastError());
            CUBLAS_CHECK(cublasDnrm2(cublas_handle, (int)elements, d_delta, 1,
                                     &delta_norm));
            CUBLAS_CHECK(cublasDnrm2(cublas_handle, (int)elements, d_current,
                                     1, &basis_norm));
            CUDA_CHECK(cudaMemcpy(&singular, d_singular, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            if (singular || !isfinite(delta_norm) ||
                !isfinite(basis_norm) || basis_norm <= 0.0) {
                status = IPT_CUDA_INVALID_ARGUMENT;
                goto cleanup;
            }

            std::swap(d_current, d_next);
            done = iter + 1;
            fixed_point_residual = delta_norm / std::max(1.0, basis_norm);
            if (fixed_point_residual <= tol) {
                break;
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        *iteration_time_sec = ipt_block_elapsed_seconds(start);
    }

    *iterations_done = done;
    if (fixed_point_residual_out != NULL) {
        *fixed_point_residual_out = fixed_point_residual;
    }
    *d_basis_out = d_current;
    if (d_current == d_a) {
        d_a = NULL;
    } else {
        d_b = NULL;
    }

cleanup:
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_y);
    cudaFree(d_h);
    cudaFree(d_delta);
    cudaFree(d_singular);
    cudaFree(d_spmm_buffer);
    if (dense_x != NULL) {
        cusparseDestroyDnMat(dense_x);
    }
    if (dense_y != NULL) {
        cusparseDestroyDnMat(dense_y);
    }
    return status;
}

static int ipt_block_cluster_rayleigh_ritz_gpu(
    cusparseHandle_t sparse_handle, cusparseSpMatDescr_t sparse_matrix,
    cublasHandle_t cublas_handle, const double *d_basis, int n, int basis_cols,
    int k, const int *d_perm, double **d_values_out, double **d_vectors_out,
    double *rayleigh_ritz_time_sec, int *used_qr_out,
    double *basis_orthogonality_frobenius_error,
    double *basis_orthogonality_max_abs_error, int *basis_has_nan_or_inf,
    double *ritz_vectors_orthogonality_frobenius_error,
    double *ritz_vectors_orthogonality_max_abs_error,
    int *ritz_vectors_has_nan_or_inf)
{
    int status = IPT_CUDA_SUCCESS;
    double *d_y = NULL;
    double *d_q = NULL;
    double *d_tau = NULL;
    double *d_gram = NULL;
    double *d_projected = NULL;
    double *d_all_values = NULL;
    double *d_sorted_vectors = NULL;
    double *d_vectors = NULL;
    double *d_work = NULL;
    int *d_info = NULL;
    void *d_spmm_buffer = NULL;
    size_t spmm_buffer_size = 0;
    int lwork = 0;
    int geqrf_lwork = 0;
    int orgqr_lwork = 0;
    int eigen_lwork = 0;
    int info = 0;
    int use_qr = ipt_cuda_env_flag("IPT_BLOCK_CLUSTER_QR");
    const double *d_basis_for_rr = d_basis;
    cusparseDnMatDescr_t dense_x = NULL;
    cusparseDnMatDescr_t dense_y = NULL;
    cusolverDnHandle_t solver = NULL;
    const int threads = 256;
    const double one = 1.0;
    const double zero = 0.0;

    *d_values_out = NULL;
    *d_vectors_out = NULL;
    *rayleigh_ritz_time_sec = 0.0;
    *used_qr_out = use_qr;
    *basis_orthogonality_frobenius_error = NAN;
    *basis_orthogonality_max_abs_error = NAN;
    *basis_has_nan_or_inf = 0;
    *ritz_vectors_orthogonality_frobenius_error = NAN;
    *ritz_vectors_orthogonality_max_abs_error = NAN;
    *ritz_vectors_has_nan_or_inf = 0;

    CUDA_CHECK(cudaMalloc((void **)&d_y,
                          (size_t)n * (size_t)basis_cols * sizeof(double)));
    if (use_qr) {
        CUDA_CHECK(cudaMalloc((void **)&d_q,
                              (size_t)n * (size_t)basis_cols *
                                  sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_tau,
                              (size_t)basis_cols * sizeof(double)));
        d_basis_for_rr = d_q;
    } else {
        CUDA_CHECK(cudaMalloc((void **)&d_gram,
                              (size_t)basis_cols * (size_t)basis_cols *
                                  sizeof(double)));
    }
    CUDA_CHECK(cudaMalloc((void **)&d_projected,
                          (size_t)basis_cols * (size_t)basis_cols *
                              sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_all_values,
                          (size_t)basis_cols * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_sorted_vectors,
                          (size_t)n * (size_t)k * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_vectors,
                          (size_t)n * (size_t)k * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_info, sizeof(int)));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_x, n, basis_cols, n,
                                       (void *)d_basis_for_rr, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_y, n, basis_cols, n, d_y,
                                       CUDA_R_64F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        sparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &one, sparse_matrix, dense_x, &zero,
        dense_y, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &spmm_buffer_size));
    if (spmm_buffer_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_spmm_buffer, spmm_buffer_size));
    }
    CUSOLVER_CHECK(cusolverDnCreate(&solver));
    if (use_qr) {
        CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(
            solver, n, basis_cols, d_q, n, &geqrf_lwork));
        CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(
            solver, n, basis_cols, basis_cols, d_q, n, d_tau,
            &orgqr_lwork));
        CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(
            solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
            basis_cols, d_projected, basis_cols, d_all_values,
            &eigen_lwork));
        lwork =
            std::max(geqrf_lwork, std::max(orgqr_lwork, eigen_lwork));
    } else {
        CUSOLVER_CHECK(cusolverDnDsygvd_bufferSize(
            solver, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER, basis_cols, d_projected, basis_cols,
            d_gram, basis_cols, d_all_values, &lwork));
    }
    CUDA_CHECK(cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(double)));
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        std::chrono::steady_clock::time_point start =
            std::chrono::steady_clock::now();

        if (use_qr) {
            CUDA_CHECK(cudaMemcpy(d_q, d_basis,
                                  (size_t)n * (size_t)basis_cols *
                                      sizeof(double),
                                  cudaMemcpyDeviceToDevice));
            CUSOLVER_CHECK(cusolverDnDgeqrf(
                solver, n, basis_cols, d_q, n, d_tau, d_work, lwork,
                d_info));
            CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            if (info != 0) {
                fprintf(stderr, "cuSOLVER Dgeqrf failed: info=%d\n", info);
                status = IPT_CUDA_CUSOLVER_ERROR;
                goto cleanup;
            }
            CUSOLVER_CHECK(cusolverDnDorgqr(
                solver, n, basis_cols, basis_cols, d_q, n, d_tau, d_work,
                lwork, d_info));
            CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            if (info != 0) {
                fprintf(stderr, "cuSOLVER Dorgqr failed: info=%d\n", info);
                status = IPT_CUDA_CUSOLVER_ERROR;
                goto cleanup;
            }
        }
        {
            std::vector<double> host_basis(
                (size_t)n * (size_t)basis_cols, 0.0);

            CUDA_CHECK(cudaMemcpy(host_basis.data(), d_basis_for_rr,
                                  host_basis.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            ipt_block_orthogonality_stats(
                host_basis, n, basis_cols,
                basis_orthogonality_frobenius_error,
                basis_orthogonality_max_abs_error, basis_has_nan_or_inf);
        }
        CUSPARSE_CHECK(ipt_cusparse_spmm(
            sparse_handle, sparse_matrix, dense_x, dense_y,
            const_cast<double *>(d_basis_for_rr), d_y, d_spmm_buffer));
        if (!use_qr) {
            CUBLAS_CHECK(cublasDgemm(
                cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, basis_cols,
                basis_cols, n, &one, d_basis, n, d_basis, n, &zero, d_gram,
                basis_cols));
        }
        CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 basis_cols, basis_cols, n, &one,
                                 d_basis_for_rr, n, d_y, n, &zero,
                                 d_projected, basis_cols));
        if (use_qr) {
            CUSOLVER_CHECK(cusolverDnDsyevd(
                solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                basis_cols, d_projected, basis_cols, d_all_values, d_work,
                lwork, d_info));
        } else {
            CUSOLVER_CHECK(cusolverDnDsygvd(
                solver, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,
                CUBLAS_FILL_MODE_LOWER, basis_cols, d_projected, basis_cols,
                d_gram, basis_cols, d_all_values, d_work, lwork, d_info));
        }
        CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (info != 0) {
            fprintf(stderr, "cuSOLVER Rayleigh-Ritz failed: info=%d\n", info);
            status = IPT_CUDA_CUSOLVER_ERROR;
            goto cleanup;
        }
        CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n,
                                 k, basis_cols, &one, d_basis_for_rr, n,
                                 d_projected, basis_cols, &zero,
                                 d_sorted_vectors, n));
        for (int col = 0; col < k; ++col) {
            double norm = 0.0;
            CUBLAS_CHECK(cublasDnrm2(cublas_handle, n,
                                     d_sorted_vectors + (size_t)col * n, 1,
                                     &norm));
            if (!isfinite(norm) || norm <= 0.0) {
                status = IPT_CUDA_INVALID_ARGUMENT;
                goto cleanup;
            }
            {
                double inverse_norm = 1.0 / norm;
                CUBLAS_CHECK(cublasDscal(
                    cublas_handle, n, &inverse_norm,
                    d_sorted_vectors + (size_t)col * n, 1));
            }
        }
        {
            std::vector<double> host_ritz((size_t)n * (size_t)k, 0.0);

            CUDA_CHECK(cudaMemcpy(host_ritz.data(), d_sorted_vectors,
                                  host_ritz.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            ipt_block_orthogonality_stats(
                host_ritz, n, k,
                ritz_vectors_orthogonality_frobenius_error,
                ritz_vectors_orthogonality_max_abs_error,
                ritz_vectors_has_nan_or_inf);
        }
        ipt_block_cluster_unpermute_kernel<<<
            (n * k + threads - 1) / threads, threads>>>(
            d_sorted_vectors, d_vectors, d_perm, n, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        *rayleigh_ritz_time_sec = ipt_block_elapsed_seconds(start);
    }

    *d_values_out = d_all_values;
    *d_vectors_out = d_vectors;
    d_all_values = NULL;
    d_vectors = NULL;

cleanup:
    cudaFree(d_y);
    cudaFree(d_q);
    cudaFree(d_tau);
    cudaFree(d_gram);
    cudaFree(d_projected);
    cudaFree(d_all_values);
    cudaFree(d_sorted_vectors);
    cudaFree(d_vectors);
    cudaFree(d_work);
    cudaFree(d_info);
    cudaFree(d_spmm_buffer);
    if (dense_x != NULL) {
        cusparseDestroyDnMat(dense_x);
    }
    if (dense_y != NULL) {
        cusparseDestroyDnMat(dense_y);
    }
    if (solver != NULL) {
        cusolverDnDestroy(solver);
    }
    return status;
}

static void ipt_block_cluster_csc_matmul(
    const std::vector<int> &col_ptr, const std::vector<int> &row_ind,
    const std::vector<double> &values, int n, const std::vector<double> &x,
    int cols, std::vector<double> *y)
{
    y->assign((size_t)n * (size_t)cols, 0.0);

    for (int col = 0; col < n; ++col) {
        for (int p = col_ptr[(size_t)col]; p < col_ptr[(size_t)col + 1U];
             ++p) {
            int row = row_ind[(size_t)p];
            double value = values[(size_t)p];

            for (int block_col = 0; block_col < cols; ++block_col) {
                (*y)[(size_t)row + (size_t)block_col * (size_t)n] +=
                    value *
                    x[(size_t)col + (size_t)block_col * (size_t)n];
            }
        }
    }
}

static int ipt_block_cluster_cholesky_lower(const std::vector<double> &a,
                                            int m,
                                            std::vector<double> *lower)
{
    lower->assign((size_t)m * (size_t)m, 0.0);

    for (int col = 0; col < m; ++col) {
        for (int row = col; row < m; ++row) {
            double sum = a[(size_t)row + (size_t)col * (size_t)m];

            for (int inner = 0; inner < col; ++inner) {
                sum -= (*lower)[(size_t)row +
                                (size_t)inner * (size_t)m] *
                       (*lower)[(size_t)col +
                                (size_t)inner * (size_t)m];
            }

            if (row == col) {
                if (sum <= 1.0e-14 || !isfinite(sum)) {
                    return 0;
                }
                (*lower)[(size_t)row + (size_t)col * (size_t)m] =
                    sqrt(sum);
            } else {
                (*lower)[(size_t)row + (size_t)col * (size_t)m] =
                    sum / (*lower)[(size_t)col +
                                   (size_t)col * (size_t)m];
            }
        }
    }
    return 1;
}

static int ipt_block_cluster_inverse_lower(const std::vector<double> &lower,
                                           int m,
                                           std::vector<double> *inverse)
{
    inverse->assign((size_t)m * (size_t)m, 0.0);

    for (int col = 0; col < m; ++col) {
        for (int row = 0; row < m; ++row) {
            double rhs = (row == col) ? 1.0 : 0.0;

            for (int inner = 0; inner < row; ++inner) {
                rhs -= lower[(size_t)row +
                             (size_t)inner * (size_t)m] *
                       (*inverse)[(size_t)inner +
                                  (size_t)col * (size_t)m];
            }
            if (fabs(lower[(size_t)row + (size_t)row * (size_t)m]) <=
                1.0e-14) {
                return 0;
            }
            (*inverse)[(size_t)row + (size_t)col * (size_t)m] =
                rhs / lower[(size_t)row + (size_t)row * (size_t)m];
        }
    }
    return 1;
}

static void ipt_block_cluster_small_matmul(const std::vector<double> &a,
                                           const std::vector<double> &b,
                                           int m,
                                           std::vector<double> *c)
{
    c->assign((size_t)m * (size_t)m, 0.0);

    for (int col = 0; col < m; ++col) {
        for (int inner = 0; inner < m; ++inner) {
            double b_value = b[(size_t)inner + (size_t)col * (size_t)m];

            if (b_value == 0.0) {
                continue;
            }
            for (int row = 0; row < m; ++row) {
                (*c)[(size_t)row + (size_t)col * (size_t)m] +=
                    a[(size_t)row + (size_t)inner * (size_t)m] *
                    b_value;
            }
        }
    }
}

static void ipt_block_cluster_symmetric_eigenpairs(
    const std::vector<double> &matrix, int m, std::vector<double> *values,
    std::vector<double> *vectors)
{
    ipt_jacobi_eigenvectors_symmetric(matrix, m, vectors);
    values->assign((size_t)m, 0.0);

    for (int col = 0; col < m; ++col) {
        double value = 0.0;

        for (int row = 0; row < m; ++row) {
            for (int inner = 0; inner < m; ++inner) {
                value += (*vectors)[(size_t)row +
                                    (size_t)col * (size_t)m] *
                         matrix[(size_t)row +
                                (size_t)inner * (size_t)m] *
                         (*vectors)[(size_t)inner +
                                    (size_t)col * (size_t)m];
            }
        }
        (*values)[(size_t)col] = value;
    }
    ipt_sort_eigenvectors_by_values(vectors, values, m);
}

static void ipt_block_cluster_add_target_clusters(
    std::vector<std::pair<int, int>> *ranges, int n, int k)
{
    std::vector<int> covered((size_t)n, 0);

    for (size_t i = 0; i < ranges->size(); ++i) {
        for (int row = (*ranges)[i].first; row <= (*ranges)[i].second; ++row) {
            if (row >= 0 && row < n) {
                covered[(size_t)row] = 1;
            }
        }
    }
    for (int target = 0; target < k; ++target) {
        if (!covered[(size_t)target]) {
            ranges->push_back(std::make_pair(target, target));
        }
    }
    std::sort(ranges->begin(), ranges->end());
}

static void ipt_block_cluster_adaptive_expand(
    const std::vector<int> &col_ptr, const std::vector<int> &row_ind,
    const std::vector<double> &values, const std::vector<double> &diagonal,
    int requested_k, int max_block_size, double coupling_tau,
    std::vector<std::pair<int, int>> *ranges,
    std::vector<int> *added_indices, int *limit_hit)
{
    const double eps_gap = 1.0e-14;
    int n = (int)diagonal.size();

    added_indices->clear();
    *limit_hit = 0;
    for (;;) {
        int target_id = -1;
        int first = -1;
        int last = -1;
        int best_outside = -1;
        double best_ratio = 0.0;

        for (size_t i = 0; i < ranges->size(); ++i) {
            if ((*ranges)[i].first <= requested_k - 1 &&
                requested_k - 1 <= (*ranges)[i].second) {
                target_id = (int)i;
                first = (*ranges)[i].first;
                last = (*ranges)[i].second;
                break;
            }
        }
        if (target_id < 0) {
            break;
        }
        for (int col = first; col <= last; ++col) {
            for (int p = col_ptr[(size_t)col];
                 p < col_ptr[(size_t)col + 1U]; ++p) {
                int row = row_ind[(size_t)p];
                double gap = 0.0;
                double ratio = 0.0;

                if (first <= row && row <= last) {
                    continue;
                }
                gap = fabs(diagonal[(size_t)col] -
                           diagonal[(size_t)row]);
                ratio = fabs(values[(size_t)p]) / std::max(gap, eps_gap);
                if (std::max(last, row) - std::min(first, row) + 1 >
                    max_block_size) {
                    if (ratio > coupling_tau) {
                        *limit_hit = 1;
                    }
                    continue;
                }
                if (ratio > best_ratio) {
                    best_ratio = ratio;
                    best_outside = row;
                }
            }
        }
        if (best_outside < 0 || best_ratio <= coupling_tau) {
            break;
        }
        {
            int proposed_first = std::min(first, best_outside);
            int proposed_last = std::max(last, best_outside);
            bool expanded = true;

            while (expanded) {
                double scale = 1.0;
                double degenerate_gap = 0.0;

                expanded = false;
                if (proposed_first > 0) {
                    scale = std::max(
                        1.0, std::max(fabs(diagonal[(size_t)proposed_first]),
                                      fabs(diagonal[(size_t)proposed_first -
                                                    1U])));
                    degenerate_gap =
                        std::max(eps_gap,
                                 1000.0 *
                                     std::numeric_limits<double>::epsilon() *
                                     scale);
                    if (fabs(diagonal[(size_t)proposed_first] -
                             diagonal[(size_t)proposed_first - 1U]) <=
                        degenerate_gap) {
                        --proposed_first;
                        expanded = true;
                    }
                }
                if (proposed_last + 1 < n) {
                    scale = std::max(
                        1.0, std::max(fabs(diagonal[(size_t)proposed_last]),
                                      fabs(diagonal[(size_t)proposed_last +
                                                    1U])));
                    degenerate_gap =
                        std::max(eps_gap,
                                 1000.0 *
                                     std::numeric_limits<double>::epsilon() *
                                     scale);
                    if (fabs(diagonal[(size_t)proposed_last + 1U] -
                             diagonal[(size_t)proposed_last]) <=
                        degenerate_gap) {
                        ++proposed_last;
                        expanded = true;
                    }
                }
            }
            if (proposed_last - proposed_first + 1 > max_block_size) {
                *limit_hit = 1;
                break;
            }
            for (int index = proposed_first; index <= proposed_last; ++index) {
                if (index < first || index > last) {
                    added_indices->push_back(index);
                }
            }
            (*ranges)[(size_t)target_id] =
                std::make_pair(proposed_first, proposed_last);
            std::sort(ranges->begin(), ranges->end());
            {
                std::vector<std::pair<int, int>> merged;

                for (size_t i = 0; i < ranges->size(); ++i) {
                    if (merged.empty() ||
                        (*ranges)[i].first > merged.back().second) {
                        merged.push_back((*ranges)[i]);
                    } else {
                        merged.back().second =
                            std::max(merged.back().second,
                                     (*ranges)[i].second);
                    }
                }
                ranges->swap(merged);
            }
        }
    }
    std::sort(added_indices->begin(), added_indices->end());
    added_indices->erase(
        std::unique(added_indices->begin(), added_indices->end()),
        added_indices->end());
}

static int ipt_block_cluster_solve_basis(
    const std::vector<int> &col_ptr, const std::vector<int> &row_ind,
    const std::vector<double> &values, const std::vector<double> &diagonal,
    int first, int last, int maxiter, double tol, double damping,
    std::vector<double> *basis, int *iterations_done)
{
    int n = (int)diagonal.size();
    int m = last - first + 1;
    std::vector<double> next_basis;
    std::vector<double> y;
    std::vector<double> h((size_t)m * (size_t)m, 0.0);
    std::vector<double> denominator((size_t)m * (size_t)m, 0.0);
    std::vector<double> inverse;
    std::vector<double> rhs((size_t)m, 0.0);
    double effective_tol = tol > 0.0 ? tol : 1.0e-12;
    double effective_damping = damping;

    if (effective_damping <= 0.0 || effective_damping > 1.0 ||
        !isfinite(effective_damping)) {
        effective_damping = 1.0;
    }

    basis->assign((size_t)n * (size_t)m, 0.0);
    for (int col = 0; col < m; ++col) {
        (*basis)[(size_t)(first + col) + (size_t)col * (size_t)n] = 1.0;
    }
    if (iterations_done != NULL) {
        *iterations_done = 0;
    }

    for (int iter = 0; iter < maxiter; ++iter) {
        double delta_norm_sq = 0.0;
        double basis_norm_sq = 0.0;

        ipt_block_cluster_csc_matmul(col_ptr, row_ind, values, n, *basis, m,
                                     &y);
        for (int col = 0; col < m; ++col) {
            for (int row = 0; row < m; ++row) {
                h[(size_t)row + (size_t)col * (size_t)m] =
                    y[(size_t)(first + row) +
                      (size_t)col * (size_t)n];
            }
        }

        next_basis.assign((size_t)n * (size_t)m, 0.0);
        for (int col = 0; col < m; ++col) {
            next_basis[(size_t)(first + col) +
                       (size_t)col * (size_t)n] = 1.0;
        }

        for (int row = 0; row < n; ++row) {
            if (row >= first && row <= last) {
                continue;
            }

            for (int col = 0; col < m; ++col) {
                rhs[(size_t)col] =
                    diagonal[(size_t)row] *
                        (*basis)[(size_t)row +
                                 (size_t)col * (size_t)n] -
                    y[(size_t)row + (size_t)col * (size_t)n];
            }

            for (int col = 0; col < m; ++col) {
                for (int local_row = 0; local_row < m; ++local_row) {
                    denominator[(size_t)local_row +
                                (size_t)col * (size_t)m] =
                        ((local_row == col) ? diagonal[(size_t)row] : 0.0) -
                        h[(size_t)local_row + (size_t)col * (size_t)m];
                }
            }
            if (!ipt_inverse_square(denominator, m, &inverse)) {
                return 0;
            }

            for (int col = 0; col < m; ++col) {
                double value = 0.0;

                for (int inner = 0; inner < m; ++inner) {
                    value += rhs[(size_t)inner] *
                             inverse[(size_t)inner +
                                     (size_t)col * (size_t)m];
                }
                if (effective_damping < 1.0) {
                    value = (1.0 - effective_damping) *
                                (*basis)[(size_t)row +
                                         (size_t)col * (size_t)n] +
                            effective_damping * value;
                }
                next_basis[(size_t)row + (size_t)col * (size_t)n] = value;
            }
        }

        for (size_t i = 0; i < next_basis.size(); ++i) {
            double delta = next_basis[i] - (*basis)[i];

            if (!isfinite(next_basis[i])) {
                return 0;
            }
            delta_norm_sq += delta * delta;
            basis_norm_sq += (*basis)[i] * (*basis)[i];
        }
        basis->swap(next_basis);
        if (iterations_done != NULL) {
            *iterations_done = iter + 1;
        }
        if (sqrt(delta_norm_sq) / std::max(1.0, sqrt(basis_norm_sq)) <=
            effective_tol) {
            break;
        }
    }
    return 1;
}

static int ipt_block_cluster_rayleigh_ritz(
    const std::vector<int> &col_ptr, const std::vector<int> &row_ind,
    const std::vector<double> &values, int n, const std::vector<double> &basis,
    int basis_cols, int k, std::vector<double> *ritz_values,
    std::vector<double> *ritz_vectors)
{
    std::vector<double> y;
    std::vector<double> gram((size_t)basis_cols * (size_t)basis_cols, 0.0);
    std::vector<double> projected((size_t)basis_cols * (size_t)basis_cols,
                                  0.0);
    std::vector<double> lower;
    std::vector<double> lower_inverse;
    std::vector<double> temp;
    std::vector<double> transformed;
    std::vector<double> all_values;
    std::vector<double> all_vectors;

    ipt_block_cluster_csc_matmul(col_ptr, row_ind, values, n, basis,
                                 basis_cols, &y);

    for (int col = 0; col < basis_cols; ++col) {
        for (int row = 0; row < basis_cols; ++row) {
            double gram_value = 0.0;
            double projected_value = 0.0;

            for (int i = 0; i < n; ++i) {
                gram_value += basis[(size_t)i +
                                    (size_t)row * (size_t)n] *
                              basis[(size_t)i +
                                    (size_t)col * (size_t)n];
                projected_value += basis[(size_t)i +
                                         (size_t)row * (size_t)n] *
                                   y[(size_t)i +
                                     (size_t)col * (size_t)n];
            }
            gram[(size_t)row + (size_t)col * (size_t)basis_cols] =
                gram_value;
            projected[(size_t)row + (size_t)col * (size_t)basis_cols] =
                projected_value;
        }
    }

    if (!ipt_block_cluster_cholesky_lower(gram, basis_cols, &lower) ||
        !ipt_block_cluster_inverse_lower(lower, basis_cols, &lower_inverse)) {
        return 0;
    }

    ipt_block_cluster_small_matmul(lower_inverse, projected, basis_cols,
                                   &temp);
    transformed.assign((size_t)basis_cols * (size_t)basis_cols, 0.0);
    for (int col = 0; col < basis_cols; ++col) {
        for (int row = 0; row < basis_cols; ++row) {
            double value = 0.0;

            for (int inner = 0; inner < basis_cols; ++inner) {
                value += temp[(size_t)row +
                              (size_t)inner * (size_t)basis_cols] *
                         lower_inverse[(size_t)col +
                                       (size_t)inner *
                                           (size_t)basis_cols];
            }
            transformed[(size_t)row + (size_t)col * (size_t)basis_cols] =
                value;
        }
    }
    for (int col = 0; col < basis_cols; ++col) {
        for (int row = 0; row < col; ++row) {
            double sym = 0.5 *
                         (transformed[(size_t)row +
                                      (size_t)col *
                                          (size_t)basis_cols] +
                          transformed[(size_t)col +
                                      (size_t)row *
                                          (size_t)basis_cols]);
            transformed[(size_t)row + (size_t)col * (size_t)basis_cols] =
                sym;
            transformed[(size_t)col + (size_t)row * (size_t)basis_cols] =
                sym;
        }
    }

    ipt_block_cluster_symmetric_eigenpairs(transformed, basis_cols,
                                           &all_values, &all_vectors);
    ritz_values->assign((size_t)k, 0.0);
    ritz_vectors->assign((size_t)n * (size_t)k, 0.0);

    for (int col = 0; col < k; ++col) {
        std::vector<double> coeff((size_t)basis_cols, 0.0);
        double norm_sq = 0.0;

        (*ritz_values)[(size_t)col] = all_values[(size_t)col];
        for (int row = 0; row < basis_cols; ++row) {
            for (int inner = 0; inner < basis_cols; ++inner) {
                coeff[(size_t)row] +=
                    lower_inverse[(size_t)inner +
                                  (size_t)row *
                                      (size_t)basis_cols] *
                    all_vectors[(size_t)inner +
                                (size_t)col *
                                    (size_t)basis_cols];
            }
        }

        for (int row = 0; row < n; ++row) {
            double value = 0.0;

            for (int inner = 0; inner < basis_cols; ++inner) {
                value += basis[(size_t)row +
                               (size_t)inner * (size_t)n] *
                         coeff[(size_t)inner];
            }
            (*ritz_vectors)[(size_t)row + (size_t)col * (size_t)n] =
                value;
            norm_sq += value * value;
        }
        if (norm_sq > 0.0) {
            double inv_norm = 1.0 / sqrt(norm_sq);

            for (int row = 0; row < n; ++row) {
                (*ritz_vectors)[(size_t)row +
                                (size_t)col * (size_t)n] *= inv_norm;
            }
        }
    }
    return 1;
}

static int ipt_cuda_sparse_csc_block_cluster_impl(
    const int *col_ptr, const int *row_ind, const double *matrix_values, int n,
    int k, int nnz, int maxiter, double tol, int use_tolerance,
    IPTCudaResult *result)
{
    int status = IPT_CUDA_SUCCESS;
    std::chrono::steady_clock::time_point preparation_start =
        std::chrono::steady_clock::now();
    std::vector<double> original_diagonal;
    std::vector<int> perm;
    std::vector<int> inverse_perm;
    std::vector<int> sorted_col_ptr;
    std::vector<int> sorted_row_ind;
    std::vector<double> sorted_values;
    std::vector<double> diagonal;
    std::vector<std::pair<int, int>> ranges;
    IPTDegeneracyOptions options;
    int too_large = 0;
    int basis_cols = 0;
    int oversample = 0;
    int basis_target_k = 0;
    int adaptive_enabled = 0;
    int adaptive_limit_hit = 0;
    double adaptive_coupling_tau = 0.1;
    std::vector<int> adaptive_added_indices;
    int block_maxiter = 0;
    int iterations_done = 0;
    double block_tol = 0.0;
    double damping = 1.0;
    int *d_col_ptr = NULL;
    int *d_row_ind = NULL;
    int *d_perm = NULL;
    double *d_values = NULL;
    double *d_diagonal = NULL;
    double *d_combined_basis = NULL;
    double *d_ritz_values = NULL;
    double *d_ritz_vectors = NULL;
    cublasHandle_t cublas_handle = NULL;
    cusparseHandle_t sparse_handle = NULL;
    cusparseSpMatDescr_t sparse_matrix = NULL;

    if (result != NULL) {
        memset(result, 0, sizeof(*result));
        result->fixed_point_residual = NAN;
        result->basis_orthogonality_frobenius_error = NAN;
        result->basis_orthogonality_max_abs_error = NAN;
        result->ritz_vectors_orthogonality_frobenius_error = NAN;
        result->ritz_vectors_orthogonality_max_abs_error = NAN;
        result->davidson_target_index = -1;
        result->davidson_residual_before = NAN;
        result->davidson_residual_after = NAN;
        result->davidson_denom_clip = NAN;
        result->adaptive_coupling_tau = NAN;
        result->adaptive_target_block_start = -1;
        result->adaptive_target_block_end = -1;
    }

    if (col_ptr == NULL || row_ind == NULL || matrix_values == NULL ||
        result == NULL || n <= 0 || k <= 0 || k > n || nnz < 0 ||
        maxiter < 0 || (use_tolerance && (tol <= 0.0 || maxiter <= 0))) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }
    if (!ipt_validate_csc(col_ptr, row_ind, n, nnz)) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    ipt_extract_csc_diagonal_host(col_ptr, row_ind, matrix_values, n,
                                  &original_diagonal);
    ipt_build_stable_sort_permutation(original_diagonal, &perm, &inverse_perm);
    if (!ipt_permute_csc(col_ptr, row_ind, matrix_values, n, inverse_perm,
                         &sorted_col_ptr, &sorted_row_ind, &sorted_values)) {
        return IPT_CUDA_ALLOCATION_FAILED;
    }
    ipt_extract_csc_diagonal_vectors(sorted_col_ptr, sorted_row_ind,
                                     sorted_values, n, &diagonal);
    if (!ipt_csc_is_hermitian_real(sorted_col_ptr, sorted_row_ind,
                                   sorted_values, n)) {
        if (ipt_preparation_debug_enabled()) {
            fprintf(stderr,
                    "IPT block-cluster path currently requires a real "
                    "Hermitian CSC matrix\n");
        }
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    options = ipt_degeneracy_options(
        ipt_cuda_env_double("IPT_DEGENERACY_THRESHOLD", 0.0));
    adaptive_enabled =
        ipt_cuda_env_flag("IPT_BLOCK_CLUSTER_ADAPTIVE");
    adaptive_coupling_tau = ipt_cuda_env_double(
        "IPT_BLOCK_CLUSTER_COUPLING_TAU", 0.1);
    oversample = ipt_cuda_env_int("IPT_BLOCK_CLUSTER_OVERSAMPLE", 0);
    if (adaptive_enabled) {
        oversample = 0;
    }
    basis_target_k = k;
    if (oversample > 0) {
        basis_target_k = oversample > n - k ? n : k + oversample;
    }
    result->oversample = oversample;
    ranges = ipt_target_degenerate_subspaces(
        sorted_col_ptr, sorted_row_ind, sorted_values, diagonal,
        basis_target_k, options, &too_large);
    if (too_large) {
        int offending_size = 0;

        for (size_t i = 0; i < ranges.size(); ++i) {
            offending_size =
                std::max(offending_size,
                         ranges[i].second - ranges[i].first + 1);
        }
        fprintf(stderr,
                "IPT block-cluster invalid argument at block detection: "
                "block_size=%d max_block_size=%d\n",
                offending_size, options.max_block_size);
        return IPT_CUDA_INVALID_ARGUMENT;
    }
    ipt_block_cluster_add_target_clusters(&ranges, n, basis_target_k);
    if (adaptive_enabled) {
        ipt_block_cluster_adaptive_expand(
            sorted_col_ptr, sorted_row_ind, sorted_values, diagonal, k,
            options.max_block_size, adaptive_coupling_tau, &ranges,
            &adaptive_added_indices, &adaptive_limit_hit);
    }
    result->adaptive_block_enabled = adaptive_enabled;
    result->adaptive_coupling_tau = adaptive_coupling_tau;
    result->adaptive_limit_hit = adaptive_limit_hit;
    {
        std::string added;

        for (size_t i = 0; i < adaptive_added_indices.size(); ++i) {
            if (!added.empty()) {
                added += ";";
            }
            added += std::to_string(adaptive_added_indices[i]);
        }
        snprintf(result->adaptive_added_indices,
                 sizeof(result->adaptive_added_indices), "%s",
                 added.empty() ? "none" : added.c_str());
    }

    for (size_t i = 0; i < ranges.size(); ++i) {
        basis_cols += ranges[i].second - ranges[i].first + 1;
        if (ranges[i].first <= k - 1 && k - 1 <= ranges[i].second) {
            result->adaptive_target_block_start = ranges[i].first;
            result->adaptive_target_block_end = ranges[i].second;
        }
    }
    if (basis_cols < k) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    block_maxiter = ipt_cuda_env_int("IPT_BLOCK_CLUSTER_MAXITER",
                                     maxiter > 0 ? maxiter : 100);
    if (block_maxiter <= 0) {
        block_maxiter = 100;
    }
    block_tol = ipt_cuda_env_double(
        "IPT_BLOCK_CLUSTER_TOL", use_tolerance ? tol : 1.0e-12);
    damping = ipt_cuda_env_double("IPT_BLOCK_CLUSTER_DAMPING", 1.0);
    result->preparation_time_sec =
        ipt_block_elapsed_seconds(preparation_start);

    if (ipt_preparation_debug_enabled()) {
        fprintf(stderr,
                "IPT block-cluster: clusters=%zu basis_cols=%d "
                "requested_k=%d basis_target_k=%d oversample=%d "
                "adaptive=%d adaptive_tau=%.3e adaptive_added=%s "
                "maxiter=%d tol=%.3e damping=%.3e\n",
                ranges.size(), basis_cols, k, basis_target_k, oversample,
                adaptive_enabled, adaptive_coupling_tau,
                result->adaptive_added_indices, block_maxiter, block_tol,
                damping);
        for (size_t i = 0; i < ranges.size(); ++i) {
            fprintf(stderr, "IPT block-cluster: block[%zu]=%d:%d size=%d\n",
                    i, ranges[i].first, ranges[i].second,
                    ranges[i].second - ranges[i].first + 1);
        }
    }

    {
        std::chrono::steady_clock::time_point setup_start =
            std::chrono::steady_clock::now();

        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CUSPARSE_CHECK(cusparseCreate(&sparse_handle));
        CUDA_CHECK(cudaMalloc((void **)&d_col_ptr,
                              (size_t)(n + 1) * sizeof(int)));
        CUDA_CHECK(
            cudaMalloc((void **)&d_row_ind, (size_t)nnz * sizeof(int)));
        CUDA_CHECK(
            cudaMalloc((void **)&d_values, (size_t)nnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_diagonal,
                              (size_t)n * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_perm, (size_t)n * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_combined_basis,
                              (size_t)n * (size_t)basis_cols *
                                  sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_col_ptr, sorted_col_ptr.data(),
                              (size_t)(n + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_row_ind, sorted_row_ind.data(),
                              (size_t)nnz * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values, sorted_values.data(),
                              (size_t)nnz * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_diagonal, diagonal.data(),
                              (size_t)n * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_perm, perm.data(), (size_t)n * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUSPARSE_CHECK(cusparseCreateCsr(
            &sparse_matrix, n, n, nnz, d_col_ptr, d_row_ind, d_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        CUDA_CHECK(cudaDeviceSynchronize());
        result->transfer_setup_time_sec =
            ipt_block_elapsed_seconds(setup_start);
    }

    {
        int offset = 0;

        for (size_t range_id = 0; range_id < ranges.size(); ++range_id) {
            double *d_block_basis = NULL;
            int block_iterations = 0;
            int first = ranges[range_id].first;
            int last = ranges[range_id].second;
            int width = last - first + 1;
            double block_iteration_time = 0.0;
            double block_fixed_point_residual = NAN;

            status = ipt_block_cluster_solve_basis_gpu(
                sparse_handle, sparse_matrix, cublas_handle, d_diagonal, n,
                width, first, last, block_maxiter, block_tol, damping,
                &d_block_basis, &block_iterations, &block_iteration_time,
                &block_fixed_point_residual);
            if (status != IPT_CUDA_SUCCESS) {
                cudaFree(d_block_basis);
                goto cleanup;
            }
            iterations_done = std::max(iterations_done, block_iterations);
            if (isfinite(block_fixed_point_residual) &&
                (!isfinite(result->fixed_point_residual) ||
                 block_fixed_point_residual >
                     result->fixed_point_residual)) {
                result->fixed_point_residual = block_fixed_point_residual;
            }
            result->iteration_time_sec += block_iteration_time;
            result->matvecs +=
                (long long)block_iterations * (long long)width;
            CUDA_CHECK(cudaMemcpy(
                d_combined_basis + (size_t)offset * (size_t)n, d_block_basis,
                (size_t)n * (size_t)width * sizeof(double),
                cudaMemcpyDeviceToDevice));
            cudaFree(d_block_basis);
            offset += width;
        }
    }

    status = ipt_block_cluster_rayleigh_ritz_gpu(
        sparse_handle, sparse_matrix, cublas_handle, d_combined_basis, n,
        basis_cols, k, d_perm, &d_ritz_values, &d_ritz_vectors,
        &result->rayleigh_ritz_time_sec, &result->rayleigh_ritz_used_qr,
        &result->basis_orthogonality_frobenius_error,
        &result->basis_orthogonality_max_abs_error,
        &result->basis_has_nan_or_inf,
        &result->ritz_vectors_orthogonality_frobenius_error,
        &result->ritz_vectors_orthogonality_max_abs_error,
        &result->ritz_vectors_has_nan_or_inf);
    if (status != IPT_CUDA_SUCCESS) {
        goto cleanup;
    }
    result->matvecs += basis_cols;
    if (ipt_cuda_env_flag("IPT_DAVIDSON_ENRICH")) {
        std::vector<double> host_ritz_values((size_t)k, 0.0);
        std::vector<double> host_ritz_vectors(
            (size_t)n * (size_t)k, 0.0);
        std::vector<double> correction;
        std::vector<double> sorted_correction((size_t)n, 0.0);
        double matrix_norm = ipt_block_host_matrix_inf_norm(
            col_ptr, row_ind, matrix_values, n);
        int worst_index = -1;

        result->davidson_attempted = 1;
        result->davidson_denom_clip = ipt_cuda_env_double(
            "IPT_DAVIDSON_DENOM_CLIP", 1.0e-8);
        if (result->davidson_denom_clip <= 0.0) {
            result->davidson_denom_clip = 1.0e-8;
        }
        CUDA_CHECK(cudaMemcpy(host_ritz_vectors.data(), d_ritz_vectors,
                              host_ritz_vectors.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_ritz_values.data(), d_ritz_values,
                              host_ritz_values.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
        result->davidson_residual_before = ipt_block_host_max_residual(
            col_ptr, row_ind, matrix_values, n, k, host_ritz_values,
            host_ritz_vectors, matrix_norm, &worst_index);
        result->davidson_target_index = worst_index;
        if (worst_index >= 0 &&
            ipt_block_build_davidson_correction(
                col_ptr, row_ind, matrix_values, n, k, original_diagonal,
                host_ritz_values, host_ritz_vectors, worst_index,
                result->davidson_denom_clip, &correction)) {
            double *d_enriched_basis = NULL;
            double *d_enriched_values = NULL;
            double *d_enriched_vectors = NULL;
            double enriched_rr_time = 0.0;
            double enriched_basis_frobenius = NAN;
            double enriched_basis_max = NAN;
            double enriched_ritz_frobenius = NAN;
            double enriched_ritz_max = NAN;
            int enriched_basis_invalid = 0;
            int enriched_ritz_invalid = 0;
            int enriched_used_qr = 0;
            std::vector<double> enriched_values((size_t)k, 0.0);
            std::vector<double> enriched_vectors(
                (size_t)n * (size_t)k, 0.0);
            int enriched_worst_index = -1;

            for (int sorted = 0; sorted < n; ++sorted) {
                sorted_correction[(size_t)sorted] =
                    correction[(size_t)perm[(size_t)sorted]];
            }
            CUDA_CHECK(cudaMalloc(
                (void **)&d_enriched_basis,
                (size_t)n * (size_t)(basis_cols + 1) * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(
                d_enriched_basis, d_combined_basis,
                (size_t)n * (size_t)basis_cols * sizeof(double),
                cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(
                d_enriched_basis + (size_t)n * (size_t)basis_cols,
                sorted_correction.data(), (size_t)n * sizeof(double),
                cudaMemcpyHostToDevice));
            status = ipt_block_cluster_rayleigh_ritz_gpu(
                sparse_handle, sparse_matrix, cublas_handle,
                d_enriched_basis, n, basis_cols + 1, k, d_perm,
                &d_enriched_values, &d_enriched_vectors, &enriched_rr_time,
                &enriched_used_qr, &enriched_basis_frobenius,
                &enriched_basis_max, &enriched_basis_invalid,
                &enriched_ritz_frobenius, &enriched_ritz_max,
                &enriched_ritz_invalid);
            cudaFree(d_enriched_basis);
            if (status != IPT_CUDA_SUCCESS) {
                cudaFree(d_enriched_values);
                cudaFree(d_enriched_vectors);
                goto cleanup;
            }
            result->rayleigh_ritz_time_sec += enriched_rr_time;
            result->matvecs += basis_cols + 1;
            CUDA_CHECK(cudaMemcpy(
                enriched_vectors.data(), d_enriched_vectors,
                enriched_vectors.size() * sizeof(double),
                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(
                enriched_values.data(), d_enriched_values,
                enriched_values.size() * sizeof(double),
                cudaMemcpyDeviceToHost));
            result->davidson_residual_after = ipt_block_host_max_residual(
                col_ptr, row_ind, matrix_values, n, k, enriched_values,
                enriched_vectors, matrix_norm, &enriched_worst_index);
            if (isfinite(result->davidson_residual_after) &&
                result->davidson_residual_after <
                    result->davidson_residual_before) {
                cudaFree(d_ritz_values);
                cudaFree(d_ritz_vectors);
                d_ritz_values = d_enriched_values;
                d_ritz_vectors = d_enriched_vectors;
                d_enriched_values = NULL;
                d_enriched_vectors = NULL;
                ++basis_cols;
                result->davidson_accepted = 1;
                result->rayleigh_ritz_used_qr = enriched_used_qr;
                result->basis_orthogonality_frobenius_error =
                    enriched_basis_frobenius;
                result->basis_orthogonality_max_abs_error =
                    enriched_basis_max;
                result->basis_has_nan_or_inf = enriched_basis_invalid;
                result->ritz_vectors_orthogonality_frobenius_error =
                    enriched_ritz_frobenius;
                result->ritz_vectors_orthogonality_max_abs_error =
                    enriched_ritz_max;
                result->ritz_vectors_has_nan_or_inf =
                    enriched_ritz_invalid;
            }
            cudaFree(d_enriched_values);
            cudaFree(d_enriched_vectors);
        }
    }
    result->solve_time_sec =
        result->iteration_time_sec + result->rayleigh_ritz_time_sec;

    result->vectors =
        (double *)calloc((size_t)n * (size_t)k, sizeof(double));
    result->values = (double *)calloc((size_t)k, sizeof(double));
    if (result->vectors == NULL || result->values == NULL) {
        status = IPT_CUDA_ALLOCATION_FAILED;
        goto cleanup;
    }

    CUDA_CHECK(cudaMemcpy(result->vectors, d_ritz_vectors,
                          (size_t)n * (size_t)k * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result->values, d_ritz_values,
                          (size_t)k * sizeof(double),
                          cudaMemcpyDeviceToHost));
    result->n = n;
    result->k = k;
    result->iterations = iterations_done;
    result->basis_cols = basis_cols;

cleanup:
    cudaFree(d_col_ptr);
    cudaFree(d_row_ind);
    cudaFree(d_perm);
    cudaFree(d_values);
    cudaFree(d_diagonal);
    cudaFree(d_combined_basis);
    cudaFree(d_ritz_values);
    cudaFree(d_ritz_vectors);
    if (sparse_matrix != NULL) {
        cusparseDestroySpMat(sparse_matrix);
    }
    if (sparse_handle != NULL) {
        cusparseDestroy(sparse_handle);
    }
    if (cublas_handle != NULL) {
        cublasDestroy(cublas_handle);
    }
    if (status != IPT_CUDA_SUCCESS) {
        ipt_cuda_free_result(result);
    }
    return status;
}

static int ipt_cuda_sparse_csc_with_initial_impl(
    const int *col_ptr, const int *row_ind, const double *matrix_values,
    const double *initial_vectors_col_major, int n, int k, int nnz,
    int maxiter, double tol, int use_tolerance, IPTCudaResult *result)
{
    int status = IPT_CUDA_SUCCESS;
    int iterations_done = 0;
    size_t vector_elements = 0;
    size_t vector_bytes = 0;
    size_t col_ptr_bytes = 0;
    size_t nnz_int_bytes = 0;
    size_t nnz_value_bytes = 0;
    cublasHandle_t handle = NULL;
    int *d_col_ptr = NULL;
    int *d_row_ind = NULL;
    double *d_matrix_values = NULL;
    IPTPreparedCsc prepared;
    double *d_x_a = NULL;
    double *d_x_b = NULL;
    double *d_diagonal = NULL;
    double *d_g = NULL;
    double *d_column_diagonal = NULL;
    double *d_delta = NULL;
    double *d_values = NULL;
    double *d_x = NULL;
    double *d_y = NULL;
    cusparseHandle_t sparse_handle = NULL;
    cusparseSpMatDescr_t sparse_matrix = NULL;
    cusparseDnMatDescr_t dense_x = NULL;
    cusparseDnMatDescr_t dense_y = NULL;
    void *d_spmm_buffer = NULL;
    size_t spmm_buffer_size = 0;
    const int block_size = 256;

    ipt_prepared_csc_init(&prepared);

    if (result != NULL) {
        result->n = 0;
        result->k = 0;
        result->iterations = 0;
        result->vectors = NULL;
        result->values = NULL;
        result->basis_cols = 0;
        result->fixed_point_residual = NAN;
    }

    if (col_ptr == NULL || row_ind == NULL || matrix_values == NULL ||
        result == NULL || n <= 0 || k <= 0 || k > n || nnz < 0 ||
        maxiter < 0 || (use_tolerance && (tol <= 0.0 || maxiter <= 0))) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    if (initial_vectors_col_major == NULL &&
        ipt_cuda_env_flag("IPT_BLOCK_CLUSTER")) {
        return ipt_cuda_sparse_csc_block_cluster_impl(
            col_ptr, row_ind, matrix_values, n, k, nnz, maxiter, tol,
            use_tolerance, result);
    }

    {
        double degeneracy_threshold =
            ipt_cuda_env_double("IPT_DEGENERACY_THRESHOLD", 0.0);
        int prepare_status =
            ipt_prepare_sparse_csc(col_ptr, row_ind, matrix_values, n, k, nnz,
                                   1, 1, degeneracy_threshold, &prepared);
        if (prepare_status == 1) {
            status = IPT_CUDA_INVALID_ARGUMENT;
            goto cleanup;
        }
        if (prepare_status == 2) {
            status = IPT_CUDA_ALLOCATION_FAILED;
            goto cleanup;
        }
    }

    vector_elements = (size_t)n * (size_t)k;

    if (vector_elements > (size_t)INT_MAX) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    vector_bytes = vector_elements * sizeof(double);
    col_ptr_bytes = ((size_t)n + 1U) * sizeof(int);
    nnz_int_bytes = (size_t)prepared.nnz * sizeof(int);
    nnz_value_bytes = (size_t)prepared.nnz * sizeof(double);

    result->vectors = (double *)calloc(vector_elements, sizeof(double));
    result->values = (double *)calloc((size_t)k, sizeof(double));

    if (result->vectors == NULL || result->values == NULL) {
        status = IPT_CUDA_ALLOCATION_FAILED;
        goto cleanup;
    }

    CUDA_CHECK(cudaMalloc((void **)&d_col_ptr, col_ptr_bytes));
    if (prepared.nnz > 0) {
        CUDA_CHECK(cudaMalloc((void **)&d_row_ind, nnz_int_bytes));
        CUDA_CHECK(cudaMalloc((void **)&d_matrix_values, nnz_value_bytes));
    }
    CUDA_CHECK(cudaMalloc((void **)&d_x_a, vector_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_x_b, vector_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_diagonal, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_g, vector_bytes));
    CUDA_CHECK(
        cudaMalloc((void **)&d_column_diagonal, (size_t)k * sizeof(double)));
    if (use_tolerance) {
        CUDA_CHECK(cudaMalloc((void **)&d_delta, vector_bytes));
    }
    CUDA_CHECK(cudaMalloc((void **)&d_values, (size_t)k * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_col_ptr, prepared.col_ptr, col_ptr_bytes,
                          cudaMemcpyHostToDevice));
    if (prepared.nnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_row_ind, prepared.row_ind, nnz_int_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_matrix_values, prepared.values,
                              nnz_value_bytes,
                              cudaMemcpyHostToDevice));
    }

    if (initial_vectors_col_major == NULL) {
        int vector_blocks = ((int)vector_elements + block_size - 1) / block_size;
        set_identity_kernel<<<vector_blocks, block_size>>>(d_x_a, n, k);
        CUDA_CHECK(cudaGetLastError());
    } else {
        CUDA_CHECK(cudaMemcpy(d_x_a, initial_vectors_col_major, vector_bytes,
                              cudaMemcpyHostToDevice));
    }

    {
        int vector_blocks = ((int)vector_elements + block_size - 1) / block_size;

        CUDA_CHECK(cudaMemcpy(d_diagonal, prepared.diagonal,
                              (size_t)n * sizeof(double),
                              cudaMemcpyHostToDevice));
        build_g_kernel<<<vector_blocks, block_size>>>(d_diagonal, d_g, n, k,
                                                      0.0, NULL);
        CUDA_CHECK(cudaGetLastError());
    }

    CUBLAS_CHECK(cublasCreate(&handle));
    CUSPARSE_CHECK(cusparseCreate(&sparse_handle));
    CUSPARSE_CHECK(cusparseCreateCsr(
        &sparse_matrix, n, n, prepared.nnz, (void *)d_col_ptr,
        (void *)d_row_ind, (void *)d_matrix_values, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_x, n, k, n, d_x_a, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_y, n, k, n, d_x_b, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    {
        const double one = 1.0;
        const double zero = 0.0;

        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            sparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &one, sparse_matrix, dense_x,
            &zero, dense_y, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
            &spmm_buffer_size));
        if (spmm_buffer_size > 0) {
            CUDA_CHECK(cudaMalloc(&d_spmm_buffer, spmm_buffer_size));
        }
    }

    d_x = d_x_a;
    d_y = d_x_b;

    for (int iter = 0; iter < maxiter; ++iter) {
        int col_blocks = (k + block_size - 1) / block_size;
        int vector_blocks = ((int)vector_elements + block_size - 1) / block_size;
        double *tmp = NULL;

        CUSPARSE_CHECK(ipt_cusparse_spmm(sparse_handle, sparse_matrix,
                                         dense_x, dense_y, d_x, d_y,
                                         d_spmm_buffer));

        column_diagonal_after_d_kernel<<<col_blocks, block_size>>>(
            d_y, d_x, d_diagonal, d_column_diagonal, n, k);
        CUDA_CHECK(cudaGetLastError());

        ipt_update_kernel<<<vector_blocks, block_size>>>(
            d_y, d_x, d_diagonal, d_g, d_column_diagonal, n, k);
        CUDA_CHECK(cudaGetLastError());

        iterations_done = iter + 1;

        if (use_tolerance) {
            double delta_norm = 0.0;
            double current_norm = 0.0;
            double fixed_point_residual = 0.0;

            fixed_point_delta_kernel<<<vector_blocks, block_size>>>(
                d_delta, d_y, d_x, (int)vector_elements);
            CUDA_CHECK(cudaGetLastError());

            CUBLAS_CHECK(cublasDnrm2(handle, (int)vector_elements, d_delta, 1,
                                     &delta_norm));
            CUBLAS_CHECK(cublasDnrm2(handle, (int)vector_elements, d_x, 1,
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
        }

        tmp = d_x;
        d_x = d_y;
        d_y = tmp;
    }

    CUSPARSE_CHECK(ipt_cusparse_spmm(sparse_handle, sparse_matrix, dense_x,
                                     dense_y, d_x, d_y, d_spmm_buffer));

    {
        int col_blocks = (k + block_size - 1) / block_size;

        gather_values_kernel<<<col_blocks, block_size>>>(d_y, d_values, n, k);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(result->vectors, d_x, vector_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result->values, d_values, (size_t)k * sizeof(double),
                          cudaMemcpyDeviceToHost));
    if (!ipt_apply_prepared_q_to_vectors(&prepared, result->vectors, k)) {
        status = IPT_CUDA_INVALID_ARGUMENT;
        goto cleanup;
    }

    result->n = n;
    result->k = k;
    result->iterations = iterations_done;
    result->basis_cols = k;

cleanup:
    if (handle != NULL) {
        cublasDestroy(handle);
    }

    cudaFree(d_col_ptr);
    cudaFree(d_row_ind);
    cudaFree(d_matrix_values);
    cudaFree(d_x_a);
    cudaFree(d_x_b);
    cudaFree(d_diagonal);
    cudaFree(d_g);
    cudaFree(d_column_diagonal);
    cudaFree(d_delta);
    cudaFree(d_values);
    cudaFree(d_spmm_buffer);
    if (dense_x != NULL) {
        cusparseDestroyDnMat(dense_x);
    }
    if (dense_y != NULL) {
        cusparseDestroyDnMat(dense_y);
    }
    if (sparse_matrix != NULL) {
        cusparseDestroySpMat(sparse_matrix);
    }
    if (sparse_handle != NULL) {
        cusparseDestroy(sparse_handle);
    }
    ipt_prepared_csc_free(&prepared);

    if (status != IPT_CUDA_SUCCESS) {
        ipt_cuda_free_result(result);
    }

    return status;
}

extern "C" int ipt_cuda_sparse_csc_with_initial(
    const int *col_ptr, const int *row_ind, const double *matrix_values,
    const double *initial_vectors_col_major, int n, int k, int nnz,
    int iterations, IPTCudaResult *result)
{
    return ipt_cuda_sparse_csc_with_initial_impl(
        col_ptr, row_ind, matrix_values, initial_vectors_col_major, n, k, nnz,
        iterations, 0.0, 0, result);
}

extern "C" int ipt_cuda_sparse_csc_with_initial_tol(
    const int *col_ptr, const int *row_ind, const double *matrix_values,
    const double *initial_vectors_col_major, int n, int k, int nnz, double tol,
    int maxiter, IPTCudaResult *result)
{
    return ipt_cuda_sparse_csc_with_initial_impl(
        col_ptr, row_ind, matrix_values, initial_vectors_col_major, n, k, nnz,
        maxiter, tol, 1, result);
}

#undef CUDA_CHECK
#undef CUBLAS_CHECK
#undef CUSPARSE_CHECK
#undef CUSOLVER_CHECK
