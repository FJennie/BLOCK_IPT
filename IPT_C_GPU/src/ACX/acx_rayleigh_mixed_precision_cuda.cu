#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <time.h>

#include "acx_mixed_precision_cuda.cu"

#ifndef ACX_RAYLEIGH_MIXED_PRECISION_CUDA_DECLS
#define ACX_RAYLEIGH_MIXED_PRECISION_CUDA_DECLS

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
    double orthogonalize_time_sec;
    double rayleigh_transform_time_sec;
    double fp64_time_sec;
    double finalization_time_sec;
    double backtransform_time_sec;
    double total_time_sec;
    double tf32_fixed_point_residual;
    double fp64_fixed_point_residual;
    double fp64_max_residual;
    double *d_vectors;
    double *d_values;
} ACXRayleighMixedTF32CudaResult;

void acx_rayleigh_mixed_tf32_cuda_reset_result(
    ACXRayleighMixedTF32CudaResult *result);
void acx_rayleigh_mixed_tf32_cuda_free_result(
    ACXRayleighMixedTF32CudaResult *result);
int acx_rayleigh_mixed_tf32_cuda_device_tol(
    const double *d_matrix_col_major, int n, int k, double tf32_tol,
    int tf32_maxiter, double fp64_tol, int fp64_maxiter,
    cublasHandle_t handle, ACXRayleighMixedTF32CudaResult *result);

#ifdef __cplusplus
}
#endif

#endif

extern "C" void acx_rayleigh_mixed_tf32_cuda_reset_result(
    ACXRayleighMixedTF32CudaResult *result)
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
    result->orthogonalize_time_sec = NAN;
    result->rayleigh_transform_time_sec = NAN;
    result->fp64_time_sec = NAN;
    result->finalization_time_sec = NAN;
    result->backtransform_time_sec = NAN;
    result->total_time_sec = NAN;
    result->tf32_fixed_point_residual = NAN;
    result->fp64_fixed_point_residual = NAN;
    result->fp64_max_residual = NAN;
    result->d_vectors = NULL;
    result->d_values = NULL;
}

extern "C" void acx_rayleigh_mixed_tf32_cuda_free_result(
    ACXRayleighMixedTF32CudaResult *result)
{
    if (result == NULL) {
        return;
    }

    cudaFree(result->d_vectors);
    cudaFree(result->d_values);
    acx_rayleigh_mixed_tf32_cuda_reset_result(result);
}

__global__ static void acx_rayleigh_symmetrize_kernel(double *matrix, int n,
                                                      int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total) {
        return;
    }

    {
        int row = idx % n;
        int col = idx / n;

        if (row < col) {
            double upper = matrix[row + (size_t)col * (size_t)n];
            double lower = matrix[col + (size_t)row * (size_t)n];
            double avg = 0.5 * (upper + lower);

            matrix[row + (size_t)col * (size_t)n] = avg;
            matrix[col + (size_t)row * (size_t)n] = avg;
        }
    }
}

static int acx_rayleigh_normalize_columns(double *d_matrix_col_major, int n,
                                          int k, cublasHandle_t handle)
{
    if (d_matrix_col_major == NULL || handle == NULL || n <= 0 || k <= 0) {
        return ACX_CUDA_INVALID_ARGUMENT;
    }

    for (int col = 0; col < k; ++col) {
        double norm = 0.0;
        double scale = 0.0;
        cublasStatus_t cublas_status =
            cublasDnrm2(handle, n, d_matrix_col_major + (size_t)col * n, 1,
                        &norm);

        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            return ACX_CUDA_CUBLAS_ERROR;
        }
        if (norm <= 0.0 || !isfinite(norm)) {
            return ACX_CUDA_INVALID_ARGUMENT;
        }

        scale = 1.0 / norm;
        cublas_status =
            cublasDscal(handle, n, &scale,
                        d_matrix_col_major + (size_t)col * n, 1);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            return ACX_CUDA_CUBLAS_ERROR;
        }
    }

    return ACX_CUDA_SUCCESS;
}

static int acx_rayleigh_orthogonalize_qr(double *d_q, int n, int k)
{
    int status = ACX_CUDA_SUCCESS;
    cusolverDnHandle_t solver = NULL;
    cusolverStatus_t solver_status;
    int lwork_geqrf = 0;
    int lwork_orgqr = 0;
    int lwork = 0;
    double *d_work = NULL;
    double *d_tau = NULL;
    int *d_info = NULL;
    int h_info = 0;
    cudaError_t cuda_status;

    if (d_q == NULL || n <= 0 || k <= 0 || k > n) {
        return ACX_CUDA_INVALID_ARGUMENT;
    }

    solver_status = cusolverDnCreate(&solver);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
        return ACX_CUDA_CUBLAS_ERROR;
    }

    solver_status = cusolverDnDgeqrf_bufferSize(solver, n, k, d_q, n,
                                                &lwork_geqrf);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
        status = ACX_CUDA_CUBLAS_ERROR;
        goto cleanup;
    }

    cuda_status = cudaMalloc((void **)&d_tau, (size_t)k * sizeof(double));
    if (cuda_status != cudaSuccess) {
        status = ACX_CUDA_CUDA_ERROR;
        goto cleanup;
    }

    solver_status = cusolverDnDorgqr_bufferSize(solver, n, k, k, d_q, n,
                                                d_tau, &lwork_orgqr);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
        status = ACX_CUDA_CUBLAS_ERROR;
        goto cleanup;
    }

    lwork = lwork_geqrf > lwork_orgqr ? lwork_geqrf : lwork_orgqr;
    if (lwork <= 0) {
        status = ACX_CUDA_INVALID_ARGUMENT;
        goto cleanup;
    }

    cuda_status = cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(double));
    if (cuda_status != cudaSuccess) {
        status = ACX_CUDA_CUDA_ERROR;
        goto cleanup;
    }
    cuda_status = cudaMalloc((void **)&d_info, sizeof(int));
    if (cuda_status != cudaSuccess) {
        status = ACX_CUDA_CUDA_ERROR;
        goto cleanup;
    }

    solver_status =
        cusolverDnDgeqrf(solver, n, k, d_q, n, d_tau, d_work, lwork, d_info);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
        status = ACX_CUDA_CUBLAS_ERROR;
        goto cleanup;
    }
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        status = ACX_CUDA_CUDA_ERROR;
        goto cleanup;
    }
    cuda_status = cudaMemcpy(&h_info, d_info, sizeof(int),
                             cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        status = ACX_CUDA_CUDA_ERROR;
        goto cleanup;
    }
    if (h_info != 0) {
        status = ACX_CUDA_INVALID_ARGUMENT;
        goto cleanup;
    }

    solver_status =
        cusolverDnDorgqr(solver, n, k, k, d_q, n, d_tau, d_work, lwork,
                         d_info);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
        status = ACX_CUDA_CUBLAS_ERROR;
        goto cleanup;
    }
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        status = ACX_CUDA_CUDA_ERROR;
        goto cleanup;
    }
    cuda_status = cudaMemcpy(&h_info, d_info, sizeof(int),
                             cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        status = ACX_CUDA_CUDA_ERROR;
        goto cleanup;
    }
    if (h_info != 0) {
        status = ACX_CUDA_INVALID_ARGUMENT;
        goto cleanup;
    }

cleanup:
    cudaFree(d_work);
    cudaFree(d_tau);
    cudaFree(d_info);
    if (solver != NULL) {
        cusolverDnDestroy(solver);
    }

    return status;
}

#define ACX_RAYLEIGH_CUDA_CHECK(call)                                        \
    do {                                                                     \
        cudaError_t rayleigh_cuda_status = (call);                           \
        if (rayleigh_cuda_status != cudaSuccess) {                           \
            status = ACX_CUDA_CUDA_ERROR;                                    \
            goto cleanup;                                                    \
        }                                                                    \
    } while (0)

#define ACX_RAYLEIGH_CUBLAS_CHECK(call)                                      \
    do {                                                                     \
        cublasStatus_t rayleigh_cublas_status = (call);                      \
        if (rayleigh_cublas_status != CUBLAS_STATUS_SUCCESS) {               \
            status = ACX_CUDA_CUBLAS_ERROR;                                  \
            goto cleanup;                                                    \
        }                                                                    \
    } while (0)

extern "C" int acx_rayleigh_mixed_tf32_cuda_device_tol(
    const double *d_matrix_col_major, int n, int k, double tf32_tol,
    int tf32_maxiter, double fp64_tol, int fp64_maxiter,
    cublasHandle_t handle, ACXRayleighMixedTF32CudaResult *result)
{
    int status = ACX_CUDA_SUCCESS;
    double *d_q = NULL;
    double *d_aq = NULL;
    double *d_b = NULL;
    double *d_vectors = NULL;
    double *d_values = NULL;
    ACXCudaResult fp64_result;
    double stage_start = 0.0;
    double tf32_residual_check_time = 0.0;
    double raw_tf32_time = 0.0;
    size_t q_elements = 0;
    size_t b_elements = 0;
    const double one = 1.0;
    const double zero = 0.0;

    if (result == NULL) {
        return ACX_CUDA_INVALID_ARGUMENT;
    }
    acx_rayleigh_mixed_tf32_cuda_reset_result(result);
    acx_cuda_reset_result(&fp64_result);

    if (d_matrix_col_major == NULL || handle == NULL || n <= 0 || k <= 0 ||
        k > n || tf32_tol <= 0.0 || fp64_tol <= 0.0 ||
        tf32_maxiter <= 0 || fp64_maxiter <= 0) {
        return ACX_CUDA_INVALID_ARGUMENT;
    }

    q_elements = (size_t)n * (size_t)k;
    b_elements = (size_t)k * (size_t)k;
    if (q_elements > (size_t)INT_MAX || b_elements > (size_t)INT_MAX) {
        return ACX_CUDA_INVALID_ARGUMENT;
    }

    stage_start = acx_mixed_now_seconds();
    status = acx_tf32_initial_device_tol(
        d_matrix_col_major, n, k, tf32_tol, tf32_maxiter, handle, &d_q,
        &result->tf32_iterations, &result->tf32_f_calls,
        &result->tf32_converged, &result->tf32_fixed_point_residual,
        &tf32_residual_check_time);
    if (status != ACX_CUDA_SUCCESS) {
        goto cleanup;
    }
    ACX_RAYLEIGH_CUDA_CHECK(cudaDeviceSynchronize());
    raw_tf32_time = acx_mixed_now_seconds() - stage_start;
    result->tf32_time_sec = raw_tf32_time - tf32_residual_check_time;
    if (result->tf32_time_sec < 0.0) {
        result->tf32_time_sec = 0.0;
    }

    stage_start = acx_mixed_now_seconds();
    status = acx_rayleigh_normalize_columns(d_q, n, k, handle);
    if (status != ACX_CUDA_SUCCESS) {
        goto cleanup;
    }
    status = acx_rayleigh_orthogonalize_qr(d_q, n, k);
    if (status != ACX_CUDA_SUCCESS) {
        goto cleanup;
    }
    ACX_RAYLEIGH_CUDA_CHECK(cudaDeviceSynchronize());
    result->orthogonalize_time_sec =
        acx_mixed_now_seconds() - stage_start;

    stage_start = acx_mixed_now_seconds();
    ACX_RAYLEIGH_CUDA_CHECK(
        cudaMalloc((void **)&d_aq, q_elements * sizeof(double)));
    ACX_RAYLEIGH_CUDA_CHECK(
        cudaMalloc((void **)&d_b, b_elements * sizeof(double)));

    ACX_RAYLEIGH_CUBLAS_CHECK(cublasDgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &one, d_matrix_col_major,
        n, d_q, n, &zero, d_aq, n));
    ACX_RAYLEIGH_CUBLAS_CHECK(cublasDgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, k, k, n, &one, d_q, n, d_aq, n,
        &zero, d_b, k));

    {
        const int block_size = 256;
        int blocks = ((int)b_elements + block_size - 1) / block_size;

        acx_rayleigh_symmetrize_kernel<<<blocks, block_size>>>(
            d_b, k, (int)b_elements);
        ACX_RAYLEIGH_CUDA_CHECK(cudaGetLastError());
    }
    ACX_RAYLEIGH_CUDA_CHECK(cudaDeviceSynchronize());
    result->rayleigh_transform_time_sec =
        acx_mixed_now_seconds() - stage_start;

    stage_start = acx_mixed_now_seconds();
    status = acx_cuda_device_tol(d_b, k, k, fp64_tol, fp64_maxiter, handle,
                                 &fp64_result);
    if (status != ACX_CUDA_SUCCESS) {
        goto cleanup;
    }
    ACX_RAYLEIGH_CUDA_CHECK(cudaDeviceSynchronize());
    result->fp64_time_sec = acx_mixed_now_seconds() - stage_start;
    result->fp64_iterations = fp64_result.iterations;
    result->fp64_f_calls = fp64_result.f_calls;
    result->fp64_converged = fp64_result.converged;
    result->finalization_time_sec = fp64_result.finalization_time_sec;
    result->fp64_fixed_point_residual = fp64_result.fixed_point_residual;
    result->fp64_max_residual = fp64_result.max_residual;

    stage_start = acx_mixed_now_seconds();
    ACX_RAYLEIGH_CUDA_CHECK(
        cudaMalloc((void **)&d_vectors, q_elements * sizeof(double)));
    ACX_RAYLEIGH_CUBLAS_CHECK(cublasDgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, k, &one, d_q, n,
        fp64_result.d_vectors, k, &zero, d_vectors, n));
    ACX_RAYLEIGH_CUDA_CHECK(cudaDeviceSynchronize());
    result->backtransform_time_sec =
        acx_mixed_now_seconds() - stage_start;

    d_values = fp64_result.d_values;
    fp64_result.d_values = NULL;

    result->total_time_sec =
        result->tf32_time_sec + result->orthogonalize_time_sec +
        result->rayleigh_transform_time_sec + result->fp64_time_sec +
        result->backtransform_time_sec;
    result->n = n;
    result->k = k;
    result->d_vectors = d_vectors;
    result->d_values = d_values;
    d_vectors = NULL;
    d_values = NULL;

cleanup:
    cudaFree(d_q);
    cudaFree(d_aq);
    cudaFree(d_b);
    cudaFree(d_vectors);
    cudaFree(d_values);
    acx_cuda_free_result(&fp64_result);
    if (status != ACX_CUDA_SUCCESS) {
        acx_rayleigh_mixed_tf32_cuda_free_result(result);
    }
    return status;
}

#undef ACX_RAYLEIGH_CUDA_CHECK
#undef ACX_RAYLEIGH_CUBLAS_CHECK
