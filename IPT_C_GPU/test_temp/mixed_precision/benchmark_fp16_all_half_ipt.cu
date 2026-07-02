#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#define DEFAULT_ROOT "/fs1/home/nudt_liujie/ftt/IPT_C_GPU"
#define DEFAULT_N 26000
#define DEFAULT_EPSILON 0.01
#define DEFAULT_TOL 1.0e-3
#define DEFAULT_MAXITER 1000
#define DEFAULT_FP64_TOL 1.0e-12
#define DEFAULT_FP64_MAXITER 1000
#define DEFAULT_SEED 20260603ULL
#define DEFAULT_WARMUP_N 128
#define ERROR_LEN 512

typedef struct {
    double initial_time_sec;
    double initial_residual_check_time_sec;
    double fp64_time_sec;
    double total_time_sec;
    double initial_fixed_point_residual;
    double fp64_fixed_point_residual;
    int initial_iterations;
    int fp64_iterations;
    int bad_count;
    char status[32];
    char initial_status[32];
    char fp64_status[32];
    char error[ERROR_LEN];
} TensorHalfResult;

static double now_seconds(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int ensure_directory(const char *path)
{
    char tmp[4096];
    size_t len = 0;

    if (path == NULL || path[0] == '\0') {
        return -1;
    }

    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    while (len > 1 && tmp[len - 1] == '/') {
        tmp[--len] = '\0';
    }

    for (char *p = tmp + 1; *p != '\0'; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0775) != 0 && errno != EEXIST) {
                return -1;
            }
            *p = '/';
        }
    }

    if (mkdir(tmp, 0775) != 0 && errno != EEXIST) {
        return -1;
    }

    return 0;
}

static const char *env_or_default(const char *name, const char *fallback)
{
    const char *value = getenv(name);

    return (value == NULL || value[0] == '\0') ? fallback : value;
}

static const char *cublas_status_name_local(cublasStatus_t status)
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

static uint64_t splitmix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static double uniform01(uint64_t seed, int n, int row, int col)
{
    uint64_t x = seed + (uint64_t)n * 0xd1b54a32d192ed03ULL;

    x ^= (uint64_t)(row + 1) * 0x9e3779b97f4a7c15ULL;
    x ^= (uint64_t)(col + 1) * 0xbf58476d1ce4e5b9ULL;
    return (double)(splitmix64(x) >> 11) * (1.0 / 9007199254740992.0);
}

static uint64_t epsilon_seed_offset(double epsilon)
{
    static const double default_eps[] = {0.01, 0.02, 0.03,
                                         0.04, 0.05, 0.06};

    for (int i = 0; i < (int)(sizeof(default_eps) / sizeof(default_eps[0]));
         ++i) {
        if (fabs(epsilon - default_eps[i]) <= 1.0e-14) {
            return (uint64_t)i * 1000003ULL;
        }
    }
    return splitmix64((uint64_t)(epsilon * 1.0e12)) % 1000000007ULL;
}

static int make_host_matrix(double **matrix_out, int n, double epsilon,
                            uint64_t seed)
{
    size_t elements = (size_t)n * (size_t)n;
    double *matrix = (double *)malloc(elements * sizeof(double));

    if (matrix == NULL) {
        return -1;
    }

    for (int col = 0; col < n; ++col) {
        for (int row = 0; row <= col; ++row) {
            double value;

            if (row == col) {
                value = (double)(row + 1) +
                        epsilon * uniform01(seed, n, row, col);
            } else {
                value =
                    0.5 * epsilon *
                    (uniform01(seed, n, row, col) +
                     uniform01(seed, n, col, row));
            }

            matrix[row + col * n] = value;
            matrix[col + row * n] = value;
        }
    }

    *matrix_out = matrix;
    return 0;
}

static int copy_matrix_to_device(double **d_matrix_out, const double *h_matrix,
                                 int n, char *error, size_t error_size)
{
    size_t bytes = (size_t)n * (size_t)n * sizeof(double);
    double *d_matrix = NULL;
    cudaError_t cuda_status = cudaMalloc((void **)&d_matrix, bytes);

    if (cuda_status != cudaSuccess) {
        snprintf(error, error_size, "cudaMalloc matrix failed: %s",
                 cudaGetErrorString(cuda_status));
        return -1;
    }

    cuda_status =
        cudaMemcpy(d_matrix, h_matrix, bytes, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        snprintf(error, error_size, "cudaMemcpy matrix failed: %s",
                 cudaGetErrorString(cuda_status));
        cudaFree(d_matrix);
        return -1;
    }

    *d_matrix_out = d_matrix;
    return 0;
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

__global__ static void extract_diagonal_kernel(const double *matrix,
                                               double *diagonal, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        diagonal[idx] = matrix[idx + idx * n];
    }
}

__global__ static void set_identity_half_kernel(__half *x, int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;

    if (idx < total) {
        int row = idx % n;
        int col = idx / n;

        x[idx] = __float2half_rn(row == col ? 1.0f : 0.0f);
    }
}

__global__ static void build_g_half_from_double_diag_kernel(
    const double *diagonal, __half *g, int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;

    if (idx < total) {
        int row = idx % n;
        int col = idx / n;

        if (row == col) {
            g[idx] = __float2half_rn(0.0f);
        } else {
            g[idx] =
                __float2half_rn((float)(1.0 / (diagonal[col] - diagonal[row])));
        }
    }
}

__global__ static void column_diagonal_after_e_half_kernel(
    const __half *y, __half *column_diagonal, int n, int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < k) {
        column_diagonal[col] = y[col + col * n];
    }
}

__global__ static void ipt_update_all_half_kernel(
    __half *y, const __half *x, const __half *g,
    const __half *column_diagonal, int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;

    if (idx < total) {
        int row = idx % n;
        int col = idx / n;

        if (row == col) {
            y[idx] = __float2half_rn(1.0f);
        } else {
            y[idx] = __hmul(__hsub(y[idx], __hmul(x[idx], column_diagonal[col])),
                            g[idx]);
        }
    }
}

__device__ static double half_bits_to_double(unsigned short bits)
{
    int sign = (bits & 0x8000u) ? -1 : 1;
    int exponent = (bits >> 10) & 0x1f;
    int mantissa = bits & 0x03ff;

    if (exponent == 0) {
        if (mantissa == 0) {
            return sign < 0 ? -0.0 : 0.0;
        }
        return (double)sign * ldexp((double)mantissa, -24);
    }
    if (exponent == 31) {
        return mantissa == 0 ? ((sign < 0) ? -INFINITY : INFINITY) : NAN;
    }
    return (double)sign *
           ldexp(1.0 + (double)mantissa * (1.0 / 1024.0), exponent - 15);
}

__global__ static void half_delta_and_current_double_kernel(
    double *delta, double *current, const __half *next, const __half *prev,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        const unsigned short *next_bits =
            reinterpret_cast<const unsigned short *>(next);
        const unsigned short *prev_bits =
            reinterpret_cast<const unsigned short *>(prev);
        double next_value = half_bits_to_double(next_bits[idx]);
        double prev_value = half_bits_to_double(prev_bits[idx]);

        delta[idx] = next_value - prev_value;
        current[idx] = prev_value;
    }
}

__global__ static void half_to_double_bits_kernel(const __half *input,
                                                  double *output, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        const unsigned short *bits =
            reinterpret_cast<const unsigned short *>(input);

        output[idx] = half_bits_to_double(bits[idx]);
    }
}

__global__ static void count_bad_half_bits_kernel(const __half *x, int total,
                                                  int *bad_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        const unsigned short *bits =
            reinterpret_cast<const unsigned short *>(x);
        unsigned short value = bits[idx];

        if ((value & 0x7c00u) == 0x7c00u) {
            atomicAdd(bad_count, 1);
        }
    }
}

__global__ static void build_g_double_kernel(const double *diagonal, double *g,
                                             int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;

    if (idx < total) {
        int row = idx % n;
        int col = idx / n;

        g[idx] = (row == col) ? 0.0
                              : 1.0 / (diagonal[col] - diagonal[row]);
    }
}

__global__ static void column_diagonal_after_d_double_kernel(
    const double *y, const double *x, const double *diagonal,
    double *column_diagonal, int n, int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < k) {
        int diag_idx = col + col * n;

        column_diagonal[col] = y[diag_idx] - diagonal[col] * x[diag_idx];
    }
}

__global__ static void ipt_update_double_kernel(
    double *y, const double *x, const double *diagonal, const double *g,
    const double *column_diagonal, int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;

    if (idx < total) {
        int row = idx % n;
        int col = idx / n;

        if (row == col) {
            y[idx] = 1.0;
        } else {
            y[idx] =
                (y[idx] - diagonal[row] * x[idx] -
                 x[idx] * column_diagonal[col]) *
                g[idx];
        }
    }
}

__global__ static void fixed_point_delta_double_kernel(double *delta,
                                                       const double *next,
                                                       const double *current,
                                                       int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        delta[idx] = next[idx] - current[idx];
    }
}

__global__ static void gather_values_double_kernel(const double *mx,
                                                   double *values, int n,
                                                   int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < k) {
        values[col] = mx[col + col * n];
    }
}

static void reset_result(TensorHalfResult *result, const char *status)
{
    result->initial_time_sec = NAN;
    result->initial_residual_check_time_sec = 0.0;
    result->fp64_time_sec = NAN;
    result->total_time_sec = NAN;
    result->initial_fixed_point_residual = NAN;
    result->fp64_fixed_point_residual = NAN;
    result->initial_iterations = 0;
    result->fp64_iterations = 0;
    result->bad_count = 0;
    snprintf(result->status, sizeof(result->status), "%s", status);
    snprintf(result->initial_status, sizeof(result->initial_status), "%s",
             status);
    snprintf(result->fp64_status, sizeof(result->fp64_status), "not_run");
    result->error[0] = '\0';
}

static void fail_result(TensorHalfResult *result, const char *message)
{
    snprintf(result->status, sizeof(result->status), "failed");
    if (strcmp(result->initial_status, "ok") == 0) {
        snprintf(result->initial_status, sizeof(result->initial_status),
                 "failed");
    }
    snprintf(result->error, sizeof(result->error), "%s", message);
}

static TensorHalfResult run_tensor_half_initial(cublasHandle_t handle,
                                                const double *d_matrix, int n,
                                                double tol, int maxiter,
                                                double **d_initial_out)
{
    TensorHalfResult result;
    size_t elements = (size_t)n * (size_t)n;
    __half *d_matrix_half = NULL;
    __half *d_x_a = NULL;
    __half *d_x_b = NULL;
    __half *d_x = NULL;
    __half *d_y = NULL;
    __half *d_g = NULL;
    __half *d_column_diagonal = NULL;
    double *d_delta = NULL;
    double *d_current = NULL;
    double *d_diagonal = NULL;
    double *d_initial = NULL;
    int *d_bad_count = NULL;
    const int block_size = 256;
    const float one = 1.0f;
    const float zero = 0.0f;
    cudaError_t cuda_status;
    cublasStatus_t cublas_status;
    double residual_check_time = 0.0;

    reset_result(&result, "ok");
    if (d_initial_out != NULL) {
        *d_initial_out = NULL;
    }

    if (d_initial_out == NULL || elements > (size_t)INT_MAX) {
        fail_result(&result, "N*N exceeds int range used by this test");
        return result;
    }

    {
        double start = now_seconds();
        int matrix_blocks = ((int)elements + block_size - 1) / block_size;
        int diag_blocks = (n + block_size - 1) / block_size;

        cuda_status = cudaMalloc((void **)&d_matrix_half,
                                 elements * sizeof(__half));
        if (cuda_status != cudaSuccess) {
            fail_result(&result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }
        cuda_status =
            cudaMalloc((void **)&d_x_a, elements * sizeof(__half));
        if (cuda_status != cudaSuccess) {
            fail_result(&result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }
        cuda_status =
            cudaMalloc((void **)&d_x_b, elements * sizeof(__half));
        if (cuda_status != cudaSuccess) {
            fail_result(&result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }
        cuda_status = cudaMalloc((void **)&d_g, elements * sizeof(__half));
        if (cuda_status != cudaSuccess) {
            fail_result(&result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }
        cuda_status =
            cudaMalloc((void **)&d_column_diagonal, (size_t)n * sizeof(__half));
        if (cuda_status != cudaSuccess) {
            fail_result(&result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }
        cuda_status =
            cudaMalloc((void **)&d_delta, elements * sizeof(double));
        if (cuda_status != cudaSuccess) {
            fail_result(&result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }
        cuda_status =
            cudaMalloc((void **)&d_current, elements * sizeof(double));
        if (cuda_status != cudaSuccess) {
            fail_result(&result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }
        cuda_status =
            cudaMalloc((void **)&d_diagonal, (size_t)n * sizeof(double));
        if (cuda_status != cudaSuccess) {
            fail_result(&result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }
        cuda_status = cudaMalloc((void **)&d_bad_count, sizeof(int));
        if (cuda_status != cudaSuccess) {
            fail_result(&result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }

        double_to_half_offdiag_kernel<<<matrix_blocks, block_size>>>(
            d_matrix, d_matrix_half, n, (int)elements);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fail_result(&result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }

        extract_diagonal_kernel<<<diag_blocks, block_size>>>(d_matrix,
                                                             d_diagonal, n);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fail_result(&result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }

        set_identity_half_kernel<<<matrix_blocks, block_size>>>(d_x_a, n, n);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fail_result(&result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }

        build_g_half_from_double_diag_kernel<<<matrix_blocks, block_size>>>(
            d_diagonal, d_g, n, n);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fail_result(&result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }

        d_x = d_x_a;
        d_y = d_x_b;

        for (int iter = 0; iter < maxiter; ++iter) {
            __half *tmp = NULL;
            double delta_norm = 0.0;
            double current_norm = 0.0;
            int bad_count = 0;

            cublas_status = cublasGemmEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one,
                d_matrix_half, CUDA_R_16F, n, d_x, CUDA_R_16F, n, &zero, d_y,
                CUDA_R_16F, n, CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                fail_result(&result, cublas_status_name_local(cublas_status));
                goto timed_done;
            }

            column_diagonal_after_e_half_kernel<<<diag_blocks, block_size>>>(
                d_y, d_column_diagonal, n, n);
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) {
                fail_result(&result, cudaGetErrorString(cuda_status));
                goto timed_done;
            }

            ipt_update_all_half_kernel<<<matrix_blocks, block_size>>>(
                d_y, d_x, d_g, d_column_diagonal, n, n);
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) {
                fail_result(&result, cudaGetErrorString(cuda_status));
                goto timed_done;
            }

            cudaMemset(d_bad_count, 0, sizeof(int));
            count_bad_half_bits_kernel<<<matrix_blocks, block_size>>>(
                d_y, (int)elements, d_bad_count);
            cudaMemcpy(&bad_count, d_bad_count, sizeof(int),
                       cudaMemcpyDeviceToHost);
            result.bad_count = bad_count;
            if (bad_count > 0) {
                snprintf(result.status, sizeof(result.status), "bad_values");
                snprintf(result.initial_status, sizeof(result.initial_status),
                         "bad_values");
                snprintf(result.error, sizeof(result.error),
                         "found %d NaN/Inf half entries", bad_count);
                result.initial_iterations = iter + 1;
                goto timed_done;
            }

            {
                double residual_start = now_seconds();

                half_delta_and_current_double_kernel<<<matrix_blocks,
                                                       block_size>>>(
                    d_delta, d_current, d_y, d_x, (int)elements);
                cuda_status = cudaGetLastError();
                if (cuda_status != cudaSuccess) {
                    fail_result(&result, cudaGetErrorString(cuda_status));
                    goto timed_done;
                }

                cublas_status =
                    cublasDnrm2(handle, (int)elements, d_delta, 1,
                                &delta_norm);
                if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                    fail_result(&result,
                                cublas_status_name_local(cublas_status));
                    goto timed_done;
                }

                cublas_status =
                    cublasDnrm2(handle, (int)elements, d_current, 1,
                                &current_norm);
                if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                    fail_result(&result,
                                cublas_status_name_local(cublas_status));
                    goto timed_done;
                }
                residual_check_time += now_seconds() - residual_start;
            }

            result.initial_iterations = iter + 1;
            if (current_norm <= 0.0 || !isfinite(current_norm) ||
                !isfinite(delta_norm)) {
                snprintf(result.status, sizeof(result.status), "bad_norm");
                snprintf(result.initial_status, sizeof(result.initial_status),
                         "bad_norm");
                snprintf(result.error, sizeof(result.error),
                         "delta_norm=%g current_norm=%g", delta_norm,
                         current_norm);
                goto timed_done;
            }

            result.initial_fixed_point_residual = delta_norm / current_norm;
            printf("  tensor-half iter %d: delta_norm=%g current_norm=%g "
                   "fixed_point_residual=%.9g bad_count=%d\n",
                   iter + 1, delta_norm, current_norm,
                   result.initial_fixed_point_residual, bad_count);

            tmp = d_x;
            d_x = d_y;
            d_y = tmp;

            if (result.initial_fixed_point_residual <= tol) {
                break;
            }
        }

        cuda_status =
            cudaMalloc((void **)&d_initial, elements * sizeof(double));
        if (cuda_status != cudaSuccess) {
            fail_result(&result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }

        half_to_double_bits_kernel<<<matrix_blocks, block_size>>>(
            d_x, d_initial, (int)elements);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fail_result(&result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }

        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            fail_result(&result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }

        *d_initial_out = d_initial;
        d_initial = NULL;

    timed_done:
        result.initial_residual_check_time_sec = residual_check_time;
        result.initial_time_sec = now_seconds() - start - residual_check_time;
        if (result.initial_time_sec < 0.0) {
            result.initial_time_sec = 0.0;
        }
    }

    cudaFree(d_initial);
    cudaFree(d_bad_count);
    cudaFree(d_diagonal);
    cudaFree(d_current);
    cudaFree(d_delta);
    cudaFree(d_column_diagonal);
    cudaFree(d_g);
    cudaFree(d_x_b);
    cudaFree(d_x_a);
    cudaFree(d_matrix_half);
    return result;
}

static void mark_fp64_failed(TensorHalfResult *result, const char *message)
{
    snprintf(result->status, sizeof(result->status), "failed");
    snprintf(result->fp64_status, sizeof(result->fp64_status), "failed");
    if (result->error[0] == '\0') {
        snprintf(result->error, sizeof(result->error), "%s", message);
    } else {
        size_t used = strlen(result->error);

        snprintf(result->error + used, sizeof(result->error) - used, " | %s",
                 message);
    }
}

static void run_fp64_refine(cublasHandle_t handle, const double *d_matrix,
                            const double *d_initial, int n, double tol,
                            int maxiter, TensorHalfResult *result)
{
    size_t elements = (size_t)n * (size_t)n;
    double *d_x_a = NULL;
    double *d_x_b = NULL;
    double *d_x = NULL;
    double *d_y = NULL;
    double *d_diagonal = NULL;
    double *d_g = NULL;
    double *d_column_diagonal = NULL;
    double *d_delta = NULL;
    double *d_values = NULL;
    const double one = 1.0;
    const double zero = 0.0;
    const int block_size = 256;
    cudaError_t cuda_status;
    cublasStatus_t cublas_status;

    snprintf(result->fp64_status, sizeof(result->fp64_status), "ok");

    if (elements > (size_t)INT_MAX) {
        mark_fp64_failed(result, "N*N exceeds int range used by this test");
        return;
    }

    {
        double start = now_seconds();
        int vector_blocks = ((int)elements + block_size - 1) / block_size;
        int diag_blocks = (n + block_size - 1) / block_size;

        cuda_status = cudaMalloc((void **)&d_x_a, elements * sizeof(double));
        if (cuda_status != cudaSuccess) {
            mark_fp64_failed(result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }
        cuda_status = cudaMalloc((void **)&d_x_b, elements * sizeof(double));
        if (cuda_status != cudaSuccess) {
            mark_fp64_failed(result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }
        cuda_status =
            cudaMalloc((void **)&d_diagonal, (size_t)n * sizeof(double));
        if (cuda_status != cudaSuccess) {
            mark_fp64_failed(result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }
        cuda_status = cudaMalloc((void **)&d_g, elements * sizeof(double));
        if (cuda_status != cudaSuccess) {
            mark_fp64_failed(result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }
        cuda_status =
            cudaMalloc((void **)&d_column_diagonal, (size_t)n * sizeof(double));
        if (cuda_status != cudaSuccess) {
            mark_fp64_failed(result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }
        cuda_status = cudaMalloc((void **)&d_delta, elements * sizeof(double));
        if (cuda_status != cudaSuccess) {
            mark_fp64_failed(result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }
        cuda_status = cudaMalloc((void **)&d_values, (size_t)n * sizeof(double));
        if (cuda_status != cudaSuccess) {
            mark_fp64_failed(result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }

        cuda_status = cudaMemcpy(d_x_a, d_initial, elements * sizeof(double),
                                 cudaMemcpyDeviceToDevice);
        if (cuda_status != cudaSuccess) {
            mark_fp64_failed(result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }

        extract_diagonal_kernel<<<diag_blocks, block_size>>>(d_matrix,
                                                             d_diagonal, n);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            mark_fp64_failed(result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }

        build_g_double_kernel<<<vector_blocks, block_size>>>(d_diagonal, d_g,
                                                             n, n);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            mark_fp64_failed(result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }

        d_x = d_x_a;
        d_y = d_x_b;

        for (int iter = 0; iter < maxiter; ++iter) {
            int col_blocks = (n + block_size - 1) / block_size;
            double delta_norm = 0.0;
            double current_norm = 0.0;
            double *tmp = NULL;

            cublas_status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n,
                                        n, n, &one, d_matrix, n, d_x, n,
                                        &zero, d_y, n);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                mark_fp64_failed(result,
                                 cublas_status_name_local(cublas_status));
                goto timed_done;
            }

            column_diagonal_after_d_double_kernel<<<col_blocks, block_size>>>(
                d_y, d_x, d_diagonal, d_column_diagonal, n, n);
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) {
                mark_fp64_failed(result, cudaGetErrorString(cuda_status));
                goto timed_done;
            }

            ipt_update_double_kernel<<<vector_blocks, block_size>>>(
                d_y, d_x, d_diagonal, d_g, d_column_diagonal, n, n);
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) {
                mark_fp64_failed(result, cudaGetErrorString(cuda_status));
                goto timed_done;
            }

            result->fp64_iterations = iter + 1;
            fixed_point_delta_double_kernel<<<vector_blocks, block_size>>>(
                d_delta, d_y, d_x, (int)elements);
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) {
                mark_fp64_failed(result, cudaGetErrorString(cuda_status));
                goto timed_done;
            }

            cublas_status =
                cublasDnrm2(handle, (int)elements, d_delta, 1, &delta_norm);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                mark_fp64_failed(result,
                                 cublas_status_name_local(cublas_status));
                goto timed_done;
            }
            cublas_status =
                cublasDnrm2(handle, (int)elements, d_x, 1, &current_norm);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                mark_fp64_failed(result,
                                 cublas_status_name_local(cublas_status));
                goto timed_done;
            }

            if (current_norm <= 0.0 || !isfinite(current_norm)) {
                mark_fp64_failed(result, "invalid FP64 current norm");
                goto timed_done;
            }

            result->fp64_fixed_point_residual = delta_norm / current_norm;
            printf("  fp64 refine iter %d: fixed_point_residual=%.9g\n",
                   iter + 1, result->fp64_fixed_point_residual);

            tmp = d_x;
            d_x = d_y;
            d_y = tmp;

            if (result->fp64_fixed_point_residual <= tol) {
                break;
            }
        }

        cublas_status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                    &one, d_matrix, n, d_x, n, &zero, d_y, n);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            mark_fp64_failed(result, cublas_status_name_local(cublas_status));
            goto timed_done;
        }

        gather_values_double_kernel<<<diag_blocks, block_size>>>(d_y, d_values,
                                                                 n, n);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            mark_fp64_failed(result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }

        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            mark_fp64_failed(result, cudaGetErrorString(cuda_status));
            goto timed_done;
        }

    timed_done:
        result->fp64_time_sec = now_seconds() - start;
    }

    cudaFree(d_values);
    cudaFree(d_delta);
    cudaFree(d_column_diagonal);
    cudaFree(d_g);
    cudaFree(d_diagonal);
    cudaFree(d_x_b);
    cudaFree(d_x_a);
}

static void append_csv(const char *path, int n, double epsilon, double tol,
                       int maxiter, double fp64_tol, int fp64_maxiter,
                       int warmup_n, const TensorHalfResult *result,
                       const char *gpu_name)
{
    FILE *fp = fopen(path, "w");

    if (fp == NULL) {
        fprintf(stderr, "could not write %s: %s\n", path, strerror(errno));
        return;
    }

    fprintf(fp,
            "N,epsilon,initial_tol,initial_maxiter,fp64_tol,fp64_maxiter,"
            "warmup_n,time_initial_sec,initial_residual_check_time_sec,"
            "time_fp64_refine_sec,total_time_sec,initial_fixed_point_residual,"
            "fp64_fixed_point_residual,initial_iterations,fp64_iterations,"
            "bad_count,status,"
            "initial_status,fp64_status,error,gpu_name\n");
    fprintf(fp,
            "%d,%.12g,%.17g,%d,%.17g,%d,%d,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%d,%d,%d,%s,%s,%s,%s,%s\n",
            n, epsilon, tol, maxiter, fp64_tol, fp64_maxiter, warmup_n,
            result->initial_time_sec,
            result->initial_residual_check_time_sec, result->fp64_time_sec,
            result->total_time_sec, result->initial_fixed_point_residual,
            result->fp64_fixed_point_residual, result->initial_iterations,
            result->fp64_iterations, result->bad_count, result->status,
            result->initial_status, result->fp64_status, result->error,
            gpu_name);
    fclose(fp);
}

static void epsilon_label(double epsilon, char *label, size_t size)
{
    char raw[64];

    snprintf(raw, sizeof(raw), "%.6f", epsilon);
    for (char *p = raw + strlen(raw) - 1; p > raw && *p == '0'; --p) {
        *p = '\0';
    }
    if (raw[strlen(raw) - 1] == '.') {
        raw[strlen(raw) - 1] = '\0';
    }
    snprintf(label, size, "%s", raw);
    for (char *p = label; *p != '\0'; ++p) {
        if (*p == '.') {
            *p = '_';
        }
    }
}

static TensorHalfResult run_case(cublasHandle_t handle, int n, double epsilon,
                                 double tol, int maxiter, double fp64_tol,
                                 int fp64_maxiter, uint64_t seed)
{
    double *h_matrix = NULL;
    double *d_matrix = NULL;
    double *d_initial = NULL;
    TensorHalfResult result;
    char error[ERROR_LEN] = "";
    double start = now_seconds();

    reset_result(&result, "ok");
    printf("Generating matrix N=%d epsilon=%.12g\n", n, epsilon);
    if (make_host_matrix(&h_matrix, n, epsilon, seed) != 0) {
        fail_result(&result, "host matrix allocation failed");
        return result;
    }
    printf("Host matrix generated in %.9g s\n", now_seconds() - start);

    if (copy_matrix_to_device(&d_matrix, h_matrix, n, error, sizeof(error)) !=
        0) {
        fail_result(&result, error);
        free(h_matrix);
        return result;
    }
    free(h_matrix);
    h_matrix = NULL;
    cudaDeviceSynchronize();

    result = run_tensor_half_initial(handle, d_matrix, n, tol, maxiter,
                                     &d_initial);
    if (strcmp(result.status, "ok") == 0 && d_initial != NULL) {
        run_fp64_refine(handle, d_matrix, d_initial, n, fp64_tol, fp64_maxiter,
                        &result);
    }
    result.total_time_sec = result.initial_time_sec + result.fp64_time_sec;
    cudaFree(d_initial);
    cudaFree(d_matrix);
    cudaDeviceSynchronize();
    return result;
}

int main(void)
{
    const char *root = env_or_default("IPT_C_ROOT", DEFAULT_ROOT);
    const char *results_dir = env_or_default("IPT_C_RESULTS_DIR", NULL);
    const char *csv_path = env_or_default("IPT_ALL_HALF_CSV", NULL);
    char default_results_dir[4096];
    char default_csv_path[4096];
    int n = getenv("IPT_ALL_HALF_N") ? atoi(getenv("IPT_ALL_HALF_N"))
                                     : DEFAULT_N;
    int maxiter = getenv("IPT_ALL_HALF_MAXITER")
                      ? atoi(getenv("IPT_ALL_HALF_MAXITER"))
                      : DEFAULT_MAXITER;
    int fp64_maxiter = getenv("IPT_FP64_MAXITER")
                           ? atoi(getenv("IPT_FP64_MAXITER"))
                           : DEFAULT_FP64_MAXITER;
    int warmup_n = getenv("IPT_ALL_HALF_WARMUP_N")
                       ? atoi(getenv("IPT_ALL_HALF_WARMUP_N"))
                       : DEFAULT_WARMUP_N;
    double epsilon = getenv("IPT_ALL_HALF_EPSILON")
                         ? atof(getenv("IPT_ALL_HALF_EPSILON"))
                         : DEFAULT_EPSILON;
    double tol = getenv("IPT_ALL_HALF_TOL") ? atof(getenv("IPT_ALL_HALF_TOL"))
                                            : DEFAULT_TOL;
    double fp64_tol =
        getenv("IPT_FP64_TOL") ? atof(getenv("IPT_FP64_TOL"))
                               : DEFAULT_FP64_TOL;
    uint64_t seed =
        getenv("IPT_SEED") ? strtoull(getenv("IPT_SEED"), NULL, 10)
                           : DEFAULT_SEED;
    cudaDeviceProp prop;
    char gpu_name[256] = "unknown_gpu";
    cublasHandle_t handle = NULL;
    cublasStatus_t cublas_status;
    TensorHalfResult warmup_result;
    TensorHalfResult result;

    if (results_dir == NULL) {
        snprintf(default_results_dir, sizeof(default_results_dir),
                 "%s/results/mixed_precision", root);
        results_dir = default_results_dir;
    }
    if (csv_path == NULL) {
        char eps_label[64];

        epsilon_label(epsilon, eps_label, sizeof(eps_label));
        snprintf(default_csv_path, sizeof(default_csv_path),
                 "%s/fp16_all_half_ipt_N%d_epsilon_%s.csv", results_dir, n,
                 eps_label);
        csv_path = default_csv_path;
    }

    ensure_directory(results_dir);

    cudaSetDevice(0);
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        snprintf(gpu_name, sizeof(gpu_name), "%s", prop.name);
    }

    cublas_status = cublasCreate(&handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed: %s\n",
                cublas_status_name_local(cublas_status));
        return 2;
    }
    (void)cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    printf("===== FP16 tensor-half IPT + FP64 refine test =====\n");
    printf("N=%d epsilon=%.12g initial_tol=%.17g initial_maxiter=%d "
           "fp64_tol=%.17g fp64_maxiter=%d warmup_n=%d\n",
           n, epsilon, tol, maxiter, fp64_tol, fp64_maxiter, warmup_n);
    printf("CSV=%s\n", csv_path);
    printf("GPU=%s\n", gpu_name);

    if (warmup_n > 0) {
        printf("===== warmup =====\n");
        warmup_result = run_case(handle, warmup_n, epsilon, tol, maxiter,
                                 fp64_tol, fp64_maxiter, seed + 17);
        printf("warmup status=%s initial_iterations=%d initial_fp=%.9g "
               "initial_residual_check_time=%.9g fp64_iterations=%d "
               "fp64_fp=%.9g error=%s\n",
               warmup_result.status, warmup_result.initial_iterations,
               warmup_result.initial_fixed_point_residual,
               warmup_result.initial_residual_check_time_sec,
               warmup_result.fp64_iterations,
               warmup_result.fp64_fixed_point_residual, warmup_result.error);
    }

    printf("===== main run =====\n");
    result = run_case(handle, n, epsilon, tol, maxiter, fp64_tol, fp64_maxiter,
                      seed + (uint64_t)n + epsilon_seed_offset(epsilon));
    printf("result status=%s initial_iterations=%d initial_fp=%.9g "
           "initial_residual_check_time=%.9g fp64_iterations=%d "
           "fp64_fp=%.9g initial_time=%.9g fp64_time=%.9g "
           "total_time=%.9g bad_count=%d error=%s\n",
           result.status, result.initial_iterations,
           result.initial_fixed_point_residual,
           result.initial_residual_check_time_sec, result.fp64_iterations,
           result.fp64_fixed_point_residual, result.initial_time_sec,
           result.fp64_time_sec, result.total_time_sec, result.bad_count,
           result.error);

    append_csv(csv_path, n, epsilon, tol, maxiter, fp64_tol, fp64_maxiter,
               warmup_n, &result, gpu_name);

    cublasDestroy(handle);
    return strcmp(result.status, "ok") == 0 ? 0 : 1;
}
