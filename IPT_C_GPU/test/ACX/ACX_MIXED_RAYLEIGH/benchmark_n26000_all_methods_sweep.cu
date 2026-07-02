#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include "../../../src/ACX/acx_rayleigh_mixed_precision_cuda.cu"
#include "../../../src/mixed_precision/ipt_rayleigh_mixed_cuda.cu"

#define DEFAULT_ROOT "/fs1/home/nudt_liujie/ftt/IPT_C_GPU"
#define DEFAULT_NS "26000"
#define DEFAULT_EPS_VALUES "0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.12"
#define DEFAULT_TF32_TOL 1.0e-6
#define DEFAULT_FP64_TOL 1.0e-12
#define DEFAULT_TF32_MAXITER 1000
#define DEFAULT_FP64_MAXITER 1000
#define DEFAULT_REPEATS 5
#define DEFAULT_WARMUP_N 128
#define DEFAULT_SEED 20260617ULL
#define MAX_VALUES 64
#define ERROR_LEN 512

typedef struct {
    const char *root;
    const char *results_dir;
    const char *logs_dir;
    int ns[MAX_VALUES];
    int n_count;
    double eps_values[MAX_VALUES];
    int eps_count;
    double tf32_tol;
    double fp64_tol;
    int tf32_maxiter;
    int fp64_maxiter;
    int repeats;
    int warmup_n;
    uint64_t seed;
    int append_results;
    int compute_residual;
    char result_csv[4096];
    char summary_txt[4096];
} Config;

typedef struct {
    double total_time_sec;
    double iteration_time_sec;
    double finalization_time_sec;
    double tf32_time_sec;
    double orthogonalize_time_sec;
    double rayleigh_transform_time_sec;
    double fp64_time_sec;
    double backtransform_time_sec;
    double absolute_residual;
    double max_apply_residual;
    double fixed_point_residual;
    double tf32_fixed_point_residual;
    double fp64_fixed_point_residual;
    int iterations;
    int tf32_iterations;
    int fp64_iterations;
    int f_calls_total;
    int tf32_f_calls;
    int fp64_f_calls;
    int converged;
    int tf32_converged;
    int fp64_converged;
    char status[32];
    char error[ERROR_LEN];
} MethodResult;

static double now_seconds(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void wall_time_string(char *buffer, size_t size)
{
    time_t t = time(NULL);
    struct tm tm_value;

    localtime_r(&t, &tm_value);
    strftime(buffer, size, "%Y-%m-%dT%H:%M:%S%z", &tm_value);
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
                fprintf(stderr, "could not create directory %s: %s\n", tmp,
                        strerror(errno));
                return -1;
            }
            *p = '/';
        }
    }

    if (mkdir(tmp, 0775) != 0 && errno != EEXIST) {
        fprintf(stderr, "could not create directory %s: %s\n", tmp,
                strerror(errno));
        return -1;
    }

    return 0;
}

static const char *env_or_default(const char *name, const char *fallback_name,
                                  const char *default_value)
{
    const char *value = getenv(name);

    if ((value == NULL || value[0] == '\0') && fallback_name != NULL) {
        value = getenv(fallback_name);
    }
    return (value == NULL || value[0] == '\0') ? default_value : value;
}

static int parse_bool_env(const char *name, int default_value)
{
    const char *value = getenv(name);

    if (value == NULL || value[0] == '\0') {
        return default_value;
    }
    return strcmp(value, "1") == 0 || strcasecmp(value, "true") == 0 ||
           strcasecmp(value, "yes") == 0 || strcasecmp(value, "y") == 0;
}

static void parse_int_list(const char *raw, int *values, int *count)
{
    char *copy = strdup(raw);
    char *token = NULL;

    if (copy == NULL) {
        fprintf(stderr, "strdup failed while parsing integer list\n");
        exit(2);
    }
    *count = 0;
    token = strtok(copy, ", \t");
    while (token != NULL && *count < MAX_VALUES) {
        int value = atoi(token);

        if (value > 0) {
            values[(*count)++] = value;
        }
        token = strtok(NULL, ", \t");
    }
    free(copy);

    if (*count == 0) {
        fprintf(stderr, "integer list did not contain valid values\n");
        exit(2);
    }
}

static void parse_double_list(const char *raw, double *values, int *count)
{
    char *copy = strdup(raw);
    char *token = NULL;

    if (copy == NULL) {
        fprintf(stderr, "strdup failed while parsing double list\n");
        exit(2);
    }
    *count = 0;
    token = strtok(copy, ", \t");
    while (token != NULL && *count < MAX_VALUES) {
        double value = atof(token);

        if (value > 0.0) {
            values[(*count)++] = value;
        }
        token = strtok(NULL, ", \t");
    }
    free(copy);

    if (*count == 0) {
        fprintf(stderr, "double list did not contain valid values\n");
        exit(2);
    }
}

static void load_config(Config *cfg)
{
    const char *result_csv = NULL;
    const char *summary = NULL;

    memset(cfg, 0, sizeof(*cfg));
    cfg->root = env_or_default("IPT_C_ROOT", NULL, DEFAULT_ROOT);
    cfg->results_dir = env_or_default("ACX_MIXED_RAYLEIGH_RESULTS_DIR",
                                      "ACX_RESULTS_DIR", NULL);
    cfg->logs_dir =
        env_or_default("ACX_MIXED_RAYLEIGH_LOG_DIR", "ACX_LOG_DIR", NULL);

    if (cfg->results_dir == NULL) {
        static char default_results[4096];
        snprintf(default_results, sizeof(default_results),
                 "%s/results/ACX/ACX_MIXED_RAYLEIGH", cfg->root);
        cfg->results_dir = default_results;
    }
    if (cfg->logs_dir == NULL) {
        static char default_logs[4096];
        snprintf(default_logs, sizeof(default_logs),
                 "%s/logs/ACX/ACX_MIXED_RAYLEIGH", cfg->root);
        cfg->logs_dir = default_logs;
    }

    parse_int_list(env_or_default("ACX_MIXED_RAYLEIGH_SWEEP_NS",
                                  "ACX_SWEEP_NS", DEFAULT_NS),
                   cfg->ns, &cfg->n_count);
    parse_double_list(env_or_default("ACX_MIXED_RAYLEIGH_EPSILON_VALUES",
                                     "ACX_EPSILON_VALUES",
                                     DEFAULT_EPS_VALUES),
                      cfg->eps_values, &cfg->eps_count);

    cfg->tf32_tol = getenv("ACX_TF32_TOL") ? atof(getenv("ACX_TF32_TOL"))
                                           : DEFAULT_TF32_TOL;
    cfg->fp64_tol = getenv("ACX_FP64_TOL") ? atof(getenv("ACX_FP64_TOL"))
                                           : DEFAULT_FP64_TOL;
    cfg->tf32_maxiter = getenv("ACX_TF32_MAXITER")
                            ? atoi(getenv("ACX_TF32_MAXITER"))
                            : DEFAULT_TF32_MAXITER;
    cfg->fp64_maxiter = getenv("ACX_FP64_MAXITER")
                            ? atoi(getenv("ACX_FP64_MAXITER"))
                            : DEFAULT_FP64_MAXITER;
    cfg->repeats = getenv("ACX_MIXED_RAYLEIGH_REPEATS")
                       ? atoi(getenv("ACX_MIXED_RAYLEIGH_REPEATS"))
                       : DEFAULT_REPEATS;
    cfg->warmup_n = getenv("ACX_MIXED_RAYLEIGH_WARMUP_N")
                        ? atoi(getenv("ACX_MIXED_RAYLEIGH_WARMUP_N"))
                        : DEFAULT_WARMUP_N;
    cfg->seed = getenv("ACX_MIXED_RAYLEIGH_SEED")
                    ? strtoull(getenv("ACX_MIXED_RAYLEIGH_SEED"), NULL, 10)
                    : DEFAULT_SEED;
    cfg->append_results =
        parse_bool_env("ACX_MIXED_RAYLEIGH_APPEND_RESULTS", 0);
    cfg->compute_residual =
        parse_bool_env("ACX_MIXED_RAYLEIGH_COMPUTE_RESIDUAL", 1);

    if (cfg->tf32_tol <= 0.0) {
        cfg->tf32_tol = DEFAULT_TF32_TOL;
    }
    if (cfg->fp64_tol <= 0.0) {
        cfg->fp64_tol = DEFAULT_FP64_TOL;
    }
    if (cfg->tf32_maxiter <= 0) {
        cfg->tf32_maxiter = DEFAULT_TF32_MAXITER;
    }
    if (cfg->fp64_maxiter <= 0) {
        cfg->fp64_maxiter = DEFAULT_FP64_MAXITER;
    }
    if (cfg->repeats <= 0) {
        cfg->repeats = DEFAULT_REPEATS;
    }
    if (cfg->warmup_n < 0) {
        cfg->warmup_n = DEFAULT_WARMUP_N;
    }

    result_csv = getenv("ACX_N26000_ALL_METHODS_CSV");
    if (result_csv == NULL || result_csv[0] == '\0') {
        result_csv = getenv("ACX_MIXED_RAYLEIGH_COMPREHENSIVE_CSV");
    }
    if (result_csv != NULL && result_csv[0] != '\0') {
        snprintf(cfg->result_csv, sizeof(cfg->result_csv), "%s", result_csv);
    } else {
        snprintf(cfg->result_csv, sizeof(cfg->result_csv),
                 "%s/n26000_all_methods_sweep.csv",
                 cfg->results_dir);
    }

    summary = getenv("ACX_N26000_ALL_METHODS_SUMMARY_TXT");
    if (summary == NULL || summary[0] == '\0') {
        summary = getenv("ACX_MIXED_RAYLEIGH_COMPREHENSIVE_SUMMARY_TXT");
    }
    if (summary != NULL && summary[0] != '\0') {
        snprintf(cfg->summary_txt, sizeof(cfg->summary_txt), "%s", summary);
    } else {
        snprintf(cfg->summary_txt, sizeof(cfg->summary_txt),
                 "%s/n26000_all_methods_sweep_summary.txt",
                 cfg->results_dir);
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
    union {
        double value;
        uint64_t bits;
    } key;

    key.value = epsilon;
    return splitmix64(key.bits) % 1000000007ULL;
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
            matrix[(size_t)row + (size_t)col * (size_t)n] = value;
            matrix[(size_t)col + (size_t)row * (size_t)n] = value;
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
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        snprintf(error, error_size, "matrix copy synchronize failed: %s",
                 cudaGetErrorString(cuda_status));
        cudaFree(d_matrix);
        return -1;
    }

    *d_matrix_out = d_matrix;
    return 0;
}

__global__ static void residual_subtract_vdiag_kernel(double *residual,
                                                      const double *vectors,
                                                      const double *values,
                                                      int n, int k)
{
    size_t total = (size_t)n * (size_t)k;
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + threadIdx.x;

    if (idx < total) {
        int col = (int)(idx / (size_t)n);
        residual[idx] -= vectors[idx] * values[col];
    }
}

static int residual_norm_gpu(cublasHandle_t handle, const double *d_matrix,
                             const double *d_vectors, const double *d_values,
                             int n, int k, double *residual_out,
                             char *error, size_t error_size)
{
    int status = 0;
    size_t elements = (size_t)n * (size_t)k;
    double *d_residual = NULL;
    const double one = 1.0;
    const double zero = 0.0;
    int block = 256;
    int grid = (int)((elements + (size_t)block - 1) / (size_t)block);
    cudaError_t cuda_status;
    cublasStatus_t cublas_status;

    if (elements > (size_t)INT_MAX) {
        snprintf(error, error_size, "residual vector too large for cuBLAS");
        return -1;
    }

    cuda_status =
        cudaMalloc((void **)&d_residual, elements * sizeof(double));
    if (cuda_status != cudaSuccess) {
        snprintf(error, error_size, "cudaMalloc residual failed: %s",
                 cudaGetErrorString(cuda_status));
        return -1;
    }

    cublas_status =
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &one, d_matrix,
                    n, d_vectors, n, &zero, d_residual, n);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        snprintf(error, error_size, "residual cublasDgemm failed");
        status = -1;
        goto cleanup;
    }

    residual_subtract_vdiag_kernel<<<grid, block>>>(
        d_residual, d_vectors, d_values, n, k);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        snprintf(error, error_size, "residual kernel failed: %s",
                 cudaGetErrorString(cuda_status));
        status = -1;
        goto cleanup;
    }

    cublas_status =
        cublasDnrm2(handle, (int)elements, d_residual, 1, residual_out);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        snprintf(error, error_size, "residual cublasDnrm2 failed");
        status = -1;
        goto cleanup;
    }

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        snprintf(error, error_size, "residual synchronize failed: %s",
                 cudaGetErrorString(cuda_status));
        status = -1;
    }

cleanup:
    cudaFree(d_residual);
    return status;
}

static const char *cusolver_status_name(cusolverStatus_t status)
{
    switch (status) {
    case CUSOLVER_STATUS_SUCCESS:
        return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
        return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
        return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
        return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_MAPPING_ERROR:
        return "CUSOLVER_STATUS_MAPPING_ERROR";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
        return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
        return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    default:
        return "CUSOLVER_STATUS_UNKNOWN";
    }
}

static void method_result_reset(MethodResult *result, const char *status)
{
    result->total_time_sec = NAN;
    result->iteration_time_sec = NAN;
    result->finalization_time_sec = NAN;
    result->tf32_time_sec = NAN;
    result->orthogonalize_time_sec = NAN;
    result->rayleigh_transform_time_sec = NAN;
    result->fp64_time_sec = NAN;
    result->backtransform_time_sec = NAN;
    result->absolute_residual = NAN;
    result->max_apply_residual = NAN;
    result->fixed_point_residual = NAN;
    result->tf32_fixed_point_residual = NAN;
    result->fp64_fixed_point_residual = NAN;
    result->iterations = 0;
    result->tf32_iterations = 0;
    result->fp64_iterations = 0;
    result->f_calls_total = 0;
    result->tf32_f_calls = 0;
    result->fp64_f_calls = 0;
    result->converged = 0;
    result->tf32_converged = 0;
    result->fp64_converged = 0;
    snprintf(result->status, sizeof(result->status), "%s", status);
    result->error[0] = '\0';
}

static double f_calls_per_iteration(const MethodResult *result)
{
    int iterations = result->iterations + result->tf32_iterations +
                     result->fp64_iterations;

    return iterations > 0 ? (double)result->f_calls_total / (double)iterations
                          : NAN;
}

static MethodResult run_cusolver_once(cusolverDnHandle_t solver_handle,
                                      cublasHandle_t blas_handle,
                                      const double *d_matrix, int n,
                                      int compute_residual)
{
    MethodResult result;
    double *d_a = NULL;
    double *d_w = NULL;
    double *d_work = NULL;
    int *d_info = NULL;
    int h_info = 0;
    int lwork = 0;
    cudaError_t cuda_status;
    cusolverStatus_t solver_status;

    method_result_reset(&result, "ok");

    cuda_status =
        cudaMalloc((void **)&d_a, (size_t)n * (size_t)n * sizeof(double));
    if (cuda_status != cudaSuccess) {
        snprintf(result.status, sizeof(result.status), "failed");
        snprintf(result.error, sizeof(result.error), "%s",
                 cudaGetErrorString(cuda_status));
        return result;
    }
    cuda_status = cudaMemcpy(d_a, d_matrix,
                             (size_t)n * (size_t)n * sizeof(double),
                             cudaMemcpyDeviceToDevice);
    if (cuda_status != cudaSuccess) {
        snprintf(result.status, sizeof(result.status), "failed");
        snprintf(result.error, sizeof(result.error), "%s",
                 cudaGetErrorString(cuda_status));
        goto cleanup;
    }
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        snprintf(result.status, sizeof(result.status), "failed");
        snprintf(result.error, sizeof(result.error), "%s",
                 cudaGetErrorString(cuda_status));
        goto cleanup;
    }

    {
        double start = now_seconds();

        cuda_status = cudaMalloc((void **)&d_w, (size_t)n * sizeof(double));
        if (cuda_status != cudaSuccess) {
            snprintf(result.status, sizeof(result.status), "failed");
            snprintf(result.error, sizeof(result.error), "%s",
                     cudaGetErrorString(cuda_status));
            goto timed_done;
        }
        cuda_status = cudaMalloc((void **)&d_info, sizeof(int));
        if (cuda_status != cudaSuccess) {
            snprintf(result.status, sizeof(result.status), "failed");
            snprintf(result.error, sizeof(result.error), "%s",
                     cudaGetErrorString(cuda_status));
            goto timed_done;
        }
        solver_status = cusolverDnDsyevd_bufferSize(
            solver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n,
            d_a, n, d_w, &lwork);
        if (solver_status != CUSOLVER_STATUS_SUCCESS) {
            snprintf(result.status, sizeof(result.status), "failed");
            snprintf(result.error, sizeof(result.error), "%s",
                     cusolver_status_name(solver_status));
            goto timed_done;
        }
        cuda_status =
            cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(double));
        if (cuda_status != cudaSuccess) {
            snprintf(result.status, sizeof(result.status), "failed");
            snprintf(result.error, sizeof(result.error), "%s",
                     cudaGetErrorString(cuda_status));
            goto timed_done;
        }
        solver_status = cusolverDnDsyevd(
            solver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n,
            d_a, n, d_w, d_work, lwork, d_info);
        if (solver_status != CUSOLVER_STATUS_SUCCESS) {
            snprintf(result.status, sizeof(result.status), "failed");
            snprintf(result.error, sizeof(result.error), "%s",
                     cusolver_status_name(solver_status));
            goto timed_done;
        }
        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            snprintf(result.status, sizeof(result.status), "failed");
            snprintf(result.error, sizeof(result.error), "%s",
                     cudaGetErrorString(cuda_status));
            goto timed_done;
        }

    timed_done:
        result.total_time_sec = now_seconds() - start;
    }

    if (strcmp(result.status, "ok") != 0) {
        goto cleanup;
    }

    cuda_status =
        cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        snprintf(result.status, sizeof(result.status), "failed");
        snprintf(result.error, sizeof(result.error), "%s",
                 cudaGetErrorString(cuda_status));
        goto cleanup;
    }
    if (h_info != 0) {
        snprintf(result.status, sizeof(result.status), "failed");
        snprintf(result.error, sizeof(result.error), "cusolver info=%d",
                 h_info);
        goto cleanup;
    }

    if (compute_residual &&
        residual_norm_gpu(blas_handle, d_matrix, d_a, d_w, n, n,
                          &result.absolute_residual, result.error,
                          sizeof(result.error)) != 0) {
        snprintf(result.status, sizeof(result.status), "failed");
    }

cleanup:
    cudaFree(d_a);
    cudaFree(d_w);
    cudaFree(d_work);
    cudaFree(d_info);
    return result;
}

static MethodResult run_original_ipt_once(cublasHandle_t handle,
                                          const double *d_matrix, int n,
                                          const Config *cfg,
                                          int compute_residual)
{
    MethodResult result;
    double *d_vectors = NULL;
    double *d_values = NULL;
    int iterations_done = 0;
    double fixed_point_residual = NAN;
    int ipt_status = IPT_CUDA_SUCCESS;
    cudaError_t cuda_status = cudaSuccess;
    double start = 0.0;

    method_result_reset(&result, "ok");

    start = now_seconds();
    ipt_status = ipt_cuda_device_tol(d_matrix, n, n, cfg->fp64_tol,
                                     cfg->fp64_maxiter, handle, &d_vectors,
                                     &d_values, &iterations_done,
                                     &fixed_point_residual);
    cuda_status = cudaDeviceSynchronize();
    result.total_time_sec = now_seconds() - start;
    result.iteration_time_sec = result.total_time_sec;
    result.finalization_time_sec = 0.0;

    if (ipt_status != IPT_CUDA_SUCCESS) {
        snprintf(result.status, sizeof(result.status), "failed");
        snprintf(result.error, sizeof(result.error), "%s",
                 ipt_cuda_status_string(ipt_status));
        goto cleanup;
    }
    if (cuda_status != cudaSuccess) {
        snprintf(result.status, sizeof(result.status), "failed");
        snprintf(result.error, sizeof(result.error), "%s",
                 cudaGetErrorString(cuda_status));
        goto cleanup;
    }

    result.iterations = iterations_done;
    result.f_calls_total = iterations_done;
    result.fixed_point_residual = fixed_point_residual;
    result.converged =
        isfinite(fixed_point_residual) && fixed_point_residual <= cfg->fp64_tol;

    if (compute_residual &&
        residual_norm_gpu(handle, d_matrix, d_vectors, d_values, n, n,
                          &result.absolute_residual, result.error,
                          sizeof(result.error)) != 0) {
        snprintf(result.status, sizeof(result.status), "failed");
    }

cleanup:
    ipt_cuda_free_device_result(d_vectors, d_values);
    return result;
}

static MethodResult run_ipt_mixed_once(cublasHandle_t handle,
                                       const double *d_matrix, int n,
                                       const Config *cfg, int compute_residual)
{
    MethodResult result;
    IPTMixedTF32CudaDeviceResult device_result;
    int status = IPT_CUDA_SUCCESS;

    method_result_reset(&result, "ok");
    ipt_mixed_tf32_reset_device_result(&device_result);

    status = ipt_mixed_precision_tf32_cuda_device_tol(
        d_matrix, n, n, cfg->tf32_tol, cfg->tf32_maxiter, cfg->fp64_tol,
        cfg->fp64_maxiter, handle, &device_result);
    if (status != IPT_CUDA_SUCCESS) {
        snprintf(result.status, sizeof(result.status), "failed");
        snprintf(result.error, sizeof(result.error), "%s",
                 ipt_cuda_status_string(status));
        goto cleanup;
    }

    result.total_time_sec = device_result.total_time_sec;
    result.tf32_time_sec = device_result.tf32_time_sec;
    result.fp64_time_sec = device_result.fp64_time_sec;
    result.tf32_fixed_point_residual =
        device_result.tf32_fixed_point_residual;
    result.fp64_fixed_point_residual =
        device_result.fp64_fixed_point_residual;
    result.fixed_point_residual = device_result.fp64_fixed_point_residual;
    result.tf32_iterations = device_result.tf32_iterations;
    result.fp64_iterations = device_result.fp64_iterations;
    result.iterations = result.tf32_iterations + result.fp64_iterations;
    result.tf32_f_calls = result.tf32_iterations;
    result.fp64_f_calls = result.fp64_iterations;
    result.f_calls_total = result.tf32_f_calls + result.fp64_f_calls;
    result.tf32_converged =
        isfinite(result.tf32_fixed_point_residual) &&
        result.tf32_fixed_point_residual <= cfg->tf32_tol;
    result.fp64_converged =
        isfinite(result.fp64_fixed_point_residual) &&
        result.fp64_fixed_point_residual <= cfg->fp64_tol;
    result.converged = result.fp64_converged;

    if (compute_residual &&
        residual_norm_gpu(handle, d_matrix, device_result.d_vectors,
                          device_result.d_values, n, n,
                          &result.absolute_residual, result.error,
                          sizeof(result.error)) != 0) {
        snprintf(result.status, sizeof(result.status), "failed");
    }

cleanup:
    ipt_mixed_precision_tf32_cuda_free_device_result(&device_result);
    return result;
}

static MethodResult run_ipt_rayleigh_once(cublasHandle_t handle,
                                          const double *d_matrix, int n,
                                          const Config *cfg,
                                          int compute_residual)
{
    MethodResult result;
    IPTRayleighMixedCudaDeviceResult device_result;
    int status = IPT_CUDA_SUCCESS;

    method_result_reset(&result, "ok");
    ipt_rayleigh_mixed_reset_device_result(&device_result);

    status = ipt_rayleigh_mixed_tf32_cuda_device_tol(
        d_matrix, n, n, cfg->tf32_tol, cfg->tf32_maxiter, cfg->fp64_tol,
        cfg->fp64_maxiter, handle, &device_result);
    if (status != IPT_CUDA_SUCCESS) {
        snprintf(result.status, sizeof(result.status), "failed");
        snprintf(result.error, sizeof(result.error), "%s",
                 ipt_cuda_status_string(status));
        goto cleanup;
    }

    result.total_time_sec = device_result.total_time_sec;
    result.tf32_time_sec = device_result.tf32_time_sec;
    result.orthogonalize_time_sec = device_result.orthogonalize_time_sec;
    result.rayleigh_transform_time_sec =
        device_result.rayleigh_transform_time_sec;
    result.fp64_time_sec = device_result.fp64_time_sec;
    result.backtransform_time_sec = device_result.backtransform_time_sec;
    result.tf32_fixed_point_residual =
        device_result.tf32_fixed_point_residual;
    result.fp64_fixed_point_residual =
        device_result.fp64_fixed_point_residual;
    result.fixed_point_residual = device_result.fp64_fixed_point_residual;
    result.tf32_iterations = device_result.tf32_iterations;
    result.fp64_iterations = device_result.fp64_iterations;
    result.iterations = result.tf32_iterations + result.fp64_iterations;
    result.tf32_f_calls = result.tf32_iterations;
    result.fp64_f_calls = result.fp64_iterations;
    result.f_calls_total = result.tf32_f_calls + result.fp64_f_calls;
    result.tf32_converged =
        isfinite(result.tf32_fixed_point_residual) &&
        result.tf32_fixed_point_residual <= cfg->tf32_tol;
    result.fp64_converged =
        isfinite(result.fp64_fixed_point_residual) &&
        result.fp64_fixed_point_residual <= cfg->fp64_tol;
    result.converged = result.fp64_converged;

    if (compute_residual &&
        residual_norm_gpu(handle, d_matrix, device_result.d_vectors,
                          device_result.d_values, n, n,
                          &result.absolute_residual, result.error,
                          sizeof(result.error)) != 0) {
        snprintf(result.status, sizeof(result.status), "failed");
    }

cleanup:
    ipt_rayleigh_mixed_cuda_free_device_result(&device_result);
    return result;
}

static MethodResult run_original_acx_once(cublasHandle_t handle,
                                          const double *d_matrix, int n,
                                          const Config *cfg,
                                          int compute_residual)
{
    MethodResult result;
    ACXCudaResult device_result;
    int status = ACX_CUDA_SUCCESS;

    method_result_reset(&result, "ok");
    acx_cuda_reset_result(&device_result);

    status = acx_cuda_device_tol(d_matrix, n, n, cfg->fp64_tol,
                                 cfg->fp64_maxiter, handle, &device_result);
    if (status != ACX_CUDA_SUCCESS) {
        snprintf(result.status, sizeof(result.status), "failed");
        snprintf(result.error, sizeof(result.error), "%s",
                 acx_cuda_status_string(status));
        goto cleanup;
    }

    result.total_time_sec = device_result.time_sec;
    result.iteration_time_sec = device_result.iteration_time_sec;
    result.finalization_time_sec = device_result.finalization_time_sec;
    result.absolute_residual = NAN;
    result.max_apply_residual = device_result.max_residual;
    result.fixed_point_residual = device_result.fixed_point_residual;
    result.iterations = device_result.iterations;
    result.f_calls_total = device_result.f_calls;
    result.converged = device_result.converged;

    if (compute_residual &&
        residual_norm_gpu(handle, d_matrix, device_result.d_vectors,
                          device_result.d_values, n, n,
                          &result.absolute_residual, result.error,
                          sizeof(result.error)) != 0) {
        snprintf(result.status, sizeof(result.status), "failed");
    }

cleanup:
    acx_cuda_free_result(&device_result);
    return result;
}

static MethodResult run_acx_mixed_once(cublasHandle_t handle,
                                       const double *d_matrix, int n,
                                       const Config *cfg, int compute_residual)
{
    MethodResult result;
    ACXMixedTF32CudaResult device_result;
    int status = ACX_CUDA_SUCCESS;

    method_result_reset(&result, "ok");
    acx_mixed_precision_tf32_cuda_reset_result(&device_result);

    status = acx_mixed_precision_tf32_cuda_device_tol(
        d_matrix, n, n, cfg->tf32_tol, cfg->tf32_maxiter, cfg->fp64_tol,
        cfg->fp64_maxiter, handle, &device_result);
    if (status != ACX_CUDA_SUCCESS) {
        snprintf(result.status, sizeof(result.status), "failed");
        snprintf(result.error, sizeof(result.error), "%s",
                 acx_cuda_status_string(status));
        goto cleanup;
    }

    result.total_time_sec = device_result.total_time_sec;
    result.tf32_time_sec = device_result.tf32_time_sec;
    result.fp64_time_sec = device_result.fp64_time_sec;
    result.finalization_time_sec = device_result.finalization_time_sec;
    result.max_apply_residual = device_result.fp64_max_residual;
    result.tf32_fixed_point_residual =
        device_result.tf32_fixed_point_residual;
    result.fp64_fixed_point_residual =
        device_result.fp64_fixed_point_residual;
    result.tf32_iterations = device_result.tf32_iterations;
    result.fp64_iterations = device_result.fp64_iterations;
    result.tf32_f_calls = device_result.tf32_f_calls;
    result.fp64_f_calls = device_result.fp64_f_calls;
    result.f_calls_total = result.tf32_f_calls + result.fp64_f_calls;
    result.tf32_converged = device_result.tf32_converged;
    result.fp64_converged = device_result.fp64_converged;
    result.converged = device_result.fp64_converged;
    result.fixed_point_residual = device_result.fp64_fixed_point_residual;

    if (compute_residual &&
        residual_norm_gpu(handle, d_matrix, device_result.d_vectors,
                          device_result.d_values, n, n,
                          &result.absolute_residual, result.error,
                          sizeof(result.error)) != 0) {
        snprintf(result.status, sizeof(result.status), "failed");
    }

cleanup:
    acx_mixed_precision_tf32_cuda_free_result(&device_result);
    return result;
}

static MethodResult run_acx_rayleigh_once(cublasHandle_t handle,
                                          const double *d_matrix, int n,
                                          const Config *cfg,
                                          int compute_residual)
{
    MethodResult result;
    ACXRayleighMixedTF32CudaResult device_result;
    int status = ACX_CUDA_SUCCESS;

    method_result_reset(&result, "ok");
    acx_rayleigh_mixed_tf32_cuda_reset_result(&device_result);

    status = acx_rayleigh_mixed_tf32_cuda_device_tol(
        d_matrix, n, n, cfg->tf32_tol, cfg->tf32_maxiter, cfg->fp64_tol,
        cfg->fp64_maxiter, handle, &device_result);
    if (status != ACX_CUDA_SUCCESS) {
        snprintf(result.status, sizeof(result.status), "failed");
        snprintf(result.error, sizeof(result.error), "%s",
                 acx_cuda_status_string(status));
        goto cleanup;
    }

    result.total_time_sec = device_result.total_time_sec;
    result.tf32_time_sec = device_result.tf32_time_sec;
    result.orthogonalize_time_sec = device_result.orthogonalize_time_sec;
    result.rayleigh_transform_time_sec =
        device_result.rayleigh_transform_time_sec;
    result.fp64_time_sec = device_result.fp64_time_sec;
    result.finalization_time_sec = device_result.finalization_time_sec;
    result.backtransform_time_sec = device_result.backtransform_time_sec;
    result.max_apply_residual = device_result.fp64_max_residual;
    result.tf32_fixed_point_residual =
        device_result.tf32_fixed_point_residual;
    result.fp64_fixed_point_residual =
        device_result.fp64_fixed_point_residual;
    result.tf32_iterations = device_result.tf32_iterations;
    result.fp64_iterations = device_result.fp64_iterations;
    result.tf32_f_calls = device_result.tf32_f_calls;
    result.fp64_f_calls = device_result.fp64_f_calls;
    result.f_calls_total = result.tf32_f_calls + result.fp64_f_calls;
    result.tf32_converged = device_result.tf32_converged;
    result.fp64_converged = device_result.fp64_converged;
    result.converged = device_result.fp64_converged;
    result.fixed_point_residual = device_result.fp64_fixed_point_residual;

    if (compute_residual &&
        residual_norm_gpu(handle, d_matrix, device_result.d_vectors,
                          device_result.d_values, n, n,
                          &result.absolute_residual, result.error,
                          sizeof(result.error)) != 0) {
        snprintf(result.status, sizeof(result.status), "failed");
    }

cleanup:
    acx_rayleigh_mixed_tf32_cuda_free_result(&device_result);
    return result;
}

typedef MethodResult (*RunOnceFn)(cublasHandle_t, const double *, int,
                                  const Config *, int);

static MethodResult run_average_device(const char *name, RunOnceFn run_once,
                                       cublasHandle_t handle,
                                       const double *d_matrix, int n,
                                       const Config *cfg)
{
    MethodResult average;
    double total_time = 0.0;
    double iteration_time = 0.0;
    double finalization_time = 0.0;
    double tf32_time = 0.0;
    double orthogonalize_time = 0.0;
    double rayleigh_transform_time = 0.0;
    double fp64_time = 0.0;
    double backtransform_time = 0.0;
    double fixed_point = 0.0;
    double tf32_fixed = 0.0;
    double fp64_fixed = 0.0;

    method_result_reset(&average, "ok");

    printf("%s warmup for N=%d\n", name, n);
    {
        MethodResult warm = run_once(handle, d_matrix, n, cfg, 0);
        if (strcmp(warm.status, "ok") != 0) {
            return warm;
        }
    }

    for (int rep = 1; rep <= cfg->repeats; ++rep) {
        int compute_residual = cfg->compute_residual && rep == cfg->repeats;
        MethodResult one = run_once(handle, d_matrix, n, cfg, compute_residual);

        printf("  %s run %d/%d: total=%.8g s, residual=%.8g, "
               "fixed=%.8g, tf32_fixed=%.8g, fp64_fixed=%.8g, "
               "iterations=%d/%d/%d, f_calls=%d/%d/%d, status=%s",
               name, rep, cfg->repeats, one.total_time_sec,
               one.absolute_residual, one.fixed_point_residual,
               one.tf32_fixed_point_residual, one.fp64_fixed_point_residual,
               one.iterations, one.tf32_iterations, one.fp64_iterations,
               one.f_calls_total, one.tf32_f_calls, one.fp64_f_calls,
               one.status);
        if (one.error[0] != '\0') {
            printf(", error=%s", one.error);
        }
        printf("\n");

        if (strcmp(one.status, "ok") != 0) {
            return one;
        }

        total_time += one.total_time_sec;
        iteration_time += one.iteration_time_sec;
        finalization_time += one.finalization_time_sec;
        tf32_time += one.tf32_time_sec;
        orthogonalize_time += one.orthogonalize_time_sec;
        rayleigh_transform_time += one.rayleigh_transform_time_sec;
        fp64_time += one.fp64_time_sec;
        backtransform_time += one.backtransform_time_sec;
        fixed_point += one.fixed_point_residual;
        tf32_fixed += one.tf32_fixed_point_residual;
        fp64_fixed += one.fp64_fixed_point_residual;

        average.absolute_residual = one.absolute_residual;
        average.max_apply_residual = one.max_apply_residual;
        average.iterations = one.iterations;
        average.tf32_iterations = one.tf32_iterations;
        average.fp64_iterations = one.fp64_iterations;
        average.f_calls_total = one.f_calls_total;
        average.tf32_f_calls = one.tf32_f_calls;
        average.fp64_f_calls = one.fp64_f_calls;
        average.converged = one.converged;
        average.tf32_converged = one.tf32_converged;
        average.fp64_converged = one.fp64_converged;
    }

    average.total_time_sec = total_time / (double)cfg->repeats;
    average.iteration_time_sec = iteration_time / (double)cfg->repeats;
    average.finalization_time_sec = finalization_time / (double)cfg->repeats;
    average.tf32_time_sec = tf32_time / (double)cfg->repeats;
    average.orthogonalize_time_sec =
        orthogonalize_time / (double)cfg->repeats;
    average.rayleigh_transform_time_sec =
        rayleigh_transform_time / (double)cfg->repeats;
    average.fp64_time_sec = fp64_time / (double)cfg->repeats;
    average.backtransform_time_sec =
        backtransform_time / (double)cfg->repeats;
    average.fixed_point_residual = fixed_point / (double)cfg->repeats;
    average.tf32_fixed_point_residual = tf32_fixed / (double)cfg->repeats;
    average.fp64_fixed_point_residual = fp64_fixed / (double)cfg->repeats;
    return average;
}

static MethodResult run_cusolver_average(cusolverDnHandle_t solver_handle,
                                         cublasHandle_t blas_handle,
                                         const double *d_matrix, int n,
                                         const Config *cfg)
{
    MethodResult average;
    double total_time = 0.0;

    method_result_reset(&average, "ok");

    printf("cusolver warmup for N=%d\n", n);
    {
        MethodResult warm =
            run_cusolver_once(solver_handle, blas_handle, d_matrix, n, 0);
        if (strcmp(warm.status, "ok") != 0) {
            return warm;
        }
    }

    for (int rep = 1; rep <= cfg->repeats; ++rep) {
        int compute_residual = cfg->compute_residual && rep == cfg->repeats;
        MethodResult one = run_cusolver_once(solver_handle, blas_handle,
                                             d_matrix, n, compute_residual);

        printf("  cusolver run %d/%d: total=%.8g s, residual=%.8g, "
               "status=%s",
               rep, cfg->repeats, one.total_time_sec, one.absolute_residual,
               one.status);
        if (one.error[0] != '\0') {
            printf(", error=%s", one.error);
        }
        printf("\n");

        if (strcmp(one.status, "ok") != 0) {
            return one;
        }

        total_time += one.total_time_sec;
        average.absolute_residual = one.absolute_residual;
    }

    average.total_time_sec = total_time / (double)cfg->repeats;
    return average;
}

static void csv_escape(const char *src, char *dst, size_t dst_size)
{
    size_t out = 0;

    if (dst_size == 0) {
        return;
    }
    for (size_t i = 0; src != NULL && src[i] != '\0' && out + 1 < dst_size;
         ++i) {
        char ch = src[i];

        if (ch == ',' || ch == '\n' || ch == '\r') {
            ch = ' ';
        }
        dst[out++] = ch;
    }
    dst[out] = '\0';
}

static void write_csv_header(const char *csv_path)
{
    FILE *fp = fopen(csv_path, "w");

    if (fp == NULL) {
        fprintf(stderr, "could not open %s for writing: %s\n", csv_path,
                strerror(errno));
        return;
    }

    fprintf(fp,
            "N,epsilon,method,seed,repeats,warmup_n,tf32_tol,fp64_tol,"
            "tf32_maxiter,fp64_maxiter,time_total_sec,time_iteration_sec,"
            "time_finalization_sec,time_tf32_initial_sec,"
            "time_orthogonalize_sec,time_rayleigh_transform_sec,"
            "time_fp64_refine_sec,time_backtransform_sec,absolute_residual,"
            "max_apply_residual,fixed_point_residual,"
            "tf32_fixed_point_residual,fp64_fixed_point_residual,iterations,"
            "tf32_iterations,fp64_iterations,f_calls_total,tf32_f_calls,"
            "fp64_f_calls,f_calls_per_iteration,converged,tf32_converged,"
            "fp64_converged,gpu_name,status,error\n");
    fclose(fp);
}

static void append_csv_row(const Config *cfg, int n, double epsilon,
                           const char *method, const MethodResult *result,
                           const char *gpu_name)
{
    FILE *fp = fopen(cfg->result_csv, "a");
    char method_clean[128];
    char gpu_clean[256];
    char status_clean[64];
    char error_clean[ERROR_LEN];

    if (fp == NULL) {
        fprintf(stderr, "could not open %s for append: %s\n", cfg->result_csv,
                strerror(errno));
        return;
    }

    csv_escape(method, method_clean, sizeof(method_clean));
    csv_escape(gpu_name, gpu_clean, sizeof(gpu_clean));
    csv_escape(result->status, status_clean, sizeof(status_clean));
    csv_escape(result->error, error_clean, sizeof(error_clean));

    fprintf(fp,
            "%d,%.17g,%s,%llu,%d,%d,%.17g,%.17g,%d,%d,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%d,%d,%d,%d,%d,%d,%.17g,%d,%d,%d,%s,%s,%s\n",
            n, epsilon, method_clean, (unsigned long long)cfg->seed,
            cfg->repeats, cfg->warmup_n, cfg->tf32_tol, cfg->fp64_tol,
            cfg->tf32_maxiter, cfg->fp64_maxiter, result->total_time_sec,
            result->iteration_time_sec, result->finalization_time_sec,
            result->tf32_time_sec, result->orthogonalize_time_sec,
            result->rayleigh_transform_time_sec, result->fp64_time_sec,
            result->backtransform_time_sec, result->absolute_residual,
            result->max_apply_residual, result->fixed_point_residual,
            result->tf32_fixed_point_residual,
            result->fp64_fixed_point_residual, result->iterations,
            result->tf32_iterations, result->fp64_iterations,
            result->f_calls_total, result->tf32_f_calls,
            result->fp64_f_calls, f_calls_per_iteration(result),
            result->converged, result->tf32_converged,
            result->fp64_converged, gpu_clean, status_clean, error_clean);
    fclose(fp);
}

static void append_summary(const Config *cfg, const char *line)
{
    FILE *fp = fopen(cfg->summary_txt, "a");

    if (fp == NULL) {
        fprintf(stderr, "could not append summary %s: %s\n", cfg->summary_txt,
                strerror(errno));
        return;
    }
    fprintf(fp, "%s\n", line);
    fclose(fp);
}

static void show_gpu_memory(const char *label)
{
    size_t free_bytes = 0;
    size_t total_bytes = 0;

    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        printf("GPU memory %s: free %.3f GiB / total %.3f GiB\n", label,
               (double)free_bytes / 1073741824.0,
               (double)total_bytes / 1073741824.0);
    }
}

static void maybe_write_headers(const Config *cfg)
{
    FILE *probe = NULL;

    if (!cfg->append_results) {
        write_csv_header(cfg->result_csv);
        probe = fopen(cfg->summary_txt, "w");
        if (probe != NULL) {
            fclose(probe);
        }
        return;
    }

    probe = fopen(cfg->result_csv, "r");
    if (probe == NULL) {
        write_csv_header(cfg->result_csv);
    } else {
        fclose(probe);
    }
}

int main(void)
{
    Config cfg;
    cudaDeviceProp prop;
    char gpu_name[256] = "unknown";
    cublasHandle_t blas_handle = NULL;
    cusolverDnHandle_t solver_handle = NULL;
    int failures = 0;
    char time_buf[64];
    char summary_line[4096];

    load_config(&cfg);
    if (ensure_directory(cfg.results_dir) != 0 ||
        ensure_directory(cfg.logs_dir) != 0) {
        return 2;
    }

    if (cudaSetDevice(0) != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed\n");
        return 2;
    }
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        snprintf(gpu_name, sizeof(gpu_name), "%s", prop.name);
    }
    if (cublasCreate(&blas_handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed\n");
        return 2;
    }
    (void)cublasSetMathMode(blas_handle, CUBLAS_TENSOR_OP_MATH);
    if (cusolverDnCreate(&solver_handle) != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "cusolverDnCreate failed\n");
        cublasDestroy(blas_handle);
        return 2;
    }

    maybe_write_headers(&cfg);
    wall_time_string(time_buf, sizeof(time_buf));
    snprintf(summary_line, sizeof(summary_line),
             "started_at=%s methods=cusolver,ipt_fp64,ipt_tf32,"
             "ipt_rayleigh,acx_fp64,acx_tf32,acx_rayleigh repeats=%d "
             "tf32_tol=%.17g fp64_tol=%.17g gpu=%s",
             time_buf, cfg.repeats, cfg.tf32_tol, cfg.fp64_tol, gpu_name);
    append_summary(&cfg, summary_line);

    printf("===== N=26000 IPT/ACX all methods sweep =====\n");
    printf("root=%s\nresults=%s\nlogs=%s\ncsv=%s\nsummary=%s\n", cfg.root,
           cfg.results_dir, cfg.logs_dir, cfg.result_csv, cfg.summary_txt);
    printf("N values = ");
    for (int i = 0; i < cfg.n_count; ++i) {
        printf("%s%d", i == 0 ? "" : ", ", cfg.ns[i]);
    }
    printf("\nepsilon values = ");
    for (int i = 0; i < cfg.eps_count; ++i) {
        printf("%s%.12g", i == 0 ? "" : ", ", cfg.eps_values[i]);
    }
    printf("\ntf32_tol=%.17g fp64_tol=%.17g tf32_maxiter=%d "
           "fp64_maxiter=%d repeats=%d warmup_n=%d seed=%llu gpu=%s\n",
           cfg.tf32_tol, cfg.fp64_tol, cfg.tf32_maxiter, cfg.fp64_maxiter,
           cfg.repeats, cfg.warmup_n, (unsigned long long)cfg.seed,
           gpu_name);
    show_gpu_memory("at start");

    for (int eps_idx = 0; eps_idx < cfg.eps_count; ++eps_idx) {
        double epsilon = cfg.eps_values[eps_idx];

        for (int n_idx = 0; n_idx < cfg.n_count; ++n_idx) {
            int n = cfg.ns[n_idx];
            double *h_matrix = NULL;
            double *d_matrix = NULL;
            MethodResult cusolver_result;
            MethodResult ipt_fp64_result;
            MethodResult ipt_tf32_result;
            MethodResult ipt_rayleigh_result;
            MethodResult acx_fp64_result;
            MethodResult acx_tf32_result;
            MethodResult acx_rayleigh_result;
            char error[ERROR_LEN] = "";
            double matrix_start = 0.0;

            method_result_reset(&cusolver_result, "not_run");
            method_result_reset(&ipt_fp64_result, "not_run");
            method_result_reset(&ipt_tf32_result, "not_run");
            method_result_reset(&ipt_rayleigh_result, "not_run");
            method_result_reset(&acx_fp64_result, "not_run");
            method_result_reset(&acx_tf32_result, "not_run");
            method_result_reset(&acx_rayleigh_result, "not_run");

            wall_time_string(time_buf, sizeof(time_buf));
            snprintf(summary_line, sizeof(summary_line),
                     "epsilon=%.12g N=%d started_at=%s", epsilon, n,
                     time_buf);
            append_summary(&cfg, summary_line);

            printf("\n===== epsilon=%.12g N=%d =====\n", epsilon, n);
            show_gpu_memory("before matrix");
            matrix_start = now_seconds();
            if (make_host_matrix(&h_matrix, n, epsilon,
                                 cfg.seed + (uint64_t)n +
                                     epsilon_seed_offset(epsilon)) != 0) {
                snprintf(error, sizeof(error), "host matrix allocation failed");
                snprintf(cusolver_result.status, sizeof(cusolver_result.status),
                         "failed");
                snprintf(ipt_fp64_result.status,
                         sizeof(ipt_fp64_result.status), "failed");
                snprintf(ipt_tf32_result.status,
                         sizeof(ipt_tf32_result.status), "failed");
                snprintf(ipt_rayleigh_result.status,
                         sizeof(ipt_rayleigh_result.status), "failed");
                snprintf(acx_fp64_result.status,
                         sizeof(acx_fp64_result.status), "failed");
                snprintf(acx_tf32_result.status,
                         sizeof(acx_tf32_result.status), "failed");
                snprintf(acx_rayleigh_result.status,
                         sizeof(acx_rayleigh_result.status), "failed");
                snprintf(cusolver_result.error, sizeof(cusolver_result.error),
                         "%s", error);
                snprintf(ipt_fp64_result.error,
                         sizeof(ipt_fp64_result.error), "%s", error);
                snprintf(ipt_tf32_result.error,
                         sizeof(ipt_tf32_result.error), "%s", error);
                snprintf(ipt_rayleigh_result.error,
                         sizeof(ipt_rayleigh_result.error), "%s", error);
                snprintf(acx_fp64_result.error,
                         sizeof(acx_fp64_result.error), "%s", error);
                snprintf(acx_tf32_result.error,
                         sizeof(acx_tf32_result.error), "%s", error);
                snprintf(acx_rayleigh_result.error,
                         sizeof(acx_rayleigh_result.error), "%s", error);
                failures += 7;
                goto row_done;
            }
            printf("host matrix generated in %.8g s\n",
                   now_seconds() - matrix_start);

            if (copy_matrix_to_device(&d_matrix, h_matrix, n, error,
                                      sizeof(error)) != 0) {
                snprintf(cusolver_result.status, sizeof(cusolver_result.status),
                         "failed");
                snprintf(ipt_fp64_result.status,
                         sizeof(ipt_fp64_result.status), "failed");
                snprintf(ipt_tf32_result.status,
                         sizeof(ipt_tf32_result.status), "failed");
                snprintf(ipt_rayleigh_result.status,
                         sizeof(ipt_rayleigh_result.status), "failed");
                snprintf(acx_fp64_result.status,
                         sizeof(acx_fp64_result.status), "failed");
                snprintf(acx_tf32_result.status,
                         sizeof(acx_tf32_result.status), "failed");
                snprintf(acx_rayleigh_result.status,
                         sizeof(acx_rayleigh_result.status), "failed");
                snprintf(cusolver_result.error, sizeof(cusolver_result.error),
                         "%s", error);
                snprintf(ipt_fp64_result.error,
                         sizeof(ipt_fp64_result.error), "%s", error);
                snprintf(ipt_tf32_result.error,
                         sizeof(ipt_tf32_result.error), "%s", error);
                snprintf(ipt_rayleigh_result.error,
                         sizeof(ipt_rayleigh_result.error), "%s", error);
                snprintf(acx_fp64_result.error,
                         sizeof(acx_fp64_result.error), "%s", error);
                snprintf(acx_tf32_result.error,
                         sizeof(acx_tf32_result.error), "%s", error);
                snprintf(acx_rayleigh_result.error,
                         sizeof(acx_rayleigh_result.error), "%s", error);
                failures += 7;
                goto row_done;
            }
            show_gpu_memory("after matrix");

            cusolver_result = run_cusolver_average(
                solver_handle, blas_handle, d_matrix, n, &cfg);
            if (strcmp(cusolver_result.status, "ok") != 0) {
                failures++;
            }
            ipt_fp64_result = run_average_device("IPT FP64",
                                                 run_original_ipt_once,
                                                 blas_handle, d_matrix, n,
                                                 &cfg);
            if (strcmp(ipt_fp64_result.status, "ok") != 0) {
                failures++;
            }
            ipt_tf32_result = run_average_device("IPT mixed TF32->FP64",
                                                 run_ipt_mixed_once,
                                                 blas_handle, d_matrix, n,
                                                 &cfg);
            if (strcmp(ipt_tf32_result.status, "ok") != 0) {
                failures++;
            }
            ipt_rayleigh_result = run_average_device(
                "IPT mixed Rayleigh TF32->FP64", run_ipt_rayleigh_once,
                blas_handle, d_matrix, n, &cfg);
            if (strcmp(ipt_rayleigh_result.status, "ok") != 0) {
                failures++;
            }
            acx_fp64_result = run_average_device("ACX FP64",
                                                 run_original_acx_once,
                                                 blas_handle, d_matrix, n,
                                                 &cfg);
            if (strcmp(acx_fp64_result.status, "ok") != 0) {
                failures++;
            }
            acx_tf32_result = run_average_device("ACX mixed TF32->FP64",
                                                 run_acx_mixed_once,
                                                 blas_handle, d_matrix, n,
                                                 &cfg);
            if (strcmp(acx_tf32_result.status, "ok") != 0) {
                failures++;
            }
            acx_rayleigh_result = run_average_device(
                "ACX mixed Rayleigh TF32->FP64", run_acx_rayleigh_once,
                blas_handle, d_matrix, n, &cfg);
            if (strcmp(acx_rayleigh_result.status, "ok") != 0) {
                failures++;
            }

        row_done:
            append_csv_row(&cfg, n, epsilon, "cusolver", &cusolver_result,
                           gpu_name);
            append_csv_row(&cfg, n, epsilon, "ipt_fp64", &ipt_fp64_result,
                           gpu_name);
            append_csv_row(&cfg, n, epsilon, "ipt_tf32", &ipt_tf32_result,
                           gpu_name);
            append_csv_row(&cfg, n, epsilon, "ipt_rayleigh",
                           &ipt_rayleigh_result, gpu_name);
            append_csv_row(&cfg, n, epsilon, "acx_fp64", &acx_fp64_result,
                           gpu_name);
            append_csv_row(&cfg, n, epsilon, "acx_tf32", &acx_tf32_result,
                           gpu_name);
            append_csv_row(&cfg, n, epsilon, "acx_rayleigh",
                           &acx_rayleigh_result, gpu_name);

            wall_time_string(time_buf, sizeof(time_buf));
            snprintf(summary_line, sizeof(summary_line),
                     "epsilon=%.12g N=%d finished_at=%s "
                     "cusolver=%.17g ipt_fp64=%.17g ipt_tf32=%.17g "
                     "ipt_rayleigh=%.17g acx_fp64=%.17g acx_tf32=%.17g "
                     "acx_rayleigh=%.17g ipt_fp64_f_calls=%d "
                     "ipt_tf32_f_calls=%d ipt_rayleigh_f_calls=%d "
                     "acx_fp64_f_calls=%d acx_tf32_f_calls=%d "
                     "acx_rayleigh_f_calls=%d csv=%s",
                     epsilon, n, time_buf, cusolver_result.total_time_sec,
                     ipt_fp64_result.total_time_sec,
                     ipt_tf32_result.total_time_sec,
                     ipt_rayleigh_result.total_time_sec,
                     acx_fp64_result.total_time_sec,
                     acx_tf32_result.total_time_sec,
                     acx_rayleigh_result.total_time_sec,
                     ipt_fp64_result.f_calls_total,
                     ipt_tf32_result.f_calls_total,
                     ipt_rayleigh_result.f_calls_total,
                     acx_fp64_result.f_calls_total,
                     acx_tf32_result.f_calls_total,
                     acx_rayleigh_result.f_calls_total, cfg.result_csv);
            append_summary(&cfg, summary_line);

            cudaFree(d_matrix);
            free(h_matrix);
            cudaDeviceSynchronize();
            show_gpu_memory("after cleanup");
        }
    }

    wall_time_string(time_buf, sizeof(time_buf));
    snprintf(summary_line, sizeof(summary_line), "finished_at=%s failures=%d",
             time_buf, failures);
    append_summary(&cfg, summary_line);
    printf("\n===== done failures=%d =====\n", failures);
    cusolverDnDestroy(solver_handle);
    cublasDestroy(blas_handle);
    return failures == 0 ? 0 : 1;
}
