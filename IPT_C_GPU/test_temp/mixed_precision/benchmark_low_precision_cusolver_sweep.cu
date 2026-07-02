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

#include "../../src/mixed_precision/ipt_mixed_precision_cuda.cu"

#define DEFAULT_ROOT "/fs1/home/nudt_liujie/ftt/IPT_C_GPU"
#define DEFAULT_NS "4096,8192,16384,32768"
#define DEFAULT_EPS_VALUES "0.01,0.02,0.03,0.04,0.05,0.06"
#define DEFAULT_FP16_TOL 1.0e-3
#define DEFAULT_FP16_MAXITER 1000
#define DEFAULT_FP32_TOL 1.0e-6
#define DEFAULT_FP32_MAXITER 1000
#define DEFAULT_FP64_TOL 1.0e-12
#define DEFAULT_FP64_MAXITER 1000
#define DEFAULT_REPEATS 5
#define DEFAULT_SEED 20260603ULL
#define DEFAULT_WARMUP_N 128
#define MAX_VALUES 64
#define ERROR_LEN 512

typedef struct {
    double time_sec;
    double residual;
    char status[32];
    char error[ERROR_LEN];
} SolverResult;

typedef struct {
    double fp16_time_sec;
    double fp64_time_sec;
    double total_time_sec;
    double residual;
    double fp16_fixed_point_residual;
    double fp64_fixed_point_residual;
    int fp16_iterations;
    int fp64_iterations;
    char status[32];
    char error[ERROR_LEN];
} MixedResult;

typedef struct {
    double time_sec;
    double residual;
    double fixed_point_residual;
    int iterations;
    char status[32];
    char error[ERROR_LEN];
} IptResult;

typedef struct {
    const char *root;
    const char *results_dir;
    const char *logs_dir;
    int ns[MAX_VALUES];
    int n_count;
    double eps_values[MAX_VALUES];
    int eps_count;
    int warmup_n;
    double fp16_tol;
    int fp16_maxiter;
    double fp32_tol;
    int fp32_maxiter;
    double fp64_tol;
    int fp64_maxiter;
    int repeats;
    uint64_t seed;
    int compute_residuals;
    int append_results;
    int skip_existing;
    char summary_txt[4096];
} Config;

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

static const char *env_or_default(const char *name, const char *default_value)
{
    const char *value = getenv(name);

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
        fprintf(stderr, "integer list did not contain any valid values\n");
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
        fprintf(stderr, "double list did not contain any valid values\n");
        exit(2);
    }
}

static void load_config(Config *cfg)
{
    const char *raw_ns = NULL;
    const char *raw_eps = NULL;

    memset(cfg, 0, sizeof(*cfg));
    cfg->root = env_or_default("IPT_C_ROOT", DEFAULT_ROOT);
    cfg->results_dir = env_or_default("IPT_C_RESULTS_DIR", NULL);
    cfg->logs_dir = env_or_default("IPT_C_LOG_DIR", NULL);

    if (cfg->results_dir == NULL) {
        static char default_results[4096];
        snprintf(default_results, sizeof(default_results),
                 "%s/results/mixed_precision", cfg->root);
        cfg->results_dir = default_results;
    }

    if (cfg->logs_dir == NULL) {
        static char default_logs[4096];
        snprintf(default_logs, sizeof(default_logs),
                 "%s/logs/mixed_precision", cfg->root);
        cfg->logs_dir = default_logs;
    }

    raw_ns = env_or_default("IPT_SWEEP_NS", DEFAULT_NS);
    raw_eps = env_or_default("IPT_EPSILON_VALUES", DEFAULT_EPS_VALUES);
    parse_int_list(raw_ns, cfg->ns, &cfg->n_count);
    parse_double_list(raw_eps, cfg->eps_values, &cfg->eps_count);

    cfg->warmup_n = getenv("IPT_WARMUP_N") ? atoi(getenv("IPT_WARMUP_N"))
                                           : DEFAULT_WARMUP_N;
    if (cfg->warmup_n <= 0) {
        cfg->warmup_n = DEFAULT_WARMUP_N;
    }

    cfg->fp16_tol = getenv("IPT_FP16_TOL") ? atof(getenv("IPT_FP16_TOL"))
                                           : DEFAULT_FP16_TOL;
    if (cfg->fp16_tol <= 0.0) {
        cfg->fp16_tol = DEFAULT_FP16_TOL;
    }

    cfg->fp16_maxiter = getenv("IPT_FP16_MAXITER")
                            ? atoi(getenv("IPT_FP16_MAXITER"))
                            : DEFAULT_FP16_MAXITER;
    if (cfg->fp16_maxiter <= 0) {
        cfg->fp16_maxiter = DEFAULT_FP16_MAXITER;
    }

    cfg->fp32_tol = getenv("IPT_FP32_TOL") ? atof(getenv("IPT_FP32_TOL"))
                                           : DEFAULT_FP32_TOL;
    if (cfg->fp32_tol <= 0.0) {
        cfg->fp32_tol = DEFAULT_FP32_TOL;
    }

    cfg->fp32_maxiter = getenv("IPT_FP32_MAXITER")
                            ? atoi(getenv("IPT_FP32_MAXITER"))
                            : DEFAULT_FP32_MAXITER;
    if (cfg->fp32_maxiter <= 0) {
        cfg->fp32_maxiter = DEFAULT_FP32_MAXITER;
    }

    cfg->fp64_tol = getenv("IPT_TOL") ? atof(getenv("IPT_TOL"))
                                      : DEFAULT_FP64_TOL;
    if (cfg->fp64_tol <= 0.0) {
        cfg->fp64_tol = DEFAULT_FP64_TOL;
    }

    cfg->fp64_maxiter = getenv("IPT_MAXITER") ? atoi(getenv("IPT_MAXITER"))
                                              : DEFAULT_FP64_MAXITER;
    if (cfg->fp64_maxiter <= 0) {
        cfg->fp64_maxiter = DEFAULT_FP64_MAXITER;
    }

    cfg->repeats =
        getenv("IPT_REPEATS") ? atoi(getenv("IPT_REPEATS")) : DEFAULT_REPEATS;
    if (cfg->repeats <= 0) {
        cfg->repeats = DEFAULT_REPEATS;
    }

    cfg->seed =
        getenv("IPT_SEED") ? strtoull(getenv("IPT_SEED"), NULL, 10)
                           : DEFAULT_SEED;
    cfg->compute_residuals = parse_bool_env("IPT_COMPUTE_RESIDUALS", 1);
    cfg->append_results = parse_bool_env("IPT_APPEND_RESULTS", 0);
    cfg->skip_existing = parse_bool_env("IPT_SKIP_EXISTING", 0);

    {
        const char *summary = getenv("IPT_SUMMARY_TXT");

        if (summary != NULL && summary[0] != '\0') {
            snprintf(cfg->summary_txt, sizeof(cfg->summary_txt), "%s",
                     summary);
        } else {
            snprintf(cfg->summary_txt, sizeof(cfg->summary_txt),
                     "%s/low_precision_cusolver_sweep_summary.txt",
                     cfg->results_dir);
        }
    }
}

static void solver_result_reset(SolverResult *result, const char *status)
{
    result->time_sec = NAN;
    result->residual = NAN;
    snprintf(result->status, sizeof(result->status), "%s", status);
    result->error[0] = '\0';
}

static void solver_result_fail(SolverResult *result, const char *status,
                               const char *message)
{
    snprintf(result->status, sizeof(result->status), "%s", status);
    snprintf(result->error, sizeof(result->error), "%s", message);
}

static void mixed_result_reset(MixedResult *result, const char *status)
{
    result->fp16_time_sec = NAN;
    result->fp64_time_sec = NAN;
    result->total_time_sec = NAN;
    result->residual = NAN;
    result->fp16_fixed_point_residual = NAN;
    result->fp64_fixed_point_residual = NAN;
    result->fp16_iterations = 0;
    result->fp64_iterations = 0;
    snprintf(result->status, sizeof(result->status), "%s", status);
    result->error[0] = '\0';
}

static void mixed_result_fail(MixedResult *result, const char *status,
                              const char *message)
{
    snprintf(result->status, sizeof(result->status), "%s", status);
    snprintf(result->error, sizeof(result->error), "%s", message);
}

static void ipt_result_reset(IptResult *result, const char *status)
{
    result->time_sec = NAN;
    result->residual = NAN;
    result->fixed_point_residual = NAN;
    result->iterations = 0;
    snprintf(result->status, sizeof(result->status), "%s", status);
    result->error[0] = '\0';
}

static void ipt_result_fail(IptResult *result, const char *status,
                            const char *message)
{
    snprintf(result->status, sizeof(result->status), "%s", status);
    snprintf(result->error, sizeof(result->error), "%s", message);
}

static const char *cuda_status_name(cudaError_t status)
{
    return cudaGetErrorString(status);
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

            matrix[row + col * n] = value;
            matrix[col + row * n] = value;
        }
    }

    *matrix_out = matrix;
    return 0;
}

__global__ static void subtract_vdiag_kernel(double *residual,
                                             const double *vectors,
                                             const double *values, int n,
                                             int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;

    if (idx < total) {
        int col = idx / n;
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
    cudaError_t cuda_status;
    cublasStatus_t cublas_status;

    if (elements > (size_t)INT_MAX) {
        snprintf(error, error_size, "residual vector too large for cuBLAS");
        return -1;
    }

    cuda_status = cudaMalloc((void **)&d_residual, elements * sizeof(double));
    if (cuda_status != cudaSuccess) {
        snprintf(error, error_size, "cudaMalloc residual failed: %s",
                 cuda_status_name(cuda_status));
        return -1;
    }

    cublas_status =
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &one, d_matrix,
                    n, d_vectors, n, &zero, d_residual, n);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        snprintf(error, error_size, "residual cublasDgemm failed: %s",
                 cublas_status_name_local(cublas_status));
        status = -1;
        goto cleanup;
    }

    {
        int block = 256;
        int grid = ((int)elements + block - 1) / block;

        subtract_vdiag_kernel<<<grid, block>>>(d_residual, d_vectors, d_values,
                                               n, k);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            snprintf(error, error_size, "residual kernel launch failed: %s",
                     cuda_status_name(cuda_status));
            status = -1;
            goto cleanup;
        }
    }

    cublas_status =
        cublasDnrm2(handle, (int)elements, d_residual, 1, residual_out);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        snprintf(error, error_size, "residual cublasDnrm2 failed: %s",
                 cublas_status_name_local(cublas_status));
        status = -1;
        goto cleanup;
    }

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        snprintf(error, error_size, "residual synchronize failed: %s",
                 cuda_status_name(cuda_status));
        status = -1;
        goto cleanup;
    }

cleanup:
    cudaFree(d_residual);
    return status;
}

static int copy_matrix_to_device(double **d_matrix_out, const double *h_matrix,
                                 int n, char *error, size_t error_size)
{
    size_t bytes = (size_t)n * (size_t)n * sizeof(double);
    double *d_matrix = NULL;
    cudaError_t cuda_status = cudaMalloc((void **)&d_matrix, bytes);

    if (cuda_status != cudaSuccess) {
        snprintf(error, error_size, "cudaMalloc matrix failed: %s",
                 cuda_status_name(cuda_status));
        return -1;
    }

    cuda_status =
        cudaMemcpy(d_matrix, h_matrix, bytes, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        snprintf(error, error_size, "cudaMemcpy matrix failed: %s",
                 cuda_status_name(cuda_status));
        cudaFree(d_matrix);
        return -1;
    }

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        snprintf(error, error_size, "matrix copy synchronize failed: %s",
                 cuda_status_name(cuda_status));
        cudaFree(d_matrix);
        return -1;
    }

    *d_matrix_out = d_matrix;
    return 0;
}

static SolverResult run_syevd(cusolverDnHandle_t solver_handle,
                              cublasHandle_t blas_handle,
                              const double *d_matrix, int n,
                              int compute_residuals)
{
    SolverResult result;
    double *d_a = NULL;
    double *d_w = NULL;
    double *d_work = NULL;
    int *d_info = NULL;
    int h_info = 0;
    int lwork = 0;
    cudaError_t cuda_status;
    cusolverStatus_t solver_status;

    solver_result_reset(&result, "ok");

    cuda_status =
        cudaMalloc((void **)&d_a, (size_t)n * (size_t)n * sizeof(double));
    if (cuda_status != cudaSuccess) {
        solver_result_fail(&result, "failed", cuda_status_name(cuda_status));
        return result;
    }

    cuda_status = cudaMemcpy(d_a, d_matrix,
                             (size_t)n * (size_t)n * sizeof(double),
                             cudaMemcpyDeviceToDevice);
    if (cuda_status != cudaSuccess) {
        solver_result_fail(&result, "failed", cuda_status_name(cuda_status));
        goto cleanup;
    }

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        solver_result_fail(&result, "failed", cuda_status_name(cuda_status));
        goto cleanup;
    }

    {
        double start = now_seconds();

        cuda_status = cudaMalloc((void **)&d_w, (size_t)n * sizeof(double));
        if (cuda_status != cudaSuccess) {
            solver_result_fail(&result, "failed",
                               cuda_status_name(cuda_status));
            goto timed_done;
        }

        cuda_status = cudaMalloc((void **)&d_info, sizeof(int));
        if (cuda_status != cudaSuccess) {
            solver_result_fail(&result, "failed",
                               cuda_status_name(cuda_status));
            goto timed_done;
        }

        solver_status = cusolverDnDsyevd_bufferSize(
            solver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n,
            d_a, n, d_w, &lwork);
        if (solver_status != CUSOLVER_STATUS_SUCCESS) {
            solver_result_fail(&result, "failed",
                               cusolver_status_name(solver_status));
            goto timed_done;
        }

        cuda_status =
            cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(double));
        if (cuda_status != cudaSuccess) {
            solver_result_fail(&result, "failed",
                               cuda_status_name(cuda_status));
            goto timed_done;
        }

        solver_status = cusolverDnDsyevd(
            solver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n,
            d_a, n, d_w, d_work, lwork, d_info);
        if (solver_status != CUSOLVER_STATUS_SUCCESS) {
            solver_result_fail(&result, "failed",
                               cusolver_status_name(solver_status));
            goto timed_done;
        }

        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            solver_result_fail(&result, "failed",
                               cuda_status_name(cuda_status));
            goto timed_done;
        }

    timed_done:
        result.time_sec = now_seconds() - start;
    }

    if (strcmp(result.status, "ok") != 0) {
        goto cleanup;
    }

    cuda_status =
        cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        solver_result_fail(&result, "failed", cuda_status_name(cuda_status));
        goto cleanup;
    }
    if (h_info != 0) {
        snprintf(result.error, sizeof(result.error),
                 "cusolverDnDsyevd info=%d", h_info);
        snprintf(result.status, sizeof(result.status), "failed");
        goto cleanup;
    }

    if (compute_residuals) {
        if (residual_norm_gpu(blas_handle, d_matrix, d_a, d_w, n, n,
                              &result.residual, result.error,
                              sizeof(result.error)) != 0) {
            snprintf(result.status, sizeof(result.status), "failed");
        }
    }

cleanup:
    cudaFree(d_a);
    cudaFree(d_w);
    cudaFree(d_work);
    cudaFree(d_info);
    return result;
}

static SolverResult run_syevd_average(cusolverDnHandle_t solver_handle,
                                      cublasHandle_t blas_handle,
                                      const double *d_matrix, int n,
                                      int compute_residuals, int repeats)
{
    SolverResult average;
    double total = 0.0;
    int successful_runs = 0;

    solver_result_reset(&average, "ok");

    printf("CUSOLVER SYEVD warmup for N=%d\n", n);
    {
        SolverResult warm =
            run_syevd(solver_handle, blas_handle, d_matrix, n, 0);
        if (strcmp(warm.status, "ok") != 0) {
            solver_result_fail(&average, "failed", warm.error);
            average.time_sec = warm.time_sec;
            return average;
        }
    }

    for (int rep = 1; rep <= repeats; ++rep) {
        SolverResult one =
            run_syevd(solver_handle, blas_handle, d_matrix, n,
                      compute_residuals && rep == repeats);

        printf("  CUSOLVER SYEVD run %d/%d: time = %.8g s, residual = %.8g, "
               "status = %s",
               rep, repeats, one.time_sec, one.residual, one.status);
        if (one.error[0] != '\0') {
            printf(", error = %s", one.error);
        }
        printf("\n");

        if (strcmp(one.status, "ok") != 0) {
            average = one;
            if (successful_runs > 0) {
                average.time_sec = total / (double)successful_runs;
            }
            return average;
        }

        total += one.time_sec;
        successful_runs += 1;
        if (rep == repeats) {
            average.residual = one.residual;
        }
    }

    average.time_sec = total / (double)successful_runs;
    return average;
}

static IptResult run_original_ipt(cublasHandle_t blas_handle,
                                  const double *d_matrix, int n,
                                  const Config *cfg, int compute_residuals)
{
    IptResult result;
    double *d_vectors = NULL;
    double *d_values = NULL;
    int ipt_status = IPT_CUDA_SUCCESS;
    cudaError_t cuda_status;

    ipt_result_reset(&result, "ok");

    {
        double start = now_seconds();

        ipt_status = ipt_cuda_device_tol(
            d_matrix, n, n, cfg->fp64_tol, cfg->fp64_maxiter, blas_handle,
            &d_vectors, &d_values, &result.iterations,
            &result.fixed_point_residual);

        if (ipt_status != IPT_CUDA_SUCCESS) {
            ipt_result_fail(&result, "failed",
                            ipt_cuda_status_string(ipt_status));
            goto timed_done;
        }

        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            ipt_result_fail(&result, "failed", cuda_status_name(cuda_status));
            goto timed_done;
        }

    timed_done:
        result.time_sec = now_seconds() - start;
    }

    if (strcmp(result.status, "ok") == 0 && compute_residuals) {
        if (residual_norm_gpu(blas_handle, d_matrix, d_vectors, d_values, n, n,
                              &result.residual, result.error,
                              sizeof(result.error)) != 0) {
            snprintf(result.status, sizeof(result.status), "failed");
        }
    }

    ipt_cuda_free_device_result(d_vectors, d_values);
    return result;
}

static IptResult run_original_ipt_average(cublasHandle_t blas_handle,
                                          const double *d_matrix, int n,
                                          const Config *cfg)
{
    IptResult average;
    double total = 0.0;
    int successful_runs = 0;

    ipt_result_reset(&average, "ok");

    printf("Original IPT warmup for N=%d\n", n);
    {
        IptResult warm = run_original_ipt(blas_handle, d_matrix, n, cfg, 0);
        if (strcmp(warm.status, "ok") != 0) {
            ipt_result_fail(&average, "failed", warm.error);
            average.time_sec = warm.time_sec;
            return average;
        }
    }

    for (int rep = 1; rep <= cfg->repeats; ++rep) {
        IptResult one =
            run_original_ipt(blas_handle, d_matrix, n, cfg,
                             cfg->compute_residuals && rep == cfg->repeats);

        printf("  original IPT run %d/%d: time = %.8g s, fp = %.8g, "
               "iterations = %d, residual = %.8g, status = %s",
               rep, cfg->repeats, one.time_sec, one.fixed_point_residual,
               one.iterations, one.residual, one.status);
        if (one.error[0] != '\0') {
            printf(", error = %s", one.error);
        }
        printf("\n");

        if (strcmp(one.status, "ok") != 0) {
            average = one;
            if (successful_runs > 0) {
                average.time_sec = total / (double)successful_runs;
            }
            return average;
        }

        total += one.time_sec;
        successful_runs += 1;
        if (rep == cfg->repeats) {
            average.residual = one.residual;
            average.fixed_point_residual = one.fixed_point_residual;
            average.iterations = one.iterations;
        }
    }

    average.time_sec = total / (double)successful_runs;
    return average;
}

static MixedResult run_mixed(cublasHandle_t blas_handle,
                             const double *d_matrix, int n,
                             const Config *cfg, int compute_residuals)
{
    MixedResult result;
    IPTMixedCudaDeviceResult device_result;
    int status = IPT_CUDA_SUCCESS;
    cudaError_t cuda_status;

    mixed_result_reset(&result, "ok");
    ipt_mixed_reset_device_result(&device_result);

    status = ipt_mixed_precision_cuda_device_tol(
        d_matrix, n, n, cfg->fp16_tol, cfg->fp16_maxiter, cfg->fp64_tol,
        cfg->fp64_maxiter, blas_handle, &device_result);
    if (status != IPT_CUDA_SUCCESS) {
        mixed_result_fail(&result, "failed", ipt_cuda_status_string(status));
        goto cleanup;
    }

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        mixed_result_fail(&result, "failed", cuda_status_name(cuda_status));
        goto cleanup;
    }

    result.fp16_time_sec = device_result.fp16_time_sec;
    result.fp64_time_sec = device_result.fp64_time_sec;
    result.total_time_sec = device_result.total_time_sec;
    result.fp16_iterations = device_result.fp16_iterations;
    result.fp64_iterations = device_result.fp64_iterations;
    result.fp16_fixed_point_residual =
        device_result.fp16_fixed_point_residual;
    result.fp64_fixed_point_residual =
        device_result.fp64_fixed_point_residual;

    if (compute_residuals) {
        if (residual_norm_gpu(blas_handle, d_matrix, device_result.d_vectors,
                              device_result.d_values, n, n, &result.residual,
                              result.error, sizeof(result.error)) != 0) {
            snprintf(result.status, sizeof(result.status), "failed");
        }
    }

cleanup:
    ipt_mixed_precision_cuda_free_device_result(&device_result);
    return result;
}

static MixedResult run_mixed_fp32(cublasHandle_t blas_handle,
                                  const double *d_matrix, int n,
                                  const Config *cfg, int compute_residuals)
{
    MixedResult result;
    IPTMixedFP32CudaDeviceResult device_result;
    int status = IPT_CUDA_SUCCESS;
    cudaError_t cuda_status;

    mixed_result_reset(&result, "ok");
    ipt_mixed_fp32_reset_device_result(&device_result);

    status = ipt_mixed_precision_fp32_cuda_device_tol(
        d_matrix, n, n, cfg->fp32_tol, cfg->fp32_maxiter, cfg->fp64_tol,
        cfg->fp64_maxiter, blas_handle, &device_result);
    if (status != IPT_CUDA_SUCCESS) {
        mixed_result_fail(&result, "failed", ipt_cuda_status_string(status));
        goto cleanup;
    }

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        mixed_result_fail(&result, "failed", cuda_status_name(cuda_status));
        goto cleanup;
    }

    result.fp16_time_sec = device_result.fp32_time_sec;
    result.fp64_time_sec = device_result.fp64_time_sec;
    result.total_time_sec = device_result.total_time_sec;
    result.fp16_iterations = device_result.fp32_iterations;
    result.fp64_iterations = device_result.fp64_iterations;
    result.fp16_fixed_point_residual =
        device_result.fp32_fixed_point_residual;
    result.fp64_fixed_point_residual =
        device_result.fp64_fixed_point_residual;

    if (compute_residuals) {
        if (residual_norm_gpu(blas_handle, d_matrix, device_result.d_vectors,
                              device_result.d_values, n, n, &result.residual,
                              result.error, sizeof(result.error)) != 0) {
            snprintf(result.status, sizeof(result.status), "failed");
        }
    }

cleanup:
    ipt_mixed_precision_fp32_cuda_free_device_result(&device_result);
    return result;
}

static MixedResult run_mixed_average(cublasHandle_t blas_handle,
                                     const double *d_matrix, int n,
                                     const Config *cfg)
{
    MixedResult average;
    double total_fp16 = 0.0;
    double total_fp64 = 0.0;
    double total_all = 0.0;
    int successful_runs = 0;

    mixed_result_reset(&average, "ok");

    printf("Mixed FP16->FP64 warmup for N=%d\n", n);
    {
        MixedResult warm = run_mixed(blas_handle, d_matrix, n, cfg, 0);
        if (strcmp(warm.status, "ok") != 0) {
            mixed_result_fail(&average, "failed", warm.error);
            average.fp16_time_sec = warm.fp16_time_sec;
            average.fp64_time_sec = warm.fp64_time_sec;
            average.total_time_sec = warm.total_time_sec;
            return average;
        }
    }

    for (int rep = 1; rep <= cfg->repeats; ++rep) {
        MixedResult one =
            run_mixed(blas_handle, d_matrix, n, cfg,
                      cfg->compute_residuals && rep == cfg->repeats);

        printf("  mixed run %d/%d: fp16 = %.8g s, fp64 = %.8g s, "
               "total = %.8g s, fp16_fp = %.8g, fp64_fp = %.8g, "
               "fp16_iters = %d, fp64_iters = %d, residual = %.8g, "
               "status = %s",
               rep, cfg->repeats, one.fp16_time_sec, one.fp64_time_sec,
               one.total_time_sec, one.fp16_fixed_point_residual,
               one.fp64_fixed_point_residual, one.fp16_iterations,
               one.fp64_iterations, one.residual, one.status);
        if (one.error[0] != '\0') {
            printf(", error = %s", one.error);
        }
        printf("\n");

        if (strcmp(one.status, "ok") != 0) {
            average = one;
            if (successful_runs > 0) {
                average.fp16_time_sec =
                    total_fp16 / (double)successful_runs;
                average.fp64_time_sec =
                    total_fp64 / (double)successful_runs;
                average.total_time_sec =
                    total_all / (double)successful_runs;
            }
            return average;
        }

        total_fp16 += one.fp16_time_sec;
        total_fp64 += one.fp64_time_sec;
        total_all += one.total_time_sec;
        successful_runs += 1;

        if (rep == cfg->repeats) {
            average.residual = one.residual;
            average.fp16_fixed_point_residual =
                one.fp16_fixed_point_residual;
            average.fp64_fixed_point_residual =
                one.fp64_fixed_point_residual;
            average.fp16_iterations = one.fp16_iterations;
            average.fp64_iterations = one.fp64_iterations;
        }
    }

    average.fp16_time_sec = total_fp16 / (double)successful_runs;
    average.fp64_time_sec = total_fp64 / (double)successful_runs;
    average.total_time_sec = total_all / (double)successful_runs;
    return average;
}

static MixedResult run_mixed_fp32_average(cublasHandle_t blas_handle,
                                          const double *d_matrix, int n,
                                          const Config *cfg)
{
    MixedResult average;
    double total_initial = 0.0;
    double total_fp64 = 0.0;
    double total_all = 0.0;
    int successful_runs = 0;

    mixed_result_reset(&average, "ok");

    printf("Mixed FP32->FP64 warmup for N=%d\n", n);
    {
        MixedResult warm = run_mixed_fp32(blas_handle, d_matrix, n, cfg, 0);
        if (strcmp(warm.status, "ok") != 0) {
            mixed_result_fail(&average, "failed", warm.error);
            average.fp16_time_sec = warm.fp16_time_sec;
            average.fp64_time_sec = warm.fp64_time_sec;
            average.total_time_sec = warm.total_time_sec;
            return average;
        }
    }

    for (int rep = 1; rep <= cfg->repeats; ++rep) {
        MixedResult one =
            run_mixed_fp32(blas_handle, d_matrix, n, cfg,
                           cfg->compute_residuals && rep == cfg->repeats);

        printf("  mixed run %d/%d: fp32 = %.8g s, fp64 = %.8g s, "
               "total = %.8g s, fp32_fp = %.8g, fp64_fp = %.8g, "
               "fp32_iters = %d, fp64_iters = %d, residual = %.8g, "
               "status = %s",
               rep, cfg->repeats, one.fp16_time_sec, one.fp64_time_sec,
               one.total_time_sec, one.fp16_fixed_point_residual,
               one.fp64_fixed_point_residual, one.fp16_iterations,
               one.fp64_iterations, one.residual, one.status);
        if (one.error[0] != '\0') {
            printf(", error = %s", one.error);
        }
        printf("\n");

        if (strcmp(one.status, "ok") != 0) {
            average = one;
            if (successful_runs > 0) {
                average.fp16_time_sec =
                    total_initial / (double)successful_runs;
                average.fp64_time_sec =
                    total_fp64 / (double)successful_runs;
                average.total_time_sec =
                    total_all / (double)successful_runs;
            }
            return average;
        }

        total_initial += one.fp16_time_sec;
        total_fp64 += one.fp64_time_sec;
        total_all += one.total_time_sec;
        successful_runs += 1;

        if (rep == cfg->repeats) {
            average.residual = one.residual;
            average.fp16_fixed_point_residual =
                one.fp16_fixed_point_residual;
            average.fp64_fixed_point_residual =
                one.fp64_fixed_point_residual;
            average.fp16_iterations = one.fp16_iterations;
            average.fp64_iterations = one.fp64_iterations;
        }
    }

    average.fp16_time_sec = total_initial / (double)successful_runs;
    average.fp64_time_sec = total_fp64 / (double)successful_runs;
    average.total_time_sec = total_all / (double)successful_runs;
    return average;
}

static void csv_clean(char *text)
{
    for (char *p = text; *p != '\0'; ++p) {
        if (*p == ',' || *p == '\n' || *p == '\r') {
            *p = ';';
        }
    }
}

static void join_errors(char *out, size_t size, const char *a, const char *b,
                        const char *c)
{
    const char *items[3] = {a, b, c};
    size_t used = 0;

    if (size == 0) {
        return;
    }
    out[0] = '\0';

    for (int i = 0; i < 3; ++i) {
        const char *item = items[i];

        if (item == NULL || item[0] == '\0') {
            continue;
        }

        if (used > 0 && used + 3 < size) {
            snprintf(out + used, size - used, " | ");
            used = strlen(out);
        }

        if (used < size - 1) {
            snprintf(out + used, size - used, "%s", item);
            used = strlen(out);
        }
    }
}

static const char *bool_text(int value)
{
    return value ? "true" : "false";
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
        } else if (*p == '+' || *p == '-') {
            *p = (*p == '+') ? 'p' : 'm';
        } else if (*p == 'e' || *p == 'E') {
            *p = 'e';
        } else if (*p < '0' || *p > '9') {
            *p = '_';
        }
    }
}

static void result_csv_path(const Config *cfg, double epsilon, char *path,
                            size_t size)
{
    char label[64];

    epsilon_label(epsilon, label, sizeof(label));
    snprintf(path, size, "%s/low_precision_cusolver_sweep_epsilon_%s.csv",
             cfg->results_dir, label);
}

static void write_csv_header(const char *csv_path)
{
    FILE *fp = fopen(csv_path, "w");

    if (fp == NULL) {
        fprintf(stderr, "could not open %s: %s\n", csv_path, strerror(errno));
        exit(2);
    }

    fprintf(fp,
            "N,epsilon,initial_precision,initial_tol,fp64_tol,"
            "initial_maxiter,fp64_maxiter,repeats,"
            "time_cusolver_syevd_sec,residual_cusolver_syevd,"
            "time_original_ipt_sec,residual_original_ipt,"
            "original_ipt_fixed_point_residual,original_ipt_iterations,"
            "time_initial_sec,time_fp64_refine_sec,time_mixed_total_sec,"
            "residual_mixed,initial_fixed_point_residual,"
            "fp64_fixed_point_residual,initial_iterations,fp64_iterations,"
            "initial_converged,fp64_converged,gpu_name,status,error\n");
    fclose(fp);
}

static void write_csv_header_if_missing(const char *csv_path)
{
    struct stat st;

    if (stat(csv_path, &st) != 0 || st.st_size == 0) {
        write_csv_header(csv_path);
    }
}

static int csv_has_row_for_precision(const char *csv_path, int n,
                                     const char *precision)
{
    FILE *fp = fopen(csv_path, "r");
    char line[8192];

    if (fp == NULL) {
        return 0;
    }

    while (fgets(line, sizeof(line), fp) != NULL) {
        char *saveptr = NULL;
        char *n_text = strtok_r(line, ",", &saveptr);
        char *epsilon_text = strtok_r(NULL, ",", &saveptr);
        char *precision_text = strtok_r(NULL, ",", &saveptr);

        (void)epsilon_text;
        if (n_text == NULL || precision_text == NULL) {
            continue;
        }
        if (atoi(n_text) == n && strcmp(precision_text, precision) == 0) {
            fclose(fp);
            return 1;
        }
    }

    fclose(fp);
    return 0;
}

static int csv_has_complete_case(const char *csv_path, int n)
{
    return csv_has_row_for_precision(csv_path, n, "fp16") &&
           csv_has_row_for_precision(csv_path, n, "fp32");
}

static void append_csv_row(const char *csv_path, const Config *cfg, int n,
                           double epsilon, const SolverResult *syevd,
                           const IptResult *original_ipt,
                           const MixedResult *mixed,
                           const char *initial_precision, double initial_tol,
                           int initial_maxiter, const char *gpu_name,
                           const char *status, const char *error)
{
    FILE *fp = fopen(csv_path, "a");
    char gpu_clean[256];
    char status_clean[64];
    char error_clean[1024];
    int initial_converged = isfinite(mixed->fp16_fixed_point_residual) &&
                            mixed->fp16_fixed_point_residual <= initial_tol;
    int fp64_converged = isfinite(mixed->fp64_fixed_point_residual) &&
                         mixed->fp64_fixed_point_residual <= cfg->fp64_tol;

    if (fp == NULL) {
        fprintf(stderr, "could not append %s: %s\n", csv_path,
                strerror(errno));
        return;
    }

    snprintf(gpu_clean, sizeof(gpu_clean), "%s", gpu_name);
    snprintf(status_clean, sizeof(status_clean), "%s", status);
    snprintf(error_clean, sizeof(error_clean), "%s", error);
    csv_clean(gpu_clean);
    csv_clean(status_clean);
    csv_clean(error_clean);

    fprintf(fp,
            "%d,%.12g,%s,%.17g,%.17g,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d,"
            "%s,%s,%s,%s,%s\n",
            n, epsilon, initial_precision, initial_tol, cfg->fp64_tol,
            initial_maxiter, cfg->fp64_maxiter, cfg->repeats, syevd->time_sec,
            syevd->residual, original_ipt->time_sec, original_ipt->residual,
            original_ipt->fixed_point_residual, original_ipt->iterations,
            mixed->fp16_time_sec, mixed->fp64_time_sec, mixed->total_time_sec,
            mixed->residual,
            mixed->fp16_fixed_point_residual,
            mixed->fp64_fixed_point_residual, mixed->fp16_iterations,
            mixed->fp64_iterations, bool_text(initial_converged),
            bool_text(fp64_converged), gpu_clean, status_clean, error_clean);
    fclose(fp);
}

static void append_summary(const Config *cfg, const char *message)
{
    FILE *fp = fopen(cfg->summary_txt, "a");

    if (fp == NULL) {
        fprintf(stderr, "could not append %s: %s\n", cfg->summary_txt,
                strerror(errno));
        return;
    }

    fprintf(fp, "%s\n", message);
    fclose(fp);
}

static void write_summary_header(const Config *cfg, const char *gpu_name)
{
    FILE *fp = fopen(cfg->summary_txt, "w");
    char time_buf[64];
    int runtime_version = 0;
    int driver_version = 0;

    if (fp == NULL) {
        fprintf(stderr, "could not open %s: %s\n", cfg->summary_txt,
                strerror(errno));
        exit(2);
    }

    wall_time_string(time_buf, sizeof(time_buf));
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);

    fprintf(fp, "IPT_C mixed precision sweep: FP16/FP32 initial + FP64 "
                "refine vs cuSolver Dsyevd\n");
    fprintf(fp, "started_at = %s\n", time_buf);
    fprintf(fp, "ipt_c_root = %s\n", cfg->root);
    fprintf(fp, "results_dir = %s\n", cfg->results_dir);
    fprintf(fp, "summary_txt = %s\n", cfg->summary_txt);
    fprintf(fp, "fp16_tol = %.17g\n", cfg->fp16_tol);
    fprintf(fp, "fp16_maxiter = %d\n", cfg->fp16_maxiter);
    fprintf(fp, "fp32_tol = %.17g\n", cfg->fp32_tol);
    fprintf(fp, "fp32_maxiter = %d\n", cfg->fp32_maxiter);
    fprintf(fp, "fp64_tol = %.17g\n", cfg->fp64_tol);
    fprintf(fp, "fp64_maxiter = %d\n", cfg->fp64_maxiter);
    fprintf(fp, "timing_repeats = %d\n", cfg->repeats);
    fprintf(fp, "per_case_warmup = true\n");
    fprintf(fp, "seed = %llu\n", (unsigned long long)cfg->seed);
    fprintf(fp, "N_values = ");
    for (int i = 0; i < cfg->n_count; ++i) {
        fprintf(fp, "%s%d", i == 0 ? "" : ", ", cfg->ns[i]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "epsilon_values = ");
    for (int i = 0; i < cfg->eps_count; ++i) {
        fprintf(fp, "%s%.12g", i == 0 ? "" : ", ", cfg->eps_values[i]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "compute_residuals = %s\n",
            cfg->compute_residuals ? "true" : "false");
    fprintf(fp, "append_results = %s\n",
            cfg->append_results ? "true" : "false");
    fprintf(fp, "skip_existing = %s\n",
            cfg->skip_existing ? "true" : "false");
    fprintf(fp, "cuda_runtime_version = %d\n", runtime_version);
    fprintf(fp, "cuda_driver_version = %d\n", driver_version);
    fprintf(fp, "gpu_name = %s\n\n", gpu_name);
    fclose(fp);
}

static void show_gpu_memory(const char *tag)
{
    size_t free_bytes = 0;
    size_t total_bytes = 0;

    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        printf("CUDA memory %s: free %.3f GiB / total %.3f GiB\n", tag,
               (double)free_bytes / 1073741824.0,
               (double)total_bytes / 1073741824.0);
    }
}

static void warmup(cublasHandle_t blas_handle, cusolverDnHandle_t solver_handle,
                   const Config *cfg)
{
    double *h_matrix = NULL;
    double *d_matrix = NULL;
    char error[ERROR_LEN] = "";
    int warm_n = cfg->warmup_n;

    printf("===== device warmup N=%d =====\n", warm_n);
    if (make_host_matrix(&h_matrix, warm_n, cfg->eps_values[0],
                         cfg->seed + 17) != 0) {
        printf("warmup host matrix allocation failed\n");
        return;
    }
    if (copy_matrix_to_device(&d_matrix, h_matrix, warm_n, error,
                              sizeof(error)) != 0) {
        printf("warmup device copy failed: %s\n", error);
        free(h_matrix);
        return;
    }

    (void)run_syevd(solver_handle, blas_handle, d_matrix, warm_n, 0);
    (void)run_original_ipt(blas_handle, d_matrix, warm_n, cfg, 0);
    (void)run_mixed(blas_handle, d_matrix, warm_n, cfg, 0);
    (void)run_mixed_fp32(blas_handle, d_matrix, warm_n, cfg, 0);

    cudaFree(d_matrix);
    free(h_matrix);
    cudaDeviceSynchronize();
    printf("===== device warmup done =====\n");
}

int main(void)
{
    Config cfg;
    cudaDeviceProp prop;
    cublasHandle_t blas_handle = NULL;
    cusolverDnHandle_t solver_handle = NULL;
    char gpu_name[256] = "unknown_gpu";
    int failures = 0;
    cudaError_t cuda_status;
    cublasStatus_t blas_status;
    cusolverStatus_t solver_status;

    load_config(&cfg);
    ensure_directory(cfg.logs_dir);
    ensure_directory(cfg.results_dir);

    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n",
                cuda_status_name(cuda_status));
        return 2;
    }

    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        snprintf(gpu_name, sizeof(gpu_name), "%s", prop.name);
    }

    blas_status = cublasCreate(&blas_handle);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed: %s\n",
                cublas_status_name_local(blas_status));
        return 2;
    }
    (void)cublasSetMathMode(blas_handle, CUBLAS_TENSOR_OP_MATH);

    solver_status = cusolverDnCreate(&solver_handle);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "cusolverDnCreate failed: %s\n",
                cusolver_status_name(solver_status));
        cublasDestroy(blas_handle);
        return 2;
    }

    write_summary_header(&cfg, gpu_name);
    for (int eps_idx = 0; eps_idx < cfg.eps_count; ++eps_idx) {
        char csv_path[4096];

        result_csv_path(&cfg, cfg.eps_values[eps_idx], csv_path,
                        sizeof(csv_path));
        if (cfg.append_results) {
            write_csv_header_if_missing(csv_path);
        } else {
            write_csv_header(csv_path);
        }
    }

    printf("===== IPT_C mixed precision FP16/FP32 sweep vs cuSolver =====\n");
    printf("ROOT = %s\n", cfg.root);
    printf("RESULT_DIR = %s\n", cfg.results_dir);
    printf("SUMMARY_TXT = %s\n", cfg.summary_txt);
    printf("N values = ");
    for (int i = 0; i < cfg.n_count; ++i) {
        printf("%s%d", i == 0 ? "" : ", ", cfg.ns[i]);
    }
    printf("\n");
    printf("epsilon values = ");
    for (int i = 0; i < cfg.eps_count; ++i) {
        printf("%s%.12g", i == 0 ? "" : ", ", cfg.eps_values[i]);
    }
    printf("\n");
    printf("fp16_tol = %.17g\n", cfg.fp16_tol);
    printf("fp16_maxiter = %d\n", cfg.fp16_maxiter);
    printf("fp32_tol = %.17g\n", cfg.fp32_tol);
    printf("fp32_maxiter = %d\n", cfg.fp32_maxiter);
    printf("fp64_tol = %.17g\n", cfg.fp64_tol);
    printf("fp64_maxiter = %d\n", cfg.fp64_maxiter);
    printf("timing_repeats = %d\n", cfg.repeats);
    printf("seed = %llu\n", (unsigned long long)cfg.seed);
    printf("compute_residuals = %s\n",
           cfg.compute_residuals ? "true" : "false");
    printf("append_results = %s\n", cfg.append_results ? "true" : "false");
    printf("skip_existing = %s\n", cfg.skip_existing ? "true" : "false");
    printf("gpu_name = %s\n", gpu_name);
    show_gpu_memory("at start");

    warmup(blas_handle, solver_handle, &cfg);

    for (int eps_idx = 0; eps_idx < cfg.eps_count; ++eps_idx) {
        double epsilon = cfg.eps_values[eps_idx];
        char csv_path[4096];

        result_csv_path(&cfg, epsilon, csv_path, sizeof(csv_path));

        for (int n_idx = 0; n_idx < cfg.n_count; ++n_idx) {
            int n = cfg.ns[n_idx];
            double *h_matrix = NULL;
            double *d_matrix = NULL;
            SolverResult syevd;
            IptResult original_ipt;
            MixedResult mixed_fp16;
            MixedResult mixed_fp32;
            char status_fp16[64] = "ok";
            char status_fp32[64] = "ok";
            char error_fp16[1024] = "";
            char error_fp32[1024] = "";
            char copy_error[1024] = "";
            char summary_line[2048];
            char time_buf[64];
            double matrix_start = 0.0;
            int fp64_converged_fp16 = 0;
            int fp64_converged_fp32 = 0;

            solver_result_reset(&syevd, "not_run");
            ipt_result_reset(&original_ipt, "not_run");
            mixed_result_reset(&mixed_fp16, "not_run");
            mixed_result_reset(&mixed_fp32, "not_run");

            printf("\n===== epsilon = %.12g, N = %d =====\n", epsilon, n);
            if (cfg.skip_existing && csv_has_complete_case(csv_path, n)) {
                wall_time_string(time_buf, sizeof(time_buf));
                snprintf(summary_line, sizeof(summary_line),
                         "epsilon=%.12g N=%d skipped_existing_at=%s csv=%s",
                         epsilon, n, time_buf, csv_path);
                append_summary(&cfg, summary_line);
                printf("Skipping epsilon=%.12g N=%d because fp16/fp32 rows "
                       "already exist in %s\n",
                       epsilon, n, csv_path);
                continue;
            }

            wall_time_string(time_buf, sizeof(time_buf));
            snprintf(summary_line, sizeof(summary_line),
                     "epsilon=%.12g N=%d started_at=%s", epsilon, n,
                     time_buf);
            append_summary(&cfg, summary_line);
            show_gpu_memory("before matrix");

            matrix_start = now_seconds();
            if (make_host_matrix(&h_matrix, n, epsilon,
                                 cfg.seed + (uint64_t)n +
                                     epsilon_seed_offset(epsilon)) != 0) {
                snprintf(status_fp16, sizeof(status_fp16), "failed");
                snprintf(status_fp32, sizeof(status_fp32), "failed");
                snprintf(error_fp16, sizeof(error_fp16),
                         "host matrix allocation failed");
                snprintf(error_fp32, sizeof(error_fp32),
                         "host matrix allocation failed");
                failures++;
                goto row_done;
            }
            printf("host matrix generated in %.8g s\n",
                   now_seconds() - matrix_start);

            if (copy_matrix_to_device(&d_matrix, h_matrix, n, copy_error,
                                      sizeof(copy_error)) != 0) {
                snprintf(status_fp16, sizeof(status_fp16), "failed");
                snprintf(status_fp32, sizeof(status_fp32), "failed");
                snprintf(error_fp16, sizeof(error_fp16), "%s", copy_error);
                snprintf(error_fp32, sizeof(error_fp32), "%s", copy_error);
                failures++;
                goto row_done;
            }

            free(h_matrix);
            h_matrix = NULL;
            show_gpu_memory("after matrix");

            printf("Running CUSOLVER SYEVD for epsilon=%.12g N=%d\n", epsilon,
                   n);
            syevd = run_syevd_average(solver_handle, blas_handle, d_matrix, n,
                                      cfg.compute_residuals, cfg.repeats);

            printf("Running original IPT for epsilon=%.12g N=%d\n", epsilon,
                   n);
            original_ipt =
                run_original_ipt_average(blas_handle, d_matrix, n, &cfg);

            printf("Running mixed FP16->FP64 for epsilon=%.12g N=%d\n",
                   epsilon, n);
            mixed_fp16 = run_mixed_average(blas_handle, d_matrix, n, &cfg);

            fp64_converged_fp16 =
                isfinite(mixed_fp16.fp64_fixed_point_residual) &&
                mixed_fp16.fp64_fixed_point_residual <= cfg.fp64_tol;
            if (strcmp(syevd.status, "ok") != 0 ||
                strcmp(original_ipt.status, "ok") != 0 ||
                strcmp(mixed_fp16.status, "ok") != 0) {
                snprintf(status_fp16, sizeof(status_fp16), "partial_failed");
                join_errors(error_fp16, sizeof(error_fp16), syevd.error,
                            original_ipt.error, mixed_fp16.error);
                if (strcmp(original_ipt.status, "ok") != 0 ||
                    strcmp(mixed_fp16.status, "ok") != 0) {
                    failures++;
                }
            } else if (!fp64_converged_fp16) {
                snprintf(status_fp16, sizeof(status_fp16), "not_converged");
                snprintf(error_fp16, sizeof(error_fp16),
                         "fp64_fixed_point_residual %.17g > tol %.17g",
                         mixed_fp16.fp64_fixed_point_residual, cfg.fp64_tol);
            }

            printf("Averages FP16: syevd = %.8g s, original_ipt = %.8g s, "
                   "mixed_total = %.8g s, "
                   "mixed_residual = %.8g, initial_fp = %.8g, fp64_fp = %.8g, "
                   "fp16_iters = %d, fp64_iters = %d, status = %s\n",
                   syevd.time_sec, original_ipt.time_sec,
                   mixed_fp16.total_time_sec,
                   mixed_fp16.residual,
                   mixed_fp16.fp16_fixed_point_residual,
                   mixed_fp16.fp64_fixed_point_residual,
                   mixed_fp16.fp16_iterations,
                   mixed_fp16.fp64_iterations, status_fp16);

            printf("Running mixed FP32->FP64 for epsilon=%.12g N=%d\n",
                   epsilon, n);
            mixed_fp32 = run_mixed_fp32_average(blas_handle, d_matrix, n, &cfg);

            fp64_converged_fp32 =
                isfinite(mixed_fp32.fp64_fixed_point_residual) &&
                mixed_fp32.fp64_fixed_point_residual <= cfg.fp64_tol;
            if (strcmp(syevd.status, "ok") != 0 ||
                strcmp(original_ipt.status, "ok") != 0 ||
                strcmp(mixed_fp32.status, "ok") != 0) {
                snprintf(status_fp32, sizeof(status_fp32), "partial_failed");
                join_errors(error_fp32, sizeof(error_fp32), syevd.error,
                            original_ipt.error, mixed_fp32.error);
                if (strcmp(original_ipt.status, "ok") != 0 ||
                    strcmp(mixed_fp32.status, "ok") != 0) {
                    failures++;
                }
            } else if (!fp64_converged_fp32) {
                snprintf(status_fp32, sizeof(status_fp32), "not_converged");
                snprintf(error_fp32, sizeof(error_fp32),
                         "fp64_fixed_point_residual %.17g > tol %.17g",
                         mixed_fp32.fp64_fixed_point_residual, cfg.fp64_tol);
            }

            printf("Averages FP32: syevd = %.8g s, original_ipt = %.8g s, "
                   "mixed_total = %.8g s, "
                   "mixed_residual = %.8g, initial_fp = %.8g, fp64_fp = %.8g, "
                   "fp32_iters = %d, fp64_iters = %d, status = %s\n",
                   syevd.time_sec, original_ipt.time_sec,
                   mixed_fp32.total_time_sec,
                   mixed_fp32.residual,
                   mixed_fp32.fp16_fixed_point_residual,
                   mixed_fp32.fp64_fixed_point_residual,
                   mixed_fp32.fp16_iterations,
                   mixed_fp32.fp64_iterations, status_fp32);

        row_done:
            append_csv_row(csv_path, &cfg, n, epsilon, &syevd, &original_ipt,
                           &mixed_fp16, "fp16", cfg.fp16_tol,
                           cfg.fp16_maxiter, gpu_name, status_fp16,
                           error_fp16);
            append_csv_row(csv_path, &cfg, n, epsilon, &syevd, &original_ipt,
                           &mixed_fp32, "fp32", cfg.fp32_tol,
                           cfg.fp32_maxiter, gpu_name, status_fp32,
                           error_fp32);
            wall_time_string(time_buf, sizeof(time_buf));
            snprintf(summary_line, sizeof(summary_line),
                     "epsilon=%.12g N=%d initial_precision=fp16 "
                     "finished_at=%s status=%s "
                     "time_syevd=%.17g residual_syevd=%.17g "
                     "time_original_ipt=%.17g residual_original_ipt=%.17g "
                     "original_ipt_iterations=%d original_ipt_fp=%.17g "
                     "time_initial=%.17g time_fp64=%.17g time_mixed=%.17g "
                     "residual_mixed=%.17g initial_iterations=%d "
                     "fp64_iterations=%d initial_fp=%.17g fp64_fp=%.17g "
                     "repeats=%d per_case_warmup=true csv=%s",
                     epsilon, n, time_buf, status_fp16, syevd.time_sec,
                     syevd.residual, original_ipt.time_sec,
                     original_ipt.residual, original_ipt.iterations,
                     original_ipt.fixed_point_residual,
                     mixed_fp16.fp16_time_sec,
                     mixed_fp16.fp64_time_sec, mixed_fp16.total_time_sec,
                     mixed_fp16.residual, mixed_fp16.fp16_iterations,
                     mixed_fp16.fp64_iterations,
                     mixed_fp16.fp16_fixed_point_residual,
                     mixed_fp16.fp64_fixed_point_residual, cfg.repeats,
                     csv_path);
            append_summary(&cfg, summary_line);
            snprintf(summary_line, sizeof(summary_line),
                     "epsilon=%.12g N=%d initial_precision=fp32 "
                     "finished_at=%s status=%s "
                     "time_syevd=%.17g residual_syevd=%.17g "
                     "time_original_ipt=%.17g residual_original_ipt=%.17g "
                     "original_ipt_iterations=%d original_ipt_fp=%.17g "
                     "time_initial=%.17g time_fp64=%.17g time_mixed=%.17g "
                     "residual_mixed=%.17g initial_iterations=%d "
                     "fp64_iterations=%d initial_fp=%.17g fp64_fp=%.17g "
                     "repeats=%d per_case_warmup=true csv=%s",
                     epsilon, n, time_buf, status_fp32, syevd.time_sec,
                     syevd.residual, original_ipt.time_sec,
                     original_ipt.residual, original_ipt.iterations,
                     original_ipt.fixed_point_residual,
                     mixed_fp32.fp16_time_sec,
                     mixed_fp32.fp64_time_sec, mixed_fp32.total_time_sec,
                     mixed_fp32.residual, mixed_fp32.fp16_iterations,
                     mixed_fp32.fp64_iterations,
                     mixed_fp32.fp16_fixed_point_residual,
                     mixed_fp32.fp64_fixed_point_residual, cfg.repeats,
                     csv_path);
            append_summary(&cfg, summary_line);
            cudaFree(d_matrix);
            free(h_matrix);
            cudaDeviceSynchronize();
            show_gpu_memory("after cleanup");
        }
    }

    {
        char time_buf[64];
        char summary_line[128];

        wall_time_string(time_buf, sizeof(time_buf));
        snprintf(summary_line, sizeof(summary_line), "finished_at = %s",
                 time_buf);
        append_summary(&cfg, summary_line);
    }

    printf("===== sweep done =====\n");
    printf("Results dir: %s\n", cfg.results_dir);
    printf("Summary: %s\n", cfg.summary_txt);

    cusolverDnDestroy(solver_handle);
    cublasDestroy(blas_handle);
    return failures == 0 ? 0 : 1;
}
