#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <errno.h>
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
#define DEFAULT_N 16384
#define DEFAULT_FP32_TOL 1.0e-6
#define DEFAULT_FP32_MAXITER 1000
#define DEFAULT_FP64_TOL 1.0e-12
#define DEFAULT_FP64_MAXITER 1000
#define DEFAULT_REPEATS 5
#define DEFAULT_SEED 20260603ULL
#define DEFAULT_WARMUP_N 128
#define MAX_EPS_VALUES 32
#define ERROR_LEN 512

typedef struct {
    double fp32_time_sec;
    double fp64_time_sec;
    double total_time_sec;
    double residual;
    double fp32_fixed_point_residual;
    double fp64_fixed_point_residual;
    int fp32_iterations;
    int fp64_iterations;
    char status[32];
    char error[ERROR_LEN];
} MixedResult;

typedef struct {
    const char *root;
    const char *results_dir;
    const char *logs_dir;
    int n;
    int warmup_n;
    double fp32_tol;
    int fp32_maxiter;
    double fp64_tol;
    int fp64_maxiter;
    int repeats;
    uint64_t seed;
    int compute_residuals;
    double eps_values[MAX_EPS_VALUES];
    int eps_count;
    char result_csv[4096];
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

static void parse_eps_values(Config *cfg)
{
    const char *raw = getenv("IPT_EPSILON_VALUES");
    const char *defaults = "0.05,0.08,0.1,0.12,0.15";
    char *copy = NULL;
    char *token = NULL;

    cfg->eps_count = 0;
    copy = strdup((raw == NULL || raw[0] == '\0') ? defaults : raw);
    if (copy == NULL) {
        fprintf(stderr, "strdup failed while parsing IPT_EPSILON_VALUES\n");
        exit(2);
    }

    token = strtok(copy, ", \t");
    while (token != NULL && cfg->eps_count < MAX_EPS_VALUES) {
        if (token[0] != '\0') {
            cfg->eps_values[cfg->eps_count++] = atof(token);
        }
        token = strtok(NULL, ", \t");
    }
    free(copy);

    if (cfg->eps_count == 0) {
        fprintf(stderr, "IPT_EPSILON_VALUES did not contain any valid values\n");
        exit(2);
    }
}

static void load_config(Config *cfg)
{
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

    cfg->n = getenv("IPT_MIXED_N") ? atoi(getenv("IPT_MIXED_N")) : DEFAULT_N;
    if (cfg->n <= 0) {
        cfg->n = DEFAULT_N;
    }

    cfg->warmup_n = getenv("IPT_WARMUP_N") ? atoi(getenv("IPT_WARMUP_N"))
                                           : DEFAULT_WARMUP_N;
    if (cfg->warmup_n <= 0 || cfg->warmup_n > cfg->n) {
        cfg->warmup_n = DEFAULT_WARMUP_N;
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

    parse_eps_values(cfg);

    {
        const char *result_csv = getenv("IPT_RESULT_CSV");
        const char *summary_txt = getenv("IPT_SUMMARY_TXT");

        if (result_csv != NULL && result_csv[0] != '\0') {
            snprintf(cfg->result_csv, sizeof(cfg->result_csv), "%s",
                     result_csv);
        } else {
            snprintf(cfg->result_csv, sizeof(cfg->result_csv),
                     "%s/mixed_precision_fp32_fp64_results.csv",
                     cfg->results_dir);
        }

        if (summary_txt != NULL && summary_txt[0] != '\0') {
            snprintf(cfg->summary_txt, sizeof(cfg->summary_txt), "%s",
                     summary_txt);
        } else {
            snprintf(cfg->summary_txt, sizeof(cfg->summary_txt),
                     "%s/mixed_precision_fp32_fp64_summary.txt",
                     cfg->results_dir);
        }
    }
}

static void mixed_result_reset(MixedResult *result, const char *status)
{
    result->fp32_time_sec = NAN;
    result->fp64_time_sec = NAN;
    result->total_time_sec = NAN;
    result->residual = NAN;
    result->fp32_fixed_point_residual = NAN;
    result->fp64_fixed_point_residual = NAN;
    result->fp32_iterations = 0;
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

    if (elements > (size_t)INT_MAX) {
        snprintf(error, error_size, "residual vector too large for cuBLAS");
        return -1;
    }

    {
        cudaError_t cuda_status =
            cudaMalloc((void **)&d_residual, elements * sizeof(double));
        if (cuda_status != cudaSuccess) {
            snprintf(error, error_size, "cudaMalloc residual failed: %s",
                     cuda_status_name(cuda_status));
            return -1;
        }
    }

    {
        cublasStatus_t cublas_status =
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &one,
                        d_matrix, n, d_vectors, n, &zero, d_residual, n);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            snprintf(error, error_size, "residual cublasDgemm failed: %s",
                     cublas_status_name_local(cublas_status));
            status = -1;
            goto cleanup;
        }
    }

    {
        int block = 256;
        int grid = ((int)elements + block - 1) / block;
        cudaError_t cuda_status;

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

    {
        cublasStatus_t cublas_status =
            cublasDnrm2(handle, (int)elements, d_residual, 1, residual_out);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            snprintf(error, error_size, "residual cublasDnrm2 failed: %s",
                     cublas_status_name_local(cublas_status));
            status = -1;
            goto cleanup;
        }
    }

    {
        cudaError_t cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            snprintf(error, error_size, "residual synchronize failed: %s",
                     cuda_status_name(cuda_status));
            status = -1;
            goto cleanup;
        }
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

static MixedResult run_mixed(cublasHandle_t blas_handle,
                             const double *d_matrix, const Config *cfg,
                             int compute_residuals)
{
    MixedResult result;
    IPTMixedFP32CudaDeviceResult device_result;
    int status = IPT_CUDA_SUCCESS;
    cudaError_t cuda_status;

    mixed_result_reset(&result, "ok");
    ipt_mixed_fp32_reset_device_result(&device_result);

    status = ipt_mixed_precision_fp32_cuda_device_tol(
        d_matrix, cfg->n, cfg->n, cfg->fp32_tol, cfg->fp32_maxiter,
        cfg->fp64_tol, cfg->fp64_maxiter, blas_handle, &device_result);
    if (status != IPT_CUDA_SUCCESS) {
        mixed_result_fail(&result, "failed", ipt_cuda_status_string(status));
        goto cleanup;
    }

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        mixed_result_fail(&result, "failed", cuda_status_name(cuda_status));
        goto cleanup;
    }

    result.fp32_time_sec = device_result.fp32_time_sec;
    result.fp64_time_sec = device_result.fp64_time_sec;
    result.total_time_sec = device_result.total_time_sec;
    result.fp32_iterations = device_result.fp32_iterations;
    result.fp64_iterations = device_result.fp64_iterations;
    result.fp32_fixed_point_residual =
        device_result.fp32_fixed_point_residual;
    result.fp64_fixed_point_residual =
        device_result.fp64_fixed_point_residual;

    if (compute_residuals) {
        if (residual_norm_gpu(blas_handle, d_matrix, device_result.d_vectors,
                              device_result.d_values, cfg->n, cfg->n,
                              &result.residual, result.error,
                              sizeof(result.error)) != 0) {
            snprintf(result.status, sizeof(result.status), "failed");
        }
    }

cleanup:
    ipt_mixed_precision_fp32_cuda_free_device_result(&device_result);
    return result;
}

static MixedResult run_mixed_average(cublasHandle_t blas_handle,
                                     const double *d_matrix,
                                     const Config *cfg, double epsilon)
{
    MixedResult average;
    double total_fp32 = 0.0;
    double total_fp64 = 0.0;
    double total_all = 0.0;
    int successful_runs = 0;

    mixed_result_reset(&average, "ok");

    printf("Mixed precision warmup for N=%d epsilon=%.17g\n", cfg->n,
           epsilon);
    {
        MixedResult warm = run_mixed(blas_handle, d_matrix, cfg, 0);
        if (strcmp(warm.status, "ok") != 0) {
            mixed_result_fail(&average, "failed", warm.error);
            average.fp32_time_sec = warm.fp32_time_sec;
            average.fp64_time_sec = warm.fp64_time_sec;
            average.total_time_sec = warm.total_time_sec;
            return average;
        }
    }

    for (int rep = 1; rep <= cfg->repeats; ++rep) {
        MixedResult one =
            run_mixed(blas_handle, d_matrix, cfg,
                      cfg->compute_residuals && rep == cfg->repeats);

        printf("  mixed run %d/%d: fp32 = %.8g s, fp64 = %.8g s, "
               "total = %.8g s, fp32_fp = %.8g, fp64_fp = %.8g, "
               "fp32_iters = %d, fp64_iters = %d, residual = %.8g, "
               "status = %s",
               rep, cfg->repeats, one.fp32_time_sec, one.fp64_time_sec,
               one.total_time_sec, one.fp32_fixed_point_residual,
               one.fp64_fixed_point_residual, one.fp32_iterations,
               one.fp64_iterations, one.residual, one.status);
        if (one.error[0] != '\0') {
            printf(", error = %s", one.error);
        }
        printf("\n");

        if (strcmp(one.status, "ok") != 0) {
            average = one;
            if (successful_runs > 0) {
                average.fp32_time_sec =
                    total_fp32 / (double)successful_runs;
                average.fp64_time_sec =
                    total_fp64 / (double)successful_runs;
                average.total_time_sec =
                    total_all / (double)successful_runs;
            }
            return average;
        }

        total_fp32 += one.fp32_time_sec;
        total_fp64 += one.fp64_time_sec;
        total_all += one.total_time_sec;
        successful_runs += 1;

        if (rep == cfg->repeats) {
            average.residual = one.residual;
            average.fp32_fixed_point_residual =
                one.fp32_fixed_point_residual;
            average.fp64_fixed_point_residual =
                one.fp64_fixed_point_residual;
            average.fp32_iterations = one.fp32_iterations;
            average.fp64_iterations = one.fp64_iterations;
        }
    }

    average.fp32_time_sec = total_fp32 / (double)successful_runs;
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

static const char *bool_text(int value)
{
    return value ? "true" : "false";
}

static void write_header(const Config *cfg)
{
    FILE *fp = fopen(cfg->result_csv, "w");

    if (fp == NULL) {
        fprintf(stderr, "could not open %s: %s\n", cfg->result_csv,
                strerror(errno));
        exit(2);
    }

    fprintf(fp,
            "N,epsilon,fp32_tol,fp64_tol,fp32_maxiter,fp64_maxiter,"
            "repeats,time_fp32_initial_sec,time_fp64_refine_sec,"
            "time_total_sec,final_residual,fp32_fixed_point_residual,"
            "fp64_fixed_point_residual,fp32_iterations,fp64_iterations,"
            "fp32_converged,fp64_converged,gpu_name,status,error\n");
    fclose(fp);
}

static void append_row(const Config *cfg, double epsilon,
                       const MixedResult *result, const char *gpu_name,
                       const char *status, const char *error)
{
    FILE *fp = fopen(cfg->result_csv, "a");
    char gpu_clean[256];
    char status_clean[64];
    char error_clean[1024];
    int fp32_converged = isfinite(result->fp32_fixed_point_residual) &&
                         result->fp32_fixed_point_residual <= cfg->fp32_tol;
    int fp64_converged = isfinite(result->fp64_fixed_point_residual) &&
                         result->fp64_fixed_point_residual <= cfg->fp64_tol;

    if (fp == NULL) {
        fprintf(stderr, "could not append %s: %s\n", cfg->result_csv,
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
            "%d,%.17g,%.17g,%.17g,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%d,%d,%s,%s,%s,%s,%s\n",
            cfg->n, epsilon, cfg->fp32_tol, cfg->fp64_tol,
            cfg->fp32_maxiter, cfg->fp64_maxiter, cfg->repeats,
            result->fp32_time_sec, result->fp64_time_sec,
            result->total_time_sec, result->residual,
            result->fp32_fixed_point_residual,
            result->fp64_fixed_point_residual, result->fp32_iterations,
            result->fp64_iterations, bool_text(fp32_converged),
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

    fprintf(fp, "IPT_C mixed precision benchmark: FP32 initial + FP64 refine\n");
    fprintf(fp, "started_at = %s\n", time_buf);
    fprintf(fp, "ipt_c_root = %s\n", cfg->root);
    fprintf(fp, "result_csv = %s\n", cfg->result_csv);
    fprintf(fp, "N = %d\n", cfg->n);
    fprintf(fp, "warmup_n = %d\n", cfg->warmup_n);
    fprintf(fp, "fp32_tol = %.17g\n", cfg->fp32_tol);
    fprintf(fp, "fp32_maxiter = %d\n", cfg->fp32_maxiter);
    fprintf(fp, "fp64_tol = %.17g\n", cfg->fp64_tol);
    fprintf(fp, "fp64_maxiter = %d\n", cfg->fp64_maxiter);
    fprintf(fp, "timing_repeats = %d\n", cfg->repeats);
    fprintf(fp, "per_epsilon_warmup = true\n");
    fprintf(fp, "seed = %llu\n", (unsigned long long)cfg->seed);
    fprintf(fp, "epsilon_values = ");
    for (int i = 0; i < cfg->eps_count; ++i) {
        fprintf(fp, "%s%.17g", i == 0 ? "" : ", ", cfg->eps_values[i]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "compute_residuals = %s\n",
            cfg->compute_residuals ? "true" : "false");
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

static void warmup(cublasHandle_t blas_handle, const Config *cfg)
{
    Config warm_cfg = *cfg;
    double *h_matrix = NULL;
    double *d_matrix = NULL;
    char error[ERROR_LEN] = "";

    warm_cfg.n = cfg->warmup_n;
    printf("===== device warmup N=%d =====\n", warm_cfg.n);

    if (make_host_matrix(&h_matrix, warm_cfg.n, cfg->eps_values[0],
                         cfg->seed + 17) != 0) {
        printf("warmup host matrix allocation failed\n");
        return;
    }
    if (copy_matrix_to_device(&d_matrix, h_matrix, warm_cfg.n, error,
                              sizeof(error)) != 0) {
        printf("warmup device copy failed: %s\n", error);
        free(h_matrix);
        return;
    }

    (void)run_mixed(blas_handle, d_matrix, &warm_cfg, 0);

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
    char gpu_name[256] = "unknown_gpu";
    int failures = 0;
    cudaError_t cuda_status;
    cublasStatus_t blas_status;

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
    (void)cublasSetMathMode(blas_handle, CUBLAS_DEFAULT_MATH);

    write_header(&cfg);
    write_summary_header(&cfg, gpu_name);

    printf("===== IPT_C mixed precision benchmark =====\n");
    printf("ROOT = %s\n", cfg.root);
    printf("RESULT_DIR = %s\n", cfg.results_dir);
    printf("RESULT_CSV = %s\n", cfg.result_csv);
    printf("SUMMARY_TXT = %s\n", cfg.summary_txt);
    printf("N = %d\n", cfg.n);
    printf("fp32_tol = %.17g\n", cfg.fp32_tol);
    printf("fp32_maxiter = %d\n", cfg.fp32_maxiter);
    printf("fp64_tol = %.17g\n", cfg.fp64_tol);
    printf("fp64_maxiter = %d\n", cfg.fp64_maxiter);
    printf("timing_repeats = %d\n", cfg.repeats);
    printf("seed = %llu\n", (unsigned long long)cfg.seed);
    printf("epsilon values = ");
    for (int i = 0; i < cfg.eps_count; ++i) {
        printf("%s%.17g", i == 0 ? "" : ", ", cfg.eps_values[i]);
    }
    printf("\n");
    printf("compute_residuals = %s\n",
           cfg.compute_residuals ? "true" : "false");
    printf("gpu_name = %s\n", gpu_name);
    show_gpu_memory("at start");

    warmup(blas_handle, &cfg);

    for (int idx = 0; idx < cfg.eps_count; ++idx) {
        double epsilon = cfg.eps_values[idx];
        double *h_matrix = NULL;
        double *d_matrix = NULL;
        MixedResult mixed;
        char status[64] = "ok";
        char error[1024] = "";
        char summary_line[1536];
        char time_buf[64];
        int fp64_converged = 0;
        double matrix_start = 0.0;

        mixed_result_reset(&mixed, "not_run");

        printf("\n===== N = %d, epsilon = %.17g =====\n", cfg.n, epsilon);
        wall_time_string(time_buf, sizeof(time_buf));
        snprintf(summary_line, sizeof(summary_line),
                 "N=%d epsilon=%.17g started_at=%s", cfg.n, epsilon,
                 time_buf);
        append_summary(&cfg, summary_line);
        show_gpu_memory("before matrix");

        matrix_start = now_seconds();
        if (make_host_matrix(&h_matrix, cfg.n, epsilon,
                             cfg.seed + (uint64_t)cfg.n +
                                 (uint64_t)idx * 1000003ULL) != 0) {
            snprintf(status, sizeof(status), "failed");
            snprintf(error, sizeof(error), "host matrix allocation failed");
            failures++;
            goto row_done;
        }
        printf("host matrix generated in %.8g s\n",
               now_seconds() - matrix_start);

        if (copy_matrix_to_device(&d_matrix, h_matrix, cfg.n, error,
                                  sizeof(error)) != 0) {
            snprintf(status, sizeof(status), "failed");
            failures++;
            goto row_done;
        }

        free(h_matrix);
        h_matrix = NULL;
        show_gpu_memory("after matrix");

        mixed = run_mixed_average(blas_handle, d_matrix, &cfg, epsilon);
        fp64_converged = isfinite(mixed.fp64_fixed_point_residual) &&
                         mixed.fp64_fixed_point_residual <= cfg.fp64_tol;
        if (strcmp(mixed.status, "ok") != 0) {
            snprintf(status, sizeof(status), "failed");
            snprintf(error, sizeof(error), "%s", mixed.error);
            failures++;
        } else if (!fp64_converged) {
            snprintf(status, sizeof(status), "not_converged");
            snprintf(error, sizeof(error),
                     "fp64_fixed_point_residual %.17g > tol %.17g",
                     mixed.fp64_fixed_point_residual, cfg.fp64_tol);
        }

        printf("Mixed average: fp32 = %.8g s, fp64 = %.8g s, "
               "total = %.8g s, residual = %.8g, fp32_fp = %.8g, "
               "fp64_fp = %.8g, fp32_iters = %d, fp64_iters = %d, "
               "status = %s\n",
               mixed.fp32_time_sec, mixed.fp64_time_sec,
               mixed.total_time_sec, mixed.residual,
               mixed.fp32_fixed_point_residual,
               mixed.fp64_fixed_point_residual, mixed.fp32_iterations,
               mixed.fp64_iterations, status);

    row_done:
        append_row(&cfg, epsilon, &mixed, gpu_name, status, error);
        wall_time_string(time_buf, sizeof(time_buf));
        snprintf(summary_line, sizeof(summary_line),
                 "N=%d epsilon=%.17g finished_at=%s status=%s "
                 "time_fp32=%.17g time_fp64=%.17g time_total=%.17g "
                 "fp32_iterations=%d fp64_iterations=%d fp32_fp=%.17g "
                 "fp64_fp=%.17g final_residual=%.17g repeats=%d "
                 "per_epsilon_warmup=true",
                 cfg.n, epsilon, time_buf, status, mixed.fp32_time_sec,
                 mixed.fp64_time_sec, mixed.total_time_sec,
                 mixed.fp32_iterations, mixed.fp64_iterations,
                 mixed.fp32_fixed_point_residual,
                 mixed.fp64_fixed_point_residual, mixed.residual,
                 cfg.repeats);
        append_summary(&cfg, summary_line);
        cudaFree(d_matrix);
        free(h_matrix);
        cudaDeviceSynchronize();
        show_gpu_memory("after cleanup");
    }

    {
        char time_buf[64];
        char summary_line[128];

        wall_time_string(time_buf, sizeof(time_buf));
        snprintf(summary_line, sizeof(summary_line), "finished_at = %s",
                 time_buf);
        append_summary(&cfg, summary_line);
    }

    printf("===== mixed benchmark done =====\n");
    printf("Results: %s\n", cfg.result_csv);
    printf("Summary: %s\n", cfg.summary_txt);

    cublasDestroy(blas_handle);
    return failures == 0 ? 0 : 1;
}
