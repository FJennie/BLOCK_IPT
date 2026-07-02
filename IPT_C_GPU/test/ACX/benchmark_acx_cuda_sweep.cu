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

#include "../../src/ACX/acx_cuda.cu"
#include "../../src/ipt_cuda.cu"

#define DEFAULT_ROOT "/fs1/home/nudt_liujie/ftt/IPT_C_GPU"
#define DEFAULT_NS "4096"
#define DEFAULT_EPS_VALUES "0.06"
#define DEFAULT_TOL 1.0e-12
#define DEFAULT_MAXITER 1000
#define DEFAULT_REPEATS 1
#define DEFAULT_WARMUP_N 128
#define DEFAULT_SEED 20260615ULL
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
    double tol;
    int maxiter;
    int repeats;
    int warmup_n;
    uint64_t seed;
    int append_results;
    int compute_residual;
    char summary_txt[4096];
} Config;

typedef struct {
    double time_sec;
    double iteration_time_sec;
    double finalization_time_sec;
    double absolute_residual;
    double max_apply_residual;
    double fixed_point_residual;
    int iterations;
    int f_calls;
    int converged;
    char status[32];
    char error[ERROR_LEN];
} RunResult;

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

    if (value == NULL || value[0] == '\0') {
        value = fallback_name == NULL ? NULL : getenv(fallback_name);
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
    const char *summary = NULL;

    memset(cfg, 0, sizeof(*cfg));
    cfg->root = env_or_default("IPT_C_ROOT", NULL, DEFAULT_ROOT);
    cfg->results_dir = env_or_default("ACX_RESULTS_DIR", "IPT_C_RESULTS_DIR",
                                      NULL);
    cfg->logs_dir = env_or_default("ACX_LOG_DIR", "IPT_C_LOG_DIR", NULL);

    if (cfg->results_dir == NULL) {
        static char default_results[4096];
        snprintf(default_results, sizeof(default_results), "%s/results/ACX",
                 cfg->root);
        cfg->results_dir = default_results;
    }
    if (cfg->logs_dir == NULL) {
        static char default_logs[4096];
        snprintf(default_logs, sizeof(default_logs), "%s/logs/ACX",
                 cfg->root);
        cfg->logs_dir = default_logs;
    }

    parse_int_list(env_or_default("ACX_SWEEP_NS", "IPT_SWEEP_NS", DEFAULT_NS),
                   cfg->ns, &cfg->n_count);
    parse_double_list(env_or_default("ACX_EPSILON_VALUES",
                                     "IPT_EPSILON_VALUES",
                                     DEFAULT_EPS_VALUES),
                      cfg->eps_values, &cfg->eps_count);

    cfg->tol = getenv("ACX_TOL") ? atof(getenv("ACX_TOL")) : DEFAULT_TOL;
    cfg->maxiter = getenv("ACX_MAXITER") ? atoi(getenv("ACX_MAXITER"))
                                         : DEFAULT_MAXITER;
    cfg->repeats = getenv("ACX_REPEATS") ? atoi(getenv("ACX_REPEATS"))
                                         : DEFAULT_REPEATS;
    cfg->warmup_n = getenv("ACX_WARMUP_N") ? atoi(getenv("ACX_WARMUP_N"))
                                           : DEFAULT_WARMUP_N;
    cfg->seed = getenv("ACX_SEED") ? strtoull(getenv("ACX_SEED"), NULL, 10)
                                   : DEFAULT_SEED;
    cfg->append_results = parse_bool_env("ACX_APPEND_RESULTS", 0);
    cfg->compute_residual = parse_bool_env("ACX_COMPUTE_RESIDUAL", 1);

    if (cfg->tol <= 0.0) {
        cfg->tol = DEFAULT_TOL;
    }
    if (cfg->maxiter <= 0) {
        cfg->maxiter = DEFAULT_MAXITER;
    }
    if (cfg->repeats <= 0) {
        cfg->repeats = DEFAULT_REPEATS;
    }
    if (cfg->warmup_n < 0) {
        cfg->warmup_n = DEFAULT_WARMUP_N;
    }

    summary = getenv("ACX_SUMMARY_TXT");
    if (summary != NULL && summary[0] != '\0') {
        snprintf(cfg->summary_txt, sizeof(cfg->summary_txt), "%s", summary);
    } else {
        snprintf(cfg->summary_txt, sizeof(cfg->summary_txt),
                 "%s/acx_cuda_sweep_summary.txt", cfg->results_dir);
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
    cudaError_t cuda_status;
    cublasStatus_t cublas_status;
    int block = 256;
    int grid = (int)((elements + (size_t)block - 1) / (size_t)block);

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

static void run_result_reset(RunResult *result, const char *status)
{
    result->time_sec = NAN;
    result->iteration_time_sec = NAN;
    result->finalization_time_sec = NAN;
    result->absolute_residual = NAN;
    result->max_apply_residual = NAN;
    result->fixed_point_residual = NAN;
    result->iterations = 0;
    result->f_calls = 0;
    result->converged = 0;
    snprintf(result->status, sizeof(result->status), "%s", status);
    result->error[0] = '\0';
}

static double f_calls_per_iteration(const RunResult *result)
{
    return (result->iterations > 0)
               ? (double)result->f_calls / (double)result->iterations
               : NAN;
}

static RunResult run_acx_once(cublasHandle_t handle, const double *d_matrix,
                              int n, const Config *cfg, int compute_residual)
{
    RunResult result;
    ACXCudaResult device_result;
    int acx_status = ACX_CUDA_SUCCESS;

    run_result_reset(&result, "ok");
    acx_cuda_reset_result(&device_result);

    acx_status = acx_cuda_device_tol(d_matrix, n, n, cfg->tol, cfg->maxiter,
                                     handle, &device_result);
    if (acx_status != ACX_CUDA_SUCCESS) {
        snprintf(result.status, sizeof(result.status), "failed");
        snprintf(result.error, sizeof(result.error), "%s",
                 acx_cuda_status_string(acx_status));
        goto cleanup;
    }

    result.time_sec = device_result.time_sec;
    result.iteration_time_sec = device_result.iteration_time_sec;
    result.finalization_time_sec = device_result.finalization_time_sec;
    result.max_apply_residual = device_result.max_residual;
    result.fixed_point_residual = device_result.fixed_point_residual;
    result.iterations = device_result.iterations;
    result.f_calls = device_result.f_calls;
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

static RunResult run_acx_average(cublasHandle_t handle, const double *d_matrix,
                                 int n, const Config *cfg)
{
    RunResult average;
    double total_time = 0.0;
    double total_iteration = 0.0;
    double total_finalization = 0.0;
    double total_fixed_point = 0.0;

    run_result_reset(&average, "ok");

    printf("ACX warmup for N=%d\n", n);
    {
        RunResult warm = run_acx_once(handle, d_matrix, n, cfg, 0);
        if (strcmp(warm.status, "ok") != 0) {
            return warm;
        }
    }

    for (int rep = 1; rep <= cfg->repeats; ++rep) {
        int compute_residual = cfg->compute_residual && rep == cfg->repeats;
        RunResult one = run_acx_once(handle, d_matrix, n, cfg,
                                     compute_residual);

        printf("  ACX run %d/%d: total=%.8g s, iter=%.8g s, "
               "finalize=%.8g s, max_apply_res=%.8g, abs_res=%.8g, "
               "fixed_point_res=%.8g, iters=%d, f_calls=%d, "
               "converged=%d, status=%s",
               rep, cfg->repeats, one.time_sec, one.iteration_time_sec,
               one.finalization_time_sec, one.max_apply_residual,
               one.absolute_residual, one.fixed_point_residual,
               one.iterations, one.f_calls, one.converged, one.status);
        if (one.error[0] != '\0') {
            printf(", error=%s", one.error);
        }
        printf("\n");

        if (strcmp(one.status, "ok") != 0) {
            return one;
        }
        total_time += one.time_sec;
        total_iteration += one.iteration_time_sec;
        total_finalization += one.finalization_time_sec;
        total_fixed_point += one.fixed_point_residual;
        average.absolute_residual = one.absolute_residual;
        average.max_apply_residual = one.max_apply_residual;
        average.iterations = one.iterations;
        average.f_calls = one.f_calls;
        average.converged = one.converged;
    }

    average.time_sec = total_time / (double)cfg->repeats;
    average.iteration_time_sec = total_iteration / (double)cfg->repeats;
    average.finalization_time_sec = total_finalization / (double)cfg->repeats;
    average.fixed_point_residual = total_fixed_point / (double)cfg->repeats;
    return average;
}

static RunResult run_original_ipt_once(cublasHandle_t handle,
                                       const double *d_matrix, int n,
                                       const Config *cfg,
                                       int compute_residual)
{
    RunResult result;
    double *d_vectors = NULL;
    double *d_values = NULL;
    int iterations_done = 0;
    double fixed_point_residual = NAN;
    int ipt_status = IPT_CUDA_SUCCESS;
    cudaError_t cuda_status = cudaSuccess;
    double start = 0.0;

    run_result_reset(&result, "ok");

    start = now_seconds();
    ipt_status = ipt_cuda_device_tol(d_matrix, n, n, cfg->tol, cfg->maxiter,
                                     handle, &d_vectors, &d_values,
                                     &iterations_done,
                                     &fixed_point_residual);
    cuda_status = cudaDeviceSynchronize();
    result.time_sec = now_seconds() - start;
    result.iteration_time_sec = result.time_sec;
    result.finalization_time_sec = 0.0;

    if (ipt_status != IPT_CUDA_SUCCESS) {
        snprintf(result.status, sizeof(result.status), "failed");
        snprintf(result.error, sizeof(result.error), "%s",
                 ipt_cuda_status_string(ipt_status));
        goto cleanup;
    }
    if (cuda_status != cudaSuccess) {
        snprintf(result.status, sizeof(result.status), "failed");
        snprintf(result.error, sizeof(result.error), "cuda sync failed: %s",
                 cudaGetErrorString(cuda_status));
        goto cleanup;
    }

    result.iterations = iterations_done;
    result.f_calls = iterations_done;
    result.fixed_point_residual = fixed_point_residual;
    result.converged = isfinite(fixed_point_residual) &&
                       fixed_point_residual <= cfg->tol;

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

static RunResult run_original_ipt_average(cublasHandle_t handle,
                                          const double *d_matrix, int n,
                                          const Config *cfg)
{
    RunResult average;
    double total_time = 0.0;
    double total_fixed_point = 0.0;

    run_result_reset(&average, "ok");

    printf("Original IPT warmup for N=%d\n", n);
    {
        RunResult warm = run_original_ipt_once(handle, d_matrix, n, cfg, 0);
        if (strcmp(warm.status, "ok") != 0) {
            return warm;
        }
    }

    for (int rep = 1; rep <= cfg->repeats; ++rep) {
        int compute_residual = cfg->compute_residual && rep == cfg->repeats;
        RunResult one = run_original_ipt_once(handle, d_matrix, n, cfg,
                                              compute_residual);

        printf("  original IPT run %d/%d: total=%.8g s, "
               "fixed_point_res=%.8g, abs_res=%.8g, iters=%d, "
               "converged=%d, status=%s",
               rep, cfg->repeats, one.time_sec, one.fixed_point_residual,
               one.absolute_residual, one.iterations, one.converged,
               one.status);
        if (one.error[0] != '\0') {
            printf(", error=%s", one.error);
        }
        printf("\n");

        if (strcmp(one.status, "ok") != 0) {
            return one;
        }
        total_time += one.time_sec;
        total_fixed_point += one.fixed_point_residual;
        average.absolute_residual = one.absolute_residual;
        average.max_apply_residual = one.max_apply_residual;
        average.iterations = one.iterations;
        average.f_calls = one.f_calls;
        average.converged = one.converged;
    }

    average.time_sec = total_time / (double)cfg->repeats;
    average.iteration_time_sec = average.time_sec;
    average.finalization_time_sec = 0.0;
    average.fixed_point_residual = total_fixed_point / (double)cfg->repeats;
    return average;
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

static void result_csv_path(const Config *cfg, double epsilon, char *path,
                            size_t size)
{
    char label[64];

    epsilon_label(epsilon, label, sizeof(label));
    snprintf(path, size, "%s/acx_cuda_sweep_epsilon_%s.csv",
             cfg->results_dir, label);
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
            "N,epsilon,method,tol,maxiter,repeats,time_total_sec,"
            "time_iteration_sec,time_finalization_sec,absolute_residual,"
            "max_apply_residual,fixed_point_residual,iterations,f_calls,"
            "f_calls_per_iteration,converged,gpu_name,status,error\n");
    fclose(fp);
}

static void append_csv_row(const char *csv_path, const Config *cfg, int n,
                           double epsilon, const char *method,
                           const RunResult *result, const char *gpu_name)
{
    FILE *fp = fopen(csv_path, "a");
    char method_clean[128];
    char gpu_clean[256];
    char status_clean[64];
    char error_clean[ERROR_LEN];

    if (fp == NULL) {
        fprintf(stderr, "could not open %s for append: %s\n", csv_path,
                strerror(errno));
        return;
    }

    csv_escape(method, method_clean, sizeof(method_clean));
    csv_escape(gpu_name, gpu_clean, sizeof(gpu_clean));
    csv_escape(result->status, status_clean, sizeof(status_clean));
    csv_escape(result->error, error_clean, sizeof(error_clean));
    fprintf(fp,
            "%d,%.17g,%s,%.17g,%d,%d,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%d,%d,%.17g,%d,%s,%s,%s\n",
            n, epsilon, method_clean, cfg->tol, cfg->maxiter, cfg->repeats,
            result->time_sec, result->iteration_time_sec,
            result->finalization_time_sec, result->absolute_residual,
            result->max_apply_residual, result->fixed_point_residual,
            result->iterations, result->f_calls,
            f_calls_per_iteration(result), result->converged, gpu_clean,
            status_clean, error_clean);
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

static int reset_summary_if_needed(const Config *cfg)
{
    FILE *fp = NULL;

    if (cfg->append_results) {
        return 0;
    }

    fp = fopen(cfg->summary_txt, "w");
    if (fp == NULL) {
        fprintf(stderr, "could not reset summary %s: %s\n",
                cfg->summary_txt, strerror(errno));
        return -1;
    }
    fclose(fp);
    return 0;
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

int main(void)
{
    Config cfg;
    cudaDeviceProp prop;
    char gpu_name[256] = "unknown";
    cublasHandle_t handle = NULL;
    int failures = 0;
    char time_buf[64];
    char summary_line[4096];

    load_config(&cfg);
    if (ensure_directory(cfg.results_dir) != 0 ||
        ensure_directory(cfg.logs_dir) != 0 ||
        reset_summary_if_needed(&cfg) != 0) {
        return 2;
    }

    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        snprintf(gpu_name, sizeof(gpu_name), "%s", prop.name);
    }
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed\n");
        return 2;
    }

    wall_time_string(time_buf, sizeof(time_buf));
    snprintf(summary_line, sizeof(summary_line), "started_at=%s", time_buf);
    append_summary(&cfg, summary_line);

    printf("===== ACX CUDA sweep =====\n");
    printf("root=%s\nresults=%s\nlogs=%s\nsummary=%s\n", cfg.root,
           cfg.results_dir, cfg.logs_dir, cfg.summary_txt);
    printf("N values = ");
    for (int i = 0; i < cfg.n_count; ++i) {
        printf("%s%d", i == 0 ? "" : ", ", cfg.ns[i]);
    }
    printf("\nepsilon values = ");
    for (int i = 0; i < cfg.eps_count; ++i) {
        printf("%s%.12g", i == 0 ? "" : ", ", cfg.eps_values[i]);
    }
    printf("\ntol=%.17g maxiter=%d repeats=%d warmup_n=%d seed=%llu gpu=%s\n",
           cfg.tol, cfg.maxiter, cfg.repeats, cfg.warmup_n,
           (unsigned long long)cfg.seed, gpu_name);
    show_gpu_memory("at start");

    for (int eps_idx = 0; eps_idx < cfg.eps_count; ++eps_idx) {
        double epsilon = cfg.eps_values[eps_idx];
        char csv_path[4096];

        result_csv_path(&cfg, epsilon, csv_path, sizeof(csv_path));
        if (!cfg.append_results) {
            write_csv_header(csv_path);
        } else {
            FILE *probe = fopen(csv_path, "r");
            if (probe == NULL) {
                write_csv_header(csv_path);
            } else {
                fclose(probe);
            }
        }

        for (int n_idx = 0; n_idx < cfg.n_count; ++n_idx) {
            int n = cfg.ns[n_idx];
            double *h_matrix = NULL;
            double *d_matrix = NULL;
            RunResult acx_result;
            RunResult ipt_result;
            char error[ERROR_LEN] = "";
            double matrix_start = 0.0;

            run_result_reset(&acx_result, "not_run");
            run_result_reset(&ipt_result, "not_run");
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
                snprintf(acx_result.status, sizeof(acx_result.status),
                         "failed");
                snprintf(acx_result.error, sizeof(acx_result.error),
                         "host matrix allocation failed");
                snprintf(ipt_result.status, sizeof(ipt_result.status),
                         "failed");
                snprintf(ipt_result.error, sizeof(ipt_result.error),
                         "host matrix allocation failed");
                failures += 2;
                goto row_done;
            }
            printf("host matrix generated in %.8g s\n",
                   now_seconds() - matrix_start);

            if (copy_matrix_to_device(&d_matrix, h_matrix, n, error,
                                      sizeof(error)) != 0) {
                snprintf(acx_result.status, sizeof(acx_result.status),
                         "failed");
                snprintf(acx_result.error, sizeof(acx_result.error), "%s",
                         error);
                snprintf(ipt_result.status, sizeof(ipt_result.status),
                         "failed");
                snprintf(ipt_result.error, sizeof(ipt_result.error), "%s",
                         error);
                failures += 2;
                goto row_done;
            }
            free(h_matrix);
            h_matrix = NULL;
            show_gpu_memory("after matrix");

            acx_result = run_acx_average(handle, d_matrix, n, &cfg);
            if (strcmp(acx_result.status, "ok") != 0) {
                failures++;
            }

            ipt_result = run_original_ipt_average(handle, d_matrix, n, &cfg);
            if (strcmp(ipt_result.status, "ok") != 0) {
                failures++;
            }

        row_done:
            append_csv_row(csv_path, &cfg, n, epsilon, "acx_gpu_fp64",
                           &acx_result, gpu_name);
            append_csv_row(csv_path, &cfg, n, epsilon,
                           "original_ipt_gpu_fp64", &ipt_result, gpu_name);
            wall_time_string(time_buf, sizeof(time_buf));
            snprintf(summary_line, sizeof(summary_line),
                     "epsilon=%.12g N=%d method=acx_gpu_fp64 finished_at=%s "
                     "status=%s "
                     "time=%.17g iter_time=%.17g final_time=%.17g "
                     "abs_residual=%.17g max_apply_residual=%.17g "
                     "fixed_point_residual=%.17g iterations=%d f_calls=%d "
                     "csv=%s",
                     epsilon, n, time_buf, acx_result.status,
                     acx_result.time_sec, acx_result.iteration_time_sec,
                     acx_result.finalization_time_sec,
                     acx_result.absolute_residual,
                     acx_result.max_apply_residual,
                     acx_result.fixed_point_residual,
                     acx_result.iterations, acx_result.f_calls, csv_path);
            append_summary(&cfg, summary_line);
            snprintf(summary_line, sizeof(summary_line),
                     "epsilon=%.12g N=%d method=original_ipt_gpu_fp64 "
                     "finished_at=%s status=%s time=%.17g "
                     "iter_time=%.17g final_time=%.17g "
                     "abs_residual=%.17g max_apply_residual=%.17g "
                     "fixed_point_residual=%.17g iterations=%d f_calls=%d "
                     "csv=%s",
                     epsilon, n, time_buf, ipt_result.status,
                     ipt_result.time_sec, ipt_result.iteration_time_sec,
                     ipt_result.finalization_time_sec,
                     ipt_result.absolute_residual,
                     ipt_result.max_apply_residual,
                     ipt_result.fixed_point_residual,
                     ipt_result.iterations, ipt_result.f_calls, csv_path);
            append_summary(&cfg, summary_line);
            snprintf(summary_line, sizeof(summary_line),
                     "epsilon=%.12g N=%d f_calls_comparison "
                     "acx_iterations=%d acx_f_calls=%d "
                     "acx_f_calls_per_iteration=%.17g "
                     "original_ipt_iterations=%d original_ipt_f_calls=%d "
                     "original_ipt_f_calls_per_iteration=%.17g "
                     "acx_minus_original_f_calls=%d "
                     "acx_over_original_f_calls=%.17g csv=%s",
                     epsilon, n, acx_result.iterations, acx_result.f_calls,
                     f_calls_per_iteration(&acx_result),
                     ipt_result.iterations, ipt_result.f_calls,
                     f_calls_per_iteration(&ipt_result),
                     acx_result.f_calls - ipt_result.f_calls,
                     (ipt_result.f_calls > 0)
                         ? (double)acx_result.f_calls /
                               (double)ipt_result.f_calls
                         : NAN,
                     csv_path);
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
    cublasDestroy(handle);
    return failures == 0 ? 0 : 1;
}
