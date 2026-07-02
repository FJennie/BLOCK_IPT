#include <cublas_v2.h>
#include <cuda_runtime.h>

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

#include "../../src/mixed_precision/ipt_rayleigh_mixed_cuda.cu"

#define DEFAULT_ROOT "/fs1/home/nudt_liujie/ftt/IPT_C_GPU"
#define DEFAULT_N 16384
#define DEFAULT_EPSILON 0.07
#define DEFAULT_TF32_TOL 1.0e-6
#define DEFAULT_TF32_MAXITER 1000
#define DEFAULT_FP64_TOL 1.0e-12
#define DEFAULT_FP64_MAXITER 1000
#define DEFAULT_REPEATS 1
#define DEFAULT_WARMUP_N 128
#define DEFAULT_SEED 20260613ULL
#define ERROR_LEN 512

typedef struct {
    double time_sec;
    double residual;
    double fixed_point_residual;
    int iterations;
    char status[32];
    char error[ERROR_LEN];
} OriginalResult;

typedef struct {
    double tf32_time_sec;
    double fp64_time_sec;
    double total_time_sec;
    double residual;
    double tf32_fixed_point_residual;
    double fp64_fixed_point_residual;
    int tf32_iterations;
    int fp64_iterations;
    char status[32];
    char error[ERROR_LEN];
} DirectResult;

typedef struct {
    double tf32_time_sec;
    double orthogonalize_time_sec;
    double rayleigh_transform_time_sec;
    double fp64_time_sec;
    double backtransform_time_sec;
    double total_time_sec;
    double residual;
    double tf32_fixed_point_residual;
    double fp64_fixed_point_residual;
    int tf32_iterations;
    int fp64_iterations;
    char status[32];
    char error[ERROR_LEN];
} RayleighResult;

typedef struct {
    const char *root;
    const char *results_dir;
    const char *logs_dir;
    int n;
    double epsilon;
    double tf32_tol;
    int tf32_maxiter;
    double fp64_tol;
    int fp64_maxiter;
    int repeats;
    int warmup_n;
    uint64_t seed;
    int compute_residuals;
} Config;

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

static void load_config(Config *cfg)
{
    memset(cfg, 0, sizeof(*cfg));
    cfg->root = env_or_default("IPT_C_ROOT", DEFAULT_ROOT);
    cfg->results_dir = env_or_default("IPT_C_RESULTS_DIR", NULL);
    cfg->logs_dir = env_or_default("IPT_C_LOG_DIR", NULL);
    if (cfg->results_dir == NULL) {
        static char default_results[4096];
        snprintf(default_results, sizeof(default_results),
                 "%s/results/rayleigh_mixed", cfg->root);
        cfg->results_dir = default_results;
    }
    if (cfg->logs_dir == NULL) {
        static char default_logs[4096];
        snprintf(default_logs, sizeof(default_logs), "%s/logs/rayleigh_mixed",
                 cfg->root);
        cfg->logs_dir = default_logs;
    }

    cfg->n = getenv("IPT_N") ? atoi(getenv("IPT_N")) : DEFAULT_N;
    cfg->epsilon = getenv("IPT_EPSILON") ? atof(getenv("IPT_EPSILON"))
                                         : DEFAULT_EPSILON;
    cfg->tf32_tol = getenv("IPT_TF32_TOL") ? atof(getenv("IPT_TF32_TOL"))
                                           : DEFAULT_TF32_TOL;
    cfg->tf32_maxiter = getenv("IPT_TF32_MAXITER")
                            ? atoi(getenv("IPT_TF32_MAXITER"))
                            : DEFAULT_TF32_MAXITER;
    cfg->fp64_tol = getenv("IPT_TOL") ? atof(getenv("IPT_TOL"))
                                      : DEFAULT_FP64_TOL;
    cfg->fp64_maxiter = getenv("IPT_MAXITER") ? atoi(getenv("IPT_MAXITER"))
                                              : DEFAULT_FP64_MAXITER;
    cfg->repeats =
        getenv("IPT_REPEATS") ? atoi(getenv("IPT_REPEATS")) : DEFAULT_REPEATS;
    cfg->warmup_n = getenv("IPT_WARMUP_N") ? atoi(getenv("IPT_WARMUP_N"))
                                           : DEFAULT_WARMUP_N;
    cfg->seed =
        getenv("IPT_SEED") ? strtoull(getenv("IPT_SEED"), NULL, 10)
                           : DEFAULT_SEED;
    cfg->compute_residuals = parse_bool_env("IPT_COMPUTE_RESIDUALS", 1);

    if (cfg->n <= 0) {
        cfg->n = DEFAULT_N;
    }
    if (cfg->tf32_tol <= 0.0) {
        cfg->tf32_tol = DEFAULT_TF32_TOL;
    }
    if (cfg->tf32_maxiter <= 0) {
        cfg->tf32_maxiter = DEFAULT_TF32_MAXITER;
    }
    if (cfg->fp64_tol <= 0.0) {
        cfg->fp64_tol = DEFAULT_FP64_TOL;
    }
    if (cfg->fp64_maxiter <= 0) {
        cfg->fp64_maxiter = DEFAULT_FP64_MAXITER;
    }
    if (cfg->repeats <= 0) {
        cfg->repeats = DEFAULT_REPEATS;
    }
    if (cfg->warmup_n <= 0) {
        cfg->warmup_n = DEFAULT_WARMUP_N;
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

__global__ static void rayleigh_subtract_vdiag_kernel(double *residual,
                                                      const double *vectors,
                                                      const double *values,
                                                      int n, int k)
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

    {
        int block = 256;
        int grid = ((int)elements + block - 1) / block;

        rayleigh_subtract_vdiag_kernel<<<grid, block>>>(
            d_residual, d_vectors, d_values, n, k);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            snprintf(error, error_size, "residual kernel failed: %s",
                     cudaGetErrorString(cuda_status));
            status = -1;
            goto cleanup;
        }
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

static void original_result_reset(OriginalResult *result, const char *status)
{
    result->time_sec = NAN;
    result->residual = NAN;
    result->fixed_point_residual = NAN;
    result->iterations = 0;
    snprintf(result->status, sizeof(result->status), "%s", status);
    result->error[0] = '\0';
}

static void direct_result_reset(DirectResult *result, const char *status)
{
    result->tf32_time_sec = NAN;
    result->fp64_time_sec = NAN;
    result->total_time_sec = NAN;
    result->residual = NAN;
    result->tf32_fixed_point_residual = NAN;
    result->fp64_fixed_point_residual = NAN;
    result->tf32_iterations = 0;
    result->fp64_iterations = 0;
    snprintf(result->status, sizeof(result->status), "%s", status);
    result->error[0] = '\0';
}

static void rayleigh_result_reset(RayleighResult *result, const char *status)
{
    result->tf32_time_sec = NAN;
    result->orthogonalize_time_sec = NAN;
    result->rayleigh_transform_time_sec = NAN;
    result->fp64_time_sec = NAN;
    result->backtransform_time_sec = NAN;
    result->total_time_sec = NAN;
    result->residual = NAN;
    result->tf32_fixed_point_residual = NAN;
    result->fp64_fixed_point_residual = NAN;
    result->tf32_iterations = 0;
    result->fp64_iterations = 0;
    snprintf(result->status, sizeof(result->status), "%s", status);
    result->error[0] = '\0';
}

static OriginalResult run_original(cublasHandle_t handle,
                                   const double *d_matrix, int n,
                                   const Config *cfg, int compute_residual)
{
    OriginalResult result;
    double *d_vectors = NULL;
    double *d_values = NULL;
    int ipt_status = IPT_CUDA_SUCCESS;
    double start = 0.0;

    original_result_reset(&result, "ok");
    start = now_seconds();
    ipt_status = ipt_cuda_device_tol(d_matrix, n, n, cfg->fp64_tol,
                                     cfg->fp64_maxiter, handle, &d_vectors,
                                     &d_values, &result.iterations,
                                     &result.fixed_point_residual);
    if (ipt_status != IPT_CUDA_SUCCESS) {
        snprintf(result.status, sizeof(result.status), "failed");
        snprintf(result.error, sizeof(result.error), "%s",
                 ipt_cuda_status_string(ipt_status));
        goto cleanup;
    }
    cudaDeviceSynchronize();
    result.time_sec = now_seconds() - start;

    if (compute_residual &&
        residual_norm_gpu(handle, d_matrix, d_vectors, d_values, n, n,
                          &result.residual, result.error,
                          sizeof(result.error)) != 0) {
        snprintf(result.status, sizeof(result.status), "failed");
    }

cleanup:
    cudaFree(d_vectors);
    cudaFree(d_values);
    return result;
}

static DirectResult run_direct_tf32(cublasHandle_t handle,
                                    const double *d_matrix, int n,
                                    const Config *cfg, int compute_residual)
{
    DirectResult result;
    IPTMixedTF32CudaDeviceResult device_result;
    int ipt_status = IPT_CUDA_SUCCESS;

    direct_result_reset(&result, "ok");
    ipt_mixed_tf32_reset_device_result(&device_result);
    ipt_status = ipt_mixed_precision_tf32_cuda_device_tol(
        d_matrix, n, n, cfg->tf32_tol, cfg->tf32_maxiter, cfg->fp64_tol,
        cfg->fp64_maxiter, handle, &device_result);
    if (ipt_status != IPT_CUDA_SUCCESS) {
        snprintf(result.status, sizeof(result.status), "failed");
        snprintf(result.error, sizeof(result.error), "%s",
                 ipt_cuda_status_string(ipt_status));
        goto cleanup;
    }
    cudaDeviceSynchronize();

    result.tf32_time_sec = device_result.tf32_time_sec;
    result.fp64_time_sec = device_result.fp64_time_sec;
    result.total_time_sec = device_result.total_time_sec;
    result.tf32_iterations = device_result.tf32_iterations;
    result.fp64_iterations = device_result.fp64_iterations;
    result.tf32_fixed_point_residual =
        device_result.tf32_fixed_point_residual;
    result.fp64_fixed_point_residual =
        device_result.fp64_fixed_point_residual;

    if (compute_residual &&
        residual_norm_gpu(handle, d_matrix, device_result.d_vectors,
                          device_result.d_values, n, n, &result.residual,
                          result.error, sizeof(result.error)) != 0) {
        snprintf(result.status, sizeof(result.status), "failed");
    }

cleanup:
    ipt_mixed_precision_tf32_cuda_free_device_result(&device_result);
    return result;
}

static RayleighResult run_rayleigh_tf32(cublasHandle_t handle,
                                        const double *d_matrix, int n,
                                        const Config *cfg,
                                        int compute_residual)
{
    RayleighResult result;
    IPTRayleighMixedCudaDeviceResult device_result;
    int ipt_status = IPT_CUDA_SUCCESS;

    rayleigh_result_reset(&result, "ok");
    ipt_rayleigh_mixed_reset_device_result(&device_result);
    ipt_status = ipt_rayleigh_mixed_tf32_cuda_device_tol(
        d_matrix, n, n, cfg->tf32_tol, cfg->tf32_maxiter, cfg->fp64_tol,
        cfg->fp64_maxiter, handle, &device_result);
    if (ipt_status != IPT_CUDA_SUCCESS) {
        snprintf(result.status, sizeof(result.status), "failed");
        snprintf(result.error, sizeof(result.error), "%s",
                 ipt_cuda_status_string(ipt_status));
        goto cleanup;
    }
    cudaDeviceSynchronize();

    result.tf32_time_sec = device_result.tf32_time_sec;
    result.orthogonalize_time_sec = device_result.orthogonalize_time_sec;
    result.rayleigh_transform_time_sec =
        device_result.rayleigh_transform_time_sec;
    result.fp64_time_sec = device_result.fp64_time_sec;
    result.backtransform_time_sec = device_result.backtransform_time_sec;
    result.total_time_sec = device_result.total_time_sec;
    result.tf32_iterations = device_result.tf32_iterations;
    result.fp64_iterations = device_result.fp64_iterations;
    result.tf32_fixed_point_residual =
        device_result.tf32_fixed_point_residual;
    result.fp64_fixed_point_residual =
        device_result.fp64_fixed_point_residual;

    if (compute_residual &&
        residual_norm_gpu(handle, d_matrix, device_result.d_vectors,
                          device_result.d_values, n, n, &result.residual,
                          result.error, sizeof(result.error)) != 0) {
        snprintf(result.status, sizeof(result.status), "failed");
    }

cleanup:
    ipt_rayleigh_mixed_cuda_free_device_result(&device_result);
    return result;
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

static void csv_clean(char *text)
{
    for (char *p = text; *p != '\0'; ++p) {
        if (*p == ',' || *p == '\n' || *p == '\r') {
            *p = ';';
        }
    }
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

static void result_csv_path(const Config *cfg, char *path, size_t size)
{
    char label[64];

    epsilon_label(cfg->epsilon, label, sizeof(label));
    snprintf(path, size, "%s/rayleigh_mixed_N%d_epsilon_%s.csv",
             cfg->results_dir, cfg->n, label);
}

static void write_csv(const char *path, const Config *cfg,
                      const char *gpu_name, const OriginalResult *original,
                      const DirectResult *direct,
                      const RayleighResult *rayleigh)
{
    FILE *fp = fopen(path, "w");
    char gpu_clean[256];
    char status_clean[128];
    char error_clean[1024];

    if (fp == NULL) {
        fprintf(stderr, "could not write %s: %s\n", path, strerror(errno));
        return;
    }

    snprintf(gpu_clean, sizeof(gpu_clean), "%s", gpu_name);
    snprintf(status_clean, sizeof(status_clean), "original=%s|direct=%s|rayleigh=%s",
             original->status, direct->status, rayleigh->status);
    snprintf(error_clean, sizeof(error_clean), "original=%s|direct=%s|rayleigh=%s",
             original->error, direct->error, rayleigh->error);
    csv_clean(gpu_clean);
    csv_clean(status_clean);
    csv_clean(error_clean);

    fprintf(fp,
            "N,epsilon,tf32_tol,fp64_tol,repeats,"
            "time_original_ipt_sec,residual_original_ipt,"
            "original_ipt_fixed_point_residual,original_ipt_iterations,"
            "time_direct_tf32_initial_sec,time_direct_fp64_refine_sec,"
            "time_direct_total_sec,residual_direct_tf32,"
            "direct_tf32_fixed_point_residual,"
            "direct_fp64_fixed_point_residual,direct_tf32_iterations,"
            "direct_fp64_iterations,"
            "time_rayleigh_tf32_initial_sec,time_rayleigh_orthogonalize_sec,"
            "time_rayleigh_transform_sec,"
            "time_rayleigh_fp64_refine_sec,time_rayleigh_backtransform_sec,"
            "time_rayleigh_total_sec,residual_rayleigh,"
            "rayleigh_tf32_fixed_point_residual,"
            "rayleigh_fp64_fixed_point_residual,rayleigh_tf32_iterations,"
            "rayleigh_fp64_iterations,gpu_name,status,error\n");

    fprintf(fp,
            "%d,%.12g,%.17g,%.17g,%d,"
            "%.17g,%.17g,%.17g,%d,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d,"
            "%s,%s,%s\n",
            cfg->n, cfg->epsilon, cfg->tf32_tol, cfg->fp64_tol, cfg->repeats,
            original->time_sec, original->residual,
            original->fixed_point_residual, original->iterations,
            direct->tf32_time_sec, direct->fp64_time_sec,
            direct->total_time_sec, direct->residual,
            direct->tf32_fixed_point_residual,
            direct->fp64_fixed_point_residual, direct->tf32_iterations,
            direct->fp64_iterations, rayleigh->tf32_time_sec,
            rayleigh->orthogonalize_time_sec,
            rayleigh->rayleigh_transform_time_sec, rayleigh->fp64_time_sec,
            rayleigh->backtransform_time_sec, rayleigh->total_time_sec,
            rayleigh->residual, rayleigh->tf32_fixed_point_residual,
            rayleigh->fp64_fixed_point_residual, rayleigh->tf32_iterations,
            rayleigh->fp64_iterations, gpu_clean, status_clean, error_clean);

    fclose(fp);
}

static void warmup(cublasHandle_t handle, const Config *cfg)
{
    double *h_matrix = NULL;
    double *d_matrix = NULL;
    char error[ERROR_LEN] = "";
    Config warm_cfg = *cfg;

    warm_cfg.n = cfg->warmup_n;
    warm_cfg.repeats = 1;
    printf("===== warmup N=%d =====\n", warm_cfg.n);
    if (make_host_matrix(&h_matrix, warm_cfg.n, cfg->epsilon,
                         cfg->seed + 17) != 0) {
        printf("warmup host matrix allocation failed\n");
        return;
    }
    if (copy_matrix_to_device(&d_matrix, h_matrix, warm_cfg.n, error,
                              sizeof(error)) != 0) {
        printf("warmup copy failed: %s\n", error);
        free(h_matrix);
        return;
    }

    (void)run_original(handle, d_matrix, warm_cfg.n, &warm_cfg, 0);
    (void)run_direct_tf32(handle, d_matrix, warm_cfg.n, &warm_cfg, 0);
    (void)run_rayleigh_tf32(handle, d_matrix, warm_cfg.n, &warm_cfg, 0);

    cudaFree(d_matrix);
    free(h_matrix);
    cudaDeviceSynchronize();
    printf("===== warmup done =====\n");
}

int main(void)
{
    Config cfg;
    cudaDeviceProp prop;
    cublasHandle_t handle = NULL;
    cublasStatus_t cublas_status;
    cudaError_t cuda_status;
    double *h_matrix = NULL;
    double *d_matrix = NULL;
    OriginalResult original;
    DirectResult direct;
    RayleighResult rayleigh;
    char gpu_name[256] = "unknown_gpu";
    char error[ERROR_LEN] = "";
    char csv_path[4096];
    double matrix_start = 0.0;

    load_config(&cfg);
    ensure_directory(cfg.logs_dir);
    ensure_directory(cfg.results_dir);
    original_result_reset(&original, "not_run");
    direct_result_reset(&direct, "not_run");
    rayleigh_result_reset(&rayleigh, "not_run");

    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n",
                cudaGetErrorString(cuda_status));
        return 2;
    }
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        snprintf(gpu_name, sizeof(gpu_name), "%s", prop.name);
    }
    cublas_status = cublasCreate(&handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed\n");
        return 2;
    }
    (void)cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    printf("===== rayleigh mixed experiment =====\n");
    printf("N=%d epsilon=%.12g tf32_tol=%.17g fp64_tol=%.17g repeats=%d\n",
           cfg.n, cfg.epsilon, cfg.tf32_tol, cfg.fp64_tol, cfg.repeats);
    printf("results_dir=%s logs_dir=%s gpu=%s\n", cfg.results_dir,
           cfg.logs_dir, gpu_name);
    show_gpu_memory("at start");

    warmup(handle, &cfg);

    matrix_start = now_seconds();
    if (make_host_matrix(&h_matrix, cfg.n, cfg.epsilon,
                         cfg.seed + (uint64_t)cfg.n +
                             epsilon_seed_offset(cfg.epsilon)) != 0) {
        fprintf(stderr, "host matrix allocation failed\n");
        cublasDestroy(handle);
        return 2;
    }
    printf("host matrix generated in %.8g s\n", now_seconds() - matrix_start);

    if (copy_matrix_to_device(&d_matrix, h_matrix, cfg.n, error,
                              sizeof(error)) != 0) {
        fprintf(stderr, "%s\n", error);
        free(h_matrix);
        cublasDestroy(handle);
        return 2;
    }
    free(h_matrix);
    h_matrix = NULL;
    show_gpu_memory("after matrix");

    for (int rep = 1; rep <= cfg.repeats; ++rep) {
        int compute_residual = cfg.compute_residuals && rep == cfg.repeats;

        printf("Original IPT run %d/%d\n", rep, cfg.repeats);
        original = run_original(handle, d_matrix, cfg.n, &cfg, compute_residual);
        printf("  original: time=%.8g fp=%.8g iters=%d residual=%.8g status=%s\n",
               original.time_sec, original.fixed_point_residual,
               original.iterations, original.residual, original.status);

        printf("Direct TF32->FP64 run %d/%d\n", rep, cfg.repeats);
        direct = run_direct_tf32(handle, d_matrix, cfg.n, &cfg,
                                 compute_residual);
        printf("  direct: tf32=%.8g fp64=%.8g total=%.8g tf32_fp=%.8g "
               "fp64_fp=%.8g iters=%d/%d residual=%.8g status=%s\n",
               direct.tf32_time_sec, direct.fp64_time_sec,
               direct.total_time_sec, direct.tf32_fixed_point_residual,
               direct.fp64_fixed_point_residual, direct.tf32_iterations,
               direct.fp64_iterations, direct.residual, direct.status);

        printf("Rayleigh mixed run %d/%d\n", rep, cfg.repeats);
        rayleigh = run_rayleigh_tf32(handle, d_matrix, cfg.n, &cfg,
                                     compute_residual);
        printf("  rayleigh: tf32=%.8g orth=%.8g transform=%.8g fp64=%.8g "
               "back=%.8g total=%.8g tf32_fp=%.8g fp64_fp=%.8g "
               "iters=%d/%d residual=%.8g status=%s\n",
               rayleigh.tf32_time_sec, rayleigh.orthogonalize_time_sec,
               rayleigh.rayleigh_transform_time_sec, rayleigh.fp64_time_sec,
               rayleigh.backtransform_time_sec, rayleigh.total_time_sec,
               rayleigh.tf32_fixed_point_residual,
               rayleigh.fp64_fixed_point_residual,
               rayleigh.tf32_iterations, rayleigh.fp64_iterations,
               rayleigh.residual, rayleigh.status);
    }

    result_csv_path(&cfg, csv_path, sizeof(csv_path));
    write_csv(csv_path, &cfg, gpu_name, &original, &direct, &rayleigh);
    printf("wrote %s\n", csv_path);
    show_gpu_memory("before cleanup");

    cudaFree(d_matrix);
    cublasDestroy(handle);
    return 0;
}
