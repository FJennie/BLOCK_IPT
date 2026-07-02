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

#include "ipt_rayleigh_mixed_no_qr_cuda.cu"

#define DEFAULT_ROOT "/fs1/home/nudt_liujie/ftt/IPT_C_GPU"
#define DEFAULT_NS "16384"
#define DEFAULT_EPS_VALUES "0.06"
#define DEFAULT_TF32_TOL 1.0e-6
#define DEFAULT_TF32_MAXITER 1000
#define DEFAULT_FP64_TOL 1.0e-12
#define DEFAULT_FP64_MAXITER 1000
#define DEFAULT_REPEATS 5
#define DEFAULT_WARMUP_N 128
#define DEFAULT_SEED 20260613ULL
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
    int tf32_maxiter;
    double fp64_tol;
    int fp64_maxiter;
    int repeats;
    int warmup_n;
    uint64_t seed;
    int append_results;
    int skip_existing;
    char summary_txt[4096];
} Config;

typedef struct {
    double tf32_time_sec;
    double orthogonalize_time_sec;
    double rayleigh_transform_time_sec;
    double fp64_time_sec;
    double backtransform_time_sec;
    double total_time_sec;
    double absolute_residual;
    double tf32_fixed_point_residual;
    double fp64_fixed_point_residual;
    int tf32_iterations;
    int fp64_iterations;
    char status[32];
    char error[ERROR_LEN];
} RayleighRunResult;

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
    const char *summary = NULL;

    memset(cfg, 0, sizeof(*cfg));
    cfg->root = env_or_default("IPT_C_ROOT", DEFAULT_ROOT);
    cfg->results_dir = env_or_default("IPT_C_RESULTS_DIR", NULL);
    cfg->logs_dir = env_or_default("IPT_C_LOG_DIR", NULL);

    if (cfg->results_dir == NULL) {
        static char default_results[4096];
        snprintf(default_results, sizeof(default_results),
                 "%s/no_qr/results", cfg->root);
        cfg->results_dir = default_results;
    }
    if (cfg->logs_dir == NULL) {
        static char default_logs[4096];
        snprintf(default_logs, sizeof(default_logs), "%s/no_qr/logs",
                 cfg->root);
        cfg->logs_dir = default_logs;
    }

    raw_ns = env_or_default("IPT_SWEEP_NS", DEFAULT_NS);
    raw_eps = env_or_default("IPT_EPSILON_VALUES", DEFAULT_EPS_VALUES);
    parse_int_list(raw_ns, cfg->ns, &cfg->n_count);
    parse_double_list(raw_eps, cfg->eps_values, &cfg->eps_count);

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
    cfg->append_results = parse_bool_env("IPT_APPEND_RESULTS", 0);
    cfg->skip_existing = parse_bool_env("IPT_SKIP_EXISTING", 0);

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

    summary = getenv("IPT_SUMMARY_TXT");
    if (summary != NULL && summary[0] != '\0') {
        snprintf(cfg->summary_txt, sizeof(cfg->summary_txt), "%s", summary);
    } else {
        snprintf(cfg->summary_txt, sizeof(cfg->summary_txt),
                 "%s/rayleigh_mixed_no_qr_sweep_summary.txt",
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

__global__ static void sweep_subtract_vdiag_kernel(double *residual,
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

        sweep_subtract_vdiag_kernel<<<grid, block>>>(
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

static void rayleigh_run_result_reset(RayleighRunResult *result,
                                      const char *status)
{
    result->tf32_time_sec = NAN;
    result->orthogonalize_time_sec = NAN;
    result->rayleigh_transform_time_sec = NAN;
    result->fp64_time_sec = NAN;
    result->backtransform_time_sec = NAN;
    result->total_time_sec = NAN;
    result->absolute_residual = NAN;
    result->tf32_fixed_point_residual = NAN;
    result->fp64_fixed_point_residual = NAN;
    result->tf32_iterations = 0;
    result->fp64_iterations = 0;
    snprintf(result->status, sizeof(result->status), "%s", status);
    result->error[0] = '\0';
}

static RayleighRunResult run_rayleigh(cublasHandle_t handle,
                                      const double *d_matrix, int n,
                                      const Config *cfg,
                                      int compute_residual)
{
    RayleighRunResult result;
    IPTRayleighMixedCudaDeviceResult device_result;
    int ipt_status = IPT_CUDA_SUCCESS;

    rayleigh_run_result_reset(&result, "ok");
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
    result.tf32_fixed_point_residual =
        device_result.tf32_fixed_point_residual;
    result.fp64_fixed_point_residual =
        device_result.fp64_fixed_point_residual;
    result.tf32_iterations = device_result.tf32_iterations;
    result.fp64_iterations = device_result.fp64_iterations;

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

static RayleighRunResult run_rayleigh_average(cublasHandle_t handle,
                                              const double *d_matrix, int n,
                                              const Config *cfg)
{
    RayleighRunResult average;
    double total_tf32 = 0.0;
    double total_orth = 0.0;
    double total_transform = 0.0;
    double total_fp64 = 0.0;
    double total_back = 0.0;
    double total_all = 0.0;
    int successful_runs = 0;

    rayleigh_run_result_reset(&average, "ok");
    printf("Rayleigh mixed no-QR warmup for N=%d\n", n);
    {
        RayleighRunResult warm = run_rayleigh(handle, d_matrix, n, cfg, 0);
        if (strcmp(warm.status, "ok") != 0) {
            return warm;
        }
    }

    for (int rep = 1; rep <= cfg->repeats; ++rep) {
        int compute_residual = rep == cfg->repeats;
        RayleighRunResult one =
            run_rayleigh(handle, d_matrix, n, cfg, compute_residual);

        printf("  rayleigh run %d/%d: tf32=%.8g s, orth=%.8g s, "
               "transform=%.8g s, fp64=%.8g s, back=%.8g s, total=%.8g s, "
               "tf32_fp=%.8g, fp64_fp=%.8g, abs_res=%.8g, iters=%d/%d, "
               "status=%s",
               rep, cfg->repeats, one.tf32_time_sec,
               one.orthogonalize_time_sec,
               one.rayleigh_transform_time_sec, one.fp64_time_sec,
               one.backtransform_time_sec, one.total_time_sec,
               one.tf32_fixed_point_residual,
               one.fp64_fixed_point_residual, one.absolute_residual,
               one.tf32_iterations, one.fp64_iterations, one.status);
        if (one.error[0] != '\0') {
            printf(", error=%s", one.error);
        }
        printf("\n");

        if (strcmp(one.status, "ok") != 0) {
            if (successful_runs > 0) {
                one.tf32_time_sec = total_tf32 / (double)successful_runs;
                one.orthogonalize_time_sec =
                    total_orth / (double)successful_runs;
                one.rayleigh_transform_time_sec =
                    total_transform / (double)successful_runs;
                one.fp64_time_sec = total_fp64 / (double)successful_runs;
                one.backtransform_time_sec = total_back / (double)successful_runs;
                one.total_time_sec = total_all / (double)successful_runs;
            }
            return one;
        }

        total_tf32 += one.tf32_time_sec;
        total_orth += one.orthogonalize_time_sec;
        total_transform += one.rayleigh_transform_time_sec;
        total_fp64 += one.fp64_time_sec;
        total_back += one.backtransform_time_sec;
        total_all += one.total_time_sec;
        successful_runs += 1;

        if (rep == cfg->repeats) {
            average.tf32_fixed_point_residual =
                one.tf32_fixed_point_residual;
            average.fp64_fixed_point_residual =
                one.fp64_fixed_point_residual;
            average.absolute_residual = one.absolute_residual;
            average.tf32_iterations = one.tf32_iterations;
            average.fp64_iterations = one.fp64_iterations;
        }
    }

    average.tf32_time_sec = total_tf32 / (double)successful_runs;
    average.orthogonalize_time_sec = total_orth / (double)successful_runs;
    average.rayleigh_transform_time_sec =
        total_transform / (double)successful_runs;
    average.fp64_time_sec = total_fp64 / (double)successful_runs;
    average.backtransform_time_sec = total_back / (double)successful_runs;
    average.total_time_sec = total_all / (double)successful_runs;
    return average;
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
    snprintf(path, size, "%s/rayleigh_mixed_no_qr_sweep_epsilon_%s.csv",
             cfg->results_dir, label);
}

static void write_csv_header(const char *csv_path)
{
    FILE *fp = fopen(csv_path, "w");

    if (fp == NULL) {
        fprintf(stderr, "could not write %s: %s\n", csv_path, strerror(errno));
        exit(2);
    }

    fprintf(fp,
            "N,epsilon,method,tf32_tol,fp64_tol,tf32_maxiter,fp64_maxiter,"
            "repeats,time_tf32_initial_sec,time_orthogonalize_sec,"
            "time_rayleigh_transform_sec,time_fp64_refine_sec,"
            "time_backtransform_sec,time_total_sec,"
            "absolute_residual,tf32_fixed_point_residual,"
            "fp64_fixed_point_residual,"
            "tf32_iterations,fp64_iterations,tf32_converged,fp64_converged,"
            "gpu_name,status,error\n");
    fclose(fp);
}

static void write_csv_header_if_missing(const char *csv_path)
{
    struct stat st;

    if (stat(csv_path, &st) != 0 || st.st_size == 0) {
        write_csv_header(csv_path);
    }
}

static int csv_has_case(const char *csv_path, int n)
{
    FILE *fp = fopen(csv_path, "r");
    char line[8192];

    if (fp == NULL) {
        return 0;
    }
    while (fgets(line, sizeof(line), fp) != NULL) {
        char *saveptr = NULL;
        char *n_text = strtok_r(line, ",", &saveptr);

        if (n_text != NULL && atoi(n_text) == n) {
            fclose(fp);
            return 1;
        }
    }
    fclose(fp);
    return 0;
}

static void append_csv_row(const char *csv_path, const Config *cfg, int n,
                           double epsilon, const RayleighRunResult *result,
                           const char *gpu_name)
{
    FILE *fp = fopen(csv_path, "a");
    char gpu_clean[256];
    char status_clean[64];
    char error_clean[ERROR_LEN];
    int tf32_converged = isfinite(result->tf32_fixed_point_residual) &&
                         result->tf32_fixed_point_residual <= cfg->tf32_tol;
    int fp64_converged = isfinite(result->fp64_fixed_point_residual) &&
                         result->fp64_fixed_point_residual <= cfg->fp64_tol;

    if (fp == NULL) {
        fprintf(stderr, "could not append %s: %s\n", csv_path,
                strerror(errno));
        return;
    }

    snprintf(gpu_clean, sizeof(gpu_clean), "%s", gpu_name);
    snprintf(status_clean, sizeof(status_clean), "%s", result->status);
    snprintf(error_clean, sizeof(error_clean), "%s", result->error);
    csv_clean(gpu_clean);
    csv_clean(status_clean);
    csv_clean(error_clean);

    fprintf(fp,
            "%d,%.12g,rayleigh_mixed_no_qr,%.17g,%.17g,%d,%d,%d,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d,"
            "%s,%s,%s,%s,%s\n",
            n, epsilon, cfg->tf32_tol, cfg->fp64_tol, cfg->tf32_maxiter,
            cfg->fp64_maxiter, cfg->repeats, result->tf32_time_sec,
            result->orthogonalize_time_sec,
            result->rayleigh_transform_time_sec, result->fp64_time_sec,
            result->backtransform_time_sec, result->total_time_sec,
            result->absolute_residual,
            result->tf32_fixed_point_residual,
            result->fp64_fixed_point_residual, result->tf32_iterations,
            result->fp64_iterations, bool_text(tf32_converged),
            bool_text(fp64_converged), gpu_clean, status_clean, error_clean);
    fclose(fp);
}

static void append_summary(const Config *cfg, const char *message)
{
    FILE *fp = fopen(cfg->summary_txt, "a");

    if (fp == NULL) {
        return;
    }
    fprintf(fp, "%s\n", message);
    fclose(fp);
}

static void warmup(cublasHandle_t handle, const Config *cfg)
{
    double *h_matrix = NULL;
    double *d_matrix = NULL;
    char error[ERROR_LEN] = "";
    Config warm_cfg = *cfg;

    warm_cfg.repeats = 1;
    printf("===== warmup N=%d =====\n", cfg->warmup_n);
    if (make_host_matrix(&h_matrix, cfg->warmup_n, cfg->eps_values[0],
                         cfg->seed + 17) != 0) {
        printf("warmup host matrix allocation failed\n");
        return;
    }
    if (copy_matrix_to_device(&d_matrix, h_matrix, cfg->warmup_n, error,
                              sizeof(error)) != 0) {
        printf("warmup copy failed: %s\n", error);
        free(h_matrix);
        return;
    }

    (void)run_rayleigh(handle, d_matrix, cfg->warmup_n, &warm_cfg, 0);
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
    char gpu_name[256] = "unknown_gpu";
    char summary_line[2048];
    char time_buf[64];
    int failures = 0;

    load_config(&cfg);
    ensure_directory(cfg.logs_dir);
    ensure_directory(cfg.results_dir);

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

    wall_time_string(time_buf, sizeof(time_buf));
    snprintf(summary_line, sizeof(summary_line),
             "started_at=%s method=rayleigh_mixed_no_qr repeats=%d tf32_tol=%.17g "
             "fp64_tol=%.17g gpu=%s",
             time_buf, cfg.repeats, cfg.tf32_tol, cfg.fp64_tol, gpu_name);
    append_summary(&cfg, summary_line);

    printf("===== Rayleigh mixed no-QR sweep =====\n");
    printf("results_dir=%s logs_dir=%s summary=%s\n", cfg.results_dir,
           cfg.logs_dir, cfg.summary_txt);
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
    printf("tf32_tol=%.17g fp64_tol=%.17g repeats=%d gpu=%s\n",
           cfg.tf32_tol, cfg.fp64_tol, cfg.repeats, gpu_name);
    show_gpu_memory("at start");

    warmup(handle, &cfg);

    for (int eps_idx = 0; eps_idx < cfg.eps_count; ++eps_idx) {
        double epsilon = cfg.eps_values[eps_idx];
        char csv_path[4096];

        result_csv_path(&cfg, epsilon, csv_path, sizeof(csv_path));
        for (int n_idx = 0; n_idx < cfg.n_count; ++n_idx) {
            int n = cfg.ns[n_idx];
            double *h_matrix = NULL;
            double *d_matrix = NULL;
            RayleighRunResult result;
            char error[ERROR_LEN] = "";
            double matrix_start = 0.0;

            rayleigh_run_result_reset(&result, "not_run");
            if (cfg.skip_existing && csv_has_case(csv_path, n)) {
                printf("Skipping epsilon=%.12g N=%d because row exists in %s\n",
                       epsilon, n, csv_path);
                continue;
            }

            wall_time_string(time_buf, sizeof(time_buf));
            snprintf(summary_line, sizeof(summary_line),
                     "epsilon=%.12g N=%d started_at=%s", epsilon, n, time_buf);
            append_summary(&cfg, summary_line);
            printf("\n===== epsilon=%.12g N=%d =====\n", epsilon, n);
            show_gpu_memory("before matrix");

            matrix_start = now_seconds();
            if (make_host_matrix(&h_matrix, n, epsilon,
                                 cfg.seed + (uint64_t)n +
                                     epsilon_seed_offset(epsilon)) != 0) {
                snprintf(result.status, sizeof(result.status), "failed");
                snprintf(result.error, sizeof(result.error),
                         "host matrix allocation failed");
                failures++;
                goto row_done;
            }
            printf("host matrix generated in %.8g s\n",
                   now_seconds() - matrix_start);

            if (copy_matrix_to_device(&d_matrix, h_matrix, n, error,
                                      sizeof(error)) != 0) {
                snprintf(result.status, sizeof(result.status), "failed");
                snprintf(result.error, sizeof(result.error), "%s", error);
                failures++;
                goto row_done;
            }
            free(h_matrix);
            h_matrix = NULL;
            show_gpu_memory("after matrix");

            result = run_rayleigh_average(handle, d_matrix, n, &cfg);
            if (strcmp(result.status, "ok") != 0) {
                failures++;
            }

        row_done:
            append_csv_row(csv_path, &cfg, n, epsilon, &result, gpu_name);
            wall_time_string(time_buf, sizeof(time_buf));
            snprintf(summary_line, sizeof(summary_line),
                     "epsilon=%.12g N=%d finished_at=%s status=%s "
                     "time_total=%.17g tf32_time=%.17g orth_time=%.17g "
                     "transform_time=%.17g fp64_time=%.17g back_time=%.17g "
                     "abs_residual=%.17g tf32_fp=%.17g fp64_fp=%.17g "
                     "iters=%d/%d csv=%s",
                     epsilon, n, time_buf, result.status,
                     result.total_time_sec, result.tf32_time_sec,
                     result.orthogonalize_time_sec,
                     result.rayleigh_transform_time_sec,
                     result.fp64_time_sec, result.backtransform_time_sec,
                     result.absolute_residual,
                     result.tf32_fixed_point_residual,
                     result.fp64_fixed_point_residual,
                     result.tf32_iterations, result.fp64_iterations, csv_path);
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
    printf("===== sweep done failures=%d =====\n", failures);

    cublasDestroy(handle);
    return failures == 0 ? 0 : 1;
}
