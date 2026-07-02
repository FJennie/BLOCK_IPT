#include <cuda_runtime.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "../src/ipt_cuda.cu"

#define DEFAULT_RESULTS_DIR "/fs1/home/nudt_liujie/ftt/IPT_C/results"
#define DEFAULT_ITERATIONS 20

typedef struct {
    int n;
    int k;
    int symmetric;
    const char *name;
} TestCase;

static const char *results_dir(void)
{
    const char *dir = getenv("IPT_C_RESULTS_DIR");

    if (dir == NULL || dir[0] == '\0') {
        return DEFAULT_RESULTS_DIR;
    }

    return dir;
}

static int ensure_directory(const char *path)
{
    if (mkdir(path, 0775) == 0 || errno == EEXIST) {
        return 0;
    }

    fprintf(stderr, "could not create directory %s: %s\n", path,
            strerror(errno));
    return -1;
}

static double deterministic_noise(int row, int col)
{
    uint32_t x = (uint32_t)(row + 1) * 747796405u;
    x ^= (uint32_t)(col + 1) * 2891336453u;
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    x ^= x >> 16;

    return ((double)(x & 0x00ffffffu) / 16777216.0) - 0.5;
}

static void fill_perturbative_matrix(double *matrix, int n, int symmetric)
{
    const double eps = 1.0e-2;

    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            double noise = deterministic_noise(row, col);

            if (symmetric) {
                noise = 0.5 *
                        (deterministic_noise(row, col) +
                         deterministic_noise(col, row));
            }

            matrix[row + col * n] = eps * noise;
        }
    }

    for (int row = 0; row < n; ++row) {
        matrix[row + row * n] += (double)(row + 1);
    }
}

static void column_major_matmul(double *out, const double *a, const double *x,
                                int n, int k)
{
    for (int col = 0; col < k; ++col) {
        for (int row = 0; row < n; ++row) {
            double sum = 0.0;

            for (int inner = 0; inner < n; ++inner) {
                sum += a[row + inner * n] * x[inner + col * n];
            }

            out[row + col * n] = sum;
        }
    }
}

static int ipt_cpu_reference(const double *matrix, int n, int k, int iterations,
                             double **vectors_out, double **values_out)
{
    size_t vector_elements = (size_t)n * (size_t)k;
    double *x = (double *)calloc(vector_elements, sizeof(double));
    double *y = (double *)calloc(vector_elements, sizeof(double));
    double *mx = (double *)calloc(vector_elements, sizeof(double));
    double *diagonal = (double *)calloc((size_t)n, sizeof(double));
    double *column_diagonal = (double *)calloc((size_t)k, sizeof(double));
    double *values = (double *)calloc((size_t)k, sizeof(double));

    if (x == NULL || y == NULL || mx == NULL || diagonal == NULL ||
        column_diagonal == NULL || values == NULL) {
        free(x);
        free(y);
        free(mx);
        free(diagonal);
        free(column_diagonal);
        free(values);
        return -1;
    }

    for (int col = 0; col < k; ++col) {
        x[col + col * n] = 1.0;
    }

    for (int row = 0; row < n; ++row) {
        diagonal[row] = matrix[row + row * n];
    }

    for (int iter = 0; iter < iterations; ++iter) {
        column_major_matmul(y, matrix, x, n, k);

        for (int col = 0; col < k; ++col) {
            int diag_idx = col + col * n;
            column_diagonal[col] =
                y[diag_idx] - diagonal[col] * x[diag_idx];
        }

        for (int col = 0; col < k; ++col) {
            for (int row = 0; row < n; ++row) {
                int idx = row + col * n;

                if (row == col) {
                    y[idx] = 1.0;
                } else {
                    double g = 1.0 / (diagonal[col] - diagonal[row]);
                    y[idx] =
                        (y[idx] - diagonal[row] * x[idx] -
                         x[idx] * column_diagonal[col]) *
                        g;
                }
            }
        }

        {
            double *tmp = x;
            x = y;
            y = tmp;
        }
    }

    column_major_matmul(mx, matrix, x, n, k);

    for (int col = 0; col < k; ++col) {
        values[col] = mx[col + col * n];
    }

    free(y);
    free(mx);
    free(diagonal);
    free(column_diagonal);

    *vectors_out = x;
    *values_out = values;
    return 0;
}

static double max_abs_diff(const double *a, const double *b, size_t count)
{
    double max_diff = 0.0;

    for (size_t i = 0; i < count; ++i) {
        double diff = fabs(a[i] - b[i]);

        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    return max_diff;
}

static double max_relative_residual(const double *matrix, const double *vectors,
                                    const double *values, int n, int k)
{
    double *mx = (double *)calloc((size_t)n * (size_t)k, sizeof(double));
    double max_residual = 0.0;

    if (mx == NULL) {
        return INFINITY;
    }

    column_major_matmul(mx, matrix, vectors, n, k);

    for (int col = 0; col < k; ++col) {
        double residual_norm_sq = 0.0;
        double vector_norm_sq = 0.0;

        for (int row = 0; row < n; ++row) {
            int idx = row + col * n;
            double residual = mx[idx] - vectors[idx] * values[col];

            residual_norm_sq += residual * residual;
            vector_norm_sq += vectors[idx] * vectors[idx];
        }

        {
            double residual_norm = sqrt(residual_norm_sq);
            double vector_norm = sqrt(vector_norm_sq);
            double denom = fmax(1.0, fabs(values[col])) * fmax(1.0, vector_norm);
            double relative = residual_norm / denom;

            if (relative > max_residual) {
                max_residual = relative;
            }
        }
    }

    free(mx);
    return max_residual;
}

static int run_case(FILE *csv, const TestCase *test_case)
{
    const double diff_tol = 5.0e-9;
    const double residual_tol = 5.0e-9;
    int status = 0;
    int passed = 0;
    size_t matrix_elements =
        (size_t)test_case->n * (size_t)test_case->n;
    size_t vector_elements =
        (size_t)test_case->n * (size_t)test_case->k;
    double *matrix = (double *)calloc(matrix_elements, sizeof(double));
    double *cpu_vectors = NULL;
    double *cpu_values = NULL;
    IPTCudaResult gpu = {0, 0, 0, NULL, NULL};
    double vector_diff = INFINITY;
    double value_diff = INFINITY;
    double residual = INFINITY;

    if (matrix == NULL) {
        fprintf(stderr, "allocation failed for test matrix\n");
        return 1;
    }

    fill_perturbative_matrix(matrix, test_case->n, test_case->symmetric);

    if (ipt_cpu_reference(matrix, test_case->n, test_case->k,
                          DEFAULT_ITERATIONS, &cpu_vectors,
                          &cpu_values) != 0) {
        fprintf(stderr, "CPU reference allocation failed\n");
        free(matrix);
        return 1;
    }

    status = ipt_cuda(matrix, test_case->n, test_case->k, DEFAULT_ITERATIONS,
                      &gpu);

    if (status == IPT_CUDA_SUCCESS) {
        vector_diff = max_abs_diff(gpu.vectors, cpu_vectors, vector_elements);
        value_diff = max_abs_diff(gpu.values, cpu_values, (size_t)test_case->k);
        residual =
            max_relative_residual(matrix, gpu.vectors, gpu.values, test_case->n,
                                  test_case->k);
        passed = (vector_diff <= diff_tol && value_diff <= diff_tol &&
                  residual <= residual_tol);
    } else {
        fprintf(stderr, "%s failed: %s\n", test_case->name,
                ipt_cuda_status_string(status));
    }

    printf("%-18s n=%3d k=%3d sym=%d vector_diff=%.3e value_diff=%.3e "
           "residual=%.3e %s\n",
           test_case->name, test_case->n, test_case->k,
           test_case->symmetric, vector_diff, value_diff, residual,
           passed ? "PASS" : "FAIL");

    if (csv != NULL) {
        fprintf(csv, "%s,%d,%d,%d,%.17g,%.17g,%.17g,%s\n",
                test_case->name, test_case->n, test_case->k,
                test_case->symmetric, vector_diff, value_diff, residual,
                passed ? "PASS" : "FAIL");
        fflush(csv);
    }

    ipt_cuda_free_result(&gpu);
    free(cpu_vectors);
    free(cpu_values);
    free(matrix);

    return passed ? 0 : 1;
}

int main(void)
{
    TestCase cases[] = {
        {32, 1, 0, "random_k1"},
        {32, 5, 0, "random_k5"},
        {32, 32, 0, "random_full"},
        {32, 1, 1, "symmetric_k1"},
        {32, 5, 1, "symmetric_k5"},
        {32, 32, 1, "symmetric_full"},
        {96, 5, 0, "random_medium"},
        {96, 16, 1, "symmetric_medium"},
    };
    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    int failures = 0;
    const char *dir = results_dir();
    char csv_path[4096];
    FILE *csv = NULL;

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n",
                cudaGetErrorString(cuda_status));
        return 77;
    }

    if (device_count <= 0) {
        fprintf(stderr, "no CUDA device available\n");
        return 77;
    }

    {
        cudaDeviceProp prop;

        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            printf("CUDA device 0: %s, compute capability %d.%d\n", prop.name,
                   prop.major, prop.minor);
        }
    }

    if (ensure_directory(dir) == 0) {
        snprintf(csv_path, sizeof(csv_path), "%s/runtests_summary.csv", dir);
        csv = fopen(csv_path, "w");

        if (csv == NULL) {
            fprintf(stderr, "could not open %s: %s\n", csv_path,
                    strerror(errno));
        } else {
            fprintf(csv,
                    "case,n,k,symmetric,vector_max_abs_diff,value_max_abs_"
                    "diff,max_relative_residual,status\n");
        }
    }

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        failures += run_case(csv, &cases[i]);
    }

    if (csv != NULL) {
        fclose(csv);
        printf("wrote %s\n", csv_path);
    }

    if (failures == 0) {
        printf("all CUDA IPT tests passed\n");
    } else {
        printf("%d CUDA IPT test(s) failed\n", failures);
    }

    return failures == 0 ? 0 : 1;
}
