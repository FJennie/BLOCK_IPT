#include <cuda_runtime.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "../../src/ipt_cuda.cu"
#include "h20_w_csc.h"

#define DEFAULT_RESULTS_DIR "/fs1/home/nudt_liujie/ftt/IPT_C_GPU/results/sparse"

typedef struct {
    const char *name;
    int n;
    int k;
    int nnz;
    int iterations;
    const int *col_ptr;
    const int *row_ind;
    const double *values;
    double diff_tol;
    double residual_tol;
} SparseCase;

static const char *env_or_default(const char *name, const char *default_value)
{
    const char *value = getenv(name);

    return (value == NULL || value[0] == '\0') ? default_value : value;
}

static const char *results_dir(void)
{
    const char *dir = getenv("IPT_SPARSE_RESULTS_DIR");

    if (dir == NULL || dir[0] == '\0') {
        dir = getenv("IPT_C_RESULTS_DIR");
    }

    return (dir == NULL || dir[0] == '\0') ? DEFAULT_RESULTS_DIR : dir;
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

static void csc_to_dense(double *dense, const int *col_ptr, const int *row_ind,
                         const double *values, int n)
{
    memset(dense, 0, (size_t)n * (size_t)n * sizeof(double));

    for (int col = 0; col < n; ++col) {
        for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
            int row = row_ind[p];

            if (row >= 0 && row < n) {
                dense[row + col * n] += values[p];
            }
        }
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

static int build_banded_csc(int n, int **col_ptr_out, int **row_ind_out,
                            double **values_out, int *nnz_out)
{
    int nnz = n + 2 * (n - 1);
    int *col_ptr = (int *)calloc((size_t)n + 1U, sizeof(int));
    int *row_ind = (int *)calloc((size_t)nnz, sizeof(int));
    double *values = (double *)calloc((size_t)nnz, sizeof(double));
    int cursor = 0;

    if (col_ptr == NULL || row_ind == NULL || values == NULL) {
        free(col_ptr);
        free(row_ind);
        free(values);
        return -1;
    }

    for (int col = 0; col < n; ++col) {
        col_ptr[col] = cursor;

        if (col > 0) {
            int edge = col - 1;
            row_ind[cursor] = col - 1;
            values[cursor] = 1.0e-2 * (0.50 + 0.03 * (double)(edge % 7));
            ++cursor;
        }

        row_ind[cursor] = col;
        values[cursor] = (double)(col + 1);
        ++cursor;

        if (col + 1 < n) {
            int edge = col;
            row_ind[cursor] = col + 1;
            values[cursor] = 1.0e-2 * (0.50 + 0.03 * (double)(edge % 7));
            ++cursor;
        }
    }

    col_ptr[n] = cursor;
    *col_ptr_out = col_ptr;
    *row_ind_out = row_ind;
    *values_out = values;
    *nnz_out = cursor;
    return 0;
}

static int run_case(FILE *csv, FILE *summary, const SparseCase *test_case)
{
    int status = 0;
    int passed = 0;
    size_t matrix_elements = (size_t)test_case->n * (size_t)test_case->n;
    size_t vector_elements = (size_t)test_case->n * (size_t)test_case->k;
    double *dense = (double *)calloc(matrix_elements, sizeof(double));
    double *cpu_vectors = NULL;
    double *cpu_values = NULL;
    IPTCudaResult gpu = {0, 0, 0, NULL, NULL};
    double vector_diff = INFINITY;
    double value_diff = INFINITY;
    double residual = INFINITY;

    if (dense == NULL) {
        fprintf(stderr, "allocation failed for dense reference matrix\n");
        return 1;
    }

    csc_to_dense(dense, test_case->col_ptr, test_case->row_ind,
                 test_case->values, test_case->n);

    if (ipt_cpu_reference(dense, test_case->n, test_case->k,
                          test_case->iterations, &cpu_vectors,
                          &cpu_values) != 0) {
        fprintf(stderr, "CPU reference allocation failed for %s\n",
                test_case->name);
        free(dense);
        return 1;
    }

    status = ipt_cuda_sparse_csc(test_case->col_ptr, test_case->row_ind,
                                 test_case->values, test_case->n,
                                 test_case->k, test_case->nnz,
                                 test_case->iterations, &gpu);

    if (status == IPT_CUDA_SUCCESS) {
        vector_diff = max_abs_diff(gpu.vectors, cpu_vectors, vector_elements);
        value_diff = max_abs_diff(gpu.values, cpu_values, (size_t)test_case->k);
        residual = max_relative_residual(dense, gpu.vectors, gpu.values,
                                         test_case->n, test_case->k);
        passed = (vector_diff <= test_case->diff_tol &&
                  value_diff <= test_case->diff_tol &&
                  residual <= test_case->residual_tol);
    } else {
        fprintf(stderr, "%s failed: %s\n", test_case->name,
                ipt_cuda_status_string(status));
    }

    printf("%-22s n=%4d k=%3d nnz=%5d it=%4d vector_diff=%.3e "
           "value_diff=%.3e residual=%.3e %s\n",
           test_case->name, test_case->n, test_case->k, test_case->nnz,
           test_case->iterations, vector_diff, value_diff, residual,
           passed ? "PASS" : "FAIL");

    if (csv != NULL) {
        fprintf(csv,
                "%s,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%s\n",
                test_case->name, test_case->n, test_case->k, test_case->nnz,
                test_case->iterations, vector_diff, value_diff, residual,
                test_case->diff_tol, test_case->residual_tol,
                passed ? "PASS" : "FAIL");
        fflush(csv);
    }

    if (summary != NULL) {
        fprintf(summary,
                "%s: vector_diff=%.17g value_diff=%.17g residual=%.17g "
                "status=%s\n",
                test_case->name, vector_diff, value_diff, residual,
                passed ? "PASS" : "FAIL");
        fflush(summary);
    }

    ipt_cuda_free_result(&gpu);
    free(cpu_vectors);
    free(cpu_values);
    free(dense);

    return passed ? 0 : 1;
}

int main(void)
{
    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    int failures = 0;
    const char *dir = results_dir();
    char csv_path[4096];
    char summary_path[4096];
    FILE *csv = NULL;
    FILE *summary = NULL;
    int *band_col_ptr = NULL;
    int *band_row_ind = NULL;
    double *band_values = NULL;
    int band_nnz = 0;

    (void)env_or_default;

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

    if (build_banded_csc(64, &band_col_ptr, &band_row_ind, &band_values,
                         &band_nnz) != 0) {
        fprintf(stderr, "could not build synthetic sparse matrix\n");
        return 2;
    }

    if (ensure_directory(dir) == 0) {
        snprintf(csv_path, sizeof(csv_path), "%s/sparse_runtests_summary.csv",
                 dir);
        snprintf(summary_path, sizeof(summary_path),
                 "%s/sparse_runtests_summary.txt", dir);
        csv = fopen(csv_path, "w");
        summary = fopen(summary_path, "w");

        if (csv == NULL) {
            fprintf(stderr, "could not open %s: %s\n", csv_path,
                    strerror(errno));
        } else {
            fprintf(csv,
                    "case,n,k,nnz,iterations,vector_max_abs_diff,value_max_"
                    "abs_diff,max_relative_residual,diff_tol,residual_tol,"
                    "status\n");
        }

        if (summary == NULL) {
            fprintf(stderr, "could not open %s: %s\n", summary_path,
                    strerror(errno));
        } else {
            fprintf(summary, "IPT CUDA sparse CSC tests\n");
        }
    }

    {
        SparseCase cases[] = {
            {"synthetic_banded_k5", 64, 5, band_nnz, 80, band_col_ptr,
             band_row_ind, band_values, 1.0e-8, 1.0e-8},
            {"h20_fci_w_k1", H20_W_CSC_N, 1, H20_W_CSC_NNZ, 80,
             H20_W_CSC_COL_PTR, H20_W_CSC_ROW_IND, H20_W_CSC_VALUES,
             1.0e-8, 1.0e-10},
        };

        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
            failures += run_case(csv, summary, &cases[i]);
        }
    }

    if (summary != NULL) {
        fprintf(summary, "failures=%d\n", failures);
        fclose(summary);
        printf("wrote %s\n", summary_path);
    }

    if (csv != NULL) {
        fclose(csv);
        printf("wrote %s\n", csv_path);
    }

    free(band_col_ptr);
    free(band_row_ind);
    free(band_values);

    if (failures == 0) {
        printf("all sparse CUDA IPT tests passed\n");
    } else {
        printf("%d sparse CUDA IPT test(s) failed\n", failures);
    }

    return failures == 0 ? 0 : 1;
}
