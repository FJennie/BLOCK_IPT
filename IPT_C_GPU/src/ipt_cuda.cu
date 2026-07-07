#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse_v2.h>

#include <chrono>
#include <algorithm>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "ipt_preparation.cu"

static double ipt_cuda_env_double(const char *name, double default_value)
{
    const char *raw = getenv(name);
    char *end = NULL;
    double value = 0.0;

    if (raw == NULL || raw[0] == '\0') {
        return default_value;
    }
    value = strtod(raw, &end);
    if (end == raw || value < 0.0 || !isfinite(value)) {
        return default_value;
    }
    return value;
}

static int ipt_cuda_env_int(const char *name, int default_value)
{
    const char *raw = getenv(name);
    char *end = NULL;
    long value = 0;

    if (raw == NULL || raw[0] == '\0') {
        return default_value;
    }
    value = strtol(raw, &end, 10);
    if (end == raw || value < 0 || value > INT_MAX) {
        return default_value;
    }
    return (int)value;
}

static int ipt_cuda_env_flag(const char *name)
{
    const char *raw = getenv(name);

    if (raw == NULL || raw[0] == '\0') {
        return 0;
    }
    if (strcmp(raw, "0") == 0 || strcmp(raw, "false") == 0 ||
        strcmp(raw, "FALSE") == 0 || strcmp(raw, "no") == 0 ||
        strcmp(raw, "NO") == 0) {
        return 0;
    }
    return 1;
}

static int ipt_cuda_env_flag_default(const char *name, int default_value)
{
    const char *raw = getenv(name);

    if (raw == NULL || raw[0] == '\0') {
        return default_value;
    }
    return ipt_cuda_env_flag(name);
}

static std::vector<double> ipt_cuda_env_double_list(
    const char *name, const char *default_value)
{
    const char *raw = getenv(name);
    const char *cursor = raw != NULL && raw[0] != '\0' ? raw
                                                        : default_value;
    std::vector<double> values;

    while (cursor != NULL && *cursor != '\0') {
        char *end = NULL;
        double value = strtod(cursor, &end);

        if (end != cursor && isfinite(value) && value > 0.0) {
            values.push_back(value);
        }
        if (end == NULL || *end == '\0') {
            break;
        }
        cursor = end + 1;
    }
    return values;
}

#ifndef IPT_CUDA_DECLS
#define IPT_CUDA_DECLS

#ifdef __cplusplus
extern "C" {
#endif

enum {
    IPT_CUDA_SUCCESS = 0,
    IPT_CUDA_INVALID_ARGUMENT = 1,
    IPT_CUDA_ALLOCATION_FAILED = 2,
    IPT_CUDA_CUDA_ERROR = 3,
    IPT_CUDA_CUBLAS_ERROR = 4,
    IPT_CUDA_CUSPARSE_ERROR = 5,
    IPT_CUDA_CUSOLVER_ERROR = 6
};

typedef struct {
    int davidson_step;
    int active_pair_index;
    double residual_before;
    double residual_after;
    int accepted;
    int basis_cols;
    double max_relative_eigen_residual;
    int max_relative_eigen_residual_index;
    double pair_28_residual;
    double pair_29_residual;
    double orthogonality_max_abs_error;
    int restarted;
    int best_so_far_updated;
    int best_so_far_step;
    double best_so_far_max_residual;
    double pair28_best_residual;
    double pair29_best_residual;
    int relaxed_accept_enabled;
    int accept_global_ok;
    int accept_active_ok;
    int accept_locked_safe;
    char accept_reason[64];
    int reject_retry_count;
    double retry_alpha;
    double retry_denom_clip;
    int accepted_steps;
    int rejected_steps;
    int min_accepted_steps;
    int early_jump_to_continuation;
    double gap28_29;
    double relative_gap28_29;
    double cluster_residual_fro_before;
    double cluster_residual_fro_after;
    double cluster_residual_max_before;
    double cluster_residual_max_after;
    double ritz_overlap_28_28;
    double ritz_overlap_28_29;
    double ritz_overlap_29_28;
    double ritz_overlap_29_29;
    int ritz_overlap_swap_detected;
    char pair28_lock_state[16];
    char pair29_lock_state[16];
    int cluster_hard_locked;
    int cluster_soft_locked_count;
} IPTDavidsonHistoryEntry;

typedef struct {
    int davidson_step;
    int pair_index;
    double residual;
    int selected_by_residual;
    int selected_forced;
    int selected_auto_cluster;
    int skipped_converged_in_cluster;
    int skipped_linear_dependent;
    int skipped_locked_old_logic_should_not_happen;
    double correction_norm_before_ortho;
    double correction_norm_after_ortho;
} IPTDavidsonSelectionEntry;

typedef struct {
    int davidson_step;
    char active_pairs[64];
    int accepted_corrections;
    int rejected_corrections;
    char correction_norm_before_ortho[256];
    char correction_norm_after_ortho[256];
    double residual_before_global;
    double residual_after_global;
    double pair_28_before;
    double pair_28_after;
    double pair_29_before;
    double pair_29_after;
    int accepted;
    char reject_reason[64];
    int basis_cols_before;
    int basis_cols_after;
    double orthogonality_error_before;
    double orthogonality_error_after;
    int best_so_far_updated;
    int best_so_far_step;
    double best_so_far_max_residual;
    double pair28_best_residual;
    double pair29_best_residual;
    int relaxed_accept_enabled;
    int accept_global_ok;
    int accept_active_ok;
    int accept_locked_safe;
    char accept_reason[64];
    int reject_retry_count;
    double retry_alpha;
    double retry_denom_clip;
    int accepted_steps;
    int rejected_steps;
    int min_accepted_steps;
    int early_jump_to_continuation;
    double gap28_29;
    double relative_gap28_29;
    double cluster_residual_fro_before;
    double cluster_residual_fro_after;
    double cluster_residual_max_before;
    double cluster_residual_max_after;
    double ritz_overlap_28_28;
    double ritz_overlap_28_29;
    double ritz_overlap_29_28;
    double ritz_overlap_29_29;
    int ritz_overlap_swap_detected;
    char pair28_lock_state[16];
    char pair29_lock_state[16];
    int cluster_hard_locked;
    int cluster_soft_locked_count;
} IPTDavidsonBlockHistoryEntry;

typedef struct {
    int jd_step;
    int active_pair_index;
    double residual_before;
    double residual_after;
    int accepted;
    int basis_cols;
    double max_relative_eigen_residual;
    int max_relative_eigen_residual_index;
    double pair_28_residual;
    double pair_29_residual;
    double orthogonality_max_abs_error;
    int best_so_far_updated;
    int best_so_far_step;
    double best_so_far_max_residual;
    double pair28_best_residual;
    double pair29_best_residual;
    int relaxed_accept_enabled;
    int accept_global_ok;
    int accept_active_ok;
    int accept_locked_safe;
    char accept_reason[64];
    int reject_retry_count;
    double retry_alpha;
    double retry_denom_clip;
    int accepted_steps;
    int rejected_steps;
    int min_accepted_steps;
    int early_jump_to_continuation;
    double gap28_29;
    double relative_gap28_29;
    double cluster_residual_fro_before;
    double cluster_residual_fro_after;
    double cluster_residual_max_before;
    double cluster_residual_max_after;
    double ritz_overlap_28_28;
    double ritz_overlap_28_29;
    double ritz_overlap_29_28;
    double ritz_overlap_29_29;
    int ritz_overlap_swap_detected;
    char pair28_lock_state[16];
    char pair29_lock_state[16];
    int cluster_hard_locked;
    int cluster_soft_locked_count;
} IPTJDLocalHistoryEntry;

typedef struct {
    int n;
    int k;
    int iterations;
    double *vectors; /* Column-major n-by-k eigenvector/eigenmatrix estimate. */
    double *values;  /* Length-k diagonal of M * vectors. */
    int basis_cols;  /* Rayleigh-Ritz/subspace columns used to produce k pairs. */
    long long matvecs;
    double preparation_time_sec;
    double transfer_setup_time_sec;
    double iteration_time_sec;
    double rayleigh_ritz_time_sec;
    double solve_time_sec;
    double fixed_point_residual;
    int oversample;
    int rayleigh_ritz_used_qr;
    double basis_orthogonality_frobenius_error;
    double basis_orthogonality_max_abs_error;
    double ritz_vectors_orthogonality_frobenius_error;
    double ritz_vectors_orthogonality_max_abs_error;
    int basis_has_nan_or_inf;
    int ritz_vectors_has_nan_or_inf;
    int davidson_attempted;
    int davidson_accepted;
    int davidson_target_index;
    double davidson_residual_before;
    double davidson_residual_after;
    double davidson_denom_clip;
    int adaptive_block_enabled;
    double adaptive_coupling_tau;
    int adaptive_limit_hit;
    int adaptive_target_block_start;
    int adaptive_target_block_end;
    char adaptive_added_indices[256];
    int davidson_history_count;
    IPTDavidsonHistoryEntry *davidson_history;
    int davidson_restart_count;
    int davidson_selection_history_count;
    IPTDavidsonSelectionEntry *davidson_selection_history;
    int davidson_block_history_count;
    IPTDavidsonBlockHistoryEntry *davidson_block_history;
    int jd_local_attempted;
    int jd_local_accepted;
    int jd_local_history_count;
    IPTJDLocalHistoryEntry *jd_local_history;
    int best_so_far_enabled;
    int best_so_far_updated;
    int best_so_far_update_count;
    int best_so_far_step;
    char best_so_far_source[32];
    int best_so_far_basis_cols;
    double best_so_far_max_residual;
    int best_so_far_max_residual_index;
    int final_returned_from_best_so_far;
    double pair28_best_residual;
    double pair29_best_residual;
    int relaxed_accept_enabled;
    int accepted_steps;
    int rejected_steps;
    int min_accepted_steps;
    int early_jump_to_continuation;
    int cluster_aware_accept_enabled;
    int soft_cluster_locking_enabled;
    double gap28_29;
    double relative_gap28_29;
    double cluster_residual_fro;
    double cluster_residual_max;
    char pair28_lock_state[16];
    char pair29_lock_state[16];
    int cluster_hard_locked;
    int cluster_soft_locked_count;
} IPTCudaResult;

const char *ipt_cuda_status_string(int status);
void ipt_cuda_free_result(IPTCudaResult *result);
void ipt_cuda_free_device_result(double *d_vectors, double *d_values);
int ipt_cuda(const double *matrix_col_major, int n, int k, int iterations,
             IPTCudaResult *result);
int ipt_cuda_tol(const double *matrix_col_major, int n, int k, double tol,
                 int maxiter, IPTCudaResult *result);
int ipt_cuda_with_initial(const double *matrix_col_major,
                          const double *initial_vectors_col_major, int n,
                          int k, int iterations, IPTCudaResult *result);
int ipt_cuda_with_initial_tol(const double *matrix_col_major,
                              const double *initial_vectors_col_major, int n,
                              int k, double tol, int maxiter,
                              IPTCudaResult *result);
int ipt_cuda_device(const double *d_matrix_col_major, int n, int k,
                    int iterations, cublasHandle_t handle,
                    double **d_vectors_out, double **d_values_out);
int ipt_cuda_device_tol(const double *d_matrix_col_major, int n, int k,
                        double tol, int maxiter, cublasHandle_t handle,
                        double **d_vectors_out, double **d_values_out,
                        int *iterations_done_out,
                        double *fixed_point_residual_out);
int ipt_cuda_sparse_csc(const int *col_ptr, const int *row_ind,
                        const double *matrix_values, int n, int k, int nnz,
                        int iterations, IPTCudaResult *result);
int ipt_cuda_sparse_csc_tol(const int *col_ptr, const int *row_ind,
                            const double *matrix_values, int n, int k,
                            int nnz, double tol, int maxiter,
                            IPTCudaResult *result);
int ipt_cuda_sparse_csc_with_initial(
    const int *col_ptr, const int *row_ind, const double *matrix_values,
    const double *initial_vectors_col_major, int n, int k, int nnz,
    int iterations, IPTCudaResult *result);
int ipt_cuda_sparse_csc_with_initial_tol(
    const int *col_ptr, const int *row_ind, const double *matrix_values,
    const double *initial_vectors_col_major, int n, int k, int nnz,
    double tol, int maxiter, IPTCudaResult *result);
int ipt_cuda_sparse_csc_device(
    const int *d_col_ptr, const int *d_row_ind,
    const double *d_matrix_values, int n, int k, int nnz, int iterations,
    cublasHandle_t handle, double **d_vectors_out, double **d_values_out);
int ipt_cuda_sparse_csc_device_tol(
    const int *d_col_ptr, const int *d_row_ind,
    const double *d_matrix_values, int n, int k, int nnz, double tol,
    int maxiter, cublasHandle_t handle, double **d_vectors_out,
    double **d_values_out, int *iterations_done_out,
    double *fixed_point_residual_out);

#ifdef __cplusplus
}
#endif

#endif

static const char *cublas_status_name(cublasStatus_t status)
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

static const char *cusparse_status_name(cusparseStatus_t status)
{
    switch (status) {
    case CUSPARSE_STATUS_SUCCESS:
        return "CUSPARSE_STATUS_SUCCESS";
    case CUSPARSE_STATUS_NOT_INITIALIZED:
        return "CUSPARSE_STATUS_NOT_INITIALIZED";
    case CUSPARSE_STATUS_ALLOC_FAILED:
        return "CUSPARSE_STATUS_ALLOC_FAILED";
    case CUSPARSE_STATUS_INVALID_VALUE:
        return "CUSPARSE_STATUS_INVALID_VALUE";
    case CUSPARSE_STATUS_ARCH_MISMATCH:
        return "CUSPARSE_STATUS_ARCH_MISMATCH";
    case CUSPARSE_STATUS_MAPPING_ERROR:
        return "CUSPARSE_STATUS_MAPPING_ERROR";
    case CUSPARSE_STATUS_EXECUTION_FAILED:
        return "CUSPARSE_STATUS_EXECUTION_FAILED";
    case CUSPARSE_STATUS_INTERNAL_ERROR:
        return "CUSPARSE_STATUS_INTERNAL_ERROR";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSPARSE_STATUS_NOT_SUPPORTED:
        return "CUSPARSE_STATUS_NOT_SUPPORTED";
    case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:
        return "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
    default:
        return "CUSPARSE_STATUS_UNKNOWN";
    }
}

// 错误码转字符串
extern "C" const char *ipt_cuda_status_string(int status)
{
    switch (status) {
    case IPT_CUDA_SUCCESS:
        return "success";
    case IPT_CUDA_INVALID_ARGUMENT:
        return "invalid argument";
    case IPT_CUDA_ALLOCATION_FAILED:
        return "allocation failed";
    case IPT_CUDA_CUDA_ERROR:
        return "CUDA runtime error";
    case IPT_CUDA_CUBLAS_ERROR:
        return "cuBLAS error";
    case IPT_CUDA_CUSPARSE_ERROR:
        return "cuSPARSE error";
    case IPT_CUDA_CUSOLVER_ERROR:
        return "cuSOLVER error";
    default:
        return "unknown error";
    }
}

// 释放 host 侧 IPTCudaResult
extern "C" void ipt_cuda_free_result(IPTCudaResult *result)
{
    if (result == NULL) {
        return;
    }

    free(result->vectors);
    free(result->values);

    result->n = 0;
    result->k = 0;
    result->iterations = 0;
    result->vectors = NULL;
    result->values = NULL;
    result->basis_cols = 0;
    result->matvecs = 0;
    result->preparation_time_sec = 0.0;
    result->transfer_setup_time_sec = 0.0;
    result->iteration_time_sec = 0.0;
    result->rayleigh_ritz_time_sec = 0.0;
    result->solve_time_sec = 0.0;
    result->fixed_point_residual = NAN;
    result->oversample = 0;
    result->rayleigh_ritz_used_qr = 0;
    result->basis_orthogonality_frobenius_error = NAN;
    result->basis_orthogonality_max_abs_error = NAN;
    result->ritz_vectors_orthogonality_frobenius_error = NAN;
    result->ritz_vectors_orthogonality_max_abs_error = NAN;
    result->basis_has_nan_or_inf = 0;
    result->ritz_vectors_has_nan_or_inf = 0;
    result->davidson_attempted = 0;
    result->davidson_accepted = 0;
    result->davidson_target_index = -1;
    result->davidson_residual_before = NAN;
    result->davidson_residual_after = NAN;
    result->davidson_denom_clip = NAN;
    result->adaptive_block_enabled = 0;
    result->adaptive_coupling_tau = NAN;
    result->adaptive_limit_hit = 0;
    result->adaptive_target_block_start = -1;
    result->adaptive_target_block_end = -1;
    result->adaptive_added_indices[0] = '\0';
    free(result->davidson_history);
    result->davidson_history_count = 0;
    result->davidson_history = NULL;
    result->davidson_restart_count = 0;
    free(result->davidson_selection_history);
    result->davidson_selection_history_count = 0;
    result->davidson_selection_history = NULL;
    free(result->davidson_block_history);
    result->davidson_block_history_count = 0;
    result->davidson_block_history = NULL;
    result->jd_local_attempted = 0;
    result->jd_local_accepted = 0;
    free(result->jd_local_history);
    result->jd_local_history_count = 0;
    result->jd_local_history = NULL;
    result->best_so_far_enabled = 0;
    result->best_so_far_updated = 0;
    result->best_so_far_update_count = 0;
    result->best_so_far_step = -1;
    result->best_so_far_source[0] = '\0';
    result->best_so_far_basis_cols = 0;
    result->best_so_far_max_residual = NAN;
    result->best_so_far_max_residual_index = -1;
    result->final_returned_from_best_so_far = 0;
    result->pair28_best_residual = NAN;
    result->pair29_best_residual = NAN;
    result->relaxed_accept_enabled = 0;
    result->accepted_steps = 0;
    result->rejected_steps = 0;
    result->min_accepted_steps = 0;
    result->early_jump_to_continuation = 0;
    result->cluster_aware_accept_enabled = 0;
    result->soft_cluster_locking_enabled = 0;
    result->gap28_29 = NAN;
    result->relative_gap28_29 = NAN;
    result->cluster_residual_fro = NAN;
    result->cluster_residual_max = NAN;
    snprintf(result->pair28_lock_state, sizeof(result->pair28_lock_state),
             "unknown");
    snprintf(result->pair29_lock_state, sizeof(result->pair29_lock_state),
             "unknown");
    result->cluster_hard_locked = 0;
    result->cluster_soft_locked_count = 0;
}

// 释放 device 侧 d_vectors/d_values
extern "C" void ipt_cuda_free_device_result(double *d_vectors,
                                             double *d_values)
{
    cudaFree(d_vectors);
    cudaFree(d_values);
}

struct IPTBestSoFarState {
    int enabled;
    int has_state;
    int updated;
    int update_count;
    int step;
    char source[32];
    int basis_cols;
    double max_residual;
    int max_residual_index;
    int state_is_current;
    int rayleigh_ritz_used_qr;
    double basis_orthogonality_frobenius_error;
    double basis_orthogonality_max_abs_error;
    double ritz_vectors_orthogonality_frobenius_error;
    double ritz_vectors_orthogonality_max_abs_error;
    int basis_has_nan_or_inf;
    int ritz_vectors_has_nan_or_inf;
    double pair28_best_residual;
    double pair29_best_residual;
    std::vector<double> values;
    std::vector<double> vectors;
    std::vector<double> residuals;
};

static void ipt_best_so_far_init(IPTBestSoFarState *best, int enabled)
{
    if (best == NULL) {
        return;
    }
    best->enabled = enabled;
    best->has_state = 0;
    best->updated = 0;
    best->update_count = 0;
    best->step = -1;
    snprintf(best->source, sizeof(best->source), "none");
    best->basis_cols = 0;
    best->max_residual = NAN;
    best->max_residual_index = -1;
    best->state_is_current = 0;
    best->rayleigh_ritz_used_qr = 0;
    best->basis_orthogonality_frobenius_error = NAN;
    best->basis_orthogonality_max_abs_error = NAN;
    best->ritz_vectors_orthogonality_frobenius_error = NAN;
    best->ritz_vectors_orthogonality_max_abs_error = NAN;
    best->basis_has_nan_or_inf = 0;
    best->ritz_vectors_has_nan_or_inf = 0;
    best->pair28_best_residual = NAN;
    best->pair29_best_residual = NAN;
    best->values.clear();
    best->vectors.clear();
    best->residuals.clear();
}

static void ipt_best_so_far_note_pair_best(IPTBestSoFarState *best,
                                           const std::vector<double> &residuals)
{
    if (best == NULL || !best->enabled) {
        return;
    }
    if (residuals.size() > 28U && isfinite(residuals[28]) &&
        (!isfinite(best->pair28_best_residual) ||
         residuals[28] < best->pair28_best_residual)) {
        best->pair28_best_residual = residuals[28];
    }
    if (residuals.size() > 29U && isfinite(residuals[29]) &&
        (!isfinite(best->pair29_best_residual) ||
         residuals[29] < best->pair29_best_residual)) {
        best->pair29_best_residual = residuals[29];
    }
}

static int ipt_best_so_far_note(
    IPTBestSoFarState *best, const char *source, int step,
    const std::vector<double> &values, const std::vector<double> &vectors,
    const std::vector<double> &residuals, double max_residual,
    int max_residual_index, int basis_cols, int rayleigh_ritz_used_qr,
    double basis_orthogonality_frobenius_error,
    double basis_orthogonality_max_abs_error, int basis_has_nan_or_inf,
    double ritz_vectors_orthogonality_frobenius_error,
    double ritz_vectors_orthogonality_max_abs_error,
    int ritz_vectors_has_nan_or_inf, int state_is_current)
{
    int improves = 0;

    if (best == NULL || !best->enabled) {
        return 0;
    }
    ipt_best_so_far_note_pair_best(best, residuals);
    if (!isfinite(max_residual) || values.empty() || vectors.empty()) {
        return 0;
    }
    improves = !best->has_state || max_residual < best->max_residual;
    if (!improves) {
        return 0;
    }
    if (best->has_state) {
        best->updated = 1;
    }
    best->has_state = 1;
    ++best->update_count;
    best->step = step;
    snprintf(best->source, sizeof(best->source), "%s",
             source == NULL ? "unknown" : source);
    best->basis_cols = basis_cols;
    best->max_residual = max_residual;
    best->max_residual_index = max_residual_index;
    best->state_is_current = state_is_current;
    best->rayleigh_ritz_used_qr = rayleigh_ritz_used_qr;
    best->basis_orthogonality_frobenius_error =
        basis_orthogonality_frobenius_error;
    best->basis_orthogonality_max_abs_error =
        basis_orthogonality_max_abs_error;
    best->basis_has_nan_or_inf = basis_has_nan_or_inf;
    best->ritz_vectors_orthogonality_frobenius_error =
        ritz_vectors_orthogonality_frobenius_error;
    best->ritz_vectors_orthogonality_max_abs_error =
        ritz_vectors_orthogonality_max_abs_error;
    best->ritz_vectors_has_nan_or_inf = ritz_vectors_has_nan_or_inf;
    best->values = values;
    best->vectors = vectors;
    best->residuals = residuals;
    return 1;
}

static void ipt_best_so_far_refresh_current_metadata(
    IPTBestSoFarState *best, int basis_cols,
    double basis_orthogonality_frobenius_error,
    double basis_orthogonality_max_abs_error, int basis_has_nan_or_inf)
{
    if (best == NULL || !best->enabled || !best->has_state ||
        !best->state_is_current) {
        return;
    }
    best->basis_cols = basis_cols;
    best->basis_orthogonality_frobenius_error =
        basis_orthogonality_frobenius_error;
    best->basis_orthogonality_max_abs_error =
        basis_orthogonality_max_abs_error;
    best->basis_has_nan_or_inf = basis_has_nan_or_inf;
}

static void ipt_best_so_far_store_result_fields(
    IPTCudaResult *result, const IPTBestSoFarState *best,
    int final_returned_from_best_so_far)
{
    if (result == NULL || best == NULL) {
        return;
    }
    result->best_so_far_enabled = best->enabled;
    result->best_so_far_updated = best->updated;
    result->best_so_far_update_count = best->update_count;
    result->best_so_far_step = best->has_state ? best->step : -1;
    snprintf(result->best_so_far_source, sizeof(result->best_so_far_source),
             "%s", best->has_state ? best->source : "none");
    result->best_so_far_basis_cols = best->has_state ? best->basis_cols : 0;
    result->best_so_far_max_residual =
        best->has_state ? best->max_residual : NAN;
    result->best_so_far_max_residual_index =
        best->has_state ? best->max_residual_index : -1;
    result->final_returned_from_best_so_far =
        final_returned_from_best_so_far;
    result->pair28_best_residual = best->pair28_best_residual;
    result->pair29_best_residual = best->pair29_best_residual;
}

struct IPTTrialAcceptDecision {
    int relaxed_accept_enabled;
    int finite_ok;
    int global_ok;
    int active_ok;
    int locked_safe;
    int accepted;
    char reason[64];
};

struct IPTClusterTrialMetrics {
    int enabled;
    int active_ok;
    double gap28_29;
    double relative_gap28_29;
    double cluster_residual_fro_before;
    double cluster_residual_fro_after;
    double cluster_residual_max_before;
    double cluster_residual_max_after;
    double overlap_28_28;
    double overlap_28_29;
    double overlap_29_28;
    double overlap_29_29;
    int overlap_swap_detected;
};

struct IPTActiveCluster {
    std::vector<int> members;
    int explicit_forced;
    int active_by_gap;
    int source;
    double min_abs_gap;
    double min_rel_gap;
};

enum {
    IPT_CLUSTER_SOURCE_AUTO_GAP = 0,
    IPT_CLUSTER_SOURCE_LEGACY_FORCE = 1
};

struct IPTGenericClusterMetrics {
    std::vector<int> members;
    std::vector<int> assignment_trial_index_for_old_member;
    std::vector<double> overlap_matrix;
    std::vector<double> matched_after_residuals;
    double cluster_residual_fro_before;
    double cluster_residual_fro_after;
    double cluster_residual_max_before;
    double cluster_residual_max_after;
    int active_ok;
    int locked_safe;
    int swap_or_permutation_detected;
};

static int ipt_vector_contains_int(const std::vector<int> &values, int needle)
{
    return std::find(values.begin(), values.end(), needle) != values.end();
}

static int ipt_cluster_has_exact_members(const std::vector<int> &members,
                                         int first, int second)
{
    return members.size() == 2U && members[0] == first &&
           members[1] == second;
}

static const char *ipt_cluster_source_name(int source)
{
    return source == IPT_CLUSTER_SOURCE_LEGACY_FORCE
               ? "legacy_force"
               : "auto_gap";
}

static std::string ipt_join_indices(const std::vector<int> &indices)
{
    std::string text;

    for (int index : indices) {
        if (!text.empty()) {
            text += ",";
        }
        text += std::to_string(index);
    }
    return text;
}

static std::vector<int> ipt_cluster_correction_members(
    const IPTActiveCluster &cluster,
    const std::vector<double> &residuals, double converged_tolerance)
{
    std::vector<int> correction_members;

    for (int member : cluster.members) {
        if (member >= 0 && member < (int)residuals.size() &&
            isfinite(residuals[(size_t)member]) &&
            residuals[(size_t)member] > converged_tolerance) {
            correction_members.push_back(member);
        }
    }
    return correction_members;
}

static std::vector<IPTActiveCluster> ipt_discover_auto_ritz_clusters(
    int k, const std::vector<double> &ritz_values,
    const std::vector<double> &residuals, double converged_tolerance,
    double gap_abs_tol, double gap_rel_tol)
{
    std::vector<IPTActiveCluster> clusters;
    int cluster_start = -1;

    if (k < 2 || ritz_values.size() < (size_t)k ||
        residuals.size() < (size_t)k) {
        return clusters;
    }
    for (int edge = 0; edge < k - 1; ++edge) {
        double left = ritz_values[(size_t)edge];
        double right = ritz_values[(size_t)edge + 1U];
        double abs_gap = fabs(right - left);
        double rel_gap =
            abs_gap /
            std::max(1.0, std::max(fabs(left), fabs(right)));
        int small_gap =
            isfinite(abs_gap) && isfinite(rel_gap) &&
            (abs_gap <= gap_abs_tol || rel_gap <= gap_rel_tol);

        if (small_gap && cluster_start < 0) {
            cluster_start = edge;
        }
        if ((!small_gap || edge == k - 2) && cluster_start >= 0) {
            int cluster_end =
                small_gap && edge == k - 2 ? edge + 1 : edge;
            IPTActiveCluster cluster = {};
            int has_unconverged_member = 0;

            cluster.explicit_forced = 0;
            cluster.active_by_gap = 1;
            cluster.source = IPT_CLUSTER_SOURCE_AUTO_GAP;
            cluster.min_abs_gap = INFINITY;
            cluster.min_rel_gap = INFINITY;
            for (int member = cluster_start; member <= cluster_end;
                 ++member) {
                cluster.members.push_back(member);
                if (isfinite(residuals[(size_t)member]) &&
                    residuals[(size_t)member] >
                        converged_tolerance) {
                    has_unconverged_member = 1;
                }
            }
            for (int member = cluster_start; member < cluster_end;
                 ++member) {
                double a = ritz_values[(size_t)member];
                double b = ritz_values[(size_t)member + 1U];
                double abs_gap = fabs(b - a);
                double rel_gap =
                    abs_gap /
                    std::max(1.0, std::max(fabs(a), fabs(b)));

                cluster.min_abs_gap =
                    std::min(cluster.min_abs_gap, abs_gap);
                cluster.min_rel_gap =
                    std::min(cluster.min_rel_gap, rel_gap);
            }
            if (cluster.members.size() >= 2U &&
                has_unconverged_member) {
                clusters.push_back(cluster);
            }
            cluster_start = -1;
        }
    }
    return clusters;
}

static void ipt_log_auto_cluster_discovery(
    int step, const std::vector<IPTActiveCluster> &clusters,
    const std::vector<double> &residuals, double converged_tolerance)
{
    fprintf(stderr,
            "IPT Davidson auto clusters: step=%d count=%zu "
            "converged_tol=%.17g active_max_used=0 "
            "correction_max_per_step_used=0\n",
            step, clusters.size(), converged_tolerance);
    for (size_t cluster_index = 0; cluster_index < clusters.size();
         ++cluster_index) {
        const IPTActiveCluster &cluster = clusters[cluster_index];
        std::vector<int> correction_members =
            ipt_cluster_correction_members(
                cluster, residuals, converged_tolerance);
        std::vector<int> skipped_converged;
        double fro = 0.0;
        double maximum = 0.0;

        for (int member : cluster.members) {
            double residual = residuals[(size_t)member];

            fro += residual * residual;
            maximum = std::max(maximum, residual);
            if (residual <= converged_tolerance) {
                skipped_converged.push_back(member);
            }
        }
        fprintf(stderr,
                "IPT Davidson auto cluster: step=%d cluster=%zu "
                "source=%s members=%s size=%zu min_abs_gap=%.17g "
                "min_rel_gap=%.17g cluster_max_residual=%.17g "
                "cluster_fro_residual=%.17g correction_members=%s "
                "correction_member_count=%zu "
                "skipped_converged_in_cluster=%s\n",
                step, cluster_index,
                ipt_cluster_source_name(cluster.source),
                ipt_join_indices(cluster.members).c_str(),
                cluster.members.size(), cluster.min_abs_gap,
                cluster.min_rel_gap, maximum, sqrt(fro),
                ipt_join_indices(correction_members).c_str(),
                correction_members.size(),
                ipt_join_indices(skipped_converged).c_str());
    }
}

static int ipt_active_clusters_contain_member(
    const std::vector<IPTActiveCluster> &clusters, int member)
{
    for (const IPTActiveCluster &cluster : clusters) {
        if (ipt_vector_contains_int(cluster.members, member)) {
            return 1;
        }
    }
    return 0;
}

static int ipt_active_clusters_include_exact(
    const std::vector<IPTActiveCluster> &clusters, int first, int second)
{
    for (const IPTActiveCluster &cluster : clusters) {
        if (ipt_cluster_has_exact_members(cluster.members, first, second)) {
            return 1;
        }
    }
    return 0;
}

static void ipt_compute_cluster_overlap_matching(
    int n, int k, const std::vector<double> &old_vectors,
    const std::vector<double> &trial_vectors,
    const std::vector<int> &members, std::vector<int> *assignment,
    std::vector<double> *overlap_matrix)
{
    int size = (int)members.size();

    assignment->assign((size_t)size, -1);
    for (int i = 0; i < size; ++i) {
        (*assignment)[(size_t)i] = i;
    }
    overlap_matrix->assign((size_t)size * (size_t)size, 0.0);
    if (size <= 0 ||
        old_vectors.size() < (size_t)n * (size_t)k ||
        trial_vectors.size() < (size_t)n * (size_t)k) {
        return;
    }
    for (int old_pos = 0; old_pos < size; ++old_pos) {
        int old_member = members[(size_t)old_pos];

        for (int trial_pos = 0; trial_pos < size; ++trial_pos) {
            int trial_member = members[(size_t)trial_pos];
            double overlap = 0.0;

            for (int row = 0; row < n; ++row) {
                overlap +=
                    old_vectors[(size_t)row +
                                (size_t)old_member * (size_t)n] *
                    trial_vectors[(size_t)row +
                                  (size_t)trial_member * (size_t)n];
            }
            (*overlap_matrix)[(size_t)old_pos * (size_t)size +
                              (size_t)trial_pos] = fabs(overlap);
        }
    }
    if (size <= 8) {
        int state_count = 1 << size;
        std::vector<double> score((size_t)state_count, -INFINITY);
        std::vector<int> previous((size_t)state_count, -1);
        std::vector<int> selected((size_t)state_count, -1);

        score[0] = 0.0;
        for (int mask = 0; mask < state_count; ++mask) {
            int old_pos = __builtin_popcount((unsigned int)mask);

            if (!isfinite(score[(size_t)mask]) || old_pos >= size) {
                continue;
            }
            for (int trial_pos = 0; trial_pos < size; ++trial_pos) {
                int bit = 1 << trial_pos;
                int next = mask | bit;
                double candidate = 0.0;

                if ((mask & bit) != 0) {
                    continue;
                }
                candidate =
                    score[(size_t)mask] +
                    (*overlap_matrix)[(size_t)old_pos * (size_t)size +
                                      (size_t)trial_pos];
                if (candidate > score[(size_t)next]) {
                    score[(size_t)next] = candidate;
                    previous[(size_t)next] = mask;
                    selected[(size_t)next] = trial_pos;
                }
            }
        }
        if (isfinite(score[(size_t)state_count - 1U])) {
            int mask = state_count - 1;
            std::vector<int> matched((size_t)size, -1);
            int valid_path = 1;

            for (int old_pos = size - 1; old_pos >= 0; --old_pos) {
                if (mask < 0 || selected[(size_t)mask] < 0 ||
                    previous[(size_t)mask] < 0) {
                    valid_path = 0;
                    break;
                }
                matched[(size_t)old_pos] = selected[(size_t)mask];
                mask = previous[(size_t)mask];
            }
            if (valid_path) {
                assignment->swap(matched);
            }
        }
    } else {
        struct IPTOverlapCandidate {
            double overlap;
            int old_pos;
            int trial_pos;
        };
        std::vector<IPTOverlapCandidate> candidates;
        std::vector<int> old_used((size_t)size, 0);
        std::vector<int> trial_used((size_t)size, 0);

        fprintf(stderr,
                "IPT Davidson: active cluster size %d exceeds exact "
                "matching limit 8; using greedy overlap matching\n",
                size);
        for (int old_pos = 0; old_pos < size; ++old_pos) {
            for (int trial_pos = 0; trial_pos < size; ++trial_pos) {
                IPTOverlapCandidate candidate = {};

                candidate.overlap =
                    (*overlap_matrix)[(size_t)old_pos * (size_t)size +
                                      (size_t)trial_pos];
                candidate.old_pos = old_pos;
                candidate.trial_pos = trial_pos;
                candidates.push_back(candidate);
            }
        }
        std::stable_sort(
            candidates.begin(), candidates.end(),
            [](const IPTOverlapCandidate &a,
               const IPTOverlapCandidate &b) {
                return a.overlap > b.overlap;
            });
        for (const IPTOverlapCandidate &candidate : candidates) {
            if (old_used[(size_t)candidate.old_pos] ||
                trial_used[(size_t)candidate.trial_pos]) {
                continue;
            }
            (*assignment)[(size_t)candidate.old_pos] =
                candidate.trial_pos;
            old_used[(size_t)candidate.old_pos] = 1;
            trial_used[(size_t)candidate.trial_pos] = 1;
        }
    }
}

static int ipt_residual_improved(double before, double after,
                                 double rel_slack, double abs_slack)
{
    double required = 0.0;

    if (!isfinite(before) || !isfinite(after)) {
        return 0;
    }
    required = std::max(abs_slack, fabs(before) * rel_slack);
    return after + required < before;
}

static void ipt_compute_generic_cluster_metrics(
    const IPTActiveCluster &cluster,
    const std::vector<double> &current_residuals,
    const std::vector<double> &trial_residuals,
    const std::vector<int> &assignment,
    const std::vector<double> &overlap_matrix, double converged_tolerance,
    double protect_tolerance, double rel_slack, double abs_slack,
    double locked_degrade_rel_slack, double locked_degrade_abs_slack,
    IPTGenericClusterMetrics *metrics)
{
    metrics->members = cluster.members;
    metrics->assignment_trial_index_for_old_member = assignment;
    metrics->overlap_matrix = overlap_matrix;
    metrics->matched_after_residuals.assign(cluster.members.size(), NAN);
    metrics->cluster_residual_fro_before = 0.0;
    metrics->cluster_residual_fro_after = 0.0;
    metrics->cluster_residual_max_before = 0.0;
    metrics->cluster_residual_max_after = 0.0;
    metrics->active_ok = 0;
    metrics->locked_safe = 1;
    metrics->swap_or_permutation_detected = 0;

    for (size_t old_pos = 0; old_pos < cluster.members.size(); ++old_pos) {
        int old_member = cluster.members[old_pos];
        int trial_pos =
            old_pos < assignment.size() ? assignment[old_pos] : -1;
        int trial_member =
            trial_pos >= 0 && trial_pos < (int)cluster.members.size()
                ? cluster.members[(size_t)trial_pos]
                : -1;
        double before = NAN;
        double after = NAN;

        if (old_member < 0 ||
            old_member >= (int)current_residuals.size() ||
            trial_member < 0 ||
            trial_member >= (int)trial_residuals.size()) {
            metrics->locked_safe = 0;
            continue;
        }
        before = current_residuals[(size_t)old_member];
        after = trial_residuals[(size_t)trial_member];
        metrics->matched_after_residuals[old_pos] = after;
        if (trial_pos != (int)old_pos) {
            metrics->swap_or_permutation_detected = 1;
        }
        if (!isfinite(before) || !isfinite(after)) {
            metrics->locked_safe = 0;
            continue;
        }
        metrics->cluster_residual_fro_before += before * before;
        metrics->cluster_residual_fro_after += after * after;
        metrics->cluster_residual_max_before =
            std::max(metrics->cluster_residual_max_before, before);
        metrics->cluster_residual_max_after =
            std::max(metrics->cluster_residual_max_after, after);
        if (before > converged_tolerance &&
            ipt_residual_improved(before, after, rel_slack, abs_slack)) {
            metrics->active_ok = 1;
        }
        if (before <= protect_tolerance ||
            before <= converged_tolerance) {
            double allowed =
                std::max(before * (1.0 + locked_degrade_rel_slack),
                         locked_degrade_abs_slack);

            if (before > converged_tolerance) {
                allowed = std::max(allowed, protect_tolerance);
            }
            if (after > allowed) {
                metrics->locked_safe = 0;
            }
        }
    }
    metrics->cluster_residual_fro_before =
        sqrt(metrics->cluster_residual_fro_before);
    metrics->cluster_residual_fro_after =
        sqrt(metrics->cluster_residual_fro_after);
    if (ipt_residual_improved(metrics->cluster_residual_fro_before,
                              metrics->cluster_residual_fro_after,
                              rel_slack, abs_slack) ||
        ipt_residual_improved(metrics->cluster_residual_max_before,
                              metrics->cluster_residual_max_after,
                              rel_slack, abs_slack)) {
        metrics->active_ok = 1;
    }
}

static std::vector<IPTGenericClusterMetrics>
ipt_compute_active_cluster_metrics(
    int n, int k, const std::vector<IPTActiveCluster> &active_clusters,
    const std::vector<double> &reference_vectors,
    const std::vector<double> &trial_vectors,
    const std::vector<double> &current_residuals,
    const std::vector<double> &trial_residuals,
    double converged_tolerance, double protect_tolerance,
    double rel_slack, double abs_slack,
    double locked_degrade_rel_slack, double locked_degrade_abs_slack)
{
    std::vector<IPTGenericClusterMetrics> metrics;

    for (const IPTActiveCluster &cluster : active_clusters) {
        IPTGenericClusterMetrics current = {};
        std::vector<int> assignment;
        std::vector<double> overlap_matrix;

        ipt_compute_cluster_overlap_matching(
            n, k, reference_vectors, trial_vectors, cluster.members,
            &assignment, &overlap_matrix);
        ipt_compute_generic_cluster_metrics(
            cluster, current_residuals, trial_residuals, assignment,
            overlap_matrix, converged_tolerance, protect_tolerance,
            rel_slack, abs_slack, locked_degrade_rel_slack,
            locked_degrade_abs_slack, &current);
        metrics.push_back(current);
    }
    return metrics;
}

static void ipt_log_generic_cluster_trial(
    int step, int retry_index,
    const std::vector<IPTActiveCluster> &clusters,
    const std::vector<IPTGenericClusterMetrics> &metrics)
{
    for (size_t cluster_index = 0;
         cluster_index < clusters.size() &&
         cluster_index < metrics.size();
         ++cluster_index) {
        const IPTActiveCluster &cluster = clusters[cluster_index];
        const IPTGenericClusterMetrics &current =
            metrics[cluster_index];
        std::string assignment;

        for (size_t old_pos = 0;
             old_pos < cluster.members.size() &&
             old_pos <
                 current.assignment_trial_index_for_old_member.size();
             ++old_pos) {
            int trial_pos =
                current.assignment_trial_index_for_old_member[old_pos];
            int trial_member =
                trial_pos >= 0 &&
                        trial_pos < (int)cluster.members.size()
                    ? cluster.members[(size_t)trial_pos]
                    : -1;

            if (!assignment.empty()) {
                assignment += ";";
            }
            assignment +=
                std::to_string(cluster.members[old_pos]) + "->" +
                std::to_string(trial_member);
        }
        fprintf(stderr,
                "IPT Davidson cluster trial: step=%d retry=%d "
                "cluster=%zu source=%s members=%s "
                "overlap_assignment=%s fro_before=%.17g "
                "fro_after=%.17g max_before=%.17g max_after=%.17g "
                "active_ok=%d locked_safe=%d permutation=%d\n",
                step, retry_index, cluster_index,
                ipt_cluster_source_name(cluster.source),
                ipt_join_indices(cluster.members).c_str(),
                assignment.c_str(),
                current.cluster_residual_fro_before,
                current.cluster_residual_fro_after,
                current.cluster_residual_max_before,
                current.cluster_residual_max_after,
                current.active_ok, current.locked_safe,
                current.swap_or_permutation_detected);
    }
}

static void ipt_cluster_set_default_metrics(IPTClusterTrialMetrics *metrics);
static void ipt_cluster_28_29_gap(
    int k, const std::vector<double> &ritz_values, double *gap,
    double *relative_gap);

static void ipt_copy_legacy_28_29_cluster_metrics(
    int k, const std::vector<double> &current_values,
    const std::vector<IPTGenericClusterMetrics> &generic_metrics,
    IPTClusterTrialMetrics *legacy)
{
    ipt_cluster_set_default_metrics(legacy);
    for (const IPTGenericClusterMetrics &metrics : generic_metrics) {
        if (!ipt_cluster_has_exact_members(metrics.members, 28, 29)) {
            continue;
        }
        legacy->enabled = 1;
        legacy->active_ok = metrics.active_ok;
        ipt_cluster_28_29_gap(k, current_values, &legacy->gap28_29,
                              &legacy->relative_gap28_29);
        legacy->cluster_residual_fro_before =
            metrics.cluster_residual_fro_before;
        legacy->cluster_residual_fro_after =
            metrics.cluster_residual_fro_after;
        legacy->cluster_residual_max_before =
            metrics.cluster_residual_max_before;
        legacy->cluster_residual_max_after =
            metrics.cluster_residual_max_after;
        if (metrics.overlap_matrix.size() >= 4U) {
            legacy->overlap_28_28 = metrics.overlap_matrix[0];
            legacy->overlap_28_29 = metrics.overlap_matrix[1];
            legacy->overlap_29_28 = metrics.overlap_matrix[2];
            legacy->overlap_29_29 = metrics.overlap_matrix[3];
        }
        legacy->overlap_swap_detected =
            metrics.swap_or_permutation_detected;
        return;
    }
}

static void ipt_cluster_set_default_metrics(IPTClusterTrialMetrics *metrics)
{
    if (metrics == NULL) {
        return;
    }
    metrics->enabled = 0;
    metrics->active_ok = 0;
    metrics->gap28_29 = NAN;
    metrics->relative_gap28_29 = NAN;
    metrics->cluster_residual_fro_before = NAN;
    metrics->cluster_residual_fro_after = NAN;
    metrics->cluster_residual_max_before = NAN;
    metrics->cluster_residual_max_after = NAN;
    metrics->overlap_28_28 = NAN;
    metrics->overlap_28_29 = NAN;
    metrics->overlap_29_28 = NAN;
    metrics->overlap_29_29 = NAN;
    metrics->overlap_swap_detected = 0;
}

static void ipt_cluster_copy_28_29_reference(
    int n, int k, const std::vector<double> &ritz_vectors,
    std::vector<double> *reference_vectors)
{
    reference_vectors->clear();
    if (k <= 29 ||
        ritz_vectors.size() < (size_t)n * (size_t)(k > 0 ? k : 1)) {
        return;
    }
    *reference_vectors = ritz_vectors;
}

static void ipt_cluster_28_29_gap(
    int k, const std::vector<double> &ritz_values, double *gap,
    double *relative_gap)
{
    *gap = NAN;
    *relative_gap = NAN;
    if (k <= 29 || ritz_values.size() <= 29U) {
        return;
    }
    *gap = fabs(ritz_values[29] - ritz_values[28]);
    *relative_gap =
        *gap / std::max(1.0, std::max(fabs(ritz_values[28]),
                                      fabs(ritz_values[29])));
}

static int ipt_cluster_28_29_active(
    int k, const std::vector<double> &ritz_values,
    const std::vector<int> &forced_pairs,
    const std::vector<int> &forced_cluster_pairs, double gap_abs_tol,
    double gap_rel_tol, double *gap, double *relative_gap)
{
    ipt_cluster_28_29_gap(k, ritz_values, gap, relative_gap);
    if (k <= 29) {
        return 0;
    }
    if ((ipt_vector_contains_int(forced_cluster_pairs, 28) &&
         ipt_vector_contains_int(forced_cluster_pairs, 29)) ||
        (ipt_vector_contains_int(forced_pairs, 28) &&
         ipt_vector_contains_int(forced_pairs, 29))) {
        return 1;
    }
    return isfinite(*gap) && isfinite(*relative_gap) &&
           (*gap <= gap_abs_tol || *relative_gap <= gap_rel_tol);
}

static void ipt_cluster_28_29_lock_states(
    const std::vector<double> &residuals, double converged_tolerance,
    int cluster_active, int soft_cluster_locking_enabled,
    int cluster_hard_locked, char *pair28_state, size_t pair28_state_size,
    char *pair29_state, size_t pair29_state_size, int *soft_locked_count)
{
    const char *state28 = "inactive";
    const char *state29 = "inactive";
    int soft_count = 0;

    if (soft_locked_count != NULL) {
        *soft_locked_count = 0;
    }
    if (!cluster_active || residuals.size() <= 29U) {
        snprintf(pair28_state, pair28_state_size, "%s", state28);
        snprintf(pair29_state, pair29_state_size, "%s", state29);
        return;
    }
    if (cluster_hard_locked) {
        state28 = "hard";
        state29 = "hard";
    } else {
        int pair28_converged =
            isfinite(residuals[28]) &&
            residuals[28] <= converged_tolerance;
        int pair29_converged =
            isfinite(residuals[29]) &&
            residuals[29] <= converged_tolerance;

        state28 = pair28_converged ? "converged" : "active";
        state29 = pair29_converged ? "converged" : "active";
        if (soft_cluster_locking_enabled && pair28_converged &&
            !pair29_converged) {
            state28 = "soft";
            ++soft_count;
        }
        if (soft_cluster_locking_enabled && pair29_converged &&
            !pair28_converged) {
            state29 = "soft";
            ++soft_count;
        }
    }
    snprintf(pair28_state, pair28_state_size, "%s", state28);
    snprintf(pair29_state, pair29_state_size, "%s", state29);
    if (soft_locked_count != NULL) {
        *soft_locked_count = soft_count;
    }
}

static void ipt_cluster_compute_28_29_metrics(
    int n, int k, const std::vector<double> &current_values,
    const std::vector<double> &current_residuals,
    const std::vector<double> &trial_vectors,
    const std::vector<double> &trial_residuals,
    const std::vector<double> &reference_vectors, int cluster_active,
    double converged_tolerance, double rel_slack, double abs_slack,
    IPTClusterTrialMetrics *metrics)
{
    IPTActiveCluster cluster = {};
    IPTGenericClusterMetrics generic = {};
    std::vector<int> assignment;
    std::vector<double> overlap_matrix;
    std::vector<IPTGenericClusterMetrics> generic_list;

    ipt_cluster_set_default_metrics(metrics);
    if (metrics == NULL || !cluster_active || k <= 29 ||
        current_residuals.size() <= 29U || trial_residuals.size() <= 29U) {
        return;
    }
    cluster.members.push_back(28);
    cluster.members.push_back(29);
    cluster.explicit_forced = 1;
    cluster.active_by_gap = 0;
    ipt_compute_cluster_overlap_matching(
        n, k, reference_vectors, trial_vectors, cluster.members,
        &assignment, &overlap_matrix);
    ipt_compute_generic_cluster_metrics(
        cluster, current_residuals, trial_residuals, assignment,
        overlap_matrix, converged_tolerance, converged_tolerance,
        rel_slack, abs_slack, rel_slack, abs_slack, &generic);
    generic_list.push_back(generic);
    ipt_copy_legacy_28_29_cluster_metrics(
        k, current_values, generic_list, metrics);
}

static IPTTrialAcceptDecision ipt_block_trial_accept_decision(
    const std::vector<double> &current_residuals,
    const std::vector<double> &trial_residuals,
    const std::vector<int> &active_indices,
    const std::vector<int> &forced_pairs, double current_maximum,
    double trial_maximum, int accept_only_if_improves,
    int relaxed_accept_enabled, double rel_slack, double abs_slack,
    int active_pair_accept, double converged_tolerance,
    double protect_tolerance, double locked_degrade_rel_slack,
    double locked_degrade_abs_slack, int basis_invalid, int ritz_invalid,
    const IPTClusterTrialMetrics *cluster_metrics,
    const std::vector<IPTGenericClusterMetrics> *generic_cluster_metrics,
    int cluster_aware_accept_enabled)
{
    IPTTrialAcceptDecision decision = {};
    const std::vector<int> &focus =
        forced_pairs.empty() ? active_indices : forced_pairs;
    int locked_unsafe_pair = -1;

    decision.relaxed_accept_enabled = relaxed_accept_enabled;
    decision.finite_ok =
        isfinite(trial_maximum) && !basis_invalid && !ritz_invalid;
    decision.global_ok =
        decision.finite_ok &&
        trial_maximum <=
            current_maximum * (1.0 + rel_slack) + abs_slack;
    decision.locked_safe = decision.finite_ok;
    decision.active_ok = 0;
    snprintf(decision.reason, sizeof(decision.reason), "not_evaluated");

    if (decision.finite_ok) {
        for (size_t col = 0; col < current_residuals.size() &&
                             col < trial_residuals.size();
             ++col) {
            double before = current_residuals[col];
            double after = trial_residuals[col];
            int handled_by_generic_cluster = 0;

            if (cluster_aware_accept_enabled &&
                generic_cluster_metrics != NULL) {
                for (const IPTGenericClusterMetrics &metrics :
                     *generic_cluster_metrics) {
                    if (ipt_vector_contains_int(metrics.members,
                                                (int)col)) {
                        handled_by_generic_cluster = 1;
                        break;
                    }
                }
            }
            if (handled_by_generic_cluster) {
                continue;
            }
            if (cluster_aware_accept_enabled && cluster_metrics != NULL &&
                cluster_metrics->enabled && (col == 28U || col == 29U)) {
                if (before > converged_tolerance) {
                    continue;
                }
                if (cluster_metrics->overlap_swap_detected) {
                    after = trial_residuals[col == 28U ? 29U : 28U];
                }
            }
            if (!isfinite(before) || !isfinite(after)) {
                decision.locked_safe = 0;
                locked_unsafe_pair = (int)col;
                break;
            }
            if (before <= protect_tolerance &&
                after >
                    std::max(before * (1.0 + locked_degrade_rel_slack),
                             locked_degrade_abs_slack)) {
                decision.locked_safe = 0;
                locked_unsafe_pair = (int)col;
                break;
            }
        }
    }
    if (decision.finite_ok && cluster_aware_accept_enabled &&
        generic_cluster_metrics != NULL &&
        !generic_cluster_metrics->empty()) {
        for (const IPTGenericClusterMetrics &metrics :
             *generic_cluster_metrics) {
            if (active_pair_accept && metrics.active_ok) {
                decision.active_ok = 1;
            }
            if (!metrics.locked_safe) {
                decision.locked_safe = 0;
            }
        }
    } else if (decision.finite_ok && active_pair_accept &&
        cluster_aware_accept_enabled && cluster_metrics != NULL &&
        cluster_metrics->enabled) {
        decision.active_ok = cluster_metrics->active_ok;
    } else if (decision.finite_ok && active_pair_accept) {
        for (int pair : focus) {
            if (pair < 0 || pair >= (int)current_residuals.size() ||
                pair >= (int)trial_residuals.size()) {
                continue;
            }
            if (current_residuals[(size_t)pair] <= converged_tolerance) {
                continue;
            }
            if (ipt_residual_improved(current_residuals[(size_t)pair],
                                      trial_residuals[(size_t)pair],
                                      rel_slack, abs_slack)) {
                decision.active_ok = 1;
                break;
            }
        }
    }

    if (!decision.finite_ok) {
        decision.accepted = 0;
        snprintf(decision.reason, sizeof(decision.reason), "nonfinite");
    } else if (!decision.locked_safe) {
        decision.accepted = 0;
        if (locked_unsafe_pair >= 0) {
            snprintf(decision.reason, sizeof(decision.reason),
                     "locked_unsafe_pair_%d", locked_unsafe_pair);
        } else {
            snprintf(decision.reason, sizeof(decision.reason),
                     "locked_unsafe");
        }
    } else if (!accept_only_if_improves) {
        decision.accepted = 1;
        snprintf(decision.reason, sizeof(decision.reason), "accept_unchecked");
    } else if (relaxed_accept_enabled) {
        decision.accepted = decision.global_ok || decision.active_ok;
        if (decision.accepted) {
            if (decision.global_ok && decision.active_ok) {
                snprintf(decision.reason, sizeof(decision.reason),
                         "global_ok+active_ok");
            } else if (decision.global_ok) {
                snprintf(decision.reason, sizeof(decision.reason),
                         "global_ok");
            } else {
                snprintf(decision.reason, sizeof(decision.reason),
                         "active_ok");
            }
        } else {
            snprintf(decision.reason, sizeof(decision.reason),
                     "not_improved");
        }
    } else {
        decision.global_ok =
            decision.finite_ok && trial_maximum < current_maximum;
        decision.accepted = decision.global_ok;
        snprintf(decision.reason, sizeof(decision.reason), "%s",
                 decision.accepted ? "strict_global" : "not_improved");
    }
    return decision;
}

static void ipt_block_fill_davidson_history_accept_fields(
    IPTDavidsonHistoryEntry *entry, const IPTTrialAcceptDecision &decision,
    int retry_count, double retry_alpha, double retry_denom_clip,
    int accepted_steps, int rejected_steps, int min_accepted_steps,
    int early_jump_to_continuation)
{
    entry->relaxed_accept_enabled = decision.relaxed_accept_enabled;
    entry->accept_global_ok = decision.global_ok;
    entry->accept_active_ok = decision.active_ok;
    entry->accept_locked_safe = decision.locked_safe;
    snprintf(entry->accept_reason, sizeof(entry->accept_reason), "%s",
             decision.reason);
    entry->reject_retry_count = retry_count;
    entry->retry_alpha = retry_alpha;
    entry->retry_denom_clip = retry_denom_clip;
    entry->accepted_steps = accepted_steps;
    entry->rejected_steps = rejected_steps;
    entry->min_accepted_steps = min_accepted_steps;
    entry->early_jump_to_continuation = early_jump_to_continuation;
}

static void ipt_block_fill_block_history_accept_fields(
    IPTDavidsonBlockHistoryEntry *entry,
    const IPTTrialAcceptDecision &decision, int retry_count,
    double retry_alpha, double retry_denom_clip, int accepted_steps,
    int rejected_steps, int min_accepted_steps,
    int early_jump_to_continuation)
{
    entry->relaxed_accept_enabled = decision.relaxed_accept_enabled;
    entry->accept_global_ok = decision.global_ok;
    entry->accept_active_ok = decision.active_ok;
    entry->accept_locked_safe = decision.locked_safe;
    snprintf(entry->accept_reason, sizeof(entry->accept_reason), "%s",
             decision.reason);
    entry->reject_retry_count = retry_count;
    entry->retry_alpha = retry_alpha;
    entry->retry_denom_clip = retry_denom_clip;
    entry->accepted_steps = accepted_steps;
    entry->rejected_steps = rejected_steps;
    entry->min_accepted_steps = min_accepted_steps;
    entry->early_jump_to_continuation = early_jump_to_continuation;
}

static void ipt_block_fill_jd_history_accept_fields(
    IPTJDLocalHistoryEntry *entry, const IPTTrialAcceptDecision &decision,
    int retry_count, double retry_alpha, double retry_denom_clip,
    int accepted_steps, int rejected_steps, int min_accepted_steps,
    int early_jump_to_continuation)
{
    entry->relaxed_accept_enabled = decision.relaxed_accept_enabled;
    entry->accept_global_ok = decision.global_ok;
    entry->accept_active_ok = decision.active_ok;
    entry->accept_locked_safe = decision.locked_safe;
    snprintf(entry->accept_reason, sizeof(entry->accept_reason), "%s",
             decision.reason);
    entry->reject_retry_count = retry_count;
    entry->retry_alpha = retry_alpha;
    entry->retry_denom_clip = retry_denom_clip;
    entry->accepted_steps = accepted_steps;
    entry->rejected_steps = rejected_steps;
    entry->min_accepted_steps = min_accepted_steps;
    entry->early_jump_to_continuation = early_jump_to_continuation;
}

static void ipt_block_fill_davidson_history_cluster_fields(
    IPTDavidsonHistoryEntry *entry, const IPTClusterTrialMetrics &metrics,
    const char *pair28_lock_state, const char *pair29_lock_state,
    int cluster_hard_locked, int cluster_soft_locked_count)
{
    entry->gap28_29 = metrics.gap28_29;
    entry->relative_gap28_29 = metrics.relative_gap28_29;
    entry->cluster_residual_fro_before =
        metrics.cluster_residual_fro_before;
    entry->cluster_residual_fro_after =
        metrics.cluster_residual_fro_after;
    entry->cluster_residual_max_before =
        metrics.cluster_residual_max_before;
    entry->cluster_residual_max_after =
        metrics.cluster_residual_max_after;
    entry->ritz_overlap_28_28 = metrics.overlap_28_28;
    entry->ritz_overlap_28_29 = metrics.overlap_28_29;
    entry->ritz_overlap_29_28 = metrics.overlap_29_28;
    entry->ritz_overlap_29_29 = metrics.overlap_29_29;
    entry->ritz_overlap_swap_detected = metrics.overlap_swap_detected;
    snprintf(entry->pair28_lock_state, sizeof(entry->pair28_lock_state),
             "%s", pair28_lock_state);
    snprintf(entry->pair29_lock_state, sizeof(entry->pair29_lock_state),
             "%s", pair29_lock_state);
    entry->cluster_hard_locked = cluster_hard_locked;
    entry->cluster_soft_locked_count = cluster_soft_locked_count;
}

static void ipt_block_fill_block_history_cluster_fields(
    IPTDavidsonBlockHistoryEntry *entry,
    const IPTClusterTrialMetrics &metrics, const char *pair28_lock_state,
    const char *pair29_lock_state, int cluster_hard_locked,
    int cluster_soft_locked_count)
{
    entry->gap28_29 = metrics.gap28_29;
    entry->relative_gap28_29 = metrics.relative_gap28_29;
    entry->cluster_residual_fro_before =
        metrics.cluster_residual_fro_before;
    entry->cluster_residual_fro_after =
        metrics.cluster_residual_fro_after;
    entry->cluster_residual_max_before =
        metrics.cluster_residual_max_before;
    entry->cluster_residual_max_after =
        metrics.cluster_residual_max_after;
    entry->ritz_overlap_28_28 = metrics.overlap_28_28;
    entry->ritz_overlap_28_29 = metrics.overlap_28_29;
    entry->ritz_overlap_29_28 = metrics.overlap_29_28;
    entry->ritz_overlap_29_29 = metrics.overlap_29_29;
    entry->ritz_overlap_swap_detected = metrics.overlap_swap_detected;
    snprintf(entry->pair28_lock_state, sizeof(entry->pair28_lock_state),
             "%s", pair28_lock_state);
    snprintf(entry->pair29_lock_state, sizeof(entry->pair29_lock_state),
             "%s", pair29_lock_state);
    entry->cluster_hard_locked = cluster_hard_locked;
    entry->cluster_soft_locked_count = cluster_soft_locked_count;
}

static void ipt_block_fill_jd_history_cluster_fields(
    IPTJDLocalHistoryEntry *entry, const IPTClusterTrialMetrics &metrics,
    const char *pair28_lock_state, const char *pair29_lock_state,
    int cluster_hard_locked, int cluster_soft_locked_count)
{
    entry->gap28_29 = metrics.gap28_29;
    entry->relative_gap28_29 = metrics.relative_gap28_29;
    entry->cluster_residual_fro_before =
        metrics.cluster_residual_fro_before;
    entry->cluster_residual_fro_after =
        metrics.cluster_residual_fro_after;
    entry->cluster_residual_max_before =
        metrics.cluster_residual_max_before;
    entry->cluster_residual_max_after =
        metrics.cluster_residual_max_after;
    entry->ritz_overlap_28_28 = metrics.overlap_28_28;
    entry->ritz_overlap_28_29 = metrics.overlap_28_29;
    entry->ritz_overlap_29_28 = metrics.overlap_29_28;
    entry->ritz_overlap_29_29 = metrics.overlap_29_29;
    entry->ritz_overlap_swap_detected = metrics.overlap_swap_detected;
    snprintf(entry->pair28_lock_state, sizeof(entry->pair28_lock_state),
             "%s", pair28_lock_state);
    snprintf(entry->pair29_lock_state, sizeof(entry->pair29_lock_state),
             "%s", pair29_lock_state);
    entry->cluster_hard_locked = cluster_hard_locked;
    entry->cluster_soft_locked_count = cluster_soft_locked_count;
}

static void ipt_block_store_result_cluster_fields(
    IPTCudaResult *result, const std::vector<double> &ritz_values,
    const std::vector<double> &residuals, int k, int cluster_active,
    int cluster_aware_accept_enabled, int soft_cluster_locking_enabled,
    double converged_tolerance, int cluster_hard_locked)
{
    int soft_count = 0;

    result->cluster_aware_accept_enabled = cluster_aware_accept_enabled;
    result->soft_cluster_locking_enabled = soft_cluster_locking_enabled;
    ipt_cluster_28_29_gap(k, ritz_values, &result->gap28_29,
                          &result->relative_gap28_29);
    if (cluster_active && residuals.size() > 29U) {
        result->cluster_residual_fro =
            sqrt(residuals[28] * residuals[28] +
                 residuals[29] * residuals[29]);
        result->cluster_residual_max =
            std::max(residuals[28], residuals[29]);
    } else {
        result->cluster_residual_fro = NAN;
        result->cluster_residual_max = NAN;
    }
    ipt_cluster_28_29_lock_states(
        residuals, converged_tolerance, cluster_active,
        soft_cluster_locking_enabled, cluster_hard_locked,
        result->pair28_lock_state, sizeof(result->pair28_lock_state),
        result->pair29_lock_state, sizeof(result->pair29_lock_state),
        &soft_count);
    result->cluster_hard_locked = cluster_hard_locked;
    result->cluster_soft_locked_count = soft_count;
}

// 生成初始 n x k 单位列向量
__global__ static void set_identity_kernel(double *x, int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;

    if (idx >= total) {
        return;
    }

    int row = idx % n;
    int col = idx / n;
    x[idx] = (row == col) ? 1.0 : 0.0;
}

// 提取矩阵对角线
__global__ static void extract_diagonal_kernel(const double *matrix,
                                               double *diagonal, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        diagonal[row] = matrix[row + row * n];
    }
}

// 构造 G 更新因子：g[row, col] = 1 / (diagonal[col] - diagonal[row])
__global__ static void build_g_kernel(const double *diagonal, double *g, int n,
                                      int k, double min_gap_allowed,
                                      int *bad_gap)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;

    if (idx >= total) {
        return;
    }

    int row = idx % n;
    int col = idx / n;

    if (row == col) {
        g[idx] = 0.0;
    } else {
        double gap = diagonal[col] - diagonal[row];
        if (!isfinite(gap) || fabs(gap) <= min_gap_allowed) {
            if (bad_gap != NULL) {
                atomicExch(bad_gap, 1);
            }
            g[idx] = NAN;
        } else {
            g[idx] = 1.0 / gap;
        }
    }
}

// 为公式里的 D(ΔZ) 准备每一列的对角修正量。
// y[diag_idx] = (M z_i)_i
// x[diag_idx] = (z_i)_i
// diagonal[col] = D_ii
// 所以这一步是在计算(M z_i)_i - D_ii (z_i)_i = ((M - D) z_i)_i = (Δ z_i)_i
__global__ static void column_diagonal_after_d_kernel(const double *y,
                                                      const double *x,
                                                      const double *diagonal,
                                                      double *column_diagonal,
                                                      int n, int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < k) {
        int diag_idx = col + col * n;
        // column_diagonal[col] 存 D(ΔZ) 的第 col 个对角元素，用它更新非对角元素
        column_diagonal[col] = y[diag_idx] - diagonal[col] * x[diag_idx];
    }
}

// 执行 IPT 迭代更新
__global__ static void ipt_update_kernel(double *y, const double *x,
                                         const double *diagonal,
                                         const double *g,
                                         const double *column_diagonal, int n,
                                         int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;

    if (idx >= total) {
        return;
    }

    int row = idx % n;
    int col = idx / n;

    if (row == col) {
        y[idx] = 1.0;
        return;
    }
    // column_diagonal[col] 存 D(ΔZ) 的第 col 个对角元素
    y[idx] =
        (y[idx] - diagonal[row] * x[idx] - x[idx] * column_diagonal[col]) *
        g[idx];
}

// 从 M * vectors 的对角位置取近似特征值
__global__ static void gather_values_kernel(const double *mx, double *values,
                                            int n, int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < k) {
        values[col] = mx[col + col * n];
    }
}

// 形成 Algorithm 3.2 的固定点残差向量 F(Z) - Z
__global__ static void fixed_point_delta_kernel(double *delta,
                                                const double *next,
                                                const double *current,
                                                int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        delta[idx] = next[idx] - current[idx];
    }
}

__global__ static void extract_csc_diagonal_kernel(const int *col_ptr,
                                                   const int *row_ind,
                                                   const double *values,
                                                   double *diagonal, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= n) {
        return;
    }

    double diag = 0.0;
    for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
        if (row_ind[p] == col) {
            diag = values[p];
            break;
        }
    }

    diagonal[col] = diag;
}

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t cuda_check_status = (call);                                \
        if (cuda_check_status != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(cuda_check_status));                    \
            status = IPT_CUDA_CUDA_ERROR;                                      \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t cublas_check_status = (call);                           \
        if (cublas_check_status != CUBLAS_STATUS_SUCCESS) {                    \
            fprintf(stderr, "cuBLAS error at %s:%d: %s\n", __FILE__,          \
                    __LINE__, cublas_status_name(cublas_check_status));        \
            status = IPT_CUDA_CUBLAS_ERROR;                                    \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

#define CUSPARSE_CHECK(call)                                                   \
    do {                                                                       \
        cusparseStatus_t cusparse_check_status = (call);                       \
        if (cusparse_check_status != CUSPARSE_STATUS_SUCCESS) {                \
            fprintf(stderr, "cuSPARSE error at %s:%d: %s\n", __FILE__,        \
                    __LINE__, cusparse_status_name(cusparse_check_status));    \
            status = IPT_CUDA_CUSPARSE_ERROR;                                  \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

#define CUSOLVER_CHECK(call)                                                   \
    do {                                                                       \
        cusolverStatus_t cusolver_check_status = (call);                       \
        if (cusolver_check_status != CUSOLVER_STATUS_SUCCESS) {                \
            fprintf(stderr, "cuSOLVER error at %s:%d: status=%d\n",           \
                    __FILE__, __LINE__, (int)cusolver_check_status);            \
            status = IPT_CUDA_CUSOLVER_ERROR;                                  \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

static cusparseStatus_t ipt_cusparse_spmm(cusparseHandle_t handle,
                                          cusparseSpMatDescr_t matrix,
                                          cusparseDnMatDescr_t x_desc,
                                          cusparseDnMatDescr_t y_desc,
                                          double *d_x, double *d_y,
                                          void *buffer)
{
    const double one = 1.0;
    const double zero = 0.0;
    cusparseStatus_t status = cusparseDnMatSetValues(x_desc, d_x);

    if (status != CUSPARSE_STATUS_SUCCESS) {
        return status;
    }

    status = cusparseDnMatSetValues(y_desc, d_y);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return status;
    }

    return cusparseSpMM(handle, CUSPARSE_OPERATION_TRANSPOSE,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matrix,
                        x_desc, &zero, y_desc, CUDA_R_64F,
                        CUSPARSE_SPMM_ALG_DEFAULT, buffer);
}

// d_matrix 已经在 GPU -> 调 GPU kernel/cuBLAS -> 结果留在 GPU
static int ipt_cuda_device_impl(const double *d_matrix_col_major, int n, int k,
                                int maxiter, double tol, int use_tolerance,
                                cublasHandle_t handle, double **d_vectors_out,
                                double **d_values_out,
                                int *iterations_done_out,
                                double *fixed_point_residual_out)
{
    int status = IPT_CUDA_SUCCESS;
    int iterations_done = 0;
    size_t vector_elements = 0;
    size_t vector_bytes = 0;
    double *d_x_a = NULL;
    double *d_x_b = NULL;
    double *d_diagonal = NULL;
    double *d_g = NULL;
    double *d_column_diagonal = NULL;
    double *d_delta = NULL;
    double *d_values = NULL;
    double *d_x = NULL;
    double *d_y = NULL;
    double fixed_point_residual = NAN;
    const double one = 1.0;
    const double zero = 0.0;
    const int block_size = 256;

    if (d_vectors_out != NULL) {
        *d_vectors_out = NULL;
    }
    if (d_values_out != NULL) {
        *d_values_out = NULL;
    }
    if (iterations_done_out != NULL) {
        *iterations_done_out = 0;
    }
    if (fixed_point_residual_out != NULL) {
        *fixed_point_residual_out = NAN;
    }

    if (d_matrix_col_major == NULL || handle == NULL || d_vectors_out == NULL ||
        d_values_out == NULL || n <= 0 || k <= 0 || k > n ||
        maxiter < 0 || (use_tolerance && (tol <= 0.0 || maxiter <= 0))) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    vector_elements = (size_t)n * (size_t)k;

    if (vector_elements > (size_t)INT_MAX) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    vector_bytes = vector_elements * sizeof(double);

    CUDA_CHECK(cudaMalloc((void **)&d_x_a, vector_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_x_b, vector_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_diagonal, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_g, vector_bytes));
    CUDA_CHECK(
        cudaMalloc((void **)&d_column_diagonal, (size_t)k * sizeof(double)));
    if (use_tolerance) {
        CUDA_CHECK(cudaMalloc((void **)&d_delta, vector_bytes));
    }
    CUDA_CHECK(cudaMalloc((void **)&d_values, (size_t)k * sizeof(double)));

    {
        int blocks = ((int)vector_elements + block_size - 1) / block_size;
        int diag_blocks = (n + block_size - 1) / block_size;

        set_identity_kernel<<<blocks, block_size>>>(d_x_a, n, k);
        CUDA_CHECK(cudaGetLastError());

        extract_diagonal_kernel<<<diag_blocks, block_size>>>(
            d_matrix_col_major, d_diagonal, n);
        CUDA_CHECK(cudaGetLastError());

        build_g_kernel<<<blocks, block_size>>>(d_diagonal, d_g, n, k, 0.0,
                                               NULL);
        CUDA_CHECK(cudaGetLastError());
    }

    d_x = d_x_a;
    d_y = d_x_b;

    for (int iter = 0; iter < maxiter; ++iter) {
        int col_blocks = (k + block_size - 1) / block_size;
        int vector_blocks = ((int)vector_elements + block_size - 1) / block_size;
        double *tmp = NULL;

        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n,
                                 &one, d_matrix_col_major, n, d_x, n, &zero,
                                 d_y, n));

        column_diagonal_after_d_kernel<<<col_blocks, block_size>>>(
            d_y, d_x, d_diagonal, d_column_diagonal, n, k);
        CUDA_CHECK(cudaGetLastError());

        ipt_update_kernel<<<vector_blocks, block_size>>>(
            d_y, d_x, d_diagonal, d_g, d_column_diagonal, n, k);
        CUDA_CHECK(cudaGetLastError());

        iterations_done = iter + 1;

        if (use_tolerance) {
            double delta_norm = 0.0;
            double current_norm = 0.0;

            fixed_point_delta_kernel<<<vector_blocks, block_size>>>(
                d_delta, d_y, d_x, (int)vector_elements);
            CUDA_CHECK(cudaGetLastError());

            CUBLAS_CHECK(cublasDnrm2(handle, (int)vector_elements, d_delta, 1,
                                     &delta_norm));
            CUBLAS_CHECK(cublasDnrm2(handle, (int)vector_elements, d_x, 1,
                                     &current_norm));

            if (current_norm <= 0.0 || !isfinite(current_norm)) {
                status = IPT_CUDA_INVALID_ARGUMENT;
                goto cleanup;
            }

            fixed_point_residual = delta_norm / current_norm;
            if (fixed_point_residual <= tol) {
                tmp = d_x;
                d_x = d_y;
                d_y = tmp;
                break;
            }
        }

        tmp = d_x;
        d_x = d_y;
        d_y = tmp;
    }

    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &one,
                             d_matrix_col_major, n, d_x, n, &zero, d_y, n));

    {
        int col_blocks = (k + block_size - 1) / block_size;

        gather_values_kernel<<<col_blocks, block_size>>>(d_y, d_values, n, k);
        CUDA_CHECK(cudaGetLastError());
    }

    *d_vectors_out = d_x;
    *d_values_out = d_values;
    if (iterations_done_out != NULL) {
        *iterations_done_out = iterations_done;
    }
    if (fixed_point_residual_out != NULL) {
        *fixed_point_residual_out = fixed_point_residual;
    }
    if (d_x == d_x_a) {
        d_x_a = NULL;
    } else if (d_x == d_x_b) {
        d_x_b = NULL;
    }
    d_values = NULL;

cleanup:
    cudaFree(d_x_a);
    cudaFree(d_x_b);
    cudaFree(d_diagonal);
    cudaFree(d_g);
    cudaFree(d_column_diagonal);
    cudaFree(d_delta);
    cudaFree(d_values);

    if (status != IPT_CUDA_SUCCESS) {
        if (d_vectors_out != NULL) {
            *d_vectors_out = NULL;
        }
        if (d_values_out != NULL) {
            *d_values_out = NULL;
        }
    }

    return status;
}

extern "C" int ipt_cuda_device(const double *d_matrix_col_major, int n, int k,
                                int iterations, cublasHandle_t handle,
                                double **d_vectors_out, double **d_values_out)
{
    return ipt_cuda_device_impl(d_matrix_col_major, n, k, iterations, 0.0, 0,
                                handle, d_vectors_out, d_values_out, NULL,
                                NULL);
}

extern "C" int ipt_cuda_device_tol(const double *d_matrix_col_major, int n,
                                    int k, double tol, int maxiter,
                                    cublasHandle_t handle,
                                    double **d_vectors_out,
                                    double **d_values_out,
                                    int *iterations_done_out,
                                    double *fixed_point_residual_out)
{
    return ipt_cuda_device_impl(d_matrix_col_major, n, k, maxiter, tol, 1,
                                handle, d_vectors_out, d_values_out,
                                iterations_done_out,
                                fixed_point_residual_out);
}

// host matrix -> 拷到 GPU -> 调 GPU kernel/cuBLAS -> 拷回 host
extern "C" int ipt_cuda(const double *matrix_col_major, int n, int k,
                         int iterations, IPTCudaResult *result)
{
    return ipt_cuda_with_initial(matrix_col_major, NULL, n, k, iterations,
                                 result);
}

extern "C" int ipt_cuda_tol(const double *matrix_col_major, int n, int k,
                             double tol, int maxiter, IPTCudaResult *result)
{
    return ipt_cuda_with_initial_tol(matrix_col_major, NULL, n, k, tol,
                                     maxiter, result);
}

static int ipt_cuda_with_initial_impl(const double *matrix_col_major,
                                      const double *initial_vectors_col_major,
                                      int n, int k, int maxiter, double tol,
                                      int use_tolerance,
                                      IPTCudaResult *result)
{
    int status = IPT_CUDA_SUCCESS;
    int iterations_done = 0;
    size_t matrix_elements = 0;
    size_t vector_elements = 0;
    size_t matrix_bytes = 0;
    size_t vector_bytes = 0;
    cublasHandle_t handle = NULL;
    double *d_matrix = NULL;
    double *d_x_a = NULL;
    double *d_x_b = NULL;
    double *d_diagonal = NULL;
    double *d_g = NULL;
    double *d_column_diagonal = NULL;
    double *d_delta = NULL;
    double *d_values = NULL;
    double *d_x = NULL;
    double *d_y = NULL;
    const double one = 1.0;
    const double zero = 0.0;
    const int block_size = 256;

    if (result != NULL) {
        result->n = 0;
        result->k = 0;
        result->iterations = 0;
        result->vectors = NULL;
        result->values = NULL;
        result->basis_cols = 0;
        result->fixed_point_residual = NAN;
    }

    if (matrix_col_major == NULL || result == NULL || n <= 0 || k <= 0 ||
        k > n || maxiter < 0 ||
        (use_tolerance && (tol <= 0.0 || maxiter <= 0))) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    matrix_elements = (size_t)n * (size_t)n;
    vector_elements = (size_t)n * (size_t)k;

    if (matrix_elements > (size_t)INT_MAX ||
        vector_elements > (size_t)INT_MAX) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    matrix_bytes = matrix_elements * sizeof(double);
    vector_bytes = vector_elements * sizeof(double);

    result->vectors = (double *)calloc(vector_elements, sizeof(double));
    result->values = (double *)calloc((size_t)k, sizeof(double));

    if (result->vectors == NULL || result->values == NULL) {
        status = IPT_CUDA_ALLOCATION_FAILED;
        goto cleanup;
    }

    CUDA_CHECK(cudaMalloc((void **)&d_matrix, matrix_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_x_a, vector_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_x_b, vector_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_diagonal, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_g, vector_bytes));
    CUDA_CHECK(
        cudaMalloc((void **)&d_column_diagonal, (size_t)k * sizeof(double)));
    if (use_tolerance) {
        CUDA_CHECK(cudaMalloc((void **)&d_delta, vector_bytes));
    }
    CUDA_CHECK(cudaMalloc((void **)&d_values, (size_t)k * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_matrix, matrix_col_major, matrix_bytes,
                          cudaMemcpyHostToDevice));

    if (initial_vectors_col_major == NULL) {
        int blocks = ((int)vector_elements + block_size - 1) / block_size;
        set_identity_kernel<<<blocks, block_size>>>(d_x_a, n, k);
        CUDA_CHECK(cudaGetLastError());
    } else {
        CUDA_CHECK(cudaMemcpy(d_x_a, initial_vectors_col_major, vector_bytes,
                              cudaMemcpyHostToDevice));
    }

    {
        int diag_blocks = (n + block_size - 1) / block_size;
        int vector_blocks = ((int)vector_elements + block_size - 1) / block_size;

        extract_diagonal_kernel<<<diag_blocks, block_size>>>(d_matrix,
                                                             d_diagonal, n);
        CUDA_CHECK(cudaGetLastError());

        build_g_kernel<<<vector_blocks, block_size>>>(d_diagonal, d_g, n, k,
                                                      0.0, NULL);
        CUDA_CHECK(cudaGetLastError());
    }

    CUBLAS_CHECK(cublasCreate(&handle));

    d_x = d_x_a;
    d_y = d_x_b;

    for (int iter = 0; iter < maxiter; ++iter) {
        int col_blocks = (k + block_size - 1) / block_size;
        int vector_blocks = ((int)vector_elements + block_size - 1) / block_size;
        double *tmp = NULL;

        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n,
                                 &one, d_matrix, n, d_x, n, &zero, d_y, n));

        column_diagonal_after_d_kernel<<<col_blocks, block_size>>>(
            d_y, d_x, d_diagonal, d_column_diagonal, n, k);
        CUDA_CHECK(cudaGetLastError());

        ipt_update_kernel<<<vector_blocks, block_size>>>(
            d_y, d_x, d_diagonal, d_g, d_column_diagonal, n, k);
        CUDA_CHECK(cudaGetLastError());

        iterations_done = iter + 1;

        if (use_tolerance) {
            double delta_norm = 0.0;
            double current_norm = 0.0;
            double fixed_point_residual = 0.0;

            fixed_point_delta_kernel<<<vector_blocks, block_size>>>(
                d_delta, d_y, d_x, (int)vector_elements);
            CUDA_CHECK(cudaGetLastError());

            CUBLAS_CHECK(cublasDnrm2(handle, (int)vector_elements, d_delta, 1,
                                     &delta_norm));
            CUBLAS_CHECK(cublasDnrm2(handle, (int)vector_elements, d_x, 1,
                                     &current_norm));

            if (current_norm <= 0.0 || !isfinite(current_norm)) {
                status = IPT_CUDA_INVALID_ARGUMENT;
                goto cleanup;
            }

            fixed_point_residual = delta_norm / current_norm;
            if (fixed_point_residual <= tol) {
                tmp = d_x;
                d_x = d_y;
                d_y = tmp;
                break;
            }
        }

        tmp = d_x;
        d_x = d_y;
        d_y = tmp;
    }

    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &one,
                             d_matrix, n, d_x, n, &zero, d_y, n));

    {
        int col_blocks = (k + block_size - 1) / block_size;

        gather_values_kernel<<<col_blocks, block_size>>>(d_y, d_values, n, k);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(result->vectors, d_x, vector_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result->values, d_values, (size_t)k * sizeof(double),
                          cudaMemcpyDeviceToHost));

    result->n = n;
    result->k = k;
    result->iterations = iterations_done;
    result->basis_cols = k;

cleanup:
    if (handle != NULL) {
        cublasDestroy(handle);
    }

    cudaFree(d_matrix);
    cudaFree(d_x_a);
    cudaFree(d_x_b);
    cudaFree(d_diagonal);
    cudaFree(d_g);
    cudaFree(d_column_diagonal);
    cudaFree(d_delta);
    cudaFree(d_values);

    if (status != IPT_CUDA_SUCCESS) {
        ipt_cuda_free_result(result);
    }

    return status;
}

extern "C" int ipt_cuda_with_initial(const double *matrix_col_major,
                                      const double *initial_vectors_col_major,
                                      int n, int k, int iterations,
                                      IPTCudaResult *result)
{
    return ipt_cuda_with_initial_impl(matrix_col_major,
                                      initial_vectors_col_major, n, k,
                                      iterations, 0.0, 0, result);
}

extern "C" int ipt_cuda_with_initial_tol(
    const double *matrix_col_major, const double *initial_vectors_col_major,
    int n, int k, double tol, int maxiter, IPTCudaResult *result)
{
    return ipt_cuda_with_initial_impl(matrix_col_major,
                                      initial_vectors_col_major, n, k,
                                      maxiter, tol, 1, result);
}

static int ipt_cuda_sparse_csc_device_impl(
    const int *d_col_ptr, const int *d_row_ind,
    const double *d_matrix_values, int n, int k, int nnz, int maxiter,
    double tol, int use_tolerance, cublasHandle_t handle,
    double **d_vectors_out, double **d_values_out, int *iterations_done_out,
    double *fixed_point_residual_out)
{
    int status = IPT_CUDA_SUCCESS;
    int iterations_done = 0;
    size_t vector_elements = 0;
    size_t vector_bytes = 0;
    double *d_x_a = NULL;
    double *d_x_b = NULL;
    double *d_diagonal = NULL;
    double *d_g = NULL;
    double *d_column_diagonal = NULL;
    double *d_delta = NULL;
    double *d_values = NULL;
    double *d_x = NULL;
    double *d_y = NULL;
    cusparseHandle_t sparse_handle = NULL;
    cusparseSpMatDescr_t sparse_matrix = NULL;
    cusparseDnMatDescr_t dense_x = NULL;
    cusparseDnMatDescr_t dense_y = NULL;
    void *d_spmm_buffer = NULL;
    size_t spmm_buffer_size = 0;
    double fixed_point_residual = NAN;
    const int block_size = 256;

    if (d_vectors_out != NULL) {
        *d_vectors_out = NULL;
    }
    if (d_values_out != NULL) {
        *d_values_out = NULL;
    }
    if (iterations_done_out != NULL) {
        *iterations_done_out = 0;
    }
    if (fixed_point_residual_out != NULL) {
        *fixed_point_residual_out = NAN;
    }

    if (d_col_ptr == NULL || d_row_ind == NULL || d_matrix_values == NULL ||
        handle == NULL || d_vectors_out == NULL || d_values_out == NULL ||
        n <= 0 || k <= 0 || k > n || nnz < 0 || maxiter < 0 ||
        (use_tolerance && (tol <= 0.0 || maxiter <= 0))) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    vector_elements = (size_t)n * (size_t)k;

    if (vector_elements > (size_t)INT_MAX) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    vector_bytes = vector_elements * sizeof(double);

    CUDA_CHECK(cudaMalloc((void **)&d_x_a, vector_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_x_b, vector_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_diagonal, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_g, vector_bytes));
    CUDA_CHECK(
        cudaMalloc((void **)&d_column_diagonal, (size_t)k * sizeof(double)));
    if (use_tolerance) {
        CUDA_CHECK(cudaMalloc((void **)&d_delta, vector_bytes));
    }
    CUDA_CHECK(cudaMalloc((void **)&d_values, (size_t)k * sizeof(double)));

    CUSPARSE_CHECK(cusparseCreate(&sparse_handle));
    CUSPARSE_CHECK(cusparseCreateCsr(
        &sparse_matrix, n, n, nnz, (void *)d_col_ptr, (void *)d_row_ind,
        (void *)d_matrix_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_x, n, k, n, d_x_a, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_y, n, k, n, d_x_b, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    {
        const double one = 1.0;
        const double zero = 0.0;

        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            sparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &one, sparse_matrix, dense_x,
            &zero, dense_y, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
            &spmm_buffer_size));
        if (spmm_buffer_size > 0) {
            CUDA_CHECK(cudaMalloc(&d_spmm_buffer, spmm_buffer_size));
        }
    }

    {
        int vector_blocks = ((int)vector_elements + block_size - 1) / block_size;
        int diag_blocks = (n + block_size - 1) / block_size;

        set_identity_kernel<<<vector_blocks, block_size>>>(d_x_a, n, k);
        CUDA_CHECK(cudaGetLastError());

        extract_csc_diagonal_kernel<<<diag_blocks, block_size>>>(
            d_col_ptr, d_row_ind, d_matrix_values, d_diagonal, n);
        CUDA_CHECK(cudaGetLastError());

        build_g_kernel<<<vector_blocks, block_size>>>(d_diagonal, d_g, n, k,
                                                      0.0, NULL);
        CUDA_CHECK(cudaGetLastError());
    }

    d_x = d_x_a;
    d_y = d_x_b;

    for (int iter = 0; iter < maxiter; ++iter) {
        int col_blocks = (k + block_size - 1) / block_size;
        int vector_blocks = ((int)vector_elements + block_size - 1) / block_size;
        double *tmp = NULL;

        CUSPARSE_CHECK(ipt_cusparse_spmm(sparse_handle, sparse_matrix,
                                         dense_x, dense_y, d_x, d_y,
                                         d_spmm_buffer));

        column_diagonal_after_d_kernel<<<col_blocks, block_size>>>(
            d_y, d_x, d_diagonal, d_column_diagonal, n, k);
        CUDA_CHECK(cudaGetLastError());

        ipt_update_kernel<<<vector_blocks, block_size>>>(
            d_y, d_x, d_diagonal, d_g, d_column_diagonal, n, k);
        CUDA_CHECK(cudaGetLastError());

        iterations_done = iter + 1;

        if (use_tolerance) {
            double delta_norm = 0.0;
            double current_norm = 0.0;

            fixed_point_delta_kernel<<<vector_blocks, block_size>>>(
                d_delta, d_y, d_x, (int)vector_elements);
            CUDA_CHECK(cudaGetLastError());

            CUBLAS_CHECK(cublasDnrm2(handle, (int)vector_elements, d_delta, 1,
                                     &delta_norm));
            CUBLAS_CHECK(cublasDnrm2(handle, (int)vector_elements, d_x, 1,
                                     &current_norm));

            if (current_norm <= 0.0 || !isfinite(current_norm)) {
                status = IPT_CUDA_INVALID_ARGUMENT;
                goto cleanup;
            }

            fixed_point_residual = delta_norm / current_norm;
            if (fixed_point_residual <= tol) {
                tmp = d_x;
                d_x = d_y;
                d_y = tmp;
                break;
            }
        }

        tmp = d_x;
        d_x = d_y;
        d_y = tmp;
    }

    CUSPARSE_CHECK(ipt_cusparse_spmm(sparse_handle, sparse_matrix, dense_x,
                                     dense_y, d_x, d_y, d_spmm_buffer));

    {
        int col_blocks = (k + block_size - 1) / block_size;

        gather_values_kernel<<<col_blocks, block_size>>>(d_y, d_values, n, k);
        CUDA_CHECK(cudaGetLastError());
    }

    *d_vectors_out = d_x;
    *d_values_out = d_values;
    if (iterations_done_out != NULL) {
        *iterations_done_out = iterations_done;
    }
    if (fixed_point_residual_out != NULL) {
        *fixed_point_residual_out = fixed_point_residual;
    }
    if (d_x == d_x_a) {
        d_x_a = NULL;
    } else if (d_x == d_x_b) {
        d_x_b = NULL;
    }
    d_values = NULL;

cleanup:
    cudaFree(d_x_a);
    cudaFree(d_x_b);
    cudaFree(d_diagonal);
    cudaFree(d_g);
    cudaFree(d_column_diagonal);
    cudaFree(d_delta);
    cudaFree(d_values);
    cudaFree(d_spmm_buffer);
    if (dense_x != NULL) {
        cusparseDestroyDnMat(dense_x);
    }
    if (dense_y != NULL) {
        cusparseDestroyDnMat(dense_y);
    }
    if (sparse_matrix != NULL) {
        cusparseDestroySpMat(sparse_matrix);
    }
    if (sparse_handle != NULL) {
        cusparseDestroy(sparse_handle);
    }

    if (status != IPT_CUDA_SUCCESS) {
        if (d_vectors_out != NULL) {
            *d_vectors_out = NULL;
        }
        if (d_values_out != NULL) {
            *d_values_out = NULL;
        }
    }

    return status;
}

extern "C" int ipt_cuda_sparse_csc_device(
    const int *d_col_ptr, const int *d_row_ind,
    const double *d_matrix_values, int n, int k, int nnz, int iterations,
    cublasHandle_t handle, double **d_vectors_out, double **d_values_out)
{
    return ipt_cuda_sparse_csc_device_impl(
        d_col_ptr, d_row_ind, d_matrix_values, n, k, nnz, iterations, 0.0, 0,
        handle, d_vectors_out, d_values_out, NULL, NULL);
}

extern "C" int ipt_cuda_sparse_csc_device_tol(
    const int *d_col_ptr, const int *d_row_ind,
    const double *d_matrix_values, int n, int k, int nnz, double tol,
    int maxiter, cublasHandle_t handle, double **d_vectors_out,
    double **d_values_out, int *iterations_done_out,
    double *fixed_point_residual_out)
{
    return ipt_cuda_sparse_csc_device_impl(
        d_col_ptr, d_row_ind, d_matrix_values, n, k, nnz, maxiter, tol, 1,
        handle, d_vectors_out, d_values_out, iterations_done_out,
        fixed_point_residual_out);
}

extern "C" int ipt_cuda_sparse_csc(
    const int *col_ptr, const int *row_ind, const double *matrix_values, int n,
    int k, int nnz, int iterations, IPTCudaResult *result)
{
    return ipt_cuda_sparse_csc_with_initial(col_ptr, row_ind, matrix_values,
                                            NULL, n, k, nnz, iterations,
                                            result);
}

extern "C" int ipt_cuda_sparse_csc_tol(
    const int *col_ptr, const int *row_ind, const double *matrix_values, int n,
    int k, int nnz, double tol, int maxiter, IPTCudaResult *result)
{
    return ipt_cuda_sparse_csc_with_initial_tol(
        col_ptr, row_ind, matrix_values, NULL, n, k, nnz, tol, maxiter,
        result);
}

__global__ static void ipt_block_cluster_identity_kernel(
    double *basis, int n, int m, int first)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * m;

    if (idx < total) {
        int row = idx % n;
        int col = idx / n;
        basis[idx] = row == first + col ? 1.0 : 0.0;
    }
}

__global__ static void ipt_block_cluster_gather_h_kernel(
    const double *y, double *h, int n, int m, int first)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < m * m) {
        int row = idx % m;
        int col = idx / m;
        h[idx] = y[first + row + col * n];
    }
}

__global__ static void ipt_block_cluster_update_kernel(
    const double *basis, const double *y, const double *diagonal,
    const double *h, double *next_basis, int n, int m, int first, int last,
    double damping, int *singular)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ unsigned char shared_raw[];
    double *aug = reinterpret_cast<double *>(shared_raw);
    int *meta = reinterpret_cast<int *>(aug + (size_t)m * (size_t)(m + 1));

    if (row >= n) {
        return;
    }
    if (row >= first && row <= last) {
        if (tid < m) {
            next_basis[row + tid * n] = row == first + tid ? 1.0 : 0.0;
        }
        return;
    }

    for (int equation = tid; equation < m; equation += blockDim.x) {
        for (int unknown = 0; unknown < m; ++unknown) {
            aug[equation * (m + 1) + unknown] =
                (equation == unknown ? diagonal[row] : 0.0) -
                h[unknown + equation * m];
        }
        aug[equation * (m + 1) + m] =
            diagonal[row] * basis[row + equation * n] -
            y[row + equation * n];
    }
    if (tid == 0) {
        meta[0] = 0;
        meta[1] = 0;
    }
    __syncthreads();

    for (int pivot_col = 0; pivot_col < m; ++pivot_col) {
        if (tid == 0) {
            int pivot_row = pivot_col;
            double pivot_abs =
                fabs(aug[pivot_col * (m + 1) + pivot_col]);

            for (int candidate = pivot_col + 1; candidate < m; ++candidate) {
                double candidate_abs =
                    fabs(aug[candidate * (m + 1) + pivot_col]);
                if (candidate_abs > pivot_abs) {
                    pivot_abs = candidate_abs;
                    pivot_row = candidate;
                }
            }
            meta[0] = pivot_row;
            if (!isfinite(pivot_abs) || pivot_abs <= 64.0 * DBL_EPSILON) {
                meta[1] = 1;
                atomicExch(singular, 1);
            }
        }
        __syncthreads();
        if (meta[1]) {
            return;
        }

        if (meta[0] != pivot_col) {
            for (int col = tid; col <= m; col += blockDim.x) {
                double temp = aug[pivot_col * (m + 1) + col];
                aug[pivot_col * (m + 1) + col] =
                    aug[meta[0] * (m + 1) + col];
                aug[meta[0] * (m + 1) + col] = temp;
            }
        }
        __syncthreads();

        {
            double pivot = aug[pivot_col * (m + 1) + pivot_col];
            for (int col = tid; col <= m; col += blockDim.x) {
                aug[pivot_col * (m + 1) + col] /= pivot;
            }
        }
        __syncthreads();

        for (int equation = tid; equation < m; equation += blockDim.x) {
            if (equation != pivot_col) {
                double factor = aug[equation * (m + 1) + pivot_col];
                for (int col = pivot_col; col <= m; ++col) {
                    aug[equation * (m + 1) + col] -=
                        factor * aug[pivot_col * (m + 1) + col];
                }
            }
        }
        __syncthreads();
    }

    if (tid < m) {
        double value = aug[tid * (m + 1) + m];
        if (damping < 1.0) {
            value = (1.0 - damping) * basis[row + tid * n] +
                    damping * value;
        }
        if (!isfinite(value)) {
            atomicExch(singular, 1);
            value = 0.0;
        }
        next_basis[row + tid * n] = value;
    }
}

__global__ static void ipt_block_cluster_unpermute_kernel(
    const double *sorted_vectors, double *vectors, const int *perm, int n,
    int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n * k) {
        int sorted_row = idx % n;
        int col = idx / n;
        vectors[perm[sorted_row] + col * n] = sorted_vectors[idx];
    }
}

static double ipt_block_elapsed_seconds(
    const std::chrono::steady_clock::time_point &start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
        .count();
}

static void ipt_block_orthogonality_stats(
    const std::vector<double> &vectors, int n, int cols,
    double *frobenius_error, double *max_abs_error, int *has_nan_or_inf)
{
    double frobenius_sq = 0.0;
    double max_error = 0.0;
    int invalid = 0;

    for (size_t i = 0; i < vectors.size(); ++i) {
        if (!isfinite(vectors[i])) {
            invalid = 1;
            break;
        }
    }
    if (!invalid) {
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < cols; ++row) {
                double dot = 0.0;

                for (int i = 0; i < n; ++i) {
                    dot += vectors[(size_t)i + (size_t)row * (size_t)n] *
                           vectors[(size_t)i + (size_t)col * (size_t)n];
                }
                {
                    double error = dot - (row == col ? 1.0 : 0.0);

                    frobenius_sq += error * error;
                    max_error = std::max(max_error, fabs(error));
                }
            }
        }
    }
    *has_nan_or_inf = invalid;
    *frobenius_error = invalid ? NAN : sqrt(frobenius_sq);
    *max_abs_error = invalid ? NAN : max_error;
}

static double ipt_block_host_matrix_inf_norm(
    const int *col_ptr, const int *row_ind, const double *values, int n)
{
    std::vector<double> row_sum((size_t)n, 0.0);
    double norm = 0.0;

    for (int col = 0; col < n; ++col) {
        for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
            row_sum[(size_t)row_ind[p]] += fabs(values[p]);
        }
    }
    for (int row = 0; row < n; ++row) {
        norm = std::max(norm, row_sum[(size_t)row]);
    }
    return norm;
}

static double ipt_block_host_pair_residual(
    const int *col_ptr, const int *row_ind, const double *values, int n,
    const double *vector, double eigenvalue, double matrix_norm,
    std::vector<double> *residual_vector)
{
    std::vector<double> av((size_t)n, 0.0);
    double residual_norm_sq = 0.0;
    double vector_norm_sq = 0.0;

    for (int col = 0; col < n; ++col) {
        double x = vector[col];

        for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
            av[(size_t)row_ind[p]] += values[p] * x;
        }
    }
    if (residual_vector != NULL) {
        residual_vector->assign((size_t)n, 0.0);
    }
    for (int row = 0; row < n; ++row) {
        double residual = av[(size_t)row] - eigenvalue * vector[row];

        if (residual_vector != NULL) {
            (*residual_vector)[(size_t)row] = residual;
        }
        residual_norm_sq += residual * residual;
        vector_norm_sq += vector[row] * vector[row];
    }
    return sqrt(residual_norm_sq) /
           (std::max(1.0, matrix_norm) *
            std::max(1.0, sqrt(vector_norm_sq)));
}

static double ipt_block_host_max_residual(
    const int *col_ptr, const int *row_ind, const double *values, int n,
    int k, const std::vector<double> &ritz_values,
    const std::vector<double> &ritz_vectors, double matrix_norm,
    int *max_index)
{
    double maximum = NAN;

    *max_index = -1;
    for (int col = 0; col < k; ++col) {
        double residual = ipt_block_host_pair_residual(
            col_ptr, row_ind, values, n,
            ritz_vectors.data() + (size_t)col * (size_t)n,
            ritz_values[(size_t)col], matrix_norm, NULL);

        if (!isfinite(maximum) || residual > maximum) {
            maximum = residual;
            *max_index = col;
        }
    }
    return maximum;
}

static void ipt_block_host_residuals(
    const int *col_ptr, const int *row_ind, const double *values, int n,
    int k, const std::vector<double> &ritz_values,
    const std::vector<double> &ritz_vectors, double matrix_norm,
    std::vector<double> *residuals, double *maximum, int *max_index)
{
    residuals->assign((size_t)k, NAN);
    *maximum = NAN;
    *max_index = -1;
    for (int col = 0; col < k; ++col) {
        double residual = ipt_block_host_pair_residual(
            col_ptr, row_ind, values, n,
            ritz_vectors.data() + (size_t)col * (size_t)n,
            ritz_values[(size_t)col], matrix_norm, NULL);

        (*residuals)[(size_t)col] = residual;
        if (!isfinite(*maximum) || residual > *maximum) {
            *maximum = residual;
            *max_index = col;
        }
    }
}

static int ipt_sort_host_ritz_pairs(
    int n, int k, std::vector<double> *ritz_values,
    std::vector<double> *ritz_vectors)
{
    std::vector<int> order((size_t)k, 0);
    int changed = 0;

    if (k <= 1 || ritz_values->size() < (size_t)k ||
        ritz_vectors->size() < (size_t)n * (size_t)k) {
        return 0;
    }
    for (int col = 0; col < k; ++col) {
        order[(size_t)col] = col;
        if (!isfinite((*ritz_values)[(size_t)col])) {
            return 0;
        }
    }
    std::stable_sort(
        order.begin(), order.end(),
        [&](int left, int right) {
            return (*ritz_values)[(size_t)left] <
                   (*ritz_values)[(size_t)right];
        });
    for (int col = 0; col < k; ++col) {
        if (order[(size_t)col] != col) {
            changed = 1;
            break;
        }
    }
    if (changed) {
        std::vector<double> sorted_values((size_t)k, 0.0);
        std::vector<double> sorted_vectors(
            (size_t)n * (size_t)k, 0.0);

        for (int col = 0; col < k; ++col) {
            int old_col = order[(size_t)col];

            sorted_values[(size_t)col] =
                (*ritz_values)[(size_t)old_col];
            std::copy(
                ritz_vectors->begin() + (size_t)old_col * (size_t)n,
                ritz_vectors->begin() +
                    (size_t)(old_col + 1) * (size_t)n,
                sorted_vectors.begin() + (size_t)col * (size_t)n);
        }
        ritz_values->swap(sorted_values);
        ritz_vectors->swap(sorted_vectors);
        fprintf(stderr,
                "IPT Davidson: Rayleigh-Ritz output required host-side "
                "ascending Ritz pair reorder\n");
    }
    return changed;
}

static void ipt_block_cluster_28_29_full_residual(
    const int *col_ptr, const int *row_ind, const double *values, int n,
    int k, const std::vector<double> &ritz_vectors, double matrix_norm,
    double *cluster_fro, double *cluster_max)
{
    std::vector<double> ax28((size_t)n, 0.0);
    std::vector<double> ax29((size_t)n, 0.0);
    double theta28_28 = 0.0;
    double theta28_29 = 0.0;
    double theta29_28 = 0.0;
    double theta29_29 = 0.0;
    double residual28_sq = 0.0;
    double residual29_sq = 0.0;
    double norm28_sq = 0.0;
    double norm29_sq = 0.0;
    double scale28 = 1.0;
    double scale29 = 1.0;
    double cluster_scale = 1.0;

    *cluster_fro = NAN;
    *cluster_max = NAN;
    if (k <= 29 ||
        ritz_vectors.size() < (size_t)n * (size_t)(k > 0 ? k : 1)) {
        return;
    }
    for (int col = 0; col < n; ++col) {
        double x28 =
            ritz_vectors[(size_t)col + (size_t)28 * (size_t)n];
        double x29 =
            ritz_vectors[(size_t)col + (size_t)29 * (size_t)n];

        for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
            int row = row_ind[p];
            double a = values[p];

            ax28[(size_t)row] += a * x28;
            ax29[(size_t)row] += a * x29;
        }
    }
    for (int row = 0; row < n; ++row) {
        double x28 =
            ritz_vectors[(size_t)row + (size_t)28 * (size_t)n];
        double x29 =
            ritz_vectors[(size_t)row + (size_t)29 * (size_t)n];

        theta28_28 += x28 * ax28[(size_t)row];
        theta28_29 += x28 * ax29[(size_t)row];
        theta29_28 += x29 * ax28[(size_t)row];
        theta29_29 += x29 * ax29[(size_t)row];
        norm28_sq += x28 * x28;
        norm29_sq += x29 * x29;
    }
    for (int row = 0; row < n; ++row) {
        double x28 =
            ritz_vectors[(size_t)row + (size_t)28 * (size_t)n];
        double x29 =
            ritz_vectors[(size_t)row + (size_t)29 * (size_t)n];
        double residual28 =
            ax28[(size_t)row] - x28 * theta28_28 -
            x29 * theta29_28;
        double residual29 =
            ax29[(size_t)row] - x28 * theta28_29 -
            x29 * theta29_29;

        residual28_sq += residual28 * residual28;
        residual29_sq += residual29 * residual29;
    }
    scale28 = std::max(1.0, matrix_norm) *
              std::max(1.0, sqrt(norm28_sq));
    scale29 = std::max(1.0, matrix_norm) *
              std::max(1.0, sqrt(norm29_sq));
    cluster_scale = std::max(1.0, matrix_norm) *
                    std::max(1.0, sqrt(norm28_sq + norm29_sq));
    *cluster_fro = sqrt(residual28_sq + residual29_sq) / cluster_scale;
    *cluster_max =
        std::max(sqrt(residual28_sq) / scale28,
                 sqrt(residual29_sq) / scale29);
}

static void ipt_cluster_refresh_full_residual_metrics(
    const int *col_ptr, const int *row_ind, const double *values, int n,
    int k, const std::vector<double> &current_vectors,
    const std::vector<double> &trial_vectors, double matrix_norm,
    double rel_slack, double abs_slack, IPTClusterTrialMetrics *metrics)
{
    double fro_before = NAN;
    double fro_after = NAN;
    double max_before = NAN;
    double max_after = NAN;

    if (metrics == NULL || !metrics->enabled) {
        return;
    }
    ipt_block_cluster_28_29_full_residual(
        col_ptr, row_ind, values, n, k, current_vectors, matrix_norm,
        &fro_before, &max_before);
    ipt_block_cluster_28_29_full_residual(
        col_ptr, row_ind, values, n, k, trial_vectors, matrix_norm,
        &fro_after, &max_after);
    if (isfinite(fro_before) && isfinite(fro_after)) {
        metrics->cluster_residual_fro_before = fro_before;
        metrics->cluster_residual_fro_after = fro_after;
        if (ipt_residual_improved(fro_before, fro_after, rel_slack,
                                  abs_slack)) {
            metrics->active_ok = 1;
        }
    }
    if (isfinite(max_before) && isfinite(max_after)) {
        metrics->cluster_residual_max_before = max_before;
        metrics->cluster_residual_max_after = max_after;
        if (ipt_residual_improved(max_before, max_after, rel_slack,
                                  abs_slack)) {
            metrics->active_ok = 1;
        }
    }
}

static int ipt_block_orthogonalize_direction(
    std::vector<double> *direction, const std::vector<double> &basis, int n,
    int basis_cols, int repeats)
{
    double norm_sq = 0.0;

    for (int pass = 0; pass < repeats; ++pass) {
        for (int col = 0; col < basis_cols; ++col) {
            const double *vector =
                basis.data() + (size_t)col * (size_t)n;
            double dot = 0.0;

            for (int row = 0; row < n; ++row) {
                dot += vector[row] * (*direction)[(size_t)row];
            }
            for (int row = 0; row < n; ++row) {
                (*direction)[(size_t)row] -= dot * vector[row];
            }
        }
    }
    for (int row = 0; row < n; ++row) {
        if (!isfinite((*direction)[(size_t)row])) {
            return 0;
        }
        norm_sq += (*direction)[(size_t)row] *
                   (*direction)[(size_t)row];
    }
    if (!isfinite(norm_sq) || norm_sq <= DBL_EPSILON) {
        return 0;
    }
    {
        double inverse_norm = 1.0 / sqrt(norm_sq);

        for (int row = 0; row < n; ++row) {
            (*direction)[(size_t)row] *= inverse_norm;
        }
    }
    return 1;
}

static double ipt_block_vector_norm(const std::vector<double> &vector)
{
    double norm_sq = 0.0;

    for (double value : vector) {
        if (!isfinite(value)) {
            return NAN;
        }
        norm_sq += value * value;
    }
    return sqrt(norm_sq);
}

static double ipt_block_max_abs_overlap(
    const std::vector<double> &direction,
    const std::vector<double> &basis, int n, int basis_cols)
{
    double maximum = 0.0;

    if (direction.size() < (size_t)n ||
        basis.size() < (size_t)n * (size_t)basis_cols) {
        return INFINITY;
    }
    for (int col = 0; col < basis_cols; ++col) {
        double dot = 0.0;
        const double *vector =
            basis.data() + (size_t)col * (size_t)n;

        for (int row = 0; row < n; ++row) {
            dot += vector[row] * direction[(size_t)row];
        }
        if (!isfinite(dot)) {
            return INFINITY;
        }
        maximum = std::max(maximum, fabs(dot));
    }
    return maximum;
}

static void ipt_block_project_out(
    std::vector<double> *direction, const std::vector<double> &basis, int n,
    int basis_cols, int repeats)
{
    for (int pass = 0; pass < repeats; ++pass) {
        for (int col = 0; col < basis_cols; ++col) {
            const double *vector =
                basis.data() + (size_t)col * (size_t)n;
            double dot = 0.0;

            for (int row = 0; row < n; ++row) {
                dot += vector[row] * (*direction)[(size_t)row];
            }
            for (int row = 0; row < n; ++row) {
                (*direction)[(size_t)row] -= dot * vector[row];
            }
        }
    }
}

static int ipt_block_normalize_direction(std::vector<double> *direction,
                                         double reference_norm,
                                         double *norm_out)
{
    double norm = ipt_block_vector_norm(*direction);

    if (norm_out != NULL) {
        *norm_out = norm;
    }
    if (!isfinite(norm) || norm <= DBL_MIN ||
        (isfinite(reference_norm) && reference_norm > 0.0 &&
         norm <= reference_norm * 1.0e-12)) {
        return 0;
    }
    for (double &value : *direction) {
        value /= norm;
    }
    return 1;
}

static int ipt_block_build_orthonormal_basis(
    const std::vector<double> &basis, int n, int basis_cols, int repeats,
    std::vector<double> *orthonormal_basis)
{
    orthonormal_basis->clear();
    orthonormal_basis->reserve((size_t)n * (size_t)basis_cols);
    for (int col = 0; col < basis_cols; ++col) {
        std::vector<double> direction(
            basis.begin() + (size_t)col * (size_t)n,
            basis.begin() + (size_t)(col + 1) * (size_t)n);
        int existing_cols =
            (int)(orthonormal_basis->size() / (size_t)n);

        if (!ipt_block_orthogonalize_direction(
                &direction, *orthonormal_basis, n, existing_cols,
                repeats)) {
            return 0;
        }
        orthonormal_basis->insert(orthonormal_basis->end(),
                                  direction.begin(), direction.end());
    }
    return 1;
}

static int ipt_block_build_davidson_correction(
    const int *col_ptr, const int *row_ind, const double *values, int n,
    const std::vector<double> &diagonal,
    const std::vector<double> &ritz_values,
    const std::vector<double> &ritz_vectors, int target_index,
    double denom_clip, std::vector<double> *correction)
{
    std::vector<double> residual;
    const double *target =
        ritz_vectors.data() + (size_t)target_index * (size_t)n;

    ipt_block_host_pair_residual(
        col_ptr, row_ind, values, n, target,
        ritz_values[(size_t)target_index], 1.0, &residual);
    correction->assign((size_t)n, 0.0);
    for (int row = 0; row < n; ++row) {
        double denominator =
            diagonal[(size_t)row] - ritz_values[(size_t)target_index];

        if (fabs(denominator) < denom_clip) {
            denominator = denominator < 0.0 ? -denom_clip : denom_clip;
        }
        (*correction)[(size_t)row] =
            residual[(size_t)row] / denominator;
    }
    return isfinite(ipt_block_vector_norm(*correction));
}

static std::vector<int> ipt_jd_parse_active_pairs(const char *raw, int k)
{
    std::vector<int> pairs;
    const char *cursor = raw;

    if (cursor == NULL || cursor[0] == '\0') {
        if (k > 0) {
            pairs.push_back(k - 1);
        }
        return pairs;
    }
    while (*cursor != '\0') {
        char *end = NULL;
        long value = 0;

        while (*cursor == ',' || *cursor == ' ' || *cursor == '\t') {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }
        value = strtol(cursor, &end, 10);
        if (end == cursor) {
            while (*cursor != '\0' && *cursor != ',') {
                ++cursor;
            }
            continue;
        }
        if (value >= 0 && value < k &&
            std::find(pairs.begin(), pairs.end(), (int)value) ==
                pairs.end()) {
            pairs.push_back((int)value);
        }
        cursor = end;
    }
    return pairs;
}

static int ipt_jd_dense_solve(std::vector<double> *matrix,
                              std::vector<double> *rhs, int n)
{
    double scale = 0.0;

    for (double value : *matrix) {
        scale = std::max(scale, fabs(value));
    }
    scale = std::max(1.0, scale);
    for (int pivot = 0; pivot < n; ++pivot) {
        int pivot_row = pivot;
        double pivot_abs =
            fabs((*matrix)[(size_t)pivot * (size_t)n + (size_t)pivot]);

        for (int row = pivot + 1; row < n; ++row) {
            double candidate =
                fabs((*matrix)[(size_t)row * (size_t)n +
                               (size_t)pivot]);

            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        if (!isfinite(pivot_abs) || pivot_abs <= DBL_EPSILON * scale) {
            return 0;
        }
        if (pivot_row != pivot) {
            for (int col = pivot; col < n; ++col) {
                std::swap(
                    (*matrix)[(size_t)pivot * (size_t)n + (size_t)col],
                    (*matrix)[(size_t)pivot_row * (size_t)n +
                              (size_t)col]);
            }
            std::swap((*rhs)[(size_t)pivot],
                      (*rhs)[(size_t)pivot_row]);
        }
        for (int row = pivot + 1; row < n; ++row) {
            double factor =
                (*matrix)[(size_t)row * (size_t)n + (size_t)pivot] /
                (*matrix)[(size_t)pivot * (size_t)n + (size_t)pivot];

            (*matrix)[(size_t)row * (size_t)n + (size_t)pivot] = 0.0;
            for (int col = pivot + 1; col < n; ++col) {
                (*matrix)[(size_t)row * (size_t)n + (size_t)col] -=
                    factor *
                    (*matrix)[(size_t)pivot * (size_t)n + (size_t)col];
            }
            (*rhs)[(size_t)row] -= factor * (*rhs)[(size_t)pivot];
        }
    }
    for (int row = n - 1; row >= 0; --row) {
        double value = (*rhs)[(size_t)row];

        for (int col = row + 1; col < n; ++col) {
            value -=
                (*matrix)[(size_t)row * (size_t)n + (size_t)col] *
                (*rhs)[(size_t)col];
        }
        (*rhs)[(size_t)row] =
            value /
            (*matrix)[(size_t)row * (size_t)n + (size_t)row];
        if (!isfinite((*rhs)[(size_t)row])) {
            return 0;
        }
    }
    return 1;
}

static int ipt_block_build_local_jd_correction(
    const int *original_col_ptr, const int *original_row_ind,
    const double *original_values, const std::vector<int> &sorted_col_ptr,
    const std::vector<int> &sorted_row_ind,
    const std::vector<double> &sorted_values,
    const std::vector<double> &sorted_diagonal,
    const std::vector<int> &perm, int n, int k,
    const std::vector<double> &ritz_values,
    const std::vector<double> &ritz_vectors,
    const std::vector<double> &ritz_residuals, int target_index,
    int window_start, int window_end, double damping,
    int use_residual_support, int support_top, int max_dim,
    int outside_diagonal, double locked_tolerance, int ortho_repeats,
    const std::vector<double> &orthonormal_basis,
    std::vector<double> *correction)
{
    std::vector<double> residual;
    std::vector<double> sorted_residual((size_t)n, 0.0);
    std::vector<int> local_indices;
    std::vector<int> local_position((size_t)n, -1);
    std::vector<double> local_matrix;
    std::vector<double> local_rhs;
    std::vector<double> locked_basis;
    const double *target =
        ritz_vectors.data() + (size_t)target_index * (size_t)n;
    double eigenvalue = ritz_values[(size_t)target_index];
    double clip = std::max(damping, 1.0e-14);

    ipt_block_host_pair_residual(
        original_col_ptr, original_row_ind, original_values, n, target,
        eigenvalue, 1.0, &residual);
    for (int sorted = 0; sorted < n; ++sorted) {
        sorted_residual[(size_t)sorted] =
            residual[(size_t)perm[(size_t)sorted]];
    }
    if (use_residual_support) {
        local_indices.resize((size_t)n);
        for (int sorted = 0; sorted < n; ++sorted) {
            local_indices[(size_t)sorted] = sorted;
        }
        std::stable_sort(
            local_indices.begin(), local_indices.end(),
            [&](int a, int b) {
                return fabs(sorted_residual[(size_t)a]) >
                       fabs(sorted_residual[(size_t)b]);
            });
        local_indices.resize(
            (size_t)std::max(
                1, std::min(n, std::min(max_dim, support_top))));
    } else {
        int last =
            std::min(window_end, window_start + max_dim - 1);

        for (int sorted = window_start; sorted <= last; ++sorted) {
            local_indices.push_back(sorted);
        }
    }
    {
        int local_dim = (int)local_indices.size();

        local_matrix.assign(
            (size_t)local_dim * (size_t)local_dim, 0.0);
        local_rhs.assign((size_t)local_dim, 0.0);
        for (int local = 0; local < local_dim; ++local) {
            local_position[(size_t)local_indices[(size_t)local]] = local;
        }
    }
    int local_dim = (int)local_indices.size();
    for (int local_col = 0; local_col < local_dim; ++local_col) {
        int sorted_col = local_indices[(size_t)local_col];

        for (int p = sorted_col_ptr[(size_t)sorted_col];
             p < sorted_col_ptr[(size_t)sorted_col + 1U]; ++p) {
            int sorted_row = sorted_row_ind[(size_t)p];
            int local_row = local_position[(size_t)sorted_row];

            if (local_row >= 0) {
                local_matrix[(size_t)local_row * (size_t)local_dim +
                             (size_t)local_col] =
                    sorted_values[(size_t)p];
            }
        }
        local_matrix[(size_t)local_col * (size_t)local_dim +
                     (size_t)local_col] +=
            -eigenvalue + damping;
        local_rhs[(size_t)local_col] =
            -sorted_residual[(size_t)sorted_col];
    }
    if (!ipt_jd_dense_solve(&local_matrix, &local_rhs, local_dim)) {
        if (ipt_preparation_debug_enabled()) {
            fprintf(stderr,
                    "IPT local JD: dense solve failed pair=%d "
                    "set=%s dim=%d window=%d:%d damping=%.3e\n",
                    target_index,
                    use_residual_support ? "residual_support" : "window",
                    local_dim, window_start, window_end, damping);
        }
        return 0;
    }

    correction->assign((size_t)n, 0.0);
    if (outside_diagonal) {
        for (int sorted = 0; sorted < n; ++sorted) {
            double denominator =
                sorted_diagonal[(size_t)sorted] - eigenvalue;

            if (local_position[(size_t)sorted] >= 0) {
                continue;
            }
            if (fabs(denominator) < clip) {
                denominator = denominator < 0.0 ? -clip : clip;
            }
            (*correction)[(size_t)sorted] =
                -sorted_residual[(size_t)sorted] / denominator;
        }
    }
    for (int local = 0; local < local_dim; ++local) {
        (*correction)[(size_t)local_indices[(size_t)local]] =
            local_rhs[(size_t)local];
    }

    for (int pair = 0; pair < k; ++pair) {
        if (ritz_residuals[(size_t)pair] > locked_tolerance) {
            continue;
        }
        for (int sorted = 0; sorted < n; ++sorted) {
            locked_basis.push_back(
                ritz_vectors[(size_t)perm[(size_t)sorted] +
                             (size_t)pair * (size_t)n]);
        }
    }
    if (!locked_basis.empty() &&
        !ipt_block_orthogonalize_direction(
            correction, locked_basis, n,
            (int)(locked_basis.size() / (size_t)n), ortho_repeats)) {
        if (ipt_preparation_debug_enabled()) {
            fprintf(stderr,
                    "IPT local JD: locked-vector orthogonalization "
                    "failed pair=%d locked=%zu\n",
                    target_index,
                    locked_basis.size() / (size_t)n);
        }
        return 0;
    }
    if (!ipt_block_orthogonalize_direction(
            correction, orthonormal_basis, n,
            (int)(orthonormal_basis.size() / (size_t)n),
            ortho_repeats)) {
        if (ipt_preparation_debug_enabled()) {
            fprintf(stderr,
                    "IPT local JD: full-basis orthogonalization "
                    "failed pair=%d basis_cols=%zu\n",
                    target_index,
                    orthonormal_basis.size() / (size_t)n);
        }
        return 0;
    }
    return 1;
}

static int ipt_block_cluster_solve_basis_gpu(
    cusparseHandle_t sparse_handle, cusparseSpMatDescr_t sparse_matrix,
    cublasHandle_t cublas_handle, const double *d_diagonal, int n, int m,
    int first, int last, int maxiter, double tol, double damping,
    double **d_basis_out, int *iterations_done, double *iteration_time_sec,
    double *fixed_point_residual_out)
{
    int status = IPT_CUDA_SUCCESS;
    size_t elements = (size_t)n * (size_t)m;
    size_t bytes = elements * sizeof(double);
    double *d_a = NULL;
    double *d_b = NULL;
    double *d_y = NULL;
    double *d_h = NULL;
    double *d_delta = NULL;
    double *d_current = NULL;
    double *d_next = NULL;
    int *d_singular = NULL;
    void *d_spmm_buffer = NULL;
    size_t spmm_buffer_size = 0;
    cusparseDnMatDescr_t dense_x = NULL;
    cusparseDnMatDescr_t dense_y = NULL;
    int done = 0;
    const int threads = 256;
    double fixed_point_residual = NAN;

    *d_basis_out = NULL;
    *iterations_done = 0;
    *iteration_time_sec = 0.0;
    if (fixed_point_residual_out != NULL) {
        *fixed_point_residual_out = NAN;
    }

    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_y, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_h, (size_t)m * (size_t)m *
                                         sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_delta, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_singular, sizeof(int)));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_x, n, m, n, d_a, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_y, n, m, n, d_y, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    {
        const double one = 1.0;
        const double zero = 0.0;
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            sparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &one, sparse_matrix, dense_x,
            &zero, dense_y, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
            &spmm_buffer_size));
    }
    if (spmm_buffer_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_spmm_buffer, spmm_buffer_size));
    }

    ipt_block_cluster_identity_kernel<<<
        (int)((elements + threads - 1) / threads), threads>>>(d_a, n, m,
                                                               first);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    d_current = d_a;
    d_next = d_b;

    {
        std::chrono::steady_clock::time_point start =
            std::chrono::steady_clock::now();

        for (int iter = 0; iter < maxiter; ++iter) {
            double delta_norm = 0.0;
            double basis_norm = 0.0;
            int singular = 0;
            size_t shared_bytes =
                (size_t)m * (size_t)(m + 1) * sizeof(double) +
                2U * sizeof(int);

            CUSPARSE_CHECK(ipt_cusparse_spmm(
                sparse_handle, sparse_matrix, dense_x, dense_y, d_current,
                d_y, d_spmm_buffer));
            ipt_block_cluster_gather_h_kernel<<<
                (m * m + threads - 1) / threads, threads>>>(
                d_y, d_h, n, m, first);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemset(d_singular, 0, sizeof(int)));
            ipt_block_cluster_update_kernel<<<n, m, shared_bytes>>>(
                d_current, d_y, d_diagonal, d_h, d_next, n, m, first, last,
                damping, d_singular);
            CUDA_CHECK(cudaGetLastError());

            fixed_point_delta_kernel<<<
                (int)((elements + threads - 1) / threads), threads>>>(
                d_delta, d_next, d_current, (int)elements);
            CUDA_CHECK(cudaGetLastError());
            CUBLAS_CHECK(cublasDnrm2(cublas_handle, (int)elements, d_delta, 1,
                                     &delta_norm));
            CUBLAS_CHECK(cublasDnrm2(cublas_handle, (int)elements, d_current,
                                     1, &basis_norm));
            CUDA_CHECK(cudaMemcpy(&singular, d_singular, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            if (singular || !isfinite(delta_norm) ||
                !isfinite(basis_norm) || basis_norm <= 0.0) {
                status = IPT_CUDA_INVALID_ARGUMENT;
                goto cleanup;
            }

            std::swap(d_current, d_next);
            done = iter + 1;
            fixed_point_residual = delta_norm / std::max(1.0, basis_norm);
            if (fixed_point_residual <= tol) {
                break;
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        *iteration_time_sec = ipt_block_elapsed_seconds(start);
    }

    *iterations_done = done;
    if (fixed_point_residual_out != NULL) {
        *fixed_point_residual_out = fixed_point_residual;
    }
    *d_basis_out = d_current;
    if (d_current == d_a) {
        d_a = NULL;
    } else {
        d_b = NULL;
    }

cleanup:
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_y);
    cudaFree(d_h);
    cudaFree(d_delta);
    cudaFree(d_singular);
    cudaFree(d_spmm_buffer);
    if (dense_x != NULL) {
        cusparseDestroyDnMat(dense_x);
    }
    if (dense_y != NULL) {
        cusparseDestroyDnMat(dense_y);
    }
    return status;
}

static int ipt_block_cluster_rayleigh_ritz_gpu(
    cusparseHandle_t sparse_handle, cusparseSpMatDescr_t sparse_matrix,
    cublasHandle_t cublas_handle, const double *d_basis, int n, int basis_cols,
    int k, const int *d_perm, double **d_values_out, double **d_vectors_out,
    double *rayleigh_ritz_time_sec, int *used_qr_out,
    double *basis_orthogonality_frobenius_error,
    double *basis_orthogonality_max_abs_error, int *basis_has_nan_or_inf,
    double *ritz_vectors_orthogonality_frobenius_error,
    double *ritz_vectors_orthogonality_max_abs_error,
    int *ritz_vectors_has_nan_or_inf)
{
    int status = IPT_CUDA_SUCCESS;
    double *d_y = NULL;
    double *d_q = NULL;
    double *d_tau = NULL;
    double *d_gram = NULL;
    double *d_projected = NULL;
    double *d_all_values = NULL;
    double *d_sorted_vectors = NULL;
    double *d_vectors = NULL;
    double *d_work = NULL;
    int *d_info = NULL;
    void *d_spmm_buffer = NULL;
    size_t spmm_buffer_size = 0;
    int lwork = 0;
    int geqrf_lwork = 0;
    int orgqr_lwork = 0;
    int eigen_lwork = 0;
    int info = 0;
    int use_qr = ipt_cuda_env_flag("IPT_BLOCK_CLUSTER_QR");
    const double *d_basis_for_rr = d_basis;
    cusparseDnMatDescr_t dense_x = NULL;
    cusparseDnMatDescr_t dense_y = NULL;
    cusolverDnHandle_t solver = NULL;
    const int threads = 256;
    const double one = 1.0;
    const double zero = 0.0;

    *d_values_out = NULL;
    *d_vectors_out = NULL;
    *rayleigh_ritz_time_sec = 0.0;
    *used_qr_out = use_qr;
    *basis_orthogonality_frobenius_error = NAN;
    *basis_orthogonality_max_abs_error = NAN;
    *basis_has_nan_or_inf = 0;
    *ritz_vectors_orthogonality_frobenius_error = NAN;
    *ritz_vectors_orthogonality_max_abs_error = NAN;
    *ritz_vectors_has_nan_or_inf = 0;

    CUDA_CHECK(cudaMalloc((void **)&d_y,
                          (size_t)n * (size_t)basis_cols * sizeof(double)));
    if (use_qr) {
        CUDA_CHECK(cudaMalloc((void **)&d_q,
                              (size_t)n * (size_t)basis_cols *
                                  sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_tau,
                              (size_t)basis_cols * sizeof(double)));
        d_basis_for_rr = d_q;
    } else {
        CUDA_CHECK(cudaMalloc((void **)&d_gram,
                              (size_t)basis_cols * (size_t)basis_cols *
                                  sizeof(double)));
    }
    CUDA_CHECK(cudaMalloc((void **)&d_projected,
                          (size_t)basis_cols * (size_t)basis_cols *
                              sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_all_values,
                          (size_t)basis_cols * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_sorted_vectors,
                          (size_t)n * (size_t)k * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_vectors,
                          (size_t)n * (size_t)k * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_info, sizeof(int)));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_x, n, basis_cols, n,
                                       (void *)d_basis_for_rr, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_y, n, basis_cols, n, d_y,
                                       CUDA_R_64F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        sparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &one, sparse_matrix, dense_x, &zero,
        dense_y, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &spmm_buffer_size));
    if (spmm_buffer_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_spmm_buffer, spmm_buffer_size));
    }
    CUSOLVER_CHECK(cusolverDnCreate(&solver));
    if (use_qr) {
        CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(
            solver, n, basis_cols, d_q, n, &geqrf_lwork));
        CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(
            solver, n, basis_cols, basis_cols, d_q, n, d_tau,
            &orgqr_lwork));
        CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(
            solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
            basis_cols, d_projected, basis_cols, d_all_values,
            &eigen_lwork));
        lwork =
            std::max(geqrf_lwork, std::max(orgqr_lwork, eigen_lwork));
    } else {
        CUSOLVER_CHECK(cusolverDnDsygvd_bufferSize(
            solver, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER, basis_cols, d_projected, basis_cols,
            d_gram, basis_cols, d_all_values, &lwork));
    }
    CUDA_CHECK(cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(double)));
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        std::chrono::steady_clock::time_point start =
            std::chrono::steady_clock::now();

        if (use_qr) {
            CUDA_CHECK(cudaMemcpy(d_q, d_basis,
                                  (size_t)n * (size_t)basis_cols *
                                      sizeof(double),
                                  cudaMemcpyDeviceToDevice));
            CUSOLVER_CHECK(cusolverDnDgeqrf(
                solver, n, basis_cols, d_q, n, d_tau, d_work, lwork,
                d_info));
            CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            if (info != 0) {
                fprintf(stderr, "cuSOLVER Dgeqrf failed: info=%d\n", info);
                status = IPT_CUDA_CUSOLVER_ERROR;
                goto cleanup;
            }
            CUSOLVER_CHECK(cusolverDnDorgqr(
                solver, n, basis_cols, basis_cols, d_q, n, d_tau, d_work,
                lwork, d_info));
            CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            if (info != 0) {
                fprintf(stderr, "cuSOLVER Dorgqr failed: info=%d\n", info);
                status = IPT_CUDA_CUSOLVER_ERROR;
                goto cleanup;
            }
        }
        {
            std::vector<double> host_basis(
                (size_t)n * (size_t)basis_cols, 0.0);

            CUDA_CHECK(cudaMemcpy(host_basis.data(), d_basis_for_rr,
                                  host_basis.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            ipt_block_orthogonality_stats(
                host_basis, n, basis_cols,
                basis_orthogonality_frobenius_error,
                basis_orthogonality_max_abs_error, basis_has_nan_or_inf);
        }
        CUSPARSE_CHECK(ipt_cusparse_spmm(
            sparse_handle, sparse_matrix, dense_x, dense_y,
            const_cast<double *>(d_basis_for_rr), d_y, d_spmm_buffer));
        if (!use_qr) {
            CUBLAS_CHECK(cublasDgemm(
                cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, basis_cols,
                basis_cols, n, &one, d_basis, n, d_basis, n, &zero, d_gram,
                basis_cols));
        }
        CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 basis_cols, basis_cols, n, &one,
                                 d_basis_for_rr, n, d_y, n, &zero,
                                 d_projected, basis_cols));
        if (use_qr) {
            CUSOLVER_CHECK(cusolverDnDsyevd(
                solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                basis_cols, d_projected, basis_cols, d_all_values, d_work,
                lwork, d_info));
        } else {
            CUSOLVER_CHECK(cusolverDnDsygvd(
                solver, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,
                CUBLAS_FILL_MODE_LOWER, basis_cols, d_projected, basis_cols,
                d_gram, basis_cols, d_all_values, d_work, lwork, d_info));
        }
        CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (info != 0) {
            fprintf(stderr, "cuSOLVER Rayleigh-Ritz failed: info=%d\n", info);
            status = IPT_CUDA_CUSOLVER_ERROR;
            goto cleanup;
        }
        CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n,
                                 k, basis_cols, &one, d_basis_for_rr, n,
                                 d_projected, basis_cols, &zero,
                                 d_sorted_vectors, n));
        for (int col = 0; col < k; ++col) {
            double norm = 0.0;
            CUBLAS_CHECK(cublasDnrm2(cublas_handle, n,
                                     d_sorted_vectors + (size_t)col * n, 1,
                                     &norm));
            if (!isfinite(norm) || norm <= 0.0) {
                status = IPT_CUDA_INVALID_ARGUMENT;
                goto cleanup;
            }
            {
                double inverse_norm = 1.0 / norm;
                CUBLAS_CHECK(cublasDscal(
                    cublas_handle, n, &inverse_norm,
                    d_sorted_vectors + (size_t)col * n, 1));
            }
        }
        {
            std::vector<double> host_ritz((size_t)n * (size_t)k, 0.0);

            CUDA_CHECK(cudaMemcpy(host_ritz.data(), d_sorted_vectors,
                                  host_ritz.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            ipt_block_orthogonality_stats(
                host_ritz, n, k,
                ritz_vectors_orthogonality_frobenius_error,
                ritz_vectors_orthogonality_max_abs_error,
                ritz_vectors_has_nan_or_inf);
        }
        ipt_block_cluster_unpermute_kernel<<<
            (n * k + threads - 1) / threads, threads>>>(
            d_sorted_vectors, d_vectors, d_perm, n, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        *rayleigh_ritz_time_sec = ipt_block_elapsed_seconds(start);
    }

    *d_values_out = d_all_values;
    *d_vectors_out = d_vectors;
    d_all_values = NULL;
    d_vectors = NULL;

cleanup:
    cudaFree(d_y);
    cudaFree(d_q);
    cudaFree(d_tau);
    cudaFree(d_gram);
    cudaFree(d_projected);
    cudaFree(d_all_values);
    cudaFree(d_sorted_vectors);
    cudaFree(d_vectors);
    cudaFree(d_work);
    cudaFree(d_info);
    cudaFree(d_spmm_buffer);
    if (dense_x != NULL) {
        cusparseDestroyDnMat(dense_x);
    }
    if (dense_y != NULL) {
        cusparseDestroyDnMat(dense_y);
    }
    if (solver != NULL) {
        cusolverDnDestroy(solver);
    }
    return status;
}

static void ipt_block_cluster_csc_matmul(
    const std::vector<int> &col_ptr, const std::vector<int> &row_ind,
    const std::vector<double> &values, int n, const std::vector<double> &x,
    int cols, std::vector<double> *y)
{
    y->assign((size_t)n * (size_t)cols, 0.0);

    for (int col = 0; col < n; ++col) {
        for (int p = col_ptr[(size_t)col]; p < col_ptr[(size_t)col + 1U];
             ++p) {
            int row = row_ind[(size_t)p];
            double value = values[(size_t)p];

            for (int block_col = 0; block_col < cols; ++block_col) {
                (*y)[(size_t)row + (size_t)block_col * (size_t)n] +=
                    value *
                    x[(size_t)col + (size_t)block_col * (size_t)n];
            }
        }
    }
}

static int ipt_block_cluster_cholesky_lower(const std::vector<double> &a,
                                            int m,
                                            std::vector<double> *lower)
{
    lower->assign((size_t)m * (size_t)m, 0.0);

    for (int col = 0; col < m; ++col) {
        for (int row = col; row < m; ++row) {
            double sum = a[(size_t)row + (size_t)col * (size_t)m];

            for (int inner = 0; inner < col; ++inner) {
                sum -= (*lower)[(size_t)row +
                                (size_t)inner * (size_t)m] *
                       (*lower)[(size_t)col +
                                (size_t)inner * (size_t)m];
            }

            if (row == col) {
                if (sum <= 1.0e-14 || !isfinite(sum)) {
                    return 0;
                }
                (*lower)[(size_t)row + (size_t)col * (size_t)m] =
                    sqrt(sum);
            } else {
                (*lower)[(size_t)row + (size_t)col * (size_t)m] =
                    sum / (*lower)[(size_t)col +
                                   (size_t)col * (size_t)m];
            }
        }
    }
    return 1;
}

static int ipt_block_cluster_inverse_lower(const std::vector<double> &lower,
                                           int m,
                                           std::vector<double> *inverse)
{
    inverse->assign((size_t)m * (size_t)m, 0.0);

    for (int col = 0; col < m; ++col) {
        for (int row = 0; row < m; ++row) {
            double rhs = (row == col) ? 1.0 : 0.0;

            for (int inner = 0; inner < row; ++inner) {
                rhs -= lower[(size_t)row +
                             (size_t)inner * (size_t)m] *
                       (*inverse)[(size_t)inner +
                                  (size_t)col * (size_t)m];
            }
            if (fabs(lower[(size_t)row + (size_t)row * (size_t)m]) <=
                1.0e-14) {
                return 0;
            }
            (*inverse)[(size_t)row + (size_t)col * (size_t)m] =
                rhs / lower[(size_t)row + (size_t)row * (size_t)m];
        }
    }
    return 1;
}

static void ipt_block_cluster_small_matmul(const std::vector<double> &a,
                                           const std::vector<double> &b,
                                           int m,
                                           std::vector<double> *c)
{
    c->assign((size_t)m * (size_t)m, 0.0);

    for (int col = 0; col < m; ++col) {
        for (int inner = 0; inner < m; ++inner) {
            double b_value = b[(size_t)inner + (size_t)col * (size_t)m];

            if (b_value == 0.0) {
                continue;
            }
            for (int row = 0; row < m; ++row) {
                (*c)[(size_t)row + (size_t)col * (size_t)m] +=
                    a[(size_t)row + (size_t)inner * (size_t)m] *
                    b_value;
            }
        }
    }
}

static void ipt_block_cluster_symmetric_eigenpairs(
    const std::vector<double> &matrix, int m, std::vector<double> *values,
    std::vector<double> *vectors)
{
    ipt_jacobi_eigenvectors_symmetric(matrix, m, vectors);
    values->assign((size_t)m, 0.0);

    for (int col = 0; col < m; ++col) {
        double value = 0.0;

        for (int row = 0; row < m; ++row) {
            for (int inner = 0; inner < m; ++inner) {
                value += (*vectors)[(size_t)row +
                                    (size_t)col * (size_t)m] *
                         matrix[(size_t)row +
                                (size_t)inner * (size_t)m] *
                         (*vectors)[(size_t)inner +
                                    (size_t)col * (size_t)m];
            }
        }
        (*values)[(size_t)col] = value;
    }
    ipt_sort_eigenvectors_by_values(vectors, values, m);
}

static void ipt_block_cluster_add_target_clusters(
    std::vector<std::pair<int, int>> *ranges, int n, int k)
{
    std::vector<int> covered((size_t)n, 0);

    for (size_t i = 0; i < ranges->size(); ++i) {
        for (int row = (*ranges)[i].first; row <= (*ranges)[i].second; ++row) {
            if (row >= 0 && row < n) {
                covered[(size_t)row] = 1;
            }
        }
    }
    for (int target = 0; target < k; ++target) {
        if (!covered[(size_t)target]) {
            ranges->push_back(std::make_pair(target, target));
        }
    }
    std::sort(ranges->begin(), ranges->end());
}

static void ipt_block_cluster_adaptive_expand(
    const std::vector<int> &col_ptr, const std::vector<int> &row_ind,
    const std::vector<double> &values, const std::vector<double> &diagonal,
    int requested_k, int max_block_size, double coupling_tau,
    std::vector<std::pair<int, int>> *ranges,
    std::vector<int> *added_indices, int *limit_hit)
{
    const double eps_gap = 1.0e-14;
    int n = (int)diagonal.size();

    added_indices->clear();
    *limit_hit = 0;
    for (;;) {
        int target_id = -1;
        int first = -1;
        int last = -1;
        int best_outside = -1;
        double best_ratio = 0.0;

        for (size_t i = 0; i < ranges->size(); ++i) {
            if ((*ranges)[i].first <= requested_k - 1 &&
                requested_k - 1 <= (*ranges)[i].second) {
                target_id = (int)i;
                first = (*ranges)[i].first;
                last = (*ranges)[i].second;
                break;
            }
        }
        if (target_id < 0) {
            break;
        }
        for (int col = first; col <= last; ++col) {
            for (int p = col_ptr[(size_t)col];
                 p < col_ptr[(size_t)col + 1U]; ++p) {
                int row = row_ind[(size_t)p];
                double gap = 0.0;
                double ratio = 0.0;

                if (first <= row && row <= last) {
                    continue;
                }
                gap = fabs(diagonal[(size_t)col] -
                           diagonal[(size_t)row]);
                ratio = fabs(values[(size_t)p]) / std::max(gap, eps_gap);
                if (std::max(last, row) - std::min(first, row) + 1 >
                    max_block_size) {
                    if (ratio > coupling_tau) {
                        *limit_hit = 1;
                    }
                    continue;
                }
                if (ratio > best_ratio) {
                    best_ratio = ratio;
                    best_outside = row;
                }
            }
        }
        if (best_outside < 0 || best_ratio <= coupling_tau) {
            break;
        }
        {
            int proposed_first = std::min(first, best_outside);
            int proposed_last = std::max(last, best_outside);
            bool expanded = true;

            while (expanded) {
                double scale = 1.0;
                double degenerate_gap = 0.0;

                expanded = false;
                if (proposed_first > 0) {
                    scale = std::max(
                        1.0, std::max(fabs(diagonal[(size_t)proposed_first]),
                                      fabs(diagonal[(size_t)proposed_first -
                                                    1U])));
                    degenerate_gap =
                        std::max(eps_gap,
                                 1000.0 *
                                     std::numeric_limits<double>::epsilon() *
                                     scale);
                    if (fabs(diagonal[(size_t)proposed_first] -
                             diagonal[(size_t)proposed_first - 1U]) <=
                        degenerate_gap) {
                        --proposed_first;
                        expanded = true;
                    }
                }
                if (proposed_last + 1 < n) {
                    scale = std::max(
                        1.0, std::max(fabs(diagonal[(size_t)proposed_last]),
                                      fabs(diagonal[(size_t)proposed_last +
                                                    1U])));
                    degenerate_gap =
                        std::max(eps_gap,
                                 1000.0 *
                                     std::numeric_limits<double>::epsilon() *
                                     scale);
                    if (fabs(diagonal[(size_t)proposed_last + 1U] -
                             diagonal[(size_t)proposed_last]) <=
                        degenerate_gap) {
                        ++proposed_last;
                        expanded = true;
                    }
                }
            }
            if (proposed_last - proposed_first + 1 > max_block_size) {
                *limit_hit = 1;
                break;
            }
            for (int index = proposed_first; index <= proposed_last; ++index) {
                if (index < first || index > last) {
                    added_indices->push_back(index);
                }
            }
            (*ranges)[(size_t)target_id] =
                std::make_pair(proposed_first, proposed_last);
            std::sort(ranges->begin(), ranges->end());
            {
                std::vector<std::pair<int, int>> merged;

                for (size_t i = 0; i < ranges->size(); ++i) {
                    if (merged.empty() ||
                        (*ranges)[i].first > merged.back().second) {
                        merged.push_back((*ranges)[i]);
                    } else {
                        merged.back().second =
                            std::max(merged.back().second,
                                     (*ranges)[i].second);
                    }
                }
                ranges->swap(merged);
            }
        }
    }
    std::sort(added_indices->begin(), added_indices->end());
    added_indices->erase(
        std::unique(added_indices->begin(), added_indices->end()),
        added_indices->end());
}

static int ipt_block_cluster_solve_basis(
    const std::vector<int> &col_ptr, const std::vector<int> &row_ind,
    const std::vector<double> &values, const std::vector<double> &diagonal,
    int first, int last, int maxiter, double tol, double damping,
    std::vector<double> *basis, int *iterations_done)
{
    int n = (int)diagonal.size();
    int m = last - first + 1;
    std::vector<double> next_basis;
    std::vector<double> y;
    std::vector<double> h((size_t)m * (size_t)m, 0.0);
    std::vector<double> denominator((size_t)m * (size_t)m, 0.0);
    std::vector<double> inverse;
    std::vector<double> rhs((size_t)m, 0.0);
    double effective_tol = tol > 0.0 ? tol : 1.0e-12;
    double effective_damping = damping;

    if (effective_damping <= 0.0 || effective_damping > 1.0 ||
        !isfinite(effective_damping)) {
        effective_damping = 1.0;
    }

    basis->assign((size_t)n * (size_t)m, 0.0);
    for (int col = 0; col < m; ++col) {
        (*basis)[(size_t)(first + col) + (size_t)col * (size_t)n] = 1.0;
    }
    if (iterations_done != NULL) {
        *iterations_done = 0;
    }

    for (int iter = 0; iter < maxiter; ++iter) {
        double delta_norm_sq = 0.0;
        double basis_norm_sq = 0.0;

        ipt_block_cluster_csc_matmul(col_ptr, row_ind, values, n, *basis, m,
                                     &y);
        for (int col = 0; col < m; ++col) {
            for (int row = 0; row < m; ++row) {
                h[(size_t)row + (size_t)col * (size_t)m] =
                    y[(size_t)(first + row) +
                      (size_t)col * (size_t)n];
            }
        }

        next_basis.assign((size_t)n * (size_t)m, 0.0);
        for (int col = 0; col < m; ++col) {
            next_basis[(size_t)(first + col) +
                       (size_t)col * (size_t)n] = 1.0;
        }

        for (int row = 0; row < n; ++row) {
            if (row >= first && row <= last) {
                continue;
            }

            for (int col = 0; col < m; ++col) {
                rhs[(size_t)col] =
                    diagonal[(size_t)row] *
                        (*basis)[(size_t)row +
                                 (size_t)col * (size_t)n] -
                    y[(size_t)row + (size_t)col * (size_t)n];
            }

            for (int col = 0; col < m; ++col) {
                for (int local_row = 0; local_row < m; ++local_row) {
                    denominator[(size_t)local_row +
                                (size_t)col * (size_t)m] =
                        ((local_row == col) ? diagonal[(size_t)row] : 0.0) -
                        h[(size_t)local_row + (size_t)col * (size_t)m];
                }
            }
            if (!ipt_inverse_square(denominator, m, &inverse)) {
                return 0;
            }

            for (int col = 0; col < m; ++col) {
                double value = 0.0;

                for (int inner = 0; inner < m; ++inner) {
                    value += rhs[(size_t)inner] *
                             inverse[(size_t)inner +
                                     (size_t)col * (size_t)m];
                }
                if (effective_damping < 1.0) {
                    value = (1.0 - effective_damping) *
                                (*basis)[(size_t)row +
                                         (size_t)col * (size_t)n] +
                            effective_damping * value;
                }
                next_basis[(size_t)row + (size_t)col * (size_t)n] = value;
            }
        }

        for (size_t i = 0; i < next_basis.size(); ++i) {
            double delta = next_basis[i] - (*basis)[i];

            if (!isfinite(next_basis[i])) {
                return 0;
            }
            delta_norm_sq += delta * delta;
            basis_norm_sq += (*basis)[i] * (*basis)[i];
        }
        basis->swap(next_basis);
        if (iterations_done != NULL) {
            *iterations_done = iter + 1;
        }
        if (sqrt(delta_norm_sq) / std::max(1.0, sqrt(basis_norm_sq)) <=
            effective_tol) {
            break;
        }
    }
    return 1;
}

static int ipt_block_cluster_rayleigh_ritz(
    const std::vector<int> &col_ptr, const std::vector<int> &row_ind,
    const std::vector<double> &values, int n, const std::vector<double> &basis,
    int basis_cols, int k, std::vector<double> *ritz_values,
    std::vector<double> *ritz_vectors)
{
    std::vector<double> y;
    std::vector<double> gram((size_t)basis_cols * (size_t)basis_cols, 0.0);
    std::vector<double> projected((size_t)basis_cols * (size_t)basis_cols,
                                  0.0);
    std::vector<double> lower;
    std::vector<double> lower_inverse;
    std::vector<double> temp;
    std::vector<double> transformed;
    std::vector<double> all_values;
    std::vector<double> all_vectors;

    ipt_block_cluster_csc_matmul(col_ptr, row_ind, values, n, basis,
                                 basis_cols, &y);

    for (int col = 0; col < basis_cols; ++col) {
        for (int row = 0; row < basis_cols; ++row) {
            double gram_value = 0.0;
            double projected_value = 0.0;

            for (int i = 0; i < n; ++i) {
                gram_value += basis[(size_t)i +
                                    (size_t)row * (size_t)n] *
                              basis[(size_t)i +
                                    (size_t)col * (size_t)n];
                projected_value += basis[(size_t)i +
                                         (size_t)row * (size_t)n] *
                                   y[(size_t)i +
                                     (size_t)col * (size_t)n];
            }
            gram[(size_t)row + (size_t)col * (size_t)basis_cols] =
                gram_value;
            projected[(size_t)row + (size_t)col * (size_t)basis_cols] =
                projected_value;
        }
    }

    if (!ipt_block_cluster_cholesky_lower(gram, basis_cols, &lower) ||
        !ipt_block_cluster_inverse_lower(lower, basis_cols, &lower_inverse)) {
        return 0;
    }

    ipt_block_cluster_small_matmul(lower_inverse, projected, basis_cols,
                                   &temp);
    transformed.assign((size_t)basis_cols * (size_t)basis_cols, 0.0);
    for (int col = 0; col < basis_cols; ++col) {
        for (int row = 0; row < basis_cols; ++row) {
            double value = 0.0;

            for (int inner = 0; inner < basis_cols; ++inner) {
                value += temp[(size_t)row +
                              (size_t)inner * (size_t)basis_cols] *
                         lower_inverse[(size_t)col +
                                       (size_t)inner *
                                           (size_t)basis_cols];
            }
            transformed[(size_t)row + (size_t)col * (size_t)basis_cols] =
                value;
        }
    }
    for (int col = 0; col < basis_cols; ++col) {
        for (int row = 0; row < col; ++row) {
            double sym = 0.5 *
                         (transformed[(size_t)row +
                                      (size_t)col *
                                          (size_t)basis_cols] +
                          transformed[(size_t)col +
                                      (size_t)row *
                                          (size_t)basis_cols]);
            transformed[(size_t)row + (size_t)col * (size_t)basis_cols] =
                sym;
            transformed[(size_t)col + (size_t)row * (size_t)basis_cols] =
                sym;
        }
    }

    ipt_block_cluster_symmetric_eigenpairs(transformed, basis_cols,
                                           &all_values, &all_vectors);
    ritz_values->assign((size_t)k, 0.0);
    ritz_vectors->assign((size_t)n * (size_t)k, 0.0);

    for (int col = 0; col < k; ++col) {
        std::vector<double> coeff((size_t)basis_cols, 0.0);
        double norm_sq = 0.0;

        (*ritz_values)[(size_t)col] = all_values[(size_t)col];
        for (int row = 0; row < basis_cols; ++row) {
            for (int inner = 0; inner < basis_cols; ++inner) {
                coeff[(size_t)row] +=
                    lower_inverse[(size_t)inner +
                                  (size_t)row *
                                      (size_t)basis_cols] *
                    all_vectors[(size_t)inner +
                                (size_t)col *
                                    (size_t)basis_cols];
            }
        }

        for (int row = 0; row < n; ++row) {
            double value = 0.0;

            for (int inner = 0; inner < basis_cols; ++inner) {
                value += basis[(size_t)row +
                               (size_t)inner * (size_t)n] *
                         coeff[(size_t)inner];
            }
            (*ritz_vectors)[(size_t)row + (size_t)col * (size_t)n] =
                value;
            norm_sq += value * value;
        }
        if (norm_sq > 0.0) {
            double inv_norm = 1.0 / sqrt(norm_sq);

            for (int row = 0; row < n; ++row) {
                (*ritz_vectors)[(size_t)row +
                                (size_t)col * (size_t)n] *= inv_norm;
            }
        }
    }
    return 1;
}

static int ipt_cuda_sparse_csc_block_cluster_impl(
    const int *col_ptr, const int *row_ind, const double *matrix_values, int n,
    int k, int nnz, int maxiter, double tol, int use_tolerance,
    IPTCudaResult *result)
{
    int status = IPT_CUDA_SUCCESS;
    std::chrono::steady_clock::time_point preparation_start =
        std::chrono::steady_clock::now();
    std::vector<double> original_diagonal;
    std::vector<int> perm;
    std::vector<int> inverse_perm;
    std::vector<int> sorted_col_ptr;
    std::vector<int> sorted_row_ind;
    std::vector<double> sorted_values;
    std::vector<double> diagonal;
    std::vector<std::pair<int, int>> ranges;
    IPTDegeneracyOptions options;
    int too_large = 0;
    int basis_cols = 0;
    int oversample = 0;
    int basis_target_k = 0;
    int adaptive_enabled = 0;
    int adaptive_limit_hit = 0;
    double adaptive_coupling_tau = 0.1;
    std::vector<int> adaptive_added_indices;
    int block_maxiter = 0;
    int iterations_done = 0;
    double block_tol = 0.0;
    double damping = 1.0;
    int *d_col_ptr = NULL;
    int *d_row_ind = NULL;
    int *d_perm = NULL;
    double *d_values = NULL;
    double *d_diagonal = NULL;
    double *d_combined_basis = NULL;
    double *d_davidson_basis = NULL;
    double *d_ritz_values = NULL;
    double *d_ritz_vectors = NULL;
    cublasHandle_t cublas_handle = NULL;
    cusparseHandle_t sparse_handle = NULL;
    cusparseSpMatDescr_t sparse_matrix = NULL;
    IPTBestSoFarState best_so_far;

    if (result != NULL) {
        memset(result, 0, sizeof(*result));
        result->fixed_point_residual = NAN;
        result->basis_orthogonality_frobenius_error = NAN;
        result->basis_orthogonality_max_abs_error = NAN;
        result->ritz_vectors_orthogonality_frobenius_error = NAN;
        result->ritz_vectors_orthogonality_max_abs_error = NAN;
        result->davidson_target_index = -1;
        result->davidson_residual_before = NAN;
        result->davidson_residual_after = NAN;
        result->davidson_denom_clip = NAN;
        result->adaptive_coupling_tau = NAN;
        result->adaptive_target_block_start = -1;
        result->adaptive_target_block_end = -1;
        result->best_so_far_enabled =
            ipt_cuda_env_flag_default("IPT_DAVIDSON_USE_BEST_SO_FAR", 1);
        result->best_so_far_step = -1;
        snprintf(result->best_so_far_source,
                 sizeof(result->best_so_far_source), "none");
        result->best_so_far_max_residual = NAN;
        result->best_so_far_max_residual_index = -1;
        result->pair28_best_residual = NAN;
        result->pair29_best_residual = NAN;
        result->relaxed_accept_enabled =
            ipt_cuda_env_flag_default("IPT_DAVIDSON_RELAXED_ACCEPT", 1);
        result->accepted_steps = 0;
        result->rejected_steps = 0;
        result->min_accepted_steps = 0;
        result->early_jump_to_continuation = 0;
        result->cluster_aware_accept_enabled =
            ipt_cuda_env_flag_default("IPT_DAVIDSON_CLUSTER_AWARE_ACCEPT", 0);
        result->soft_cluster_locking_enabled =
            ipt_cuda_env_flag_default("IPT_DAVIDSON_SOFT_CLUSTER_LOCKING", 0);
    }
    ipt_best_so_far_init(
        &best_so_far,
        ipt_cuda_env_flag_default("IPT_DAVIDSON_USE_BEST_SO_FAR", 1));

    if (col_ptr == NULL || row_ind == NULL || matrix_values == NULL ||
        result == NULL || n <= 0 || k <= 0 || k > n || nnz < 0 ||
        maxiter < 0 || (use_tolerance && (tol <= 0.0 || maxiter <= 0))) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }
    if (!ipt_validate_csc(col_ptr, row_ind, n, nnz)) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    ipt_extract_csc_diagonal_host(col_ptr, row_ind, matrix_values, n,
                                  &original_diagonal);
    ipt_build_stable_sort_permutation(original_diagonal, &perm, &inverse_perm);
    if (!ipt_permute_csc(col_ptr, row_ind, matrix_values, n, inverse_perm,
                         &sorted_col_ptr, &sorted_row_ind, &sorted_values)) {
        return IPT_CUDA_ALLOCATION_FAILED;
    }
    ipt_extract_csc_diagonal_vectors(sorted_col_ptr, sorted_row_ind,
                                     sorted_values, n, &diagonal);
    if (!ipt_csc_is_hermitian_real(sorted_col_ptr, sorted_row_ind,
                                   sorted_values, n)) {
        if (ipt_preparation_debug_enabled()) {
            fprintf(stderr,
                    "IPT block-cluster path currently requires a real "
                    "Hermitian CSC matrix\n");
        }
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    options = ipt_degeneracy_options(
        ipt_cuda_env_double("IPT_DEGENERACY_THRESHOLD", 0.0));
    adaptive_enabled =
        ipt_cuda_env_flag("IPT_BLOCK_CLUSTER_ADAPTIVE");
    adaptive_coupling_tau = ipt_cuda_env_double(
        "IPT_BLOCK_CLUSTER_COUPLING_TAU", 0.1);
    oversample = ipt_cuda_env_int("IPT_BLOCK_CLUSTER_OVERSAMPLE", 0);
    if (adaptive_enabled) {
        oversample = 0;
    }
    basis_target_k = k;
    if (oversample > 0) {
        basis_target_k = oversample > n - k ? n : k + oversample;
    }
    result->oversample = oversample;
    ranges = ipt_target_degenerate_subspaces(
        sorted_col_ptr, sorted_row_ind, sorted_values, diagonal,
        basis_target_k, options, &too_large);
    if (too_large) {
        int offending_size = 0;

        for (size_t i = 0; i < ranges.size(); ++i) {
            offending_size =
                std::max(offending_size,
                         ranges[i].second - ranges[i].first + 1);
        }
        fprintf(stderr,
                "IPT block-cluster invalid argument at block detection: "
                "block_size=%d max_block_size=%d\n",
                offending_size, options.max_block_size);
        return IPT_CUDA_INVALID_ARGUMENT;
    }
    ipt_block_cluster_add_target_clusters(&ranges, n, basis_target_k);
    if (adaptive_enabled) {
        ipt_block_cluster_adaptive_expand(
            sorted_col_ptr, sorted_row_ind, sorted_values, diagonal, k,
            options.max_block_size, adaptive_coupling_tau, &ranges,
            &adaptive_added_indices, &adaptive_limit_hit);
    }
    result->adaptive_block_enabled = adaptive_enabled;
    result->adaptive_coupling_tau = adaptive_coupling_tau;
    result->adaptive_limit_hit = adaptive_limit_hit;
    {
        std::string added;

        for (size_t i = 0; i < adaptive_added_indices.size(); ++i) {
            if (!added.empty()) {
                added += ";";
            }
            added += std::to_string(adaptive_added_indices[i]);
        }
        snprintf(result->adaptive_added_indices,
                 sizeof(result->adaptive_added_indices), "%s",
                 added.empty() ? "none" : added.c_str());
    }

    for (size_t i = 0; i < ranges.size(); ++i) {
        basis_cols += ranges[i].second - ranges[i].first + 1;
        if (ranges[i].first <= k - 1 && k - 1 <= ranges[i].second) {
            result->adaptive_target_block_start = ranges[i].first;
            result->adaptive_target_block_end = ranges[i].second;
        }
    }
    if (basis_cols < k) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    block_maxiter = ipt_cuda_env_int("IPT_BLOCK_CLUSTER_MAXITER",
                                     maxiter > 0 ? maxiter : 100);
    if (block_maxiter <= 0) {
        block_maxiter = 100;
    }
    block_tol = ipt_cuda_env_double(
        "IPT_BLOCK_CLUSTER_TOL", use_tolerance ? tol : 1.0e-12);
    damping = ipt_cuda_env_double("IPT_BLOCK_CLUSTER_DAMPING", 1.0);
    result->preparation_time_sec =
        ipt_block_elapsed_seconds(preparation_start);

    if (ipt_preparation_debug_enabled()) {
        fprintf(stderr,
                "IPT block-cluster: clusters=%zu basis_cols=%d "
                "requested_k=%d basis_target_k=%d oversample=%d "
                "adaptive=%d adaptive_tau=%.3e adaptive_added=%s "
                "maxiter=%d tol=%.3e damping=%.3e\n",
                ranges.size(), basis_cols, k, basis_target_k, oversample,
                adaptive_enabled, adaptive_coupling_tau,
                result->adaptive_added_indices, block_maxiter, block_tol,
                damping);
        for (size_t i = 0; i < ranges.size(); ++i) {
            fprintf(stderr, "IPT block-cluster: block[%zu]=%d:%d size=%d\n",
                    i, ranges[i].first, ranges[i].second,
                    ranges[i].second - ranges[i].first + 1);
        }
    }

    {
        std::chrono::steady_clock::time_point setup_start =
            std::chrono::steady_clock::now();

        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CUSPARSE_CHECK(cusparseCreate(&sparse_handle));
        CUDA_CHECK(cudaMalloc((void **)&d_col_ptr,
                              (size_t)(n + 1) * sizeof(int)));
        CUDA_CHECK(
            cudaMalloc((void **)&d_row_ind, (size_t)nnz * sizeof(int)));
        CUDA_CHECK(
            cudaMalloc((void **)&d_values, (size_t)nnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_diagonal,
                              (size_t)n * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_perm, (size_t)n * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_combined_basis,
                              (size_t)n * (size_t)basis_cols *
                                  sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_col_ptr, sorted_col_ptr.data(),
                              (size_t)(n + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_row_ind, sorted_row_ind.data(),
                              (size_t)nnz * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values, sorted_values.data(),
                              (size_t)nnz * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_diagonal, diagonal.data(),
                              (size_t)n * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_perm, perm.data(), (size_t)n * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUSPARSE_CHECK(cusparseCreateCsr(
            &sparse_matrix, n, n, nnz, d_col_ptr, d_row_ind, d_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        CUDA_CHECK(cudaDeviceSynchronize());
        result->transfer_setup_time_sec =
            ipt_block_elapsed_seconds(setup_start);
    }

    {
        int offset = 0;

        for (size_t range_id = 0; range_id < ranges.size(); ++range_id) {
            double *d_block_basis = NULL;
            int block_iterations = 0;
            int first = ranges[range_id].first;
            int last = ranges[range_id].second;
            int width = last - first + 1;
            double block_iteration_time = 0.0;
            double block_fixed_point_residual = NAN;

            status = ipt_block_cluster_solve_basis_gpu(
                sparse_handle, sparse_matrix, cublas_handle, d_diagonal, n,
                width, first, last, block_maxiter, block_tol, damping,
                &d_block_basis, &block_iterations, &block_iteration_time,
                &block_fixed_point_residual);
            if (status != IPT_CUDA_SUCCESS) {
                cudaFree(d_block_basis);
                goto cleanup;
            }
            iterations_done = std::max(iterations_done, block_iterations);
            if (isfinite(block_fixed_point_residual) &&
                (!isfinite(result->fixed_point_residual) ||
                 block_fixed_point_residual >
                     result->fixed_point_residual)) {
                result->fixed_point_residual = block_fixed_point_residual;
            }
            result->iteration_time_sec += block_iteration_time;
            result->matvecs +=
                (long long)block_iterations * (long long)width;
            CUDA_CHECK(cudaMemcpy(
                d_combined_basis + (size_t)offset * (size_t)n, d_block_basis,
                (size_t)n * (size_t)width * sizeof(double),
                cudaMemcpyDeviceToDevice));
            cudaFree(d_block_basis);
            offset += width;
        }
    }

    status = ipt_block_cluster_rayleigh_ritz_gpu(
        sparse_handle, sparse_matrix, cublas_handle, d_combined_basis, n,
        basis_cols, k, d_perm, &d_ritz_values, &d_ritz_vectors,
        &result->rayleigh_ritz_time_sec, &result->rayleigh_ritz_used_qr,
        &result->basis_orthogonality_frobenius_error,
        &result->basis_orthogonality_max_abs_error,
        &result->basis_has_nan_or_inf,
        &result->ritz_vectors_orthogonality_frobenius_error,
        &result->ritz_vectors_orthogonality_max_abs_error,
        &result->ritz_vectors_has_nan_or_inf);
    if (status != IPT_CUDA_SUCCESS) {
        goto cleanup;
    }
    result->matvecs += basis_cols;
    if (ipt_cuda_env_flag("IPT_DAVIDSON_ENRICH")) {
        const char *accept_raw =
            getenv("IPT_DAVIDSON_ACCEPT_ONLY_IF_IMPROVES");
        int baseline_steps =
            ipt_cuda_env_int("IPT_DAVIDSON_STEPS", 1);
        int extra_steps =
            ipt_cuda_env_int("IPT_DAVIDSON_EXTRA_STEPS", 0);
        int steps = baseline_steps + extra_steps;
        int debug_active_max_enabled =
            ipt_cuda_env_flag("IPT_DAVIDSON_DEBUG_ACTIVE_MAX");
        int debug_active_max = debug_active_max_enabled
                                   ? ipt_cuda_env_int(
                                         "IPT_DAVIDSON_ACTIVE_MAX", 1)
                                   : 0;
        int ortho_repeats =
            ipt_cuda_env_int("IPT_DAVIDSON_ORTHO_REPEATS", 2);
        int restart_every =
            ipt_cuda_env_int("IPT_DAVIDSON_RESTART_EVERY", 20);
        int restart_keep_extra =
            ipt_cuda_env_int("IPT_DAVIDSON_RESTART_KEEP_EXTRA", 5);
        int relaxed_accept_enabled =
            ipt_cuda_env_flag_default("IPT_DAVIDSON_RELAXED_ACCEPT", 1);
        int active_pair_accept =
            ipt_cuda_env_flag_default("IPT_DAVIDSON_ACTIVE_PAIR_ACCEPT", 1);
        int cluster_aware_accept_enabled = 1;
        int soft_cluster_locking_enabled =
            ipt_cuda_env_flag_default("IPT_DAVIDSON_SOFT_CLUSTER_LOCKING", 0);
        int retry_on_reject =
            ipt_cuda_env_flag_default("IPT_DAVIDSON_RETRY_ON_REJECT", 1);
        int min_accepted_steps =
            ipt_cuda_env_int("IPT_DAVIDSON_MIN_ACCEPTED_STEPS", 0);
        int accept_only_if_improves =
            accept_raw == NULL
                ? 1
                : ipt_cuda_env_flag(
                      "IPT_DAVIDSON_ACCEPT_ONLY_IF_IMPROVES");
        double accept_rel_slack = ipt_cuda_env_double(
            "IPT_DAVIDSON_ACCEPT_REL_SLACK", 1.0e-12);
        double accept_abs_slack = ipt_cuda_env_double(
            "IPT_DAVIDSON_ACCEPT_ABS_SLACK", 1.0e-15);
        double legacy_locked_degrade_slack = ipt_cuda_env_double(
            "IPT_DAVIDSON_LOCKED_DEGRADE_SLACK", 1.0e-8);
        double locked_degrade_rel_slack = ipt_cuda_env_double(
            "IPT_DAVIDSON_LOCKED_DEGRADE_REL_SLACK",
            legacy_locked_degrade_slack);
        double locked_degrade_abs_slack = ipt_cuda_env_double(
            "IPT_DAVIDSON_LOCKED_DEGRADE_ABS_SLACK", 1.0e-12);
        double auto_gap_abs_tol = ipt_cuda_env_double(
            "IPT_DAVIDSON_AUTO_GAP_ABS_TOL", 1.0e-12);
        double auto_gap_rel_tol = ipt_cuda_env_double(
            "IPT_DAVIDSON_AUTO_GAP_REL_TOL", 5.0e-4);
        double converged_tolerance = ipt_cuda_env_double(
            "IPT_DAVIDSON_CONVERGED_TOL", 1.0e-13);
        double protect_tolerance = ipt_cuda_env_double(
            "IPT_DAVIDSON_PROTECT_TOL", 1.0e-10);
        double matrix_norm = ipt_block_host_matrix_inf_norm(
            col_ptr, row_ind, matrix_values, n);
        std::vector<double> host_ritz_values((size_t)k, 0.0);
        std::vector<double> host_ritz_vectors(
            (size_t)n * (size_t)k, 0.0);
        std::vector<double> current_residuals;
        std::vector<double> raw_basis(
            (size_t)n * (size_t)basis_cols, 0.0);
        std::vector<double> orthonormal_basis;
        std::vector<std::vector<double>> recent_corrections;
        std::vector<IPTDavidsonHistoryEntry> history;
        std::vector<IPTDavidsonSelectionEntry> selection_history;
        std::vector<IPTDavidsonBlockHistoryEntry> block_history;
        std::vector<int> forced_pairs;
        std::vector<IPTActiveCluster> active_clusters;
        std::vector<double> cluster_reference_vectors;
        std::vector<double> retry_damping_list =
            ipt_cuda_env_double_list("IPT_DAVIDSON_RETRY_DAMPING_LIST",
                                     "0.5,0.25");
        std::vector<double> retry_denom_clip_mults =
            ipt_cuda_env_double_list("IPT_DAVIDSON_RETRY_DENOM_CLIP_MULTS",
                                     "10,100");
        double current_maximum = NAN;
        int current_max_index = -1;
        int accepted_since_restart = 0;
        int accepted_steps_count = 0;
        int rejected_steps_count = 0;
        int early_jump_to_continuation = 0;
        int cluster_active = 0;
        int cluster_hard_locked = 0;
        int cluster_stable_count = 0;
        int cluster_soft_locked_count = 0;
        char pair28_lock_state[16] = "inactive";
        char pair29_lock_state[16] = "inactive";

        if (getenv("IPT_DAVIDSON_FORCE_ACTIVE_PAIRS") != NULL) {
            forced_pairs = ipt_jd_parse_active_pairs(
                getenv("IPT_DAVIDSON_FORCE_ACTIVE_PAIRS"), k);
            if (!forced_pairs.empty()) {
                fprintf(stderr,
                        "IPT Davidson: FORCE_ACTIVE_PAIRS is "
                        "legacy/debug-only and does not alter automatic "
                        "Ritz cluster discovery or correction selection\n");
                forced_pairs.clear();
            }
        }
        baseline_steps = std::max(0, baseline_steps);
        extra_steps = std::max(0, extra_steps);
        steps = std::max(1, baseline_steps + extra_steps);
        debug_active_max =
            std::max(0, std::min(debug_active_max, k));
        ortho_repeats = std::max(1, ortho_repeats);
        restart_every = std::max(0, restart_every);
        restart_keep_extra = std::max(1, restart_keep_extra);
        min_accepted_steps = std::max(0, min_accepted_steps);
        result->davidson_attempted = 1;
        result->relaxed_accept_enabled = relaxed_accept_enabled;
        result->min_accepted_steps = min_accepted_steps;
        result->cluster_aware_accept_enabled =
            cluster_aware_accept_enabled;
        result->soft_cluster_locking_enabled =
            soft_cluster_locking_enabled;
        result->davidson_denom_clip = ipt_cuda_env_double(
            "IPT_DAVIDSON_DENOM_CLIP", 1.0e-8);
        if (result->davidson_denom_clip <= 0.0) {
            result->davidson_denom_clip = 1.0e-8;
        }
        fprintf(stderr,
                "IPT Davidson automatic Ritz cluster discovery enabled: "
                "auto_gap_abs_tol=%.17g auto_gap_rel_tol=%.17g "
                "converged_tol=%.17g active_tol_used=0 "
                "active_clusters_env_used=0 active_max_used=0 "
                "correction_max_per_step_used=0 "
                "preparation_blocks_are_not_ritz_clusters=1\n",
                auto_gap_abs_tol, auto_gap_rel_tol,
                converged_tolerance);
        if (debug_active_max_enabled) {
            fprintf(stderr,
                    "IPT Davidson: debug-only ACTIVE_MAX cap enabled: %d\n",
                    debug_active_max);
        }
        CUDA_CHECK(cudaMemcpy(host_ritz_vectors.data(), d_ritz_vectors,
                              host_ritz_vectors.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_ritz_values.data(), d_ritz_values,
                              host_ritz_values.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
        if (ipt_sort_host_ritz_pairs(
                n, k, &host_ritz_values, &host_ritz_vectors)) {
            CUDA_CHECK(cudaMemcpy(
                d_ritz_values, host_ritz_values.data(),
                host_ritz_values.size() * sizeof(double),
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(
                d_ritz_vectors, host_ritz_vectors.data(),
                host_ritz_vectors.size() * sizeof(double),
                cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(raw_basis.data(), d_combined_basis,
                              raw_basis.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
        ipt_block_host_residuals(
            col_ptr, row_ind, matrix_values, n, k, host_ritz_values,
            host_ritz_vectors, matrix_norm, &current_residuals,
            &current_maximum, &current_max_index);
        {
            active_clusters = ipt_discover_auto_ritz_clusters(
                k, host_ritz_values, current_residuals,
                converged_tolerance, auto_gap_abs_tol,
                auto_gap_rel_tol);
            ipt_log_auto_cluster_discovery(
                0, active_clusters, current_residuals,
                converged_tolerance);
            cluster_active = ipt_active_clusters_include_exact(
                active_clusters, 28, 29);
            cluster_reference_vectors = host_ritz_vectors;
            ipt_cluster_28_29_lock_states(
                current_residuals, converged_tolerance, cluster_active,
                soft_cluster_locking_enabled, cluster_hard_locked,
                pair28_lock_state, sizeof(pair28_lock_state),
                pair29_lock_state, sizeof(pair29_lock_state),
                &cluster_soft_locked_count);
            ipt_block_store_result_cluster_fields(
                result, host_ritz_values, current_residuals, k,
                cluster_active, cluster_aware_accept_enabled,
                soft_cluster_locking_enabled, converged_tolerance,
                cluster_hard_locked);
        }
        ipt_best_so_far_note(
            &best_so_far, "davidson_initial", 0, host_ritz_values,
            host_ritz_vectors, current_residuals, current_maximum,
            current_max_index, basis_cols, result->rayleigh_ritz_used_qr,
            result->basis_orthogonality_frobenius_error,
            result->basis_orthogonality_max_abs_error,
            result->basis_has_nan_or_inf,
            result->ritz_vectors_orthogonality_frobenius_error,
            result->ritz_vectors_orthogonality_max_abs_error,
            result->ritz_vectors_has_nan_or_inf, 1);
        result->davidson_residual_before = current_maximum;
        result->davidson_target_index = current_max_index;
        if (ipt_block_build_orthonormal_basis(
                raw_basis, n, basis_cols, ortho_repeats,
                &orthonormal_basis)) {
            for (int step = 1; step <= steps; ++step) {
                int continuation = extra_steps > 0 &&
                                   step > baseline_steps;
                int auto_cluster_mode = 0;
                int legacy_force_mode = 0;
                int round_limit = k;
                std::vector<int> candidates;
                std::vector<int> active_indices;
                std::vector<double> trial_orthonormal_basis =
                    orthonormal_basis;
                std::vector<double> corrections_sorted;
                std::vector<double> correction_norms_before;
                std::vector<double> correction_norms_after;
                std::vector<double> protected_ritz_basis;
                int rejected_corrections = 0;
                double *d_trial_basis = NULL;
                double *d_trial_values = NULL;
                double *d_trial_vectors = NULL;
                double trial_rr_time = 0.0;
                double trial_basis_frobenius = NAN;
                double trial_basis_max = NAN;
                double trial_ritz_frobenius = NAN;
                double trial_ritz_max = NAN;
                int trial_basis_invalid = 0;
                int trial_ritz_invalid = 0;
                int trial_used_qr = 0;
                std::vector<double> trial_values((size_t)k, 0.0);
                std::vector<double> trial_vectors(
                    (size_t)n * (size_t)k, 0.0);
                std::vector<double> trial_residuals;
                double trial_maximum = NAN;
                int trial_max_index = -1;
                int accepted = 0;
                int trial_best_updated = 0;
                int any_trial_best_updated = 0;
                int retry_count = 0;
                double retry_alpha = 1.0;
                double retry_denom_clip = result->davidson_denom_clip;
                IPTTrialAcceptDecision accept_decision = {};
                IPTClusterTrialMetrics cluster_metrics = {};
                std::vector<IPTGenericClusterMetrics>
                    generic_cluster_metrics;
                int log_cluster_hard_locked = 0;
                int log_cluster_soft_locked_count = 0;
                char log_pair28_lock_state[16] = "inactive";
                char log_pair29_lock_state[16] = "inactive";

                ipt_cluster_set_default_metrics(&cluster_metrics);
                {
                    active_clusters = ipt_discover_auto_ritz_clusters(
                        k, host_ritz_values, current_residuals,
                        converged_tolerance, auto_gap_abs_tol,
                        auto_gap_rel_tol);
                    ipt_log_auto_cluster_discovery(
                        step, active_clusters, current_residuals,
                        converged_tolerance);
                    cluster_active =
                        ipt_active_clusters_include_exact(
                            active_clusters, 28, 29);
                    ipt_cluster_28_29_lock_states(
                        current_residuals, converged_tolerance,
                        cluster_active, soft_cluster_locking_enabled,
                        cluster_hard_locked, pair28_lock_state,
                        sizeof(pair28_lock_state), pair29_lock_state,
                        sizeof(pair29_lock_state),
                        &cluster_soft_locked_count);
                }
                for (const IPTActiveCluster &cluster : active_clusters) {
                    std::vector<int> correction_members =
                        ipt_cluster_correction_members(
                            cluster, current_residuals,
                            converged_tolerance);

                    if (cluster.source == IPT_CLUSTER_SOURCE_AUTO_GAP) {
                        auto_cluster_mode = 1;
                    } else {
                        legacy_force_mode = 1;
                    }
                    for (int member : cluster.members) {
                        if (current_residuals[(size_t)member] <=
                            converged_tolerance) {
                            IPTDavidsonSelectionEntry selection = {};

                            selection.davidson_step = step;
                            selection.pair_index = member;
                            selection.residual =
                                current_residuals[(size_t)member];
                            selection.selected_auto_cluster =
                                cluster.source ==
                                IPT_CLUSTER_SOURCE_AUTO_GAP;
                            selection.skipped_converged_in_cluster = 1;
                            selection_history.push_back(selection);
                        }
                    }
                    for (int member : correction_members) {
                        if (!ipt_vector_contains_int(candidates, member)) {
                            candidates.push_back(member);
                        }
                    }
                }
                if (!candidates.empty()) {
                    std::stable_sort(
                        candidates.begin(), candidates.end(),
                        [&](int left, int right) {
                            return current_residuals[(size_t)left] >
                                   current_residuals[(size_t)right];
                        });
                    round_limit = k;
                } else {
                    candidates.resize((size_t)k);
                    for (int i = 0; i < k; ++i) {
                        candidates[(size_t)i] = i;
                    }
                    std::stable_sort(
                        candidates.begin(), candidates.end(),
                        [&](int a, int b) {
                            return current_residuals[(size_t)a] >
                                   current_residuals[(size_t)b];
                        });
                    round_limit =
                        debug_active_max_enabled &&
                                debug_active_max > 0
                            ? debug_active_max
                            : 1;
                }
                fprintf(stderr,
                        "IPT Davidson active selection: step=%d "
                        "auto_cluster_mode=%d legacy_force_mode=%d "
                        "cluster_count=%zu correction_members=%s "
                        "correction_member_count=%zu fallback_cap=%d "
                        "active_max_used=%d\n",
                        step, auto_cluster_mode, legacy_force_mode,
                        active_clusters.size(),
                        ipt_join_indices(candidates).c_str(),
                        candidates.size(), round_limit,
                        debug_active_max_enabled ? 1 : 0);
                for (int pair = 0; pair < k; ++pair) {
                    if (current_residuals[(size_t)pair] >=
                        protect_tolerance) {
                        continue;
                    }
                    for (int sorted = 0; sorted < n; ++sorted) {
                        protected_ritz_basis.push_back(
                            host_ritz_vectors
                                [(size_t)perm[(size_t)sorted] +
                                 (size_t)pair * (size_t)n]);
                    }
                }
                for (int index : candidates) {
                    IPTDavidsonSelectionEntry selection = {};
                    double norm_before = NAN;
                    double norm_after = NAN;

                    selection.davidson_step = step;
                    selection.pair_index = index;
                    selection.residual =
                        current_residuals[(size_t)index];
                    selection
                        .skipped_locked_old_logic_should_not_happen = 0;
                    for (const IPTActiveCluster &cluster :
                         active_clusters) {
                        if (ipt_vector_contains_int(cluster.members,
                                                    index) &&
                            cluster.source ==
                                IPT_CLUSTER_SOURCE_AUTO_GAP) {
                            selection.selected_auto_cluster = 1;
                            break;
                        }
                    }
                    selection.selected_forced =
                        !selection.selected_auto_cluster &&
                        ipt_vector_contains_int(forced_pairs, index);
                    selection.selected_by_residual =
                        active_clusters.empty() &&
                        !selection.selected_forced;
                    if ((int)active_indices.size() >= round_limit) {
                        break;
                    }
                    if (current_residuals[(size_t)index] <=
                        converged_tolerance) {
                        selection.skipped_converged_in_cluster = 1;
                        selection_history.push_back(selection);
                        ++rejected_corrections;
                        continue;
                    }
                    {
                        std::vector<double> correction;
                        std::vector<double> sorted_correction(
                            (size_t)n, 0.0);

                        if (!ipt_block_build_davidson_correction(
                                col_ptr, row_ind, matrix_values, n,
                                original_diagonal, host_ritz_values,
                                host_ritz_vectors, index,
                                result->davidson_denom_clip,
                                &correction)) {
                            selection.skipped_linear_dependent = 1;
                            selection_history.push_back(selection);
                            ++rejected_corrections;
                            continue;
                        }
                        norm_before =
                            ipt_block_vector_norm(correction);
                        for (int sorted = 0; sorted < n; ++sorted) {
                            sorted_correction[(size_t)sorted] =
                                correction[(size_t)perm[(size_t)sorted]];
                        }
                        if (!protected_ritz_basis.empty()) {
                            ipt_block_project_out(
                                &sorted_correction,
                                protected_ritz_basis, n,
                                (int)(protected_ritz_basis.size() /
                                      (size_t)n),
                                ortho_repeats);
                        }
                        ipt_block_project_out(
                            &sorted_correction,
                            trial_orthonormal_basis, n,
                            (int)(trial_orthonormal_basis.size() /
                                  (size_t)n),
                            ortho_repeats);
                        if (!ipt_block_normalize_direction(
                                &sorted_correction, norm_before,
                                &norm_after)) {
                            selection.skipped_linear_dependent = 1;
                            selection.correction_norm_before_ortho =
                                norm_before;
                            selection.correction_norm_after_ortho =
                                norm_after;
                            selection_history.push_back(selection);
                            ++rejected_corrections;
                            continue;
                        }
                        if (ipt_block_max_abs_overlap(
                                sorted_correction,
                                trial_orthonormal_basis, n,
                                (int)(trial_orthonormal_basis.size() /
                                      (size_t)n)) > 1.0e-10 ||
                            (!protected_ritz_basis.empty() &&
                             ipt_block_max_abs_overlap(
                                 sorted_correction,
                                 protected_ritz_basis, n,
                                 (int)(protected_ritz_basis.size() /
                                       (size_t)n)) > 1.0e-10)) {
                            selection.skipped_linear_dependent = 1;
                            selection.correction_norm_before_ortho =
                                norm_before;
                            selection.correction_norm_after_ortho =
                                norm_after;
                            selection_history.push_back(selection);
                            ++rejected_corrections;
                            continue;
                        }
                        selection.correction_norm_before_ortho =
                            norm_before;
                        selection.correction_norm_after_ortho =
                            norm_after;
                        selection_history.push_back(selection);
                        trial_orthonormal_basis.insert(
                            trial_orthonormal_basis.end(),
                            sorted_correction.begin(),
                            sorted_correction.end());
                        corrections_sorted.insert(
                            corrections_sorted.end(),
                            sorted_correction.begin(),
                            sorted_correction.end());
                        active_indices.push_back(index);
                        correction_norms_before.push_back(norm_before);
                        correction_norms_after.push_back(norm_after);
                    }
                }
                if (active_indices.empty()) {
                    if (continuation) {
                        IPTDavidsonBlockHistoryEntry entry = {};

                        entry.davidson_step = step;
                        snprintf(entry.active_pairs,
                                 sizeof(entry.active_pairs), "none");
                        entry.accepted_corrections = 0;
                        entry.rejected_corrections =
                            rejected_corrections;
                        entry.residual_before_global =
                            current_maximum;
                        entry.residual_after_global =
                            current_maximum;
                        entry.pair_28_before =
                            k > 28 ? current_residuals[28] : NAN;
                        entry.pair_28_after = entry.pair_28_before;
                        entry.pair_29_before =
                            k > 29 ? current_residuals[29] : NAN;
                        entry.pair_29_after = entry.pair_29_before;
                        entry.accepted = 0;
                        snprintf(entry.reject_reason,
                                 sizeof(entry.reject_reason),
                                 "no_valid_correction");
                        entry.basis_cols_before = basis_cols;
                        entry.basis_cols_after = basis_cols;
                        entry.orthogonality_error_before =
                            result
                                ->ritz_vectors_orthogonality_max_abs_error;
                        entry.orthogonality_error_after =
                            entry.orthogonality_error_before;
                        entry.best_so_far_updated = 0;
                        entry.best_so_far_step = best_so_far.step;
                        entry.best_so_far_max_residual =
                            best_so_far.max_residual;
                        entry.pair28_best_residual =
                            best_so_far.pair28_best_residual;
                        entry.pair29_best_residual =
                            best_so_far.pair29_best_residual;
                        {
                            IPTTrialAcceptDecision no_correction = {};

                            no_correction.relaxed_accept_enabled =
                                relaxed_accept_enabled;
                            no_correction.finite_ok = 1;
                            no_correction.locked_safe = 1;
                            snprintf(no_correction.reason,
                                     sizeof(no_correction.reason),
                                     "no_valid_correction");
                            ipt_block_fill_block_history_accept_fields(
                                &entry, no_correction, 0, 1.0,
                                result->davidson_denom_clip,
                                accepted_steps_count,
                                rejected_steps_count + 1,
                                min_accepted_steps, 0);
                        }
                        generic_cluster_metrics =
                            ipt_compute_active_cluster_metrics(
                                n, k, active_clusters,
                                cluster_reference_vectors,
                                host_ritz_vectors, current_residuals,
                                current_residuals,
                                converged_tolerance,
                                protect_tolerance, accept_rel_slack,
                                accept_abs_slack,
                                locked_degrade_rel_slack,
                                locked_degrade_abs_slack);
                        ipt_copy_legacy_28_29_cluster_metrics(
                            k, host_ritz_values,
                            generic_cluster_metrics,
                            &cluster_metrics);
                        ipt_block_fill_block_history_cluster_fields(
                            &entry, cluster_metrics, pair28_lock_state,
                            pair29_lock_state, cluster_hard_locked,
                            cluster_soft_locked_count);
                        block_history.push_back(entry);
                        ++rejected_steps_count;
                        break;
                    }
                    if (extra_steps > 0 &&
                        accepted_steps_count >= min_accepted_steps) {
                        early_jump_to_continuation = 1;
                        step = baseline_steps;
                        continue;
                    }
                    if (extra_steps > 0) {
                        continue;
                    }
                    break;
                }
                {
                    int trial_basis_cols =
                        basis_cols + (int)active_indices.size();
                    const double *current_basis =
                        d_davidson_basis != NULL ? d_davidson_basis
                                                : d_combined_basis;

                    CUDA_CHECK(cudaMalloc(
                        (void **)&d_trial_basis,
                        (size_t)n * (size_t)trial_basis_cols *
                            sizeof(double)));
                    CUDA_CHECK(cudaMemcpy(
                        d_trial_basis, current_basis,
                        (size_t)n * (size_t)basis_cols * sizeof(double),
                        cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemcpy(
                        d_trial_basis + (size_t)n * (size_t)basis_cols,
                        corrections_sorted.data(),
                        corrections_sorted.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
                    status = ipt_block_cluster_rayleigh_ritz_gpu(
                        sparse_handle, sparse_matrix, cublas_handle,
                        d_trial_basis, n, trial_basis_cols, k, d_perm,
                        &d_trial_values, &d_trial_vectors, &trial_rr_time,
                        &trial_used_qr, &trial_basis_frobenius,
                        &trial_basis_max, &trial_basis_invalid,
                        &trial_ritz_frobenius, &trial_ritz_max,
                        &trial_ritz_invalid);
                    if (status != IPT_CUDA_SUCCESS) {
                        cudaFree(d_trial_basis);
                        cudaFree(d_trial_values);
                        cudaFree(d_trial_vectors);
                        goto cleanup;
                    }
                    result->rayleigh_ritz_time_sec += trial_rr_time;
                    result->matvecs += trial_basis_cols;
                    CUDA_CHECK(cudaMemcpy(
                        trial_vectors.data(), d_trial_vectors,
                        trial_vectors.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(
                        trial_values.data(), d_trial_values,
                        trial_values.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
                    if (ipt_sort_host_ritz_pairs(
                            n, k, &trial_values, &trial_vectors)) {
                        CUDA_CHECK(cudaMemcpy(
                            d_trial_values, trial_values.data(),
                            trial_values.size() * sizeof(double),
                            cudaMemcpyHostToDevice));
                        CUDA_CHECK(cudaMemcpy(
                            d_trial_vectors, trial_vectors.data(),
                            trial_vectors.size() * sizeof(double),
                            cudaMemcpyHostToDevice));
                    }
                    ipt_block_host_residuals(
                        col_ptr, row_ind, matrix_values, n, k, trial_values,
                        trial_vectors, matrix_norm, &trial_residuals,
                        &trial_maximum, &trial_max_index);
                    trial_best_updated = ipt_best_so_far_note(
                        &best_so_far,
                        continuation ? "davidson_continuation"
                                     : "davidson",
                        step, trial_values, trial_vectors, trial_residuals,
                        trial_maximum, trial_max_index, trial_basis_cols,
                        trial_used_qr, trial_basis_frobenius,
                        trial_basis_max, trial_basis_invalid,
                        trial_ritz_frobenius, trial_ritz_max,
                        trial_ritz_invalid, 0);
                    any_trial_best_updated = trial_best_updated;
                    generic_cluster_metrics =
                        ipt_compute_active_cluster_metrics(
                            n, k, active_clusters,
                            cluster_reference_vectors, trial_vectors,
                            current_residuals, trial_residuals,
                            converged_tolerance, protect_tolerance,
                            accept_rel_slack, accept_abs_slack,
                            locked_degrade_rel_slack,
                            locked_degrade_abs_slack);
                    ipt_copy_legacy_28_29_cluster_metrics(
                        k, host_ritz_values, generic_cluster_metrics,
                        &cluster_metrics);
                    accept_decision = ipt_block_trial_accept_decision(
                        current_residuals, trial_residuals,
                        active_indices, forced_pairs, current_maximum,
                        trial_maximum, accept_only_if_improves,
                        relaxed_accept_enabled, accept_rel_slack,
                        accept_abs_slack, active_pair_accept,
                        converged_tolerance, protect_tolerance,
                        locked_degrade_rel_slack,
                        locked_degrade_abs_slack, trial_basis_invalid,
                        trial_ritz_invalid, &cluster_metrics,
                        &generic_cluster_metrics,
                        cluster_aware_accept_enabled);
                    ipt_log_generic_cluster_trial(
                        step, 0, active_clusters,
                        generic_cluster_metrics);
                    accepted = accept_decision.accepted;
                    if (!accepted && retry_on_reject) {
                        std::vector<int> base_active_indices =
                            active_indices;
                        std::vector<double> base_corrections_sorted =
                            corrections_sorted;
                        std::vector<double> base_trial_orthonormal_basis =
                            trial_orthonormal_basis;
                        std::vector<double> base_norms_before =
                            correction_norms_before;
                        std::vector<double> base_norms_after =
                            correction_norms_after;
                        auto free_trial_buffers = [&]() {
                            cudaFree(d_trial_basis);
                            cudaFree(d_trial_values);
                            cudaFree(d_trial_vectors);
                            d_trial_basis = NULL;
                            d_trial_values = NULL;
                            d_trial_vectors = NULL;
                        };
                        auto evaluate_retry =
                            [&](const std::vector<int> &retry_active,
                                const std::vector<double> &retry_basis,
                                const std::vector<double> &retry_corrections,
                                const std::vector<double> &retry_norms_before,
                                const std::vector<double> &retry_norms_after,
                                double alpha, double denom_clip,
                                int retry_index) -> int {
                            cudaError_t cuda_status = cudaSuccess;
                            int retry_basis_cols =
                                basis_cols + (int)retry_active.size();
                            const double *current_basis =
                                d_davidson_basis != NULL
                                    ? d_davidson_basis
                                    : d_combined_basis;

                            if (retry_active.empty()) {
                                return IPT_CUDA_SUCCESS;
                            }
                            free_trial_buffers();
                            active_indices = retry_active;
                            trial_orthonormal_basis = retry_basis;
                            corrections_sorted = retry_corrections;
                            correction_norms_before = retry_norms_before;
                            correction_norms_after = retry_norms_after;
                            trial_basis_cols = retry_basis_cols;
                            retry_alpha = alpha;
                            retry_denom_clip = denom_clip;
                            retry_count = retry_index;
                            trial_rr_time = 0.0;
                            trial_basis_frobenius = NAN;
                            trial_basis_max = NAN;
                            trial_ritz_frobenius = NAN;
                            trial_ritz_max = NAN;
                            trial_basis_invalid = 0;
                            trial_ritz_invalid = 0;
                            trial_used_qr = 0;
                            cuda_status = cudaMalloc(
                                (void **)&d_trial_basis,
                                (size_t)n * (size_t)trial_basis_cols *
                                    sizeof(double));
                            if (cuda_status != cudaSuccess) {
                                fprintf(stderr,
                                        "CUDA error at %s:%d: %s\n",
                                        __FILE__, __LINE__,
                                        cudaGetErrorString(cuda_status));
                                return IPT_CUDA_CUDA_ERROR;
                            }
                            cuda_status = cudaMemcpy(
                                d_trial_basis, current_basis,
                                (size_t)n * (size_t)basis_cols *
                                    sizeof(double),
                                cudaMemcpyDeviceToDevice);
                            if (cuda_status != cudaSuccess) {
                                fprintf(stderr,
                                        "CUDA error at %s:%d: %s\n",
                                        __FILE__, __LINE__,
                                        cudaGetErrorString(cuda_status));
                                return IPT_CUDA_CUDA_ERROR;
                            }
                            cuda_status = cudaMemcpy(
                                d_trial_basis +
                                    (size_t)n * (size_t)basis_cols,
                                corrections_sorted.data(),
                                corrections_sorted.size() * sizeof(double),
                                cudaMemcpyHostToDevice);
                            if (cuda_status != cudaSuccess) {
                                fprintf(stderr,
                                        "CUDA error at %s:%d: %s\n",
                                        __FILE__, __LINE__,
                                        cudaGetErrorString(cuda_status));
                                return IPT_CUDA_CUDA_ERROR;
                            }
                            status = ipt_block_cluster_rayleigh_ritz_gpu(
                                sparse_handle, sparse_matrix, cublas_handle,
                                d_trial_basis, n, trial_basis_cols, k,
                                d_perm, &d_trial_values, &d_trial_vectors,
                                &trial_rr_time, &trial_used_qr,
                                &trial_basis_frobenius, &trial_basis_max,
                                &trial_basis_invalid, &trial_ritz_frobenius,
                                &trial_ritz_max, &trial_ritz_invalid);
                            if (status != IPT_CUDA_SUCCESS) {
                                return status;
                            }
                            result->rayleigh_ritz_time_sec +=
                                trial_rr_time;
                            result->matvecs += trial_basis_cols;
                            cuda_status = cudaMemcpy(
                                trial_vectors.data(), d_trial_vectors,
                                trial_vectors.size() * sizeof(double),
                                cudaMemcpyDeviceToHost);
                            if (cuda_status != cudaSuccess) {
                                fprintf(stderr,
                                        "CUDA error at %s:%d: %s\n",
                                        __FILE__, __LINE__,
                                        cudaGetErrorString(cuda_status));
                                return IPT_CUDA_CUDA_ERROR;
                            }
                            cuda_status = cudaMemcpy(
                                trial_values.data(), d_trial_values,
                                trial_values.size() * sizeof(double),
                                cudaMemcpyDeviceToHost);
                            if (cuda_status != cudaSuccess) {
                                fprintf(stderr,
                                        "CUDA error at %s:%d: %s\n",
                                        __FILE__, __LINE__,
                                        cudaGetErrorString(cuda_status));
                                return IPT_CUDA_CUDA_ERROR;
                            }
                            if (ipt_sort_host_ritz_pairs(
                                    n, k, &trial_values,
                                    &trial_vectors)) {
                                cuda_status = cudaMemcpy(
                                    d_trial_values,
                                    trial_values.data(),
                                    trial_values.size() *
                                        sizeof(double),
                                    cudaMemcpyHostToDevice);
                                if (cuda_status == cudaSuccess) {
                                    cuda_status = cudaMemcpy(
                                        d_trial_vectors,
                                        trial_vectors.data(),
                                        trial_vectors.size() *
                                            sizeof(double),
                                        cudaMemcpyHostToDevice);
                                }
                                if (cuda_status != cudaSuccess) {
                                    fprintf(
                                        stderr,
                                        "CUDA error at %s:%d: %s\n",
                                        __FILE__, __LINE__,
                                        cudaGetErrorString(cuda_status));
                                    return IPT_CUDA_CUDA_ERROR;
                                }
                            }
                            ipt_block_host_residuals(
                                col_ptr, row_ind, matrix_values, n, k,
                                trial_values, trial_vectors, matrix_norm,
                                &trial_residuals, &trial_maximum,
                                &trial_max_index);
                            if (ipt_best_so_far_note(
                                    &best_so_far,
                                    continuation
                                        ? "davidson_continuation_retry"
                                        : "davidson_retry",
                                    step, trial_values, trial_vectors,
                                    trial_residuals, trial_maximum,
                                    trial_max_index, trial_basis_cols,
                                    trial_used_qr, trial_basis_frobenius,
                                    trial_basis_max, trial_basis_invalid,
                                    trial_ritz_frobenius, trial_ritz_max,
                                    trial_ritz_invalid, 0)) {
                                any_trial_best_updated = 1;
                            }
                            generic_cluster_metrics =
                                ipt_compute_active_cluster_metrics(
                                    n, k, active_clusters,
                                    cluster_reference_vectors,
                                    trial_vectors, current_residuals,
                                    trial_residuals,
                                    converged_tolerance,
                                    protect_tolerance,
                                    accept_rel_slack, accept_abs_slack,
                                    locked_degrade_rel_slack,
                                    locked_degrade_abs_slack);
                            ipt_copy_legacy_28_29_cluster_metrics(
                                k, host_ritz_values,
                                generic_cluster_metrics,
                                &cluster_metrics);
                            accept_decision =
                                ipt_block_trial_accept_decision(
                                    current_residuals, trial_residuals,
                                    active_indices, forced_pairs,
                                    current_maximum, trial_maximum,
                                    accept_only_if_improves,
                                    relaxed_accept_enabled,
                                    accept_rel_slack, accept_abs_slack,
                                    active_pair_accept,
                                    converged_tolerance, protect_tolerance,
                                    locked_degrade_rel_slack,
                                    locked_degrade_abs_slack,
                                    trial_basis_invalid,
                                    trial_ritz_invalid, &cluster_metrics,
                                    &generic_cluster_metrics,
                                    cluster_aware_accept_enabled);
                            ipt_log_generic_cluster_trial(
                                step, retry_index, active_clusters,
                                generic_cluster_metrics);
                            accepted = accept_decision.accepted;
                            return IPT_CUDA_SUCCESS;
                        };
                        auto build_retry_corrections =
                            [&](double denom_clip,
                                std::vector<int> *retry_active,
                                std::vector<double> *retry_basis,
                                std::vector<double> *retry_corrections,
                                std::vector<double> *retry_norms_before,
                                std::vector<double> *retry_norms_after) {
                            retry_active->clear();
                            retry_corrections->clear();
                            retry_norms_before->clear();
                            retry_norms_after->clear();
                            *retry_basis = orthonormal_basis;
                            for (int index : candidates) {
                                double norm_before = NAN;
                                double norm_after = NAN;
                                std::vector<double> correction;
                                std::vector<double> sorted_correction(
                                    (size_t)n, 0.0);

                                if ((int)retry_active->size() >=
                                    round_limit) {
                                    break;
                                }
                                if (current_residuals[(size_t)index] <=
                                    converged_tolerance) {
                                    continue;
                                }
                                if (!ipt_block_build_davidson_correction(
                                        col_ptr, row_ind, matrix_values, n,
                                        original_diagonal, host_ritz_values,
                                        host_ritz_vectors, index,
                                        denom_clip, &correction)) {
                                    continue;
                                }
                                norm_before =
                                    ipt_block_vector_norm(correction);
                                for (int sorted = 0; sorted < n;
                                     ++sorted) {
                                    sorted_correction[(size_t)sorted] =
                                        correction[(size_t)perm
                                                       [(size_t)sorted]];
                                }
                                if (!protected_ritz_basis.empty()) {
                                    ipt_block_project_out(
                                        &sorted_correction,
                                        protected_ritz_basis, n,
                                        (int)(protected_ritz_basis.size() /
                                              (size_t)n),
                                        ortho_repeats);
                                }
                                ipt_block_project_out(
                                    &sorted_correction, *retry_basis, n,
                                    (int)(retry_basis->size() /
                                          (size_t)n),
                                    ortho_repeats);
                                if (!ipt_block_normalize_direction(
                                        &sorted_correction, norm_before,
                                        &norm_after)) {
                                    continue;
                                }
                                if (ipt_block_max_abs_overlap(
                                        sorted_correction, *retry_basis,
                                        n,
                                        (int)(retry_basis->size() /
                                              (size_t)n)) >
                                        1.0e-10 ||
                                    (!protected_ritz_basis.empty() &&
                                     ipt_block_max_abs_overlap(
                                         sorted_correction,
                                         protected_ritz_basis, n,
                                         (int)(
                                             protected_ritz_basis.size() /
                                             (size_t)n)) > 1.0e-10)) {
                                    continue;
                                }
                                retry_basis->insert(
                                    retry_basis->end(),
                                    sorted_correction.begin(),
                                    sorted_correction.end());
                                retry_corrections->insert(
                                    retry_corrections->end(),
                                    sorted_correction.begin(),
                                    sorted_correction.end());
                                retry_active->push_back(index);
                                retry_norms_before->push_back(norm_before);
                                retry_norms_after->push_back(norm_after);
                            }
                        };
                        int next_retry = 1;
                        int retry_status = IPT_CUDA_SUCCESS;

                        for (double alpha : retry_damping_list) {
                            std::vector<double> damped_corrections =
                                base_corrections_sorted;

                            for (double &value : damped_corrections) {
                                value *= alpha;
                            }
                            retry_status = evaluate_retry(
                                base_active_indices,
                                base_trial_orthonormal_basis,
                                damped_corrections, base_norms_before,
                                base_norms_after, alpha,
                                result->davidson_denom_clip, next_retry++);
                            if (retry_status != IPT_CUDA_SUCCESS) {
                                status = retry_status;
                                goto cleanup;
                            }
                            if (accepted) {
                                break;
                            }
                        }
                        if (!accepted) {
                            for (double mult : retry_denom_clip_mults) {
                                std::vector<int> retry_active;
                                std::vector<double> retry_basis;
                                std::vector<double> retry_corrections;
                                std::vector<double> retry_norms_before;
                                std::vector<double> retry_norms_after;
                                double denom_clip =
                                    result->davidson_denom_clip * mult;

                                build_retry_corrections(
                                    denom_clip, &retry_active, &retry_basis,
                                    &retry_corrections, &retry_norms_before,
                                    &retry_norms_after);
                                retry_status = evaluate_retry(
                                    retry_active, retry_basis,
                                    retry_corrections, retry_norms_before,
                                    retry_norms_after, 1.0, denom_clip,
                                    next_retry++);
                                if (retry_status != IPT_CUDA_SUCCESS) {
                                    status = retry_status;
                                    goto cleanup;
                                }
                                if (accepted) {
                                    break;
                                }
                            }
                        }
                    }
                    trial_best_updated = any_trial_best_updated;
                    fprintf(stderr,
                            "IPT Davidson trial decision: step=%d "
                            "selected_active_pairs=%s accepted=%d "
                            "global_ok=%d active_ok=%d locked_safe=%d "
                            "reason=%s retry_count=%d basis_cols_before=%d "
                            "basis_cols_trial=%d\n",
                            step,
                            ipt_join_indices(active_indices).c_str(),
                            accepted, accept_decision.global_ok,
                            accept_decision.active_ok,
                            accept_decision.locked_safe,
                            accept_decision.reason, retry_count,
                            basis_cols, trial_basis_cols);
                    {
                        int accepted_steps_after =
                            accepted_steps_count + (accepted ? 1 : 0);
                        int rejected_steps_after =
                            rejected_steps_count + (accepted ? 0 : 1);
                        int step_early_jump =
                            !accepted && !continuation && extra_steps > 0 &&
                            accepted_steps_count >= min_accepted_steps;
                        log_cluster_hard_locked = cluster_hard_locked;
                        log_cluster_soft_locked_count =
                            cluster_soft_locked_count;
                        snprintf(log_pair28_lock_state,
                                 sizeof(log_pair28_lock_state), "%s",
                                 pair28_lock_state);
                        snprintf(log_pair29_lock_state,
                                 sizeof(log_pair29_lock_state), "%s",
                                 pair29_lock_state);
                        if (accepted && cluster_active) {
                            int cluster_stable =
                                trial_residuals.size() > 29U &&
                                trial_residuals[28] <=
                                    converged_tolerance &&
                                trial_residuals[29] <=
                                    converged_tolerance &&
                                cluster_metrics
                                        .cluster_residual_fro_after <=
                                    converged_tolerance;

                            if (cluster_stable) {
                                ++cluster_stable_count;
                            } else {
                                cluster_stable_count = 0;
                            }
                            if (soft_cluster_locking_enabled &&
                                cluster_stable_count >= 2) {
                                cluster_hard_locked = 1;
                            }
                            ipt_cluster_28_29_lock_states(
                                trial_residuals, converged_tolerance,
                                cluster_active,
                                soft_cluster_locking_enabled,
                                cluster_hard_locked,
                                log_pair28_lock_state,
                                sizeof(log_pair28_lock_state),
                                log_pair29_lock_state,
                                sizeof(log_pair29_lock_state),
                                &log_cluster_soft_locked_count);
                            log_cluster_hard_locked = cluster_hard_locked;
                        }

                    if (continuation) {
                        IPTDavidsonBlockHistoryEntry block_entry = {};
                        std::string active_text;
                        std::string norm_before_text;
                        std::string norm_after_text;

                        for (size_t active = 0;
                             active < active_indices.size(); ++active) {
                            char value[64];

                            if (!active_text.empty()) {
                                active_text += ";";
                                norm_before_text += ";";
                                norm_after_text += ";";
                            }
                            active_text +=
                                std::to_string(active_indices[active]);
                            snprintf(
                                value, sizeof(value), "%.17g",
                                correction_norms_before[active]);
                            norm_before_text += value;
                            snprintf(
                                value, sizeof(value), "%.17g",
                                correction_norms_after[active]);
                            norm_after_text += value;
                        }
                        block_entry.davidson_step = step;
                        snprintf(
                            block_entry.active_pairs,
                            sizeof(block_entry.active_pairs), "%s",
                            active_text.c_str());
                        block_entry.accepted_corrections =
                            accepted ? (int)active_indices.size() : 0;
                        block_entry.rejected_corrections =
                            rejected_corrections +
                            (accepted ? 0
                                      : (int)active_indices.size());
                        snprintf(
                            block_entry.correction_norm_before_ortho,
                            sizeof(block_entry
                                       .correction_norm_before_ortho),
                            "%s", norm_before_text.c_str());
                        snprintf(
                            block_entry.correction_norm_after_ortho,
                            sizeof(block_entry
                                       .correction_norm_after_ortho),
                            "%s", norm_after_text.c_str());
                        block_entry.residual_before_global =
                            current_maximum;
                        block_entry.residual_after_global =
                            trial_maximum;
                        block_entry.pair_28_before =
                            k > 28 ? current_residuals[28] : NAN;
                        block_entry.pair_28_after =
                            k > 28 ? trial_residuals[28] : NAN;
                        block_entry.pair_29_before =
                            k > 29 ? current_residuals[29] : NAN;
                        block_entry.pair_29_after =
                            k > 29 ? trial_residuals[29] : NAN;
                        block_entry.accepted = accepted;
                        snprintf(
                            block_entry.reject_reason,
                            sizeof(block_entry.reject_reason), "%s",
                            accepted ? "none"
                                     : accept_decision.reason);
                        block_entry.basis_cols_before = basis_cols;
                        block_entry.basis_cols_after =
                            accepted ? trial_basis_cols : basis_cols;
                        block_entry.orthogonality_error_before =
                            result
                                ->ritz_vectors_orthogonality_max_abs_error;
                        block_entry.orthogonality_error_after =
                            trial_ritz_max;
                        block_entry.best_so_far_updated =
                            trial_best_updated;
                        block_entry.best_so_far_step =
                            best_so_far.step;
                        block_entry.best_so_far_max_residual =
                            best_so_far.max_residual;
                        block_entry.pair28_best_residual =
                            best_so_far.pair28_best_residual;
                        block_entry.pair29_best_residual =
                            best_so_far.pair29_best_residual;
                        ipt_block_fill_block_history_accept_fields(
                            &block_entry, accept_decision, retry_count,
                            retry_alpha, retry_denom_clip,
                            accepted_steps_after, rejected_steps_after,
                            min_accepted_steps, step_early_jump);
                        ipt_block_fill_block_history_cluster_fields(
                            &block_entry, cluster_metrics,
                            log_pair28_lock_state,
                            log_pair29_lock_state,
                            log_cluster_hard_locked,
                            log_cluster_soft_locked_count);
                        block_history.push_back(block_entry);
                    }
                    for (int index : active_indices) {
                        IPTDavidsonHistoryEntry entry = {};

                        entry.davidson_step = step;
                        entry.active_pair_index = index;
                        entry.residual_before =
                            current_residuals[(size_t)index];
                        entry.residual_after =
                            trial_residuals[(size_t)index];
                        entry.accepted = accepted;
                        entry.basis_cols =
                            accepted ? trial_basis_cols : basis_cols;
                        entry.max_relative_eigen_residual =
                            accepted ? trial_maximum : current_maximum;
                        entry.max_relative_eigen_residual_index =
                            accepted ? trial_max_index : current_max_index;
                        entry.pair_28_residual =
                            k > 28
                                ? (accepted
                                       ? trial_residuals[28]
                                       : current_residuals[28])
                                : NAN;
                        entry.pair_29_residual =
                            k > 29
                                ? (accepted
                                       ? trial_residuals[29]
                                       : current_residuals[29])
                                : NAN;
                        entry.orthogonality_max_abs_error =
                            accepted
                                ? trial_ritz_max
                                : result
                                      ->ritz_vectors_orthogonality_max_abs_error;
                        entry.best_so_far_updated =
                            trial_best_updated;
                        entry.best_so_far_step = best_so_far.step;
                        entry.best_so_far_max_residual =
                            best_so_far.max_residual;
                        entry.pair28_best_residual =
                            best_so_far.pair28_best_residual;
                        entry.pair29_best_residual =
                            best_so_far.pair29_best_residual;
                        ipt_block_fill_davidson_history_accept_fields(
                            &entry, accept_decision, retry_count,
                            retry_alpha, retry_denom_clip,
                            accepted_steps_after, rejected_steps_after,
                            min_accepted_steps, step_early_jump);
                        ipt_block_fill_davidson_history_cluster_fields(
                            &entry, cluster_metrics,
                            log_pair28_lock_state,
                            log_pair29_lock_state,
                            log_cluster_hard_locked,
                            log_cluster_soft_locked_count);
                        history.push_back(entry);
                    }
                    }
                    result->davidson_residual_after = trial_maximum;
                    if (accepted) {
                        if (trial_best_updated) {
                            best_so_far.state_is_current = 1;
                        }
                        cudaFree(d_davidson_basis);
                        d_davidson_basis = d_trial_basis;
                        d_trial_basis = NULL;
                        cudaFree(d_ritz_values);
                        cudaFree(d_ritz_vectors);
                        d_ritz_values = d_trial_values;
                        d_ritz_vectors = d_trial_vectors;
                        d_trial_values = NULL;
                        d_trial_vectors = NULL;
                        basis_cols = trial_basis_cols;
                        orthonormal_basis.swap(
                            trial_orthonormal_basis);
                        host_ritz_values.swap(trial_values);
                        host_ritz_vectors.swap(trial_vectors);
                        current_residuals.swap(trial_residuals);
                        current_maximum = trial_maximum;
                        current_max_index = trial_max_index;
                        snprintf(pair28_lock_state,
                                 sizeof(pair28_lock_state), "%s",
                                 log_pair28_lock_state);
                        snprintf(pair29_lock_state,
                                 sizeof(pair29_lock_state), "%s",
                                 log_pair29_lock_state);
                        cluster_soft_locked_count =
                            log_cluster_soft_locked_count;
                        cluster_reference_vectors = host_ritz_vectors;
                        ipt_block_store_result_cluster_fields(
                            result, host_ritz_values, current_residuals, k,
                            cluster_active,
                            cluster_aware_accept_enabled,
                            soft_cluster_locking_enabled,
                            converged_tolerance, cluster_hard_locked);
                        result->davidson_accepted = 1;
                        result->rayleigh_ritz_used_qr = trial_used_qr;
                        result->basis_orthogonality_frobenius_error =
                            trial_basis_frobenius;
                        result->basis_orthogonality_max_abs_error =
                            trial_basis_max;
                        result->basis_has_nan_or_inf =
                            trial_basis_invalid;
                        result
                            ->ritz_vectors_orthogonality_frobenius_error =
                            trial_ritz_frobenius;
                        result
                            ->ritz_vectors_orthogonality_max_abs_error =
                            trial_ritz_max;
                        result->ritz_vectors_has_nan_or_inf =
                            trial_ritz_invalid;
                        for (size_t active = 0;
                             active < active_indices.size(); ++active) {
                            recent_corrections.push_back(
                                std::vector<double>(
                                    corrections_sorted.begin() +
                                        active * (size_t)n,
                                    corrections_sorted.begin() +
                                        (active + 1U) * (size_t)n));
                        }
                        while ((int)recent_corrections.size() >
                               restart_keep_extra) {
                            recent_corrections.erase(
                                recent_corrections.begin());
                        }
                        accepted_since_restart +=
                            (int)active_indices.size();
                        if (restart_every > 0 &&
                            accepted_since_restart >= restart_every) {
                            std::vector<double> restart_basis;
                            std::vector<double> restart_orthonormal_basis;
                            std::vector<std::vector<double>>
                                retained_corrections;
                            int restarted = 0;

                            restart_basis.reserve(
                                (size_t)n *
                                (size_t)(k + restart_keep_extra));
                            for (int col = 0; col < k; ++col) {
                                for (int sorted = 0; sorted < n;
                                     ++sorted) {
                                    restart_basis.push_back(
                                        host_ritz_vectors
                                            [(size_t)perm[(size_t)sorted] +
                                             (size_t)col * (size_t)n]);
                                }
                            }
                            if (ipt_block_build_orthonormal_basis(
                                    restart_basis, n, k,
                                    ortho_repeats,
                                    &restart_orthonormal_basis)) {
                                for (const std::vector<double> &saved :
                                     recent_corrections) {
                                    std::vector<double> direction = saved;

                                    if (!ipt_block_orthogonalize_direction(
                                            &direction,
                                            restart_orthonormal_basis, n,
                                            (int)(
                                                restart_orthonormal_basis
                                                    .size() /
                                                (size_t)n),
                                            ortho_repeats)) {
                                        continue;
                                    }
                                    restart_basis.insert(
                                        restart_basis.end(),
                                        direction.begin(),
                                        direction.end());
                                    restart_orthonormal_basis.insert(
                                        restart_orthonormal_basis.end(),
                                        direction.begin(),
                                        direction.end());
                                    retained_corrections.push_back(
                                        direction);
                                }
                                {
                                    double *d_restart_basis = NULL;
                                    int restart_basis_cols =
                                        (int)(restart_basis.size() /
                                              (size_t)n);

                                    CUDA_CHECK(cudaMalloc(
                                        (void **)&d_restart_basis,
                                        restart_basis.size() *
                                            sizeof(double)));
                                    CUDA_CHECK(cudaMemcpy(
                                        d_restart_basis,
                                        restart_basis.data(),
                                        restart_basis.size() *
                                            sizeof(double),
                                        cudaMemcpyHostToDevice));
                                    cudaFree(d_davidson_basis);
                                    d_davidson_basis = d_restart_basis;
                                    basis_cols = restart_basis_cols;
                                    orthonormal_basis.swap(
                                        restart_orthonormal_basis);
                                    recent_corrections.swap(
                                        retained_corrections);
                                    accepted_since_restart = 0;
                                    ++result->davidson_restart_count;
                                    restarted = 1;
                                    ipt_block_orthogonality_stats(
                                        orthonormal_basis, n,
                                        basis_cols,
                                        &result
                                             ->basis_orthogonality_frobenius_error,
                                        &result
                                             ->basis_orthogonality_max_abs_error,
                                        &result
                                             ->basis_has_nan_or_inf);
                                    ipt_best_so_far_refresh_current_metadata(
                                        &best_so_far, basis_cols,
                                        result
                                            ->basis_orthogonality_frobenius_error,
                                        result
                                            ->basis_orthogonality_max_abs_error,
                                        result->basis_has_nan_or_inf);
                                }
                            }
                            if (restarted) {
                                if (continuation &&
                                    !block_history.empty()) {
                                    block_history.back().basis_cols_after =
                                        basis_cols;
                                }
                                for (size_t offset = 0;
                                     offset < active_indices.size();
                                     ++offset) {
                                    IPTDavidsonHistoryEntry &entry =
                                        history[history.size() - 1U -
                                                offset];

                                    entry.basis_cols = basis_cols;
                                    entry.restarted = 1;
                                }
                            }
                        }
                        ++accepted_steps_count;
                    }
                    cudaFree(d_trial_basis);
                    cudaFree(d_trial_values);
                    cudaFree(d_trial_vectors);
                    if (!accepted) {
                        ++rejected_steps_count;
                        if (!continuation && extra_steps > 0 &&
                            accepted_steps_count >= min_accepted_steps) {
                            early_jump_to_continuation = 1;
                            step = baseline_steps;
                            continue;
                        }
                        if (!continuation) {
                            continue;
                        }
                        break;
                    }
                }
            }
        }
        result->davidson_residual_after = current_maximum;
        result->accepted_steps = accepted_steps_count;
        result->rejected_steps = rejected_steps_count;
        result->early_jump_to_continuation = early_jump_to_continuation;
        ipt_block_store_result_cluster_fields(
            result, host_ritz_values, current_residuals, k,
            cluster_active, cluster_aware_accept_enabled,
            soft_cluster_locking_enabled, converged_tolerance,
            cluster_hard_locked);
        if (!history.empty()) {
            result->davidson_history = (IPTDavidsonHistoryEntry *)calloc(
                history.size(), sizeof(IPTDavidsonHistoryEntry));
            if (result->davidson_history == NULL) {
                status = IPT_CUDA_ALLOCATION_FAILED;
                goto cleanup;
            }
            memcpy(result->davidson_history, history.data(),
                   history.size() * sizeof(IPTDavidsonHistoryEntry));
            result->davidson_history_count = (int)history.size();
        }
        if (!selection_history.empty()) {
            result->davidson_selection_history =
                (IPTDavidsonSelectionEntry *)calloc(
                    selection_history.size(),
                    sizeof(IPTDavidsonSelectionEntry));
            if (result->davidson_selection_history == NULL) {
                status = IPT_CUDA_ALLOCATION_FAILED;
                goto cleanup;
            }
            memcpy(result->davidson_selection_history,
                   selection_history.data(),
                   selection_history.size() *
                       sizeof(IPTDavidsonSelectionEntry));
            result->davidson_selection_history_count =
                (int)selection_history.size();
        }
        if (!block_history.empty()) {
            result->davidson_block_history =
                (IPTDavidsonBlockHistoryEntry *)calloc(
                    block_history.size(),
                    sizeof(IPTDavidsonBlockHistoryEntry));
            if (result->davidson_block_history == NULL) {
                status = IPT_CUDA_ALLOCATION_FAILED;
                goto cleanup;
            }
            memcpy(result->davidson_block_history,
                   block_history.data(),
                   block_history.size() *
                       sizeof(IPTDavidsonBlockHistoryEntry));
            result->davidson_block_history_count =
                (int)block_history.size();
        }
    }
    if (ipt_cuda_env_flag("IPT_JD_LOCAL_CORRECTION")) {
        const char *accept_raw =
            getenv("IPT_JD_ACCEPT_ONLY_IF_IMPROVES");
        const char *outside_raw =
            getenv("IPT_JD_LOCAL_OUTSIDE_CORRECTION");
        const char *local_set_raw =
            getenv("IPT_JD_LOCAL_SET");
        int steps = ipt_cuda_env_int("IPT_JD_LOCAL_STEPS", 1);
        int window_start =
            ipt_cuda_env_int("IPT_JD_LOCAL_WINDOW_START", 25);
        int window_end =
            ipt_cuda_env_int("IPT_JD_LOCAL_WINDOW_END", 40);
        int max_dim =
            ipt_cuda_env_int("IPT_JD_LOCAL_MAX_DIM", 80);
        int support_top =
            ipt_cuda_env_int("IPT_JD_LOCAL_SUPPORT_TOP", 80);
        int ortho_repeats =
            ipt_cuda_env_int("IPT_DAVIDSON_ORTHO_REPEATS", 2);
        int relaxed_accept_enabled =
            ipt_cuda_env_flag_default("IPT_DAVIDSON_RELAXED_ACCEPT", 1);
        int active_pair_accept =
            ipt_cuda_env_flag_default("IPT_DAVIDSON_ACTIVE_PAIR_ACCEPT", 1);
        int cluster_aware_accept_enabled =
            ipt_cuda_env_flag_default("IPT_DAVIDSON_CLUSTER_AWARE_ACCEPT", 0);
        int soft_cluster_locking_enabled =
            ipt_cuda_env_flag_default("IPT_DAVIDSON_SOFT_CLUSTER_LOCKING", 0);
        int accept_only_if_improves =
            accept_raw == NULL
                ? 1
                : ipt_cuda_env_flag(
                      "IPT_JD_ACCEPT_ONLY_IF_IMPROVES");
        double accept_rel_slack = ipt_cuda_env_double(
            "IPT_DAVIDSON_ACCEPT_REL_SLACK", 1.0e-12);
        double accept_abs_slack = ipt_cuda_env_double(
            "IPT_DAVIDSON_ACCEPT_ABS_SLACK", 1.0e-15);
        double legacy_locked_degrade_slack = ipt_cuda_env_double(
            "IPT_DAVIDSON_LOCKED_DEGRADE_SLACK", 1.0e-8);
        double locked_degrade_rel_slack = ipt_cuda_env_double(
            "IPT_DAVIDSON_LOCKED_DEGRADE_REL_SLACK",
            legacy_locked_degrade_slack);
        double locked_degrade_abs_slack = ipt_cuda_env_double(
            "IPT_DAVIDSON_LOCKED_DEGRADE_ABS_SLACK", 1.0e-12);
        double cluster_gap_abs_tol = ipt_cuda_env_double(
            "IPT_DAVIDSON_CLUSTER_GAP_ABS_TOL", 1.0e-12);
        double cluster_gap_rel_tol = ipt_cuda_env_double(
            "IPT_DAVIDSON_CLUSTER_GAP_REL_TOL", 1.0e-10);
        int outside_diagonal =
            outside_raw != NULL &&
            (strcmp(outside_raw, "diagonal") == 0 ||
             strcmp(outside_raw, "DIAGONAL") == 0 ||
             strcmp(outside_raw, "1") == 0);
        int use_residual_support =
            local_set_raw != NULL &&
            (strcmp(local_set_raw, "residual_support") == 0 ||
             strcmp(local_set_raw, "RESIDUAL_SUPPORT") == 0 ||
             strcmp(local_set_raw, "support") == 0);
        double damping = ipt_cuda_env_double(
            "IPT_JD_LOCAL_DAMPING", 1.0e-8);
        double protect_tolerance = ipt_cuda_env_double(
            "IPT_DAVIDSON_PROTECT_TOL", 1.0e-10);
        double locked_tolerance = ipt_cuda_env_double(
            "IPT_DAVIDSON_LOCKED_TOL", 1.0e-12);
        double matrix_norm = ipt_block_host_matrix_inf_norm(
            col_ptr, row_ind, matrix_values, n);
        std::vector<int> active_pairs = ipt_jd_parse_active_pairs(
            getenv("IPT_JD_LOCAL_ACTIVE_PAIRS"), k);
        std::vector<int> forced_cluster_pairs;
        std::vector<double> host_ritz_values((size_t)k, 0.0);
        std::vector<double> host_ritz_vectors(
            (size_t)n * (size_t)k, 0.0);
        std::vector<double> current_residuals;
        std::vector<double> raw_basis;
        std::vector<double> orthonormal_basis;
        std::vector<IPTJDLocalHistoryEntry> history;
        double current_maximum = NAN;
        int current_max_index = -1;
        int accepted_steps_count = result->accepted_steps;
        int rejected_steps_count = result->rejected_steps;
        int min_accepted_steps = result->min_accepted_steps;
        std::vector<double> cluster_reference_vectors;
        int cluster_active = 0;
        int cluster_hard_locked = result->cluster_hard_locked;
        int cluster_stable_count = cluster_hard_locked ? 2 : 0;
        int cluster_soft_locked_count = 0;
        char pair28_lock_state[16] = "inactive";
        char pair29_lock_state[16] = "inactive";

        steps = std::max(1, steps);
        ortho_repeats = std::max(2, ortho_repeats);
        max_dim = std::max(1, std::min(max_dim, n));
        support_top = std::max(1, std::min(support_top, n));
        window_start = std::max(0, std::min(window_start, n - 1));
        window_end = std::max(window_start, std::min(window_end, n - 1));
        if (window_end - window_start + 1 > max_dim) {
            window_end = window_start + max_dim - 1;
        }
        if (damping <= 0.0) {
            damping = 1.0e-8;
        }
        result->jd_local_attempted = 1;
        CUDA_CHECK(cudaMemcpy(host_ritz_vectors.data(), d_ritz_vectors,
                              host_ritz_vectors.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_ritz_values.data(), d_ritz_values,
                              host_ritz_values.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
        {
            const double *current_basis =
                d_davidson_basis != NULL ? d_davidson_basis
                                        : d_combined_basis;

            raw_basis.assign((size_t)n * (size_t)basis_cols, 0.0);
            CUDA_CHECK(cudaMemcpy(
                raw_basis.data(), current_basis,
                raw_basis.size() * sizeof(double),
                cudaMemcpyDeviceToHost));
        }
        ipt_block_host_residuals(
            col_ptr, row_ind, matrix_values, n, k, host_ritz_values,
            host_ritz_vectors, matrix_norm, &current_residuals,
            &current_maximum, &current_max_index);
        {
            double gap = NAN;
            double relative_gap = NAN;

            cluster_active = ipt_cluster_28_29_active(
                k, host_ritz_values, active_pairs, forced_cluster_pairs,
                cluster_gap_abs_tol, cluster_gap_rel_tol, &gap,
                &relative_gap);
            ipt_cluster_copy_28_29_reference(n, k, host_ritz_vectors,
                                             &cluster_reference_vectors);
            ipt_cluster_28_29_lock_states(
                current_residuals, locked_tolerance, cluster_active,
                soft_cluster_locking_enabled, cluster_hard_locked,
                pair28_lock_state, sizeof(pair28_lock_state),
                pair29_lock_state, sizeof(pair29_lock_state),
                &cluster_soft_locked_count);
            ipt_block_store_result_cluster_fields(
                result, host_ritz_values, current_residuals, k,
                cluster_active, cluster_aware_accept_enabled,
                soft_cluster_locking_enabled, locked_tolerance,
                cluster_hard_locked);
        }
        ipt_best_so_far_note(
            &best_so_far, "jd_initial", 0, host_ritz_values,
            host_ritz_vectors, current_residuals, current_maximum,
            current_max_index, basis_cols, result->rayleigh_ritz_used_qr,
            result->basis_orthogonality_frobenius_error,
            result->basis_orthogonality_max_abs_error,
            result->basis_has_nan_or_inf,
            result->ritz_vectors_orthogonality_frobenius_error,
            result->ritz_vectors_orthogonality_max_abs_error,
            result->ritz_vectors_has_nan_or_inf, 1);
        if (ipt_block_build_orthonormal_basis(
                raw_basis, n, basis_cols, ortho_repeats,
                &orthonormal_basis)) {
            for (int step = 1; step <= steps; ++step) {
                std::vector<int> step_active_pairs;
                std::vector<double> trial_orthonormal_basis =
                    orthonormal_basis;
                std::vector<double> corrections_sorted;
                double *d_trial_basis = NULL;
                double *d_trial_values = NULL;
                double *d_trial_vectors = NULL;
                double trial_rr_time = 0.0;
                double trial_basis_frobenius = NAN;
                double trial_basis_max = NAN;
                double trial_ritz_frobenius = NAN;
                double trial_ritz_max = NAN;
                int trial_basis_invalid = 0;
                int trial_ritz_invalid = 0;
                int trial_used_qr = 0;
                std::vector<double> trial_values((size_t)k, 0.0);
                std::vector<double> trial_vectors(
                    (size_t)n * (size_t)k, 0.0);
                std::vector<double> trial_residuals;
                double trial_maximum = NAN;
                int trial_max_index = -1;
                int accepted = 0;
                int trial_best_updated = 0;
                IPTTrialAcceptDecision accept_decision = {};
                IPTClusterTrialMetrics cluster_metrics = {};

                ipt_cluster_set_default_metrics(&cluster_metrics);
                {
                    double gap = NAN;
                    double relative_gap = NAN;

                    cluster_active = ipt_cluster_28_29_active(
                        k, host_ritz_values, active_pairs,
                        forced_cluster_pairs, cluster_gap_abs_tol,
                        cluster_gap_rel_tol, &gap, &relative_gap);
                    ipt_cluster_28_29_lock_states(
                        current_residuals, locked_tolerance,
                        cluster_active, soft_cluster_locking_enabled,
                        cluster_hard_locked, pair28_lock_state,
                        sizeof(pair28_lock_state), pair29_lock_state,
                        sizeof(pair29_lock_state),
                        &cluster_soft_locked_count);
                }

                for (int index : active_pairs) {
                    std::vector<double> correction;

                    if (current_residuals[(size_t)index] <=
                        locked_tolerance) {
                        continue;
                    }
                    if (!ipt_block_build_local_jd_correction(
                            col_ptr, row_ind, matrix_values,
                            sorted_col_ptr, sorted_row_ind, sorted_values,
                            diagonal, perm, n, k, host_ritz_values,
                            host_ritz_vectors, current_residuals, index,
                            window_start, window_end, damping,
                            use_residual_support, support_top, max_dim,
                            outside_diagonal, locked_tolerance,
                            ortho_repeats, trial_orthonormal_basis,
                            &correction)) {
                        continue;
                    }
                    trial_orthonormal_basis.insert(
                        trial_orthonormal_basis.end(), correction.begin(),
                        correction.end());
                    corrections_sorted.insert(
                        corrections_sorted.end(), correction.begin(),
                        correction.end());
                    step_active_pairs.push_back(index);
                }
                if (step_active_pairs.empty()) {
                    break;
                }
                {
                    int trial_basis_cols =
                        basis_cols + (int)step_active_pairs.size();
                    const double *current_basis =
                        d_davidson_basis != NULL ? d_davidson_basis
                                                : d_combined_basis;

                    CUDA_CHECK(cudaMalloc(
                        (void **)&d_trial_basis,
                        (size_t)n * (size_t)trial_basis_cols *
                            sizeof(double)));
                    CUDA_CHECK(cudaMemcpy(
                        d_trial_basis, current_basis,
                        (size_t)n * (size_t)basis_cols * sizeof(double),
                        cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemcpy(
                        d_trial_basis + (size_t)n * (size_t)basis_cols,
                        corrections_sorted.data(),
                        corrections_sorted.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
                    status = ipt_block_cluster_rayleigh_ritz_gpu(
                        sparse_handle, sparse_matrix, cublas_handle,
                        d_trial_basis, n, trial_basis_cols, k, d_perm,
                        &d_trial_values, &d_trial_vectors, &trial_rr_time,
                        &trial_used_qr, &trial_basis_frobenius,
                        &trial_basis_max, &trial_basis_invalid,
                        &trial_ritz_frobenius, &trial_ritz_max,
                        &trial_ritz_invalid);
                    if (status != IPT_CUDA_SUCCESS) {
                        cudaFree(d_trial_basis);
                        cudaFree(d_trial_values);
                        cudaFree(d_trial_vectors);
                        goto cleanup;
                    }
                    result->rayleigh_ritz_time_sec += trial_rr_time;
                    result->matvecs += trial_basis_cols;
                    CUDA_CHECK(cudaMemcpy(
                        trial_vectors.data(), d_trial_vectors,
                        trial_vectors.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(
                        trial_values.data(), d_trial_values,
                        trial_values.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
                    ipt_block_host_residuals(
                        col_ptr, row_ind, matrix_values, n, k,
                        trial_values, trial_vectors, matrix_norm,
                        &trial_residuals, &trial_maximum,
                        &trial_max_index);
                    trial_best_updated = ipt_best_so_far_note(
                        &best_so_far, "jd_local", step, trial_values,
                        trial_vectors, trial_residuals, trial_maximum,
                        trial_max_index, trial_basis_cols, trial_used_qr,
                        trial_basis_frobenius, trial_basis_max,
                        trial_basis_invalid, trial_ritz_frobenius,
                        trial_ritz_max, trial_ritz_invalid, 0);
                    ipt_cluster_compute_28_29_metrics(
                        n, k, host_ritz_values, current_residuals,
                        trial_vectors, trial_residuals,
                        cluster_reference_vectors, cluster_active,
                        locked_tolerance, accept_rel_slack,
                        accept_abs_slack, &cluster_metrics);
                    ipt_cluster_refresh_full_residual_metrics(
                        col_ptr, row_ind, matrix_values, n, k,
                        host_ritz_vectors, trial_vectors, matrix_norm,
                        accept_rel_slack, accept_abs_slack,
                        &cluster_metrics);
                    accept_decision = ipt_block_trial_accept_decision(
                        current_residuals, trial_residuals,
                        step_active_pairs, std::vector<int>(),
                        current_maximum, trial_maximum,
                        accept_only_if_improves, relaxed_accept_enabled,
                        accept_rel_slack, accept_abs_slack,
                        active_pair_accept, locked_tolerance,
                        protect_tolerance, locked_degrade_rel_slack,
                        locked_degrade_abs_slack, trial_basis_invalid,
                        trial_ritz_invalid, &cluster_metrics, NULL,
                        cluster_aware_accept_enabled);
                    accepted = accept_decision.accepted;
                    {
                        int log_cluster_hard_locked =
                            cluster_hard_locked;
                        int log_cluster_soft_locked_count =
                            cluster_soft_locked_count;
                        char log_pair28_lock_state[16];
                        char log_pair29_lock_state[16];

                        snprintf(log_pair28_lock_state,
                                 sizeof(log_pair28_lock_state), "%s",
                                 pair28_lock_state);
                        snprintf(log_pair29_lock_state,
                                 sizeof(log_pair29_lock_state), "%s",
                                 pair29_lock_state);
                        if (accepted && cluster_active) {
                            int cluster_stable =
                                trial_residuals.size() > 29U &&
                                trial_residuals[28] <= locked_tolerance &&
                                trial_residuals[29] <= locked_tolerance &&
                                cluster_metrics
                                        .cluster_residual_fro_after <=
                                    locked_tolerance;

                            if (cluster_stable) {
                                ++cluster_stable_count;
                            } else {
                                cluster_stable_count = 0;
                            }
                            if (soft_cluster_locking_enabled &&
                                cluster_stable_count >= 2) {
                                cluster_hard_locked = 1;
                            }
                            ipt_cluster_28_29_lock_states(
                                trial_residuals, locked_tolerance,
                                cluster_active,
                                soft_cluster_locking_enabled,
                                cluster_hard_locked,
                                log_pair28_lock_state,
                                sizeof(log_pair28_lock_state),
                                log_pair29_lock_state,
                                sizeof(log_pair29_lock_state),
                                &log_cluster_soft_locked_count);
                            log_cluster_hard_locked = cluster_hard_locked;
                        }
                    for (int index : step_active_pairs) {
                        IPTJDLocalHistoryEntry entry = {};
                        int accepted_steps_after =
                            accepted_steps_count + (accepted ? 1 : 0);
                        int rejected_steps_after =
                            rejected_steps_count + (accepted ? 0 : 1);

                        entry.jd_step = step;
                        entry.active_pair_index = index;
                        entry.residual_before =
                            current_residuals[(size_t)index];
                        entry.residual_after =
                            trial_residuals[(size_t)index];
                        entry.accepted = accepted;
                        entry.basis_cols =
                            accepted ? trial_basis_cols : basis_cols;
                        entry.max_relative_eigen_residual =
                            accepted ? trial_maximum : current_maximum;
                        entry.max_relative_eigen_residual_index =
                            accepted ? trial_max_index : current_max_index;
                        entry.pair_28_residual =
                            k > 28
                                ? (accepted
                                       ? trial_residuals[28]
                                       : current_residuals[28])
                                : NAN;
                        entry.pair_29_residual =
                            k > 29
                                ? (accepted
                                       ? trial_residuals[29]
                                       : current_residuals[29])
                                : NAN;
                        entry.orthogonality_max_abs_error =
                            accepted
                                ? trial_ritz_max
                                : result
                                      ->ritz_vectors_orthogonality_max_abs_error;
                        entry.best_so_far_updated =
                            trial_best_updated;
                        entry.best_so_far_step = best_so_far.step;
                        entry.best_so_far_max_residual =
                            best_so_far.max_residual;
                        entry.pair28_best_residual =
                            best_so_far.pair28_best_residual;
                        entry.pair29_best_residual =
                            best_so_far.pair29_best_residual;
                        ipt_block_fill_jd_history_accept_fields(
                            &entry, accept_decision, 0, 1.0, damping,
                            accepted_steps_after, rejected_steps_after,
                            min_accepted_steps, 0);
                        ipt_block_fill_jd_history_cluster_fields(
                            &entry, cluster_metrics,
                            log_pair28_lock_state,
                            log_pair29_lock_state,
                            log_cluster_hard_locked,
                            log_cluster_soft_locked_count);
                        history.push_back(entry);
                    }
                    }
                    if (accepted) {
                        if (trial_best_updated) {
                            best_so_far.state_is_current = 1;
                        }
                        cudaFree(d_davidson_basis);
                        d_davidson_basis = d_trial_basis;
                        d_trial_basis = NULL;
                        cudaFree(d_ritz_values);
                        cudaFree(d_ritz_vectors);
                        d_ritz_values = d_trial_values;
                        d_ritz_vectors = d_trial_vectors;
                        d_trial_values = NULL;
                        d_trial_vectors = NULL;
                        basis_cols = trial_basis_cols;
                        orthonormal_basis.swap(
                            trial_orthonormal_basis);
                        host_ritz_values.swap(trial_values);
                        host_ritz_vectors.swap(trial_vectors);
                        current_residuals.swap(trial_residuals);
                        current_maximum = trial_maximum;
                        current_max_index = trial_max_index;
                        ipt_cluster_28_29_lock_states(
                            current_residuals, locked_tolerance,
                            cluster_active, soft_cluster_locking_enabled,
                            cluster_hard_locked, pair28_lock_state,
                            sizeof(pair28_lock_state),
                            pair29_lock_state,
                            sizeof(pair29_lock_state),
                            &cluster_soft_locked_count);
                        ipt_cluster_copy_28_29_reference(
                            n, k, host_ritz_vectors,
                            &cluster_reference_vectors);
                        ipt_block_store_result_cluster_fields(
                            result, host_ritz_values, current_residuals, k,
                            cluster_active,
                            cluster_aware_accept_enabled,
                            soft_cluster_locking_enabled, locked_tolerance,
                            cluster_hard_locked);
                        result->jd_local_accepted = 1;
                        result->rayleigh_ritz_used_qr = trial_used_qr;
                        result->basis_orthogonality_frobenius_error =
                            trial_basis_frobenius;
                        result->basis_orthogonality_max_abs_error =
                            trial_basis_max;
                        result->basis_has_nan_or_inf =
                            trial_basis_invalid;
                        result
                            ->ritz_vectors_orthogonality_frobenius_error =
                            trial_ritz_frobenius;
                        result
                            ->ritz_vectors_orthogonality_max_abs_error =
                            trial_ritz_max;
                        result->ritz_vectors_has_nan_or_inf =
                            trial_ritz_invalid;
                        ++accepted_steps_count;
                    }
                    cudaFree(d_trial_basis);
                    cudaFree(d_trial_values);
                    cudaFree(d_trial_vectors);
                    if (!accepted) {
                        ++rejected_steps_count;
                        break;
                    }
                }
            }
        } else if (ipt_preparation_debug_enabled()) {
            fprintf(stderr,
                    "IPT local JD: failed to rebuild current basis "
                    "basis_cols=%d\n",
                    basis_cols);
        }
        if (!history.empty()) {
            result->jd_local_history = (IPTJDLocalHistoryEntry *)calloc(
                history.size(), sizeof(IPTJDLocalHistoryEntry));
            if (result->jd_local_history == NULL) {
                status = IPT_CUDA_ALLOCATION_FAILED;
                goto cleanup;
            }
            memcpy(result->jd_local_history, history.data(),
                   history.size() * sizeof(IPTJDLocalHistoryEntry));
            result->jd_local_history_count = (int)history.size();
        }
        result->accepted_steps = accepted_steps_count;
        result->rejected_steps = rejected_steps_count;
        ipt_block_store_result_cluster_fields(
            result, host_ritz_values, current_residuals, k,
            cluster_active, cluster_aware_accept_enabled,
            soft_cluster_locking_enabled, locked_tolerance,
            cluster_hard_locked);
    }
    result->solve_time_sec =
        result->iteration_time_sec + result->rayleigh_ritz_time_sec;

    result->vectors =
        (double *)calloc((size_t)n * (size_t)k, sizeof(double));
    result->values = (double *)calloc((size_t)k, sizeof(double));
    if (result->vectors == NULL || result->values == NULL) {
        status = IPT_CUDA_ALLOCATION_FAILED;
        goto cleanup;
    }

    {
        int return_best =
            best_so_far.enabled && best_so_far.has_state &&
            best_so_far.values.size() >= (size_t)k &&
            best_so_far.vectors.size() >=
                (size_t)n * (size_t)k;

        if (return_best) {
            memcpy(result->vectors, best_so_far.vectors.data(),
                   (size_t)n * (size_t)k * sizeof(double));
            memcpy(result->values, best_so_far.values.data(),
                   (size_t)k * sizeof(double));
            basis_cols = best_so_far.basis_cols;
            result->rayleigh_ritz_used_qr =
                best_so_far.rayleigh_ritz_used_qr;
            result->basis_orthogonality_frobenius_error =
                best_so_far.basis_orthogonality_frobenius_error;
            result->basis_orthogonality_max_abs_error =
                best_so_far.basis_orthogonality_max_abs_error;
            result->basis_has_nan_or_inf =
                best_so_far.basis_has_nan_or_inf;
            result->ritz_vectors_orthogonality_frobenius_error =
                best_so_far.ritz_vectors_orthogonality_frobenius_error;
            result->ritz_vectors_orthogonality_max_abs_error =
                best_so_far.ritz_vectors_orthogonality_max_abs_error;
            result->ritz_vectors_has_nan_or_inf =
                best_so_far.ritz_vectors_has_nan_or_inf;
            if (result->davidson_attempted) {
                result->davidson_residual_after =
                    best_so_far.max_residual;
            }
        } else {
            CUDA_CHECK(cudaMemcpy(result->vectors, d_ritz_vectors,
                                  (size_t)n * (size_t)k * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(result->values, d_ritz_values,
                                  (size_t)k * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
        ipt_best_so_far_store_result_fields(
            result, &best_so_far,
            return_best && !best_so_far.state_is_current);
    }
    if (k > 29 && result->vectors != NULL && result->values != NULL) {
        std::vector<double> final_values(result->values,
                                         result->values + (size_t)k);
        std::vector<double> final_vectors(
            result->vectors, result->vectors + (size_t)n * (size_t)k);
        std::vector<double> final_residuals;
        std::vector<IPTActiveCluster> final_active_clusters;
        double final_maximum = NAN;
        int final_max_index = -1;
        double final_matrix_norm = ipt_block_host_matrix_inf_norm(
            col_ptr, row_ind, matrix_values, n);
        int final_cluster_active = 0;
        double final_converged_tolerance = ipt_cuda_env_double(
            "IPT_DAVIDSON_CONVERGED_TOL", 1.0e-13);
        double final_cluster_gap_abs_tol = ipt_cuda_env_double(
            "IPT_DAVIDSON_AUTO_GAP_ABS_TOL", 1.0e-12);
        double final_cluster_gap_rel_tol = ipt_cuda_env_double(
            "IPT_DAVIDSON_AUTO_GAP_REL_TOL", 5.0e-4);

        if (ipt_sort_host_ritz_pairs(
                n, k, &final_values, &final_vectors)) {
            std::copy(final_values.begin(), final_values.end(),
                      result->values);
            std::copy(final_vectors.begin(), final_vectors.end(),
                      result->vectors);
        }
        ipt_block_host_residuals(
            col_ptr, row_ind, matrix_values, n, k, final_values,
            final_vectors, final_matrix_norm, &final_residuals,
            &final_maximum, &final_max_index);
        final_active_clusters = ipt_discover_auto_ritz_clusters(
            k, final_values, final_residuals,
            final_converged_tolerance, final_cluster_gap_abs_tol,
            final_cluster_gap_rel_tol);
        final_cluster_active = ipt_active_clusters_include_exact(
            final_active_clusters, 28, 29);
        ipt_block_store_result_cluster_fields(
            result, final_values, final_residuals, k, final_cluster_active,
            result->cluster_aware_accept_enabled,
            result->soft_cluster_locking_enabled,
            final_converged_tolerance, result->cluster_hard_locked);
        if (final_cluster_active) {
            double final_cluster_fro = NAN;
            double final_cluster_max = NAN;

            ipt_block_cluster_28_29_full_residual(
                col_ptr, row_ind, matrix_values, n, k, final_vectors,
                final_matrix_norm, &final_cluster_fro,
                &final_cluster_max);
            result->cluster_residual_fro = final_cluster_fro;
            result->cluster_residual_max = final_cluster_max;
        }
    }
    result->n = n;
    result->k = k;
    result->iterations = iterations_done;
    result->basis_cols = basis_cols;

cleanup:
    cudaFree(d_col_ptr);
    cudaFree(d_row_ind);
    cudaFree(d_perm);
    cudaFree(d_values);
    cudaFree(d_diagonal);
    cudaFree(d_combined_basis);
    cudaFree(d_davidson_basis);
    cudaFree(d_ritz_values);
    cudaFree(d_ritz_vectors);
    if (sparse_matrix != NULL) {
        cusparseDestroySpMat(sparse_matrix);
    }
    if (sparse_handle != NULL) {
        cusparseDestroy(sparse_handle);
    }
    if (cublas_handle != NULL) {
        cublasDestroy(cublas_handle);
    }
    if (status != IPT_CUDA_SUCCESS) {
        ipt_cuda_free_result(result);
    }
    return status;
}

static int ipt_cuda_sparse_csc_with_initial_impl(
    const int *col_ptr, const int *row_ind, const double *matrix_values,
    const double *initial_vectors_col_major, int n, int k, int nnz,
    int maxiter, double tol, int use_tolerance, IPTCudaResult *result)
{
    int status = IPT_CUDA_SUCCESS;
    int iterations_done = 0;
    size_t vector_elements = 0;
    size_t vector_bytes = 0;
    size_t col_ptr_bytes = 0;
    size_t nnz_int_bytes = 0;
    size_t nnz_value_bytes = 0;
    cublasHandle_t handle = NULL;
    int *d_col_ptr = NULL;
    int *d_row_ind = NULL;
    double *d_matrix_values = NULL;
    IPTPreparedCsc prepared;
    double *d_x_a = NULL;
    double *d_x_b = NULL;
    double *d_diagonal = NULL;
    double *d_g = NULL;
    double *d_column_diagonal = NULL;
    double *d_delta = NULL;
    double *d_values = NULL;
    double *d_x = NULL;
    double *d_y = NULL;
    cusparseHandle_t sparse_handle = NULL;
    cusparseSpMatDescr_t sparse_matrix = NULL;
    cusparseDnMatDescr_t dense_x = NULL;
    cusparseDnMatDescr_t dense_y = NULL;
    void *d_spmm_buffer = NULL;
    size_t spmm_buffer_size = 0;
    const int block_size = 256;

    ipt_prepared_csc_init(&prepared);

    if (result != NULL) {
        result->n = 0;
        result->k = 0;
        result->iterations = 0;
        result->vectors = NULL;
        result->values = NULL;
        result->basis_cols = 0;
        result->fixed_point_residual = NAN;
    }

    if (col_ptr == NULL || row_ind == NULL || matrix_values == NULL ||
        result == NULL || n <= 0 || k <= 0 || k > n || nnz < 0 ||
        maxiter < 0 || (use_tolerance && (tol <= 0.0 || maxiter <= 0))) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    if (initial_vectors_col_major == NULL &&
        ipt_cuda_env_flag("IPT_BLOCK_CLUSTER")) {
        return ipt_cuda_sparse_csc_block_cluster_impl(
            col_ptr, row_ind, matrix_values, n, k, nnz, maxiter, tol,
            use_tolerance, result);
    }

    {
        double degeneracy_threshold =
            ipt_cuda_env_double("IPT_DEGENERACY_THRESHOLD", 0.0);
        int prepare_status =
            ipt_prepare_sparse_csc(col_ptr, row_ind, matrix_values, n, k, nnz,
                                   1, 1, degeneracy_threshold, &prepared);
        if (prepare_status == 1) {
            status = IPT_CUDA_INVALID_ARGUMENT;
            goto cleanup;
        }
        if (prepare_status == 2) {
            status = IPT_CUDA_ALLOCATION_FAILED;
            goto cleanup;
        }
    }

    vector_elements = (size_t)n * (size_t)k;

    if (vector_elements > (size_t)INT_MAX) {
        return IPT_CUDA_INVALID_ARGUMENT;
    }

    vector_bytes = vector_elements * sizeof(double);
    col_ptr_bytes = ((size_t)n + 1U) * sizeof(int);
    nnz_int_bytes = (size_t)prepared.nnz * sizeof(int);
    nnz_value_bytes = (size_t)prepared.nnz * sizeof(double);

    result->vectors = (double *)calloc(vector_elements, sizeof(double));
    result->values = (double *)calloc((size_t)k, sizeof(double));

    if (result->vectors == NULL || result->values == NULL) {
        status = IPT_CUDA_ALLOCATION_FAILED;
        goto cleanup;
    }

    CUDA_CHECK(cudaMalloc((void **)&d_col_ptr, col_ptr_bytes));
    if (prepared.nnz > 0) {
        CUDA_CHECK(cudaMalloc((void **)&d_row_ind, nnz_int_bytes));
        CUDA_CHECK(cudaMalloc((void **)&d_matrix_values, nnz_value_bytes));
    }
    CUDA_CHECK(cudaMalloc((void **)&d_x_a, vector_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_x_b, vector_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_diagonal, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_g, vector_bytes));
    CUDA_CHECK(
        cudaMalloc((void **)&d_column_diagonal, (size_t)k * sizeof(double)));
    if (use_tolerance) {
        CUDA_CHECK(cudaMalloc((void **)&d_delta, vector_bytes));
    }
    CUDA_CHECK(cudaMalloc((void **)&d_values, (size_t)k * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_col_ptr, prepared.col_ptr, col_ptr_bytes,
                          cudaMemcpyHostToDevice));
    if (prepared.nnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_row_ind, prepared.row_ind, nnz_int_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_matrix_values, prepared.values,
                              nnz_value_bytes,
                              cudaMemcpyHostToDevice));
    }

    if (initial_vectors_col_major == NULL) {
        int vector_blocks = ((int)vector_elements + block_size - 1) / block_size;
        set_identity_kernel<<<vector_blocks, block_size>>>(d_x_a, n, k);
        CUDA_CHECK(cudaGetLastError());
    } else {
        CUDA_CHECK(cudaMemcpy(d_x_a, initial_vectors_col_major, vector_bytes,
                              cudaMemcpyHostToDevice));
    }

    {
        int vector_blocks = ((int)vector_elements + block_size - 1) / block_size;

        CUDA_CHECK(cudaMemcpy(d_diagonal, prepared.diagonal,
                              (size_t)n * sizeof(double),
                              cudaMemcpyHostToDevice));
        build_g_kernel<<<vector_blocks, block_size>>>(d_diagonal, d_g, n, k,
                                                      0.0, NULL);
        CUDA_CHECK(cudaGetLastError());
    }

    CUBLAS_CHECK(cublasCreate(&handle));
    CUSPARSE_CHECK(cusparseCreate(&sparse_handle));
    CUSPARSE_CHECK(cusparseCreateCsr(
        &sparse_matrix, n, n, prepared.nnz, (void *)d_col_ptr,
        (void *)d_row_ind, (void *)d_matrix_values, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_x, n, k, n, d_x_a, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_y, n, k, n, d_x_b, CUDA_R_64F,
                                       CUSPARSE_ORDER_COL));
    {
        const double one = 1.0;
        const double zero = 0.0;

        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            sparse_handle, CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &one, sparse_matrix, dense_x,
            &zero, dense_y, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
            &spmm_buffer_size));
        if (spmm_buffer_size > 0) {
            CUDA_CHECK(cudaMalloc(&d_spmm_buffer, spmm_buffer_size));
        }
    }

    d_x = d_x_a;
    d_y = d_x_b;

    for (int iter = 0; iter < maxiter; ++iter) {
        int col_blocks = (k + block_size - 1) / block_size;
        int vector_blocks = ((int)vector_elements + block_size - 1) / block_size;
        double *tmp = NULL;

        CUSPARSE_CHECK(ipt_cusparse_spmm(sparse_handle, sparse_matrix,
                                         dense_x, dense_y, d_x, d_y,
                                         d_spmm_buffer));

        column_diagonal_after_d_kernel<<<col_blocks, block_size>>>(
            d_y, d_x, d_diagonal, d_column_diagonal, n, k);
        CUDA_CHECK(cudaGetLastError());

        ipt_update_kernel<<<vector_blocks, block_size>>>(
            d_y, d_x, d_diagonal, d_g, d_column_diagonal, n, k);
        CUDA_CHECK(cudaGetLastError());

        iterations_done = iter + 1;

        if (use_tolerance) {
            double delta_norm = 0.0;
            double current_norm = 0.0;
            double fixed_point_residual = 0.0;

            fixed_point_delta_kernel<<<vector_blocks, block_size>>>(
                d_delta, d_y, d_x, (int)vector_elements);
            CUDA_CHECK(cudaGetLastError());

            CUBLAS_CHECK(cublasDnrm2(handle, (int)vector_elements, d_delta, 1,
                                     &delta_norm));
            CUBLAS_CHECK(cublasDnrm2(handle, (int)vector_elements, d_x, 1,
                                     &current_norm));

            if (current_norm <= 0.0 || !isfinite(current_norm)) {
                status = IPT_CUDA_INVALID_ARGUMENT;
                goto cleanup;
            }

            fixed_point_residual = delta_norm / current_norm;
            if (fixed_point_residual <= tol) {
                tmp = d_x;
                d_x = d_y;
                d_y = tmp;
                break;
            }
        }

        tmp = d_x;
        d_x = d_y;
        d_y = tmp;
    }

    CUSPARSE_CHECK(ipt_cusparse_spmm(sparse_handle, sparse_matrix, dense_x,
                                     dense_y, d_x, d_y, d_spmm_buffer));

    {
        int col_blocks = (k + block_size - 1) / block_size;

        gather_values_kernel<<<col_blocks, block_size>>>(d_y, d_values, n, k);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(result->vectors, d_x, vector_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result->values, d_values, (size_t)k * sizeof(double),
                          cudaMemcpyDeviceToHost));
    if (!ipt_apply_prepared_q_to_vectors(&prepared, result->vectors, k)) {
        status = IPT_CUDA_INVALID_ARGUMENT;
        goto cleanup;
    }

    result->n = n;
    result->k = k;
    result->iterations = iterations_done;
    result->basis_cols = k;

cleanup:
    if (handle != NULL) {
        cublasDestroy(handle);
    }

    cudaFree(d_col_ptr);
    cudaFree(d_row_ind);
    cudaFree(d_matrix_values);
    cudaFree(d_x_a);
    cudaFree(d_x_b);
    cudaFree(d_diagonal);
    cudaFree(d_g);
    cudaFree(d_column_diagonal);
    cudaFree(d_delta);
    cudaFree(d_values);
    cudaFree(d_spmm_buffer);
    if (dense_x != NULL) {
        cusparseDestroyDnMat(dense_x);
    }
    if (dense_y != NULL) {
        cusparseDestroyDnMat(dense_y);
    }
    if (sparse_matrix != NULL) {
        cusparseDestroySpMat(sparse_matrix);
    }
    if (sparse_handle != NULL) {
        cusparseDestroy(sparse_handle);
    }
    ipt_prepared_csc_free(&prepared);

    if (status != IPT_CUDA_SUCCESS) {
        ipt_cuda_free_result(result);
    }

    return status;
}

extern "C" int ipt_cuda_sparse_csc_with_initial(
    const int *col_ptr, const int *row_ind, const double *matrix_values,
    const double *initial_vectors_col_major, int n, int k, int nnz,
    int iterations, IPTCudaResult *result)
{
    return ipt_cuda_sparse_csc_with_initial_impl(
        col_ptr, row_ind, matrix_values, initial_vectors_col_major, n, k, nnz,
        iterations, 0.0, 0, result);
}

extern "C" int ipt_cuda_sparse_csc_with_initial_tol(
    const int *col_ptr, const int *row_ind, const double *matrix_values,
    const double *initial_vectors_col_major, int n, int k, int nnz, double tol,
    int maxiter, IPTCudaResult *result)
{
    return ipt_cuda_sparse_csc_with_initial_impl(
        col_ptr, row_ind, matrix_values, initial_vectors_col_major, n, k, nnz,
        maxiter, tol, 1, result);
}

#undef CUDA_CHECK
#undef CUBLAS_CHECK
#undef CUSPARSE_CHECK
#undef CUSOLVER_CHECK
