#define main ch4_ipt_benchmark_unused_main
#include "benchmark_ch4_sto6g_fci_15876_ipt.cu"
#undef main

static double eigen_residual_norm(const CscMatrixView *matrix,
                                  const double *vector,
                                  double eigenvalue)
{
    std::vector<double> av((size_t)matrix->n, 0.0);
    double residual_norm_sq = 0.0;

    csc_matvec_block(matrix, vector, matrix->n, av.data(), matrix->n, 1);
    for (int row = 0; row < matrix->n; ++row) {
        double residual = av[(size_t)row] - eigenvalue * vector[row];

        residual_norm_sq += residual * residual;
    }
    return sqrt(residual_norm_sq);
}

static double vector_norm2(int n, const double *vector)
{
    double norm_sq = 0.0;

    for (int row = 0; row < n; ++row) {
        norm_sq += vector[row] * vector[row];
    }
    return sqrt(norm_sq);
}

static void primme_relative_eigen_residual_convtest(double *eval, void *evec,
                                                    double *rnorm,
                                                    int *isconv,
                                                    primme_params *primme,
                                                    int *ierr)
{
    (void)eval;
    (void)evec;

    *isconv = fabs(*rnorm) <= primme->eps * primme->aNorm;
    *ierr = 0;
}

static primme_preset_method primme_method_from_name(const char *name)
{
    if (name == NULL || name[0] == '\0' ||
        strcmp(name, "PRIMME_JDQMR_ETol") == 0 ||
        strcmp(name, "JDQMR_ETol") == 0) {
        return PRIMME_JDQMR_ETol;
    }
    if (strcmp(name, "PRIMME_DYNAMIC") == 0 || strcmp(name, "DYNAMIC") == 0) {
        return PRIMME_DYNAMIC;
    }
    if (strcmp(name, "PRIMME_JDQMR") == 0 || strcmp(name, "JDQMR") == 0) {
        return PRIMME_JDQMR;
    }
    if (strcmp(name, "PRIMME_JDQR") == 0 || strcmp(name, "JDQR") == 0) {
        return PRIMME_JDQR;
    }
    if (strcmp(name, "PRIMME_GD_plusK") == 0 || strcmp(name, "GD_plusK") == 0) {
        return PRIMME_GD_plusK;
    }
    if (strcmp(name, "PRIMME_DEFAULT_MIN_MATVECS") == 0 ||
        strcmp(name, "DEFAULT_MIN_MATVECS") == 0) {
        return PRIMME_DEFAULT_MIN_MATVECS;
    }
    return PRIMME_JDQMR_ETol;
}

static const char *primme_output_dir(void)
{
    const char *path = getenv("PRIMME_CH4_OUTPUT_DIR");

    return path == NULL || path[0] == '\0'
               ? "/fs1/home/nudt_liujie/ftt/IPT_C_GPU/test/sparse/CH4/"
                 "primme_k30_results"
               : path;
}

int main(void)
{
    const char *default_cache =
        "/fs1/home/nudt_liujie/ftt/IPT_C_GPU/results/sparse/"
        "CH4_K30_debug/ch4_sto6g_fci_15876_csc.bin";
    const char *cache_path = getenv("IPT_CH4_MATRIX_CACHE");
    const char *output_dir = primme_output_dir();
    int k = env_int("PRIMME_K", 30);
    int repeats = env_int("PRIMME_REPEATS", 1);
    double relative_tolerance =
        env_double("PRIMME_RELATIVE_EIGEN_RESIDUAL_TOL",
                   env_double("PRIMME_TOL", 1.0e-12));
    double required_relative_residual =
        env_double("PRIMME_REQUIRED_RELATIVE_EIGEN_RESIDUAL", 1.0e-12);
    long long max_matvecs = env_ll("PRIMME_MAX_MATVECS", 200000);
    long long max_outer_iterations =
        env_ll("PRIMME_MAX_OUTER_ITERATIONS", 0);
    int max_basis_size = env_int("PRIMME_MAX_BASIS_SIZE", 160);
    int min_restart_size = env_int("PRIMME_MIN_RESTART_SIZE", 80);
    int max_block_size = env_int("PRIMME_MAX_BLOCK_SIZE", 8);
    int locking = env_int("PRIMME_LOCKING", 1);
    int print_level = env_int("PRIMME_PRINT_LEVEL", 0);
    const char *method_name = getenv("PRIMME_METHOD");
    primme_preset_method method = primme_method_from_name(method_name);
    ActiveSpaceData active = {};
    CscMatrix owned_matrix = {};
    std::vector<double> cached_diagonal;
    double matrix_norm = NAN;
    CscMatrixView matrix = {};
    PrimmeGpuMatrix gpu_matrix = {};
    cublasHandle_t cublas_handle = NULL;
    int *d_col_ptr = NULL;
    int *d_row_ind = NULL;
    double *d_values = NULL;
    double *d_evec = NULL;
    double *d_warm_x = NULL;
    double *d_warm_y = NULL;
    FILE *pairs_csv = NULL;
    FILE *timing_csv = NULL;
    FILE *summary = NULL;
    char pairs_path[4096];
    char timing_path[4096];
    char summary_path[4096];
    std::vector<double> solve_times;
    int process_status = 0;

    if (cache_path == NULL || cache_path[0] == '\0') {
        cache_path = default_cache;
    }
    if (!read_ch4_matrix_cache(cache_path, &owned_matrix, &matrix_norm,
                               &cached_diagonal, &active)) {
        fprintf(stderr, "failed to load fixed CH4 matrix cache: %s\n",
                cache_path);
        return 1;
    }
    matrix = matrix_view(owned_matrix);
    if (k <= 0 || k > matrix.n) {
        fprintf(stderr, "invalid PRIMME_K=%d for n=%d\n", k, matrix.n);
        return 2;
    }
    ensure_directory(output_dir);
    snprintf(pairs_path, sizeof(pairs_path),
             "%s/ch4_sto6g_fci_15876_primme_k30_pairs.csv", output_dir);
    snprintf(timing_path, sizeof(timing_path),
             "%s/ch4_sto6g_fci_15876_primme_k30_timing.csv", output_dir);
    snprintf(summary_path, sizeof(summary_path),
             "%s/ch4_sto6g_fci_15876_primme_k30_summary.txt", output_dir);
    pairs_csv = fopen(pairs_path, "w");
    timing_csv = fopen(timing_path, "w");
    summary = fopen(summary_path, "w");
    if (pairs_csv == NULL || timing_csv == NULL || summary == NULL) {
        fprintf(stderr, "failed to open PRIMME outputs under %s\n",
                output_dir);
        process_status = 2;
        goto cleanup;
    }
    fprintf(pairs_csv,
            "repeat,pair_index,lambda,relative_eigen_residual,vector_norm\n");
    fprintf(timing_csv,
            "repeat,status,returned_k,solve_time_sec,outer_iterations,"
            "matvecs,max_relative_eigen_residual,"
            "max_relative_eigen_residual_index,"
            "passed_relative_threshold\n");

    if (cudaMalloc((void **)&d_col_ptr,
                   (size_t)(matrix.n + 1) * sizeof(int)) != cudaSuccess ||
        cudaMalloc((void **)&d_row_ind,
                   (size_t)matrix.nnz * sizeof(int)) != cudaSuccess ||
        cudaMalloc((void **)&d_values,
                   (size_t)matrix.nnz * sizeof(double)) != cudaSuccess ||
        cudaMalloc((void **)&d_evec,
                   (size_t)matrix.n * (size_t)k * sizeof(double)) !=
            cudaSuccess ||
        cudaMalloc((void **)&d_warm_x,
                   (size_t)matrix.n * (size_t)k * sizeof(double)) !=
            cudaSuccess ||
        cudaMalloc((void **)&d_warm_y,
                   (size_t)matrix.n * (size_t)k * sizeof(double)) !=
            cudaSuccess) {
        fprintf(stderr, "PRIMME GPU allocation failed\n");
        process_status = 3;
        goto cleanup;
    }
    if (cudaMemcpy(d_col_ptr, matrix.col_ptr,
                   (size_t)(matrix.n + 1) * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_row_ind, matrix.row_ind,
                   (size_t)matrix.nnz * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_values, matrix.values,
                   (size_t)matrix.nnz * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(d_warm_x, 0,
                   (size_t)matrix.n * (size_t)k * sizeof(double)) !=
            cudaSuccess ||
        cudaMemset(d_warm_y, 0,
                   (size_t)matrix.n * (size_t)k * sizeof(double)) !=
            cudaSuccess) {
        fprintf(stderr, "PRIMME matrix transfer failed\n");
        process_status = 3;
        goto cleanup;
    }
    if (cusparseCreate(&gpu_matrix.sparse_handle) !=
            CUSPARSE_STATUS_SUCCESS ||
        cusparseCreateCsr(
            &gpu_matrix.matrix, matrix.n, matrix.n, matrix.nnz, d_col_ptr,
            d_row_ind, d_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) !=
            CUSPARSE_STATUS_SUCCESS ||
        cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "PRIMME GPU handle/descriptor setup failed\n");
        process_status = 4;
        goto cleanup;
    }

    /* Prime the maximum-width SpMM buffer and CUDA libraries outside the
       measured eigensolve interval. */
    {
        primme_params warmup_primme;
        PRIMME_INT leading_dimension = matrix.n;
        int block_size = k;
        int ierr = 0;
        double dummy_norm = 0.0;

        primme_initialize(&warmup_primme);
        warmup_primme.n = matrix.n;
        warmup_primme.nLocal = matrix.n;
        warmup_primme.matrix = &gpu_matrix;
        primme_gpu_matvec(d_warm_x, &leading_dimension, d_warm_y,
                          &leading_dimension, &block_size, &warmup_primme,
                          &ierr);
        cublasDnrm2(cublas_handle, matrix.n, d_warm_y, 1, &dummy_norm);
        cudaDeviceSynchronize();
        primme_free(&warmup_primme);
        if (ierr != 0) {
            fprintf(stderr, "PRIMME warmup SpMM failed\n");
            process_status = 4;
            goto cleanup;
        }
    }

    fprintf(summary,
            "matrix_source=cache\ncache_path=%s\nn=%d\nnnz=%d\n"
            "matrix_inf_norm=%.17g\nrequested_k=%d\nrepeats=%d\n"
            "convergence_metric=max_relative_eigen_residual\n"
            "convergence_threshold=%.17g\n"
            "primme_relative_eigen_residual_tol=%.17g\n"
            "primme_convergence_test=relative_eigen_residual\n"
            "relative_eigen_residual_definition="
            "||A v - lambda v||_2/(||A||_inf*||v||_2)\n"
            "primme_method=%s\n"
            "primme_max_matvecs=%lld\n"
            "primme_max_outer_iterations=%lld\n"
            "primme_max_basis_size=%d\n"
            "primme_min_restart_size=%d\n"
            "primme_max_block_size=%d\n"
            "primme_locking=%d\n"
            "timing_metric=solve_time_sec\n"
            "timing_scope=cublas_dprimme_plus_final_cuda_synchronize\n"
            "timing_excludes=cache_read,matrix_h2d,gpu_setup,warmup_spmm,"
            "eigenvector_d2h,residual_recomputation\n",
            cache_path, matrix.n, matrix.nnz, matrix_norm, k, repeats,
            required_relative_residual, relative_tolerance,
            method_name == NULL || method_name[0] == '\0'
                ? "PRIMME_JDQMR_ETol"
                : method_name,
            max_matvecs, max_outer_iterations, max_basis_size,
            min_restart_size, max_block_size, locking);

    for (int repeat = 1; repeat <= repeats; ++repeat) {
        primme_params primme;
        std::vector<double> eigenvalues((size_t)k, 0.0);
        std::vector<double> reported_rnorms((size_t)k, 0.0);
        std::vector<double> eigenvectors(
            (size_t)matrix.n * (size_t)k, 0.0);
        std::vector<double> relative_residuals((size_t)k, NAN);
        double solve_time = NAN;
        double max_relative_residual = NAN;
        int max_relative_index = -1;
        int ret = 0;
        int returned_k = 0;
        int passed_relative_threshold = 0;

        primme_initialize(&primme);
        primme.n = matrix.n;
        primme.numEvals = k;
        primme.matrix = &gpu_matrix;
        primme.matrixMatvec = primme_gpu_matvec;
        primme.target = primme_smallest;
        primme.outputFile = stdout;
        primme.queue = &cublas_handle;
        primme_set_method(method, &primme);
        primme.eps = relative_tolerance;
        primme.aNorm = matrix_norm;
        primme.maxMatvecs = max_matvecs;
        if (max_outer_iterations > 0) {
            primme.maxOuterIterations = max_outer_iterations;
        }
        if (max_basis_size > 0) {
            primme.maxBasisSize = max_basis_size;
        }
        if (min_restart_size > 0) {
            primme.minRestartSize = min_restart_size;
        }
        if (max_block_size > 0) {
            primme.maxBlockSize = max_block_size;
        }
        primme.locking = locking;
        primme.printLevel = print_level;
        primme.convTestFun = primme_relative_eigen_residual_convtest;
        primme.convTestFun_type = primme_op_double;
        cudaMemset(d_evec, 0,
                   (size_t)matrix.n * (size_t)k * sizeof(double));
        cudaDeviceSynchronize();
        {
            double start = now_seconds();

            ret = cublas_dprimme(eigenvalues.data(), d_evec,
                                 reported_rnorms.data(), &primme);
            cudaDeviceSynchronize();
            solve_time = now_seconds() - start;
        }
        returned_k =
            ret == 0 ? k : std::max(0, std::min(k, primme.initSize));
        solve_times.push_back(solve_time);
        if (cudaMemcpy(eigenvectors.data(), d_evec,
                       eigenvectors.size() * sizeof(double),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "PRIMME eigenvector D2H failed\n");
            primme_free(&primme);
            process_status = 5;
            goto cleanup;
        }
        for (int pair = 0; pair < returned_k; ++pair) {
            const double *vector =
                eigenvectors.data() + (size_t)pair * (size_t)matrix.n;
            double vector_norm = vector_norm2(matrix.n, vector);
            double denominator = matrix_norm * vector_norm;
            double residual_norm =
                eigen_residual_norm(&matrix, vector,
                                    eigenvalues[(size_t)pair]);

            relative_residuals[(size_t)pair] =
                denominator > 0.0
                    ? residual_norm / denominator
                    : NAN;
            if (!isfinite(max_relative_residual) ||
                relative_residuals[(size_t)pair] >
                    max_relative_residual) {
                max_relative_residual =
                    relative_residuals[(size_t)pair];
                max_relative_index = pair;
            }
            fprintf(pairs_csv, "%d,%d,%.17g,%.17g,%.17g\n", repeat,
                    pair, eigenvalues[(size_t)pair],
                    relative_residuals[(size_t)pair],
                    vector_norm);
        }
        passed_relative_threshold =
            returned_k == k && isfinite(max_relative_residual) &&
            max_relative_residual <= required_relative_residual;
        fprintf(timing_csv,
                "%d,%d,%d,%.17g,%lld,%lld,%.17g,%d,%d\n",
                repeat, ret, returned_k, solve_time,
                (long long)primme.stats.numOuterIterations,
                (long long)primme.stats.numMatvecs, max_relative_residual,
                max_relative_index, passed_relative_threshold);
        fprintf(summary,
                "repeat=%d status=%d returned_k=%d solve_time_sec=%.17g "
                "outer_iterations=%lld matvecs=%lld "
                "max_relative_eigen_residual=%.17g "
                "max_relative_eigen_residual_index=%d "
                "passed_relative_threshold=%d\n",
                repeat, ret, returned_k, solve_time,
                (long long)primme.stats.numOuterIterations,
                (long long)primme.stats.numMatvecs, max_relative_residual,
                max_relative_index, passed_relative_threshold);
        printf("PRIMME repeat=%d status=%d returned_k=%d solve=%.6f sec "
               "max_relative_eigen_residual=%.3e index=%d\n",
               repeat, ret, returned_k, solve_time,
               max_relative_residual, max_relative_index);
        fflush(stdout);
        fflush(pairs_csv);
        fflush(timing_csv);
        fflush(summary);
        if (ret != 0) {
            process_status = 6;
        } else if (!passed_relative_threshold && process_status == 0) {
            process_status = 7;
        }
        primme_free(&primme);
    }
    if (!solve_times.empty()) {
        double mean =
            std::accumulate(solve_times.begin(), solve_times.end(), 0.0) /
            (double)solve_times.size();
        double variance = 0.0;

        for (double value : solve_times) {
            double delta = value - mean;

            variance += delta * delta;
        }
        variance /= (double)solve_times.size();
        fprintf(summary,
                "mean_solve_time_sec=%.17g\n"
                "stddev_solve_time_sec=%.17g\n",
                mean, sqrt(variance));
    }

cleanup:
    if (pairs_csv != NULL) {
        fclose(pairs_csv);
    }
    if (timing_csv != NULL) {
        fclose(timing_csv);
    }
    if (summary != NULL) {
        fclose(summary);
    }
    cudaFree(d_col_ptr);
    cudaFree(d_row_ind);
    cudaFree(d_values);
    cudaFree(d_evec);
    cudaFree(d_warm_x);
    cudaFree(d_warm_y);
    cudaFree(gpu_matrix.buffer);
    if (gpu_matrix.matrix != NULL) {
        cusparseDestroySpMat(gpu_matrix.matrix);
    }
    if (gpu_matrix.sparse_handle != NULL) {
        cusparseDestroy(gpu_matrix.sparse_handle);
    }
    if (cublas_handle != NULL) {
        cublasDestroy(cublas_handle);
    }
    if (process_status == 0) {
        printf("wrote %s\nwrote %s\nwrote %s\n", pairs_path, timing_path,
               summary_path);
    }
    return process_status;
}
