#define main ch4_ipt_benchmark_unused_main
#include "benchmark_ch4_sto6g_fci_15876_ipt.cu"
#undef main

static double absolute_eigen_residual(const CscMatrixView *matrix,
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
    double tolerance = env_double("PRIMME_TOL", 1.0e-12);
    long long max_matvecs = env_ll("PRIMME_MAX_MATVECS", 50000);
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
            "repeat,pair_index,lambda,absolute_eigen_residual,"
            "primme_reported_rnorm\n");
    fprintf(timing_csv,
            "repeat,status,returned_k,solve_time_sec,outer_iterations,"
            "matvecs,max_absolute_eigen_residual,"
            "max_absolute_eigen_residual_index\n");

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
            "primme_tolerance=%.17g\nprimme_max_matvecs=%lld\n"
            "timing_metric=solve_time_sec\n"
            "timing_scope=cublas_dprimme_plus_final_cuda_synchronize\n"
            "timing_excludes=cache_read,matrix_h2d,gpu_setup,warmup_spmm,"
            "eigenvector_d2h,residual_recomputation\n",
            cache_path, matrix.n, matrix.nnz, matrix_norm, k, repeats,
            tolerance, max_matvecs);

    for (int repeat = 1; repeat <= repeats; ++repeat) {
        primme_params primme;
        std::vector<double> eigenvalues((size_t)k, 0.0);
        std::vector<double> reported_rnorms((size_t)k, 0.0);
        std::vector<double> eigenvectors(
            (size_t)matrix.n * (size_t)k, 0.0);
        std::vector<double> absolute_residuals((size_t)k, NAN);
        double solve_time = NAN;
        double max_absolute_residual = NAN;
        int max_absolute_index = -1;
        int ret = 0;
        int returned_k = 0;

        primme_initialize(&primme);
        primme.n = matrix.n;
        primme.numEvals = k;
        primme.matrix = &gpu_matrix;
        primme.matrixMatvec = primme_gpu_matvec;
        primme.target = primme_smallest;
        primme.eps = tolerance;
        primme.aNorm = matrix_norm;
        primme.maxMatvecs = max_matvecs;
        primme.printLevel = 0;
        primme.outputFile = stdout;
        primme.queue = &cublas_handle;
        primme_set_method(PRIMME_DYNAMIC, &primme);
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
            absolute_residuals[(size_t)pair] = absolute_eigen_residual(
                &matrix,
                eigenvectors.data() + (size_t)pair * (size_t)matrix.n,
                eigenvalues[(size_t)pair]);
            if (!isfinite(max_absolute_residual) ||
                absolute_residuals[(size_t)pair] >
                    max_absolute_residual) {
                max_absolute_residual =
                    absolute_residuals[(size_t)pair];
                max_absolute_index = pair;
            }
            fprintf(pairs_csv, "%d,%d,%.17g,%.17g,%.17g\n", repeat,
                    pair, eigenvalues[(size_t)pair],
                    absolute_residuals[(size_t)pair],
                    reported_rnorms[(size_t)pair]);
        }
        fprintf(timing_csv, "%d,%d,%d,%.17g,%lld,%lld,%.17g,%d\n",
                repeat, ret, returned_k, solve_time,
                (long long)primme.stats.numOuterIterations,
                (long long)primme.stats.numMatvecs,
                max_absolute_residual, max_absolute_index);
        fprintf(summary,
                "repeat=%d status=%d returned_k=%d solve_time_sec=%.17g "
                "outer_iterations=%lld matvecs=%lld "
                "max_absolute_eigen_residual=%.17g "
                "max_absolute_eigen_residual_index=%d\n",
                repeat, ret, returned_k, solve_time,
                (long long)primme.stats.numOuterIterations,
                (long long)primme.stats.numMatvecs,
                max_absolute_residual, max_absolute_index);
        printf("PRIMME repeat=%d status=%d returned_k=%d solve=%.6f sec "
               "max_abs_residual=%.3e index=%d\n",
               repeat, ret, returned_k, solve_time,
               max_absolute_residual, max_absolute_index);
        fflush(stdout);
        fflush(pairs_csv);
        fflush(timing_csv);
        fflush(summary);
        if (ret != 0) {
            process_status = 6;
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
