#include <Python.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "/fs1/home/nudt_liujie/ftt/primme-3.2.3/include/primme.h"

#include "../../../src/ipt_cuda.cu"

#define DEFAULT_RESULTS_DIR "/fs1/home/nudt_liujie/ftt/IPT_C_GPU/results/sparse"
#define DEFAULT_PYSCF_SITE_PACKAGES                                               \
    "/fs1/software/hpcsystem/THL7/software/pyscf/2.0.1-py3.8/lib/python3.8/site-packages"
#define DEFAULT_REPEATS 5
#define DEFAULT_TOL 1.0e-12
#define DEFAULT_IPT_MAXITER 1000
#define DEFAULT_IPT_K 10
#define DEFAULT_PRIMME_MAX_MATVECS 1000

typedef struct {
    int n;
    int nnz;
    const int *col_ptr;
    const int *row_ind;
    const double *values;
} CscMatrixView;

typedef struct {
    int n;
    int nnz;
    std::vector<int> col_ptr;
    std::vector<int> row_ind;
    std::vector<double> values;
} CscMatrix;

typedef struct {
    int norb;
    int nelec;
    int na;
    int nb;
    double nuclear_repulsion;
    double hartree_fock_energy;
    double generation_time_sec;
} H2OGenerationInfo;

typedef struct {
    const char *method;
    int repeat;
    double time_sec;
    int iterations;
    long long matvecs;
    double eigenvalue;
    double residual;
    int status;
    char error[128];
} TrialResult;

static const char *H2O_FCI_CSC_PYTHON = R"PY(
def build_h2o_fci_csc():
    import numpy as np
    from functools import reduce
    from numpy import dot
    from pyscf import ao2mo, fci, gto, scf

    coordinates = """
O 0 0 0;
H 0.2774 0.8929 0.2544;
H 0.6068, -0.2383, -0.7169
"""
    basis = "sto-6g"

    mol = gto.M(atom=coordinates, basis=basis, symmetry=True, verbose=0)
    hf = scf.HF(mol)
    hf.verbose = 0
    hf.kernel()
    if not hf.converged:
        raise RuntimeError("PySCF HF did not converge for H2O/sto-6g")

    nelec = mol.nelectron
    norb = hf.mo_coeff.shape[0]
    neleca, nelecb = fci.addons._unpack_nelec(nelec)
    na = fci.cistring.num_strings(norb, neleca)
    nb = fci.cistring.num_strings(norb, nelecb)
    n = int(na * nb)

    h1e = reduce(dot, (hf.mo_coeff.T, hf.get_hcore(), hf.mo_coeff))
    eri = ao2mo.incore.general(hf._eri, (hf.mo_coeff,) * 4, compact=False)
    h2e = fci.direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5)
    e_nuc = float(hf.energy_nuc())

    dense = np.empty((n, n), dtype=np.float64)
    for col in range(n):
        c = np.zeros(n, dtype=np.float64)
        c[col] = 1.0
        hc = fci.direct_spin1.contract_2e(
            h2e, c.reshape(na, nb), norb, nelec
        ).ravel()
        dense[:, col] = hc + e_nuc * c

    perm = np.argsort(np.real(np.diag(dense)), kind="mergesort")
    dense = dense[np.ix_(perm, perm)]

    col_ptr = [0]
    row_ind = []
    values = []
    for col in range(n):
        column = dense[:, col]
        nz_rows = np.nonzero(column != 0.0)[0]
        row_ind.extend(int(row) for row in nz_rows)
        values.extend(float(column[row]) for row in nz_rows)
        col_ptr.append(len(row_ind))

    return (
        n,
        len(row_ind),
        col_ptr,
        row_ind,
        values,
        e_nuc,
        int(norb),
        int(nelec),
        int(na),
        int(nb),
        float(hf.e_tot),
    )

)PY";

static double now_seconds(void)
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch())
        .count();
}

static const char *results_dir(void)
{
    const char *dir = getenv("IPT_C_RESULTS_DIR");

    if (dir == NULL || dir[0] == '\0') {
        dir = getenv("IPT_SPARSE_RESULTS_DIR");
    }

    return (dir == NULL || dir[0] == '\0') ? DEFAULT_RESULTS_DIR : dir;
}

static int env_int(const char *name, int default_value)
{
    const char *raw = getenv(name);

    if (raw == NULL || raw[0] == '\0') {
        return default_value;
    }

    int value = atoi(raw);
    return value > 0 ? value : default_value;
}

static double env_double(const char *name, double default_value)
{
    const char *raw = getenv(name);

    if (raw == NULL || raw[0] == '\0') {
        return default_value;
    }

    double value = atof(raw);
    return value > 0.0 ? value : default_value;
}

static long long env_ll(const char *name, long long default_value)
{
    const char *raw = getenv(name);

    if (raw == NULL || raw[0] == '\0') {
        return default_value;
    }

    long long value = atoll(raw);
    return value > 0 ? value : default_value;
}

static const char *pyscf_site_packages(void)
{
    const char *site = getenv("PYSCF_SITE_PACKAGES");

    return (site == NULL || site[0] == '\0') ? DEFAULT_PYSCF_SITE_PACKAGES
                                             : site;
}

static void prepend_env_path(const char *name, const char *path)
{
    const char *old_value = getenv(name);
    std::string new_value;

    if (path == NULL || path[0] == '\0') {
        return;
    }

    if (old_value == NULL || old_value[0] == '\0') {
        new_value = path;
    } else {
        new_value = std::string(path) + ":" + old_value;
    }

    setenv(name, new_value.c_str(), 1);
}

static int ensure_directory(const char *path)
{
    char command[4096];

    snprintf(command, sizeof(command), "mkdir -p '%s'", path);
    return system(command);
}

static CscMatrixView matrix_view(const CscMatrix &matrix)
{
    CscMatrixView view = {matrix.n, matrix.nnz, matrix.col_ptr.data(),
                          matrix.row_ind.data(), matrix.values.data()};
    return view;
}

static bool py_long_to_int(PyObject *object, int *value, const char *name)
{
    long parsed = PyLong_AsLong(object);

    if (PyErr_Occurred() != NULL || parsed < 0 ||
        parsed > (long)std::numeric_limits<int>::max()) {
        fprintf(stderr, "could not parse Python integer field %s\n", name);
        PyErr_Clear();
        return false;
    }

    *value = (int)parsed;
    return true;
}

static bool py_float_to_double(PyObject *object, double *value,
                               const char *name)
{
    double parsed = PyFloat_AsDouble(object);

    if (PyErr_Occurred() != NULL || !std::isfinite(parsed)) {
        fprintf(stderr, "could not parse Python float field %s\n", name);
        PyErr_Clear();
        return false;
    }

    *value = parsed;
    return true;
}

static bool py_sequence_to_int_vector(PyObject *object,
                                      std::vector<int> *values,
                                      const char *name)
{
    PyObject *sequence = PySequence_Fast(object, name);

    if (sequence == NULL) {
        PyErr_Print();
        return false;
    }

    Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence);
    values->resize((size_t)size);

    for (Py_ssize_t i = 0; i < size; ++i) {
        int parsed = 0;
        if (!py_long_to_int(PySequence_Fast_GET_ITEM(sequence, i), &parsed,
                            name)) {
            Py_DECREF(sequence);
            return false;
        }
        (*values)[(size_t)i] = parsed;
    }

    Py_DECREF(sequence);
    return true;
}

static bool py_sequence_to_double_vector(PyObject *object,
                                         std::vector<double> *values,
                                         const char *name)
{
    PyObject *sequence = PySequence_Fast(object, name);

    if (sequence == NULL) {
        PyErr_Print();
        return false;
    }

    Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence);
    values->resize((size_t)size);

    for (Py_ssize_t i = 0; i < size; ++i) {
        double parsed = 0.0;
        if (!py_float_to_double(PySequence_Fast_GET_ITEM(sequence, i),
                                &parsed, name)) {
            Py_DECREF(sequence);
            return false;
        }
        (*values)[(size_t)i] = parsed;
    }

    Py_DECREF(sequence);
    return true;
}

static bool parse_h2o_tuple(PyObject *tuple, CscMatrix *matrix,
                            H2OGenerationInfo *info)
{
    if (!PyTuple_Check(tuple) || PyTuple_Size(tuple) != 11) {
        fprintf(stderr, "PySCF generator returned an unexpected object\n");
        return false;
    }

    if (!py_long_to_int(PyTuple_GET_ITEM(tuple, 0), &matrix->n, "n") ||
        !py_long_to_int(PyTuple_GET_ITEM(tuple, 1), &matrix->nnz, "nnz") ||
        !py_sequence_to_int_vector(PyTuple_GET_ITEM(tuple, 2),
                                   &matrix->col_ptr, "col_ptr") ||
        !py_sequence_to_int_vector(PyTuple_GET_ITEM(tuple, 3),
                                   &matrix->row_ind, "row_ind") ||
        !py_sequence_to_double_vector(PyTuple_GET_ITEM(tuple, 4),
                                      &matrix->values, "values") ||
        !py_float_to_double(PyTuple_GET_ITEM(tuple, 5),
                            &info->nuclear_repulsion,
                            "nuclear_repulsion") ||
        !py_long_to_int(PyTuple_GET_ITEM(tuple, 6), &info->norb, "norb") ||
        !py_long_to_int(PyTuple_GET_ITEM(tuple, 7), &info->nelec, "nelec") ||
        !py_long_to_int(PyTuple_GET_ITEM(tuple, 8), &info->na, "na") ||
        !py_long_to_int(PyTuple_GET_ITEM(tuple, 9), &info->nb, "nb") ||
        !py_float_to_double(PyTuple_GET_ITEM(tuple, 10),
                            &info->hartree_fock_energy,
                            "hartree_fock_energy")) {
        return false;
    }

    if (matrix->n != 441) {
        fprintf(stderr, "expected H2O/sto-6g FCI n=441, got n=%d\n",
                matrix->n);
        return false;
    }

    if (matrix->nnz < 0 || matrix->col_ptr.size() != (size_t)matrix->n + 1 ||
        matrix->row_ind.size() != (size_t)matrix->nnz ||
        matrix->values.size() != (size_t)matrix->nnz ||
        matrix->col_ptr.front() != 0 ||
        matrix->col_ptr.back() != matrix->nnz) {
        fprintf(stderr, "invalid CSC structure from PySCF generator\n");
        return false;
    }

    return true;
}

static bool generate_h2o_fci_csc(CscMatrix *matrix, H2OGenerationInfo *info)
{
    const char *site = pyscf_site_packages();
    double start = now_seconds();

    prepend_env_path("PYTHONPATH", site);

    printf("Generating H2O/sto-6g FCI Hamiltonian with PySCF...\n");
    printf("PYSCF_SITE_PACKAGES=%s\n", site);
    fflush(stdout);

    Py_Initialize();
    if (!Py_IsInitialized()) {
        fprintf(stderr, "failed to initialize embedded Python\n");
        return false;
    }

    PyObject *main_module = PyImport_AddModule("__main__");
    PyObject *globals = PyModule_GetDict(main_module);
    PyObject *module_result =
        PyRun_String(H2O_FCI_CSC_PYTHON, Py_file_input, globals, globals);

    if (module_result == NULL) {
        PyErr_Print();
        Py_Finalize();
        return false;
    }
    Py_DECREF(module_result);

    PyObject *result =
        PyRun_String("build_h2o_fci_csc()", Py_eval_input, globals, globals);

    if (result == NULL) {
        PyErr_Print();
        Py_Finalize();
        return false;
    }

    bool ok = parse_h2o_tuple(result, matrix, info);
    Py_DECREF(result);
    Py_Finalize();

    info->generation_time_sec = now_seconds() - start;
    return ok;
}

static void csc_matvec_block(const CscMatrixView *matrix, const double *x,
                             int ldx, double *y, int ldy, int block_size)
{
    for (int block = 0; block < block_size; ++block) {
        const double *xb = x + (size_t)block * (size_t)ldx;
        double *yb = y + (size_t)block * (size_t)ldy;

        std::fill(yb, yb + matrix->n, 0.0);

        for (int col = 0; col < matrix->n; ++col) {
            double x_value = xb[col];

            if (x_value == 0.0) {
                continue;
            }

            for (int p = matrix->col_ptr[col]; p < matrix->col_ptr[col + 1];
                 ++p) {
                yb[matrix->row_ind[p]] += matrix->values[p] * x_value;
            }
        }
    }
}

static void primme_matvec(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy,
                          int *block_size, primme_params *primme, int *ierr)
{
    const CscMatrixView *matrix = (const CscMatrixView *)primme->matrix;

    csc_matvec_block(matrix, (const double *)x, (int)*ldx, (double *)y,
                     (int)*ldy, *block_size);
    *ierr = 0;
}

static double inf_norm(const CscMatrixView *matrix)
{
    std::vector<double> row_sum((size_t)matrix->n, 0.0);
    double norm = 0.0;

    for (int col = 0; col < matrix->n; ++col) {
        for (int p = matrix->col_ptr[col]; p < matrix->col_ptr[col + 1]; ++p) {
            row_sum[(size_t)matrix->row_ind[p]] += fabs(matrix->values[p]);
        }
    }

    for (double value : row_sum) {
        norm = std::max(norm, value);
    }

    return norm;
}

static double relative_residual(const CscMatrixView *matrix,
                                const double *vector, double eigenvalue,
                                double matrix_norm)
{
    std::vector<double> av((size_t)matrix->n, 0.0);
    double residual_norm_sq = 0.0;
    double vector_norm_sq = 0.0;

    csc_matvec_block(matrix, vector, matrix->n, av.data(), matrix->n, 1);

    for (int row = 0; row < matrix->n; ++row) {
        double residual = av[(size_t)row] - eigenvalue * vector[row];

        residual_norm_sq += residual * residual;
        vector_norm_sq += vector[row] * vector[row];
    }

    {
        double denom = std::max(1.0, matrix_norm) *
                       std::max(1.0, sqrt(vector_norm_sq));
        return sqrt(residual_norm_sq) / denom;
    }
}

static TrialResult run_primme_once(const CscMatrixView *matrix, double tol,
                                   long long max_matvecs, double matrix_norm,
                                   int k, int repeat)
{
    TrialResult result = {"PRIMME_DYNAMIC", repeat, NAN, 0, 0, NAN, NAN, 0,
                          ""};
    primme_params primme;
    std::vector<double> eval((size_t)k, 0.0);
    std::vector<double> rnorm((size_t)k, 0.0);
    std::vector<double> evec((size_t)matrix->n * (size_t)k, 0.0);
    int ret = 0;

    primme_initialize(&primme);
    primme.n = matrix->n;
    primme.numEvals = k;
    primme.matrix = (void *)matrix;
    primme.matrixMatvec = primme_matvec;
    primme.target = primme_smallest;
    primme.eps = tol;
    primme.aNorm = matrix_norm;
    primme.maxMatvecs = max_matvecs;
    primme.printLevel = 0;
    primme.outputFile = stdout;
    primme_set_method(PRIMME_DYNAMIC, &primme);

    {
        double start = now_seconds();
        ret = dprimme(eval.data(), evec.data(), rnorm.data(), &primme);
        result.time_sec = now_seconds() - start;
    }

    result.iterations = (int)primme.stats.numOuterIterations;
    result.matvecs = (long long)primme.stats.numMatvecs;
    result.eigenvalue = eval[0];

    if (ret != 0) {
        result.status = ret;
        snprintf(result.error, sizeof(result.error), "dprimme returned %d",
                 ret);
    } else {
        double max_residual = 0.0;
        for (int col = 0; col < k; ++col) {
            const double *vector =
                evec.data() + (size_t)col * (size_t)matrix->n;
            double residual =
                relative_residual(matrix, vector, eval[col], matrix_norm);
            max_residual = std::max(max_residual, residual);
        }
        result.residual = max_residual;
    }

    primme_free(&primme);
    return result;
}

static TrialResult run_ipt_once(const CscMatrixView *matrix, double tol,
                                int maxiter, double matrix_norm, int k,
                                int repeat)
{
    TrialResult result = {"IPT_CUDA_SPARSE", repeat, NAN, 0, 0, NAN, NAN, 0,
                          ""};
    IPTCudaResult ipt = {0, 0, 0, NULL, NULL};
    int status = IPT_CUDA_SUCCESS;

    {
        double start = now_seconds();
        status = ipt_cuda_sparse_csc_tol(matrix->col_ptr, matrix->row_ind,
                                         matrix->values, matrix->n, k,
                                         matrix->nnz, tol, maxiter, &ipt);
        cudaDeviceSynchronize();
        (void)(now_seconds() - start);
    }

    result.status = status;
    if (status != IPT_CUDA_SUCCESS) {
        snprintf(result.error, sizeof(result.error), "%s",
                 ipt_cuda_status_string(status));
    } else {
        double max_residual = 0.0;
        int result_k = ipt.k > 0 ? ipt.k : k;

        result.iterations = ipt.iterations;
        result.matvecs = ipt.matvecs;
        result.time_sec = ipt.solve_time_sec;
        result.eigenvalue = ipt.values[0];
        for (int col = 0; col < result_k; ++col) {
            const double *vector =
                ipt.vectors + (size_t)col * (size_t)matrix->n;
            double residual =
                relative_residual(matrix, vector, ipt.values[col], matrix_norm);
            max_residual = std::max(max_residual, residual);
        }
        result.residual = max_residual;
    }

    ipt_cuda_free_result(&ipt);
    return result;
}

static double average(const std::vector<TrialResult> &results,
                      double TrialResult::*field)
{
    double sum = 0.0;

    for (const TrialResult &result : results) {
        sum += result.*field;
    }

    return sum / (double)results.size();
}

static double stddev(const std::vector<TrialResult> &results,
                     double TrialResult::*field, double avg)
{
    double sum = 0.0;

    for (const TrialResult &result : results) {
        double delta = result.*field - avg;
        sum += delta * delta;
    }

    return sqrt(sum / (double)results.size());
}

static double average_ll(const std::vector<TrialResult> &results,
                         long long TrialResult::*field)
{
    double sum = 0.0;

    for (const TrialResult &result : results) {
        sum += (double)(result.*field);
    }

    return sum / (double)results.size();
}

static double average_int(const std::vector<TrialResult> &results,
                          int TrialResult::*field)
{
    double sum = 0.0;

    for (const TrialResult &result : results) {
        sum += (double)(result.*field);
    }

    return sum / (double)results.size();
}

static void write_trial(FILE *csv, const TrialResult &result, double tol,
                        int maxiter, int k)
{
    fprintf(csv,
            "%s,%d,%d,%.17g,%d,%lld,%.17g,%.17g,%d,%s,%.17g,%d\n",
            result.method, result.repeat, k, result.time_sec,
            result.iterations, result.matvecs, result.eigenvalue,
            result.residual, result.status, result.error, tol, maxiter);
    fflush(csv);
}

static void write_summary(FILE *summary, const char *method,
                          const std::vector<TrialResult> &results)
{
    double avg_time = average(results, &TrialResult::time_sec);
    double std_time = stddev(results, &TrialResult::time_sec, avg_time);

    fprintf(summary,
            "%s avg_time_sec=%.17g std_time_sec=%.17g avg_iterations=%.17g "
            "avg_matvecs=%.17g avg_eigenvalue=%.17g avg_residual=%.17g\n",
            method, avg_time, std_time,
            average_int(results, &TrialResult::iterations),
            average_ll(results, &TrialResult::matvecs),
            average(results, &TrialResult::eigenvalue),
            average(results, &TrialResult::residual));
}

int main(void)
{
    CscMatrix owned_matrix = {0, 0, std::vector<int>(), std::vector<int>(),
                              std::vector<double>()};
    H2OGenerationInfo generation = {0, 0, 0, 0, NAN, NAN, NAN};
    int repeats = env_int("IPT_REPEATS", DEFAULT_REPEATS);
    double tol = env_double("IPT_TOL", DEFAULT_TOL);
    int ipt_maxiter = env_int("IPT_MAXITER", DEFAULT_IPT_MAXITER);
    int ipt_k = env_int("IPT_K", DEFAULT_IPT_K);
    long long primme_max_matvecs =
        env_ll("PRIMME_MAX_MATVECS", DEFAULT_PRIMME_MAX_MATVECS);
    const char *dir = results_dir();
    char csv_path[4096];
    char summary_path[4096];
    FILE *csv = NULL;
    FILE *summary = NULL;
    std::vector<TrialResult> primme_results;
    std::vector<TrialResult> ipt_results;

    if (repeats <= 0) {
        repeats = DEFAULT_REPEATS;
    }

    if (!generate_h2o_fci_csc(&owned_matrix, &generation)) {
        fprintf(stderr, "failed to generate H2O/sto-6g FCI CSC matrix\n");
        return 1;
    }

    CscMatrixView matrix = matrix_view(owned_matrix);
    double matrix_norm = inf_norm(&matrix);

    ensure_directory(dir);
    snprintf(csv_path, sizeof(csv_path),
             "%s/h2o_fci_pyscf_441_primme_vs_ipt_trials.csv", dir);
    snprintf(summary_path, sizeof(summary_path),
             "%s/h2o_fci_pyscf_441_primme_vs_ipt_summary.txt", dir);

    csv = fopen(csv_path, "w");
    summary = fopen(summary_path, "w");
    if (csv == NULL || summary == NULL) {
        fprintf(stderr, "could not open result files in %s\n", dir);
        if (csv != NULL) {
            fclose(csv);
        }
        if (summary != NULL) {
            fclose(summary);
        }
        return 2;
    }

    fprintf(csv,
            "method,repeat,k,time_sec,iterations,matvecs,first_eigenvalue,"
            "relative_residual,status,error,tol,maxiter\n");

    printf("H2O/sto-6g FCI PySCF CSC benchmark: n=%d nnz=%d k=%d "
           "repeats=%d tol=%.3e\n",
           matrix.n, matrix.nnz, ipt_k, repeats, tol);
    printf("Matrix generation time %.6f sec is excluded from PRIMME/IPT "
           "timings.\n",
           generation.generation_time_sec);
    printf("Warm up once for each method...\n");

    {
        TrialResult warm_primme =
            run_primme_once(&matrix, tol, primme_max_matvecs, matrix_norm,
                            ipt_k, 0);
        TrialResult warm_ipt =
            run_ipt_once(&matrix, tol, ipt_maxiter, matrix_norm, ipt_k, 0);

        fprintf(summary,
                "warmup PRIMME status=%d time_sec=%.17g eigenvalue=%.17g "
                "residual=%.17g error=%s\n",
                warm_primme.status, warm_primme.time_sec,
                warm_primme.eigenvalue, warm_primme.residual,
                warm_primme.error);
        fprintf(summary,
                "warmup IPT status=%d time_sec=%.17g eigenvalue=%.17g "
                "residual=%.17g error=%s\n\n",
                warm_ipt.status, warm_ipt.time_sec, warm_ipt.eigenvalue,
                warm_ipt.residual, warm_ipt.error);
    }

    primme_results.reserve((size_t)repeats);
    ipt_results.reserve((size_t)repeats);

    for (int repeat = 1; repeat <= repeats; ++repeat) {
        TrialResult primme_result =
            run_primme_once(&matrix, tol, primme_max_matvecs, matrix_norm,
                            ipt_k, repeat);
        TrialResult ipt_result =
            run_ipt_once(&matrix, tol, ipt_maxiter, matrix_norm, ipt_k,
                         repeat);

        primme_results.push_back(primme_result);
        ipt_results.push_back(ipt_result);
        write_trial(csv, primme_result, tol, (int)primme_max_matvecs, ipt_k);
        write_trial(csv, ipt_result, tol, ipt_maxiter, ipt_k);

        printf("repeat %d PRIMME time=%.6e residual=%.3e; IPT time=%.6e "
               "residual=%.3e\n",
               repeat, primme_result.time_sec, primme_result.residual,
               ipt_result.time_sec, ipt_result.residual);
    }

    fprintf(summary,
            "matrix=h2o_sto6g_fci_generated_by_embedded_pyscf\nn=%d\n"
            "nnz=%d\nk=%d\nrepeats=%d\ntol=%.17g\nipt_maxiter=%d\n"
            "primme_max_matvecs=%lld\nmatrix_inf_norm=%.17g\n"
            "matrix_generation_time_sec=%.17g\n"
            "matrix_generation_excluded_from_timings=true\n"
            "pyscf_site_packages=%s\nnorb=%d\nnelec=%d\nna=%d\nnb=%d\n"
            "nuclear_repulsion=%.17g\nhartree_fock_energy=%.17g\n\n",
            matrix.n, matrix.nnz, ipt_k, repeats, tol, ipt_maxiter,
            primme_max_matvecs, matrix_norm, generation.generation_time_sec,
            pyscf_site_packages(), generation.norb, generation.nelec,
            generation.na, generation.nb, generation.nuclear_repulsion,
            generation.hartree_fock_energy);
    write_summary(summary, "PRIMME_DYNAMIC", primme_results);
    write_summary(summary, "IPT_CUDA_SPARSE", ipt_results);

    fclose(csv);
    fclose(summary);

    printf("wrote %s\n", csv_path);
    printf("wrote %s\n", summary_path);
    return 0;
}
