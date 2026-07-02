#include <Python.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "/fs1/home/nudt_liujie/ftt/primme-3.2.3/include/primme.h"

#include "../../../src/ipt_cuda.cu"

#define BENCH_CUDA_CHECK(call)                                                \
    do {                                                                       \
        cudaError_t check_status = (call);                                     \
        if (check_status != cudaSuccess) {                                     \
            fprintf(stderr, "benchmark CUDA error at %s:%d: %s\n", __FILE__, \
                    __LINE__, cudaGetErrorString(check_status));                \
            status = IPT_CUDA_CUDA_ERROR;                                      \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

#define BENCH_CUBLAS_CHECK(call)                                               \
    do {                                                                       \
        cublasStatus_t check_status = (call);                                  \
        if (check_status != CUBLAS_STATUS_SUCCESS) {                           \
            fprintf(stderr, "benchmark cuBLAS error at %s:%d: %d\n",         \
                    __FILE__, __LINE__, (int)check_status);                     \
            status = IPT_CUDA_CUBLAS_ERROR;                                    \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

#define BENCH_CUSPARSE_CHECK(call)                                             \
    do {                                                                       \
        cusparseStatus_t check_status = (call);                                \
        if (check_status != CUSPARSE_STATUS_SUCCESS) {                         \
            fprintf(stderr, "benchmark cuSPARSE error at %s:%d: %d\n",       \
                    __FILE__, __LINE__, (int)check_status);                     \
            status = IPT_CUDA_CUSPARSE_ERROR;                                  \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

#define DEFAULT_RESULTS_DIR "/fs1/home/nudt_liujie/ftt/IPT_C_GPU/results/sparse"
#define DEFAULT_PYSCF_SITE_PACKAGES                                               \
    "/fs1/software/hpcsystem/THL7/software/pyscf/2.0.1-py3.8/lib/python3.8/site-packages"
#define DEFAULT_REPEATS 5
#define DEFAULT_TOL 1.0e-12
#define DEFAULT_IPT_MAXITER 1000
#define DEFAULT_IPT_K 30
#define DEFAULT_PRIMME_MAX_MATVECS 5000
#define CONVERGENCE_THRESHOLD 1.0e-12

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
    int active_electrons;
    int neleca;
    int nelecb;
    int na;
    int nb;
    double ecore;
    double nuclear_repulsion;
    double hartree_fock_energy;
    double generation_time_sec;
    std::vector<double> h1;
    std::vector<double> eri;
} ActiveSpaceData;

typedef struct {
    int p;
    int q;
    int target;
    int sign;
} OneLink;

typedef struct {
    int p;
    int q;
    int r;
    int s;
    int target;
    int sign;
} TwoLink;

typedef struct {
    const char *method;
    int repeat;
    int requested_k;
    int basis_cols;
    int returned_k;
    int oversample;
    int rayleigh_ritz_used_qr;
    double time_sec;
    double api_total_time_sec;
    double preparation_time_sec;
    double transfer_setup_time_sec;
    double iteration_time_sec;
    double rayleigh_ritz_time_sec;
    int iterations;
    long long matvecs;
    double max_relative_eigen_residual;
    int max_relative_eigen_residual_index;
    double relative_fixed_point_residual;
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
    std::string adaptive_added_indices;
    std::vector<IPTDavidsonHistoryEntry> davidson_history;
    std::vector<IPTDavidsonSelectionEntry> davidson_selection_history;
    std::vector<IPTDavidsonBlockHistoryEntry> davidson_block_history;
    int davidson_restart_count;
    int jd_local_attempted;
    int jd_local_accepted;
    std::vector<IPTJDLocalHistoryEntry> jd_local_history;
    std::vector<double> eigenvalues;
    std::vector<double> eigenvectors;
    std::vector<double> relative_eigen_residuals;
    int status;
    char error[128];
} TrialResult;

typedef struct {
    char magic[8];
    uint32_t version;
    int32_t n;
    int32_t nnz;
    double matrix_inf_norm;
    int32_t norb;
    int32_t nelec;
    int32_t neleca;
    int32_t nelecb;
    int32_t na;
    int32_t nb;
} Ch4MatrixCacheHeader;

typedef struct {
    std::vector<double> original_diagonal;
    std::vector<int> perm;
    std::vector<int> inverse_perm;
    std::vector<double> sorted_diagonal;
    std::vector<int> sorted_col_ptr;
    std::vector<int> sorted_row_ind;
    std::vector<double> sorted_values;
    std::vector<std::pair<int, int>> blocks;
    int oversample;
    int basis_target_k;
    int basis_cols;
    int max_block_size;
    int too_large;
    int offending_block_size;
    int target_block_id;
    int adaptive_enabled;
    double adaptive_coupling_tau;
    int adaptive_limit_hit;
    std::vector<int> adaptive_added_indices;
} Ch4Diagnostics;

static const char *CH4_STO6G_FCI_PYTHON = R"PY(
def build_ch4_sto6g_full_fci_integrals():
    import numpy as np
    from functools import reduce
    from numpy import dot
    from pyscf import ao2mo, fci, gto, scf

    coordinates = """
C 0.000000 0.000000 0.000000;
H 0.628736 0.628736 0.628736;
H -0.628736 -0.628736 0.628736;
H -0.628736 0.628736 -0.628736;
H 0.628736 -0.628736 -0.628736
"""

    mol = gto.M(atom=coordinates, basis="sto-6g", symmetry=False, verbose=0)
    hf = scf.RHF(mol)
    hf.verbose = 0
    hf.kernel()
    if not hf.converged:
        raise RuntimeError("PySCF RHF did not converge for CH4/STO-6G full FCI")

    nelec = mol.nelectron
    norb = hf.mo_coeff.shape[1]
    neleca, nelecb = fci.addons._unpack_nelec(nelec)
    na = fci.cistring.num_strings(norb, neleca)
    nb = fci.cistring.num_strings(norb, nelecb)

    h1eff = reduce(dot, (hf.mo_coeff.T, hf.get_hcore(), hf.mo_coeff))
    eri = ao2mo.incore.general(hf._eri, (hf.mo_coeff,) * 4, compact=False)
    ecore = float(hf.energy_nuc())

    return (
        int(norb),
        int(nelec),
        int(neleca),
        int(nelecb),
        int(na),
        int(nb),
        float(ecore),
        float(hf.energy_nuc()),
        float(hf.e_tot),
        np.asarray(h1eff, dtype=np.float64).reshape(-1).tolist(),
        np.asarray(eri, dtype=np.float64).reshape(-1).tolist(),
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

static const char *pyscf_site_packages(void)
{
    const char *site = getenv("PYSCF_SITE_PACKAGES");

    return (site == NULL || site[0] == '\0') ? DEFAULT_PYSCF_SITE_PACKAGES
                                             : site;
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

static long long env_ll(const char *name, long long default_value)
{
    const char *raw = getenv(name);

    if (raw == NULL || raw[0] == '\0') {
        return default_value;
    }

    long long value = atoll(raw);
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

static int env_bool(const char *name, int default_value)
{
    const char *raw = getenv(name);

    if (raw == NULL || raw[0] == '\0') {
        return default_value;
    }

    return atoi(raw) != 0;
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

static bool parse_integral_tuple(PyObject *tuple, ActiveSpaceData *data)
{
    if (!PyTuple_Check(tuple) || PyTuple_Size(tuple) != 11) {
        fprintf(stderr, "PySCF generator returned an unexpected object\n");
        return false;
    }

    if (!py_long_to_int(PyTuple_GET_ITEM(tuple, 0), &data->norb, "norb") ||
        !py_long_to_int(PyTuple_GET_ITEM(tuple, 1),
                        &data->active_electrons, "active_electrons") ||
        !py_long_to_int(PyTuple_GET_ITEM(tuple, 2), &data->neleca,
                        "neleca") ||
        !py_long_to_int(PyTuple_GET_ITEM(tuple, 3), &data->nelecb,
                        "nelecb") ||
        !py_long_to_int(PyTuple_GET_ITEM(tuple, 4), &data->na, "na") ||
        !py_long_to_int(PyTuple_GET_ITEM(tuple, 5), &data->nb, "nb") ||
        !py_float_to_double(PyTuple_GET_ITEM(tuple, 6), &data->ecore,
                            "ecore") ||
        !py_float_to_double(PyTuple_GET_ITEM(tuple, 7),
                            &data->nuclear_repulsion,
                            "nuclear_repulsion") ||
        !py_float_to_double(PyTuple_GET_ITEM(tuple, 8),
                            &data->hartree_fock_energy,
                            "hartree_fock_energy") ||
        !py_sequence_to_double_vector(PyTuple_GET_ITEM(tuple, 9), &data->h1,
                                      "h1") ||
        !py_sequence_to_double_vector(PyTuple_GET_ITEM(tuple, 10),
                                      &data->eri, "eri")) {
        return false;
    }

    if (data->norb != 9 || data->active_electrons != 10 ||
        data->neleca != 5 || data->nelecb != 5 || data->na != 126 ||
        data->nb != 126 || data->h1.size() != 81U ||
        data->eri.size() != 6561U) {
        fprintf(stderr,
                "unexpected CH4/STO-6G full FCI active space: norb=%d "
                "nelec=%d (%d,%d) na=%d nb=%d h1=%zu eri=%zu\n",
                data->norb, data->active_electrons, data->neleca,
                data->nelecb, data->na, data->nb, data->h1.size(),
                data->eri.size());
        return false;
    }

    return true;
}

static bool generate_active_space_integrals(ActiveSpaceData *data)
{
    const char *site = pyscf_site_packages();
    double start = now_seconds();

    prepend_env_path("PYTHONPATH", site);

    printf("Generating CH4/STO-6G full FCI integrals with "
           "PySCF...\n");
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
        PyRun_String(CH4_STO6G_FCI_PYTHON, Py_file_input, globals, globals);

    if (module_result == NULL) {
        PyErr_Print();
        Py_Finalize();
        return false;
    }
    Py_DECREF(module_result);

    PyObject *result = PyRun_String(
        "build_ch4_sto6g_full_fci_integrals()", Py_eval_input, globals,
        globals);

    if (result == NULL) {
        PyErr_Print();
        Py_Finalize();
        return false;
    }

    bool ok = parse_integral_tuple(result, data);
    Py_DECREF(result);
    Py_Finalize();

    data->generation_time_sec = now_seconds() - start;
    return ok;
}

static inline int eri_index(int p, int q, int r, int s, int norb)
{
    return (((p * norb + q) * norb + r) * norb + s);
}

static inline int h1_index(int p, int q, int norb)
{
    return p * norb + q;
}

static int popcount_u32(unsigned int value)
{
    return __builtin_popcount(value);
}

static bool apply_annihilate(unsigned int bits, int orbital,
                             unsigned int *out_bits, int *sign)
{
    unsigned int mask = 1U << orbital;

    if ((bits & mask) == 0U) {
        return false;
    }

    if ((popcount_u32(bits & (mask - 1U)) & 1) != 0) {
        *sign = -*sign;
    }
    *out_bits = bits ^ mask;
    return true;
}

static bool apply_create(unsigned int bits, int orbital, unsigned int *out_bits,
                         int *sign)
{
    unsigned int mask = 1U << orbital;

    if ((bits & mask) != 0U) {
        return false;
    }

    if ((popcount_u32(bits & (mask - 1U)) & 1) != 0) {
        *sign = -*sign;
    }
    *out_bits = bits | mask;
    return true;
}

static bool apply_one_body(unsigned int bits, int p, int q,
                           unsigned int *out_bits, int *sign)
{
    unsigned int work = bits;

    *sign = 1;
    if (!apply_annihilate(work, q, &work, sign)) {
        return false;
    }
    if (!apply_create(work, p, &work, sign)) {
        return false;
    }

    *out_bits = work;
    return true;
}

static bool apply_two_body(unsigned int bits, int p, int q, int r, int s,
                           unsigned int *out_bits, int *sign)
{
    unsigned int work = bits;

    *sign = 1;
    if (!apply_annihilate(work, q, &work, sign)) {
        return false;
    }
    if (!apply_annihilate(work, s, &work, sign)) {
        return false;
    }
    if (!apply_create(work, r, &work, sign)) {
        return false;
    }
    if (!apply_create(work, p, &work, sign)) {
        return false;
    }

    *out_bits = work;
    return true;
}

static void generate_strings_rec(int norb, int nelec, int start, int left,
                                 unsigned int bits,
                                 std::vector<unsigned int> *strings)
{
    if (left == 0) {
        strings->push_back(bits);
        return;
    }

    for (int orb = start; orb <= norb - left; ++orb) {
        generate_strings_rec(norb, nelec, orb + 1, left - 1,
                             bits | (1U << orb), strings);
    }
}

static std::vector<unsigned int> generate_strings(int norb, int nelec)
{
    std::vector<unsigned int> strings;

    generate_strings_rec(norb, nelec, 0, nelec, 0U, &strings);
    return strings;
}

static void build_address_table(const std::vector<unsigned int> &strings,
                                int norb, std::vector<int> *address)
{
    address->assign((size_t)1U << (size_t)norb, -1);

    for (size_t i = 0; i < strings.size(); ++i) {
        (*address)[(size_t)strings[i]] = (int)i;
    }
}

static void build_links(const std::vector<unsigned int> &strings,
                        const std::vector<int> &address, int norb,
                        std::vector<std::vector<OneLink>> *one_links,
                        std::vector<std::vector<TwoLink>> *two_links)
{
    one_links->assign(strings.size(), std::vector<OneLink>());
    two_links->assign(strings.size(), std::vector<TwoLink>());

    for (size_t idx = 0; idx < strings.size(); ++idx) {
        unsigned int bits = strings[idx];
        std::vector<int> occ;

        for (int orb = 0; orb < norb; ++orb) {
            if (((bits >> orb) & 1U) != 0U) {
                occ.push_back(orb);
            }
        }

        (*one_links)[idx].reserve((size_t)occ.size() *
                                  (size_t)(norb - (int)occ.size() + 1));
        for (int q : occ) {
            for (int p = 0; p < norb; ++p) {
                unsigned int target_bits = 0U;
                int sign = 1;
                if (apply_one_body(bits, p, q, &target_bits, &sign)) {
                    int target = address[(size_t)target_bits];
                    if (target >= 0) {
                        OneLink link = {p, q, target, sign};
                        (*one_links)[idx].push_back(link);
                    }
                }
            }
        }

        (*two_links)[idx].reserve((size_t)occ.size() * (size_t)occ.size() *
                                  (size_t)norb * (size_t)norb / 4U);
        for (int q : occ) {
            for (int s : occ) {
                for (int r = 0; r < norb; ++r) {
                    for (int p = 0; p < norb; ++p) {
                        unsigned int target_bits = 0U;
                        int sign = 1;
                        if (apply_two_body(bits, p, q, r, s, &target_bits,
                                           &sign)) {
                            int target = address[(size_t)target_bits];
                            if (target >= 0) {
                                TwoLink link = {p, q, r, s, target, sign};
                                (*two_links)[idx].push_back(link);
                            }
                        }
                    }
                }
            }
        }
    }
}

static double determinant_diagonal(const ActiveSpaceData &data,
                                   unsigned int alpha_bits,
                                   unsigned int beta_bits)
{
    int norb = data.norb;
    double value = data.ecore;
    std::vector<int> occ_alpha;
    std::vector<int> occ_beta;

    for (int orb = 0; orb < norb; ++orb) {
        if (((alpha_bits >> orb) & 1U) != 0U) {
            occ_alpha.push_back(orb);
            value += data.h1[(size_t)h1_index(orb, orb, norb)];
        }
        if (((beta_bits >> orb) & 1U) != 0U) {
            occ_beta.push_back(orb);
            value += data.h1[(size_t)h1_index(orb, orb, norb)];
        }
    }

    for (const std::vector<int> *occ : {&occ_alpha, &occ_beta}) {
        for (int i : *occ) {
            for (int j : *occ) {
                value += 0.5 *
                         (data.eri[(size_t)eri_index(i, i, j, j, norb)] -
                          data.eri[(size_t)eri_index(i, j, j, i, norb)]);
            }
        }
    }

    for (int i : occ_alpha) {
        for (int j : occ_beta) {
            value += data.eri[(size_t)eri_index(i, i, j, j, norb)];
        }
    }

    return value;
}

static void add_entry(std::vector<std::pair<int, double>> *entries,
                      const std::vector<int> &inverse_perm, int old_row,
                      double value)
{
    if (value != 0.0) {
        entries->push_back(
            std::make_pair(inverse_perm[(size_t)old_row], value));
    }
}

static bool generate_hamiltonian_csc(const ActiveSpaceData &data,
                                     CscMatrix *matrix)
{
    int norb = data.norb;
    int na = data.na;
    int nb = data.nb;
    int n = na * nb;
    std::vector<unsigned int> alpha_strings =
        generate_strings(norb, data.neleca);
    std::vector<unsigned int> beta_strings =
        generate_strings(norb, data.nelecb);
    std::vector<int> alpha_address;
    std::vector<int> beta_address;
    std::vector<std::vector<OneLink>> alpha_one;
    std::vector<std::vector<OneLink>> beta_one;
    std::vector<std::vector<TwoLink>> alpha_two;
    std::vector<std::vector<TwoLink>> beta_two;
    std::vector<double> diagonal((size_t)n, 0.0);
    std::vector<int> perm((size_t)n, 0);
    std::vector<int> inverse_perm((size_t)n, 0);
    double start = now_seconds();
    const char *reserve_raw = getenv("IPT_EST_NNZ_PER_COL");
    long long reserve_per_col =
        reserve_raw == NULL || reserve_raw[0] == '\0' ? 1500LL
                                                      : atoll(reserve_raw);

    if ((int)alpha_strings.size() != na || (int)beta_strings.size() != nb) {
        fprintf(stderr, "CI string count mismatch\n");
        return false;
    }

    build_address_table(alpha_strings, norb, &alpha_address);
    build_address_table(beta_strings, norb, &beta_address);
    build_links(alpha_strings, alpha_address, norb, &alpha_one, &alpha_two);
    build_links(beta_strings, beta_address, norb, &beta_one, &beta_two);

    printf("Generated CI strings: na=%d nb=%d n=%d\n", na, nb, n);
    printf("Link counts per string: one=%zu two=%zu\n", alpha_one[0].size(),
           alpha_two[0].size());
    fflush(stdout);

    for (int ia = 0; ia < na; ++ia) {
        for (int ib = 0; ib < nb; ++ib) {
            int idx = ia * nb + ib;
            diagonal[(size_t)idx] = determinant_diagonal(
                data, alpha_strings[(size_t)ia], beta_strings[(size_t)ib]);
            perm[(size_t)idx] = idx;
        }
    }

    std::stable_sort(perm.begin(), perm.end(),
                     [&diagonal](int a, int b) {
                         return diagonal[(size_t)a] < diagonal[(size_t)b];
                     });
    for (int sorted = 0; sorted < n; ++sorted) {
        inverse_perm[(size_t)perm[(size_t)sorted]] = sorted;
    }

    matrix->n = n;
    matrix->nnz = 0;
    matrix->col_ptr.assign((size_t)n + 1U, 0);
    matrix->row_ind.clear();
    matrix->values.clear();
    if (reserve_per_col > 0) {
        size_t reserve_nnz =
            (size_t)std::min<long long>((long long)n * reserve_per_col,
                                        (long long)std::numeric_limits<int>::max());
        matrix->row_ind.reserve(reserve_nnz);
        matrix->values.reserve(reserve_nnz);
    }

    for (int sorted_col = 0; sorted_col < n; ++sorted_col) {
        int old_col = perm[(size_t)sorted_col];
        int ia = old_col / nb;
        int ib = old_col - ia * nb;
        std::vector<std::pair<int, double>> entries;

        entries.reserve(2048U);
        matrix->col_ptr[(size_t)sorted_col] = (int)matrix->row_ind.size();

        add_entry(&entries, inverse_perm, old_col, data.ecore);

        for (const OneLink &link : alpha_one[(size_t)ia]) {
            int old_row = link.target * nb + ib;
            double value =
                data.h1[(size_t)h1_index(link.p, link.q, norb)] *
                (double)link.sign;
            add_entry(&entries, inverse_perm, old_row, value);
        }
        for (const OneLink &link : beta_one[(size_t)ib]) {
            int old_row = ia * nb + link.target;
            double value =
                data.h1[(size_t)h1_index(link.p, link.q, norb)] *
                (double)link.sign;
            add_entry(&entries, inverse_perm, old_row, value);
        }
        for (const TwoLink &link : alpha_two[(size_t)ia]) {
            int old_row = link.target * nb + ib;
            double value = 0.5 *
                           data.eri[(size_t)eri_index(link.p, link.q, link.r,
                                                      link.s, norb)] *
                           (double)link.sign;
            add_entry(&entries, inverse_perm, old_row, value);
        }
        for (const TwoLink &link : beta_two[(size_t)ib]) {
            int old_row = ia * nb + link.target;
            double value = 0.5 *
                           data.eri[(size_t)eri_index(link.p, link.q, link.r,
                                                      link.s, norb)] *
                           (double)link.sign;
            add_entry(&entries, inverse_perm, old_row, value);
        }
        for (const OneLink &a : alpha_one[(size_t)ia]) {
            for (const OneLink &b : beta_one[(size_t)ib]) {
                int old_row = a.target * nb + b.target;
                double value =
                    data.eri[(size_t)eri_index(a.p, a.q, b.p, b.q, norb)] *
                    (double)(a.sign * b.sign);
                add_entry(&entries, inverse_perm, old_row, value);
            }
        }

        std::sort(entries.begin(), entries.end(),
                  [](const std::pair<int, double> &a,
                     const std::pair<int, double> &b) {
                      return a.first < b.first;
                  });

        for (size_t i = 0; i < entries.size(); ++i) {
            int row = entries[i].first;
            double value = entries[i].second;

            while (i + 1U < entries.size() && entries[i + 1U].first == row) {
                ++i;
                value += entries[i].second;
            }

            if (value != 0.0) {
                if (matrix->row_ind.size() >=
                    (size_t)std::numeric_limits<int>::max()) {
                    fprintf(stderr, "CSC nnz exceeds int range\n");
                    return false;
                }
                matrix->row_ind.push_back(row);
                matrix->values.push_back(value);
            }
        }

        if ((sorted_col + 1) % 5000 == 0 || sorted_col + 1 == n) {
            double elapsed = now_seconds() - start;
            printf("CSC generation progress %d/%d columns nnz=%zu "
                   "elapsed=%.2f sec\n",
                   sorted_col + 1, n, matrix->row_ind.size(), elapsed);
            fflush(stdout);
        }
    }

    matrix->col_ptr[(size_t)n] = (int)matrix->row_ind.size();
    matrix->nnz = (int)matrix->row_ind.size();
    return true;
}

static CscMatrixView matrix_view(const CscMatrix &matrix)
{
    CscMatrixView view = {matrix.n, matrix.nnz, matrix.col_ptr.data(),
                          matrix.row_ind.data(), matrix.values.data()};
    return view;
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

typedef struct {
    cusparseHandle_t sparse_handle;
    cusparseSpMatDescr_t matrix;
    void *buffer;
    size_t buffer_size;
} PrimmeGpuMatrix;

static void primme_gpu_matvec(void *x, PRIMME_INT *ldx, void *y,
                              PRIMME_INT *ldy, int *block_size,
                              primme_params *primme, int *ierr)
{
    PrimmeGpuMatrix *matrix = (PrimmeGpuMatrix *)primme->matrix;
    cusparseDnMatDescr_t dense_x = NULL;
    cusparseDnMatDescr_t dense_y = NULL;
    const double one = 1.0;
    const double zero = 0.0;
    size_t required = 0;
    cusparseStatus_t sparse_status = CUSPARSE_STATUS_SUCCESS;

    sparse_status = cusparseCreateDnMat(
        &dense_x, primme->nLocal, *block_size, *ldx, x, CUDA_R_64F,
        CUSPARSE_ORDER_COL);
    if (sparse_status == CUSPARSE_STATUS_SUCCESS) {
        sparse_status = cusparseCreateDnMat(
            &dense_y, primme->nLocal, *block_size, *ldy, y, CUDA_R_64F,
            CUSPARSE_ORDER_COL);
    }
    if (sparse_status == CUSPARSE_STATUS_SUCCESS) {
        sparse_status = cusparseSpMM_bufferSize(
            matrix->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matrix->matrix, dense_x,
            &zero, dense_y, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &required);
    }
    if (sparse_status == CUSPARSE_STATUS_SUCCESS &&
        required > matrix->buffer_size) {
        cudaError_t cuda_status = cudaFree(matrix->buffer);
        if (cuda_status == cudaSuccess) {
            cuda_status = cudaMalloc(&matrix->buffer, required);
        }
        if (cuda_status != cudaSuccess) {
            sparse_status = CUSPARSE_STATUS_ALLOC_FAILED;
        } else {
            matrix->buffer_size = required;
        }
    }
    if (sparse_status == CUSPARSE_STATUS_SUCCESS) {
        sparse_status = cusparseSpMM(
            matrix->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matrix->matrix, dense_x,
            &zero, dense_y, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
            matrix->buffer);
    }
    if (dense_x != NULL) {
        cusparseDestroyDnMat(dense_x);
    }
    if (dense_y != NULL) {
        cusparseDestroyDnMat(dense_y);
    }
    *ierr = sparse_status == CUSPARSE_STATUS_SUCCESS ? 0 : 1;
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

static bool write_ch4_matrix_cache(const char *path, const CscMatrix &matrix,
                                   double matrix_norm,
                                   const std::vector<double> &diagonal,
                                   const ActiveSpaceData &active)
{
    const char magic[8] = {'I', 'P', 'T', 'C', 'H', '4', 'C', '1'};
    Ch4MatrixCacheHeader header = {};
    char temporary_path[4096];
    FILE *file = NULL;
    bool ok = false;

    if (path == NULL || path[0] == '\0' ||
        diagonal.size() != (size_t)matrix.n) {
        return false;
    }
    memcpy(header.magic, magic, sizeof(magic));
    header.version = 1;
    header.n = matrix.n;
    header.nnz = matrix.nnz;
    header.matrix_inf_norm = matrix_norm;
    header.norb = active.norb;
    header.nelec = active.active_electrons;
    header.neleca = active.neleca;
    header.nelecb = active.nelecb;
    header.na = active.na;
    header.nb = active.nb;
    snprintf(temporary_path, sizeof(temporary_path), "%s.tmp", path);
    file = fopen(temporary_path, "wb");
    if (file == NULL) {
        return false;
    }
    ok = fwrite(&header, sizeof(header), 1, file) == 1 &&
         fwrite(matrix.col_ptr.data(), sizeof(int), matrix.col_ptr.size(),
                file) == matrix.col_ptr.size() &&
         fwrite(matrix.row_ind.data(), sizeof(int), matrix.row_ind.size(),
                file) == matrix.row_ind.size() &&
         fwrite(matrix.values.data(), sizeof(double), matrix.values.size(),
                file) == matrix.values.size() &&
         fwrite(diagonal.data(), sizeof(double), diagonal.size(), file) ==
             diagonal.size() &&
         fflush(file) == 0;
    if (fclose(file) != 0) {
        ok = false;
    }
    if (!ok || rename(temporary_path, path) != 0) {
        remove(temporary_path);
        return false;
    }
    return true;
}

static bool read_ch4_matrix_cache(const char *path, CscMatrix *matrix,
                                  double *matrix_norm,
                                  std::vector<double> *diagonal,
                                  ActiveSpaceData *active)
{
    const char magic[8] = {'I', 'P', 'T', 'C', 'H', '4', 'C', '1'};
    Ch4MatrixCacheHeader header = {};
    FILE *file = NULL;
    bool ok = false;

    if (path == NULL || path[0] == '\0') {
        return false;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return false;
    }
    if (fread(&header, sizeof(header), 1, file) != 1 ||
        memcmp(header.magic, magic, sizeof(magic)) != 0 ||
        header.version != 1 || header.n <= 0 || header.nnz < 0) {
        fclose(file);
        return false;
    }
    matrix->n = header.n;
    matrix->nnz = header.nnz;
    matrix->col_ptr.assign((size_t)header.n + 1U, 0);
    matrix->row_ind.assign((size_t)header.nnz, 0);
    matrix->values.assign((size_t)header.nnz, 0.0);
    diagonal->assign((size_t)header.n, 0.0);
    ok = fread(matrix->col_ptr.data(), sizeof(int), matrix->col_ptr.size(),
               file) == matrix->col_ptr.size() &&
         fread(matrix->row_ind.data(), sizeof(int), matrix->row_ind.size(),
               file) == matrix->row_ind.size() &&
         fread(matrix->values.data(), sizeof(double), matrix->values.size(),
               file) == matrix->values.size() &&
         fread(diagonal->data(), sizeof(double), diagonal->size(), file) ==
             diagonal->size();
    fclose(file);
    if (!ok ||
        !ipt_validate_csc(matrix->col_ptr.data(), matrix->row_ind.data(),
                          matrix->n, matrix->nnz) ||
        !isfinite(header.matrix_inf_norm) || header.matrix_inf_norm < 0.0) {
        *matrix = CscMatrix();
        diagonal->clear();
        return false;
    }
    active->norb = header.norb;
    active->active_electrons = header.nelec;
    active->neleca = header.neleca;
    active->nelecb = header.nelecb;
    active->na = header.na;
    active->nb = header.nb;
    active->generation_time_sec = 0.0;
    *matrix_norm = header.matrix_inf_norm;
    return true;
}

static bool build_ch4_diagnostics(const CscMatrixView *matrix,
                                  int requested_k,
                                  Ch4Diagnostics *diagnostics)
{
    IPTDegeneracyOptions options = ipt_degeneracy_options(
        ipt_cuda_env_double("IPT_DEGENERACY_THRESHOLD", 0.0));

    diagnostics->oversample =
        ipt_cuda_env_int("IPT_BLOCK_CLUSTER_OVERSAMPLE", 0);
    diagnostics->adaptive_enabled =
        ipt_cuda_env_flag("IPT_BLOCK_CLUSTER_ADAPTIVE");
    diagnostics->adaptive_coupling_tau = ipt_cuda_env_double(
        "IPT_BLOCK_CLUSTER_COUPLING_TAU", 0.1);
    diagnostics->adaptive_limit_hit = 0;
    diagnostics->adaptive_added_indices.clear();
    if (diagnostics->adaptive_enabled) {
        diagnostics->oversample = 0;
    }
    diagnostics->basis_target_k = requested_k;
    if (diagnostics->oversample > 0) {
        diagnostics->basis_target_k =
            diagnostics->oversample > matrix->n - requested_k
                ? matrix->n
                : requested_k + diagnostics->oversample;
    }
    diagnostics->basis_cols = 0;
    diagnostics->max_block_size = options.max_block_size;
    diagnostics->too_large = 0;
    diagnostics->offending_block_size = 0;
    diagnostics->target_block_id = -1;
    ipt_extract_csc_diagonal_host(matrix->col_ptr, matrix->row_ind,
                                  matrix->values, matrix->n,
                                  &diagnostics->original_diagonal);
    ipt_build_stable_sort_permutation(diagnostics->original_diagonal,
                                      &diagnostics->perm,
                                      &diagnostics->inverse_perm);
    if (!ipt_permute_csc(matrix->col_ptr, matrix->row_ind, matrix->values,
                         matrix->n, diagnostics->inverse_perm,
                         &diagnostics->sorted_col_ptr,
                         &diagnostics->sorted_row_ind,
                         &diagnostics->sorted_values)) {
        return false;
    }
    ipt_extract_csc_diagonal_vectors(
        diagnostics->sorted_col_ptr, diagnostics->sorted_row_ind,
        diagnostics->sorted_values, matrix->n,
        &diagnostics->sorted_diagonal);
    diagnostics->blocks = ipt_target_degenerate_subspaces(
        diagnostics->sorted_col_ptr, diagnostics->sorted_row_ind,
        diagnostics->sorted_values, diagnostics->sorted_diagonal,
        diagnostics->basis_target_k, options, &diagnostics->too_large);
    for (size_t i = 0; i < diagnostics->blocks.size(); ++i) {
        diagnostics->offending_block_size =
            std::max(diagnostics->offending_block_size,
                     diagnostics->blocks[i].second -
                         diagnostics->blocks[i].first + 1);
    }
    if (!diagnostics->too_large) {
        ipt_block_cluster_add_target_clusters(
            &diagnostics->blocks, matrix->n, diagnostics->basis_target_k);
        if (diagnostics->adaptive_enabled) {
            ipt_block_cluster_adaptive_expand(
                diagnostics->sorted_col_ptr, diagnostics->sorted_row_ind,
                diagnostics->sorted_values,
                diagnostics->sorted_diagonal, requested_k,
                diagnostics->max_block_size,
                diagnostics->adaptive_coupling_tau, &diagnostics->blocks,
                &diagnostics->adaptive_added_indices,
                &diagnostics->adaptive_limit_hit);
        }
    }
    for (size_t i = 0; i < diagnostics->blocks.size(); ++i) {
        int first = diagnostics->blocks[i].first;
        int last = diagnostics->blocks[i].second;

        diagnostics->basis_cols += last - first + 1;
        if (first <= requested_k - 1 && requested_k - 1 <= last) {
            diagnostics->target_block_id = (int)i;
        }
    }
    return true;
}

static double relative_eigen_residual(const CscMatrixView *matrix,
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
    TrialResult result = {};
    primme_params primme;
    std::vector<double> eval((size_t)k, 0.0);
    std::vector<double> rnorm((size_t)k, 0.0);
    std::vector<double> evec((size_t)matrix->n * (size_t)k, 0.0);
    PrimmeGpuMatrix gpu_matrix = {};
    cublasHandle_t cublas_handle = NULL;
    int *d_col_ptr = NULL;
    int *d_row_ind = NULL;
    double *d_values = NULL;
    double *d_evec = NULL;
    int status = IPT_CUDA_SUCCESS;
    int ret = 0;
    bool primme_initialized = false;

    result.method = "PRIMME_CUBLAS";
    result.repeat = repeat;
    result.requested_k = k;
    result.basis_cols = k;
    result.returned_k = 0;
    result.oversample = 0;
    result.rayleigh_ritz_used_qr = 0;
    result.time_sec = NAN;
    result.api_total_time_sec = NAN;
    result.max_relative_eigen_residual = NAN;
    result.max_relative_eigen_residual_index = -1;
    result.relative_fixed_point_residual = NAN;
    result.basis_orthogonality_frobenius_error = NAN;
    result.basis_orthogonality_max_abs_error = NAN;
    result.ritz_vectors_orthogonality_frobenius_error = NAN;
    result.ritz_vectors_orthogonality_max_abs_error = NAN;
    result.davidson_target_index = -1;
    result.davidson_residual_before = NAN;
    result.davidson_residual_after = NAN;
    result.davidson_denom_clip = NAN;
    result.adaptive_coupling_tau = NAN;
    result.adaptive_target_block_start = -1;
    result.adaptive_target_block_end = -1;
    result.adaptive_added_indices = "none";

    if (k <= 0 || k > matrix->n) {
        result.status = IPT_CUDA_INVALID_ARGUMENT;
        snprintf(result.error, sizeof(result.error), "invalid k=%d", k);
        return result;
    }

    primme_initialize(&primme);
    primme_initialized = true;
    primme.n = matrix->n;
    primme.numEvals = k;
    primme.matrix = (void *)&gpu_matrix;
    primme.matrixMatvec = primme_gpu_matvec;
    primme.target = primme_smallest;
    primme.eps = tol;
    primme.aNorm = matrix_norm;
    primme.maxMatvecs = max_matvecs;
    primme.printLevel = 0;
    primme.outputFile = stdout;
    primme_set_method(PRIMME_DYNAMIC, &primme);

    {
        double setup_start = now_seconds();

        BENCH_CUDA_CHECK(cudaMalloc((void **)&d_col_ptr,
                              (size_t)(matrix->n + 1) * sizeof(int)));
        BENCH_CUDA_CHECK(cudaMalloc((void **)&d_row_ind,
                              (size_t)matrix->nnz * sizeof(int)));
        BENCH_CUDA_CHECK(cudaMalloc((void **)&d_values,
                              (size_t)matrix->nnz * sizeof(double)));
        BENCH_CUDA_CHECK(cudaMalloc((void **)&d_evec,
                              (size_t)matrix->n * (size_t)k *
                                  sizeof(double)));
        BENCH_CUDA_CHECK(cudaMemcpy(d_col_ptr, matrix->col_ptr,
                              (size_t)(matrix->n + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));
        BENCH_CUDA_CHECK(cudaMemcpy(d_row_ind, matrix->row_ind,
                              (size_t)matrix->nnz * sizeof(int),
                              cudaMemcpyHostToDevice));
        BENCH_CUDA_CHECK(cudaMemcpy(d_values, matrix->values,
                              (size_t)matrix->nnz * sizeof(double),
                              cudaMemcpyHostToDevice));
        BENCH_CUSPARSE_CHECK(cusparseCreate(&gpu_matrix.sparse_handle));
        /* The Hamiltonian is symmetric, so its CSC arrays are also a valid
           CSR representation of the same matrix. This matches PRIMME's
           official cuSPARSE example path. */
        BENCH_CUSPARSE_CHECK(cusparseCreateCsr(
            &gpu_matrix.matrix, matrix->n, matrix->n, matrix->nnz, d_col_ptr,
            d_row_ind, d_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        BENCH_CUBLAS_CHECK(cublasCreate(&cublas_handle));
        primme.queue = &cublas_handle;
        BENCH_CUDA_CHECK(cudaDeviceSynchronize());
        result.transfer_setup_time_sec = now_seconds() - setup_start;
    }

    {
        double start = now_seconds();

        ret = cublas_dprimme(eval.data(), d_evec, rnorm.data(), &primme);
        BENCH_CUDA_CHECK(cudaDeviceSynchronize());
        result.time_sec = now_seconds() - start;
        result.api_total_time_sec =
            result.transfer_setup_time_sec + result.time_sec;
    }

    result.iterations = (int)primme.stats.numOuterIterations;
    result.matvecs = (long long)primme.stats.numMatvecs;
    if (ret != 0) {
        result.status = ret;
        snprintf(result.error, sizeof(result.error), "dprimme returned %d",
                 ret);
    } else {
        double max_residual = 0.0;

        result.returned_k = k;
        BENCH_CUDA_CHECK(cudaMemcpy(evec.data(), d_evec,
                              (size_t)matrix->n * (size_t)k *
                                  sizeof(double),
                              cudaMemcpyDeviceToHost));
        ipt_block_orthogonality_stats(
            evec, matrix->n, k,
            &result.ritz_vectors_orthogonality_frobenius_error,
            &result.ritz_vectors_orthogonality_max_abs_error,
            &result.ritz_vectors_has_nan_or_inf);

        const int pairs_to_check = std::min(result.requested_k,
                                            result.returned_k);
        for (int col = 0; col < pairs_to_check; ++col) {
            const double *vector =
                evec.data() + (size_t)col * (size_t)matrix->n;
            double residual =
                relative_eigen_residual(matrix, vector, eval[col],
                                        matrix_norm);

            if (col == 0 || residual > max_residual) {
                max_residual = residual;
                result.max_relative_eigen_residual_index = col;
            }
            result.eigenvalues.push_back(eval[col]);
            result.relative_eigen_residuals.push_back(residual);
        }
        result.max_relative_eigen_residual =
            pairs_to_check > 0 ? max_residual : NAN;
    }

cleanup:
    if (status != IPT_CUDA_SUCCESS && result.status == 0) {
        result.status = status;
        snprintf(result.error, sizeof(result.error), "%s",
                 ipt_cuda_status_string(status));
    }
    cudaFree(d_col_ptr);
    cudaFree(d_row_ind);
    cudaFree(d_values);
    cudaFree(d_evec);
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
    if (primme_initialized) {
        primme_free(&primme);
    }
    return result;
}


static TrialResult run_ipt_once(const CscMatrixView *matrix, double tol,
                                int maxiter, double matrix_norm, int k,
                                int repeat)
{
    TrialResult result = {};
    IPTCudaResult ipt = {};
    int status = IPT_CUDA_SUCCESS;

    result.method = "IPT_CUDA_BLOCK_CLUSTER";
    result.repeat = repeat;
    result.requested_k = k;
    result.basis_cols = 0;
    result.returned_k = 0;
    result.oversample = 0;
    result.rayleigh_ritz_used_qr = 0;
    result.time_sec = NAN;
    result.api_total_time_sec = NAN;
    result.max_relative_eigen_residual = NAN;
    result.max_relative_eigen_residual_index = -1;
    result.relative_fixed_point_residual = NAN;
    result.basis_orthogonality_frobenius_error = NAN;
    result.basis_orthogonality_max_abs_error = NAN;
    result.ritz_vectors_orthogonality_frobenius_error = NAN;
    result.ritz_vectors_orthogonality_max_abs_error = NAN;
    result.davidson_target_index = -1;
    result.davidson_residual_before = NAN;
    result.davidson_residual_after = NAN;
    result.davidson_denom_clip = NAN;
    result.adaptive_coupling_tau = NAN;
    result.adaptive_target_block_start = -1;
    result.adaptive_target_block_end = -1;
    result.adaptive_added_indices = "none";

    if (k <= 0 || k > matrix->n) {
        result.status = IPT_CUDA_INVALID_ARGUMENT;
        snprintf(result.error, sizeof(result.error), "invalid k=%d", k);
        return result;
    }

    {
        double start = now_seconds();
        status = ipt_cuda_sparse_csc_tol(matrix->col_ptr, matrix->row_ind,
                                         matrix->values, matrix->n, k,
                                         matrix->nnz, tol, maxiter, &ipt);
        cudaDeviceSynchronize();
        result.api_total_time_sec = now_seconds() - start;
    }

    result.status = status;
    if (status != IPT_CUDA_SUCCESS) {
        snprintf(result.error, sizeof(result.error), "%s",
                 ipt_cuda_status_string(status));
    } else {
        double max_residual = 0.0;
        int pairs_to_check = 0;

        result.basis_cols = ipt.basis_cols;
        result.returned_k = ipt.k;
        result.oversample = ipt.oversample;
        result.rayleigh_ritz_used_qr = ipt.rayleigh_ritz_used_qr;
        result.iterations = ipt.iterations;
        result.matvecs = ipt.matvecs;
        result.preparation_time_sec = ipt.preparation_time_sec;
        result.transfer_setup_time_sec = ipt.transfer_setup_time_sec;
        result.iteration_time_sec = ipt.iteration_time_sec;
        result.rayleigh_ritz_time_sec = ipt.rayleigh_ritz_time_sec;
        result.time_sec = ipt.solve_time_sec;
        result.relative_fixed_point_residual = ipt.fixed_point_residual;
        result.basis_orthogonality_frobenius_error =
            ipt.basis_orthogonality_frobenius_error;
        result.basis_orthogonality_max_abs_error =
            ipt.basis_orthogonality_max_abs_error;
        result.ritz_vectors_orthogonality_frobenius_error =
            ipt.ritz_vectors_orthogonality_frobenius_error;
        result.ritz_vectors_orthogonality_max_abs_error =
            ipt.ritz_vectors_orthogonality_max_abs_error;
        result.basis_has_nan_or_inf = ipt.basis_has_nan_or_inf;
        result.ritz_vectors_has_nan_or_inf =
            ipt.ritz_vectors_has_nan_or_inf;
        result.davidson_attempted = ipt.davidson_attempted;
        result.davidson_accepted = ipt.davidson_accepted;
        result.davidson_target_index = ipt.davidson_target_index;
        result.davidson_residual_before = ipt.davidson_residual_before;
        result.davidson_residual_after = ipt.davidson_residual_after;
        result.davidson_denom_clip = ipt.davidson_denom_clip;
        result.adaptive_block_enabled = ipt.adaptive_block_enabled;
        result.adaptive_coupling_tau = ipt.adaptive_coupling_tau;
        result.adaptive_limit_hit = ipt.adaptive_limit_hit;
        result.adaptive_target_block_start =
            ipt.adaptive_target_block_start;
        result.adaptive_target_block_end = ipt.adaptive_target_block_end;
        result.adaptive_added_indices = ipt.adaptive_added_indices;
        result.davidson_restart_count = ipt.davidson_restart_count;
        if (ipt.davidson_history_count > 0 &&
            ipt.davidson_history != NULL) {
            result.davidson_history.assign(
                ipt.davidson_history,
                ipt.davidson_history + ipt.davidson_history_count);
        }
        if (ipt.davidson_selection_history_count > 0 &&
            ipt.davidson_selection_history != NULL) {
            result.davidson_selection_history.assign(
                ipt.davidson_selection_history,
                ipt.davidson_selection_history +
                    ipt.davidson_selection_history_count);
        }
        if (ipt.davidson_block_history_count > 0 &&
            ipt.davidson_block_history != NULL) {
            result.davidson_block_history.assign(
                ipt.davidson_block_history,
                ipt.davidson_block_history +
                    ipt.davidson_block_history_count);
        }
        result.jd_local_attempted = ipt.jd_local_attempted;
        result.jd_local_accepted = ipt.jd_local_accepted;
        if (ipt.jd_local_history_count > 0 &&
            ipt.jd_local_history != NULL) {
            result.jd_local_history.assign(
                ipt.jd_local_history,
                ipt.jd_local_history + ipt.jd_local_history_count);
        }
        pairs_to_check = std::min(result.requested_k, result.returned_k);
        result.eigenvectors.assign(
            ipt.vectors,
            ipt.vectors + (size_t)matrix->n * (size_t)pairs_to_check);
        for (int col = 0; col < pairs_to_check; ++col) {
            const double *vector = ipt.vectors + (size_t)col * (size_t)matrix->n;
            double residual =
                relative_eigen_residual(matrix, vector, ipt.values[col],
                                        matrix_norm);

            if (col == 0 || residual > max_residual) {
                max_residual = residual;
                result.max_relative_eigen_residual_index = col;
            }
            result.eigenvalues.push_back(ipt.values[col]);
            result.relative_eigen_residuals.push_back(residual);
        }
        result.max_relative_eigen_residual =
            pairs_to_check > 0 ? max_residual : NAN;
    }

    ipt_cuda_free_result(&ipt);
    return result;
}


static const char *trial_status(const TrialResult &result)
{
    if (result.status != IPT_CUDA_SUCCESS) {
        return "api_error";
    }
    if (result.returned_k != result.requested_k) {
        return "returned_k_mismatch";
    }
    return isfinite(result.max_relative_eigen_residual) &&
                   result.max_relative_eigen_residual <=
                       CONVERGENCE_THRESHOLD
               ? "success"
               : "failed";
}

static bool trial_succeeded(const TrialResult &result)
{
    return strcmp(trial_status(result), "success") == 0;
}

static void write_trial_csv(FILE *csv, const TrialResult &result)
{
    fprintf(csv, "%s,%d,%d,%d,%d,%d,%s,%.17g,%.17g,%d,%.17g\n",
            result.method, result.repeat, result.requested_k,
            result.basis_cols, result.returned_k, result.iterations,
            trial_status(result), result.api_total_time_sec,
            result.max_relative_eigen_residual,
            result.max_relative_eigen_residual_index,
            result.relative_fixed_point_residual);
}

static void write_trial_summary(FILE *summary, const TrialResult &result)
{
    auto max_in_range = [&](size_t first, size_t last) {
        double maximum = NAN;
        size_t count = result.relative_eigen_residuals.size();

        last = std::min(last, count);
        for (size_t i = first; i < last; ++i) {
            if (!isfinite(maximum) ||
                result.relative_eigen_residuals[i] > maximum) {
                maximum = result.relative_eigen_residuals[i];
            }
        }
        return maximum;
    };

    fprintf(summary,
            "method=%s repeat=%d requested_k=%d basis_cols=%d returned_k=%d "
            "iterations=%d status=%s time_total_sec=%.17g "
            "max_relative_eigen_residual=%.17g "
            "max_relative_eigen_residual_index=%d "
            "relative_fixed_point_residual=%.17g "
            "oversample=%d rayleigh_ritz_used_qr=%d "
            "basis_orthogonality_frobenius_error=%.17g "
            "basis_orthogonality_max_abs_error=%.17g "
            "ritz_vectors_orthogonality_frobenius_error=%.17g "
            "ritz_vectors_orthogonality_max_abs_error=%.17g "
            "basis_has_nan_or_inf=%d ritz_vectors_has_nan_or_inf=%d "
            "davidson_attempted=%d davidson_accepted=%d "
            "davidson_target_index=%d davidson_residual_before=%.17g "
            "davidson_residual_after=%.17g davidson_denom_clip=%.17g "
            "davidson_history_count=%zu "
            "davidson_restart_count=%d "
            "davidson_selection_history_count=%zu "
            "davidson_block_history_count=%zu "
            "jd_local_attempted=%d jd_local_accepted=%d "
            "jd_local_history_count=%zu "
            "adaptive_block_enabled=%d adaptive_coupling_tau=%.17g "
            "adaptive_limit_hit=%d adaptive_added_indices=%s "
            "adaptive_final_target_block_start=%d "
            "adaptive_final_target_block_end=%d "
            "first_10_max_relative_eigen_residual=%.17g "
            "middle_10_max_relative_eigen_residual=%.17g "
            "last_10_max_relative_eigen_residual=%.17g\n",
            result.method, result.repeat, result.requested_k,
            result.basis_cols, result.returned_k, result.iterations,
            trial_status(result), result.api_total_time_sec,
            result.max_relative_eigen_residual,
            result.max_relative_eigen_residual_index,
            result.relative_fixed_point_residual, result.oversample,
            result.rayleigh_ritz_used_qr,
            result.basis_orthogonality_frobenius_error,
            result.basis_orthogonality_max_abs_error,
            result.ritz_vectors_orthogonality_frobenius_error,
            result.ritz_vectors_orthogonality_max_abs_error,
            result.basis_has_nan_or_inf,
            result.ritz_vectors_has_nan_or_inf,
            result.davidson_attempted, result.davidson_accepted,
            result.davidson_target_index, result.davidson_residual_before,
            result.davidson_residual_after, result.davidson_denom_clip,
            result.davidson_history.size(),
            result.davidson_restart_count,
            result.davidson_selection_history.size(),
            result.davidson_block_history.size(),
            result.jd_local_attempted, result.jd_local_accepted,
            result.jd_local_history.size(),
            result.adaptive_block_enabled,
            result.adaptive_coupling_tau, result.adaptive_limit_hit,
            result.adaptive_added_indices.c_str(),
            result.adaptive_target_block_start,
            result.adaptive_target_block_end, max_in_range(0, 10),
            max_in_range(10, 20), max_in_range(20, 30));
}

static void write_pair_residuals(FILE *pair_csv, const TrialResult &result)
{
    size_t count = std::min(result.eigenvalues.size(),
                            result.relative_eigen_residuals.size());

    for (size_t pair_index = 0; pair_index < count; ++pair_index) {
        fprintf(pair_csv, "%s,%d,%d,%d,%d,%zu,%.17g,%.17g\n",
                result.method, result.repeat, result.requested_k,
                result.basis_cols, result.returned_k, pair_index,
                result.eigenvalues[pair_index],
                result.relative_eigen_residuals[pair_index]);
    }
}

static void write_davidson_history(FILE *history_csv,
                                   const TrialResult &result)
{
    for (const IPTDavidsonHistoryEntry &entry :
         result.davidson_history) {
        fprintf(history_csv,
                "%d,%d,%.17g,%.17g,%d,%d,%.17g,%d,%.17g,%.17g,%.17g,%d\n",
                entry.davidson_step, entry.active_pair_index,
                entry.residual_before, entry.residual_after,
                entry.accepted, entry.basis_cols,
                entry.max_relative_eigen_residual,
                entry.max_relative_eigen_residual_index,
                entry.pair_28_residual, entry.pair_29_residual,
                entry.orthogonality_max_abs_error, entry.restarted);
    }
}

static void write_davidson_selection_history(
    FILE *history_csv, const TrialResult &result)
{
    for (const IPTDavidsonSelectionEntry &entry :
         result.davidson_selection_history) {
        fprintf(history_csv,
                "%d,%d,%.17g,%d,%d,%d,%d,%d,%d,%.17g,%.17g\n",
                entry.davidson_step, entry.pair_index, entry.residual,
                entry.selected_by_residual, entry.selected_forced,
                entry.skipped_converged, entry.skipped_active_tol,
                entry.skipped_linear_dependent,
                entry.skipped_locked_old_logic_should_not_happen,
                entry.correction_norm_before_ortho,
                entry.correction_norm_after_ortho);
    }
}

static void write_davidson_block_history(
    FILE *history_csv, const TrialResult &result)
{
    for (const IPTDavidsonBlockHistoryEntry &entry :
         result.davidson_block_history) {
        fprintf(history_csv,
                "%d,%s,%d,%d,%s,%s,%.17g,%.17g,%.17g,%.17g,"
                "%.17g,%.17g,%d,%s,%d,%d,%.17g,%.17g\n",
                entry.davidson_step, entry.active_pairs,
                entry.accepted_corrections,
                entry.rejected_corrections,
                entry.correction_norm_before_ortho,
                entry.correction_norm_after_ortho,
                entry.residual_before_global,
                entry.residual_after_global,
                entry.pair_28_before, entry.pair_28_after,
                entry.pair_29_before, entry.pair_29_after,
                entry.accepted, entry.reject_reason,
                entry.basis_cols_before, entry.basis_cols_after,
                entry.orthogonality_error_before,
                entry.orthogonality_error_after);
    }
}

static void write_jd_local_history(FILE *history_csv,
                                   const TrialResult &result)
{
    for (const IPTJDLocalHistoryEntry &entry : result.jd_local_history) {
        fprintf(history_csv,
                "%d,%d,%.17g,%.17g,%d,%d,%.17g,%d,%.17g,%.17g,%.17g\n",
                entry.jd_step, entry.active_pair_index,
                entry.residual_before, entry.residual_after,
                entry.accepted, entry.basis_cols,
                entry.max_relative_eigen_residual,
                entry.max_relative_eigen_residual_index,
                entry.pair_28_residual, entry.pair_29_residual,
                entry.orthogonality_max_abs_error);
    }
}

static void write_residual_support(
    FILE *csv, const CscMatrixView *matrix,
    const Ch4Diagnostics &diagnostics, const TrialResult &result,
    int pair_index, int top)
{
    std::vector<double> av((size_t)matrix->n, 0.0);
    std::vector<double> residual((size_t)matrix->n, 0.0);
    std::vector<int> order((size_t)matrix->n, 0);
    int target_start = -1;
    int target_end = -1;

    if (csv == NULL || pair_index < 0 ||
        pair_index >= (int)result.eigenvalues.size() ||
        (size_t)(pair_index + 1) * (size_t)matrix->n >
            result.eigenvectors.size()) {
        return;
    }
    if (diagnostics.target_block_id >= 0 &&
        diagnostics.target_block_id < (int)diagnostics.blocks.size()) {
        target_start =
            diagnostics.blocks[(size_t)diagnostics.target_block_id].first;
        target_end =
            diagnostics.blocks[(size_t)diagnostics.target_block_id].second;
    }
    {
        const double *vector =
            result.eigenvectors.data() +
            (size_t)pair_index * (size_t)matrix->n;
        double eigenvalue = result.eigenvalues[(size_t)pair_index];

        csc_matvec_block(matrix, vector, matrix->n, av.data(), matrix->n, 1);
        for (int original = 0; original < matrix->n; ++original) {
            residual[(size_t)original] =
                av[(size_t)original] - eigenvalue * vector[original];
            order[(size_t)original] = original;
        }
        std::stable_sort(
            order.begin(), order.end(), [&](int a, int b) {
                return fabs(residual[(size_t)a]) >
                       fabs(residual[(size_t)b]);
            });
        top = std::max(0, std::min(top, matrix->n));
        for (int rank = 0; rank < top; ++rank) {
            int original = order[(size_t)rank];
            int sorted =
                diagnostics.inverse_perm[(size_t)original];
            double diagonal =
                diagnostics.original_diagonal[(size_t)original];

            fprintf(csv,
                    "%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%d,%d,%d,%d\n",
                    pair_index, rank + 1, original, sorted,
                    residual[(size_t)original],
                    fabs(residual[(size_t)original]), diagonal,
                    diagonal - eigenvalue,
                    sorted >= target_start && sorted <= target_end,
                    sorted >= 0 && sorted < 30,
                    sorted >= 30 && sorted < 40,
                    sorted >= 40 && sorted < 80);
        }
    }
}

static double min_adjacent_gap(const std::vector<double> &diagonal,
                               int count)
{
    double minimum = NAN;
    int limit = std::min(count, (int)diagonal.size());

    for (int i = 0; i + 1 < limit; ++i) {
        double gap = fabs(diagonal[(size_t)i + 1U] - diagonal[(size_t)i]);

        if (!isfinite(minimum) || gap < minimum) {
            minimum = gap;
        }
    }
    return minimum;
}

static double adjacent_gap(const std::vector<double> &diagonal, int index)
{
    return index >= 0 && index + 1 < (int)diagonal.size()
               ? fabs(diagonal[(size_t)index + 1U] -
                      diagonal[(size_t)index])
               : NAN;
}

static void write_diagonal_diagnostics(FILE *summary, const char *path,
                                       const Ch4Diagnostics &diagnostics,
                                       int dump_csv)
{
    FILE *csv = NULL;

    fprintf(summary, "diagonal_indexing=0-based_sorted\n");
    for (int i = 25; i <= 40; ++i) {
        fprintf(summary, "d[%d]=%.17g\n", i,
                i < (int)diagnostics.sorted_diagonal.size()
                    ? diagnostics.sorted_diagonal[(size_t)i]
                    : NAN);
    }
    for (int i = 25; i <= 39; ++i) {
        fprintf(summary, "gap_%d_%d=%.17g\n", i, i + 1,
                adjacent_gap(diagnostics.sorted_diagonal, i));
    }
    fprintf(summary,
            "min_gap_first_30=%.17g\nmin_gap_first_40=%.17g\n"
            "min_gap_first_60=%.17g\nboundary_gap_29_30=%.17g\n"
            "boundary_gap_30_31=%.17g\nboundary_gap_31_32=%.17g\n"
            "boundary_gap_32_33=%.17g\n",
            min_adjacent_gap(diagnostics.sorted_diagonal, 30),
            min_adjacent_gap(diagnostics.sorted_diagonal, 40),
            min_adjacent_gap(diagnostics.sorted_diagonal, 60),
            adjacent_gap(diagnostics.sorted_diagonal, 29),
            adjacent_gap(diagnostics.sorted_diagonal, 30),
            adjacent_gap(diagnostics.sorted_diagonal, 31),
            adjacent_gap(diagnostics.sorted_diagonal, 32));
    if (!dump_csv) {
        return;
    }
    csv = fopen(path, "w");
    if (csv == NULL) {
        fprintf(stderr, "could not open diagonal-gap output %s\n", path);
        return;
    }
    fprintf(csv,
            "sorted_index_0based,sorted_index_1based,original_index,"
            "diagonal_value,adjacent_gap_to_next\n");
    {
        int limit =
            std::min(80, (int)diagnostics.sorted_diagonal.size());

        for (int i = 0; i < limit; ++i) {
            fprintf(csv, "%d,%d,%d,%.17g,%.17g\n", i, i + 1,
                    diagnostics.perm[(size_t)i],
                    diagnostics.sorted_diagonal[(size_t)i],
                    adjacent_gap(diagnostics.sorted_diagonal, i));
        }
    }
    fclose(csv);
}

static void write_block_diagnostics(FILE *summary,
                                    const Ch4Diagnostics &diagnostics,
                                    int requested_k, int returned_k)
{
    int target_first = -1;
    int target_last = -1;
    std::string adaptive_added;

    if (diagnostics.target_block_id >= 0) {
        target_first =
            diagnostics.blocks[(size_t)diagnostics.target_block_id].first;
        target_last =
            diagnostics.blocks[(size_t)diagnostics.target_block_id].second;
    }
    for (size_t i = 0; i < diagnostics.adaptive_added_indices.size(); ++i) {
        if (!adaptive_added.empty()) {
            adaptive_added += ";";
        }
        adaptive_added +=
            std::to_string(diagnostics.adaptive_added_indices[i]);
    }
    fprintf(summary,
            "block_indexing=0-based_sorted\nrequested_k=%d\nbasis_cols=%d\n"
            "returned_k=%d\noversample=%d\nmax_block_size=%d\n"
            "number_of_blocks=%zu\ntarget_block_start=%d\n"
            "target_block_end=%d\ntarget_block_size=%d\n"
            "max_block_size_truncated=0\nmax_block_size_error=%d\n"
            "max_block_size_offending_block_size=%d\n"
            "adaptive_block_enabled=%d\n"
            "adaptive_coupling_tau=%.17g\nadaptive_limit_hit=%d\n"
            "adaptive_added_indices=%s\n"
            "adaptive_final_target_block_start=%d\n"
            "adaptive_final_target_block_end=%d\n",
            requested_k, diagnostics.basis_cols, returned_k,
            diagnostics.oversample, diagnostics.max_block_size,
            diagnostics.blocks.size(), target_first, target_last,
            target_first >= 0 ? target_last - target_first + 1 : 0,
            diagnostics.too_large, diagnostics.offending_block_size,
            diagnostics.adaptive_enabled,
            diagnostics.adaptive_coupling_tau,
            diagnostics.adaptive_limit_hit,
            adaptive_added.empty() ? "none" : adaptive_added.c_str(),
            target_first, target_last);
    fprintf(summary,
            "target_block_contains_indices_30_31_32_33_0based=%d,%d,%d,%d\n"
            "target_block_contains_positions_30_31_32_33_1based=%d,%d,%d,%d\n",
            target_first <= 30 && 30 <= target_last,
            target_first <= 31 && 31 <= target_last,
            target_first <= 32 && 32 <= target_last,
            target_first <= 33 && 33 <= target_last,
            target_first <= 29 && 29 <= target_last,
            target_first <= 30 && 30 <= target_last,
            target_first <= 31 && 31 <= target_last,
            target_first <= 32 && 32 <= target_last);
    for (size_t block_id = 0; block_id < diagnostics.blocks.size();
         ++block_id) {
        int first = diagnostics.blocks[block_id].first;
        int last = diagnostics.blocks[block_id].second;
        double internal_min_gap = NAN;

        for (int i = first; i < last; ++i) {
            double gap = adjacent_gap(diagnostics.sorted_diagonal, i);

            if (!isfinite(internal_min_gap) || gap < internal_min_gap) {
                internal_min_gap = gap;
            }
        }
        fprintf(summary,
                "block_id=%zu start=%d end=%d size=%d min_diag=%.17g "
                "max_diag=%.17g internal_min_gap=%.17g "
                "boundary_left_gap=%.17g boundary_right_gap=%.17g\n",
                block_id, first, last, last - first + 1,
                diagnostics.sorted_diagonal[(size_t)first],
                diagnostics.sorted_diagonal[(size_t)last],
                internal_min_gap,
                first > 0
                    ? adjacent_gap(diagnostics.sorted_diagonal, first - 1)
                    : NAN,
                adjacent_gap(diagnostics.sorted_diagonal, last));
    }
}

static void write_coupling_gap_diagnostics(
    FILE *summary, const char *path, const Ch4Diagnostics &diagnostics,
    int requested_k, int dump_csv)
{
    const double eps_gap = 1.0e-14;
    FILE *csv = NULL;
    double max_inside = NAN;
    double max_boundary = NAN;
    double max_block_outside = NAN;
    int inside_i = -1;
    int inside_j = -1;
    int boundary_i = -1;
    int boundary_j = -1;
    int block_i = -1;
    int block_j = -1;
    int limit = std::min(80, (int)diagnostics.sorted_diagonal.size());
    int target_first = -1;
    int target_last = -1;

    if (diagnostics.target_block_id >= 0) {
        target_first =
            diagnostics.blocks[(size_t)diagnostics.target_block_id].first;
        target_last =
            diagnostics.blocks[(size_t)diagnostics.target_block_id].second;
    }
    if (dump_csv) {
        csv = fopen(path, "w");
        if (csv != NULL) {
            fprintf(csv,
                    "i_sorted,j_sorted,i_original,j_original,d_i,d_j,"
                    "abs_gap,abs_Aij,coupling_over_gap,relation\n");
        } else {
            fprintf(stderr, "could not open coupling-gap output %s\n", path);
        }
    }
    for (int i = 0; i < limit; ++i) {
        for (int j = i + 1; j < limit; ++j) {
            double a_ij = ipt_csc_find_value(
                diagnostics.sorted_col_ptr, diagnostics.sorted_row_ind,
                diagnostics.sorted_values, i, j);
            double a_ji = ipt_csc_find_value(
                diagnostics.sorted_col_ptr, diagnostics.sorted_row_ind,
                diagnostics.sorted_values, j, i);
            double abs_aij = std::max(fabs(a_ij), fabs(a_ji));
            double gap = fabs(diagnostics.sorted_diagonal[(size_t)i] -
                              diagnostics.sorted_diagonal[(size_t)j]);
            double ratio = abs_aij / std::max(gap, eps_gap);
            const char *relation = "outside_candidate";

            if (j < requested_k) {
                relation = "inside_requested_k";
                if (!isfinite(max_inside) || ratio > max_inside) {
                    max_inside = ratio;
                    inside_i = i;
                    inside_j = j;
                }
            } else if (i < requested_k && j < requested_k + 10) {
                relation = "boundary_around_k";
                if (!isfinite(max_boundary) || ratio > max_boundary) {
                    max_boundary = ratio;
                    boundary_i = i;
                    boundary_j = j;
                }
            }
            if (target_first >= 0) {
                bool i_in = target_first <= i && i <= target_last;
                bool j_in = target_first <= j && j <= target_last;

                if (i_in != j_in &&
                    (!isfinite(max_block_outside) ||
                     ratio > max_block_outside)) {
                    max_block_outside = ratio;
                    block_i = i;
                    block_j = j;
                }
            }
            if (csv != NULL) {
                fprintf(csv,
                        "%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%s\n",
                        i, j, diagnostics.perm[(size_t)i],
                        diagnostics.perm[(size_t)j],
                        diagnostics.sorted_diagonal[(size_t)i],
                        diagnostics.sorted_diagonal[(size_t)j], gap,
                        abs_aij, ratio, relation);
            }
        }
    }
    if (csv != NULL) {
        fclose(csv);
    }
    fprintf(summary,
            "coupling_gap_eps=%.17g\n"
            "coupling_abs_Aij_definition=max(abs(Aij),abs(Aji))\n"
            "max_coupling_over_gap_inside_first_30=%.17g\n"
            "max_coupling_over_gap_inside_first_30_argmax=%d,%d\n"
            "max_coupling_over_gap_between_first_30_and_31_40=%.17g\n"
            "max_coupling_over_gap_between_first_30_and_31_40_argmax=%d,%d\n"
            "max_coupling_over_gap_between_block_and_outside=%.17g\n"
            "max_coupling_over_gap_between_block_and_outside_argmax=%d,%d\n",
            eps_gap, max_inside, inside_i, inside_j, max_boundary,
            boundary_i, boundary_j, max_block_outside, block_i, block_j);
}

int main(void)
{
    ActiveSpaceData active = {0, 0, 0, 0, 0, 0, NAN, NAN, NAN, NAN,
                              std::vector<double>(), std::vector<double>()};
    CscMatrix owned_matrix = {0, 0, std::vector<int>(), std::vector<int>(),
                              std::vector<double>()};
    int repeats = env_int("IPT_REPEATS", DEFAULT_REPEATS);
    int run_primme = env_bool("RUN_PRIMME", 0);
    int run_warmup = env_bool("RUN_WARMUP", 0);
    double tol = env_double("IPT_TOL", DEFAULT_TOL);
    int ipt_maxiter = env_int("IPT_MAXITER", DEFAULT_IPT_MAXITER);
    int ipt_k = env_int("IPT_K", DEFAULT_IPT_K);
    long long primme_max_matvecs =
        env_ll("PRIMME_MAX_MATVECS", DEFAULT_PRIMME_MAX_MATVECS);
    const char *dir = results_dir();
    const char *cache_path = getenv("IPT_CH4_MATRIX_CACHE");
    int load_matrix = env_bool("IPT_LOAD_MATRIX", 0);
    int save_matrix = env_bool("IPT_SAVE_MATRIX", 0);
    int dump_diag_gaps = env_bool("IPT_DUMP_DIAG_GAPS", 0);
    int dump_coupling_gap = env_bool("IPT_DUMP_COUPLING_GAP", 0);
    int dump_residual_support =
        env_bool("IPT_DUMP_RESIDUAL_SUPPORT", 0);
    int residual_support_pair =
        env_int("IPT_RESIDUAL_SUPPORT_PAIR", 29);
    int residual_support_top =
        env_int("IPT_RESIDUAL_SUPPORT_TOP", 100);
    int ritz_check_interval = env_int("IPT_RITZ_CHECK_INTERVAL", 0);
    const char *matrix_source = "generated";
    double matrix_norm = NAN;
    std::vector<double> cached_diagonal;
    Ch4Diagnostics diagnostics = {};
    char csv_path[4096];
    char summary_path[4096];
    char pair_csv_path[4096];
    char diag_gaps_path[4096];
    char coupling_gap_path[4096];
    char davidson_history_path[4096];
    char davidson_selection_history_path[4096];
    char davidson_block_history_path[4096];
    char jd_local_history_path[4096];
    char residual_support_path[4096];
    FILE *csv = NULL;
    FILE *summary = NULL;
    FILE *pair_csv = NULL;
    FILE *davidson_history_csv = NULL;
    FILE *davidson_selection_history_csv = NULL;
    FILE *davidson_block_history_csv = NULL;
    FILE *jd_local_history_csv = NULL;
    FILE *residual_support_csv = NULL;
    std::vector<TrialResult> primme_results;
    std::vector<TrialResult> ipt_results;
    bool benchmark_succeeded = true;

    ensure_directory(dir);
    if (load_matrix &&
        read_ch4_matrix_cache(cache_path, &owned_matrix, &matrix_norm,
                              &cached_diagonal, &active)) {
        matrix_source = "cache";
        printf("Loaded fixed CH4 CSC matrix cache from %s\n", cache_path);
    } else {
        if (load_matrix) {
            fprintf(stderr,
                    "CH4 matrix cache unavailable or invalid; generating "
                    "matrix: %s\n",
                    cache_path == NULL ? "(unset)" : cache_path);
        }
        if (!generate_active_space_integrals(&active)) {
            fprintf(stderr,
                    "failed to generate CH4/STO-6G full FCI active "
                    "integrals\n");
            return 1;
        }
        if (!generate_hamiltonian_csc(active, &owned_matrix)) {
            fprintf(stderr,
                    "failed to generate CH4/STO-6G full FCI CSC matrix\n");
            return 1;
        }
        {
            CscMatrixView generated_matrix = matrix_view(owned_matrix);

            matrix_norm = inf_norm(&generated_matrix);
            ipt_extract_csc_diagonal_host(
                generated_matrix.col_ptr, generated_matrix.row_ind,
                generated_matrix.values, generated_matrix.n,
                &cached_diagonal);
        }
        if (save_matrix) {
            if (!write_ch4_matrix_cache(cache_path, owned_matrix, matrix_norm,
                                        cached_diagonal, active)) {
                fprintf(stderr, "failed to save CH4 matrix cache to %s\n",
                        cache_path == NULL ? "(unset)" : cache_path);
                return 1;
            }
            printf("Saved fixed CH4 CSC matrix cache to %s\n", cache_path);
        }
    }

    CscMatrixView matrix = matrix_view(owned_matrix);

    if (ipt_k <= 0 || ipt_k > matrix.n) {
        fprintf(stderr, "invalid IPT_K=%d for n=%d\n", ipt_k, matrix.n);
        return 2;
    }
    if (!build_ch4_diagnostics(&matrix, ipt_k, &diagnostics)) {
        fprintf(stderr, "failed to build CH4 matrix diagnostics\n");
        return 2;
    }

    snprintf(csv_path, sizeof(csv_path),
             "%s/ch4_sto6g_fci_15876_ipt_relative_trials.csv", dir);
    snprintf(summary_path, sizeof(summary_path),
             "%s/ch4_sto6g_fci_15876_ipt_relative_summary.txt", dir);
    snprintf(pair_csv_path, sizeof(pair_csv_path),
             "%s/ch4_sto6g_fci_15876_pair_residuals.csv", dir);
    snprintf(diag_gaps_path, sizeof(diag_gaps_path),
             "%s/ch4_sto6g_fci_15876_diag_gaps.csv", dir);
    snprintf(coupling_gap_path, sizeof(coupling_gap_path),
             "%s/ch4_sto6g_fci_15876_coupling_gap.csv", dir);
    snprintf(davidson_history_path, sizeof(davidson_history_path),
             "%s/ch4_sto6g_fci_15876_davidson_history.csv", dir);
    snprintf(
        davidson_selection_history_path,
        sizeof(davidson_selection_history_path),
        "%s/ch4_sto6g_fci_15876_davidson_selection_history.csv", dir);
    snprintf(davidson_block_history_path,
             sizeof(davidson_block_history_path),
             "%s/ch4_sto6g_fci_15876_davidson_block_history.csv", dir);
    snprintf(jd_local_history_path, sizeof(jd_local_history_path),
             "%s/ch4_sto6g_fci_15876_jd_local_history.csv", dir);
    snprintf(residual_support_path, sizeof(residual_support_path),
             "%s/ch4_sto6g_fci_15876_residual_support.csv", dir);

    csv = fopen(csv_path, "w");
    summary = fopen(summary_path, "w");
    pair_csv = fopen(pair_csv_path, "w");
    davidson_history_csv = fopen(davidson_history_path, "w");
    davidson_selection_history_csv =
        fopen(davidson_selection_history_path, "w");
    davidson_block_history_csv =
        fopen(davidson_block_history_path, "w");
    jd_local_history_csv = fopen(jd_local_history_path, "w");
    if (dump_residual_support) {
        residual_support_csv = fopen(residual_support_path, "w");
    }
    if (csv == NULL || summary == NULL || pair_csv == NULL ||
        davidson_history_csv == NULL || jd_local_history_csv == NULL ||
        davidson_selection_history_csv == NULL ||
        davidson_block_history_csv == NULL ||
        (dump_residual_support && residual_support_csv == NULL)) {
        fprintf(stderr, "could not open result files in %s\n", dir);
        if (csv != NULL) {
            fclose(csv);
        }
        if (summary != NULL) {
            fclose(summary);
        }
        if (pair_csv != NULL) {
            fclose(pair_csv);
        }
        if (davidson_history_csv != NULL) {
            fclose(davidson_history_csv);
        }
        if (jd_local_history_csv != NULL) {
            fclose(jd_local_history_csv);
        }
        if (davidson_selection_history_csv != NULL) {
            fclose(davidson_selection_history_csv);
        }
        if (davidson_block_history_csv != NULL) {
            fclose(davidson_block_history_csv);
        }
        if (residual_support_csv != NULL) {
            fclose(residual_support_csv);
        }
        return 2;
    }

    fprintf(csv,
            "method,repeat,requested_k,basis_cols,returned_k,iterations,status,"
            "time_total_sec,max_relative_eigen_residual,"
            "max_relative_eigen_residual_index,"
            "relative_fixed_point_residual\n");
    fprintf(pair_csv,
            "method,repeat,requested_k,basis_cols,returned_k,pair_index,"
            "lambda,relative_eigen_residual\n");
    fprintf(davidson_history_csv,
            "davidson_step,active_pair_index,residual_before,residual_after,"
            "accepted,basis_cols,max_relative_eigen_residual,"
            "max_relative_eigen_residual_index,pair_28_residual,"
            "pair_29_residual,orthogonality_error,restarted\n");
    fprintf(davidson_selection_history_csv,
            "davidson_step,pair_index,residual,selected_by_residual,"
            "selected_forced,skipped_converged,skipped_active_tol,"
            "skipped_linear_dependent,"
            "skipped_locked_old_logic_should_not_happen,"
            "correction_norm_before_ortho,"
            "correction_norm_after_ortho\n");
    fprintf(davidson_block_history_csv,
            "davidson_step,active_pairs,accepted_corrections,"
            "rejected_corrections,correction_norm_before_ortho,"
            "correction_norm_after_ortho,residual_before_global,"
            "residual_after_global,pair_28_before,pair_28_after,"
            "pair_29_before,pair_29_after,accepted,reject_reason,"
            "basis_cols_before,basis_cols_after,"
            "orthogonality_error_before,"
            "orthogonality_error_after\n");
    fprintf(jd_local_history_csv,
            "jd_step,active_pair_index,residual_before,residual_after,"
            "accepted,basis_cols,max_relative_eigen_residual,"
            "max_relative_eigen_residual_index,pair_28_residual,"
            "pair_29_residual,orthogonality_error\n");
    if (residual_support_csv != NULL) {
        fprintf(residual_support_csv,
                "pair_index,rank,original_index,sorted_index,"
                "residual_value,abs_residual_value,diagonal_value,"
                "denom,inside_target_block,inside_first_30,"
                "inside_31_40,inside_41_80\n");
    }

    printf("CH4/STO-6G full FCI CSC benchmark: n=%d "
           "nnz=%d k=%d repeats=%d tol=%.3e run_primme=%d\n",
           matrix.n, matrix.nnz, ipt_k, repeats, tol, run_primme);
    printf("Matrix/integral generation time %.6f sec is excluded from "
           "PRIMME/IPT timings.\n",
           active.generation_time_sec);
    fflush(stdout);

    fprintf(summary,
            "convergence_metric=max_relative_eigen_residual\n"
            "convergence_threshold=1e-12\n"
            "fixed_point_residual_is_diagnostic_only=1\n"
            "matrix_source=%s\nn=%d\nnnz=%d\nmatrix_inf_norm=%.17g\n"
            "matrix_cache_path=%s\n"
            "davidson_steps=%d\ndavidson_active_max=%d\n"
            "davidson_select_tol=%.17g\ndavidson_denom_clip=%.17g\n"
            "davidson_accept_only_if_improves=%d\n"
            "davidson_ortho_repeats=%d\n"
            "davidson_protect_tol=%.17g\n"
            "davidson_locked_tol=%.17g\n"
            "davidson_restart_every=%d\n"
            "davidson_restart_keep_extra=%d\n",
            matrix_source, matrix.n, matrix.nnz, matrix_norm,
            cache_path == NULL || cache_path[0] == '\0' ? "(unset)"
                                                        : cache_path,
            ipt_cuda_env_int("IPT_DAVIDSON_STEPS", 1),
            ipt_cuda_env_int("IPT_DAVIDSON_ACTIVE_MAX", 1),
            ipt_cuda_env_double("IPT_DAVIDSON_SELECT_TOL", 1.0e-12),
            ipt_cuda_env_double("IPT_DAVIDSON_DENOM_CLIP", 1.0e-8),
            getenv("IPT_DAVIDSON_ACCEPT_ONLY_IF_IMPROVES") == NULL
                ? 1
                : ipt_cuda_env_flag(
                      "IPT_DAVIDSON_ACCEPT_ONLY_IF_IMPROVES"),
            ipt_cuda_env_int("IPT_DAVIDSON_ORTHO_REPEATS", 2),
            ipt_cuda_env_double("IPT_DAVIDSON_PROTECT_TOL", 1.0e-10),
            ipt_cuda_env_double("IPT_DAVIDSON_LOCKED_TOL", 1.0e-12),
            ipt_cuda_env_int("IPT_DAVIDSON_RESTART_EVERY", 20),
            ipt_cuda_env_int("IPT_DAVIDSON_RESTART_KEEP_EXTRA", 5));
    fprintf(summary,
            "baseline_steps=%d\n"
            "davidson_extra_steps=%d\n"
            "davidson_converged_tol=%.17g\n"
            "davidson_active_tol=%.17g\n"
            "davidson_protect_tol_role=orthogonalization_and_pollution_"
            "protection_only\n"
            "davidson_locked_old_logic_active_selection=0\n"
            "davidson_force_active_pairs=%s\n"
            "davidson_block_active=%d\n",
            ipt_cuda_env_int("IPT_DAVIDSON_STEPS", 1),
            ipt_cuda_env_int("IPT_DAVIDSON_EXTRA_STEPS", 0),
            ipt_cuda_env_double("IPT_DAVIDSON_CONVERGED_TOL", 1.0e-13),
            ipt_cuda_env_double("IPT_DAVIDSON_ACTIVE_TOL", 1.0e-13),
            getenv("IPT_DAVIDSON_FORCE_ACTIVE_PAIRS") == NULL
                ? "none"
                : getenv("IPT_DAVIDSON_FORCE_ACTIVE_PAIRS"),
            ipt_cuda_env_flag("IPT_DAVIDSON_BLOCK_ACTIVE"));
    fprintf(summary,
            "dump_residual_support=%d\n"
            "residual_support_pair=%d\n"
            "residual_support_top=%d\n"
            "jd_local_correction=%d\n"
            "jd_local_active_pairs=%s\n"
            "jd_local_window_start=%d\n"
            "jd_local_window_end=%d\n"
            "jd_local_damping=%.17g\n"
            "jd_local_max_dim=%d\n"
            "jd_local_set=%s\n"
            "jd_local_support_top=%d\n"
            "jd_local_outside_correction=%s\n"
            "jd_local_steps=%d\n"
            "jd_accept_only_if_improves=%d\n",
            dump_residual_support, residual_support_pair,
            residual_support_top,
            ipt_cuda_env_flag("IPT_JD_LOCAL_CORRECTION"),
            getenv("IPT_JD_LOCAL_ACTIVE_PAIRS") == NULL
                ? "29"
                : getenv("IPT_JD_LOCAL_ACTIVE_PAIRS"),
            ipt_cuda_env_int("IPT_JD_LOCAL_WINDOW_START", 25),
            ipt_cuda_env_int("IPT_JD_LOCAL_WINDOW_END", 40),
            ipt_cuda_env_double("IPT_JD_LOCAL_DAMPING", 1.0e-8),
            ipt_cuda_env_int("IPT_JD_LOCAL_MAX_DIM", 80),
            getenv("IPT_JD_LOCAL_SET") == NULL
                ? "window"
                : getenv("IPT_JD_LOCAL_SET"),
            ipt_cuda_env_int("IPT_JD_LOCAL_SUPPORT_TOP", 80),
            getenv("IPT_JD_LOCAL_OUTSIDE_CORRECTION") == NULL
                ? "none"
                : getenv("IPT_JD_LOCAL_OUTSIDE_CORRECTION"),
            ipt_cuda_env_int("IPT_JD_LOCAL_STEPS", 1),
            getenv("IPT_JD_ACCEPT_ONLY_IF_IMPROVES") == NULL
                ? 1
                : ipt_cuda_env_flag(
                      "IPT_JD_ACCEPT_ONLY_IF_IMPROVES"));
    write_diagonal_diagnostics(summary, diag_gaps_path, diagnostics,
                               dump_diag_gaps);
    write_coupling_gap_diagnostics(summary, coupling_gap_path, diagnostics,
                                   ipt_k, dump_coupling_gap);
    if (ritz_check_interval > 0) {
        fprintf(summary,
                "ritz_checkpoint_supported=0\n"
                "ritz_checkpoint_requested_interval=%d\n"
                "ritz_checkpoint_insertion_point="
                "ipt_block_cluster_solve_basis_gpu_after_basis_swap_and_"
                "ipt_cuda_sparse_csc_block_cluster_impl_before_final_"
                "rayleigh_ritz\n"
                "ritz_checkpoint_required_device_data="
                "d_combined_basis,d_values,d_col_ptr,d_row_ind,d_perm\n",
                ritz_check_interval);
        fprintf(stderr,
                "IPT_RITZ_CHECK_INTERVAL=%d requested, but synchronized "
                "multi-block Ritz checkpoints are diagnostic-only and not "
                "implemented in the current sequential block solver.\n",
                ritz_check_interval);
    }

    if (run_warmup) {
        TrialResult warmup_ipt =
            run_ipt_once(&matrix, tol, ipt_maxiter, matrix_norm, ipt_k, 0);
        printf("warmup IPT_GPU_BLOCK requested_k=%d basis_cols=%d "
               "returned_k=%d solve=%.6e preparation=%.6e "
               "setup=%.6e iteration=%.6e rr=%.6e api_total=%.6e "
               "relative_fixed_point_residual=%.3e "
               "max_relative_eigen_residual=%.3e status=%s\n",
               warmup_ipt.requested_k, warmup_ipt.basis_cols,
               warmup_ipt.returned_k, warmup_ipt.time_sec,
               warmup_ipt.preparation_time_sec,
               warmup_ipt.transfer_setup_time_sec,
               warmup_ipt.iteration_time_sec,
               warmup_ipt.rayleigh_ritz_time_sec,
               warmup_ipt.api_total_time_sec,
               warmup_ipt.relative_fixed_point_residual,
               warmup_ipt.max_relative_eigen_residual,
               trial_status(warmup_ipt));
        if (run_primme) {
            TrialResult warmup_primme = run_primme_once(
                &matrix, tol, primme_max_matvecs, matrix_norm, ipt_k, 0);
            printf("warmup PRIMME_CUBLAS requested_k=%d basis_cols=%d "
                   "returned_k=%d solve=%.6e setup=%.6e "
                   "max_relative_eigen_residual=%.3e status=%s\n",
                   warmup_primme.requested_k, warmup_primme.basis_cols,
                   warmup_primme.returned_k, warmup_primme.time_sec,
                   warmup_primme.transfer_setup_time_sec,
                   warmup_primme.max_relative_eigen_residual,
                   trial_status(warmup_primme));
        }
        fflush(stdout);
    }

    for (int repeat = 1; repeat <= repeats; ++repeat) {
        TrialResult ipt_result =
            run_ipt_once(&matrix, tol, ipt_maxiter, matrix_norm, ipt_k,
                         repeat);
        ipt_results.push_back(ipt_result);

        printf("repeat %d IPT_GPU_BLOCK requested_k=%d basis_cols=%d "
               "returned_k=%d solve=%.6e preparation=%.6e "
               "setup=%.6e iteration=%.6e rr=%.6e api_total=%.6e "
               "relative_fixed_point_residual=%.3e "
               "max_relative_eigen_residual=%.3e status=%s\n",
               repeat, ipt_result.requested_k, ipt_result.basis_cols,
               ipt_result.returned_k, ipt_result.time_sec,
               ipt_result.preparation_time_sec,
               ipt_result.transfer_setup_time_sec,
               ipt_result.iteration_time_sec,
               ipt_result.rayleigh_ritz_time_sec,
               ipt_result.api_total_time_sec,
               ipt_result.relative_fixed_point_residual,
               ipt_result.max_relative_eigen_residual,
               trial_status(ipt_result));
        fflush(stdout);

        if (run_primme) {
            TrialResult primme_result =
                run_primme_once(&matrix, tol, primme_max_matvecs, matrix_norm,
                                ipt_k, repeat);
            primme_results.push_back(primme_result);
            printf("repeat %d PRIMME_CUBLAS requested_k=%d basis_cols=%d "
                   "returned_k=%d solve=%.6e setup=%.6e "
                   "max_relative_eigen_residual=%.3e status=%s\n",
                   repeat, primme_result.requested_k,
                   primme_result.basis_cols, primme_result.returned_k,
                   primme_result.time_sec,
                   primme_result.transfer_setup_time_sec,
                   primme_result.max_relative_eigen_residual,
                   trial_status(primme_result));
            fflush(stdout);
        }
    }

    for (const TrialResult &result : ipt_results) {
        write_trial_csv(csv, result);
        write_pair_residuals(pair_csv, result);
        write_davidson_history(davidson_history_csv, result);
        write_davidson_selection_history(
            davidson_selection_history_csv, result);
        write_davidson_block_history(
            davidson_block_history_csv, result);
        write_jd_local_history(jd_local_history_csv, result);
        if (residual_support_csv != NULL) {
            write_residual_support(
                residual_support_csv, &matrix, diagnostics, result,
                residual_support_pair, residual_support_top);
        }
        benchmark_succeeded =
            benchmark_succeeded && trial_succeeded(result);
    }
    for (const TrialResult &result : primme_results) {
        write_trial_csv(csv, result);
        write_pair_residuals(pair_csv, result);
        benchmark_succeeded =
            benchmark_succeeded && trial_succeeded(result);
    }
    fflush(csv);
    fflush(pair_csv);
    fflush(davidson_history_csv);
    fflush(davidson_selection_history_csv);
    fflush(davidson_block_history_csv);
    fflush(jd_local_history_csv);
    if (residual_support_csv != NULL) {
        fflush(residual_support_csv);
    }

    write_block_diagnostics(
        summary, diagnostics, ipt_k,
        ipt_results.empty() ? 0 : ipt_results.back().returned_k);
    if (!ipt_results.empty() &&
        ipt_results.back().status == IPT_CUDA_INVALID_ARGUMENT) {
        fprintf(summary, "invalid_argument_trigger=%s\n",
                diagnostics.too_large ? "block_size_exceeds_max"
                                      : "solver_or_preparation");
        fprintf(summary, "invalid_argument_block_size=%d\n",
                diagnostics.offending_block_size);
    }
    for (const TrialResult &result : ipt_results) {
        write_trial_summary(summary, result);
    }
    for (const TrialResult &result : primme_results) {
        write_trial_summary(summary, result);
    }

    fclose(csv);
    fclose(summary);
    fclose(pair_csv);
    fclose(davidson_history_csv);
    fclose(davidson_selection_history_csv);
    fclose(davidson_block_history_csv);
    fclose(jd_local_history_csv);
    if (residual_support_csv != NULL) {
        fclose(residual_support_csv);
    }

    printf("wrote %s\n", csv_path);
    printf("wrote %s\n", summary_path);
    printf("wrote %s\n", pair_csv_path);
    printf("wrote %s\n", davidson_history_path);
    printf("wrote %s\n", davidson_selection_history_path);
    printf("wrote %s\n", davidson_block_history_path);
    printf("wrote %s\n", jd_local_history_path);
    if (dump_residual_support) {
        printf("wrote %s\n", residual_support_path);
    }
    return benchmark_succeeded ? 0 : 3;
}
