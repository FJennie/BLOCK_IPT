#ifndef IPT_PREPARATION_CU
#define IPT_PREPARATION_CU

#include "ipt_preparation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

static const double IPT_DEFAULT_EPS_MULTIPLIER = 1000.0;
static const double IPT_DEFAULT_REL_GAP_TOL = 1.0e-12;
static const double IPT_DEFAULT_COUPLING_GAP_RATIO_TOL = 0.2;
static const double IPT_DEFAULT_COUPLING_ETA = 0.02;
static const int IPT_DEFAULT_MAX_BLOCK_SIZE = 64;
static const int IPT_DEFAULT_MAX_LIFT_ROUNDS = 3;

extern "C" void dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda,
                       double *wr, double *wi, double *vl, int *ldvl,
                       double *vr, int *ldvr, double *work, int *lwork,
                       int *info) __attribute__((weak));

static void ipt_prepared_csc_init(IPTPreparedCsc *prepared)
{
    if (prepared == NULL) {
        return;
    }

    prepared->n = 0;
    prepared->nnz = 0;
    prepared->sorted = 0;
    prepared->hermitian = 0;
    prepared->lifted_degeneracies = 0;
    prepared->rotation_count = 0;
    prepared->rotation_values_count = 0;
    prepared->permuted = 0;
    prepared->col_ptr = NULL;
    prepared->row_ind = NULL;
    prepared->values = NULL;
    prepared->diagonal = NULL;
    prepared->rotation_starts = NULL;
    prepared->rotation_sizes = NULL;
    prepared->rotation_values = NULL;
    prepared->basis_to_original = NULL;
}

static void ipt_prepared_csc_free(IPTPreparedCsc *prepared)
{
    if (prepared == NULL) {
        return;
    }

    free(prepared->col_ptr);
    free(prepared->row_ind);
    free(prepared->values);
    free(prepared->diagonal);
    free(prepared->rotation_starts);
    free(prepared->rotation_sizes);
    free(prepared->rotation_values);
    free(prepared->basis_to_original);
    ipt_prepared_csc_init(prepared);
}

static int ipt_preparation_debug_enabled()
{
    const char *raw = getenv("IPT_DEBUG_PREPARATION");

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

static double ipt_preparation_env_double(const char *name, double default_value)
{
    const char *raw = getenv(name);
    char *end = NULL;
    double value = 0.0;

    if (raw == NULL || raw[0] == '\0') {
        return default_value;
    }
    value = strtod(raw, &end);
    if (end == raw || value < 0.0 || !std::isfinite(value)) {
        return default_value;
    }
    return value;
}

static int ipt_preparation_env_int(const char *name, int default_value)
{
    const char *raw = getenv(name);
    char *end = NULL;
    long value = 0;

    if (raw == NULL || raw[0] == '\0') {
        return default_value;
    }
    value = strtol(raw, &end, 10);
    if (end == raw || value <= 0 || value > std::numeric_limits<int>::max()) {
        return default_value;
    }
    return (int)value;
}

typedef struct {
    double absolute_gap;
    double eps_multiplier;
    double rel_gap_tol;
    double coupling_gap_ratio_tol;
    double eta;
    int max_block_size;
    int max_lift_rounds;
} IPTDegeneracyOptions;

static IPTDegeneracyOptions
ipt_degeneracy_options(double degeneracy_threshold)
{
    IPTDegeneracyOptions options;

    options.absolute_gap = degeneracy_threshold > 0.0 ? degeneracy_threshold : 0.0;
    options.eps_multiplier = ipt_preparation_env_double(
        "IPT_EPS_MULTIPLIER", IPT_DEFAULT_EPS_MULTIPLIER);
    options.rel_gap_tol = ipt_preparation_env_double(
        "IPT_REL_GAP_TOL", IPT_DEFAULT_REL_GAP_TOL);
    options.coupling_gap_ratio_tol = ipt_preparation_env_double(
        "IPT_COUPLING_GAP_RATIO_TOL",
        IPT_DEFAULT_COUPLING_GAP_RATIO_TOL);
    options.eta = ipt_preparation_env_double(
        "IPT_COUPLING_ETA", IPT_DEFAULT_COUPLING_ETA);
    options.max_block_size = ipt_preparation_env_int(
        "IPT_MAX_LIFT_BLOCK_SIZE", IPT_DEFAULT_MAX_BLOCK_SIZE);
    options.max_lift_rounds = ipt_preparation_env_int(
        "IPT_MAX_LIFT_ROUNDS", IPT_DEFAULT_MAX_LIFT_ROUNDS);
    return options;
}

static int ipt_validate_csc(const int *col_ptr, const int *row_ind, int n,
                            int nnz)
{
    if (col_ptr == NULL || row_ind == NULL || n <= 0 || nnz < 0 ||
        col_ptr[0] != 0 || col_ptr[n] != nnz) {
        return 0;
    }

    for (int col = 0; col < n; ++col) {
        if (col_ptr[col] > col_ptr[col + 1]) {
            return 0;
        }
    }

    for (int p = 0; p < nnz; ++p) {
        if (row_ind[p] < 0 || row_ind[p] >= n) {
            return 0;
        }
    }

    return 1;
}

static int ipt_extract_csc_diagonal_host(const int *col_ptr,
                                         const int *row_ind,
                                         const double *values, int n,
                                         std::vector<double> *diagonal)
{
    diagonal->assign((size_t)n, 0.0);

    for (int col = 0; col < n; ++col) {
        for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
            if (row_ind[p] == col) {
                (*diagonal)[(size_t)col] = values[p];
                break;
            }
        }
    }

    return 1;
}

static void ipt_build_stable_sort_permutation(const std::vector<double> &diag,
                                              std::vector<int> *perm,
                                              std::vector<int> *inverse_perm)
{
    int n = (int)diag.size();
    perm->resize((size_t)n);
    inverse_perm->resize((size_t)n);

    for (int i = 0; i < n; ++i) {
        (*perm)[(size_t)i] = i;
    }

    std::stable_sort(perm->begin(), perm->end(),
                     [&diag](int a, int b) { return diag[(size_t)a] < diag[(size_t)b]; });

    for (int sorted = 0; sorted < n; ++sorted) {
        (*inverse_perm)[(size_t)(*perm)[(size_t)sorted]] = sorted;
    }
}

static int ipt_permute_csc(const int *col_ptr, const int *row_ind,
                           const double *values, int n,
                           const std::vector<int> &inverse_perm,
                           std::vector<int> *out_col_ptr,
                           std::vector<int> *out_row_ind,
                           std::vector<double> *out_values)
{
    std::vector<std::vector<std::pair<int, double>>> columns((size_t)n);

    for (int old_col = 0; old_col < n; ++old_col) {
        int new_col = inverse_perm[(size_t)old_col];

        for (int p = col_ptr[old_col]; p < col_ptr[old_col + 1]; ++p) {
            int new_row = inverse_perm[(size_t)row_ind[p]];
            columns[(size_t)new_col].push_back(
                std::make_pair(new_row, values[p]));
        }
    }

    out_col_ptr->assign((size_t)n + 1U, 0);
    out_row_ind->clear();
    out_values->clear();

    for (int col = 0; col < n; ++col) {
        std::vector<std::pair<int, double>> &entries = columns[(size_t)col];

        std::sort(entries.begin(), entries.end(),
                  [](const std::pair<int, double> &a,
                     const std::pair<int, double> &b) {
                      return a.first < b.first;
                  });

        (*out_col_ptr)[(size_t)col] = (int)out_row_ind->size();

        for (size_t i = 0; i < entries.size(); ++i) {
            int row = entries[i].first;
            double value = entries[i].second;

            while (i + 1U < entries.size() && entries[i + 1U].first == row) {
                ++i;
                value += entries[i].second;
            }

            if (value != 0.0) {
                out_row_ind->push_back(row);
                out_values->push_back(value);
            }
        }
    }

    (*out_col_ptr)[(size_t)n] = (int)out_row_ind->size();
    return 1;
}

static std::vector<std::pair<int, int>>
ipt_degenerate_subspaces(const std::vector<double> &diag, int k,
                         double threshold)
{
    std::vector<std::pair<int, int>> subspaces;
    int n = (int)diag.size();
    int head = 0;
    int tail = 0;
    int degenerate = 0;

    while (tail <= k - 2 && tail + 1 < n) {
        if (fabs(diag[(size_t)tail] - diag[(size_t)(tail + 1)]) < threshold) {
            degenerate = 1;
            ++tail;
        } else {
            if (degenerate) {
                subspaces.push_back(std::make_pair(head, tail));
            }
            degenerate = 0;
            head = tail = tail + 1;
        }
    }

    if (degenerate) {
        subspaces.push_back(std::make_pair(head, std::min(tail, k - 1)));
    }

    return subspaces;
}

typedef struct {
    std::vector<int> parent;
    std::vector<int> rank;
} IPTUnionFind;

static void ipt_uf_init(IPTUnionFind *uf, int n)
{
    uf->parent.resize((size_t)n);
    uf->rank.assign((size_t)n, 0);
    for (int i = 0; i < n; ++i) {
        uf->parent[(size_t)i] = i;
    }
}

static int ipt_uf_find(IPTUnionFind *uf, int x)
{
    int parent = uf->parent[(size_t)x];
    if (parent != x) {
        parent = ipt_uf_find(uf, parent);
        uf->parent[(size_t)x] = parent;
    }
    return parent;
}

static void ipt_uf_union(IPTUnionFind *uf, int a, int b)
{
    int root_a = ipt_uf_find(uf, a);
    int root_b = ipt_uf_find(uf, b);

    if (root_a == root_b) {
        return;
    }
    if (uf->rank[(size_t)root_a] < uf->rank[(size_t)root_b]) {
        std::swap(root_a, root_b);
    }
    uf->parent[(size_t)root_b] = root_a;
    if (uf->rank[(size_t)root_a] == uf->rank[(size_t)root_b]) {
        ++uf->rank[(size_t)root_a];
    }
}

static double ipt_max_abs_diagonal(const std::vector<double> &diag)
{
    double max_abs = 0.0;

    for (double value : diag) {
        max_abs = std::max(max_abs, fabs(value));
    }
    return max_abs;
}

static double ipt_threshold_num(const std::vector<double> &diag,
                                const IPTDegeneracyOptions &options)
{
    double diag_scale = std::max(1.0, ipt_max_abs_diagonal(diag));
    return options.eps_multiplier *
           std::numeric_limits<double>::epsilon() * diag_scale;
}

static void ipt_csc_offdiag_strengths(const std::vector<int> &col_ptr,
                                      const std::vector<int> &row_ind,
                                      const std::vector<double> &values, int n,
                                      std::vector<double> *strength)
{
    std::vector<double> col_strength((size_t)n, 0.0);
    std::vector<double> row_strength((size_t)n, 0.0);

    strength->assign((size_t)n, 0.0);
    for (int col = 0; col < n; ++col) {
        for (int p = col_ptr[(size_t)col]; p < col_ptr[(size_t)col + 1U];
             ++p) {
            int row = row_ind[(size_t)p];
            double abs_value = fabs(values[(size_t)p]);

            if (row == col) {
                continue;
            }
            col_strength[(size_t)col] =
                std::max(col_strength[(size_t)col], abs_value);
            row_strength[(size_t)row] =
                std::max(row_strength[(size_t)row], abs_value);
        }
    }
    for (int i = 0; i < n; ++i) {
        (*strength)[(size_t)i] =
            std::max(col_strength[(size_t)i], row_strength[(size_t)i]);
    }
}

static double ipt_csc_find_value(const std::vector<int> &col_ptr,
                                 const std::vector<int> &row_ind,
                                 const std::vector<double> &values,
                                 int row, int col)
{
    for (int p = col_ptr[(size_t)col]; p < col_ptr[(size_t)col + 1U]; ++p) {
        if (row_ind[(size_t)p] == row) {
            return values[(size_t)p];
        }
    }
    return 0.0;
}

static double ipt_coupling_strength(const std::vector<int> &col_ptr,
                                    const std::vector<int> &row_ind,
                                    const std::vector<double> &values,
                                    const std::vector<double> &offdiag,
                                    const IPTDegeneracyOptions &options,
                                    int row, int col)
{
    double a_row_col = ipt_csc_find_value(col_ptr, row_ind, values, row, col);
    double a_col_row = ipt_csc_find_value(col_ptr, row_ind, values, col, row);
    double direct = std::max(fabs(a_row_col), fabs(a_col_row));

    if (direct > 0.0) {
        return direct;
    }
    return options.eta *
           std::min(offdiag[(size_t)row], offdiag[(size_t)col]);
}

static double ipt_numeric_gap_floor(double d_row, double d_col,
                                    double tau_num,
                                    const IPTDegeneracyOptions &options)
{
    double scale = std::max(1.0, std::max(fabs(d_row), fabs(d_col)));
    double tau = options.absolute_gap;

    tau = std::max(tau, tau_num);
    tau = std::max(tau, options.rel_gap_tol * scale);
    return tau;
}

static double ipt_target_window_threshold(double d_col, double tau_num,
                                          double delta_col,
                                          const IPTDegeneracyOptions &options)
{
    double tau = ipt_numeric_gap_floor(d_col, d_col, tau_num, options);
    double ratio_tol =
        std::max(options.coupling_gap_ratio_tol,
                 std::numeric_limits<double>::epsilon());

    /*
     * Candidate filtering only. If a pair could satisfy coupling/gap >=
     * ratio_tol, it must fall inside delta_col / ratio_tol.
     */
    tau = std::max(tau, delta_col / ratio_tol);
    return tau;
}

static int ipt_pairwise_should_union(double d_row, double d_col,
                                     double tau_num, double coupling,
                                     const IPTDegeneracyOptions &options,
                                     double *gap_out,
                                     double *floor_out,
                                     double *ratio_out,
                                     double *allowed_out)
{
    double gap = fabs(d_row - d_col);
    double gap_floor =
        ipt_numeric_gap_floor(d_row, d_col, tau_num, options);
    double denominator = std::max(gap, gap_floor);
    double ratio = coupling > 0.0 ? coupling / denominator : 0.0;
    double ratio_tol =
        std::max(options.coupling_gap_ratio_tol,
                 std::numeric_limits<double>::epsilon());
    double allowed = std::max(gap_floor, coupling / ratio_tol);

    if (gap_out != NULL) {
        *gap_out = gap;
    }
    if (floor_out != NULL) {
        *floor_out = gap_floor;
    }
    if (ratio_out != NULL) {
        *ratio_out = ratio;
    }
    if (allowed_out != NULL) {
        *allowed_out = allowed;
    }

    if (gap <= gap_floor) {
        return 1;
    }
    return coupling > 0.0 && ratio >= ratio_tol;
}

static std::vector<std::pair<int, int>> ipt_merge_ranges(
    std::vector<std::pair<int, int>> ranges)
{
    std::vector<std::pair<int, int>> merged;

    if (ranges.empty()) {
        return merged;
    }
    std::sort(ranges.begin(), ranges.end());
    merged.push_back(ranges[0]);
    for (size_t i = 1; i < ranges.size(); ++i) {
        std::pair<int, int> &last = merged.back();
        if (ranges[i].first <= last.second + 1) {
            last.second = std::max(last.second, ranges[i].second);
        } else {
            merged.push_back(ranges[i]);
        }
    }
    return merged;
}

static std::vector<std::pair<int, int>>
ipt_target_degenerate_subspaces(const std::vector<int> &col_ptr,
                                const std::vector<int> &row_ind,
                                const std::vector<double> &values,
                                const std::vector<double> &diag, int k,
                                const IPTDegeneracyOptions &options,
                                int *too_large)
{
    int n = (int)diag.size();
    double tau_num = ipt_threshold_num(diag, options);
    std::vector<double> offdiag;
    std::vector<int> sorted_order((size_t)n, 0);
    std::vector<double> sorted_diag((size_t)n, 0.0);
    IPTUnionFind uf;
    std::vector<int> component_min((size_t)n, n);
    std::vector<int> component_max((size_t)n, -1);
    std::vector<int> component_has_target((size_t)n, 0);
    std::vector<std::pair<int, int>> ranges;
    int debug = ipt_preparation_debug_enabled();
    int debug_union_printed = 0;
    int debug_union_count = 0;
    int debug_candidate_count = 0;
    double debug_max_ratio = 0.0;
    int debug_max_ratio_col = -1;
    int debug_max_ratio_row = -1;
    double debug_max_ratio_gap = 0.0;
    double debug_max_ratio_coupling = 0.0;

    *too_large = 0;
    ipt_csc_offdiag_strengths(col_ptr, row_ind, values, n, &offdiag);
    for (int i = 0; i < n; ++i) {
        sorted_order[(size_t)i] = i;
    }
    std::stable_sort(sorted_order.begin(), sorted_order.end(),
                     [&](int a, int b) {
                         if (diag[(size_t)a] < diag[(size_t)b]) {
                             return true;
                         }
                         if (diag[(size_t)b] < diag[(size_t)a]) {
                             return false;
                         }
                         return a < b;
                     });
    for (int i = 0; i < n; ++i) {
        sorted_diag[(size_t)i] = diag[(size_t)sorted_order[(size_t)i]];
    }

    ipt_uf_init(&uf, n);
    for (int col = 0; col < k; ++col) {
        double d_col = diag[(size_t)col];
        double tau_c = ipt_target_window_threshold(
            d_col, tau_num, offdiag[(size_t)col], options);
        std::vector<double>::const_iterator lower = std::lower_bound(
            sorted_diag.begin(), sorted_diag.end(), d_col - tau_c);
        std::vector<double>::const_iterator upper = std::upper_bound(
            sorted_diag.begin(), sorted_diag.end(), d_col + tau_c);
        int first = (int)(lower - sorted_diag.begin());
        int last = (int)(upper - sorted_diag.begin());

        for (int pos = first; pos < last; ++pos) {
            int row = sorted_order[(size_t)pos];
            double d_row = diag[(size_t)row];
            double coupling = 0.0;
            double gap = 0.0;
            double gap_floor = 0.0;
            double ratio = 0.0;
            double allowed = 0.0;
            int should_union = 0;

            if (row == col) {
                continue;
            }
            ++debug_candidate_count;
            coupling = ipt_coupling_strength(col_ptr, row_ind, values, offdiag,
                                             options, row, col);
            should_union = ipt_pairwise_should_union(
                d_row, d_col, tau_num, coupling, options, &gap, &gap_floor,
                &ratio, &allowed);
            if (ratio > debug_max_ratio) {
                debug_max_ratio = ratio;
                debug_max_ratio_col = col;
                debug_max_ratio_row = row;
                debug_max_ratio_gap = gap;
                debug_max_ratio_coupling = coupling;
            }
            if (should_union) {
                ipt_uf_union(&uf, row, col);
                ++debug_union_count;
                if (debug && debug_union_printed < 64) {
                    printf("[IPT preparation] union target=%d row=%d "
                           "gap=%.17g gap_floor=%.17g coupling=%.17g "
                           "coupling_gap_ratio=%.17g ratio_tol=%.17g "
                           "allowed_gap=%.17g\n",
                           col, row, gap, gap_floor, coupling, ratio,
                           options.coupling_gap_ratio_tol, allowed);
                    ++debug_union_printed;
                }
            }
        }
    }
    if (debug) {
        printf("[IPT preparation] target_vs_all_candidates=%d unions=%d "
               "max_coupling_gap_ratio=%.17g at target=%d row=%d "
               "gap=%.17g coupling=%.17g\n",
               debug_candidate_count, debug_union_count, debug_max_ratio,
               debug_max_ratio_col, debug_max_ratio_row, debug_max_ratio_gap,
               debug_max_ratio_coupling);
        if (debug_union_count > debug_union_printed) {
            printf("[IPT preparation] union log truncated after %d pairs\n",
                   debug_union_printed);
        }
        fflush(stdout);
    }

    for (int i = 0; i < n; ++i) {
        int root = ipt_uf_find(&uf, i);
        component_min[(size_t)root] =
            std::min(component_min[(size_t)root], i);
        component_max[(size_t)root] =
            std::max(component_max[(size_t)root], i);
        if (i < k) {
            component_has_target[(size_t)root] = 1;
        }
    }

    for (int root = 0; root < n; ++root) {
        if (!component_has_target[(size_t)root]) {
            continue;
        }
        if (component_min[(size_t)root] >= component_max[(size_t)root]) {
            continue;
        }
        ranges.push_back(std::make_pair(component_min[(size_t)root],
                                        component_max[(size_t)root]));
    }
    ranges = ipt_merge_ranges(ranges);

    for (size_t i = 0; i < ranges.size(); ++i) {
        int size = ranges[i].second - ranges[i].first + 1;
        if (size > options.max_block_size) {
            *too_large = 1;
        }
    }
    return ranges;
}

typedef struct {
    int bad_count;
    double min_abs_gap;
    int first_col;
    int first_row;
    double first_gap;
    double first_allowed;
    double first_coupling;
    double first_ratio;
} IPTGapCheck;

static IPTGapCheck ipt_validate_target_gaps(
    const std::vector<int> &col_ptr, const std::vector<int> &row_ind,
    const std::vector<double> &values, const std::vector<double> &diag, int k,
    const IPTDegeneracyOptions &options)
{
    int n = (int)diag.size();
    double tau_num = ipt_threshold_num(diag, options);
    std::vector<double> offdiag;
    std::vector<int> sorted_order((size_t)n, 0);
    std::vector<int> sorted_rank((size_t)n, 0);
    std::vector<double> sorted_diag((size_t)n, 0.0);
    IPTGapCheck check;

    check.bad_count = 0;
    check.min_abs_gap = std::numeric_limits<double>::infinity();
    check.first_col = -1;
    check.first_row = -1;
    check.first_gap = 0.0;
    check.first_allowed = 0.0;
    check.first_coupling = 0.0;
    check.first_ratio = 0.0;

    ipt_csc_offdiag_strengths(col_ptr, row_ind, values, n, &offdiag);
    for (int i = 0; i < n; ++i) {
        sorted_order[(size_t)i] = i;
    }
    std::stable_sort(sorted_order.begin(), sorted_order.end(),
                     [&](int a, int b) {
                         if (diag[(size_t)a] < diag[(size_t)b]) {
                             return true;
                         }
                         if (diag[(size_t)b] < diag[(size_t)a]) {
                             return false;
                         }
                         return a < b;
                     });
    for (int i = 0; i < n; ++i) {
        sorted_diag[(size_t)i] = diag[(size_t)sorted_order[(size_t)i]];
        sorted_rank[(size_t)sorted_order[(size_t)i]] = i;
    }

    for (int col = 0; col < k; ++col) {
        double d_col = diag[(size_t)col];
        double window = ipt_target_window_threshold(
            d_col, tau_num, offdiag[(size_t)col], options);
        int rank = sorted_rank[(size_t)col];

        if (rank > 0) {
            int row = sorted_order[(size_t)(rank - 1)];
            check.min_abs_gap =
                std::min(check.min_abs_gap,
                         fabs(d_col - diag[(size_t)row]));
        }
        if (rank + 1 < n) {
            int row = sorted_order[(size_t)(rank + 1)];
            check.min_abs_gap =
                std::min(check.min_abs_gap,
                         fabs(d_col - diag[(size_t)row]));
        }

        std::vector<double>::const_iterator lower = std::lower_bound(
            sorted_diag.begin(), sorted_diag.end(), d_col - window);
        std::vector<double>::const_iterator upper = std::upper_bound(
            sorted_diag.begin(), sorted_diag.end(), d_col + window);
        int first = (int)(lower - sorted_diag.begin());
        int last = (int)(upper - sorted_diag.begin());

        for (int pos = first; pos < last; ++pos) {
            int row = sorted_order[(size_t)pos];
            double d_row = diag[(size_t)row];
            double coupling = 0.0;
            double gap = 0.0;
            double gap_floor = 0.0;
            double ratio = 0.0;
            double allowed = 0.0;

            if (row == col) {
                continue;
            }
            coupling = ipt_coupling_strength(col_ptr, row_ind, values, offdiag,
                                             options, row, col);
            if (!ipt_pairwise_should_union(d_row, d_col, tau_num, coupling,
                                           options, &gap, &gap_floor, &ratio,
                                           &allowed)) {
                continue;
            }
            check.min_abs_gap = std::min(check.min_abs_gap, gap);
            ++check.bad_count;
            if (check.first_col < 0) {
                check.first_col = col;
                check.first_row = row;
                check.first_gap = gap;
                check.first_allowed = allowed;
                check.first_coupling = coupling;
                check.first_ratio = ratio;
            }
        }
    }
    return check;
}

static void ipt_jacobi_eigenvectors_symmetric(const std::vector<double> &a,
                                               int m,
                                               std::vector<double> *vectors)
{
    std::vector<double> work = a;
    const int max_sweeps = 64 * m * m;
    const double eps = 1.0e-14;

    vectors->assign((size_t)m * (size_t)m, 0.0);
    for (int i = 0; i < m; ++i) {
        (*vectors)[(size_t)i + (size_t)i * (size_t)m] = 1.0;
    }

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        int p = 0;
        int q = 1;
        double max_offdiag = 0.0;

        for (int col = 1; col < m; ++col) {
            for (int row = 0; row < col; ++row) {
                double value = fabs(work[(size_t)row + (size_t)col * (size_t)m]);
                if (value > max_offdiag) {
                    max_offdiag = value;
                    p = row;
                    q = col;
                }
            }
        }

        if (max_offdiag <= eps) {
            break;
        }

        double app = work[(size_t)p + (size_t)p * (size_t)m];
        double aqq = work[(size_t)q + (size_t)q * (size_t)m];
        double apq = work[(size_t)p + (size_t)q * (size_t)m];
        double tau = (aqq - app) / (2.0 * apq);
        double t = ((tau >= 0.0) ? 1.0 : -1.0) /
                   (fabs(tau) + sqrt(1.0 + tau * tau));
        double c = 1.0 / sqrt(1.0 + t * t);
        double s = t * c;

        for (int r = 0; r < m; ++r) {
            if (r != p && r != q) {
                double arp = work[(size_t)r + (size_t)p * (size_t)m];
                double arq = work[(size_t)r + (size_t)q * (size_t)m];
                double new_rp = c * arp - s * arq;
                double new_rq = s * arp + c * arq;
                work[(size_t)r + (size_t)p * (size_t)m] = new_rp;
                work[(size_t)p + (size_t)r * (size_t)m] = new_rp;
                work[(size_t)r + (size_t)q * (size_t)m] = new_rq;
                work[(size_t)q + (size_t)r * (size_t)m] = new_rq;
            }
        }

        work[(size_t)p + (size_t)p * (size_t)m] =
            c * c * app - 2.0 * s * c * apq + s * s * aqq;
        work[(size_t)q + (size_t)q * (size_t)m] =
            s * s * app + 2.0 * s * c * apq + c * c * aqq;
        work[(size_t)p + (size_t)q * (size_t)m] = 0.0;
        work[(size_t)q + (size_t)p * (size_t)m] = 0.0;

        for (int r = 0; r < m; ++r) {
            double vrp = (*vectors)[(size_t)r + (size_t)p * (size_t)m];
            double vrq = (*vectors)[(size_t)r + (size_t)q * (size_t)m];
            (*vectors)[(size_t)r + (size_t)p * (size_t)m] = c * vrp - s * vrq;
            (*vectors)[(size_t)r + (size_t)q * (size_t)m] = s * vrp + c * vrq;
        }
    }
}

static double ipt_csc_value_at(const std::vector<int> &col_ptr,
                               const std::vector<int> &row_ind,
                               const std::vector<double> &values, int row,
                               int col)
{
    double value = 0.0;

    for (int p = col_ptr[(size_t)col]; p < col_ptr[(size_t)(col + 1)]; ++p) {
        if (row_ind[(size_t)p] == row) {
            value += values[(size_t)p];
        }
    }

    return value;
}

static void ipt_extract_csc_diagonal_vectors(const std::vector<int> &col_ptr,
                                             const std::vector<int> &row_ind,
                                             const std::vector<double> &values,
                                             int n,
                                             std::vector<double> *diagonal)
{
    diagonal->assign((size_t)n, 0.0);

    for (int col = 0; col < n; ++col) {
        (*diagonal)[(size_t)col] =
            ipt_csc_value_at(col_ptr, row_ind, values, col, col);
    }
}

static int ipt_csc_is_hermitian_real(const std::vector<int> &col_ptr,
                                     const std::vector<int> &row_ind,
                                     const std::vector<double> &values, int n)
{
    const double tol = 1.0e-12;

    for (int col = 0; col < n; ++col) {
        for (int p = col_ptr[(size_t)col]; p < col_ptr[(size_t)(col + 1)];
             ++p) {
            int row = row_ind[(size_t)p];
            double value = values[(size_t)p];
            double transpose_value =
                ipt_csc_value_at(col_ptr, row_ind, values, col, row);
            double scale = std::max(1.0, std::max(fabs(value),
                                                  fabs(transpose_value)));

            if (fabs(value - transpose_value) > tol * scale) {
                return 0;
            }
        }
    }

    return 1;
}

static void ipt_extract_csc_block(const std::vector<int> &col_ptr,
                                  const std::vector<int> &row_ind,
                                  const std::vector<double> &values, int first,
                                  int m, std::vector<double> *block)
{
    block->assign((size_t)m * (size_t)m, 0.0);

    for (int col = 0; col < m; ++col) {
        int global_col = first + col;

        for (int p = col_ptr[(size_t)global_col];
             p < col_ptr[(size_t)(global_col + 1)]; ++p) {
            int global_row = row_ind[(size_t)p];

            if (global_row >= first && global_row < first + m) {
                int row = global_row - first;
                (*block)[(size_t)row + (size_t)col * (size_t)m] +=
                    values[(size_t)p];
            }
        }
    }
}

static void ipt_transpose_square(const std::vector<double> &a, int m,
                                 std::vector<double> *transpose)
{
    transpose->assign((size_t)m * (size_t)m, 0.0);

    for (int col = 0; col < m; ++col) {
        for (int row = 0; row < m; ++row) {
            (*transpose)[(size_t)row + (size_t)col * (size_t)m] =
                a[(size_t)col + (size_t)row * (size_t)m];
        }
    }
}

static void ipt_sort_eigenvectors_by_values(std::vector<double> *vectors,
                                            std::vector<double> *eigenvalues,
                                            int m)
{
    std::vector<int> order((size_t)m, 0);
    std::vector<double> sorted_vectors((size_t)m * (size_t)m, 0.0);
    std::vector<double> sorted_values((size_t)m, 0.0);

    for (int i = 0; i < m; ++i) {
        order[(size_t)i] = i;
    }

    std::stable_sort(order.begin(), order.end(),
                     [&eigenvalues](int a, int b) {
                         return (*eigenvalues)[(size_t)a] <
                                (*eigenvalues)[(size_t)b];
                     });

    for (int new_col = 0; new_col < m; ++new_col) {
        int old_col = order[(size_t)new_col];
        sorted_values[(size_t)new_col] = (*eigenvalues)[(size_t)old_col];

        for (int row = 0; row < m; ++row) {
            sorted_vectors[(size_t)row + (size_t)new_col * (size_t)m] =
                (*vectors)[(size_t)row + (size_t)old_col * (size_t)m];
        }
    }

    vectors->swap(sorted_vectors);
    eigenvalues->swap(sorted_values);
}

static void ipt_symmetric_eigenvectors_sorted(const std::vector<double> &block,
                                              int m,
                                              std::vector<double> *vectors)
{
    std::vector<double> eigenvalues((size_t)m, 0.0);

    ipt_jacobi_eigenvectors_symmetric(block, m, vectors);

    for (int col = 0; col < m; ++col) {
        double value = 0.0;

        for (int row = 0; row < m; ++row) {
            for (int inner = 0; inner < m; ++inner) {
                value += (*vectors)[(size_t)row + (size_t)col * (size_t)m] *
                         block[(size_t)row + (size_t)inner * (size_t)m] *
                         (*vectors)[(size_t)inner +
                                    (size_t)col * (size_t)m];
            }
        }

        eigenvalues[(size_t)col] = value;
    }

    ipt_sort_eigenvectors_by_values(vectors, &eigenvalues, m);
}

static int ipt_general_real_eigenvectors_sorted(const std::vector<double> &block,
                                                int m,
                                                std::vector<double> *vectors)
{
    if (dgeev_ == NULL) {
        return 0;
    }

    std::vector<double> a = block;
    std::vector<double> wr((size_t)m, 0.0);
    std::vector<double> wi((size_t)m, 0.0);
    std::vector<double> vl(1U, 0.0);
    std::vector<double> vr((size_t)m * (size_t)m, 0.0);
    double work_query = 0.0;
    char jobvl = 'N';
    char jobvr = 'V';
    int n_local = m;
    int lda = m;
    int ldvl = 1;
    int ldvr = m;
    int lwork = -1;
    int info = 0;

    dgeev_(&jobvl, &jobvr, &n_local, a.data(), &lda, wr.data(), wi.data(),
           vl.data(), &ldvl, vr.data(), &ldvr, &work_query, &lwork, &info);
    if (info != 0 || work_query < 1.0) {
        return 0;
    }

    lwork = std::max(1, (int)ceil(work_query));
    std::vector<double> work((size_t)lwork, 0.0);
    a = block;
    info = 0;

    dgeev_(&jobvl, &jobvr, &n_local, a.data(), &lda, wr.data(), wi.data(),
           vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, &info);
    if (info != 0) {
        return 0;
    }

    for (int i = 0; i < m; ++i) {
        double scale = std::max(1.0, fabs(wr[(size_t)i]));

        if (fabs(wi[(size_t)i]) > 1.0e-12 * scale) {
            return 0;
        }
    }

    vectors->swap(vr);
    ipt_sort_eigenvectors_by_values(vectors, &wr, m);
    return 1;
}

static int ipt_inverse_square(const std::vector<double> &a, int m,
                              std::vector<double> *inverse)
{
    int width = 2 * m;
    std::vector<double> aug((size_t)m * (size_t)width, 0.0);

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < m; ++col) {
            aug[(size_t)row * (size_t)width + (size_t)col] =
                a[(size_t)row + (size_t)col * (size_t)m];
        }
        aug[(size_t)row * (size_t)width + (size_t)(m + row)] = 1.0;
    }

    for (int col = 0; col < m; ++col) {
        int pivot = col;
        double pivot_abs =
            fabs(aug[(size_t)pivot * (size_t)width + (size_t)col]);

        for (int row = col + 1; row < m; ++row) {
            double candidate =
                fabs(aug[(size_t)row * (size_t)width + (size_t)col]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }

        if (pivot_abs <= 1.0e-14) {
            return 0;
        }

        if (pivot != col) {
            for (int j = 0; j < width; ++j) {
                std::swap(aug[(size_t)col * (size_t)width + (size_t)j],
                          aug[(size_t)pivot * (size_t)width + (size_t)j]);
            }
        }

        {
            double pivot_value =
                aug[(size_t)col * (size_t)width + (size_t)col];
            for (int j = 0; j < width; ++j) {
                aug[(size_t)col * (size_t)width + (size_t)j] /= pivot_value;
            }
        }

        for (int row = 0; row < m; ++row) {
            if (row == col) {
                continue;
            }

            double factor =
                aug[(size_t)row * (size_t)width + (size_t)col];
            if (factor == 0.0) {
                continue;
            }

            for (int j = 0; j < width; ++j) {
                aug[(size_t)row * (size_t)width + (size_t)j] -=
                    factor *
                    aug[(size_t)col * (size_t)width + (size_t)j];
            }
        }
    }

    inverse->assign((size_t)m * (size_t)m, 0.0);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < m; ++col) {
            (*inverse)[(size_t)row + (size_t)col * (size_t)m] =
                aug[(size_t)row * (size_t)width + (size_t)(m + col)];
        }
    }

    return 1;
}

static int ipt_columns_to_csc(
    std::vector<std::vector<std::pair<int, double>>> *columns,
    std::vector<int> *col_ptr, std::vector<int> *row_ind,
    std::vector<double> *values)
{
    int n = (int)columns->size();

    col_ptr->assign((size_t)n + 1U, 0);
    row_ind->clear();
    values->clear();

    for (int col = 0; col < n; ++col) {
        std::vector<std::pair<int, double>> &entries = (*columns)[(size_t)col];

        std::sort(entries.begin(), entries.end(),
                  [](const std::pair<int, double> &a,
                     const std::pair<int, double> &b) {
                      return a.first < b.first;
                  });

        (*col_ptr)[(size_t)col] = (int)row_ind->size();

        for (size_t i = 0; i < entries.size(); ++i) {
            int row = entries[i].first;
            double value = entries[i].second;

            while (i + 1U < entries.size() && entries[i + 1U].first == row) {
                ++i;
                value += entries[i].second;
            }

            if (value != 0.0) {
                if (row_ind->size() >=
                    (size_t)std::numeric_limits<int>::max()) {
                    return 0;
                }
                row_ind->push_back(row);
                values->push_back(value);
            }
        }
    }

    (*col_ptr)[(size_t)n] = (int)row_ind->size();
    return 1;
}

static int ipt_apply_local_similarity_csc(
    std::vector<int> *col_ptr, std::vector<int> *row_ind,
    std::vector<double> *values, int n, int first, int last,
    const std::vector<double> &right, const std::vector<double> &left)
{
    int m = last - first + 1;
    std::vector<std::vector<std::pair<int, double>>> columns((size_t)n);

    for (int col = 0; col < n; ++col) {
        int col_local = col - first;
        int col_in_block = (col_local >= 0 && col_local < m);

        for (int p = (*col_ptr)[(size_t)col];
             p < (*col_ptr)[(size_t)(col + 1)]; ++p) {
            int row = (*row_ind)[(size_t)p];
            int row_local = row - first;
            int row_in_block = (row_local >= 0 && row_local < m);
            double value = (*values)[(size_t)p];

            if (!row_in_block && !col_in_block) {
                columns[(size_t)col].push_back(std::make_pair(row, value));
            } else if (!row_in_block && col_in_block) {
                for (int j = 0; j < m; ++j) {
                    double new_value =
                        value *
                        right[(size_t)col_local + (size_t)j * (size_t)m];
                    if (new_value != 0.0) {
                        columns[(size_t)(first + j)].push_back(
                            std::make_pair(row, new_value));
                    }
                }
            } else if (row_in_block && !col_in_block) {
                for (int i = 0; i < m; ++i) {
                    double new_value =
                        left[(size_t)i + (size_t)row_local * (size_t)m] *
                        value;
                    if (new_value != 0.0) {
                        columns[(size_t)col].push_back(
                            std::make_pair(first + i, new_value));
                    }
                }
            } else {
                for (int j = 0; j < m; ++j) {
                    double right_value =
                        right[(size_t)col_local + (size_t)j * (size_t)m];
                    if (right_value == 0.0) {
                        continue;
                    }

                    for (int i = 0; i < m; ++i) {
                        double new_value =
                            left[(size_t)i +
                                 (size_t)row_local * (size_t)m] *
                            value * right_value;
                        if (new_value != 0.0) {
                            columns[(size_t)(first + j)].push_back(
                                std::make_pair(first + i, new_value));
                        }
                    }
                }
            }
        }
    }

    return ipt_columns_to_csc(&columns, col_ptr, row_ind, values);
}

static int ipt_apply_prepared_q_to_vectors(const IPTPreparedCsc *prepared,
                                           double *vectors_col_major, int k)
{
    int value_offset = prepared != NULL ? prepared->rotation_values_count : 0;

    if (prepared == NULL || vectors_col_major == NULL || k <= 0 ||
        prepared->n <= 0) {
        return 0;
    }

    for (int rotation = prepared->rotation_count - 1; rotation >= 0;
         --rotation) {
        int first = prepared->rotation_starts[rotation];
        int m = prepared->rotation_sizes[rotation];
        const double *q = NULL;
        std::vector<double> temp;

        if (first < 0 || m <= 0 || first + m > prepared->n ||
            value_offset - m * m < 0) {
            return 0;
        }
        value_offset -= m * m;
        q = prepared->rotation_values + value_offset;
        temp.assign((size_t)m, 0.0);

        for (int col = 0; col < k; ++col) {
            for (int row = 0; row < m; ++row) {
                double sum = 0.0;

                for (int inner = 0; inner < m; ++inner) {
                    sum += q[(size_t)row + (size_t)inner * (size_t)m] *
                           vectors_col_major[(size_t)(first + inner) +
                                             (size_t)col *
                                                 (size_t)prepared->n];
                }

                temp[(size_t)row] = sum;
            }

            for (int row = 0; row < m; ++row) {
                vectors_col_major[(size_t)(first + row) +
                                  (size_t)col * (size_t)prepared->n] =
                    temp[(size_t)row];
            }
        }
    }

    if (value_offset != 0) {
        return 0;
    }

    if (prepared->permuted && prepared->basis_to_original != NULL) {
        std::vector<double> temp((size_t)prepared->n, 0.0);

        for (int col = 0; col < k; ++col) {
            double *vector =
                vectors_col_major + (size_t)col * (size_t)prepared->n;

            std::fill(temp.begin(), temp.end(), 0.0);
            for (int basis = 0; basis < prepared->n; ++basis) {
                int original = prepared->basis_to_original[basis];

                if (original < 0 || original >= prepared->n) {
                    return 0;
                }
                temp[(size_t)original] = vector[basis];
            }
            memcpy(vector, temp.data(), (size_t)prepared->n * sizeof(double));
        }
    }

    return 1;
}

static int ipt_lift_csc_degeneracies(
    std::vector<int> *col_ptr, std::vector<int> *row_ind,
    std::vector<double> *values, std::vector<double> *diagonal, int n,
    int hermitian, const std::vector<std::pair<int, int>> &subspaces,
    std::vector<int> *rotation_starts, std::vector<int> *rotation_sizes,
    std::vector<double> *rotation_values)
{
    for (size_t s = 0; s < subspaces.size(); ++s) {
        int first = subspaces[s].first;
        int last = subspaces[s].second;
        int m = last - first + 1;
        std::vector<double> block((size_t)m * (size_t)m, 0.0);
        std::vector<double> vectors;
        std::vector<double> left;

        ipt_extract_csc_block(*col_ptr, *row_ind, *values, first, m, &block);

        if (hermitian) {
            ipt_symmetric_eigenvectors_sorted(block, m, &vectors);
            ipt_transpose_square(vectors, m, &left);
        } else {
            if (!ipt_general_real_eigenvectors_sorted(block, m, &vectors)) {
                return 1;
            }
            if (!ipt_inverse_square(vectors, m, &left)) {
                return 1;
            }
        }

        if (!ipt_apply_local_similarity_csc(col_ptr, row_ind, values, n, first,
                                            last, vectors, left)) {
            return 2;
        }

        rotation_starts->push_back(first);
        rotation_sizes->push_back(m);
        rotation_values->insert(rotation_values->end(), vectors.begin(),
                                vectors.end());
    }

    ipt_extract_csc_diagonal_vectors(*col_ptr, *row_ind, *values, n, diagonal);
    return 0;
}

static int ipt_copy_vector_to_malloc(const std::vector<int> &source, int **out)
{
    size_t bytes = source.size() * sizeof(int);
    if (bytes == 0) {
        *out = NULL;
        return 1;
    }
    *out = (int *)malloc(bytes);
    if (*out == NULL) {
        return 0;
    }
    if (bytes > 0) {
        memcpy(*out, source.data(), bytes);
    }
    return 1;
}

static int ipt_copy_vector_to_malloc(const std::vector<double> &source,
                                     double **out)
{
    size_t bytes = source.size() * sizeof(double);
    if (bytes == 0) {
        *out = NULL;
        return 1;
    }
    *out = (double *)malloc(bytes);
    if (*out == NULL) {
        return 0;
    }
    if (bytes > 0) {
        memcpy(*out, source.data(), bytes);
    }
    return 1;
}

static int ipt_prepare_sparse_csc(const int *col_ptr, const int *row_ind,
                                  const double *values, int n, int k,
                                  int nnz, int sort_diagonal,
                                  int lift_degeneracies,
                                  double degeneracy_threshold,
                                  IPTPreparedCsc *prepared)
{
    std::vector<double> diagonal;
    std::vector<int> work_col_ptr;
    std::vector<int> work_row_ind;
    std::vector<double> work_values;
    std::vector<int> perm;
    std::vector<int> inverse_perm;
    std::vector<int> basis_to_original;
    std::vector<int> rotation_starts;
    std::vector<int> rotation_sizes;
    std::vector<double> rotation_values;

    if (prepared == NULL) {
        return 1;
    }
    ipt_prepared_csc_init(prepared);

    if (col_ptr == NULL || row_ind == NULL || values == NULL || n <= 0 ||
        k <= 0 || k > n || nnz < 0 ||
        !ipt_validate_csc(col_ptr, row_ind, n, nnz)) {
        return 1;
    }

    ipt_extract_csc_diagonal_host(col_ptr, row_ind, values, n, &diagonal);

    if (sort_diagonal) {
        int changed = 0;

        ipt_build_stable_sort_permutation(diagonal, &perm, &inverse_perm);
        for (int i = 0; i < n; ++i) {
            if (perm[(size_t)i] != i) {
                changed = 1;
                break;
            }
        }

        if (changed) {
            if (!ipt_permute_csc(col_ptr, row_ind, values, n, inverse_perm,
                                 &work_col_ptr, &work_row_ind, &work_values)) {
                return 2;
            }

            std::vector<double> sorted_diagonal((size_t)n, 0.0);
            for (int i = 0; i < n; ++i) {
                sorted_diagonal[(size_t)i] = diagonal[(size_t)perm[(size_t)i]];
            }
            diagonal.swap(sorted_diagonal);
            prepared->sorted = 1;
            prepared->permuted = 1;
            basis_to_original = perm;
        }
    }

    if (work_col_ptr.empty()) {
        work_col_ptr.assign(col_ptr, col_ptr + (size_t)n + 1U);
        work_row_ind.assign(row_ind, row_ind + (size_t)nnz);
        work_values.assign(values, values + (size_t)nnz);
    }

    if (lift_degeneracies) {
        IPTDegeneracyOptions options =
            ipt_degeneracy_options(degeneracy_threshold);
        IPTGapCheck final_gap_check;
        int gap_clean = 0;

        prepared->hermitian = ipt_csc_is_hermitian_real(
            work_col_ptr, work_row_ind, work_values, n);
        if (ipt_preparation_debug_enabled()) {
            printf("[IPT preparation] n=%d k=%d abs_threshold=%.17g "
                   "eps_multiplier=%.17g rel_gap_tol=%.17g "
                   "coupling_gap_ratio_tol=%.17g eta=%.17g "
                   "max_block_size=%d max_lift_rounds=%d sorted_changed=%d "
                   "hermitian=%d\n",
                   n, k, options.absolute_gap, options.eps_multiplier,
                   options.rel_gap_tol, options.coupling_gap_ratio_tol,
                   options.eta, options.max_block_size,
                   options.max_lift_rounds, prepared->sorted,
                   prepared->hermitian);
            fflush(stdout);
        }

        for (int round = 0; round < options.max_lift_rounds; ++round) {
            int too_large = 0;
            std::vector<std::pair<int, int>> subspaces;

            ipt_extract_csc_diagonal_vectors(work_col_ptr, work_row_ind,
                                             work_values, n, &diagonal);
            subspaces = ipt_target_degenerate_subspaces(
                work_col_ptr, work_row_ind, work_values, diagonal, k, options,
                &too_large);

            if (ipt_preparation_debug_enabled()) {
                int count = std::min(n, 10);

                printf("[IPT preparation] round=%d target_vs_all_blocks=%zu\n",
                       round + 1, subspaces.size());
                for (int i = 0; i < count; ++i) {
                    if (i + 1 < count) {
                        printf("[IPT preparation] diag[%d]=%.17g "
                               "gap_to_next_index=%.17g\n",
                               i, diagonal[(size_t)i],
                               diagonal[(size_t)(i + 1)] -
                                   diagonal[(size_t)i]);
                    } else {
                        printf("[IPT preparation] diag[%d]=%.17g\n", i,
                               diagonal[(size_t)i]);
                    }
                }
                for (size_t s = 0; s < subspaces.size(); ++s) {
                    int first = subspaces[s].first;
                    int last = subspaces[s].second;
                    double max_gap = 0.0;

                    for (int i = first; i < last; ++i) {
                        max_gap = std::max(
                            max_gap, fabs(diagonal[(size_t)(i + 1)] -
                                          diagonal[(size_t)i]));
                    }
                    printf("[IPT preparation] block[%zu]=%d:%d size=%d "
                           "diag_first=%.17g diag_last=%.17g "
                           "max_adjacent_index_gap=%.17g\n",
                           s, first, last, last - first + 1,
                           diagonal[(size_t)first], diagonal[(size_t)last],
                           max_gap);
                }
                fflush(stdout);
            }

            if (too_large) {
                if (ipt_preparation_debug_enabled()) {
                    printf("[IPT preparation] near-degenerate block exceeds "
                           "max_block_size=%d; fallback eigensolver needed\n",
                           options.max_block_size);
                    fflush(stdout);
                }
                return 1;
            }

            if (subspaces.empty()) {
                final_gap_check =
                    ipt_validate_target_gaps(work_col_ptr, work_row_ind,
                                             work_values, diagonal, k,
                                             options);
                if (final_gap_check.bad_count == 0) {
                    gap_clean = 1;
                }
                break;
            }

            int lift_status = ipt_lift_csc_degeneracies(
                &work_col_ptr, &work_row_ind, &work_values, &diagonal, n,
                prepared->hermitian, subspaces, &rotation_starts,
                &rotation_sizes, &rotation_values);
            if (lift_status != 0) {
                if (ipt_preparation_debug_enabled()) {
                    printf("[IPT preparation] lift failed status=%d\n",
                           lift_status);
                    fflush(stdout);
                }
                return lift_status;
            }
            prepared->lifted_degeneracies = 1;

            final_gap_check = ipt_validate_target_gaps(
                work_col_ptr, work_row_ind, work_values, diagonal, k,
                options);
            if (ipt_preparation_debug_enabled()) {
                int count = std::min(n, std::min(k, 10));

                printf("[IPT preparation] round=%d lift succeeded "
                       "total_rotations=%zu bad_gaps=%d min_abs_gap=%.17g\n",
                       round + 1, rotation_starts.size(),
                       final_gap_check.bad_count,
                       final_gap_check.min_abs_gap);
                for (int i = 0; i < count; ++i) {
                    if (i + 1 < count) {
                        double gap =
                            diagonal[(size_t)(i + 1)] - diagonal[(size_t)i];
                        printf("[IPT preparation] post_lift_diag[%d]=%.17g "
                               "gap_to_next=%.17g\n",
                               i, diagonal[(size_t)i], gap);
                    } else {
                        printf("[IPT preparation] post_lift_diag[%d]=%.17g\n",
                               i, diagonal[(size_t)i]);
                    }
                }
                if (final_gap_check.bad_count > 0) {
                    printf("[IPT preparation] first_bad_gap col=%d row=%d "
                           "gap=%.17g allowed=%.17g coupling=%.17g "
                           "coupling_gap_ratio=%.17g\n",
                           final_gap_check.first_col,
                           final_gap_check.first_row,
                           final_gap_check.first_gap,
                           final_gap_check.first_allowed,
                           final_gap_check.first_coupling,
                           final_gap_check.first_ratio);
                }
                fflush(stdout);
            }
            if (final_gap_check.bad_count == 0) {
                gap_clean = 1;
                break;
            }
        }

        if (!gap_clean) {
            ipt_extract_csc_diagonal_vectors(work_col_ptr, work_row_ind,
                                             work_values, n, &diagonal);
            final_gap_check = ipt_validate_target_gaps(
                work_col_ptr, work_row_ind, work_values, diagonal, k,
                options);
            if (final_gap_check.bad_count > 0) {
                if (ipt_preparation_debug_enabled()) {
                    printf("[IPT preparation] unresolved target gap after "
                           "%d lift rounds: bad_gaps=%d min_abs_gap=%.17g "
                           "first_col=%d first_row=%d gap=%.17g "
                           "allowed=%.17g coupling=%.17g "
                           "coupling_gap_ratio=%.17g\n",
                           options.max_lift_rounds,
                           final_gap_check.bad_count,
                           final_gap_check.min_abs_gap,
                           final_gap_check.first_col,
                           final_gap_check.first_row,
                           final_gap_check.first_gap,
                           final_gap_check.first_allowed,
                           final_gap_check.first_coupling,
                           final_gap_check.first_ratio);
                    fflush(stdout);
                }
                return 1;
            }
        }
    }

    prepared->n = n;
    prepared->nnz = (int)work_row_ind.size();
    if (work_row_ind.size() > (size_t)std::numeric_limits<int>::max()) {
        ipt_prepared_csc_free(prepared);
        return 1;
    }

    if (rotation_starts.size() > (size_t)std::numeric_limits<int>::max() ||
        rotation_values.size() > (size_t)std::numeric_limits<int>::max()) {
        ipt_prepared_csc_free(prepared);
        return 1;
    }

    prepared->rotation_count = (int)rotation_starts.size();
    prepared->rotation_values_count = (int)rotation_values.size();
    prepared->permuted = !basis_to_original.empty();

    if (!ipt_copy_vector_to_malloc(work_col_ptr, &prepared->col_ptr) ||
        !ipt_copy_vector_to_malloc(work_row_ind, &prepared->row_ind) ||
        !ipt_copy_vector_to_malloc(work_values, &prepared->values) ||
        !ipt_copy_vector_to_malloc(diagonal, &prepared->diagonal) ||
        !ipt_copy_vector_to_malloc(rotation_starts,
                                   &prepared->rotation_starts) ||
        !ipt_copy_vector_to_malloc(rotation_sizes, &prepared->rotation_sizes) ||
        !ipt_copy_vector_to_malloc(rotation_values,
                                   &prepared->rotation_values) ||
        !ipt_copy_vector_to_malloc(basis_to_original,
                                   &prepared->basis_to_original)) {
        ipt_prepared_csc_free(prepared);
        return 2;
    }

    return 0;
}

#endif
