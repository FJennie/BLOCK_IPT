#ifndef IPT_PREPARATION_H
#define IPT_PREPARATION_H

typedef struct {
    int n;
    int nnz;
    int sorted;
    int hermitian;
    int lifted_degeneracies;
    int rotation_count;
    int rotation_values_count;
    int permuted;
    int *col_ptr;
    int *row_ind;
    double *values;
    double *diagonal;
    int *rotation_starts;
    int *rotation_sizes;
    double *rotation_values;
    int *basis_to_original;
} IPTPreparedCsc;

static void ipt_prepared_csc_init(IPTPreparedCsc *prepared);
static void ipt_prepared_csc_free(IPTPreparedCsc *prepared);
static int ipt_apply_prepared_q_to_vectors(const IPTPreparedCsc *prepared,
                                           double *vectors_col_major, int k);
static int ipt_prepare_sparse_csc(const int *col_ptr, const int *row_ind,
                                  const double *values, int n, int k,
                                  int nnz, int sort_diagonal,
                                  int lift_degeneracies,
                                  double degeneracy_threshold,
                                  IPTPreparedCsc *prepared);

#endif
