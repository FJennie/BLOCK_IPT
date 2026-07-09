#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/fs1/home/nudt_liujie/ftt/IPT_C_GPU")
TEST_ROOT = ROOT / "test/sparse/test_initial"
RESULT_ROOT = ROOT / "results/sparse/test_initial"
LOG_ROOT = ROOT / "logs/sparse/test_initial"
K_LIST = ("10", "20", "30")
NEAR_DIAGONAL_SUFFICIENT_RATIO = 3.0 - 2.0 * math.sqrt(2.0)


def read_kv_file(path):
    out = {}
    if not path.is_file():
        return out
    for raw in path.read_text(errors="replace").splitlines():
        if "\t" in raw:
            key, value = raw.split("\t", 1)
        elif "=" in raw:
            key, value = raw.split("=", 1)
        else:
            continue
        out[key.strip()] = value.strip()
    return out


def parse_summary(path):
    data = {}
    methods = []
    if not path.is_file():
        return data, methods
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("method="):
            fields = {}
            for part in line.split():
                if "=" in part:
                    key, value = part.split("=", 1)
                    fields[key] = value
            methods.append(fields)
        elif "=" in line:
            key, value = line.split("=", 1)
            data[key] = value
    return data, methods


def method(methods, name):
    return next((m for m in methods if m.get("method") == name), {})


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def read_log(path):
    if not path.is_file():
        return ""
    return path.read_text(errors="replace")


def first_match(pattern, text, default=""):
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else default


def max_coupling_ratio(summary):
    keys = [
        "max_coupling_over_gap_inside_first_30",
        "max_coupling_over_gap_between_first_30_and_31_40",
        "max_coupling_over_gap_between_block_and_outside",
    ]
    values = [to_float(summary.get(k)) for k in keys]
    finite = [v for v in values if math.isfinite(v)]
    return max(finite) if finite else math.nan


def classify(row, summary, ipt, log_text, status):
    if status.get("ipt_status") == "matrix_cache_missing":
        return "matrix_cache_missing"
    if not ipt and not Path(row["ipt_summary_path"]).is_file():
        return "no_ipt_result"
    if ipt.get("status") == "success":
        return "ok"

    lowered = log_text.lower()
    max_block_error = to_int(summary.get("max_block_size_error"))
    offending = to_int(summary.get("max_block_size_offending_block_size"))
    configured_max = row.get("ipt_max_lift_block_size", "")
    configured_number = to_int(first_match(r"([0-9]+)", configured_max))

    if max_block_error or "invalid argument at block detection" in lowered:
        if offending > 64 and (configured_number == 0 or configured_number <= 64):
            return "block_gt_64_retry_needed"
        return "block_gt_max"
    if "shared memory bytes" in lowered and "exceeds device" in lowered:
        return "block_gt_max"
    if "block solve failed" in lowered:
        return "riccati_singular_or_sep_small"
    if "singular" in lowered or "nan" in lowered and "block" in lowered:
        return "riccati_singular_or_sep_small"

    ratio = max_coupling_ratio(summary)
    if math.isfinite(ratio) and ratio > NEAR_DIAGONAL_SUFFICIENT_RATIO:
        return "not_near_diagonal"
    if "unresolved target gap" in lowered or "coupling_gap_ratio" in lowered:
        return "not_near_diagonal"
    if "out of memory" in lowered or "cudaerrormemoryallocation" in lowered:
        return "memory"
    if ipt.get("status") == "failed":
        return "not_converged"
    if ipt.get("status") == "api_error":
        return "api_error_other"
    return "other"


def bottleneck_notes(row):
    notes = []
    compute = to_float(row.get("ipt_compute_time_sec"))
    iteration = to_float(row.get("ipt_iteration_time_sec"))
    rr = to_float(row.get("ipt_rayleigh_ritz_time_sec"))
    davidson = to_float(row.get("ipt_davidson_time_sec"))
    residual = to_float(row.get("ipt_max_rel_residual"))
    ratio = to_float(row.get("max_coupling_over_gap"))
    failure = row.get("ipt_failure_class", "")
    basis_cols = to_int(row.get("ipt_basis_cols"))
    k = to_int(row.get("k"))
    restarts = to_int(row.get("ipt_restart_count"))
    target_block_size = to_int(row.get("target_block_size"))
    offending_block_size = to_int(row.get("max_block_size_offending_block_size"))

    if failure == "block_gt_64_retry_needed":
        notes.append(
            f"block size {offending_block_size} exceeds default 64; rerun with larger IPT_MAX_LIFT_BLOCK_SIZE or split the dangerous block"
        )
    elif failure == "block_gt_max":
        notes.append(
            f"dangerous block remains too large (offending={offending_block_size}); preparation needs a better split/fallback policy"
        )
    elif failure == "riccati_singular_or_sep_small":
        notes.append(
            "Riccati block solve hit a singular/near-SEP-small case; consider regularized denominators or alternative small-block solver"
        )
    elif failure == "not_near_diagonal":
        notes.append(
            f"coupling/gap ratio {ratio:.3g} exceeds the paper sufficient threshold {NEAR_DIAGONAL_SUFFICIENT_RATIO:.3g}; fixed-point IPT may be outside its useful domain"
        )
    elif failure == "not_converged":
        notes.append(
            f"residual stopped at {residual:.3g}; try deeper Davidson or adjust dangerous-block preparation before comparing performance"
        )

    if math.isfinite(compute) and compute > 0:
        phases = [
            ("Riccati", iteration),
            ("RR", rr),
            ("Davidson", davidson),
        ]
        phase, seconds = max(phases, key=lambda item: -math.inf if not math.isfinite(item[1]) else item[1])
        if math.isfinite(seconds):
            notes.append(f"{phase} dominates IPT compute time ({seconds / compute:.0%})")
    if basis_cols > max(k, 0) * 2 and k > 0:
        notes.append(
            f"basis_cols={basis_cols} is large for k={k}; RR/Davidson cost may be inflated by block lifting/oversampling"
        )
    if restarts > 0:
        notes.append(
            f"Davidson restarted {restarts} times; restart threshold is fixed in the algorithm and is worth exposing as a tuning parameter"
        )
    if target_block_size > 64:
        notes.append(
            f"target block size {target_block_size} is above 64; preparation split quality is a likely bottleneck"
        )
    return " | ".join(notes)


def speedup(primme_time, ipt_time):
    p = to_float(primme_time)
    i = to_float(ipt_time)
    if math.isfinite(p) and math.isfinite(i) and i > 0:
        return f"{p / i:.6g}"
    return ""


def discover_targets(result_root, param):
    matrix_ids = set()
    for cache in TEST_ROOT.glob("*/*_csc.bin"):
        matrix_ids.add(cache.parent.name)
    for path in result_root.glob(f"*/K*/{param}"):
        matrix_ids.add(path.parents[1].name)
    for status in result_root.glob(f"*/K*/{param}/run_status.tsv"):
        matrix_ids.add(status.parents[2].name)
    return sorted(matrix_ids)


def build_rows(args):
    result_root = Path(args.result_root)
    log_root = Path(args.log_root)
    rows = []

    for matrix_id in discover_targets(result_root, args.param_set):
        cache_files = sorted((TEST_ROOT / matrix_id).glob("*_csc.bin"))
        for k in args.k_list.split():
            base = result_root / matrix_id / f"K{k}" / args.param_set
            status = read_kv_file(base / "run_status.tsv")
            params = read_kv_file(base / "params.env")
            ipt_summary_path = base / "ipt" / f"{matrix_id}_summary.txt"
            primme_summary_path = base / "gpu_primme" / f"{matrix_id}_summary.txt"
            log_path = log_root / matrix_id / f"K{k}" / args.param_set / "latest.log"
            ipt_summary, ipt_methods = parse_summary(ipt_summary_path)
            primme_summary, primme_methods = parse_summary(primme_summary_path)
            ipt = method(ipt_methods, "IPT_CUDA_BLOCK_CLUSTER")
            primme = method(primme_methods, "PRIMME_CUBLAS")
            log_text = read_log(log_path)

            row = {
                "matrix_id": matrix_id,
                "k": k,
                "param_set": args.param_set,
                "matrix_cache": str(cache_files[0]) if cache_files else status.get("matrix_cache", ""),
                "tested": "1" if (base / "run_status.tsv").is_file() or ipt_summary_path.is_file() else "0",
                "n": ipt_summary.get("n", primme_summary.get("n", "")),
                "nnz": ipt_summary.get("nnz", primme_summary.get("nnz", "")),
                "matrix_source": ipt_summary.get("matrix_source", primme_summary.get("matrix_source", "")),
                "ipt_status": ipt.get("status", status.get("ipt_status", "")),
                "ipt_rc": status.get("ipt_rc", ""),
                "ipt_compute_time_sec": ipt.get("compute_time_sec", ""),
                "ipt_total_time_sec": ipt.get("time_total_sec", ""),
                "ipt_preparation_time_sec": ipt.get("preparation_time_sec", ""),
                "ipt_transfer_setup_time_sec": ipt.get("transfer_setup_time_sec", ""),
                "ipt_iteration_time_sec": ipt.get("iteration_time_sec", ""),
                "ipt_rayleigh_ritz_time_sec": ipt.get("rayleigh_ritz_time_sec", ""),
                "ipt_davidson_time_sec": ipt.get("davidson_time_sec", ""),
                "ipt_max_rel_residual": ipt.get("max_relative_eigen_residual", ""),
                "ipt_max_rel_residual_index": ipt.get("max_relative_eigen_residual_index", ""),
                "ipt_basis_cols": ipt.get("basis_cols", ""),
                "ipt_iterations": ipt.get("iterations", ""),
                "ipt_accepted_steps": ipt.get("accepted_steps", ""),
                "ipt_rejected_steps": ipt.get("rejected_steps", ""),
                "ipt_restart_count": ipt.get("davidson_restart_count", ""),
                "primme_status": primme.get("status", ""),
                "primme_rc": status.get("primme_rc", ""),
                "primme_compute_time_sec": primme.get("compute_time_sec", ""),
                "primme_total_time_sec": primme.get("time_total_sec", ""),
                "primme_max_rel_residual": primme.get("max_relative_eigen_residual", ""),
                "primme_over_ipt_time": speedup(primme.get("compute_time_sec"), ipt.get("compute_time_sec")),
                "target_block_size": ipt_summary.get("target_block_size", ""),
                "max_block_size": ipt_summary.get("max_block_size", params.get("IPT_MAX_LIFT_BLOCK_SIZE", "")),
                "max_block_size_error": ipt_summary.get("max_block_size_error", ""),
                "max_block_size_offending_block_size": ipt_summary.get("max_block_size_offending_block_size", ""),
                "max_coupling_over_gap": f"{max_coupling_ratio(ipt_summary):.17g}",
                "ipt_max_lift_block_size": params.get("IPT_MAX_LIFT_BLOCK_SIZE", ""),
                "ipt_max_lift_rounds": params.get("IPT_MAX_LIFT_ROUNDS", ""),
                "ipt_rel_gap_tol": params.get("IPT_REL_GAP_TOL", ""),
                "ipt_coupling_gap_ratio_tol": params.get("IPT_COUPLING_GAP_RATIO_TOL", ""),
                "ipt_coupling_eta": params.get("IPT_COUPLING_ETA", ""),
                "ipt_davidson_steps": params.get("IPT_DAVIDSON_STEPS", ""),
                "ipt_davidson_extra_steps": params.get("IPT_DAVIDSON_EXTRA_STEPS", ""),
                "ipt_summary_path": str(ipt_summary_path),
                "primme_summary_path": str(primme_summary_path),
                "log_path": str(log_path),
            }
            row["ipt_failure_class"] = classify(row, ipt_summary, ipt, log_text, status)
            row["bottleneck_and_optimization_notes"] = bottleneck_notes(row)
            rows.append(row)
    return rows


def write_report(rows, csv_path, report_path, param_set):
    fields = [
        "matrix_id",
        "k",
        "param_set",
        "tested",
        "n",
        "nnz",
        "matrix_cache",
        "matrix_source",
        "ipt_status",
        "ipt_failure_class",
        "ipt_compute_time_sec",
        "ipt_iteration_time_sec",
        "ipt_rayleigh_ritz_time_sec",
        "ipt_davidson_time_sec",
        "ipt_max_rel_residual",
        "ipt_max_rel_residual_index",
        "ipt_basis_cols",
        "ipt_iterations",
        "ipt_accepted_steps",
        "ipt_rejected_steps",
        "ipt_restart_count",
        "primme_status",
        "primme_compute_time_sec",
        "primme_max_rel_residual",
        "primme_over_ipt_time",
        "target_block_size",
        "max_block_size",
        "max_block_size_error",
        "max_block_size_offending_block_size",
        "max_coupling_over_gap",
        "ipt_max_lift_block_size",
        "ipt_max_lift_rounds",
        "ipt_rel_gap_tol",
        "ipt_coupling_gap_ratio_tol",
        "ipt_coupling_eta",
        "ipt_davidson_steps",
        "ipt_davidson_extra_steps",
        "bottleneck_and_optimization_notes",
        "ipt_summary_path",
        "primme_summary_path",
        "log_path",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_failure = Counter(r["ipt_failure_class"] for r in rows)
    tested = [r for r in rows if r["tested"] == "1"]
    success = [r for r in rows if r["ipt_failure_class"] == "ok"]
    primme_done = [r for r in success if r.get("primme_status") == "success"]
    by_matrix = defaultdict(list)
    for row in rows:
        by_matrix[row["matrix_id"]].append(row)

    lines = []
    lines.append(f"# test_initial cache-only report: {param_set}")
    lines.append("")
    lines.append(f"- rows: {len(rows)}")
    lines.append(f"- tested rows: {len(tested)}")
    lines.append(f"- IPT converged rows: {len(success)}")
    lines.append(f"- GPU PRIMME comparison rows: {len(primme_done)}")
    lines.append(f"- near-diagonal sufficient ratio used for diagnostics: {NEAR_DIAGONAL_SUFFICIENT_RATIO:.17g}")
    lines.append("")
    lines.append("## Failure classes")
    for key, count in sorted(by_failure.items()):
        lines.append(f"- {key}: {count}")
    lines.append("")
    lines.append("## Matrix summary")
    lines.append("| matrix | K results | notes |")
    lines.append("|---|---|---|")
    for matrix_id, matrix_rows in sorted(by_matrix.items()):
        k_bits = []
        notes = []
        for row in sorted(matrix_rows, key=lambda r: int(r["k"])):
            status = row["ipt_failure_class"]
            detail = row.get("ipt_max_rel_residual") or ""
            if status == "ok" and row.get("primme_over_ipt_time"):
                detail = f"speedup={row['primme_over_ipt_time']}"
            k_bits.append(f"K{row['k']}:{status} {detail}".strip())
            note = row.get("bottleneck_and_optimization_notes", "")
            if note:
                notes.append(f"K{row['k']} {note}")
        lines.append(
            f"| {matrix_id} | {'; '.join(k_bits)} | {'<br>'.join(notes[:6])} |"
        )
    report_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default=str(RESULT_ROOT))
    parser.add_argument("--log-root", default=str(LOG_ROOT))
    parser.add_argument("--param-set", default="default")
    parser.add_argument("--k-list", default=" ".join(K_LIST))
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--out-report", default="")
    args = parser.parse_args()

    out_base = Path(args.result_root)
    csv_path = Path(args.out_csv) if args.out_csv else out_base / f"cache_results_{args.param_set}.csv"
    report_path = Path(args.out_report) if args.out_report else out_base / f"cache_report_{args.param_set}.md"
    rows = build_rows(args)
    write_report(rows, csv_path, report_path, args.param_set)
    print(csv_path)
    print(report_path)


if __name__ == "__main__":
    main()
