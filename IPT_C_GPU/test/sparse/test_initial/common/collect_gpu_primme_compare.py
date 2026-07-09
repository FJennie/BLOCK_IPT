#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


def parse_kv_line(path: Path, method: str):
    if not path.is_file():
        return {}
    for line in path.read_text(errors="replace").splitlines():
        if f"method={method}" not in line:
            continue
        out = {}
        for tok in line.split():
            if "=" in tok:
                key, val = tok.split("=", 1)
                out[key] = val
        return out
    return {}


def first_cuda_error(log_path: Path):
    if not log_path.is_file():
        return ""
    for line in log_path.read_text(errors="replace").splitlines():
        if "benchmark CUDA error" in line or "PRIMME" in line and "error" in line.lower():
            return line.strip()
    return ""


def to_float(text):
    try:
        val = float(text)
    except (TypeError, ValueError):
        return math.nan
    return val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ipt-results",
        default="/fs1/home/nudt_liujie/ftt/IPT_C_GPU/results/sparse/test_initial/selected_100_k_results.csv",
    )
    parser.add_argument(
        "--gpu-base",
        default="/fs1/home/nudt_liujie/ftt/IPT_C_GPU/results/sparse/test_initial_gpu_primme",
    )
    parser.add_argument(
        "--gpu-log-base",
        default="/fs1/home/nudt_liujie/ftt/IPT_C_GPU/logs/sparse/test_initial_gpu_primme",
    )
    parser.add_argument(
        "--out-csv",
        default="/fs1/home/nudt_liujie/ftt/IPT_C_GPU/results/sparse/test_initial_gpu_primme/gpu_primme_vs_ipt_success_k.csv",
    )
    parser.add_argument(
        "--out-report",
        default="/fs1/home/nudt_liujie/ftt/IPT_C_GPU/results/sparse/test_initial_gpu_primme/gpu_primme_vs_ipt_success_k_report.txt",
    )
    args = parser.parse_args()

    gpu_base = Path(args.gpu_base)
    gpu_log_base = Path(args.gpu_log_base)
    rows = []
    with open(args.ipt_results, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("ipt_status") != "success":
                continue
            matrix_id = row["matrix_id"]
            k = row["k"]
            summary = gpu_base / matrix_id / f"K{k}" / f"{matrix_id}_summary.txt"
            log = gpu_log_base / matrix_id / f"K{k}" / "latest.log"
            primme = parse_kv_line(summary, "PRIMME_CUBLAS")

            ipt_time_text = row.get("ipt_solve_time_sec") or row.get("ipt_api_total_sec", "")
            ipt_time = to_float(ipt_time_text)
            primme_time_text = primme.get("compute_time_sec") or primme.get(
                "time_total_sec", ""
            )
            primme_time = to_float(primme_time_text)
            speedup = ""
            if math.isfinite(ipt_time) and math.isfinite(primme_time) and ipt_time > 0:
                speedup = f"{primme_time / ipt_time:.6g}"

            rows.append(
                {
                    "matrix_id": matrix_id,
                    "k": k,
                    "n": row.get("n", ""),
                    "nnz": row.get("nnz", ""),
                    "ipt_time_sec": ipt_time_text,
                    "ipt_max_rel_residual": row.get("ipt_max_rel_residual", ""),
                    "ipt_basis_cols": row.get("ipt_basis_cols", ""),
                    "ipt_accepted_steps": row.get("ipt_accepted_steps", ""),
                    "ipt_restart_count": row.get("ipt_restart_count", ""),
                    "primme_gpu_status": primme.get("status", "missing"),
                    "primme_gpu_time_sec": primme_time_text,
                    "primme_gpu_iterations": primme.get("iterations", ""),
                    "primme_gpu_max_rel_residual": primme.get("max_relative_eigen_residual", ""),
                    "primme_gpu_over_ipt_time": speedup,
                    "primme_gpu_failure_detail": ""
                    if primme.get("status") == "success"
                    else first_cuda_error(log),
                    "ipt_summary_path": row.get("summary_path", ""),
                    "primme_gpu_summary_path": str(summary),
                    "primme_gpu_log_path": str(log),
                }
            )

    fieldnames = [
        "matrix_id",
        "k",
        "n",
        "nnz",
        "ipt_time_sec",
        "ipt_max_rel_residual",
        "ipt_basis_cols",
        "ipt_accepted_steps",
        "ipt_restart_count",
        "primme_gpu_status",
        "primme_gpu_time_sec",
        "primme_gpu_iterations",
        "primme_gpu_max_rel_residual",
        "primme_gpu_over_ipt_time",
        "primme_gpu_failure_detail",
        "ipt_summary_path",
        "primme_gpu_summary_path",
        "primme_gpu_log_path",
    ]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successes = [r for r in rows if r["primme_gpu_status"] == "success"]
    failures = [r for r in rows if r["primme_gpu_status"] != "success"]
    ratios = [to_float(r["primme_gpu_over_ipt_time"]) for r in successes]
    ratios = [x for x in ratios if math.isfinite(x)]
    faster = sum(1 for x in ratios if x < 1.0)
    slower = sum(1 for x in ratios if x > 1.0)
    avg_ratio = sum(ratios) / len(ratios) if ratios else math.nan

    out_report = Path(args.out_report)
    with out_report.open("w") as f:
        f.write(f"ipt_success_k_count={len(rows)}\n")
        f.write(f"primme_gpu_success_count={len(successes)}\n")
        f.write(f"primme_gpu_failure_count={len(failures)}\n")
        f.write(f"primme_gpu_faster_than_ipt_count={faster}\n")
        f.write(f"primme_gpu_slower_than_ipt_count={slower}\n")
        f.write(f"mean_primme_gpu_over_ipt_time={avg_ratio:.6g}\n")
        f.write(f"csv={out_csv}\n")
        if failures:
            f.write("\nfailures:\n")
            for r in failures:
                f.write(
                    f"{r['matrix_id']} K{r['k']}: {r['primme_gpu_status']} "
                    f"{r['primme_gpu_failure_detail']}\n"
                )

    print(f"wrote {out_csv}")
    print(f"wrote {out_report}")
    print(f"rows={len(rows)} primme_gpu_success={len(successes)} failures={len(failures)}")


if __name__ == "__main__":
    main()
