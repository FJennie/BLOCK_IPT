#!/usr/bin/env python3
import csv
import math
import re
import sys
from pathlib import Path


ROOT = Path("/fs1/home/nudt_liujie/ftt/IPT_C_GPU")
TEST_ROOT = ROOT / "test/sparse/test_initial"
RESULT_ROOT = ROOT / "results/sparse/test_initial"
DEFAULT_K_LIST = ["10", "20", "30"]


def parse_key_value_summary(path):
    data = {}
    method_lines = []
    if not path.exists():
        return data, method_lines
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("method="):
            method_lines.append(line)
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            data[key] = value
    return data, method_lines


def parse_method_line(line):
    values = {}
    for part in line.split():
        if "=" in part:
            key, value = part.split("=", 1)
            values[key] = value
    return values


def parse_case_env(path):
    data = {}
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        data[key.strip()] = value
    return data


def classify_failure(summary, ipt, log_text):
    if ipt.get("status") == "success":
        return "ok"
    text = (log_text or "").lower()
    if "invalid argument at block detection" in text and "max_block_size" in text:
        return "block_gt_max"
    if "block solve failed" in text:
        return "riccati_block_solve_failed"
    if "singular" in text or "riccati" in text:
        return "riccati_singular_or_sep_small"
    if (
        "out of memory" in text
        or "cudaerrormemoryallocation" in text
        or "cuda error: memory allocation" in text
    ):
        return "memory"
    if "invalid closed-shell cas core/orbital window" in text:
        return "invalid_cas_window"
    if "failed to generate molecular fci/cas active integrals" in text:
        return "matrix_generation_failed"
    if "unresolved target gap" in text or "delta/gap" in text or "coupling_gap_ratio" in text:
        return "large_coupling_over_gap"
    if ipt.get("status") == "failed":
        return "not_converged"
    if not ipt:
        return "no_ipt_result"
    return "other"


def first_match(pattern, text, group=1):
    match = re.search(pattern, text or "", re.IGNORECASE)
    if not match:
        return ""
    return match.group(group)


def failure_detail(log_text):
    text = log_text or ""
    block_limit = re.search(
        r"block_size=(\d+)\s+max_block_size=(\d+)", text, re.IGNORECASE
    )
    if block_limit:
        return f"block_size={block_limit.group(1)} max_block_size={block_limit.group(2)}"
    block_failed = re.search(
        r"block solve failed status=([-\d]+).*?block=(\d+):(\d+) size=(\d+)",
        text,
        re.IGNORECASE,
    )
    if block_failed:
        return (
            f"block_solve_status={block_failed.group(1)} "
            f"block={block_failed.group(2)}:{block_failed.group(3)} "
            f"size={block_failed.group(4)}"
        )
    gap_failed = re.search(
        r"unresolved target gap.*", text, re.IGNORECASE
    )
    if gap_failed:
        return gap_failed.group(0)[:240]
    if "invalid closed-shell CAS core/orbital window" in text:
        return "invalid closed-shell CAS core/orbital window"
    if "failed to generate molecular FCI/CAS active integrals" in text:
        return "failed to generate molecular FCI/CAS active integrals"
    return ""


def main():
    rows = []
    targets = []
    for case_env in sorted(TEST_ROOT.glob("*/case.env")):
        case_data = parse_case_env(case_env)
        matrix_id = case_data.get("IPT_MATRIX_ID", case_env.parent.name)
        for k in DEFAULT_K_LIST:
            targets.append((matrix_id, k, case_env))

    seen = {(matrix_id, k) for matrix_id, k, _ in targets}
    for summary in sorted(RESULT_ROOT.glob("*/K*/*_summary.txt")):
        matrix_id = summary.parents[1].name
        k_dir = summary.parent.name
        k = k_dir[1:] if k_dir.startswith("K") else k_dir
        if (matrix_id, k) not in seen:
            targets.append((matrix_id, k, None))
            seen.add((matrix_id, k))

    for matrix_id, k, case_env in targets:
        summary = RESULT_ROOT / matrix_id / f"K{k}" / f"{matrix_id}_summary.txt"
        data, method_lines = parse_key_value_summary(summary)
        methods = [parse_method_line(line) for line in method_lines]
        ipt = next(
            (m for m in methods if m.get("method") == "IPT_CUDA_BLOCK_CLUSTER"),
            {},
        )
        primme = next(
            (m for m in methods if m.get("method", "").startswith("PRIMME")),
            {},
        )
        log_path = (
            ROOT
            / "logs/sparse/test_initial"
            / matrix_id
            / f"K{k}"
            / "latest.log"
        )
        tested = summary.exists() or log_path.exists()
        log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
        max_lift_block_size = first_match(
            r"IPT_MAX_LIFT_BLOCK_SIZE=([^\s]+)", log_text
        )
        block_size = first_match(r"block_size=(\d+)", log_text)
        rows.append(
            {
                "matrix_id": matrix_id,
                "k": k,
                "case_env": str(case_env) if case_env is not None else "",
                "tested": "1" if tested else "0",
                "n": data.get("n", ""),
                "nnz": data.get("nnz", ""),
                "basis": data.get("molecule_basis", ""),
                "active_orbitals": data.get("active_orbitals", ""),
                "active_electrons": data.get("active_electrons", ""),
                "ipt_status": ipt.get("status", ""),
                "ipt_failure_class": classify_failure(data, ipt, log_text),
                "ipt_failure_detail": failure_detail(log_text),
                "ipt_max_lift_block_size": max_lift_block_size,
                "ipt_block_size": block_size,
                "ipt_solve_time_sec": ipt.get("time_sec", ""),
                "ipt_api_total_sec": ipt.get("time_total_sec", ""),
                "ipt_max_rel_residual": ipt.get("max_relative_eigen_residual", ""),
                "ipt_max_rel_residual_index": ipt.get(
                    "max_relative_eigen_residual_index", ""
                ),
                "ipt_basis_cols": ipt.get("basis_cols", ""),
                "ipt_accepted_steps": ipt.get("accepted_steps", ""),
                "ipt_rejected_steps": ipt.get("rejected_steps", ""),
                "ipt_restart_count": ipt.get("davidson_restart_count", ""),
                "primme_status": primme.get("status", ""),
                "primme_solve_time_sec": primme.get("time_sec", ""),
                "primme_max_rel_residual": primme.get(
                    "max_relative_eigen_residual", ""
                ),
                "summary_path": str(summary),
                "log_path": str(log_path),
            }
        )

    out = RESULT_ROOT / "initial_results_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "matrix_id",
        "k",
        "case_env",
        "tested",
        "n",
        "nnz",
        "basis",
        "active_orbitals",
        "active_electrons",
        "ipt_status",
        "ipt_failure_class",
        "ipt_failure_detail",
        "ipt_max_lift_block_size",
        "ipt_block_size",
        "ipt_solve_time_sec",
        "ipt_api_total_sec",
        "ipt_max_rel_residual",
        "ipt_max_rel_residual_index",
        "ipt_basis_cols",
        "ipt_accepted_steps",
        "ipt_rejected_steps",
        "ipt_restart_count",
        "primme_status",
        "primme_solve_time_sec",
        "primme_max_rel_residual",
        "summary_path",
        "log_path",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
