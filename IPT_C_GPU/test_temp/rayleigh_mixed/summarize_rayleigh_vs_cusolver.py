#!/usr/bin/env python3
import csv
import glob
import os
import statistics
from collections import defaultdict


def parse_list(value, cast):
    out = []
    for item in (value or "").split(","):
        item = item.strip()
        if item:
            out.append(cast(item))
    return out


def as_float(value):
    try:
        return float(value)
    except Exception:
        return None


def as_int(value):
    try:
        return int(float(value))
    except Exception:
        return None


def eps_tag(eps):
    return f"{eps:.12g}".replace(".", "_")


root = os.environ.get("IPT_C_ROOT", "/fs1/home/nudt_liujie/ftt/IPT_C_GPU")
rayleigh_dir = os.environ.get(
    "IPT_RAYLEIGH_RESULTS_DIR",
    os.path.join(root, "results", "rayleigh_mixed"),
)
tf32_dir = os.environ.get(
    "IPT_TF32_RESULTS_DIR",
    os.path.join(root, "results", "mixed_precision_tf32"),
)

target_ns = set(parse_list(os.environ.get("IPT_COMPARE_NS", "8192,16384,26000"), int))
target_eps = set(parse_list(os.environ.get("IPT_COMPARE_EPSILONS", ""), float))
if not target_eps:
    target_eps = None

detail_csv = os.environ.get(
    "IPT_COMPARE_DETAIL_CSV",
    os.path.join(rayleigh_dir, "rayleigh_vs_cusolver_detail.csv"),
)
threshold_csv = os.environ.get(
    "IPT_COMPARE_THRESHOLD_CSV",
    os.path.join(rayleigh_dir, "rayleigh_vs_cusolver_threshold.csv"),
)
missing_eps_file = os.environ.get("IPT_MISSING_THRESHOLD_EPS_FILE", "")


rayleigh = {}
for path in sorted(glob.glob(os.path.join(rayleigh_dir, "rayleigh_mixed_qr_sweep_epsilon_*.csv"))):
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "ok":
                continue
            n = as_int(row.get("N"))
            eps = as_float(row.get("epsilon"))
            if n is None or eps is None:
                continue
            if target_ns and n not in target_ns:
                continue
            if target_eps is not None and eps not in target_eps:
                continue
            rayleigh[(n, eps)] = row


baseline = {}
cusolver_by_n = defaultdict(list)
for path in sorted(glob.glob(os.path.join(tf32_dir, "tf32_initial_cusolver_sweep_epsilon_*.csv"))):
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("initial_precision") != "tf32":
                continue
            if row.get("status") not in ("ok", "partial_failed"):
                continue
            n = as_int(row.get("N"))
            eps = as_float(row.get("epsilon"))
            if n is None or eps is None:
                continue
            cusolver_time = as_float(row.get("time_cusolver_syevd_sec"))
            if cusolver_time is not None and cusolver_time > 0.0:
                cusolver_by_n[n].append(cusolver_time)
            if target_ns and n not in target_ns:
                continue
            if target_eps is not None and eps not in target_eps:
                continue
            key = (n, eps)
            if key not in baseline or (
                baseline[key].get("status") != "ok" and row.get("status") == "ok"
            ):
                baseline[key] = row

cusolver_ref_by_n = {}
for n, values in cusolver_by_n.items():
    if values:
        cusolver_ref_by_n[n] = statistics.median(values)


detail_fields = [
    "epsilon",
    "N",
    "rayleigh_total_sec",
    "cusolver_sec",
    "cusolver_source",
    "speedup_cusolver_over_rayleigh",
    "rayleigh_slower_than_cusolver",
    "original_fp64_ipt_sec",
    "tf32_mixed_ipt_sec",
    "speedup_original_over_rayleigh",
    "speedup_tf32_over_rayleigh",
    "rayleigh_tf32_iterations",
    "rayleigh_fp64_iterations",
    "tf32_mixed_initial_iterations",
    "tf32_mixed_fp64_iterations",
    "original_fp64_iterations",
    "rayleigh_fp64_fixed_point_residual",
    "rayleigh_absolute_residual",
    "tf32_mixed_fp64_fixed_point_residual",
    "original_fp64_fixed_point_residual",
    "status",
]

details = []
for key in sorted(rayleigh, key=lambda item: (item[1], item[0])):
    n, eps = key
    r = rayleigh[key]
    b = baseline.get(key)
    ray_time = as_float(r.get("time_total_sec"))
    exact_cusolver = as_float(b.get("time_cusolver_syevd_sec")) if b else None
    if exact_cusolver is not None and exact_cusolver > 0.0:
        cusolver_time = exact_cusolver
        cusolver_source = "exact"
    else:
        cusolver_time = cusolver_ref_by_n.get(n)
        cusolver_source = "median_reference" if cusolver_time is not None else "missing"

    original_time = as_float(b.get("time_original_ipt_sec")) if b else None
    tf32_time = as_float(b.get("time_mixed_total_sec")) if b else None
    if tf32_time is None and b:
        initial = as_float(b.get("time_initial_sec"))
        refine = as_float(b.get("time_fp64_refine_sec"))
        if initial is not None and refine is not None:
            tf32_time = initial + refine

    speed_cu = cusolver_time / ray_time if cusolver_time and ray_time else None
    speed_orig = original_time / ray_time if original_time and ray_time else None
    speed_tf32 = tf32_time / ray_time if tf32_time and ray_time else None
    slower = bool(ray_time and cusolver_time and ray_time > cusolver_time)

    details.append({
        "epsilon": f"{eps:.12g}",
        "N": n,
        "rayleigh_total_sec": ray_time,
        "cusolver_sec": cusolver_time,
        "cusolver_source": cusolver_source,
        "speedup_cusolver_over_rayleigh": speed_cu,
        "rayleigh_slower_than_cusolver": "true" if slower else "false",
        "original_fp64_ipt_sec": original_time,
        "tf32_mixed_ipt_sec": tf32_time,
        "speedup_original_over_rayleigh": speed_orig,
        "speedup_tf32_over_rayleigh": speed_tf32,
        "rayleigh_tf32_iterations": as_int(r.get("tf32_iterations")),
        "rayleigh_fp64_iterations": as_int(r.get("fp64_iterations")),
        "tf32_mixed_initial_iterations": as_int(b.get("initial_iterations")) if b else None,
        "tf32_mixed_fp64_iterations": as_int(b.get("fp64_iterations")) if b else None,
        "original_fp64_iterations": as_int(b.get("original_ipt_iterations")) if b else None,
        "rayleigh_fp64_fixed_point_residual": as_float(r.get("fp64_fixed_point_residual")),
        "rayleigh_absolute_residual": as_float(r.get("absolute_residual")),
        "tf32_mixed_fp64_fixed_point_residual": as_float(b.get("fp64_fixed_point_residual")) if b else None,
        "original_fp64_fixed_point_residual": as_float(b.get("original_ipt_fixed_point_residual")) if b else None,
        "status": r.get("status"),
    })

os.makedirs(os.path.dirname(detail_csv), exist_ok=True)
with open(detail_csv, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=detail_fields)
    writer.writeheader()
    writer.writerows(details)

threshold_fields = [
    "N",
    "first_epsilon_rayleigh_slower_than_cusolver",
    "rayleigh_total_sec",
    "cusolver_sec",
    "cusolver_source",
    "speedup_cusolver_over_rayleigh",
    "original_fp64_ipt_sec",
    "tf32_mixed_ipt_sec",
    "original_fp64_iterations",
    "tf32_mixed_initial_iterations",
    "tf32_mixed_fp64_iterations",
    "rayleigh_tf32_iterations",
    "rayleigh_fp64_iterations",
    "note",
]

threshold_rows = []
missing_eps = set()
for n in sorted(target_ns):
    n_rows = [row for row in details if int(row["N"]) == n]
    n_rows.sort(key=lambda row: float(row["epsilon"]))
    hit = next((row for row in n_rows if row["rayleigh_slower_than_cusolver"] == "true"), None)
    if hit:
        if hit["cusolver_source"] != "exact":
            missing_eps.add(float(hit["epsilon"]))
        note = "exact" if hit["cusolver_source"] == "exact" else "estimated_needs_exact_baseline"
        threshold_rows.append({
            "N": n,
            "first_epsilon_rayleigh_slower_than_cusolver": hit["epsilon"],
            "rayleigh_total_sec": hit["rayleigh_total_sec"],
            "cusolver_sec": hit["cusolver_sec"],
            "cusolver_source": hit["cusolver_source"],
            "speedup_cusolver_over_rayleigh": hit["speedup_cusolver_over_rayleigh"],
            "original_fp64_ipt_sec": hit["original_fp64_ipt_sec"],
            "tf32_mixed_ipt_sec": hit["tf32_mixed_ipt_sec"],
            "original_fp64_iterations": hit["original_fp64_iterations"],
            "tf32_mixed_initial_iterations": hit["tf32_mixed_initial_iterations"],
            "tf32_mixed_fp64_iterations": hit["tf32_mixed_fp64_iterations"],
            "rayleigh_tf32_iterations": hit["rayleigh_tf32_iterations"],
            "rayleigh_fp64_iterations": hit["rayleigh_fp64_iterations"],
            "note": note,
        })
    else:
        threshold_rows.append({
            "N": n,
            "first_epsilon_rayleigh_slower_than_cusolver": "",
            "rayleigh_total_sec": "",
            "cusolver_sec": cusolver_ref_by_n.get(n, ""),
            "cusolver_source": "median_reference" if n in cusolver_ref_by_n else "missing",
            "speedup_cusolver_over_rayleigh": "",
            "original_fp64_ipt_sec": "",
            "tf32_mixed_ipt_sec": "",
            "original_fp64_iterations": "",
            "tf32_mixed_initial_iterations": "",
            "tf32_mixed_fp64_iterations": "",
            "rayleigh_tf32_iterations": "",
            "rayleigh_fp64_iterations": "",
            "note": "not_reached_in_requested_range",
        })

with open(threshold_csv, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=threshold_fields)
    writer.writeheader()
    writer.writerows(threshold_rows)

if missing_eps_file:
    with open(missing_eps_file, "w") as fh:
        fh.write(",".join(f"{eps:.12g}" for eps in sorted(missing_eps)))
        if missing_eps:
            fh.write("\n")

print(f"detail_csv={detail_csv}")
print(f"threshold_csv={threshold_csv}")
if missing_eps_file:
    print(f"missing_threshold_eps_file={missing_eps_file}")
    print("missing_threshold_eps=" + ",".join(f"{eps:.12g}" for eps in sorted(missing_eps)))
