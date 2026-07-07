import os
import struct
import numpy as np

path = r"D:\study\matrix\library\IPT\BLOCK_IPT\IPT_C_GPU\results\sparse\CH4_K30_debug\ch4_sto6g_fci_15876_csc.bin"

HEADER_BYTES = 56

def read_header(f):
    f.seek(0)
    raw = f.read(HEADER_BYTES)

    magic, mol, version, n = struct.unpack_from("<4s4sii", raw, 0)
    nnz = struct.unpack_from("<q", raw, 16)[0]
    const_value = struct.unpack_from("<d", raw, 24)[0]
    n_orb, n_elec, n_alpha, n_beta, n_alpha_det, n_beta_det = struct.unpack_from("<iiiiii", raw, 32)

    return {
        "magic": magic.decode(errors="replace"),
        "mol": mol.decode(errors="replace"),
        "version": version,
        "n": n,
        "nnz": nnz,
        "const_value": const_value,
        "n_orb": n_orb,
        "n_elec": n_elec,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "n_alpha_det": n_alpha_det,
        "n_beta_det": n_beta_det,
    }

def main():
    file_size = os.path.getsize(path)

    with open(path, "rb") as f:
        header = read_header(f)

        n = int(header["n"])
        nnz = int(header["nnz"])

        print("header:")
        for k, v in header.items():
            print(f"  {k}: {v}")
        print(f"file_size: {file_size}")

        expected_size = HEADER_BYTES + 4 * (n + 1) + 4 * nnz + 8 * nnz + 8 * n
        print(f"expected_size: {expected_size}")
        if expected_size != file_size:
            raise RuntimeError(f"文件大小不匹配：expected={expected_size}, actual={file_size}")

        f.seek(HEADER_BYTES)

        col_ptr = np.fromfile(f, dtype=np.int32, count=n + 1)
        row_idx = np.fromfile(f, dtype=np.int32, count=nnz)
        values = np.fromfile(f, dtype=np.float64, count=nnz)
        diag_tail = np.fromfile(f, dtype=np.float64, count=n)

    print("\nCSC check:")
    print("  col_ptr[0] =", int(col_ptr[0]))
    print("  col_ptr[-1] =", int(col_ptr[-1]))
    print("  row_idx min/max =", int(row_idx.min()), int(row_idx.max()))
    print("  values min/max =", float(values.min()), float(values.max()))

    if int(col_ptr[0]) != 0 or int(col_ptr[-1]) != nnz:
        raise RuntimeError("col_ptr 不合法，说明格式推断可能不对。")

    # 从 CSC values 里再提取一份对角线，用来和文件尾部 diag 校验
    diag_from_csc = np.zeros(n, dtype=np.float64)

    for col in range(n):
        start = int(col_ptr[col])
        end = int(col_ptr[col + 1])
        rows = row_idx[start:end]
        vals = values[start:end]

        hit = np.where(rows == col)[0]
        if hit.size > 0:
            diag_from_csc[col] = vals[hit[0]]

    max_diag_diff = np.max(np.abs(diag_from_csc - diag_tail))
    print("\ndiagonal check:")
    print("  max |diag_from_csc - diag_tail| =", max_diag_diff)

    # 如果文件尾部 diag 和 CSC 对角线一致，就用 diag_tail；
    # 否则用 CSC 里真正的对角线。
    if max_diag_diff < 1e-10:
        diag = diag_tail
        print("  use diag_tail")
    else:
        diag = diag_from_csc
        print("  use diag_from_csc")

    # 找全局最小的 40 个对角元
    top = np.argsort(diag)[:40]
    target_mask = np.zeros(n, dtype=bool)
    target_mask[top] = True

    # 计算每个目标 index 的列最大非对角元
    col_max_abs = {}
    col_max_row = {}
    col_max_val = {}

    for col in top:
        col = int(col)
        start = int(col_ptr[col])
        end = int(col_ptr[col + 1])
        rows = row_idx[start:end]
        vals = values[start:end]

        mask = rows != col
        if np.any(mask):
            off_rows = rows[mask]
            off_vals = vals[mask]
            p = int(np.argmax(np.abs(off_vals)))
            col_max_abs[col] = float(abs(off_vals[p]))
            col_max_row[col] = int(off_rows[p])
            col_max_val[col] = float(off_vals[p])
        else:
            col_max_abs[col] = 0.0
            col_max_row[col] = -1
            col_max_val[col] = 0.0

    # 计算每个目标 index 的行最大非对角元
    # 因为是 CSC 格式，按行找要扫一遍所有列。
    row_max_abs = {int(i): 0.0 for i in top}
    row_max_col = {int(i): -1 for i in top}
    row_max_val = {int(i): 0.0 for i in top}

    for col in range(n):
        start = int(col_ptr[col])
        end = int(col_ptr[col + 1])

        rows = row_idx[start:end]
        vals = values[start:end]

        # 只保留 row 属于 top40 的非零元，并排除对角元
        mask = target_mask[rows] & (rows != col)
        if not np.any(mask):
            continue

        hit_rows = rows[mask]
        hit_vals = vals[mask]

        for r, v in zip(hit_rows, hit_vals):
            r = int(r)
            av = float(abs(v))
            if av > row_max_abs[r]:
                row_max_abs[r] = av
                row_max_col[r] = int(col)
                row_max_val[r] = float(v)

    print("\n最小的 40 个对角元，以及同行/同列最大非对角元：")
    print(
        "rank,idx,diag,"
        "col_max_abs,col_max_row,col_max_value,"
        "row_max_abs,row_max_col,row_max_value,"
        "overall_max_abs,overall_where"
    )

    for rank, idx in enumerate(top, start=1):
        idx = int(idx)

        c_abs = col_max_abs[idx]
        c_row = col_max_row[idx]
        c_val = col_max_val[idx]

        r_abs = row_max_abs[idx]
        r_col = row_max_col[idx]
        r_val = row_max_val[idx]

        if c_abs >= r_abs:
            overall_abs = c_abs
            overall_where = f"A[{c_row},{idx}]"
        else:
            overall_abs = r_abs
            overall_where = f"A[{idx},{r_col}]"

        print(
            f"{rank},{idx},{diag[idx]:.17e},"
            f"{c_abs:.17e},{c_row},{c_val:.17e},"
            f"{r_abs:.17e},{r_col},{r_val:.17e},"
            f"{overall_abs:.17e},{overall_where}"
        )

if __name__ == "__main__":
    main()