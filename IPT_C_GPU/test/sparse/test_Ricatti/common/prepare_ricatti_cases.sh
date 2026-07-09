#!/usr/bin/env bash
set -euo pipefail

ROOT="${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
MANIFEST="${1:-$ROOT/test/sparse/test_Ricatti/common/cases_k10.txt}"
INITIAL_ROOT="$ROOT/test/sparse/test_initial"
RICATTI_ROOT="$ROOT/test/sparse/test_Ricatti"

mkdir -p "$RICATTI_ROOT/common"

while read -r molecule k_value; do
    if [ -z "${molecule:-}" ] || [[ "$molecule" == \#* ]]; then
        continue
    fi

    source_env="$INITIAL_ROOT/$molecule/case.env"
    target_dir="$RICATTI_ROOT/$molecule"
    target_env="$target_dir/case.env"
    run_file="$target_dir/run_K${k_value}_no_davidson.sh"

    if [ ! -f "$source_env" ]; then
        echo "missing source case env: $source_env" >&2
        continue
    fi

    mkdir -p "$target_dir"
    cp "$source_env" "$target_env"
    {
        echo "export IPT_MATRIX_CACHE=\"\$ROOT/test/sparse/test_initial/$molecule/${molecule}_csc.bin\""
        echo "export IPT_LOAD_MATRIX=1"
        echo "export IPT_SAVE_MATRIX=1"
        echo "export IPT_DAVIDSON_ENRICH=0"
    } >> "$target_env"

    cat > "$run_file" <<EOF
#!/usr/bin/env bash
set -euo pipefail
ROOT="\${IPT_C_ROOT:-/fs1/home/nudt_liujie/ftt/IPT_C_GPU}"
export IPT_RESULT_BASE="\${IPT_RESULT_BASE:-\$ROOT/results/sparse/test_Ricatti}"
export IPT_LOG_BASE="\${IPT_LOG_BASE:-\$ROOT/logs/sparse/test_Ricatti}"
export RUN_PRIMME="\${RUN_PRIMME:-0}"
export RUN_IPT="\${RUN_IPT:-1}"
export IPT_DAVIDSON_ENRICH=0
bash "\$ROOT/test/sparse/test_Ricatti/common/run_ricatti_case.sh" "\$ROOT/test/sparse/test_Ricatti/$molecule/case.env" "$k_value"
EOF
    chmod +x "$run_file"
done < "$MANIFEST"
