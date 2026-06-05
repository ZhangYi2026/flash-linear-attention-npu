#!/usr/bin/env bash
set -euo pipefail

mode="${1:---summary}"

if ! command -v npu-smi >/dev/null 2>&1; then
    echo "npu-smi is not available" >&2
    exit 1
fi

raw="$(npu-smi info 2>/dev/null || true)"
if [[ -z "$raw" ]]; then
    echo "npu-smi returned no data" >&2
    exit 1
fi

declare -a ids=()
declare -A names=()
declare -A health=()
declare -A free=()

while IFS= read -r line; do
    if [[ "$line" =~ ^\|[[:space:]]*([0-9]+)[[:space:]]+([^[:space:]\|]*[A-Za-z][^[:space:]\|]*)[[:space:]]*\|[[:space:]]*([A-Za-z]+) ]]; then
        id="${BASH_REMATCH[1]}"
        ids+=("$id")
        names["$id"]="${BASH_REMATCH[2]}"
        health["$id"]="${BASH_REMATCH[3]}"
    elif [[ "$line" =~ No[[:space:]]running[[:space:]]processes[[:space:]]found[[:space:]]in[[:space:]]NPU[[:space:]]+([0-9]+) ]]; then
        free["${BASH_REMATCH[1]}"]=1
    fi
done <<< "$raw"

if [[ "${#ids[@]}" -eq 0 ]]; then
    echo "No NPU device was found in npu-smi output" >&2
    exit 1
fi

soc_from_name() {
    local name="$1"
    case "$name" in
        *910B*|*910b*) echo "ascend910b" ;;
        *910_93*|*910*93*) echo "ascend910_93" ;;
        *950*) echo "ascend950" ;;
        *) echo "unknown" ;;
    esac
}

select_device() {
    local id
    for id in "${ids[@]}"; do
        if [[ "${health[$id]}" == "OK" && "${free[$id]:-0}" == "1" ]]; then
            echo "$id"
            return
        fi
    done
    for id in "${ids[@]}"; do
        if [[ "${free[$id]:-0}" == "1" ]]; then
            echo "$id"
            return
        fi
    done
    for id in "${ids[@]}"; do
        if [[ "${health[$id]}" == "OK" ]]; then
            echo "$id"
            return
        fi
    done
    echo "${ids[0]}"
}

selected="$(select_device)"
selected_soc="$(soc_from_name "${names[$selected]}")"
selected_free="${free[$selected]:-0}"

case "$mode" in
    --first-free|--selected)
        echo "$selected"
        ;;
    --soc)
        echo "$selected_soc"
        ;;
    --env)
        echo "NPU_AVAILABLE=1"
        echo "NPU_SELECTED_DEVICE=$selected"
        echo "NPU_SELECTED_NAME=${names[$selected]}"
        echo "NPU_SELECTED_HEALTH=${health[$selected]}"
        echo "NPU_SELECTED_FREE=$selected_free"
        echo "NPU_SOC=$selected_soc"
        ;;
    --summary)
        echo "Detected NPU devices:"
        for id in "${ids[@]}"; do
            echo "  - id=$id name=${names[$id]} health=${health[$id]} free=${free[$id]:-0} soc=$(soc_from_name "${names[$id]}")"
        done
        echo "Selected NPU: id=$selected name=${names[$selected]} health=${health[$selected]} free=$selected_free soc=$selected_soc"
        ;;
    *)
        echo "Usage: $0 [--summary|--env|--first-free|--selected|--soc]" >&2
        exit 2
        ;;
esac
