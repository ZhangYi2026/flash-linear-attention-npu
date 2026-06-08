#!/bin/bash

# 构建自定义算子
code_path="/data/huangjunzhe/GDN/perf/flash-linear-attention-npu/"
custom_path="/data/zs/custom"
source /data/zs/run/Ascend/ascend-toolkit/set_env.sh
cd ${code_path}
rm -rf ${code_path}/build
rm -rf ${code_path}/build_out

echo "[INFO] Running build.sh ..."
bash ./build.sh --pkg --ops=chunk_bwd_dqkwg
if [ $? -ne 0 ]; then
    echo "[ERROR] build.sh failed!" >&2
    exit 1
fi

# 安装自定义算子包
echo "[INFO] Installing custom operator package ..."
/data/huangjunzhe/GDN/perf/flash-linear-attention-npu/build/cann-ops-transformer-custom_linux-aarch64.run --install-path=${custom_path}
if [ $? -ne 0 ]; then
    echo "[ ERROR ] Operator package installation failed!" >&2
    exit 1
fi

# 设置 CANN 环境变量
echo "[INFO] Sourcing CANN environment ..."
source /data/zs/run/8.5/cann-8.5.0/set_env.sh
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to source CANN environment!" >&2
    exit 1
fi

# 设置自定义算子环境变量
echo "[INFO] Sourcing custom operator environment ..."
source ${custom_path}/vendors/custom_transformer/bin/set_env.bash
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to source custom operator environment!" >&2
    exit 1
fi

# 设置 LD_LIBRARY_PATH（这一步通常不会失败，但为一致性也保留）
export LD_LIBRARY_PATH=/data/zs/run/8.5/cann-8.5.0/aarch64-linux/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH

# 运行 msprof 性能分析
# cd /data/huangjunzhe/GDN/github/flash-linear-attention-npu/chunk_gated_delta_rule/chunk_bwd_dqkwg/tests
echo "[INFO] Running msprof op simulator ..."
msprof op simulator python ${code_path}/chunk_gated_delta_rule/chunk_bwd_dqkwg/tests/sim/pta_simulator.py
if [ $? -ne 0 ]; then
    echo "[ERROR] msprof command failed!" >&2
    exit 1
fi

echo "[INFO] All steps completed successfully."