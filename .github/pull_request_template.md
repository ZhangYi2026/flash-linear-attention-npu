# Pull Request 模板

## 关联 Issue / 背景

- 关联 Issue:
- 背景与目标:
- 本 PR 解决的问题:

## 变更类型

- [ ] Bug 修复
- [ ] 新功能 / 新算子
- [ ] 算子性能优化
- [ ] 算子精度 / 稳定性修复
- [ ] Tiling / InferShape / OpApi 调整
- [ ] 构建 / CI / 脚本变更
- [ ] 文档 / 示例 / 测试更新

## 算子责任范围

请明确本 PR 修改的算子责任边界，便于审查、验收和后续维护。

- 涉及算子名称:
- 涉及目录 / 文件:
- 责任人 / 提交人:
- 修改范围:
  - [ ] `op_host` / 算子定义
  - [ ] `InferShape` / 参数校验
  - [ ] `Tiling` 策略
  - [ ] `op_kernel` / Ascend C Kernel
  - [ ] `op_api` / aclnn 接口
  - [ ] `torch_custom` 适配
  - [ ] `Triton` 算子
  - [ ] 单算子测试
  - [ ] `example / ST` 测试
  - [ ] 文档
- 输入 / 输出 / shape / dtype / format 支持范围:
- 明确不包含的范围:
- 是否影响公共模块或其他算子:
  - [ ] 否
  - [ ] 是，请说明影响面、兼容策略和回归范围:

## 版本与产品编译矩阵

本 PR 必须保证配套版本满足以下编译要求。当前工程中产品与 `--soc` 参数的对应关系如下:

| 产品 | `--soc` |
| --- | --- |
| A2 | `ascend910b` |
| A3 | `ascend910_93` |
| A5 | `ascend950` |

## 验证方法

请按本节方法执行验证，并把实际命令、结果和日志填写到后续矩阵中。`<op_name>` 替换为本 PR 修改的算子名；多个算子用逗号分隔，或逐个填写。

### 1. 环境确认

在每套 CANN / 产品环境中先确认基础信息。

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cat /usr/local/Ascend/ascend-toolkit/latest/opp/version.info
npu-smi info
```

记录项:

- CANN 版本:
- 产品 / 芯片类型:
- 机器或 CI 链接:

### 2. 编译验证

CANN 8.5 及以上需要验证 A2 / A3:

```sh
bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=fla_npu
bash build.sh --pkg --soc=ascend910_93 --ops=<op_name> --vendor_name=fla_npu
```

CANN 9.0 及以上需要验证 A2 / A3 / A5:

```sh
bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=fla_npu
bash build.sh --pkg --soc=ascend910_93 --ops=<op_name> --vendor_name=fla_npu
bash build.sh --pkg --soc=ascend950 --ops=<op_name> --vendor_name=fla_npu
```

通过标准:

- 命令退出码为 0
- `build_out/` 下生成对应 `.run` 包
- 编译日志无 `ERROR` / `FAILED` / 关键 warning

### 3. 修改算子 test 验证

根据本 PR 修改范围选择对应测试入口；涉及多个入口时均需执行。

工程 UT:

```sh
bash build.sh -u --ops=<op_name> --soc=<soc>
```

按模块精确验证:

```sh
bash build.sh --ophost_test --ops=<op_name> --soc=<soc>
bash build.sh --opapi_test --ops=<op_name> --soc=<soc>
bash build.sh --opkernel_test --ops=<op_name> --soc=<soc>
```

`torch_custom` 适配验证:

```sh
cd torch_custom/fla_npu
bash build.sh
cd test
python3 test_npu_<op_name>.py
# 或执行全部 GDN 单算子测试
bash test.sh
```

`examples/fast_kernel_launch_example` 验证:

```sh
cd examples/fast_kernel_launch_example
bash build_and_test.sh <op_name>
# 不传 <op_name> 时执行该 example 下全部测试
bash build_and_test.sh
```

如果对应算子目录下已有专用测试脚本或 ATK 用例，请填写实际脚本命令，并说明覆盖的 shape / dtype / format / 边界场景。

通过标准:

- 命令退出码为 0
- 测试日志显示 `PASS` / `passed` / `execute samples success`
- 精度误差满足该算子 README、测试脚本或需求说明中的阈值

### 4. 整体 example ST 验证

整体 example ST 用于确认修改没有破坏 GDN 端到端调用链。请在 A2 / A3 / A5 对应环境分别执行。

```sh
python examples/flash_gated_delta_rule.py
```

如本 PR 修改了 aclnn example 或新增了算子 example，同时执行:

```sh
bash build.sh --run_example <op_name> eager cust --vendor_name=fla_npu --soc=<soc>
```

通过标准:

- 命令退出码为 0
- 端到端结果与 golden / PyTorch 参考实现一致
- 日志无精度异常、运行时异常、fallback 异常或 device error

### CANN 8.5 及以上

要求: 可以编译出 A2 / A3 目标产物。

| 产品 | `--soc` | CANN 实际版本 | 实际执行命令 | 结果 | 产物 / 日志 |
| --- | --- | --- | --- | --- | --- |
| A2 | `ascend910b` |  | `bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=fla_npu` | 通过 / 未通过 / 未执行 |  |
| A3 | `ascend910_93` |  | `bash build.sh --pkg --soc=ascend910_93 --ops=<op_name> --vendor_name=fla_npu` | 通过 / 未通过 / 未执行 |  |

### CANN 9.0 及以上

要求: 可以编译出 A2 / A3 / A5 目标产物。

| 产品 | `--soc` | CANN 实际版本 | 实际执行命令 | 结果 | 产物 / 日志 |
| --- | --- | --- | --- | --- | --- |
| A2 | `ascend910b` |  | `bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=fla_npu` | 通过 / 未通过 / 未执行 |  |
| A3 | `ascend910_93` |  | `bash build.sh --pkg --soc=ascend910_93 --ops=<op_name> --vendor_name=fla_npu` | 通过 / 未通过 / 未执行 |  |
| A5 | `ascend950` |  | `bash build.sh --pkg --soc=ascend950 --ops=<op_name> --vendor_name=fla_npu` | 通过 / 未通过 / 未执行 |  |

如结果为“未通过”或“未执行”，请在日志列填写原因、当前阻塞点、责任人和预计补齐时间。

## 修改算子测试

本 PR 修改到的算子必须完成对应 test 测试，并在 A2 / A3 / A5 覆盖验证结果。

| 产品 | `--soc` | 实际执行命令 | 结果 | 日志 / 精度结论 |
| --- | --- | --- | --- | --- |
| A2 | `ascend910b` | `bash build.sh -u --ops=<op_name> --soc=ascend910b` | 通过 / 未通过 / 未执行 |  |
| A3 | `ascend910_93` | `bash build.sh -u --ops=<op_name> --soc=ascend910_93` | 通过 / 未通过 / 未执行 |  |
| A5 | `ascend950` | `bash build.sh -u --ops=<op_name> --soc=ascend950` | 通过 / 未通过 / 未执行 |  |

- 精度对比:
- 性能对比:
- 边界 / 异常场景:
- 未覆盖风险:

## 整体 Example ST 验证

整体 example 的 ST 测试必须在 A2 / A3 / A5 通过。

| 产品 | `--soc` | 实际执行命令 | 结果 | 日志 / 结论 |
| --- | --- | --- | --- | --- |
| A2 | `ascend910b` | `python examples/flash_gated_delta_rule.py` | 通过 / 未通过 / 未执行 |  |
| A3 | `ascend910_93` | `python examples/flash_gated_delta_rule.py` | 通过 / 未通过 / 未执行 |  |
| A5 | `ascend950` | `python examples/flash_gated_delta_rule.py` | 通过 / 未通过 / 未执行 |  |

- 端到端场景说明:
- 与本 PR 修改算子的关联:
- 未覆盖原因及补充计划:

## 兼容性、风险与回退

- 兼容性说明:
- 风险点:
- 回退方案:
- 回退责任确认:
  - [ ] 若本 PR 未满足上述编译 / 测试要求，或引入阻塞性 bug，PR 提交人承诺第一时间回退或配合维护者完成回退
  - [ ] 回退后会同步说明影响范围、恢复版本、后续修复计划

## Checklist

- [ ] 已关联 Issue，或说明无需 Issue 的原因
- [ ] 已明确本 PR 的算子责任范围和不包含范围
- [ ] CANN 8.5 及以上已验证 A2 / A3 可以编译通过
- [ ] CANN 9.0 及以上已验证 A2 / A3 / A5 可以编译通过
- [ ] 已验证修改算子的对应 test 在 A2 / A3 / A5 通过
- [ ] 已验证整体 example ST 在 A2 / A3 / A5 通过
- [ ] 已完成必要的精度、性能或回归验证
- [ ] 已确认没有引入与本 PR 无关的格式化或大规模重构
- [ ] 已更新相关 README、算子说明或变更记录，如适用
- [ ] PR 标题已使用合适类型标签，如 `feat:`, `fix:`, `perf:`, `docs:`
- [ ] 已确认如不满足准入要求或引入 bug，将第一时间回退

## 其他信息

在这里补充评审人需要关注的实现细节、限制条件或后续计划。
