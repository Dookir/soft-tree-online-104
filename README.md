# Soft Tree Online Ramping · v4.0.2

**分楼软决策树 + 分尺度在线寻优，可复现课程实验。**

本版冻结实验 `formal-v4-20261114-01`：从原始 **2,048** 行数据开始，独立执行 **20 轮 × 50 组**真实评估，最低真实 **R = 103.85699706606101**。目标 **R ≤104 已达到**。这里的真实 R 来自冻结的逆向评分后端，不声称运行了 Windows 原始 EXE。

适合做报告的材料：[实验报告](docs/REPORT.zh-CN.md) · [模型与算法](docs/METHOD.zh-CN.md) · [复现说明](docs/REPRODUCIBILITY.md) · [最优参数](results/formal-v4/best.json) · [逐轮结果](results/formal-v4/progress.csv)。

## 结果

各实验均使用1,000组新增真实评估；以下为单次实验对照，不能解释为多次重复的统计结论。

| 方法 | 最低真实 R（越低越好） |
|---|---:|
| **本版：软决策树 + 新寻优器** | **103.856997** |
| 前版v3 | 104.140308 |
| 原发布版v1 | 104.503950 |
| v2独立实验 | 105.284698 |
| 原软决策树在线流程 | 106.684391 |
| TabM 在线流程 | 107.018991 |
| 软决策树短程继承更新版 | 108.284816 |

v4相比v3的104.140308降低 **0.283311**，低于104目标约0.143003。实验约6小时43分钟；最优点为**第20轮第49组**，从第二栋楼的ramping向resilience转移0.0005权重。最后一批反馈已进入最终重训，没有第21次评估。

本实验从原始2048行独立冷启动，所有轮次按原配置从头全量加权重训；没有加载旧模型、旧反馈或旧最好点。历史完整版本保留在 [v1.0.0](https://github.com/Dookir/soft-tree-online-104/tree/v1.0.0) 和 [v3.0.0](https://github.com/Dookir/soft-tree-online-104/tree/v3.0.0)。

## v4.0.2：降低运行随机性

本补丁增强新运行的确定性；上面的103.856997仍是原v4.0.0实验的归档成绩，没有重新评分。原版保留在[v4.0.0](https://github.com/Dookir/soft-tree-online-104/tree/v4.0.0)。

- 启动前固定Python哈希、UTF-8及数学库线程；必要时自动重启解释器，使设置真正生效。
- 每个训练作业固定Python、NumPy和PyTorch随机数；原有五种子、打乱顺序和候选随机数种子保留。
- 默认CPU、float64、单线程，严格确定性算法；关闭可选加速路径、TF32、cuDNN、MKLDNN及可选einsum路径优化。训练和寻优的Adam显式使用普通逐张量实现。
- 固定相同数值时的排序，并记录生效的运行设置；Git检出保留原始字节，避免Windows换行转换破坏完整性核验。

```bash
.venv/bin/python run.py check-determinism --output outputs/determinism
```

此检查在两个独立CPU进程中，使用全部原始2048行、五种子各3轮短训练及每阶段8步寻优，故意传入不同的哈希种子、线程设置和全局随机状态，检查模型、预测和50组候选是否完全一致。它不调用评分器、不消耗评估预算，**不能替代完整训练或跨系统效果验收**。Windows使用 `.venv\Scripts\python.exe`。可加 `--reference validation/cpu-smoke-reference.npz` 与随包提供的Mac CPU短流程参考签名比较。两台电脑可比较导出的`run-1-signature.npz`，通过`--reference <另一台电脑的签名文件>`测量差异。

为减少计算路径差异，正式复现建议保持CPU默认设置。GPU仍需显式指定；严格确定性不受支持的运算会报错。随机探索仍保留，只使用固定随机序列；关闭这些运行优化可能增加耗时，跨系统数值差异也不能完全消除。

## 快速开始

支持 Windows x86-64、Linux x86-64 和 macOS arm64。固定CPU环境为 **Python3.10.21 / Torch2.6.0 / NumPy1.26.4 / float64 / 单线程**。

Windows：双击 `01_SETUP_WINDOWS.bat` 安装项目自己的环境，再双击 `02_REPRODUCE_WINDOWS.bat` 核验并导出本次最佳结果。双击 `04_CHECK_DETERMINISM_WINDOWS.bat` 可运行两进程一致性检查，并与随包参考签名比较。

macOS / Linux：

```bash
python3 bootstrap.py
.venv/bin/python run.py doctor
.venv/bin/python run.py reproduce
```

首次安装可能联网；依赖已准备好时可直接运行。安装器使用固定版本与wheel哈希，不修改系统包。Windows命令中的解释器替换为 `.venv\Scripts\python.exe`。

`reproduce` 会逐项核验发布文件、原始CSV、20批输入与真实回传R、1,000行反馈索引及最优点，并从最优点保存的8周净负荷重算R，输出到 `outputs/reproduced/`。**这是对归档真实结果的复核和导出，不会重新调用评分器或声称重新完成训练。**

## v4的寻优逻辑

每轮50组分为冲分候选与学习候选，按固定计划逐步偏向冲分：

| 轮次 | 冲分 | 学习 |
|---|---:|---:|
| 1–6 | 30 | 20 |
| 7–14 | 35 | 15 |
| 15–19 | 40 | 10 |
| 20 | 50 | 0 |

累计710组冲分、290组学习。冲分候选保留模型低值、权重转移、楼宇重组、联合扰动等来源；学习候选包含成对局部比较、已知真实变化区间的新中点、局部有竞争力且模型分歧大的点。所有新点均真实评估，不用插值制造标签。角色只控制选点，不作为模型输入或额外权重。

保留v3的停滞后精修及影子去偏门槛；**本次运行未触发精修，也未在候选选择中使用去偏**。第18轮首次低于104，第20轮两组补充扫描中的一组取得最终最好成绩。

在第4/8/12/16轮后，另训单种子诊断模型，排除刚返回的学习组，比较对下一批实际候选的预反馈预测。前两次加入学习组后的MAE较差，后两次较好；这些是条件性诊断，不能推断所有学习点都有价值。详情见报告。

## 重新训练与寻优

```bash
# 仅用原始2048行，从固定种子重新训练五个模型（不是加载最终模型）
.venv/bin/python run.py train --round 0 --from-scratch --output outputs/original-refit

# 用原始2048行 + 本实验1000行反馈训练最终模型
.venv/bin/python run.py train --round 20 --output outputs/final-refit

# 顺序重训v00至v20；每个版本只能看到当时已返回的反馈，不新增评分
.venv/bin/python run.py replay --from-scratch --output outputs/replay

# 用归档最终模型按冻结规则提出50个新候选；只有预测R，不做真实评估
.venv/bin/python run.py optimize --round 20 --output outputs/proposals
```

预算用完后的 `optimize --round 20` 复用末轮50组冲分策略，仅生成未评分提案，不执行第21次评估。

完整重训耗时取决于CPU/GPU。GPU可用 `--device cuda --gpu 0`；重训后会报告相对归档模型的固定探针预测差异。**跨设备强调结构、数据、配置与数值行为的保真，不保证逐位相同或新一轮在线轨迹完全相同。**

`replay`使用归档的精确输入/R配对重建训练过程，不把旧R赋给重新生成的相近权重。重新运行寻优得到的新参数，需要评分器返回新R后才能称为真实成绩。

## 接入评分器，重新跑在线20轮

学习程序只依赖原始CSV和当前实验自己得到的标量反馈；评分器通过标准输入/输出连接。仓库不包含原闭源EXE、恢复字节码、设备运行数据或第三方温度模型。它们不属于本项目自行实现的学习与寻优源码；**本仓库可独立复核历史结果、训练和生成候选，新的真实评估需自行提供兼容后端。**

将评分器启动命令写成UTF-8 JSON数组文件，例如 `solver-command.json`：

```json
["C:\\your-evaluator\\solver.exe"]
```

评分器读取 `{"weights":[...50组3×7权重...]}`，只向stdout输出 `{"R":[...50个有限非负数...]}`，日志写stderr。无需shell拼接，路径可以包含空格。

```bash
.venv/bin/python run.py online --backend external --solver-command solver-command.json --session sessions/my-run --device cpu
```

在项目根目录生成持久预算账本，最多20次、每次50组。未知状态的失败调用不会自动重试。恢复时使用完全相同参数并加 `--resume`。不要修改或删除运行中的账本。

已拥有逆向运行包时，也可使用仓库提供的 `solver_bridge.py` 和 `--backend recovered --workflow <你的运行包> --evaluator-python <独立Python3.11>`。桥接器保留原JSON协议；这不需要把运行包发布到GitHub。

## 仓库内容

- `adaptive/`：3692参数模型、全量加权训练、冻结v1探索骨架、停滞门槛、多步长扫描、影子去偏、信息采样、训练价值诊断与在线预算状态机。
- `adaptive/data/train.csv`：原始2048行；SHA256为 `877b5f8ee265245dd693148a5d7b11f28f48b426e43ac3ac1be8b7a934249023`。
- `results/formal-v4/models/v00/`：本实验从头训练的原始数据专用五种子模型；无需旧预训练存档。
- `results/formal-v4/`：全部1000组反馈、20批请求与响应、v00–v20全部105个主模型及4个单种子诊断模型、每版训练样本和权重、诊断、耗时与最终报告。
- `config/`、`requirements-common.lock`：Windows/Linux/macOS的固定CPU依赖及文件哈希。
- `MANIFEST.json`、`ORIGIN.json`：发布完整性清单、历史源码标识和打包变更说明。核心模型/损失/寻优代码保留实验版本；服务器路径已脱敏，入口增加了可移植适配。

测试：`.venv/bin/python run.py test`。测试只核验本地记录与数学契约，不发起真实评估。Windows启动脚本已提供；发布前实测平台记录见 [复现说明](docs/REPRODUCIBILITY.md)。

源码使用MIT许可证，数据和记录的来源及适用范围见 [DATA_NOTICE.md](DATA_NOTICE.md)。引用本项目时，请同时注明原始数据量、1,000组在线反馈预算及“单次实验、逆向后端”的条件。
