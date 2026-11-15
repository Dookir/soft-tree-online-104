# Soft Tree Online Ramping · v3.0.0

**分楼软决策树 + 分尺度在线寻优，可复现课程实验。**

本版冻结实验 `formal-v3-20261114-01`：从原始 **2,048** 行数据开始，独立执行 **20 轮 × 50 组**真实评估，最低真实 **R = 104.14030796010047**。目标 **R ≤104 尚未达到**。这里的真实 R 来自冻结的逆向评分后端，不声称运行了 Windows 原始 EXE。

适合做报告的材料：[实验报告](docs/REPORT.zh-CN.md) · [模型与算法](docs/METHOD.zh-CN.md) · [复现说明](docs/REPRODUCIBILITY.md) · [最优参数](results/formal-v3/best.json) · [逐轮结果](results/formal-v3/progress.csv)。

## 结果

各实验均使用1,000组新增真实评估；以下为单次实验对照，不能解释为多次重复的统计结论。

| 方法 | 最低真实 R（越低越好） |
|---|---:|
| **本版：软决策树 + 新寻优器** | **104.140308** |
| 原发布版v1 | 104.503950 |
| v2独立实验 | 105.284698 |
| 原软决策树在线流程 | 106.684391 |
| TabM 在线流程 | 107.018991 |
| 软决策树短程继承更新版 | 108.284816 |

v3相比原发布版v1的104.503950降低 **0.363642**。实验约4小时24分钟，训练/寻优在共享A100上运行，真实评估使用CPU。第16轮确认停滞后，从第17轮启用多步长扫描及通过影子验证的局部去偏。最优点为**第18轮第9组**：第三栋楼从degradation向peak转移0.01权重。最后一批反馈已进入最终重训，没有第21次评估。

本发布的20轮均从原始2048行**从头训练**，没有加载旧模型、旧反馈或旧最好点。原发布版的完整源码与记录继续固定在 [v1.0.0](https://github.com/Dookir/soft-tree-online-104/tree/v1.0.0)，本次固定为v3.0.0。

## 快速开始

支持 Windows x86-64、Linux x86-64 和 macOS arm64。固定CPU环境为 **Python3.10.21 / Torch2.6.0 / NumPy1.26.4 / float64 / 单线程**。

Windows：双击 `01_SETUP_WINDOWS.bat` 安装项目自己的环境，再双击 `02_REPRODUCE_WINDOWS.bat` 核验并导出本次最佳结果。

macOS / Linux：

```bash
python3 bootstrap.py
.venv/bin/python run.py doctor
.venv/bin/python run.py reproduce
```

首次安装可能联网；依赖已准备好时可直接运行。安装器使用固定版本与wheel哈希，不修改系统包。Windows命令中的解释器替换为 `.venv\Scripts\python.exe`。

`reproduce` 会逐项核验发布文件、原始CSV、20批输入与真实回传R、1,000行反馈索引及最优点，输出到 `outputs/reproduced/`。**这是对归档真实结果的复核和导出，不会重新调用评分器或声称重新完成训练。**

## v3的寻优逻辑

探索阶段保持冻结v1规则。至少返回500组反馈、连续两轮最好真实R改善均小于0.02后，下一轮才进入精修；本次触发于第16轮。

精修每批仍50组：模型低值4、四档转移12、楼宇重组6、联合扰动8、多步长扫描20。扫描步长0.0005/0.002/0.008/0.03/0.08，按供体约束截断、去重；已评估步长用其他方向或区间二分补齐，不通过插值赋真实R。

每次评分前保存中心锚定预测 `R(c)+p(q)-p(c)`。只有过去三个有效批次至少30个局部点、MAE降低至少20%、且前五组选点遗憾值不恶化，精修时才允许启用。当前批真实R只影响下一批选择；候选文件也有预反馈哈希保护。该检验局限于已评估候选，不代表校准的置信区间。

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

- `adaptive/`：3692参数模型、全量加权训练、冻结v1探索骨架、停滞门槛、多步长扫描、影子去偏与在线预算状态机。
- `adaptive/data/train.csv`：原始2048行；SHA256为 `877b5f8ee265245dd693148a5d7b11f28f48b426e43ac3ac1be8b7a934249023`。
- `results/formal-v3/models/v00/`：本实验从头训练的原始数据专用五种子模型；无需旧预训练存档。
- `results/formal-v3/`：全部1000组反馈、20批请求与响应、v00–v20全部模型、每版训练样本和权重、诊断、耗时与最终报告。
- `config/`、`requirements-common.lock`：Windows/Linux/macOS的固定CPU依赖及文件哈希。
- `MANIFEST.json`、`ORIGIN.json`：发布完整性清单、历史源码标识和打包变更说明。核心模型/损失/寻优代码保留实验版本；服务器路径已脱敏，入口增加了可移植适配。

测试：`.venv/bin/python run.py test`。测试只核验本地记录与数学契约，不发起真实评估。Windows启动脚本已提供；发布前实测平台记录见 [复现说明](docs/REPRODUCIBILITY.md)。

源码使用MIT许可证，数据和记录的来源及适用范围见 [DATA_NOTICE.md](DATA_NOTICE.md)。引用本项目时，请同时注明原始数据量、1,000组在线反馈预算及“单次实验、逆向后端”的条件。
