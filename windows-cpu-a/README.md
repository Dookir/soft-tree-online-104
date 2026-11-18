# Windows CPU 在线寻优

程序为三栋楼各选择七项权重。老师原版评分器接收完整的三楼组合并返回 R，分数越低越好。

已完成的实验使用原版 Windows EXE，从课程原始 2,048 行数据开始，完成 **20 轮、1,000 组**新评分。最优为 **R = 103.63530439231543**，来自第 20 轮第 5 组。

## 从头运行

1. 用 64 位 Windows，把项目解压到较短的可写路径，例如 `C:\ML\soft-tree`。进入 `windows-cpu-a`。
2. 双击 `01_SETUP.bat` 安装固定版本的 Python 3.10.21、NumPy 1.26.4、PyTorch 2.6.0+cpu。首次安装需要联网，此步骤会核对源码和数据并运行离线测试。
3. 从 [v4.1.0 发布附件](https://github.com/Dookir/soft-tree-online-104/releases/tag/v4.1.0) 下载原版 `solver.exe`，放到本目录的 `teacher_solver/solver.exe`。启动时会核对 SHA-256。
4. 双击 `03_FORMAL_TRAIN.bat`。程序将自动跑 20 轮，每轮用原版评分器评估 50 组。运行结束后，`best-result.json` 保存最低实测 R，`best-weights.json` 保存完整权重。
5. 如果中途意外中断，保留整个目录，再用 `04_RESUME.bat`。已调用但缺少可校验返回的评分会保留记录，程序会停止以免重复评分。

新运行的数据和模型写入 `project/sessions/formal/`。训练使用本次已有的实测反馈。发布包中的归档位于单独的 `evidence/` 目录。

## 每轮怎样选点

1. 用原始数据和已经返回的评分，从头训练五个模型。模型用于预测哪组权重值得尝试。
2. 前 6 轮用共同的开局规则找起点。从第 7 轮开始，优先跟进已取得较低 R 的重组点。程序会在保持权重和不变的情况下，从某项减一点给另一项，再实测这次调整是否降低 R。
3. 稳定保留必要的学习样本和原有精修候选。第 19–20 轮优先尝试有实测改善的方向，方向记录绑定到原来的精确父点。
4. EXE 返回的 50 个实测结果逐项对应本轮请求，并加入下一轮训练。20 轮累计 710 组冲分和 290 组学习。最后一轮结束后再用全部 3,048 行训练最终模型。

核心实现在 [训练循环](project/adaptive/runner.py)、[候选入口](project/adaptive/acquisition.py) 和 [重组点跟进规则](project/adaptive/paired_selection.py)。

## 已发布模型与训练产出

| 文件 | 内容 |
|---|---|
| [evidence/best-result.json](evidence/best-result.json) | 最优分数、候选编号、评分后端 |
| [evidence/best-weights.json](evidence/best-weights.json) | 可直接送入原版评分器的完整权重 |
| [evidence/round-results.csv](evidence/round-results.csv) | 每轮当批最低分、累计最低分和预测误差 |
| [evidence/formal/rounds/](evidence/formal/rounds/) | 每轮候选及预测、精确请求、原版 EXE 返回、反馈和耗时 |
| [evidence/formal/models/](evidence/formal/models/) | v00 到 v20 的 21 版训练集、每版五个模型、模型 SHA-256 |
| [evidence/formal/learning-audits/](evidence/formal/learning-audits/) | 四个诊断模型及对下一批候选的预测比较 |
| [evidence/strategy-feedback.json](evidence/strategy-feedback.json) | 特化点的实测 R、父点和相对父点的改善 |

模型为 NumPy `.npz` 检查点。v00 只使用初始数据，v20 使用全部数据。原始运行的 105 个主模型与 4 个诊断模型均已归档。

## 离线核验和重训

安装环境后双击 `02_VERIFY_ARCHIVE.bat`，可核验 1,000 组配对、输入与评分文件、训练集和模型哈希。此步骤不需要评分器。公开归档的配置和账本使用相对路径，保留了输入、返回和模型文件的原始内容。

在本目录打开 PowerShell 可执行以下命令。

~~~powershell
.\02_VERIFY_ARCHIVE.bat --replay-selection
.\project\.venv\Scripts\python.exe refit.py --round 0
.\project\.venv\Scripts\python.exe refit.py --round 20
~~~

第一条使用归档模型和当时可见的反馈重新计算特化选点，要求与归档一致。后两条分别重训初始和最终版本，输出两组模型在固定输入上的预测差异及模型哈希是否一致。离线重训遵循归档的真实输入和分数。重新从头在线运行时，硬件浮点差异可能影响选点轨迹和最终 R。
