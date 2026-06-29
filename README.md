# Storm Surge Project

本项目包含西北太平洋风暴潮 CNN 论文复现、厦门站数据处理，以及短时滚动预测的前期试验。

## 当前状态

- 原始厦门 GESLA 潮位和 1985 年 ERA20C 数据保留在 `data/`。
- 旧 `data/processed_xiamen_1985/` 使用了修正前的错误分潮标签，已于 2026-06-24 删除。
- 旧 `runs/xiamen_1985/` 基于错误标签训练，模型、指标和图片已同步删除。
- 后续正式复现必须使用当前修正后的 UTide 代码重新预处理，再训练 CNN。
- 2026-06-24 的短时预测脚本、汇报和结果统一放在 `short_term_forecast/`。

## 目录地图

```text
storm_surge/
├─ code_my/                 自己整理和修改的论文复现代码
├─ data/                    原始数据，不提交 Git
├─ docs/                    论文、步骤文档和汇报 PPT
├─ notebooks/               独立测试 notebook
├─ paper_reference/         论文作者原始代码及原始压缩包
├─ short_term_forecast/     本周短时滚动预测试验
├─ .agents/                 Codex/Agent 本地技能配置
├─ .claude/                 Claude 本地配置
├─ .idea/                   PyCharm/IDEA 工程配置
├─ .gitattributes           Git 属性配置
├─ .gitignore               Git 忽略规则
├─ requirements.txt         Python 依赖列表
└─ skills-lock.json         Agent 技能版本锁定文件
```

## 各目录说明

### `code_my/`

自己的复现工作，是当前主要代码目录。

- `Data_preprocessing.ipynb`：notebook 版数据预处理，包含 GESLA 清洗、UTide 分潮、ERA20C 裁剪/插值和样本构建。
- `Model_training.ipynb`：notebook 版 CNN 训练和评价。
- `xiamen/`：整理后的厦门站脚本版实现。
  - `01_preprocessing/`：模块化预处理代码。
  - `02_model_training/`：模块化 CNN 训练代码。
  - `01_xiamen_preprocess_all_in_one.py`：预处理单文件版本。
  - `02_xiamen_train_all_in_one.py`：训练单文件版本。
  - 目录内多个 README 分别说明模块版、两文件版和 GitHub 整理版。

当前正确的分潮关键逻辑在：

```text
code_my/xiamen/01_preprocessing/tide_processing.py
```

### `data/`

仅保存原始输入数据：

- `ERA20C_1985/`：1985 年 U10、V10、SLP 三个 ERA20C GRIB 文件。
- `xiamen_GESLA/`：厦门验潮站 GESLA 原始小时潮位，覆盖 1954-1997 年。

这里目前没有可信的预处理 `.npy`。需要重新运行修正版预处理生成。

### `docs/`

- `利用卷积神经网络重建西太平洋北部风暴潮的数据集.pdf`：复现依据的中文论文资料。
- `步骤.docx`：早期操作步骤记录。
- `风暴潮.pptx`：已有汇报 PPT。

### `notebooks/`

- `GPU环境测试.ipynb`：只用于检查 PyTorch、CUDA 和显卡是否可用，不参与模型训练流程。

### `paper_reference/`

- `SurgeReconstructed_code/`：论文作者提供或下载得到的原始 notebook 代码，用于对照，不直接修改。
- `surgeReconstructed_WNP.zip`：上述原始资料的压缩备份。

### `short_term_forecast/`

2026-06-24 的短时预测阶段成果，具体说明见该目录的 `README.md`。

- `short_term_rolling_baseline.py`：只使用历史增水的 Ridge 滚动基线，用于验证窗口和递归评价流程，尚未使用 CNN、U10、V10、SLP。
- `make_report_figures.py`：生成正式 CNN 滚动流程和现有数据处理展示图。
- `本周汇报_短时风暴潮预测.md`：任务理解、实验结果、限制和下一步计划。
- `results/`：本次试验的图、指标和 CSV，已被 Git 忽略。
- `figures_for_report/`：本周汇报建议使用的三张图，已被 Git 忽略。

正式短时模型计划使用 U10、V10、SLP 三类 `40×40` 气象网格 CNN，并融合增水历史状态；新预测的增水会回填到下一次预测窗口。

## 建议工作顺序

1. 先阅读 `docs/` 中的论文和 `paper_reference/SurgeReconstructed_code/`。
2. 以 `code_my/xiamen/` 为正式复现代码，确认数据路径后重新运行预处理。
3. 检查新生成风暴增水的均值、范围和日最大值，再训练 CNN。
4. 短时预测讨论和基线结果从 `short_term_forecast/` 继续迭代。

## 环境

实验室电脑建议使用 conda 环境 `jjq`：

```powershell
pip install -r requirements.txt
```

主要依赖包括 NumPy、Pandas、Matplotlib、scikit-learn、xarray、UTide、cfgrib、ecCodes、PyTorch 和 Jupyter。

## 数据与输出规则

`.grb`、`.npy`、`.csv`、`.pth`、训练图片和大型压缩文件不提交 Git。重新生成数据前，不要恢复已经确认错误的旧 `processed_xiamen_1985` 和 `runs/xiamen_1985` 文件。
