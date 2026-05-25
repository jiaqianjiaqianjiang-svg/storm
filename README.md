# Storm Surge CNN Reconstruction

本仓库用于复现论文 **A dataset of storm surge reconstructions in the Western North Pacific using CNN** 的核心流程。当前版本先以厦门站 `Xiamen` 为例，完成：

1. GESLA 潮位数据读取与清洗；
2. UTide 潮汐分离，得到 storm surge；
3. ERA20C 的 U10、V10、SLP 读取、裁剪、插值和标准化；
4. 构建 CNN 输入 `X shape = (N, 48, 40, 40)` 与标签 `y shape = (N,)`；
5. 按时间顺序划分训练集和验证集；
6. 训练并验证厦门站 CNN 风暴潮重建模型。

本仓库只提交代码、说明和 notebook 示例。真实数据、`.npy` 输出、模型权重和训练图表都不会提交到 GitHub。

## 数据路径

实验室远程电脑上的真实数据路径：

```python
ERA20C_DIR = r"F:\ERA20C"
GESLA_DIR = r"F:\GESLA\GESLA3"
SITE_FILE = r"F:\GESLA\GESLA3\xiamen-376a-chn-uhslc"
SITE_NAME = "Xiamen"
SITE_LAT = 24.45
SITE_LON = 118.067
```

ERA20C 目录结构应为：

```text
F:\ERA20C\10U\*.grb
F:\ERA20C\10V\*.grb
F:\ERA20C\SLP\*.grb
```

## 环境

建议在实验室远程电脑的 conda 环境 `jjq` 中运行：

```bash
pip install -r requirements.txt
```

当前代码使用：

```text
numpy pandas matplotlib scipy scikit-learn tqdm xarray netCDF4 utide cfgrib eccodes torch
```

## 预处理

如果使用 notebook：

```text
code_my/Data_preprocessing.ipynb
```

该 notebook 已经补齐到数据划分阶段，会生成：

```text
data/processed_xiamen_1985/X_train.npy
data/processed_xiamen_1985/y_train.npy
data/processed_xiamen_1985/X_val.npy
data/processed_xiamen_1985/y_val.npy
data/processed_xiamen_1985/dates_train.npy
data/processed_xiamen_1985/dates_val.npy
data/processed_xiamen_1985/y_original.npy
data/processed_xiamen_1985/dates_all.npy
```

如果使用模块化脚本：

```bash
python src/preprocess_xiamen.py --start-year 1985 --end-year 1985
```

模块化脚本默认输出到：

```text
outputs/xiamen/
```

## CNN 输入说明

对某一天 `D` 的 daily maximum storm surge，输入使用 `D-1` 和 `D` 两天的 ERA20C 气象场。

- ERA20C 为 3 小时分辨率；
- 每天 8 个时间片；
- 两天共 16 个时间片；
- 变量为 U10、V10、SLP；
- 每个变量插值到 `40 x 40`；
- 最终单个样本 `X` 的 shape 为 `(48, 40, 40)`。

训练脚本会检查 `X_train.npy` 是否满足 `(N, 48, 40, 40)`，如果预处理输出不正确会直接报错。

## 模型训练

先完成预处理，然后运行：

```bash
python src/train_xiamen.py --data-dir data/processed_xiamen_1985 --epochs 100
```

如果你使用模块化预处理脚本生成了 `outputs/xiamen/`，也可以运行：

```bash
python src/train_xiamen.py --data-dir outputs/xiamen --epochs 100
```

如果不传 `--data-dir`，训练脚本会按顺序自动寻找：

1. `data/processed_xiamen_1985`
2. `outputs/xiamen`

常用参数：

```bash
python src/train_xiamen.py ^
  --data-dir data/processed_xiamen_1985 ^
  --run-dir runs/xiamen_1985 ^
  --epochs 100 ^
  --batch-size 16 ^
  --lr 0.001 ^
  --device auto
```

训练输出默认保存到：

```text
runs/xiamen_1985/
```

包括：

```text
best_model.pth
last_loaded_best_model.pth
training_history.csv
metrics.json
val_predictions.csv
loss_curve.png
val_timeseries.png
val_scatter.png
```

这些文件都在 `.gitignore` 中，不会上传到 GitHub。

## 验证指标

训练完成后会在验证集上输出：

- MSE
- RMSE
- MAE
- R2
- 相关系数 corr

其中 `metrics.json` 同时保存标准化单位和原始 storm surge 单位下的指标。

## 注意事项

- 1985 单年样本很少，训练结果只能用于流程验证，不能代表论文最终重建效果。
- 论文级复现应使用更长时间范围，例如厦门站完整可用期，再重新划分训练/验证。
- `code_my/Data_preprocessing.ipynb` 中的标签标准化是对全部 `y` 拟合；训练脚本会通过 `y_original.npy` 自动反推出反标准化参数。
- 不要提交 `.grb`、`.npy`、`.csv`、`.pth`、训练图或模型权重。
