# Storm Surge CNN Reconstruction

这个仓库用于复现论文 **A dataset of storm surge reconstructions in the Western North Pacific using CNN** 的核心流程。

当前先以厦门站 `Xiamen`、1985 年为例，按照 notebook 形式完成：

1. GESLA 潮位数据读取与清洗；
2. UTide 潮汐分离，得到 storm surge；
3. ERA20C 气象场读取、裁剪、插值和标准化；
4. 构建 CNN 输入数据；
5. 划分训练集和验证集；
6. 训练 CNN 并做验证分析。

## 推荐阅读顺序

主要看这两个 notebook：

```text
code_my/Data_preprocessing.ipynb
code_my/Model_training.ipynb
```

第一个 notebook 负责从原始数据生成 CNN 输入。

第二个 notebook 负责读取 `.npy` 数据、训练模型、验证、画图和保存结果。

## 数据预处理 notebook

文件：

```text
code_my/Data_preprocessing.ipynb
```

它会完成：

- 读取厦门站 GESLA 潮位文件；
- 跳过 `#` 开头的元数据；
- 解析 `date`、`time`、`sea_level`、`qc_flag`、`use_flag`；
- 清理缺测值和明显异常值；
- 使用 UTide 做潮汐分离；
- 计算 `storm surge = observed sea level - predicted tide`；
- 按日提取 `daily maximum storm surge` 作为标签 `y`；
- 读取 ERA20C 的 U10、V10、SLP；
- 裁剪厦门站周围 `10° x 10°` 区域；
- 插值到 `40 x 40` 网格；
- 构建 CNN 输入：

```text
单个样本 shape = (48, 40, 40)
总体 X shape = (N, 48, 40, 40)
y shape = (N,)
```

其中 `48 = 2 天 x 每天 8 个 3小时时间片 x 3 个变量`。

默认输出目录：

```text
data/processed_xiamen_1985/
```

生成文件：

```text
X_train.npy
y_train.npy
X_val.npy
y_val.npy
dates_train.npy
dates_val.npy
y_original.npy
dates_all.npy
y_scaler.json
```

## 模型训练 notebook

文件：

```text
code_my/Model_training.ipynb
```

它会完成：

- 检查预处理输出文件是否存在；
- 检查 `X_train` 和 `X_val` 是否为 `(N, 48, 40, 40)`；
- 用 PyTorch `Dataset` 和 `DataLoader` 读取 `.npy`；
- 定义适配 48 通道输入的 CNN；
- 训练模型；
- 使用验证集选择最佳模型；
- 计算 RMSE、MAE、R2、相关系数；
- 保存预测结果和图。

默认训练输出目录：

```text
runs/xiamen_1985/
```

生成文件：

```text
best_model.pth
training_history.csv
metrics.json
val_predictions.csv
loss_curve.png
val_timeseries.png
val_scatter.png
```

这些文件都不会上传到 GitHub。

## 依赖环境

建议在实验室远程电脑的 conda 环境 `jjq` 中运行。

如果缺包，可以执行：

```bash
pip install -r requirements.txt
```

需要的主要库：

```text
numpy
pandas
matplotlib
scikit-learn
xarray
utide
cfgrib
eccodes
torch
jupyter
```

## 数据路径

当前 notebook 示例中，本地测试路径是：

```text
E:\AAAqian\code\storm_surge\data
```

实验室电脑上的真实路径是：

```text
ERA20C_DIR = F:\ERA20C
GESLA_DIR = F:\GESLA\GESLA3
SITE_FILE = F:\GESLA\GESLA3\xiamen-376a-chn-uhslc
```

如果在实验室电脑运行，需要在 `Data_preprocessing.ipynb` 的路径设置 Cell 中改成实验室路径。

## 注意

- 1985 单年样本较少，模型结果主要用于跑通流程，不代表论文最终效果。
- 要更接近论文结果，需要使用多年 ERA20C 和 GESLA 数据重新预处理和训练。
- 不要提交 `.grb`、`.npy`、`.csv`、`.pth`、训练图片或任何大数据文件。
