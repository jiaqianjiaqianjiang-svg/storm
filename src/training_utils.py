"""训练与验证阶段的通用工具函数。"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class TargetScaler:
    """标签标准化参数。

    预处理 notebook 里保存的 ``y_train.npy`` 和 ``y_val.npy`` 是标准化后的标签。
    为了画图和计算真实单位的 RMSE/MAE，需要把预测值反标准化。
    """

    mean: float
    std: float

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """把标准化值还原到原始 storm surge 单位。"""

        return values * self.std + self.mean


class SurgeNpyDataset(Dataset):
    """从 `.npy` 文件按需读取 CNN 样本。

    ``X_train.npy`` 可能比较大，所以这里用 ``mmap_mode='r'`` 映射读取，
    不一次性把所有样本加载到内存。每次 ``__getitem__`` 只取一个样本。
    """

    def __init__(self, x_path: str | Path, y_path: str | Path) -> None:
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)

        if not self.x_path.exists():
            raise FileNotFoundError(f"找不到输入数组: {self.x_path}")
        if not self.y_path.exists():
            raise FileNotFoundError(f"找不到标签数组: {self.y_path}")

        self.x = np.load(self.x_path, mmap_mode="r")
        self.y = np.load(self.y_path)

        if self.x.ndim != 4:
            raise ValueError(f"{self.x_path.name} 应为 4 维数组 (N, 48, 40, 40)，实际为 {self.x.shape}")
        if self.x.shape[1:] != (48, 40, 40):
            raise ValueError(f"{self.x_path.name} shape 异常，应为 (N, 48, 40, 40)，实际为 {self.x.shape}")
        if self.y.ndim != 1:
            raise ValueError(f"{self.y_path.name} 应为 1 维数组 (N,)，实际为 {self.y.shape}")
        if len(self.x) != len(self.y):
            raise ValueError(f"X/y 样本数不一致: {len(self.x)} vs {len(self.y)}")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.asarray(self.x[index], dtype=np.float32))
        y = torch.tensor(float(self.y[index]), dtype=torch.float32)
        return x, y


def set_random_seed(seed: int) -> None:
    """固定随机种子，让训练结果尽量可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_preprocessed_dir(user_dir: str | None = None) -> Path:
    """寻找预处理输出目录。

    优先使用命令行传入路径；如果没传，则先找 notebook 默认输出
    ``data/processed_xiamen_1985``，再找模块化脚本默认输出 ``outputs/xiamen``。
    """

    if user_dir:
        path = Path(user_dir)
        if not path.exists():
            raise FileNotFoundError(f"指定的数据目录不存在: {path}")
        return path

    candidates = [
        Path("data/processed_xiamen_1985"),
        Path("outputs/xiamen"),
    ]
    for path in candidates:
        if (path / "X_train.npy").exists():
            return path

    raise FileNotFoundError(
        "没有找到预处理结果。请先运行 Data_preprocessing.ipynb，"
        "或执行 python src/preprocess_xiamen.py --start-year 1985 --end-year 1985。"
    )


def load_dates(data_dir: Path, split: str) -> np.ndarray | None:
    """读取训练或验证日期。日期文件不存在时返回 None。"""

    path = data_dir / f"dates_{split}.npy"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def load_target_scaler(data_dir: Path) -> TargetScaler:
    """读取或反推出 y 的标准化参数。

    两种情况都兼容：

    1. 模块化预处理脚本会保存 ``y_scaler.json``；
    2. notebook 只保存 ``y_original.npy``，此时用 ``y_train/y_val`` 与原始 y
       做线性拟合，反推 ``original = scaled * std + mean``。
    """

    scaler_path = data_dir / "y_scaler.json"
    if scaler_path.exists():
        payload = json.loads(scaler_path.read_text(encoding="utf-8"))
        return TargetScaler(mean=float(payload["mean"]), std=float(payload["std"]))

    y_original_path = data_dir / "y_original.npy"
    if not y_original_path.exists():
        raise FileNotFoundError("缺少 y_scaler.json 或 y_original.npy，无法把预测值还原到原始单位。")

    y_original = np.load(y_original_path).astype("float64")
    y_scaled = np.concatenate([
        np.load(data_dir / "y_train.npy"),
        np.load(data_dir / "y_val.npy"),
    ]).astype("float64")

    if len(y_original) != len(y_scaled):
        raise ValueError("y_original 与 y_train/y_val 拼接后的长度不一致，无法反推标准化参数。")

    design = np.column_stack([y_scaled, np.ones_like(y_scaled)])
    std, mean = np.linalg.lstsq(design, y_original, rcond=None)[0]

    if abs(std) < 1e-12:
        raise ValueError("反推出的 y 标准差接近 0，请检查预处理输出。")

    return TargetScaler(mean=float(mean), std=float(std))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """计算回归验证指标。"""

    y_true = np.asarray(y_true, dtype="float64").reshape(-1)
    y_pred = np.asarray(y_pred, dtype="float64").reshape(-1)
    error = y_pred - y_true

    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))

    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(error ** 2) / denom) if denom > 0 else float("nan")
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "corr": corr,
    }


def save_json(path: str | Path, payload: dict) -> None:
    """以 UTF-8 JSON 保存字典。"""

    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_predictions_csv(
    path: str | Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: np.ndarray | None,
) -> None:
    """保存验证集逐日预测结果。"""

    frame = pd.DataFrame({
        "observed": y_true.reshape(-1),
        "predicted": y_pred.reshape(-1),
        "error": y_pred.reshape(-1) - y_true.reshape(-1),
    })
    if dates is not None:
        frame.insert(0, "date", pd.to_datetime(dates).astype(str))
    frame.to_csv(path, index=False, encoding="utf-8-sig")
