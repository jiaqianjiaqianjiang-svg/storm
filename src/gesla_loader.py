"""GESLA 潮位文件读取与基础清洗。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import MISSING_VALUE_MARKERS, SEA_LEVEL_ABS_LIMIT


def find_data_start_line(file_path: str | Path) -> int:
    """自动寻找 GESLA 文件中真正数据开始的行号。

    GESLA 文件开头有多行以 ``#`` 开头的站点元数据。真正数据从第一行
    非 ``#`` 内容开始。
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"GESLA 文件不存在: {path}")

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f):
            if line.strip() and not line.startswith("#"):
                return line_no

    raise ValueError(f"没有找到 GESLA 数据开始行: {path}")


def read_gesla_file(file_path: str | Path) -> pd.DataFrame:
    """读取并初步清洗 GESLA 潮位数据。

    返回的 DataFrame 以 ``datetime`` 为索引，至少包含：

    - ``sea_level``: 观测水位；
    - ``qc_flag``: 质量控制标记；
    - ``use_flag``: 使用标记，缺失时默认设为 1。
    """

    path = Path(file_path)
    start_line = find_data_start_line(path)

    raw = pd.read_csv(path, skiprows=start_line, sep=r"\s+", header=None)
    if raw.shape[1] >= 5:
        raw = raw.iloc[:, :5]
        raw.columns = ["date", "time", "sea_level", "qc_flag", "use_flag"]
    elif raw.shape[1] == 4:
        raw.columns = ["date", "time", "sea_level", "qc_flag"]
        raw["use_flag"] = 1
    else:
        raise ValueError(f"GESLA 列数异常，应为 4 或 5 列，实际为 {raw.shape[1]} 列")

    raw["datetime"] = pd.to_datetime(
        raw["date"].astype(str) + " " + raw["time"].astype(str),
        errors="coerce",
    )
    raw["sea_level"] = pd.to_numeric(raw["sea_level"], errors="coerce")
    raw["use_flag"] = pd.to_numeric(raw["use_flag"], errors="coerce").fillna(1)

    data = raw[["datetime", "sea_level", "qc_flag", "use_flag"]].copy()
    data = data.dropna(subset=["datetime", "sea_level"])
    data = data.sort_values("datetime").drop_duplicates(subset=["datetime"])

    data.loc[data["sea_level"].isin(MISSING_VALUE_MARKERS), "sea_level"] = np.nan
    data.loc[data["sea_level"].abs() > SEA_LEVEL_ABS_LIMIT, "sea_level"] = np.nan
    data = data[data["use_flag"] != 0]
    data = data.dropna(subset=["sea_level"])

    if data.empty:
        raise ValueError(f"GESLA 文件清洗后为空: {path}")

    return data.set_index("datetime").sort_index()


def restrict_years(data: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """按年份范围裁剪 GESLA 数据。"""

    if start_year > end_year:
        raise ValueError("start_year 不能大于 end_year")

    mask = (data.index.year >= start_year) & (data.index.year <= end_year)
    subset = data.loc[mask].copy()

    if subset.empty:
        raise ValueError(f"年份范围 {start_year}-{end_year} 内没有 GESLA 数据")

    return subset


def mark_outliers_mad(data: pd.DataFrame, column: str, threshold: float) -> pd.Series:
    """使用宽松 MAD 阈值识别明显异常点。

    MAD 比均值标准差更不容易被极端风暴潮影响。阈值设置较大，只用于去除
    明显坏点，不用于删除真实极端事件。
    """

    values = data[column].astype(float)
    median = values.median()
    mad = (values - median).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(False, index=data.index)

    robust_z = 0.6745 * (values - median) / mad
    return robust_z.abs() > threshold


def clean_observed_sea_level(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """对观测潮位做异常值标记和短缺口插值。"""

    cleaned = data.copy()
    outliers = mark_outliers_mad(cleaned, "sea_level", threshold)
    cleaned.loc[outliers, "sea_level"] = np.nan
    cleaned["sea_level"] = cleaned["sea_level"].interpolate(
        method="time",
        limit=6,
        limit_direction="both",
    )
    cleaned = cleaned.dropna(subset=["sea_level"])

    if cleaned.empty:
        raise ValueError("潮位异常值处理后数据为空")

    return cleaned
