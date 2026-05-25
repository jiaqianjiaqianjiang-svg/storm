"""UTide 潮汐分离与逐日最大风暴潮标签生成。"""

from __future__ import annotations

import matplotlib.dates as mdates
import pandas as pd
from utide import reconstruct, solve

from config import OBS_MAD_THRESHOLD, SURGE_MAD_THRESHOLD
from gesla_loader import clean_observed_sea_level, mark_outliers_mad


def separate_tide_with_utide(data: pd.DataFrame, lat: float) -> pd.DataFrame:
    """用 UTide 从观测水位中分离天文潮和 storm surge。

    参数
    ----
    data:
        以 ``datetime`` 为索引，包含 ``sea_level`` 的 DataFrame。
    lat:
        潮位站纬度，UTide 解调和分析时需要。

    返回
    ----
    pd.DataFrame
        增加 ``predicted_tide`` 和 ``storm_surge`` 两列。
    """

    cleaned = clean_observed_sea_level(data, threshold=OBS_MAD_THRESHOLD)
    time_num = mdates.date2num(cleaned.index.to_pydatetime())
    sea_level = cleaned["sea_level"].to_numpy()

    coef = solve(time_num, sea_level, lat=lat, method="ols", conf_int="linear")
    tide = reconstruct(time_num, coef).h

    result = cleaned.copy()
    result["predicted_tide"] = tide
    result["storm_surge"] = result["sea_level"] - result["predicted_tide"]

    surge_outliers = mark_outliers_mad(result, "storm_surge", SURGE_MAD_THRESHOLD)
    result.loc[surge_outliers, "storm_surge"] = pd.NA
    result["storm_surge"] = result["storm_surge"].interpolate(
        method="time",
        limit=6,
        limit_direction="both",
    )
    result = result.dropna(subset=["storm_surge"])

    if result.empty:
        raise ValueError("潮汐分离后没有可用 storm surge 数据")

    return result


def daily_maximum_surge(surge: pd.DataFrame) -> pd.DataFrame:
    """按日提取 daily maximum storm surge，作为 CNN 标签 y。"""

    if "storm_surge" not in surge.columns:
        raise KeyError("输入数据缺少 storm_surge 列")

    daily = surge["storm_surge"].resample("D").max().dropna().to_frame("daily_max_surge")
    if daily.empty:
        raise ValueError("逐日最大 storm surge 为空")
    return daily
