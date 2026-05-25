"""ERA20C GRIB 读取、裁剪、插值和标准化。"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from config import (
    ERA20C_VARIABLE_CANDIDATES,
    ERA20C_VARIABLE_DIRS,
    GRID_SIZE,
    REGION_HALF_SIZE_DEG,
    SITE_LAT,
    SITE_LON,
    VARIABLE_ORDER,
)


def _find_coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    """在 Dataset 中从候选名里找到实际坐标名。"""

    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"无法识别坐标名，候选={candidates}，实际={list(ds.coords)}")


def _find_variable_name(ds: xr.Dataset, logical_name: str) -> str:
    """自动识别 u10/v10/slp 在 GRIB 文件中的真实变量名。"""

    for name in ERA20C_VARIABLE_CANDIDATES[logical_name]:
        if name in ds.data_vars:
            return name
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    raise KeyError(f"无法识别 {logical_name} 变量名，文件变量={list(ds.data_vars)}")


def find_year_file(variable: str, year: int) -> Path | None:
    """在对应变量目录下寻找某一年的 GRIB 文件。"""

    folder = ERA20C_VARIABLE_DIRS[variable]
    if not folder.exists():
        return None

    for pattern in (f"*{year}*.grb", f"*{year}*.grib", f"*{year}*"):
        matches = sorted(path for path in folder.glob(pattern) if path.is_file())
        if matches:
            return matches[0]
    return None


def open_era20c_grib(path: Path, variable: str) -> xr.DataArray:
    """读取一个 GRIB 文件并返回目标变量 DataArray。"""

    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    var_name = _find_variable_name(ds, variable)
    lat_name = _find_coord_name(ds, ("latitude", "lat"))
    lon_name = _find_coord_name(ds, ("longitude", "lon"))
    time_name = _find_coord_name(ds, ("time", "valid_time"))

    da = ds[var_name].rename({lat_name: "lat", lon_name: "lon", time_name: "time"})
    return da


def subset_and_interp_station_region(
    da: xr.DataArray,
    site_lat: float = SITE_LAT,
    site_lon: float = SITE_LON,
    half_size: float = REGION_HALF_SIZE_DEG,
    grid_size: int = GRID_SIZE,
) -> xr.DataArray:
    """提取站点周围 10° x 10° 区域，并插值到 40 x 40 网格。"""

    lon = da["lon"]
    site_lon_for_data = site_lon
    if float(lon.max()) > 180 and site_lon < 0:
        site_lon_for_data = site_lon % 360

    lat_min = site_lat - half_size
    lat_max = site_lat + half_size
    lon_min = site_lon_for_data - half_size
    lon_max = site_lon_for_data + half_size

    lat_values = da["lat"].to_numpy()
    lat_slice = slice(lat_max, lat_min) if lat_values[0] > lat_values[-1] else slice(lat_min, lat_max)
    region = da.sel(lat=lat_slice, lon=slice(lon_min, lon_max))

    if region.sizes.get("lat", 0) < 2 or region.sizes.get("lon", 0) < 2:
        raise ValueError("站点周围 ERA 区域为空或格点过少，请检查经纬度和 ERA 坐标")

    target_lat = np.linspace(lat_min, lat_max, grid_size)
    target_lon = np.linspace(lon_min, lon_max, grid_size)
    return region.interp(lat=target_lat, lon=target_lon).transpose("time", "lat", "lon")


@dataclass
class EraStats:
    """一个 ERA 变量的均值和标准差。"""

    mean: float
    std: float


class Era20cReader:
    """按年份读取 ERA20C，并为 CNN 构建标准化输入样本。"""

    def __init__(self, max_cache_items: int = 6) -> None:
        self.stats: dict[str, EraStats] = {}
        self.max_cache_items = max_cache_items
        self._cache: OrderedDict[tuple[str, int], xr.DataArray] = OrderedDict()

    def _put_cache(self, key: tuple[str, int], value: xr.DataArray) -> None:
        """把某变量某年份数据放入简单 LRU 缓存，避免反复读 GRIB。"""

        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_cache_items:
            self._cache.popitem(last=False)

    def available_years(self, years: list[int], skip_missing: bool = False) -> list[int]:
        """检查三种变量文件是否齐全，返回可用年份。"""

        available: list[int] = []
        for year in years:
            missing = [var for var in VARIABLE_ORDER if find_year_file(var, year) is None]
            if missing:
                message = f"{year} 年缺少 ERA20C 文件: {', '.join(missing)}"
                if skip_missing:
                    print("[ERA]", message, "跳过该年")
                    continue
                raise FileNotFoundError(message)
            available.append(year)

        if not available:
            raise FileNotFoundError("没有找到任何可用 ERA20C 年份")

        print(f"[ERA] 可用年份: {available[0]}-{available[-1]}，共 {len(available)} 年")
        return available

    def _load_raw_year(self, variable: str, year: int) -> xr.DataArray:
        """读取并插值某变量某一年，返回未标准化 DataArray。"""

        key = (variable, year)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        path = find_year_file(variable, year)
        if path is None:
            raise FileNotFoundError(f"{year} 年 {variable} 文件不存在")

        print(f"[ERA] 读取 {year} {variable}: {path}")
        da = open_era20c_grib(path, variable)
        da = subset_and_interp_station_region(da).astype("float32")
        self._put_cache(key, da)
        return da

    def compute_standardization(self, years: list[int]) -> None:
        """对 u10/v10/slp 分别计算研究期均值和标准差。"""

        for variable in VARIABLE_ORDER:
            total = 0.0
            total_sq = 0.0
            count = 0
            for year in years:
                arr = self._load_raw_year(variable, year).to_numpy().astype("float64")
                values = arr[np.isfinite(arr)]
                total += float(values.sum())
                total_sq += float(np.square(values).sum())
                count += int(values.size)
            if count == 0:
                raise ValueError(f"{variable} 没有可用于标准化的有效值")
            mean = total / count
            variance = max(total_sq / count - mean * mean, 1e-12)
            self.stats[variable] = EraStats(mean=float(mean), std=float(np.sqrt(variance)))
            print(f"[ERA] {variable} mean={mean:.6f}, std={self.stats[variable].std:.6f}")

    def get_normalized_year(self, variable: str, year: int) -> xr.DataArray:
        """返回标准化后的某变量某一年 DataArray。"""

        if variable not in self.stats:
            raise RuntimeError("请先调用 compute_standardization()")
        raw = self._load_raw_year(variable, year)
        stat = self.stats[variable]
        return ((raw - stat.mean) / stat.std).astype("float32")

    def build_predictor_for_day(self, date: pd.Timestamp) -> np.ndarray | None:
        """为某一天 D 构建 CNN 输入，shape=(48, 40, 40)。

        如果 D-1 到 D 的 16 个 3 小时时间片不完整，返回 None。
        """

        date = pd.Timestamp(date).normalize()
        start = date - pd.Timedelta(days=1)
        expected_times = pd.date_range(start, periods=16, freq="3h")

        channels: list[np.ndarray] = []
        for variable in VARIABLE_ORDER:
            pieces = [self.get_normalized_year(variable, year) for year in sorted({start.year, date.year})]
            da = xr.concat(pieces, dim="time").sortby("time")
            try:
                selected = da.sel(time=expected_times)
            except KeyError:
                return None
            if selected.sizes.get("time", 0) != 16:
                return None
            channels.append(selected.to_numpy().astype("float32"))

        sample = np.concatenate(channels, axis=0)
        if sample.shape != (48, GRID_SIZE, GRID_SIZE):
            raise ValueError(f"样本 shape 异常: {sample.shape}")
        return sample
