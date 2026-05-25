"""项目集中配置。

路径、站点信息、网格大小、清洗阈值都放在这里，避免散落在 notebook
或脚本中。实验室远程电脑上通常只需要检查这里的数据路径是否正确。
"""

from pathlib import Path


# =========================
# 1. 真实数据路径
# =========================

ERA20C_DIR = Path(r"F:\ERA20C")
GESLA_DIR = Path(r"F:\GESLA\GESLA3")


# =========================
# 2. 厦门站信息
# =========================

SITE_NAME = "Xiamen"
SITE_FILE = Path(r"F:\GESLA\GESLA3\xiamen-376a-chn-uhslc")
SITE_LAT = 24.45
SITE_LON = 118.067

XIAMEN_START_YEAR = 1954
XIAMEN_END_YEAR = 1997


# =========================
# 3. ERA20C 变量目录
# =========================

ERA20C_VARIABLE_DIRS = {
    "u10": ERA20C_DIR / "10U",
    "v10": ERA20C_DIR / "10V",
    "slp": ERA20C_DIR / "SLP",
}

ERA20C_VARIABLE_CANDIDATES = {
    "u10": ("u10", "10u", "u", "var165"),
    "v10": ("v10", "10v", "v", "var166"),
    "slp": ("msl", "slp", "sp", "var151"),
}


# =========================
# 4. CNN 输入参数
# =========================

REGION_HALF_SIZE_DEG = 5.0
GRID_SIZE = 40
HOURS_PER_STEP = 3
STEPS_PER_DAY = 8
INPUT_DAYS = 2
STEPS_PER_SAMPLE = STEPS_PER_DAY * INPUT_DAYS
VARIABLE_ORDER = ("u10", "v10", "slp")
INPUT_CHANNELS = STEPS_PER_SAMPLE * len(VARIABLE_ORDER)


# =========================
# 5. 输出目录
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
XIAMEN_OUTPUT_DIR = OUTPUT_ROOT / "xiamen"


# =========================
# 6. 清洗参数
# =========================

MISSING_VALUE_MARKERS = {-99, -999, -9999, 9999, 99999}
SEA_LEVEL_ABS_LIMIT = 10_000.0
OBS_MAD_THRESHOLD = 15.0
SURGE_MAD_THRESHOLD = 15.0
