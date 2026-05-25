"""厦门站 Xiamen 预处理命令行入口。

示例：

python src/preprocess_xiamen.py --start-year 1985 --end-year 1985
python src/preprocess_xiamen.py --all-years
"""

from __future__ import annotations

import argparse
import sys


def configure_console_encoding() -> None:
    """尽量避免 Windows 终端打印中文时乱码。"""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except OSError:
                pass


configure_console_encoding()

from config import (  # noqa: E402
    SITE_FILE,
    SITE_LAT,
    SITE_NAME,
    XIAMEN_END_YEAR,
    XIAMEN_OUTPUT_DIR,
    XIAMEN_START_YEAR,
)
from dataset_builder import collect_available_samples, save_train_val_arrays  # noqa: E402
from era20c_loader import Era20cReader  # noqa: E402
from gesla_loader import read_gesla_file, restrict_years  # noqa: E402
from tide_processing import daily_maximum_surge, separate_tide_with_utide  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="预处理厦门站 GESLA + ERA20C，生成 CNN 可用数据集。")
    parser.add_argument("--all-years", action="store_true", help="使用厦门站全部可用年份 1954-1997")
    parser.add_argument("--start-year", type=int, help="开始年份，例如 1985")
    parser.add_argument("--end-year", type=int, help="结束年份，例如 1985")
    parser.add_argument("--skip-missing-era", action="store_true", help="ERA 文件缺失时跳过该年")
    return parser.parse_args()


def resolve_years(args: argparse.Namespace) -> tuple[int, int]:
    """根据命令行参数确定年份范围。"""

    if args.all_years:
        return XIAMEN_START_YEAR, XIAMEN_END_YEAR
    if args.start_year is None or args.end_year is None:
        raise SystemExit("请使用 --all-years，或同时提供 --start-year 和 --end-year")
    if args.start_year > args.end_year:
        raise SystemExit("--start-year 不能大于 --end-year")
    return args.start_year, args.end_year


def main() -> None:
    """执行完整预处理流程。"""

    args = parse_args()
    start_year, end_year = resolve_years(args)
    years = list(range(start_year, end_year + 1))

    print("=" * 80)
    print(f"[RUN] 站点: {SITE_NAME}")
    print(f"[RUN] 年份范围: {start_year}-{end_year}")
    print(f"[RUN] GESLA 文件: {SITE_FILE}")
    print(f"[RUN] 输出目录: {XIAMEN_OUTPUT_DIR}")
    print("=" * 80)

    gesla = read_gesla_file(SITE_FILE)
    gesla = restrict_years(gesla, start_year, end_year)

    surge = separate_tide_with_utide(gesla, lat=SITE_LAT)
    XIAMEN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    surge.to_csv(XIAMEN_OUTPUT_DIR / "cleaned_surge.csv", encoding="utf-8-sig")

    daily = daily_maximum_surge(surge)
    daily.to_csv(XIAMEN_OUTPUT_DIR / "daily_max_surge.csv", encoding="utf-8-sig")

    era_reader = Era20cReader()
    available_years = era_reader.available_years(years, skip_missing=args.skip_missing_era)
    era_reader.compute_standardization(available_years)

    sample_dates, y = collect_available_samples(daily, era_reader)
    save_train_val_arrays(sample_dates, y, era_reader, XIAMEN_OUTPUT_DIR)

    print("=" * 80)
    print("[DONE] 厦门站预处理完成")
    print(f"[DONE] 输出目录: {XIAMEN_OUTPUT_DIR.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
