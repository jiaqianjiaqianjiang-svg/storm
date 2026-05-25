"""训练并验证厦门站 1985 风暴潮 CNN。

运行前请先完成预处理，确保数据目录中存在：

- X_train.npy / y_train.npy
- X_val.npy / y_val.npy
- dates_val.npy
- y_original.npy 或 y_scaler.json

示例：

python src/train_xiamen.py --data-dir data/processed_xiamen_1985 --epochs 100
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import StormSurgeCNN, count_trainable_parameters
from training_utils import (
    SurgeNpyDataset,
    load_dates,
    load_target_scaler,
    regression_metrics,
    resolve_preprocessed_dir,
    save_json,
    save_predictions_csv,
    set_random_seed,
)


def configure_console_encoding() -> None:
    """尽量让 Windows 终端用 UTF-8 输出中文日志。"""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except OSError:
                pass


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="训练厦门站 CNN 风暴潮重建模型。")
    parser.add_argument("--data-dir", type=str, default=None, help="预处理输出目录，默认自动寻找。")
    parser.add_argument("--run-dir", type=str, default="runs/xiamen_1985", help="训练结果保存目录。")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数。1985 单年样本少，默认 100。")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size。")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率。")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW 权重衰减。")
    parser.add_argument("--dropout", type=float, default=0.30, help="全连接层 dropout。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数。Windows 建议先用 0。")
    parser.add_argument("--patience", type=int, default=25, help="验证集 loss 连续多少轮不提升则早停。")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"), help="训练设备。")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    """根据命令行和 CUDA 可用性选择训练设备。"""

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前 PyTorch 检测不到 CUDA。")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    """运行一个训练或验证 epoch。

    当 ``optimizer`` 为 None 时只做验证，不更新参数。
    """

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = len(y_batch)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict_scaled(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """在验证集上预测，返回标准化单位下的真实值和预测值。"""

    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        pred = model(x_batch).cpu().numpy()
        all_pred.append(pred)
        all_true.append(y_batch.numpy())

    return np.concatenate(all_true), np.concatenate(all_pred)


def plot_training_curves(run_dir: Path, history: list[dict[str, float]]) -> None:
    """保存训练/验证 loss 曲线。"""

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (standardized)")
    plt.title("Xiamen CNN training curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=200)
    plt.close()


def plot_validation_figures(
    run_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: np.ndarray | None,
) -> None:
    """保存验证集时间序列图和散点图。"""

    x_axis = np.arange(len(y_true)) if dates is None else np.asarray(dates, dtype="datetime64[D]")

    plt.figure(figsize=(10, 4))
    plt.plot(x_axis, y_true, label="Observed", linewidth=1.2)
    plt.plot(x_axis, y_pred, label="Predicted", linewidth=1.2)
    plt.xlabel("Date" if dates is not None else "Validation sample")
    plt.ylabel("Daily maximum storm surge")
    plt.title("Xiamen validation reconstruction")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "val_timeseries.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=18, alpha=0.8)
    low = min(float(np.min(y_true)), float(np.min(y_pred)))
    high = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([low, high], [low, high], color="black", linewidth=1, linestyle="--")
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title("Observed vs predicted")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "val_scatter.png", dpi=200)
    plt.close()


def save_history(run_dir: Path, history: list[dict[str, float]]) -> None:
    """保存每个 epoch 的训练日志。"""

    path = run_dir / "training_history.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    """训练入口。"""

    configure_console_encoding()
    args = parse_args()
    set_random_seed(args.seed)

    data_dir = resolve_preprocessed_dir(args.data_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[RUN] 厦门站 CNN 风暴潮重建训练")
    print(f"[RUN] 数据目录: {data_dir.resolve()}")
    print(f"[RUN] 输出目录: {run_dir.resolve()}")
    print("=" * 80)

    train_dataset = SurgeNpyDataset(data_dir / "X_train.npy", data_dir / "y_train.npy")
    val_dataset = SurgeNpyDataset(data_dir / "X_val.npy", data_dir / "y_val.npy")
    dates_val = load_dates(data_dir, "val")
    scaler = load_target_scaler(data_dir)

    print(f"[DATA] X_train: {train_dataset.x.shape}, y_train: {train_dataset.y.shape}")
    print(f"[DATA] X_val:   {val_dataset.x.shape}, y_val:   {val_dataset.y.shape}")
    print(f"[DATA] y 反标准化参数: mean={scaler.mean:.6f}, std={scaler.std:.6f}")

    device = choose_device(args.device)
    print(f"[RUN] device: {device}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = StormSurgeCNN(in_channels=48, dropout=args.dropout).to(device)
    print(f"[MODEL] 可训练参数量: {count_trainable_parameters(model):,}")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(args.patience // 3, 3),
    )

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    best_model_path = run_dir / "best_model.pth"

    progress = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
    for epoch in progress:
        train_loss = run_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = run_one_epoch(model, val_loader, loss_fn, None, device)
        scheduler.step(val_loss)

        lr = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})

        progress.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", lr=f"{lr:.1e}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "args": vars(args),
                    "target_scaler": {"mean": scaler.mean, "std": scaler.std},
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"[EARLY STOP] 验证集 loss 连续 {args.patience} 轮未提升，提前停止。")
            break

    save_history(run_dir, history)
    plot_training_curves(run_dir, history)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "target_scaler": {"mean": scaler.mean, "std": scaler.std},
        },
        run_dir / "last_loaded_best_model.pth",
    )

    y_true_scaled, y_pred_scaled = predict_scaled(model, val_loader, device)
    metrics_scaled = regression_metrics(y_true_scaled, y_pred_scaled)

    y_true = scaler.inverse_transform(y_true_scaled)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    metrics_original = regression_metrics(y_true, y_pred)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "scaled": metrics_scaled,
        "original_units": metrics_original,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
    }
    save_json(run_dir / "metrics.json", metrics)
    save_predictions_csv(run_dir / "val_predictions.csv", y_true, y_pred, dates_val)
    plot_validation_figures(run_dir, y_true, y_pred, dates_val)

    print("=" * 80)
    print("[DONE] 训练完成")
    print(f"[DONE] best_epoch: {best_epoch}")
    print(f"[DONE] val RMSE: {metrics_original['rmse']:.6f}")
    print(f"[DONE] val MAE:  {metrics_original['mae']:.6f}")
    print(f"[DONE] val R2:   {metrics_original['r2']:.6f}")
    print(f"[DONE] 输出目录: {run_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
