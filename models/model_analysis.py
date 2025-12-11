"""
Post-training analysis script.

For each config file listed in CONFIGS:
- Read metrics from: ./runs/<config_name>/metrics.csv
- Create and save (in ./outputs/<config_name>/):
    - loss curves (train / val) vs epochs
    - accuracy curves (train / val) vs epochs
    - epoch time vs epochs
    - cumulative time vs epochs
- Save a small textual summary (summary.txt) with useful model info
  extracted from the config (dataset, layers, performer variant, m, etc.).
"""
import os
import csv
import yaml
import matplotlib.pyplot as plt

# List of config files to analyze
CONFIGS = [
    "configs/mnist_baseline.yaml",
    "configs/cifar10_baseline.yaml",
]


def load_cfg(path: str):
    """Load a YAML configuration file into a Python dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_metrics(metrics_path: str):
    """
    Load metrics.csv into simple Python lists.

    Expected header:
        epoch,epoch_time_sec,train_loss,train_acc,val_loss,val_acc
    """
    epochs = []
    epoch_time = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    with open(metrics_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            epoch_time.append(float(row["epoch_time_sec"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_acc"]))

    return {
        "epochs": epochs,
        "epoch_time": epoch_time,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }


def compute_cumulative_time(epoch_time):
    """Given a list of epoch_time, return cumulative time list."""
    cum = []
    running = 0.0
    for t in epoch_time:
        running += t
        cum.append(running)
    return cum


def build_model_info_string(cfg):
    """
    Build a compact string describing the model, for titles / summaries:

    Includes:
    - dataset
    - d_model, heads, depth, mlp_ratio, patch
    - performer m, variant (if present)
    - layers pattern
    """
    dataset_name = cfg["dataset"]["name"]
    patch = cfg["dataset"]["patch"]

    model_cfg = cfg["model"]
    layers = model_cfg.get("layers", [])

    performer_cfg = model_cfg.get("performer", {})
    m = performer_cfg.get("m", None)
    variant = performer_cfg.get("variant", None)

    parts = []
    parts.append(f"dataset={dataset_name}")

    # Performer-specific info (only meaningful if you actually use Perf layers)
    if m is not None:
        parts.append(f"m={m}")
    if variant is not None:
        parts.append(f"variant={variant}")

    if layers:
        # layers is a list like ["Reg", "Perf", ...]
        parts.append(f"layers={layers}")

    return " | ".join(parts)


def plot_loss_curves(out_dir, cfg_name, metrics, model_info):
    """
    Plot train and val loss vs epochs and save to out_dir.
    """
    epochs = metrics["epochs"]
    train_loss = metrics["train_loss"]
    val_loss = metrics["val_loss"]

    plt.figure()
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_loss, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss curves ({cfg_name})\n{model_info}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(out_dir, "loss_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_accuracy_curves(out_dir, cfg_name, metrics, model_info):
    """
    Plot train and val accuracy vs epochs and save to out_dir.
    """
    epochs = metrics["epochs"]
    train_acc = metrics["train_acc"]
    val_acc = metrics["val_acc"]

    plt.figure()
    plt.plot(epochs, train_acc, label="train acc")
    plt.plot(epochs, val_acc, label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy curves ({cfg_name})\n{model_info}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(out_dir, "accuracy_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_epoch_time(out_dir, cfg_name, metrics, model_info):
    """
    Plot epoch time vs epochs and cumulative time vs epochs.
    Saves two separate figures in out_dir.
    """
    epochs = metrics["epochs"]
    epoch_time = metrics["epoch_time"]
    cum_time = compute_cumulative_time(epoch_time)

    # Epoch time per epoch
    plt.figure()
    plt.bar(epochs, epoch_time)
    plt.xlabel("Epoch")
    plt.ylabel("Time per epoch (s)")
    plt.title(f"Epoch time per epoch ({cfg_name})\n{model_info}")
    plt.grid(True, axis="y", alpha=0.3)

    out_path = os.path.join(out_dir, "epoch_time.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Cumulative time
    plt.figure()
    plt.plot(epochs, cum_time, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative time (s)")
    plt.title(f"Cumulative training time ({cfg_name})\n{model_info}")
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(out_dir, "cumulative_time.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_run_summary(out_dir, cfg_name, cfg, metrics, model_info):
    """
    Write a small textual summary for this run into out_dir:
    - config name
    - model info
    - number of epochs
    - best val accuracy and epoch
    - total training time
    """
    epochs = metrics["epochs"]
    val_acc = metrics["val_acc"]
    epoch_time = metrics["epoch_time"]

    # Best val accuracy
    best_idx = max(range(len(val_acc)), key=lambda i: val_acc[i])
    best_epoch = epochs[best_idx]
    best_val_acc = val_acc[best_idx]

    cum_time = compute_cumulative_time(epoch_time)
    total_time = cum_time[-1] if cum_time else 0.0

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Config name: {cfg_name}\n")
        f.write(f"Model info: {model_info}\n\n")
        f.write(f"Num epochs: {len(epochs)}\n")
        f.write(f"Best val accuracy: {best_val_acc:.4f}% (epoch {best_epoch})\n")
        f.write(f"Total training time: {total_time:.2f} s\n")


def analyze_config(cfg_path: str):
    """
    For a given config file:
    - read metrics from runs/<config_name>/metrics.csv
    - create plots + summary in outputs/<config_name>/
    """
    if not os.path.exists(cfg_path):
        print(f"[WARN] Config file does not exist: {cfg_path}")
        return

    cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]

    # Where training wrote metrics.csv
    runs_dir = os.path.join("runs", cfg_name)
    metrics_path = os.path.join(runs_dir, "metrics.csv")

    # Where we will save analysis outputs
    out_dir = os.path.join("outputs", cfg_name)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(runs_dir):
        print(f"[WARN] Run directory not found for {cfg_name}: {runs_dir}")
        return

    if not os.path.exists(metrics_path):
        print(f"[WARN] metrics.csv not found for {cfg_name} in {runs_dir}")
        return

    print(f"\nAnalyzing config: {cfg_path}")
    print(f"Metrics directory : {runs_dir}")
    print(f"Metrics file      : {metrics_path}")
    print(f"Output directory  : {out_dir}")

    cfg = load_cfg(cfg_path)
    metrics = load_metrics(metrics_path)
    model_info = build_model_info_string(cfg)

    # Create plots in outputs/<cfg_name>/
    plot_loss_curves(out_dir, cfg_name, metrics, model_info)
    plot_accuracy_curves(out_dir, cfg_name, metrics, model_info)
    plot_epoch_time(out_dir, cfg_name, metrics, model_info)

    # Write summary in outputs/<cfg_name>/
    write_run_summary(out_dir, cfg_name, cfg, metrics, model_info)

    print("  -> Saved loss_curves.png, accuracy_curves.png, "
          "epoch_time.png, cumulative_time.png, summary.txt")


def main():
    """Loop over all config files listed in CONFIGS and analyze each one."""
    for cfg_path in CONFIGS:
        analyze_config(cfg_path)


if __name__ == "__main__":
    main()
