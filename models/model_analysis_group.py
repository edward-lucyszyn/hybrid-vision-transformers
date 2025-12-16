"""
Group post-training analysis script (multi-model overlay).

How to run (example):
    python3 models/model_analysis_group.py

This file also supports CLI:
    python3 models/model_analysis_group.py --name "..." --configs configs/a.yaml configs/b.yaml
"""

import os
import csv
import yaml
import argparse
import matplotlib.pyplot as plt
import statistics

# ----------------------------
# IO helpers
# ----------------------------
def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_metrics(metrics_path: str):
    """
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
    cum = []
    running = 0.0
    for t in epoch_time:
        running += t
        cum.append(running)
    return cum


# ----------------------------
# Params extraction (for title/legend)
# ----------------------------
def deep_get(d, path, default=None):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


DEFAULT_PARAM_PATHS = {
    "dataset": "dataset.name",
    "layers": "model.layers",
    "m": "model.performer.m",
    "variant": "model.performer.variant",
}


def extract_params(cfg, param_paths):
    out = {}
    for alias, path in param_paths.items():
        out[alias] = deep_get(cfg, path, default=None)
    return out


def common_and_varying_params(list_of_param_dicts):
    if not list_of_param_dicts:
        return {}, set()

    keys = list(list_of_param_dicts[0].keys())
    common = {}
    varying = set()

    for k in keys:
        vals = [p.get(k, None) for p in list_of_param_dicts]
        first = vals[0]
        if first is None:
            varying.add(k)
            continue
        if all(v == first for v in vals):
            common[k] = first
        else:
            varying.add(k)

    return common, varying


def fmt_value(v):
    if isinstance(v, list):
        return "[" + ",".join(map(str, v)) + "]"
    return str(v)


def format_params_line(params_dict, keys_order=None):
    if not params_dict:
        return ""
    keys = keys_order if keys_order is not None else list(params_dict.keys())
    parts = []
    for k in keys:
        if k in params_dict and params_dict[k] is not None:
            parts.append(f"{k}={fmt_value(params_dict[k])}")
    return " | ".join(parts)


# ----------------------------
# Plotting
# ----------------------------
def plot_multi_line(out_dir, filename, plot_title_main, params_line, series, y_label):
    """
    Two-line title:
      line 1: main title
      line 2: params line (common params)
    """
    plt.figure()
    for s in series:
        plt.plot(s["x"], s["y"], label=s["label"])

    plt.xlabel("Epoch")
    plt.ylabel(y_label)

    if params_line:
        plt.title(f"{plot_title_main}\n{params_line}")
    else:
        plt.title(plot_title_main)

    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_group(group_name, out_dir, runs):
    all_params = [r["params"] for r in runs]
    common, varying = common_and_varying_params(all_params)

    # line 2 of the title: common params only
    common_str = format_params_line(common, keys_order=list(DEFAULT_PARAM_PATHS.keys()))

    # legend: only varying params
    varying_order = [k for k in DEFAULT_PARAM_PATHS.keys() if k in varying]

    def legend_label(r):
        v = {k: r["params"].get(k, None) for k in varying_order}
        label = format_params_line(v, keys_order=varying_order)
        return label if label else r["cfg_name"]

    train_loss_series = []
    val_loss_series = []
    train_acc_series = []
    val_acc_series = []
    epoch_time_series = []
    cum_time_series = []

    for r in runs:
        m = r["metrics"]
        epochs = m["epochs"]
        lbl = legend_label(r)

        train_loss_series.append({"x": epochs, "y": m["train_loss"], "label": lbl})
        val_loss_series.append({"x": epochs, "y": m["val_loss"], "label": lbl})
        train_acc_series.append({"x": epochs, "y": m["train_acc"], "label": lbl})
        val_acc_series.append({"x": epochs, "y": m["val_acc"], "label": lbl})
        epoch_time_series.append({"x": epochs, "y": m["epoch_time"], "label": lbl})

        cum = compute_cumulative_time(m["epoch_time"])
        cum_time_series.append({"x": epochs, "y": cum, "label": lbl})

    plot_multi_line(
        out_dir=out_dir,
        filename="train_loss.png",
        plot_title_main=f"{group_name} - Train loss",
        params_line=common_str,
        series=train_loss_series,
        y_label="Loss",
    )
    plot_multi_line(
        out_dir=out_dir,
        filename="val_loss.png",
        plot_title_main=f"{group_name} - Val loss",
        params_line=common_str,
        series=val_loss_series,
        y_label="Loss",
    )
    plot_multi_line(
        out_dir=out_dir,
        filename="train_acc.png",
        plot_title_main=f"{group_name} - Train accuracy",
        params_line=common_str,
        series=train_acc_series,
        y_label="Accuracy (%)",
    )
    plot_multi_line(
        out_dir=out_dir,
        filename="val_acc.png",
        plot_title_main=f"{group_name} - Val accuracy",
        params_line=common_str,
        series=val_acc_series,
        y_label="Accuracy (%)",
    )
    plot_multi_line(
        out_dir=out_dir,
        filename="epoch_time.png",
        plot_title_main=f"{group_name} - Time per epoch",
        params_line=common_str,
        series=epoch_time_series,
        y_label="Time per epoch (s)",
    )
    plot_multi_line(
        out_dir=out_dir,
        filename="cumulative_time.png",
        plot_title_main=f"{group_name} - Cumulative time",
        params_line=common_str,
        series=cum_time_series,
        y_label="Cumulative time (s)",
    )

    return common, varying_order


def write_group_summary(out_dir, group_name, runs, common, varying_order):
    summary_path = os.path.join(out_dir, "summary.txt")

    def best_val(m):
        vals = m["val_acc"]
        if not vals:
            return None, None
        best_idx = max(range(len(vals)), key=lambda i: vals[i])
        return m["epochs"][best_idx], vals[best_idx]

    with open(summary_path, "w") as f:
        f.write(f"Group name: {group_name}\n")
        common_line = format_params_line(common, keys_order=list(DEFAULT_PARAM_PATHS.keys()))
        if common_line:
            f.write(f"Common params: {common_line}\n")
        if varying_order:
            f.write("Varying params (legend): " + ", ".join(varying_order) + "\n")
        f.write("\n")

        for r in runs:
            m = r["metrics"]
            total_time = compute_cumulative_time(m["epoch_time"])[-1] if m["epoch_time"] else 0.0
            be, bv = best_val(m)
            var_part = format_params_line(
                {k: r["params"].get(k, None) for k in varying_order},
                keys_order=varying_order
            )
            f.write(f"- {r['cfg_name']}\n")
            if var_part:
                f.write(f"  label: {var_part}\n")
            f.write(f"  epochs: {len(m['epochs'])}\n")
            if be is not None:
                f.write(f"  best val acc: {bv:.4f}% at epoch {be}\n")
            f.write(f"  total time: {total_time:.2f} s\n")
            f.write("\n")


# ----------------------------
# Main
# ----------------------------
def analyze_group(group_name, cfg_paths):
    runs = []
    for cfg_path in cfg_paths:
        if not os.path.exists(cfg_path):
            print(f"[WARN] Config file does not exist: {cfg_path}")
            continue

        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
        runs_dir = os.path.join("runs", cfg_name)
        metrics_path = os.path.join(runs_dir, "metrics.csv")

        if not os.path.exists(metrics_path):
            print(f"[WARN] metrics.csv not found for {cfg_name}: {metrics_path}")
            continue

        cfg = load_cfg(cfg_path)
        metrics = load_metrics(metrics_path)
        params = extract_params(cfg, DEFAULT_PARAM_PATHS)

        runs.append({
            "cfg_name": cfg_name,
            "cfg_path": cfg_path,
            "cfg": cfg,
            "metrics": metrics,
            "params": params,
        })

    if not runs:
        print("[ERROR] No valid runs found (check paths + runs/<cfg_name>/metrics.csv).")
        return

    out_dir = os.path.join("outputs", group_name)
    os.makedirs(out_dir, exist_ok=True)

    common, varying_order = plot_group(group_name, out_dir, runs)
    write_group_summary(out_dir, group_name, runs, common, varying_order)

    print(f"\nSaved group plots in: {out_dir}")
    print("Files:")
    print("  train_loss.png")
    print("  val_loss.png")
    print("  train_acc.png")
    print("  val_acc.png")
    print("  epoch_time.png")
    print("  cumulative_time.png")
    print("  summary.txt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, default=None, help="Group output folder name (under outputs/).")
    p.add_argument("--configs", nargs="+", default=None, help="List of YAML config paths.")
    return p.parse_args()

# ----------------------------
# Full analysis (per-dataset txt tables)
# ----------------------------
def infer_dataset(cfg_name: str, cfg: dict):
    # treat cifar10_high as its own dataset
    if "cifar10_high" in cfg_name:
        return "cifar10_high"
    ds = deep_get(cfg, "dataset.name", default=None)
    return ds if ds is not None else "unknown"


def infer_setting(cfg_name: str):
    n = cfg_name.lower()
    if "baseline" in n:
        return "Full Reg"
    if "interwined_perf" in n:
        return "Intertwined (start=Perf)"
    if "interwined_reg" in n:
        return "Intertwined (start=Reg)"
    if "perf_reg" in n:
        return "Perf->Reg"
    if "reg_perf" in n:
        return "Reg->Perf"
    # names like cifar10_perf_d=...
    if "_perf_d=" in n or n.startswith("perf_d=") or "mnist_perf_d=" in n or "cifar10_perf_d=" in n:
        return "Full Perf"
    return "Unknown"


def safe_median(xs):
    return statistics.median(xs) if xs else 0.0


def best_epoch_and_value(epochs, values, mode="max"):
    if not values:
        return None, None
    if mode == "max":
        idx = max(range(len(values)), key=lambda i: values[i])
    else:
        idx = min(range(len(values)), key=lambda i: values[i])
    return epochs[idx], values[idx]


def full_analysis(groups_dict, out_root="outputs/full_analysis"):
    """
    Generate one TSV-like .txt file per dataset containing a single-row summary per config:
      - best_val_acc, best_train_acc, best_train_loss, best_val_loss
      - total_epochs, epoch_best_val_acc
      - m, d (layers), setting, median_epoch_time, kernel
    """
    # collect unique config paths across all groups
    all_cfg_paths = []
    seen = set()
    for _, cfgs in groups_dict.items():
        for p in cfgs:
            if p not in seen:
                seen.add(p)
                all_cfg_paths.append(p)

    rows_by_dataset = {}

    for cfg_path in all_cfg_paths:
        if not os.path.exists(cfg_path):
            print(f"[WARN] Config file does not exist: {cfg_path}")
            continue

        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
        runs_dir = os.path.join("runs", cfg_name)
        metrics_path = os.path.join(runs_dir, "metrics.csv")

        if not os.path.exists(metrics_path):
            print(f"[WARN] metrics.csv not found for {cfg_name}: {metrics_path}")
            continue

        cfg = load_cfg(cfg_path)
        m = load_metrics(metrics_path)

        dataset = infer_dataset(cfg_name, cfg)
        setting = infer_setting(cfg_name)

        layers = deep_get(cfg, "model.layers", default=None)
        m_features = deep_get(cfg, "model.performer.m", default=None)
        variant = deep_get(cfg, "model.performer.variant", default=None)

        # baseline has no performer.variant; kernel is still softmax in regular attention
        kernel = variant if variant is not None else "softmax"

        total_epochs = len(m["epochs"])
        med_epoch_time = safe_median(m["epoch_time"])

        best_val_epoch, best_val_acc = best_epoch_and_value(m["epochs"], m["val_acc"], mode="max")
        _, best_train_acc = best_epoch_and_value(m["epochs"], m["train_acc"], mode="max")
        _, best_train_loss = best_epoch_and_value(m["epochs"], m["train_loss"], mode="min")
        _, best_val_loss = best_epoch_and_value(m["epochs"], m["val_loss"], mode="min")

        row = {
            "cfg_name": cfg_name,
            "dataset": dataset,
            "setting": setting,
            "d": layers,
            "m": m_features,
            "kernel": kernel,
            "best_val_acc": best_val_acc,
            "best_val_epoch": best_val_epoch,
            "best_train_acc": best_train_acc,
            "best_train_loss": best_train_loss,
            "best_val_loss": best_val_loss,
            "total_epochs": total_epochs,
            "median_epoch_time_sec": med_epoch_time,
        }

        rows_by_dataset.setdefault(dataset, []).append(row)

    os.makedirs(out_root, exist_ok=True)

    header = [
        "cfg_name",
        "setting",
        "d",
        "m",
        "kernel",
        "best_val_acc",
        "best_val_epoch",
        "best_train_acc",
        "best_train_loss",
        "best_val_loss",
        "total_epochs",
        "median_epoch_time_sec",
    ]

    for dataset, rows in rows_by_dataset.items():
        # sort: put "Full Reg" then others, and by d then m for readability
        def sort_key(r):
            s = r["setting"]
            s_rank = 0 if s == "Full Reg" else (1 if s == "Full Perf" else 2)
            d_val = r["d"] if r["d"] is not None else 10**9
            m_val = r["m"] if r["m"] is not None else 10**9
            return (s_rank, s, d_val, m_val, r["cfg_name"])

        rows = sorted(rows, key=sort_key)

        out_path = os.path.join(out_root, f"{dataset}.txt")
        with open(out_path, "w") as f:
            f.write("\t".join(header) + "\n")
            for r in rows:
                def fmt(v):
                    if v is None:
                        return ""
                    if isinstance(v, float):
                        return f"{v:.6f}"
                    return str(v)

                f.write("\t".join(fmt(r.get(k)) for k in header) + "\n")

        print(f"[INFO] Wrote full analysis table: {out_path}")

# ----------------------------
# Easy-run presets (just have to write python3 models/model_analysis_group.py and it runs all groups)
# ----------------------------
GROUPS = {
    "Regular Attention Depth (cifar10)": [
        "configs/cifar10_baseline_d=1.yaml",
        "configs/cifar10_baseline_d=2.yaml",
        "configs/cifar10_baseline_d=8.yaml",
    ],
    "Regular Attention Depth (mnist)": [
        "configs/mnist_baseline_d=1.yaml",
        "configs/mnist_baseline_d=2.yaml",
        "configs/mnist_baseline_d=6.yaml",
    ],

    "Best setting for the layers (cifar10, d=2)": [
        "configs/cifar10_interwined_perf_d=2_m=64_variant=softmax.yaml",
        "configs/cifar10_interwined_reg_d=2_m=64_variant=softmax.yaml",
        "configs/cifar10_perf_d=2_m=64_variant=softmax.yaml",
        "configs/cifar10_perf_reg_d=2_m=64_variant=softmax.yaml",
        "configs/cifar10_reg_perf_d=2_m=64_variant=softmax.yaml",
        "configs/cifar10_baseline_d=2.yaml",
    ],
    "Best setting for the layers (cifar10, d=8)": [
        "configs/cifar10_interwined_perf_d=8_m=64_variant=softmax.yaml",
        "configs/cifar10_interwined_reg_d=8_m=64_variant=softmax.yaml",
        "configs/cifar10_perf_d=8_m=64_variant=softmax.yaml",
        "configs/cifar10_perf_reg_d=8_m=64_variant=softmax.yaml",
        "configs/cifar10_reg_perf_d=8_m=64_variant=softmax.yaml",
        "configs/cifar10_baseline_d=8.yaml",
    ],

    "Best setting for the layers (mnist, d=2)": [
        "configs/mnist_interwined_perf_d=2_m=64_variant=softmax.yaml",
        "configs/mnist_interwined_reg_d=2_m=64_variant=softmax.yaml",
        "configs/mnist_perf_d=2_m=64_variant=softmax.yaml",
        "configs/mnist_perf_reg_d=2_m=64_variant=softmax.yaml",
        "configs/mnist_reg_perf_d=2_m=64_variant=softmax.yaml",
        "configs/mnist_baseline_d=2.yaml",
    ],
    "Best setting for the layers (mnist, d=6)": [
        "configs/mnist_interwined_perf_d=6_m=64_variant=softmax.yaml",
        "configs/mnist_interwined_reg_d=6_m=64_variant=softmax.yaml",
        "configs/mnist_perf_d=6_m=64_variant=softmax.yaml",
        "configs/mnist_perf_reg_d=6_m=64_variant=softmax.yaml",
        "configs/mnist_reg_perf_d=6_m=64_variant=softmax.yaml",
        "configs/mnist_baseline_d=6.yaml",
    ],

    "ReLU vs Softmax (cifar10, d=1, m=64)": [
        "configs/cifar10_perf_d=1_m=64_variant=relu.yaml",
        "configs/cifar10_perf_d=1_m=64_variant=softmax.yaml",
    ],
    "ReLU vs Softmax (cifar10, d=2, m=64)": [
        "configs/cifar10_perf_d=2_m=64_variant=relu.yaml",
        "configs/cifar10_perf_d=2_m=64_variant=softmax.yaml",
    ],
    "ReLU vs Softmax (cifar10, d=2, m=32)": [
        "configs/cifar10_perf_d=2_m=32_variant=relu.yaml",
        "configs/cifar10_perf_d=2_m=32_variant=softmax.yaml",
    ],
    "ReLU vs Softmax (cifar10, d=8, m=64)": [
        "configs/cifar10_perf_d=8_m=64_variant=relu.yaml",
        "configs/cifar10_perf_d=8_m=64_variant=softmax.yaml",
    ],
    "ReLU vs Softmax (cifar10, d=2, m=128)": [
        "configs/cifar10_perf_d=2_m=128_variant=relu.yaml",
        "configs/cifar10_perf_d=2_m=128_variant=softmax.yaml",
    ],

    "Number of random features comparison (cifar10, softmax, d=2)": [
        "configs/cifar10_perf_d=2_m=32_variant=softmax.yaml",
        "configs/cifar10_perf_d=2_m=64_variant=softmax.yaml",
        "configs/cifar10_perf_d=2_m=128_variant=softmax.yaml",
        "configs/cifar10_perf_d=2_m=1024_variant=softmax.yaml",
        "configs/cifar10_perf_d=2_m=4096_variant=softmax.yaml",
    ],
    "Number of random features comparison (cifar10, relu, d=2)": [
        "configs/cifar10_perf_d=2_m=32_variant=relu.yaml",
        "configs/cifar10_perf_d=2_m=64_variant=relu.yaml",
        "configs/cifar10_perf_d=2_m=128_variant=relu.yaml",
        "configs/cifar10_perf_d=2_m=4096_variant=relu.yaml",
        "configs/cifar10_perf_d=2_m=1024_variant=relu.yaml",  
    ],

    "Number of random features comparison (mnist, softmax, d=2)": [
        "configs/mnist_perf_d=2_m=32_variant=softmax.yaml",
        "configs/mnist_perf_d=2_m=64_variant=softmax.yaml",
        "configs/mnist_perf_d=2_m=128_variant=softmax.yaml",
        "configs/mnist_perf_d=2_m=1024_variant=softmax.yaml",
        "configs/mnist_perf_d=2_m=4096_variant=softmax.yaml",
    ],
    "Number of random features comparison (mnist, relu, d=2)": [
        "configs/mnist_perf_d=2_m=32_variant=relu.yaml",
        "configs/mnist_perf_d=2_m=64_variant=relu.yaml",
        "configs/mnist_perf_d=2_m=128_variant=relu.yaml",
        "configs/mnist_perf_d=2_m=1024_variant=relu.yaml",
        "configs/mnist_perf_d=2_m=4096_variant=relu.yaml",
    ],

    "ReLU vs Softmax (mnist, d=1, m=64)": [
        "configs/mnist_perf_d=1_m=64_variant=relu.yaml",
        "configs/mnist_perf_d=1_m=64_variant=softmax.yaml",
    ],
    "ReLU vs Softmax (mnist, d=2, m=64)": [
        "configs/mnist_perf_d=2_m=64_variant=relu.yaml",
        "configs/mnist_perf_d=2_m=64_variant=softmax.yaml",
    ],
    "ReLU vs Softmax (mnist, d=2, m=32)": [
        "configs/mnist_perf_d=2_m=32_variant=relu.yaml",
        "configs/mnist_perf_d=2_m=32_variant=softmax.yaml",
    ],
    "ReLU vs Softmax (mnist, d=6, m=64)": [
        "configs/mnist_perf_d=6_m=64_variant=relu.yaml",
        "configs/mnist_perf_d=6_m=64_variant=softmax.yaml",
    ],
    "ReLU vs Softmax (mnist, d=2, m=128)": [
        "configs/mnist_perf_d=2_m=128_variant=relu.yaml",
        "configs/mnist_perf_d=2_m=128_variant=softmax.yaml",
    ],

    "Depth Comparison (mnist, relu, m=64)": [
        "configs/mnist_perf_d=1_m=64_variant=relu.yaml",
        "configs/mnist_perf_d=2_m=64_variant=relu.yaml",
        "configs/mnist_perf_d=6_m=64_variant=relu.yaml",
    ],
    "Depth Comparison (mnist, softmax, m=64)": [
        "configs/mnist_perf_d=1_m=64_variant=softmax.yaml",
        "configs/mnist_perf_d=2_m=64_variant=softmax.yaml",
        "configs/mnist_perf_d=6_m=64_variant=softmax.yaml",
    ],
    "Depth Comparison (cifar10, relu, m=64)": [
        "configs/cifar10_perf_d=1_m=64_variant=relu.yaml",
        "configs/cifar10_perf_d=2_m=64_variant=relu.yaml",
        "configs/cifar10_perf_d=8_m=64_variant=relu.yaml",
    ],
    "Depth Comparison (cifar10, softmax, m=64)": [
        "configs/cifar10_perf_d=1_m=64_variant=softmax.yaml",
        "configs/cifar10_perf_d=2_m=64_variant=softmax.yaml",
        "configs/cifar10_perf_d=8_m=64_variant=softmax.yaml",
    ],

    "High setting comparisons (cifar10_high, softmax, d=8, m=64)": [
        "configs/cifar10_high_interwined_perf_d=8_m=64_variant=softmax.yaml",
        "configs/cifar10_high_interwined_reg_d=8_m=64_variant=softmax.yaml",
        "configs/cifar10_high_perf_d=8_m=64_variant=softmax.yaml",
        "configs/cifar10_high_perf_reg_d=8_m=64_variant=softmax.yaml",
        "configs/cifar10_high_reg_perf_d=8_m=64_variant=softmax.yaml",
    ],
}


def main():
    full_analysis(GROUPS)

    for group_name, cfgs in GROUPS.items():
        print(f"\n=== Analyzing group: {group_name} ===")
        analyze_group(group_name, cfgs)


if __name__ == "__main__":
    main()
