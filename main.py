"""
Entry point to launch one or multiple training runs from config files.

For each config in HYBRID_CONFIGS, this script:
- trains a HybridViT model defined in models/hybrid_vit.py
- logs metrics to ./runs/<config_name>/metrics.csv
- then runs post-training analysis, which:
    - reads metrics from ./runs/<config_name>/metrics.csv
    - creates plots + summary in ./outputs/<config_name>/
"""

import torch

from models.hybrid_vit import train_hybrid_performer_from_cfg
from models.model_analysis import analyze_config  # <- NEW


# List of config files you want to run
HYBRID_CONFIGS = [
    "configs/mnist_test.yaml",
    "configs/cifar10_test.yaml",
]


def main():
    """
    Loop over all config files in HYBRID_CONFIGS:
    - train one model per config
    - run post-training analysis right after.
    """
    # Choose device once and pass it to all runs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    for cfg_path in HYBRID_CONFIGS:
        # 1) Train model (writes metrics.csv in ./runs/<config_name>/)
        train_hybrid_performer_from_cfg(cfg_path, device=device)

        # 2) Run analysis (reads ./runs/<config_name>/metrics.csv
        #    and writes plots + summary to ./outputs/<config_name>/)
        try:
            analyze_config(cfg_path)
        except Exception as e:
            print(f"[WARN] Post-training analysis failed for {cfg_path}: {e}")


if __name__ == "__main__":
    main()
