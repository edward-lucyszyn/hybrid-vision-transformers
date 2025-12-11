# Hybrid Vision Transformers with Performer Attention

This project implements small hybrid Vision Transformers combining regular self-attention and Performer attention.
We compare:

* regular ViT (all softmax attention)
* full-Performer models (ReLU and softmax-kernel variants)
* hybrid models with different layer patterns (Performer→Reg, Reg→Performer, intertwined)

on MNIST and CIFAR-10, and log accuracy, parameter counts, and training time.

---

## 1. Installation and dependencies

It is recommended to use a virtual environment (for example `.venv` or `.env`).

Create and activate a virtual environment (example with `.venv`):

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

Install the core dependencies (you can adapt torch/torchvision versions to your system):

```bash
pip install torch torchvision torchaudio
pip install pyyaml tqdm
```

If you already have a working environment for the project and just want to generate a `requirements.txt` that captures the exact versions you used, simply run:

```bash
# Assuming your virtualenv (.venv / .env) is already activated
pip freeze > requirements.txt
```

To reproduce the environment on another machine:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Repository structure

The repository is organized as follows:

```text
.
├── main.py
├── models/
│   ├── regular_vit.py     # regular vit architecture (did at first, not useful anymore for the hybrid vit)
│   ├── model_analysis.py  # generates graphs from runs results
│   └── hybrid_vit.py      # main model
├── configs/
│   ├── mnist_baseline_*.yaml
│   ├── mnist_perf_*.yaml
│   ├── mnist_reg_perf_*.yaml
│   ├── mnist_perf_reg_*.yaml
│   ├── mnist_interwined_*.yaml
│   ├── cifar10_baseline_*.yaml
│   ├── cifar10_perf_*.yaml
│   ├── cifar10_reg_perf_*.yaml
│   ├── cifar10_perf_reg_*.yaml
│   └── cifar10_interwined_*.yaml
├── data/
│   ├── dataloaders.py
│   ├── cache/          # downloaded datasets are stored here
│   └── data.ipynb      # quick notebook to inspect how data is loaded
├── runs/
│   └── ...             # created automatically on first run
├── outputs/
│   └── ...             # created for aggregated metrics / plots
└── README.md
```

Notes:

* `models/hybrid_vit.py` contains:

  * the PatchEmbedding module
  * regular multi-head self-attention
  * Performer attention (ReLU and softmax-kernel variants)
  * the HybridPerformer model and training/evaluation loops

* `data/dataloaders.py` builds the train/val/test dataloaders based on the YAML config (MNIST or CIFAR-10, image size, patches, augmentation, batch size, etc.).

* `data/data.ipynb` is a small notebook to quickly visualize and understand how the data are loaded and preprocessed before focusing on model/optimization details.

* `runs/` and `outputs/` are created automatically when you run the code:

  * each experiment gets its own subdirectory in `runs/` (one per config file)
  * metrics, parameter counts, and checkpoints are stored there
  * `outputs/` can be used by analysis scripts/notebooks to save aggregated results and plots.

---

## 3. Configuration files

Each experiment is defined by a YAML config in `configs/`.
A typical config contains:

* dataset settings (MNIST or CIFAR-10, image size, patch size, augmentation)

* model settings:

  * `d_model`, `heads`, `mlp_ratio`, `depth`
  * `layers`: list of "Reg" or "Perf" defining the stack of blocks
  * `performer.variant`: "softmax" or "relu"
  * `performer.m`: number of random features

* optimization settings (batch size, learning rate, weight decay, number of epochs)

* logging/output settings (`out_dir`, `save_every`)

* misc settings (random seed)

Example excerpt:

```yaml
experiment: cifar10_perf_d=2_m=64_variant=softmax
dataset:
  name: cifar10
  root: ./data/cache
  img_size: 32
  patch: 4
  augment: true
model:
  d_model: 64
  heads: 4
  mlp_ratio: 2.0
  depth: 6
  layers: ["Perf", "Perf", "Perf", "Perf", "Perf", "Perf"]
  performer:
    variant: "softmax"
    kind: "favor+"
    m: 64
optim:
  batch_size: 128
  lr: 0.0003
  weight_decay: 0.05
  epochs: 40
log:
  out_dir: ./runs
  save_every: 5
misc:
  seed: 17092003
```

---

## 4. Running experiments

The main entry point is `main.py`.
It imports the high-level training function from `models/hybrid_vit.py` and specifies which configs to run.

Example structure of `main.py`:

```python
import torch
from models.hybrid_vit import train_hybrid_performer_from_cfg

HYBRID_CONFIGS = [
    "configs/mnist_baseline_d=6.yaml",
    "configs/mnist_perf_d=2_m=64_variant=softmax.yaml",
    "configs/cifar10_baseline_d=2.yaml",
    "configs/cifar10_perf_d=2_m=64_variant=softmax.yaml",
    # add or remove any config file you want to run
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    for cfg_path in HYBRID_CONFIGS:
        train_hybrid_performer_from_cfg(cfg_path, device=device)

if __name__ == "__main__":
    main()
```

To launch the trainings, simply activate your environment and run:

```bash
python3 main.py
```

The code will:

1. Load each YAML config listed in `HYBRID_CONFIGS`.
2. Build the corresponding dataloaders (MNIST or CIFAR-10).
3. Construct the HybridPerformer model according to the `layers` and `performer` settings.
4. Train the model for the specified number of epochs.
5. Evaluate on the validation set after each epoch.
6. Optionally save checkpoints every `save_every` epochs.

---

## 5. Outputs and logging

For each config file `configs/XYZ.yaml`, the code creates a run directory:

```text
runs/XYZ/
    metrics.csv
    number_of_parameters.txt
    checkpoint_epoch_10.pt
    checkpoint_epoch_20.pt
    ...
```

* `metrics.csv`: one row per epoch, containing:

  * epoch index
  * epoch time
  * training loss and accuracy
  * validation loss and accuracy

* `number_of_parameters.txt`: total number of trainable parameters for that model.

* `checkpoint_epoch_*.pt`: model and optimizer state dicts saved periodically according to `save_every` in the config.

The `outputs/` directory is intended for higher-level analysis artifacts, for example:

* aggregated CSV files comparing models
* plots of accuracy vs. number of random features m
* plots of accuracy vs. number of parameters
* training/inference time comparisons

You can generate these from your own analysis scripts or notebooks by reading the CSV files and parameter counts from the `runs/` directory and writing your results into `outputs/`.

---

## 6. Inspecting the data (optional)

The notebook:

```text
data/data.ipynb
```

provides a quick way to:

* visualize sample images from MNIST and CIFAR-10
* check image sizes and patching behavior
* verify that the training/validation/test splits and augmentations are as expected

This is mainly for understanding and debugging the data pipeline before running the full set of experiments.

---

With this setup, you can:

* plug different configs into `HYBRID_CONFIGS` in `main.py`
* run `python3 main.py`
* then compare the models using the logs in `runs/` and any additional plots in `outputs/`.
