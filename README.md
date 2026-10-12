# Resource-Constrained CNN for Vegetable Image Classification

**EN3150 Pattern Recognition: Assignment 03** (Simple convolutional neural network to perform classification)

This project builds a **lightweight image classifier for embedded / edge devices** (microcontrollers, Raspberry Pi, edge nodes) that recognises 15 types of vegetables from low-resolution **64×64** images. We:

1. Design two custom CNNs in TensorFlow/Keras:
   - **Model A (Standard CNN):** interleaved `Conv2D` + `MaxPooling2D` layers.
   - **Model B (Lightweight CNN):** depthwise-separable convolutions, **≤ 100,000 trainable parameters**.
2. Compare optimizers: Adam vs SGD vs SGD + Momentum, including a sweep over the momentum value.
3. Train and evaluate both custom models: loss curves, accuracy, confusion matrix, precision, recall, model size, and time per epoch.
4. Fine-tune two **pre-trained lightweight SOTA networks** (MobileNetV2, EfficientNet-B0) on the same data splits.
5. Compare the custom sub-100k model against the SOTA models on **accuracy vs memory footprint vs computational cost**.

---

## Group members

| Name | Index No. | GitHub |
|---|---|---|
| _Member 1_ | _xxxxxxX_ | [@thevinufernando](https://github.com/thevinufernando) |
| _Member 2_ | _xxxxxxX_ | |
| _Member 3_ | _xxxxxxX_ | |
| _Member 4_ | _xxxxxxX_ | |

Repository: <https://github.com/thevinufernando/cnn-based-vegetable-classification-model>

---

## Progress

| # | Assignment task | Marks | Notebook / script | Status |
|---|---|---|---|---|
| 1 | Data preparation (download, 64×64 downscale, 70/15/15 split) | – | `scripts/download_data.py`, `scripts/prepare_data.py` | ✅ Done |
| 1 | Dataset exploration | – | `notebooks/01_data_exploration.ipynb` | ⬜ To do |
| 2 | Custom architecture design (Model A & B, param counts, activations) | 20 | `notebooks/02_architectures.ipynb` | ⬜ To do |
| 3 | Optimizer selection & tuning (Adam vs SGD vs SGD+Momentum) | 15 | `notebooks/03_optimizer_comparison.ipynb` | ⬜ To do |
| 4 | Custom model training & evaluation + A vs B comparison | 25 | `notebooks/04_train_eval_custom.ipynb` | ⬜ To do |
| 5 | SOTA fine-tuning & evaluation (MobileNetV2, EfficientNet-B0) | 20 | `notebooks/05_sota_finetuning.ipynb` | ⬜ To do |
| 6 | Final comparison & trade-off discussion | 20 | `notebooks/06_final_comparison.ipynb` | ⬜ To do |
| – | Report | – | _(Overleaf / Word, see Submission)_ | ⬜ To do |

> Update the **Status** column (and add your name if you own a task) whenever you push work.

---

## Dataset

**Vegetable Image Dataset** by M. Israk Ahmed: <https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset>

- **15 classes:** Bean, Bitter_Gourd, Bottle_Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish, Tomato
- **21,000** RGB JPG images at 224×224 (1,400 per class), shipped with a train / validation / test split that the author describes as 70 / 15 / 15.
- To emulate a low-resource sensor, every image is **downscaled to 64×64** (LANCZOS resampling) and stored as a lossless PNG.

**Split used in this project** (Kaggle's provided split, after removing exact duplicate files):

| Split | Images | Per class | Share |
|---|---:|---:|---:|
| Train | 14,996 | ~1,000 | 71.4% |
| Validation | 3,000 | 200 | 14.3% |
| Test | 3,000 | 200 | 14.3% |
| **Total** | **20,996** | | |

4 exact-duplicate images inside the training set were removed by `prepare_data.py`. The full image → split mapping is saved in `data/processed/split_manifest.csv`. All models (custom and SOTA) use **exactly these splits**.

---

## Workspace structure

```
vegCNN/
├── data/                         # (git-ignored) everyone generates it locally with the scripts
│   ├── raw/                      #   original Kaggle download: Vegetable Images/{train,validation,test}/<Class>/*.jpg
│   └── processed/                #   64×64 PNGs: {train,val,test}/<Class>/*.png + split_manifest.csv
├── scripts/
│   ├── download_data.py          # downloads the dataset into data/raw with kagglehub
│   └── prepare_data.py           # de-duplicates, splits and downscales images into data/processed
├── src/                          # shared Python code imported by the notebooks
│   ├── config.py                 #   paths, IMG_SIZE=64, BATCH_SIZE, SEED, class list          (planned)
│   ├── data.py                   #   tf.data loaders for train/val/test                         (planned)
│   ├── models.py                 #   Model A, Model B, MobileNetV2, EfficientNet-B0 builders    (planned)
│   ├── train_utils.py            #   training loop helpers, epoch timer, model saving & size   (planned)
│   └── eval_utils.py             #   metrics, confusion matrix, loss-curve plots               (planned)
├── notebooks/                    # one notebook per assignment task (01 → 06)                  (planned)
├── models/                       # (git-ignored) saved .keras / .tflite models                  (planned)
├── results/
│   ├── figures/                  # loss curves, confusion matrices (used in the report)         (planned)
│   └── metrics/                  # per-model metrics JSON + comparison tables (CSV)             (planned)
├── taskdoc/                      # (git-ignored) assignment brief PDF
├── requirements.txt              # pinned dependency versions
├── .gitignore
└── README.md
```

---

## Dependencies

Developed and tested with **Python 3.13.12** on Windows 11 (CPU only).

| Package | Version | Used for |
|---|---|---|
| tensorflow | 2.21.0 | Building / training CNNs, TFLite export |
| keras | 3.15.1 | High-level model API (bundled with TensorFlow) |
| kagglehub | 1.0.2 | Downloading the dataset from Kaggle |
| numpy | 2.5.4 | Array operations |
| pandas | 3.0.6 | Result tables / comparisons |
| pillow | 12.3.0 | Image loading and downscaling |
| scikit-learn | 1.9.1 | Stratified splitting, precision / recall / confusion matrix |
| matplotlib | 3.11.2 | Plots (loss curves, sample grids) |
| seaborn | 0.13.2 | Confusion-matrix heatmaps |
| tqdm | 4.70.1 | Progress bars |
| jupyter, ipykernel | 1.1.1, 7.4.0 | Running the notebooks |

The exact versions of all packages (including sub-dependencies) are pinned in `requirements.txt`.

> **Note:** Native-Windows TensorFlow runs on **CPU only**. That is fine here: at 64×64 the models are small.
> GPU training on Windows requires WSL2. On Linux, `pip install tensorflow[and-cuda]` enables an NVIDIA GPU.

---

## Getting started

### 1. Clone the repository
```bash
git clone https://github.com/thevinufernando/cnn-based-vegetable-classification-model.git
cd cnn-based-vegetable-classification-model
```

> **Windows users:** TensorFlow has very deep file paths. Clone into a **short path** (e.g. `D:\Projects\vegCNN`)
> **or** enable long paths (Admin PowerShell):
> `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`.
> Otherwise `pip install tensorflow` fails with an `OSError: No such file or directory`.

### 2. Create a virtual environment and install dependencies

**Windows (PowerShell)**
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check the installation:
```bash
python -c "import tensorflow as tf, keras; print(tf.__version__, keras.__version__)"
```

Register the venv as a Jupyter kernel (then choose **"Python (veg-cnn)"** in VS Code / Jupyter):
```bash
python -m ipykernel install --user --name veg-cnn --display-name "Python (veg-cnn)"
```

### 3. Download and prepare the dataset
Run from the project root with the venv activated:
```bash
python scripts/download_data.py      # ~545 MB into data/raw (auto-extracted)
python scripts/prepare_data.py       # ~180 MB data/processed (64x64 PNGs + split_manifest.csv)
```

| `prepare_data.py` option | Default | Description |
|---|---|---|
| `--split-mode {original,resplit}` | `original` | Keep Kaggle's split, or make a fresh stratified 70/15/15 split |
| `--size N` | `64` | Output resolution (max 64, per the assignment) |
| `--seed N` | `42` | Random seed (resplit mode only) |
| `--overwrite` | off | Delete and rebuild `data/processed` |
| `--raw-dir`, `--out-dir` | `data/raw`, `data/processed` | Custom input / output folders |

The dataset is public, so no Kaggle login is normally needed. If the download fails with 401/403, create an API token on
kaggle.com (Settings → API → *Create New Token*) and put `kaggle.json` in `~/.kaggle/` (Windows: `%USERPROFILE%\.kaggle\`).

**Everyone must use the default split** (`--split-mode original`) so results from different members are comparable.

### 4. Run the experiments
Run the notebooks in order: `01` → `06`. Each notebook reads `data/processed`, uses the shared code in `src/`, and
writes figures to `results/figures/` and metrics to `results/metrics/`. _(Added as the project progresses.)_

---

## Methodology

### Data pipeline
1. **Collect:** all images in `data/raw`. The class comes from the parent folder, the original split from the grand-parent folder.
2. **De-duplicate:** MD5 hash of each file. Exact copies are dropped, which prevents train/test leakage.
3. **Split:** Kaggle's provided train/validation/test split (or a stratified 70/15/15 re-split with `--split-mode resplit`).
4. **Downscale:** 224×224 → 64×64 RGB, lossless PNG.
5. **Load:** `keras.utils.image_dataset_from_directory` → `tf.data` pipeline with caching and prefetching. Light augmentation (flip / rotation / zoom) is applied during training only.

### Models (planned)

| Model | Key layers | Trainable params (target) |
|---|---|---|
| **A: Standard CNN** | 4 × [Conv2D 3×3 (32→64→128→256) + BatchNorm + ReLU + MaxPool] → GlobalAvgPool → Dense 128 → Dense 15 (softmax) | ≈ 424 k |
| **B: Lightweight CNN** | Same topology with **SeparableConv2D** (depthwise 3×3 + pointwise 1×1) | ≈ 81 k (< 100 k) |
| **MobileNetV2** | ImageNet pre-trained, fine-tuned, new 15-class head | ≈ 2.3 M |
| **EfficientNet-B0** | ImageNet pre-trained, fine-tuned, new 15-class head | ≈ 4.1 M |

- **Activation:** ReLU in hidden layers (a single comparison, no exponentials, friendly to int8 quantization on edge hardware). Softmax only at the output.
- **Optimizers compared:** Adam (lr = 1e-3), SGD (lr = 1e-2), and SGD + Momentum (lr = 1e-2, β ∈ {0.5, 0.9, 0.99}).
- **Training:** at least 20 epochs per custom model, sparse categorical cross-entropy, best-val-loss checkpointing.
- **Evaluation:** test accuracy, per-class and macro precision / recall, confusion matrix, parameter count, model size (KB / MB, `.keras` and `.tflite`), training time per epoch.

---

## Collaboration guide

### Branching workflow
- `main` always holds **working, runnable** code. Don't push broken notebooks to `main`.
- Create a branch per task: `git checkout -b feature/<task-name>` (e.g. `feature/model-b`, `feature/sota-mobilenet`).
- Push your branch and open a **Pull Request** into `main`. Another member reviews and merges it.
- Before you start work: `git checkout main && git pull`, then rebase or merge `main` into your branch.

### Commit regularly
The lecturer checks the commit history to see that the work was done over a reasonable period. **Commit small, commit often** and use clear messages:
```
Add Model B with separable convolutions
Plot optimizer comparison curves
Fix class order in confusion matrix
```

### Notebooks and merge conflicts
- **One owner per notebook** at a time. Notebook JSON is hard to merge, so agree in the group chat who is editing which notebook.
- Put reusable code (data loading, model builders, plotting) in `src/` instead of copying it between notebooks. `.py` files merge cleanly.
- Before committing a notebook, run **Kernel → Restart & Run All** so its outputs are complete and in order.

### What NOT to commit
`data/`, `models/`, `.venv/` and `taskdoc/` are git-ignored. Never force-add them. Large files make the repo unusable.
Commit **small result files** (`results/figures/*.png`, `results/metrics/*.json|csv`). These are what the report uses.

### Adding a new package
```bash
pip install <package>
pip freeze > requirements.txt        # then commit requirements.txt
```
Tell the group so everyone runs `pip install -r requirements.txt` again.
Windows-only packages (e.g. `pywinpty`) need a `; sys_platform == "win32"` marker in `requirements.txt` so the file still installs on macOS / Linux.

### Reproducibility
- Use the shared seed (`SEED = 42`) and the shared splits in `data/processed`.
- Record the hardware a timing was measured on (CPU model), because training times differ between laptops. Run all timing comparisons on **one** machine.

---

## Submission

- **Report (PDF)** and **code (zip)** are uploaded separately to Moodle, named `<GroupNo>_A03_EN3150`.
- The report must include the group members' names and index numbers, and the link to this GitHub repository.
- The code must be complete, commented and runnable (the steps in *Getting started* reproduce everything).

## References
- Dataset: M. I. Ahmed et al., *Vegetable Image Dataset*, Kaggle.
- [Keras: Image classification from scratch](https://keras.io/examples/vision/image_classification_from_scratch/)
- [Keras: Image classification via fine-tuning with EfficientNet](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/)
- A. G. Howard et al., *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*, 2017.
- M. Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018.
- M. Tan and Q. V. Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019.
