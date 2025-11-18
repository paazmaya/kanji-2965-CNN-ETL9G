# Tsujimoto - Kanji Recognition Training

This project trains multiple neural network architectures for Japanese kanji character recognition using the expanded ETL6-9 dataset (4,154 classes, 934,622 samples - 53% more data than ETL9G alone).

## 📚 Documentation

| Document                                   | Purpose                                                                           |
| ------------------------------------------ | --------------------------------------------------------------------------------- |
| [**PROJECT_DIARY.md**](PROJECT_DIARY.md)   | Complete project history, all training phases, research references, key learnings |
| [**RESEARCH.md**](RESEARCH.md)             | Research findings, architecture comparisons, citations                            |
| [**model-card.md**](model-card.md)         | HuggingFace model card with carbon footprint analysis                             |

## 🚀 Quick Start

### Setup

```ps1
# Expects CUDA 13 to be available
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# Install dependencies with uv
uv sync

# Verify environment
uv run python scripts/preflight_check.py
```

### Prepare Multi-ETL Dataset

```ps1
# Process and combine ETL6, ETL7, ETL8G, ETL9G into single dataset
# (934K+ samples, 4,154 classes - includes kanji, hiragana, katakana, symbols, numerals)
uv run python scripts/prepare_dataset.py
uv run python scripts/generate_chunk_metadata.py
```

### Training

Gotta have CUA 13 available
https://developer.nvidia.com/cuda-downloads

```ps1
# CNN (fast baseline, 97.18% accuracy on ETL9G)
# Automatically uses combined_all_etl (934K) if available, else ETL9G (607K)
uv run python scripts/train_cnn_model.py --data-dir dataset

# RNN (best accuracy, 98.4% on ETL9G)
# Automatically uses combined_all_etl (934K) if available, else ETL9G (607K)
uv run python scripts/train_radical_rnn.py --data-dir dataset

# HierCode (recommended, 95.56% + quantizable on ETL9G)
# Automatically uses combined_all_etl (934K) if available, else ETL9G (607K)
uv run python scripts/train_hiercode.py --data-dir dataset --epochs 30 --checkpoint-dir models/checkpoints

# With checkpoint resume (crash-safe)
uv run python scripts/train_hiercode.py --data-dir dataset --resume-from models/checkpoints/checkpoint_epoch_015.pt --epochs 30

# QAT (lightweight deployment, 1.7 MB)
# Automatically uses combined_all_etl (934K) if available, else ETL9G (607K)
uv run python scripts/train_qat.py --data-dir dataset --checkpoint-dir models/checkpoints
```

**Dataset Selection**: All scripts automatically select the best available dataset in this priority: `combined_all_etl` → `etl9g` → `etl8g` → `etl7` → `etl6`. See [Dataset Auto-Detection Priority](#dataset-auto-detection-priority) above.

## ✅ Checkpoint Management

All training scripts support automatic checkpoint management with resume-from-latest functionality. This enables crash-safe training without manual intervention.

**How it works**:
1. **Auto-save**: After each epoch, checkpoints save to `models/checkpoints/{approach_name}/checkpoint_epoch_NNN.pt`
2. **Auto-detect**: When you re-run the training command, it automatically finds and resumes from the latest checkpoint
3. **Auto-cleanup**: Keeps only the 5 most recent checkpoints per approach to save disk space

**Directory structure**:
```
models/
└── checkpoints/
    ├── cnn/                          ← CNN checkpoints
    │   ├── checkpoint_epoch_001.pt
    │   ├── checkpoint_epoch_002.pt
    │   └── checkpoint_best.pt        ← Best accuracy so far
    ├── qat/                          ← QAT checkpoints
    ├── rnn/                          ← RNN checkpoints
    ├── vit/                          ← Vision Transformer checkpoints
    ├── hiercode/                     ← HierCode checkpoints
    └── hiercode_higita/              ← Hi-GITA variant checkpoints
```

**Usage examples**:

```ps1
# Run 1: Trains from scratch (epochs 1-15), saves checkpoints
uv run python scripts/train_cnn_model.py --data-dir dataset --epochs 30

# Interrupted at epoch 15? Just re-run - automatically resumes from epoch 16
uv run python scripts/train_cnn_model.py --data-dir dataset --epochs 30

# Resume from specific checkpoint
uv run python scripts/train_cnn_model.py --data-dir dataset --resume-from models/checkpoints/cnn/checkpoint_epoch_010.pt

# Start fresh (ignore existing checkpoints)
uv run python scripts/train_cnn_model.py --data-dir dataset --no-checkpoint

# Keep more checkpoints (default is 5, keeps last 10)
uv run python scripts/train_cnn_model.py --data-dir dataset --keep-last-n 10

# Change checkpoint directory
uv run python scripts/train_cnn_model.py --data-dir dataset --checkpoint-dir models/my_checkpoints
```

### Checkpoint Manager API

For advanced usage, you can also use the `CheckpointManager` class directly:

```python
from scripts.checkpoint_manager import CheckpointManager

# Create manager for your approach
manager = CheckpointManager("models/checkpoints", "cnn")

# Find latest checkpoint
latest = manager.find_latest_checkpoint()

# Auto-load latest checkpoint
checkpoint_data, start_epoch = manager.find_and_load_latest_checkpoint(model, optimizer, scheduler)
if checkpoint_data:
    print(f"Resumed from epoch {start_epoch}")
else:
    print("Starting fresh training")

# Save checkpoint after epoch
manager.save_checkpoint(
    epoch=10,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    metrics={"val_accuracy": 0.975, "val_loss": 0.112},
    is_best=True
)

# List all checkpoints for this approach
for checkpoint_path in manager.list_all_checkpoints():
    print(checkpoint_path)
```

### Development

```ps1
# Development commands
uv sync                    # Sync dependencies
uv sync --all-extras       # Sync with dev dependencies
uv run pytest tests/       # Run tests
uv run ruff format .       # Format code
uv run ruff check . --fix  # Lint and fix
uv run jupyter notebook    # Start Jupyter
```

## 📊 Dependency Management

This project uses **uv** for fast, reliable Python dependency management.

**Install uv (one-time)**:
- Windows: `irm https://astral.sh/uv/install.ps1 | iex`
- macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## 🔄 Model Training Pipeline

**Complete HierCode training + quantization + export:**

```ps1
# Step 1: Train HierCode (30 epochs)
uv run python scripts/train_hiercode.py --data-dir dataset --epochs 30 --checkpoint-dir models/checkpoints

# Step 2: Quantize to INT8
uv run python scripts/quantize_model.py --model-path models/hiercode_model_best.pth --calibrate --evaluate

# Step 3: Export to ONNX (float32)
uv run python scripts/export_to_onnx_hiercode.py --model-path models/hiercode_model_best.pth --verify

# Step 4: Export quantized INT8 to ONNX
uv run python scripts/export_quantized_to_onnx.py --model-path models/quantized_hiercode_int8.pth --verify --test-inference

# Step 5: Export with additional ONNX quantization (ultra-lightweight)
uv run python scripts/export_4bit_quantized_onnx.py --model-path models/quantized_hiercode_int8.pth --verify
```

**Result models**:

| File | Size | Format | Use |
|------|------|--------|-----|
| `models/hiercode_model_best.pth` | 9.56 MB | PyTorch | Fine-tuning |
| `models/quantized_hiercode_int8.pth` | 2.10 MB | PyTorch INT8 | Fast CPU |
| `models/hiercode_opset14.onnx` | 6.86 MB | ONNX | Cross-platform |
| `models/hiercode_int8_opset14.onnx` | 6.86 MB | ONNX INT8 | Portable |
| `models/hiercode_int8_4bit_opset14_quantized.onnx` | 1.75 MB | ONNX INT8 | **Edge** |

### Deployment

```ps1
# Export to ONNX
uv run python scripts/convert_to_onnx.py --model-path models/best_kanji_model.pth

# Export to SafeTensors
uv run python scripts/convert_to_safetensors.py --model-path models/best_kanji_model.pth

# Quantize INT8 PyTorch to ultra-lightweight ONNX
uv run python scripts/convert_int8_pytorch_to_quantized_onnx.py --model-path models/quantized_hiercode_int8.pth
```

### Inference (Python)

```python
import onnxruntime as ort
import numpy as np

# GPU providers auto-fallback to CPU if not available
providers = [
    ("CUDAExecutionProvider", {"device_id": 0}),
    ("CPUExecutionProvider", {}),
]
sess = ort.InferenceSession('models/hiercode_int8_quantized_quantized_int8_onnx_opset14.onnx', providers=providers)
image = np.random.randn(1, 1, 64, 64).astype(np.float32)  # 64x64 grayscale
logits = sess.run(None, {'input_image': image})[0]
prediction = np.argmax(logits[0])
```

## 📁 Project Structure

```
kanji-2965-CNN-ETL9G/
├── PROJECT_DIARY.md          ← Full project documentation (START HERE)
├── README.md                 ← This file (quick reference)
├── model-card.md             ← HuggingFace model card
├── RESEARCH.md               ← Research findings and references
│
├── scripts/                  ← Training and deployment scripts
│   ├── train_cnn_model.py  ← CNN baseline
│   ├── train_qat.py          ← Quantization-aware training
│   ├── train_radical_rnn.py  ← RNN variant
│   ├── train_hiercode.py     ← HierCode approach
│   ├── train_vit.py          ← Vision Transformer
│   ├── convert_to_onnx.py    ← ONNX export
│   ├── convert_to_safetensors.py ← SafeTensors export
│   └── rnn/                  ← RNN-specific utilities
│
├── models/                   ← Trained model checkpoints
│   ├── best_kanji_model.pth
│   ├── kanji_model.onnx
│   ├── checkpoints/          ← Resume checkpoints
│   └── ...
│
├── dataset/                  ← Preprocessed datasets
│   ├── etl9g/               ← Current ETL9G (3,036 classes, 607K samples)
│   ├── etl8g/               ← Optional ETL8G (956 classes, 153K samples)
│   ├── etl7/                ← Optional ETL7 (48 classes, 16.8K samples)
│   ├── etl6/                ← Optional ETL6 (114 classes, 157K samples)
│   ├── etl6789_combined/    ← Optional combined (4,154 classes, 934K samples)
│   ├── character_mapping.json
│   └── metadata.json
│
├── ETL9G/                   ← Raw ETL9G files (download separately)
│   ├── ETL9G_01 - ETL9G_50
│   └── ETL9GINFO
│
├── ETL8G/, ETL7/, ETL6/     ← Optional raw dataset files (download separately)
```

## 📊 Results Comparison (on ETL9G)

| Architecture           | Accuracy | Model Size    | Speed    | Format       | Deployment  | Status     |
| ---------------------- | -------- | ------------- | -------- | ------------ | ----------- | ---------- |
| **CNN**                | 97.18%   | 6.6 MB        | ⚡⚡⚡   | PyTorch      | Python/ONNX | ✅ Prod    |
| **RNN**                | 98.4%    | 23 MB         | ⚡⚡     | PyTorch      | Python/ONNX | ✅ Prod    |
| **HierCode**           | 95.56%   | 2.1 MB (INT8) | ⚡⚡⚡   | PyTorch/ONNX | Python/ONNX | ✅ Prod    |
| **HierCode INT8 ONNX** | 95.56%   | **1.67 MB**   | ⚡⚡⚡   | ONNX         | Edge/Mobile | ✅ Prod    |
| **QAT**                | 62%      | 1.7 MB        | ⚡⚡⚡⚡ | ONNX         | Embedded    | ✅ Done    |
| **ViT**                | —        | —             | —        | —            | —           | 📋 Explored |

**Dataset Expansion**: ETL6-9 combines 4 datasets (ETL6, ETL7, ETL8G, ETL9G) → 934K samples, ~4,154 classes | Expected accuracy gain: +2-3%

## 🎯 Unified Dataset Preparation

Automatically prepare all available ETLCDB datasets:

```ps1
# Download datasets from: http://etlcdb.db.aist.go.jp/download-links/
# Extract to: ETL1/, ETL2/, ..., ETL9G/ directories as needed

# Auto-detect and prepare ALL available datasets + combine:
uv run python scripts/prepare_dataset.py

# Process specific datasets only:
uv run python scripts/prepare_dataset.py --only etl9g etl8g etl7

# Process but don't combine:
uv run python scripts/prepare_dataset.py --no-combine

# Custom output directory:
uv run python scripts/prepare_dataset.py --output-dir my_datasets
```

**Features**:
- ✅ Auto-detects available ETL directories
- ✅ Processes ETL1-9G (all formats supported)
- ✅ Combines into single unified dataset
- ✅ Handles chunked output for large datasets
- ✅ Generates metadata for each dataset

### Dataset Auto-Detection Priority

All training scripts **automatically select the best available dataset** using this priority order:

```
1. combined_all_etl  ← 934K samples, 4,154 classes (recommended if available)
2. etl9g             ← 607K samples, 3,036 classes (default if combined not available)
3. etl8g             ← 153K samples, 956 classes
4. etl7              ← 16.8K samples, 48 classes
5. etl6              ← 157K samples, 114 classes
6. etl1              ← Legacy format support
```

**What this means**:
- If you prepare the combined dataset → all training scripts automatically use it (+53% more data)
- If only ETL9G exists → scripts use ETL9G
- No need to modify training commands - they adapt automatically!

```ps1
# Example: Prepare combined dataset
uv run python scripts/prepare_dataset.py

# Then train - automatically uses combined_all_etl (934K samples)
uv run python scripts/train_cnn_model.py --data-dir dataset
uv run python scripts/train_qat.py --data-dir dataset
uv run python scripts/train_radical_rnn.py --data-dir dataset
uv run python scripts/train_vit.py --data-dir dataset
uv run python scripts/train_hiercode.py --data-dir dataset
```

**Training Impact**:
| Dataset | Classes | Samples | Per-Epoch Time | Expected Accuracy |
|---------|---------|---------|---|---|
| ETL9G only | 3,036 | 607K | ~1.0x | Baseline |
| Combined (ETL6-9) | 4,154 | 934K (+53%) | ~1.5-1.8x | **+2-3% gain** |

### Character Coverage Expansion

```
Current (ETL9G only):
├─ Kanji: 2,965 (JIS Level 1)
├─ Hiragana: 71
└─ Total: 3,036 classes, 607K samples

Expanded (ETL6-9):
├─ Kanji: 2,965 (JIS Level 1)
├─ Hiragana: ~75 (ETL8G + ETL9G)
├─ Katakana: 46 (ETL6)
├─ Numerals: 10 (ETL6)
├─ Symbols: 32 (ETL6)
├─ ASCII: 26 (ETL6)
└─ Total: ~4,154 classes, 934K samples (+53%)
```

### Training Integration

```python
from scripts.load_multi_etl import load_etl_dataset

# Load combined ETL6-9 dataset
X, y, metadata = load_etl_dataset("dataset/etl6789_combined")
num_classes = metadata["num_classes"]  # ~4,154

# Use in training (compatible with all existing scripts)
model = train(X, y, num_classes=num_classes, ...)
```

**Performance Impact**:
- Training time: ~1.5-2.0x longer per epoch
- Expected accuracy gain: +2-3%
- Memory: ~7.5 GB (vs 4.6 GB for ETL9G alone)

### How Dataset Auto-Detection Works

All training scripts use intelligent dataset selection:

```python
# Priority order (checked in this sequence)
dataset_priority = [
    "combined_all_etl",  # 934K samples, 4,154 classes (best)
    "etl9g",             # 607K samples, 3,036 classes (default)
    "etl8g",             # 153K samples, 956 classes
    "etl7",              # 16.8K samples, 48 classes
    "etl6",              # 157K samples, 114 classes
    "etl1",              # Legacy format
]

# Script automatically selects first one it finds:
for dataset in dataset_priority:
    if Path(f"dataset/{dataset}").exists():
        return load_dataset(f"dataset/{dataset}")  # Use this one!
```

**Affected Scripts**:
- `scripts/train_cnn_model.py` - CNN baseline
- `scripts/train_qat.py` - Quantization-aware training
- `scripts/train_radical_rnn.py` - RNN variant
- `scripts/train_vit.py` - Vision Transformer
- `scripts/train_hiercode.py` - HierCode approach
- `scripts/train_hiercode_higita.py` - Hi-GITA variant

**Example**:
```
Your dataset/ directory contains:
├── combined_all_etl/      ← This exists
├── etl9g/
└── etl8g/

When you run: uv run python scripts/train_cnn_model.py --data-dir dataset
↓
Script finds combined_all_etl/ first → uses it (934K samples)
↓
Training automatically benefits from +53% more data!
```

See **Phase 7: Dataset Expansion** in [PROJECT_DIARY.md](PROJECT_DIARY.md) for complete details.

### Dataset Details

| Dataset | Classes | Samples | Content |
|---------|---------|---------|---------|
| **ETL6** | 114 | 157,662 | Katakana + Numerals + Symbols + ASCII |
| **ETL7** | 48 | 16,800 | Hiragana |
| **ETL8G** | 956 | 152,960 | Educational Kanji + Hiragana |
| **ETL9G** | 3,036 | 607,200 | JIS Level 1 Kanji + Hiragana |
| **Combined** | ~4,154 | 934,622 | Complete character set |

See **Phase 7: Dataset Expansion** in [PROJECT_DIARY.md](PROJECT_DIARY.md) for complete details.

## 🎯 Model Recommendations

- **Best Accuracy**: RNN (98.4%) - Use for high-precision applications
- **Best Balance**: CNN (97.18%) - Fast, accurate, easy to deploy
- **Best for Deployment**: HierCode INT8 ONNX (1.67 MB) - Ultra-lightweight, 82% size reduction
- **Best for Edge**: HierCode INT8 ONNX - Runs on Raspberry Pi, Jetson Nano, IoT devices
- **Best for Mobile**: HierCode INT8 ONNX - CoreML/ONNX Runtime support

## 🔗 Resources

- **ETL9G Dataset**: http://etlcdb.db.aist.go.jp/download-links/
- **Research References**: See RESEARCH.md or PROJECT_DIARY.md
- **Model Card**: [model-card.md](model-card.md) (HuggingFace format)
- **Deployment Guide**: See model-specific sections in PROJECT_DIARY.md

## 📖 Related Papers (Building on HierCode)

Recent research (2022-2025) extends HierCode with improved techniques:

- **Hi-GITA** (2505.24837, May 2025): Hierarchical multi-granularity image-text alignment, 20% improvement
- **RZCR** (2207.05842, July 2022): Knowledge graph reasoning over radicals
- **STAR** (2210.08490, October 2022): Stroke + radical level decompositions
- **MegaHan97K** (2506.04807, June 2025): 97K character benchmark dataset

See [PROJECT_DIARY.md](PROJECT_DIARY.md) Phase 6 for detailed analysis and integration opportunities.

## ❓ FAQ

**Q: Where's the full project documentation?**  
A: [PROJECT_DIARY.md](PROJECT_DIARY.md) - Complete history, all approaches, results, references.

**Q: Which model should I use?**  
A: See "Model Recommendations" above. CNN for speed (97.18%), RNN for accuracy (98.4%), HierCode INT8 for deployment (1.67 MB, 82% reduction).

**Q: Which dataset do the training scripts use?**  
A: Scripts auto-detect in this priority: `combined_all_etl` (934K, recommended) → `etl9g` (607K, default) → `etl8g` → `etl7` → `etl6`. Prepare the combined dataset with `uv run python scripts/prepare_dataset.py` and training scripts will automatically use it (+53% more data).

**Q: Why isn't my training using the combined dataset?**  
A: Prepare it first: `uv run python scripts/prepare_dataset.py`. Scripts check for `dataset/combined_all_etl/chunk_info.json` before falling back to `dataset/etl9g/`.

**Q: How do I handle training crashes?**  
A: All scripts have checkpoint/resume system built in. Checkpoints auto-save after each epoch in `models/checkpoints/{approach}/`. Resume with `--resume-from models/checkpoints/{approach}/checkpoint_epoch_015.pt` or it auto-detects the latest checkpoint if you just re-run the command.

**Q: What approaches support automatic checkpoint resumption?**  
A: All 6 training scripts: `train_cnn_model.py`, `train_qat.py`, `train_radical_rnn.py`, `train_vit.py`, `train_hiercode.py`, `train_hiercode_higita.py`. Each uses its own checkpoint folder: `models/checkpoints/cnn/`, `models/checkpoints/qat/`, `models/checkpoints/rnn/`, `models/checkpoints/vit/`, `models/checkpoints/hiercode/`, `models/checkpoints/hiercode_higita/`.

**Q: How do I resume from the latest checkpoint?**  
A: Just re-run the training command and it automatically resumes from the latest checkpoint found:
```ps1
# First run - trains from scratch
uv run python scripts/train_cnn_model.py --data-dir dataset --epochs 30

# If interrupted, just run again - resumes automatically
uv run python scripts/train_cnn_model.py --data-dir dataset --epochs 30
```

**Q: How do I start fresh training and ignore old checkpoints?**  
A: Use the `--no-checkpoint` flag:
```ps1
uv run python scripts/train_cnn_model.py --data-dir dataset --no-checkpoint
```

**Q: Can I manually specify which checkpoint to resume from?**  
A: Yes, use `--resume-from`:
```ps1
uv run python scripts/train_cnn_model.py --data-dir dataset --resume-from models/checkpoints/cnn/checkpoint_epoch_010.pt
```

**Q: How many checkpoints are kept?**  
A: By default, the 5 most recent checkpoints are kept per approach. Older checkpoints auto-delete to save disk space. Customize with `--keep-last-n N` flag.

**Q: How do I deploy to edge/mobile?**  
A: Use `hiercode_int8_quantized_quantized_int8_onnx_opset14.onnx` (1.67 MB). Supports ONNX Runtime, TensorRT, CoreML, TVM.

**Q: Can I use pre-trained models?**  
A: Models in `models/` are ready to use. Load with `torch.load()` or `ort.InferenceSession()` for ONNX.

**Q: How do I add new training approaches?**  
A: See `scripts/optimization_config.py` for unified config system. Inherit from `OptimizationConfig` class.

**Q: Why use `uv run python` instead of just `python`?**  
A: `uv` provides isolated, reproducible environments with locked dependency versions. Prevents version conflicts and ensures consistency across machines.

## 📝 System Requirements

- **OS**: Windows 11, Linux, or macOS
- **Python**: 3.11+ (tested with 3.13)
- **GPU**: NVIDIA GPU with CUDA 11.8+ recommended
- **RAM**: 8+ GB (16+ GB recommended)
- **Storage**: 15+ GB free space

## 🚀 Next Steps

1. **Read** [PROJECT_DIARY.md](PROJECT_DIARY.md) for complete project overview
2. **Setup** environment with `uv sync` (or `uv pip install -r requirements.txt`)
3. **Prepare** datasets with `uv run python scripts/prepare_dataset.py`
4. **Train** model with `uv run python scripts/train_cnn_model.py --data-dir dataset`
5. **Export** to ONNX for deployment with `uv run python scripts/convert_to_onnx.py`

---

## 📋 Project Summary

**Goal**: Multi-architecture platform for Japanese kanji recognition  
**Current Dataset**: ETL9G (3,036 classes, 607,200 samples, 64×64 grayscale images)  
**Expanded Dataset**: ETL6-9 (4,154 classes, 934,622 samples - includes kanji, hiragana, katakana, symbols, numerals)  
**Best Result (ETL9G)**: RNN 98.4% accuracy | HierCode 95.56% accuracy at 1.67 MB (quantized ONNX)  
**Key Achievement**: 82% size reduction while maintaining 95.56% accuracy; now expandable to 934K samples with ETL6-9  
**Dependency Management**: All scripts run via `uv run python` for reproducible, isolated environments

**Repository**: https://github.com/paazmaya/kanji-2965-CNN-ETL9G  
**Owner**: Jukka Paazmaya (@paazmaya)  
**License**: MIT (see LICENSE file)  
**Last Updated**: November 17, 2025 (Unified dataset prep + uv standardization)
