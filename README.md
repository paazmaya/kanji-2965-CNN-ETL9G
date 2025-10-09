# ETL9G Kanji Recognition Training

This project trains an enhanced Lightweight [Convolutional Neural Network (CNN)](https://learnopencv.com/understanding-convolutional-neural-networks-cnn/) with **SENet-style channel attention** for Japanese kanji character recognition using the ETL9G dataset.

## Overview

### Enhanced CNN Architecture (v2.1)

- **5-Layer CNN**: Depthwise separable convolutions with progressive channel expansion (1→32→64→128→256→512)
- **Channel Attention**: 3 SENet-style attention modules for adaptive feature weighting
- **Accuracy Target**: 90-92% validation accuracy (improved from 85% baseline)
- **Web-Optimized**: ~15MB model size, still suitable for WASM deployment

### ETL9G Dataset

The ETL9G dataset contains:

- **3,036 character classes** (2,965 JIS Level 1 Kanji + 71 Hiragana)
- **607,200 total samples** from 4,000 different writers
- **128×127 pixel images** with 16 grayscale levels
- Comprehensive coverage including common kanji like rice field (田)

Details at http://etlcdb.db.aist.go.jp/database-development/#ETL9

Downloaded from http://etlcdb.db.aist.go.jp/download-links/

Data set definition http://etlcdb.db.aist.go.jp/etlcdb/etln/form_e9g.htm

## Prerequisites

### System Requirements

- **OS**: Windows 11
- **Shell**: PowerShell 5.1 or PowerShell 7+
- **RAM**: 8+ GB (16+ GB recommended)
- **GPU**: NVIDIA GPU with CUDA support (GTX 1060/RTX 2060 or better recommended)
- **Disk Space**: 10+ GB free
- **Python**: 3.8 or later
- **CUDA**: CUDA 11.8 or 12.x (automatically installed with PyTorch)

### Python Dependencies

This project uses [uv](https://docs.astral.sh/uv/) for fast and reliable Python package management:

```powershell
# Install uv (if not already installed)
# Via pip: pip install uv
# Via pipx: pipx install uv
# Via winget: winget install astral-sh.uv

# Create project with uv (automatically creates virtual environment)
uv sync

# Install PyTorch with CUDA support (special index configuration in pyproject.toml)
uv add torch torchvision --index pytorch

# Activate the virtual environment
.venv\Scripts\Activate.ps1

# Alternatively, run commands directly with uv:
uv run python scripts/preflight_check.py
```

**Alternative pip installation** (if you prefer traditional pip):

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA support for NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# Install other required packages
pip install -r requirements.txt
```

**Note**: Always ensure your virtual environment is activated before running any Python scripts. You should see `(venv)` or `(.venv)` in your PowerShell prompt. Ensure your NVIDIA drivers are up to date.

## Dataset Setup

1. **Download ETL9G Dataset**

- Obtain ETL9G files from [ETL Character Database](http://etlcdb.db.aist.go.jp/download-links/)
- Place all files in the `ETL9G/` subdirectory

2. **Verify Dataset Structure**

```
training/
├── ETL9G/
│   ├── ETL9G_01
│   ├── ETL9G_02
│   ├── ...
│   ├── ETL9G_50
│   └── ETL9INFO
```

The `ETL9G_` files are a bit over 94 MB each.

## Project Structure

After setting up and running the training pipeline, your project will have this structure:

```
kanji-2965-CNN-ETL9G/
├── scripts/                    # Python training scripts
│   ├── train_etl9g_model.py   # Main training script
│   ├── convert_to_onnx.py     # ONNX conversion for web deployment
│   ├── convert_to_safetensors.py # SafeTensors conversion
│   ├── prepare_etl9g_dataset.py  # Data preprocessing
│   └── ... (11 more scripts)
├── models/                     # Generated model files
│   ├── best_kanji_model.pth    # Best PyTorch model (training output)
│   ├── best_model_info.json   # Model metadata
│   ├── training_progress.json # Training metrics
│   ├── *.onnx                 # ONNX models for web deployment
│   ├── *.safetensors          # SafeTensors models for secure deployment
│   └── *_mapping.json         # Character mappings for each model
├── dataset/                    # Processed training data
│   ├── etl9g_dataset_chunk_*.npz  # Training data chunks
│   ├── character_mapping.json     # Character class mappings
│   └── metadata.json             # Dataset information
├── ETL9G/                      # Original ETL9G dataset files
│   ├── ETL9G_01 through ETL9G_50
│   └── ETL9INFO
├── pyproject.toml              # Project configuration and dependencies (uv)
├── .python-version             # Python version specification
├── uv.lock                     # Locked dependencies (auto-generated)
├── requirements.txt            # Legacy pip dependencies (kept for compatibility)
└── README.md                  # This documentation
```

## Training Workflow

### Step 0: Environment Setup

Before running any training commands, set up your environment with uv:

```powershell
# Install uv (if not already installed)
winget install astral-sh.uv

# Create project environment and install dependencies
uv sync

# Activate the virtual environment
.venv\Scripts\Activate.ps1
```

**Alternative with pip**:

```powershell
# Create and activate virtual environment (first time only)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies with CUDA support (first time only)
pip install -r requirements.txt
```

**Important**: Always ensure your virtual environment is activated before running any Python scripts. You should see `(.venv)` or `(venv)` in your PowerShell prompt.

### Step 1: Pre-flight Check

Verify your system setup and requirements:

```powershell
# With uv (recommended)
uv run python scripts/preflight_check.py

# Or with activated virtual environment
python scripts/preflight_check.py
```

**Expected Output:**

- ✅ All Python packages available
- ✅ ETL9G data files present
- ✅ Sufficient system resources
- ✅ Training scripts ready

### Step 2: Data Preparation

Convert ETL9G binary files to training-ready format:

```powershell
# With uv (recommended)
uv run python scripts/prepare_etl9g_dataset.py --etl-dir ETL9G --output-dir dataset --size 64

# Or with activated virtual environment
python scripts/prepare_etl9g_dataset.py --etl-dir ETL9G --output-dir dataset --size 64
```

**Parameters:**

- `--etl-dir`: Directory containing ETL9G files
- `--output-dir`: Output directory for processed dataset
- `--size`: Target image size (default: 64x64 pixels)
- `--workers`: Number of parallel workers (default: auto)

**Expected Output:**

- `dataset/etl9g_dataset_chunk_*.npz`: Chunked dataset files
- `dataset/metadata.json`: Dataset metadata
- `dataset/character_mapping.json`: Character mappings
- `dataset/chunk_info.json`: Chunk information

**Processing Time:** 30-60 minutes (depending on system)

### Step 3: Setup Verification

Test the prepared dataset and model architecture:

```powershell
python scripts/test_etl9g_setup.py --data-dir dataset --test-model
```

**Expected Output:**

- Dataset statistics and sample visualization
- Model architecture verification
- Character mapping analysis (including rice field kanji)
- Estimated model size, for 64 pixel size, about 6.6 MB

### Step 4: Training Test Run

Perform a quick test with limited data to verify the training pipeline:

**Recommended for initial testing (smaller sample size for faster training):**

```powershell
python scripts/train_etl9g_model.py --data-dir dataset --epochs 2 --batch-size 32 --sample-limit 15000
```

**Alternative commands:**

```powershell
# More samples for better testing (still limited for speed)
python scripts/train_etl9g_model.py --data-dir dataset --epochs 2 --batch-size 32 --sample-limit 50000

# Very quick test (may show lower accuracy due to too few samples per class)
python scripts/train_etl9g_model.py --data-dir dataset --epochs 2 --batch-size 32 --sample-limit 5000
```

**Parameters:**

- `--sample-limit 15000`: Use 15,000 samples total (distributed across all 3,036 classes)
- ETL9G dataset uses all 3,036 character classes (fixed)

**Expected Output:**

- Quick training completion (5-10 minutes)
- Model convergence verification
- Training logs and progress files

### Step 5: Full Model Training

Train the complete model for production use:

```powershell
python scripts/train_etl9g_model.py --data-dir dataset --epochs 30 --batch-size 64
```

**Parameters:**

- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Training batch size (default: 64)
- `--learning-rate`: Learning rate (default: 0.001)
- `--sample-limit`: Limit samples for testing (optional)

**Expected Training Time with NVIDIA GPU:**

- **RTX 4090/4080**: 45-90 minutes
- **RTX 3080/3070**: 1.5-2.5 hours
- **GTX 1660/RTX 2060**: 2.5-4 hours

### Step 6: ONNX Export for WASM

Convert trained model to ONNX for web deployment:

```powershell
python scripts/convert_to_onnx.py --model-path models/best_kanji_model.pth
```

Convert trained model to SafeTensors format:

```powershell
python scripts/convert_to_safetensors.py --model-path models/best_kanji_model.pth --verify
```

**Parameters:**

- `--model-path`: Path to trained model (default: best_kanji_model.pth)
- `--onnx-path`: Output ONNX file (auto-generated: `kanji_model_etl9g_64x64_3036classes_tract.onnx`)
- `--output-path`: Output SafeTensors file (auto-generated: `kanji_model_etl9g_64x64_3036classes.safetensors`)
- `--image-size`: Image size used in training (default: 64)
- `--pooling-type`: Pooling layer configuration (default: adaptive_avg)
- `--target-backend`: Target inference backend (default: tract)

## Model Output Format Comparison

The training pipeline supports multiple output formats, each optimized for different deployment scenarios:

| Format          | File Extension | Size   | Security        | Loading Speed | Cross-Platform | Metadata | Best For                |
| --------------- | -------------- | ------ | --------------- | ------------- | -------------- | -------- | ----------------------- |
| **PyTorch**     | `.pth`         | ~15 MB | ⚠️ Pickle-based | Good          | Python only    | Limited  | Development/Training    |
| **ONNX**        | `.onnx`        | ~15 MB | ✅ Safe         | Excellent     | Universal      | Good     | **Web/WASM Deployment** |
| **SafeTensors** | `.safetensors` | ~15 MB | ✅ Secure       | Excellent     | Universal      | Rich     | Production/Security     |

**Note**: Enhanced model (v2.1) with 5 layers + channel attention has increased from ~6.6MB to ~15MB while maintaining web-deployment viability.

### Format Details:

#### 🌐 **ONNX (.onnx) - Recommended for Web**

- **Purpose**: Universal inference standard, perfect for WASM
- **Backends**: Tract, ORT-Tract, ONNX Runtime
- **Advantages**: Wide ecosystem support, optimized for inference
- **File**: `kanji_model_etl9g_64x64_3036classes_tract.onnx`

#### 🔒 **SafeTensors (.safetensors) - Secure Alternative**

- **Purpose**: Secure, fast model weight storage
- **Advantages**: No arbitrary code execution, memory-mapped loading
- **Metadata**: Embedded training info, architecture details
- **File**: `kanji_model_etl9g_64x64_3036classes.safetensors`

#### 🐍 **PyTorch (.pth) - Development Only**

- **Purpose**: Native PyTorch format for training/research
- **Limitations**: Python-only, pickle security concerns
- **Use Case**: Model checkpoints, development workflow
- **File**: `best_kanji_model.pth`

### Deployment Recommendations:

- **🌐 Web/WASM**: Use ONNX format with Tract backend
- **🔒 Production**: Use SafeTensors for secure model distribution
- **🐍 Development**: Use PyTorch format for training and experimentation
- **📱 Mobile**: ONNX with quantization for smaller size

### Quick Format Conversion Reference:

```powershell
# ONNX for web deployment (recommended)
python scripts/convert_to_onnx.py --model-path models/best_kanji_model.pth
# → models/kanji_model_etl9g_64x64_3036classes_tract.onnx

# SafeTensors for secure deployment
python scripts/convert_to_safetensors.py --model-path models/best_kanji_model.pth --verify
# → models/kanji_model_etl9g_64x64_3036classes.safetensors

# Enhanced character mapping (all formats)
python scripts/generate_enhanced_mapping.py
# → kanji_etl9g_enhanced_mapping.json
```

**Backend Configuration Options:**

| Backend                                           | Description                  | ONNX Opset | Use Case                       | Pooling Support |
| ------------------------------------------------- | ---------------------------- | ---------- | ------------------------------ | --------------- |
| [`tract`](https://github.com/sonos/tract)         | Direct Sonos Tract (default) | 12         | Maximum performance            | All types ✅    |
| [`ort-tract`](https://ort.pyke.io/backends/tract) | ORT-Tract via ort crate      | 11         | ONNX Runtime API compatibility | Fixed only 🔷   |
| `strict`                                          | Ultra-compatible mode        | 11         | Universal compatibility        | Fixed only ⚠️   |

**Pooling Compatibility:**

- **`tract`**: Supports all pooling types including `adaptive_avg`, `adaptive_max` (GlobalAveragePool/GlobalMaxPool)
- **`ort-tract` & `strict`**: Auto-converts `adaptive_avg` → `fixed_avg`, `adaptive_max` → `fixed_max` (no GlobalAveragePool)

**Model Architecture Compatibility:**

The converter automatically uses the correct trained model architecture from `train_etl9g_model.py`:

- **Input**: 2D image tensors `(batch, 1, 64, 64)` - matches training format
- **Architecture**: Depthwise separable convolutions `1→32→64→128→256`
- **Pooling**: Backend-aware automatic conversion for maximum compatibility
- **Output**: `(batch, 3036)` classification scores

⚠️ **Important**: The converter imports the exact model architecture used during training to ensure weight compatibility.

**Export Method Selection:**

The converter automatically selects the optimal ONNX export method based on your target backend:

| Backend     | Export Method                | PyTorch Feature | Reason                                    |
| ----------- | ---------------------------- | --------------- | ----------------------------------------- |
| `tract`     | `torch.export` (dynamo=True) | Modern exporter | Better optimization for inference engines |
| `ort-tract` | TorchScript (dynamo=False)   | Legacy exporter | API compatibility with ORT                |
| `strict`    | TorchScript (dynamo=False)   | Legacy exporter | Maximum universal compatibility           |

**Automatic Fallback:** If `torch.export` fails, the converter automatically falls back to TorchScript method with status reporting.

**Backend Examples:**

```powershell
# Default: Direct Sonos Tract (recommended for performance)
python scripts/convert_to_onnx.py --model-path models/best_kanji_model.pth
# Output: models/kanji_model_etl9g_64x64_3036classes_tract.onnx

# ORT-Tract: Better integration with existing ort-based code
python scripts/convert_to_onnx.py --model-path models/best_kanji_model.pth --target-backend ort-tract
# Output: models/kanji_model_etl9g_64x64_3036classes_ort-tract.onnx

# Strict: Maximum compatibility across all inference engines
python scripts/convert_to_onnx.py --model-path models/best_kanji_model.pth --target-backend strict
# Output: models/kanji_model_etl9g_64x64_3036classes_strict.onnx
```

**Pooling Configuration Options:**

| Pooling Type   | Description                      | Output Size | Classifier Input | Model Size Impact |
| -------------- | -------------------------------- | ----------- | ---------------- | ----------------- |
| `adaptive_avg` | Global Average Pooling (default) | 1×1         | 256 features     | Smallest          |
| `adaptive_max` | Global Max Pooling               | 1×1         | 256 features     | Smallest          |
| `avg_2x2`      | 2×2 Average Pooling              | 2×2         | 1,024 features   | 4x larger         |
| `max_2x2`      | 2×2 Max Pooling                  | 2×2         | 1,024 features   | 4x larger         |
| `fixed_avg`    | Fixed 4×4 Average Pool           | 1×1         | 256 features     | Smallest          |
| `fixed_max`    | Fixed 4×4 Max Pool               | 1×1         | 256 features     | Smallest          |

**Pooling Examples:**

```powershell
# Default global average pooling (recommended)
python scripts/convert_to_onnx.py --model-path models/best_kanji_model.pth
# Output: models/kanji_model_etl9g_64x64_3036classes_tract.onnx

# Global max pooling (alternative feature aggregation)
python scripts/convert_to_onnx.py --model-path models/best_kanji_model.pth --pooling-type adaptive_max
# Output: models/kanji_model_etl9g_64x64_3036classes_tract.onnx

# 2x2 pooling (larger model, potentially better accuracy)
python scripts/convert_to_onnx.py --model-path models/best_kanji_model.pth --pooling-type avg_2x2
# Output: kanji_model_etl9g_64x64_3036classes_tract.onnx
```

**Note:** ONNX/SafeTensors conversion is done separately after training - you can train once and export multiple times with different settings.

## Output Files

### Training Outputs

- `models/best_kanji_model.pth`: Best PyTorch model checkpoint
- `models/training_progress.json`: Training metrics and curves
- `models/best_model_info.json`: Best model metadata

### WASM Deployment Files

- `models/kanji_model_etl9g_64x64_3036classes_tract.onnx`: Optimized ONNX model (6.6 MB)
- `models/kanji_model_etl9g_64x64_3036classes_tract_mapping.json`: Class-to-character mappings
- `models/kanji_etl9g_enhanced_mapping.json`: Enhanced character mapping with Unicode characters and stroke counts
- `models/kanji_model_etl9g_64x64_3036classes.safetensors`: SafeTensors model format (6.6 MB)

## Memory Management

The training pipeline is optimized for memory efficiency:

### Data Preparation

- **Chunked Processing**: Large dataset split into manageable chunks
- **Multiprocessing**: Parallel file processing with memory limits
- **Compressed Storage**: NumPy compressed format for disk efficiency

### Training

- **Progressive Loading**: Chunks loaded as needed
- **Memory Monitoring**: Automatic memory usage tracking
- **Batch Processing**: Configurable batch sizes for different hardware

## Model Architecture

### Enhanced LightweightKanjiNet Features

- **Depthwise Separable Convolutions**: Efficient feature extraction with MobileNet-style blocks
- **SENet-Style Channel Attention**: Adaptive feature recalibration for better stroke discrimination
- **5-Layer CNN Architecture**: Enhanced capacity with progressive channel expansion
- **Global Average Pooling**: Reduces parameters vs. large FC layers
- **Progressive Feature Maps**: 1→32→64→128→256→512 channels (enhanced capacity)
- **Dropout Regularization**: Prevents overfitting with 3,036 classes

### Architecture Enhancements (v2.0)

#### **Core Improvements:**

- **Added 5th Convolutional Layer**: 256→512 channels for deeper feature learning
- **Channel Attention Modules**: 3 SENet-style attention layers for adaptive feature weighting
- **Enhanced Classifier**: 512→1024→3036 neurons for improved pattern recognition

#### **Channel Attention (SENet Integration):**

```
Attention Flow: Input → Global Pool → FC → ReLU → FC → Sigmoid → Scale
Applied after: conv3 (128ch), conv4 (256ch), conv5 (512ch)
Purpose: Focus on discriminative kanji stroke patterns, suppress noise
```

#### **Architecture Comparison:**

| Component             | Original Model  | **Enhanced Model (v2.0)** |
| --------------------- | --------------- | ------------------------- |
| **Conv Layers**       | 4 layers        | **5 layers**              |
| **Channels**          | 1→32→64→128→256 | **1→32→64→128→256→512**   |
| **Attention**         | None            | **3 SENet modules**       |
| **Classifier**        | 256→512→3036    | **512→1024→3036**         |
| **Parameters**        | ~1.7M           | **~3.9M**                 |
| **Expected Accuracy** | 85%             | **90-92%**                |

### Model Specifications

- **Input**: 64×64 grayscale images (flattened to 4,096 values)
- **Output**: 3,036 class probabilities
- **Parameters**: ~3.9M (enhanced capacity for complex kanji patterns)
- **Model Size**: ~15 MB (ONNX format) - still web-deployment friendly
- **Architecture**: 5-layer CNN with channel attention for optimal kanji recognition

## Training Monitoring

### Progress Tracking

- **Real-time Metrics**: Loss, accuracy, learning rate per epoch
- **Best Model Saving**: Automatic checkpoint of best validation accuracy
- **Early Stopping**: Prevents overfitting with patience mechanism
- **Progress Logs**: JSON files for analysis and visualization

### Training Features

- **Label Smoothing**: Better generalization for large class sets
- **Cosine Annealing**: Optimal learning rate scheduling
- **Data Augmentation**: Noise injection for robustness
- **Gradient Clipping**: Training stability

## Model Optimization History

### Architecture Evolution

The model has undergone significant optimizations to achieve state-of-the-art accuracy for kanji recognition:

#### **Version 1.0 (Original)**

- **Architecture**: 4-layer CNN with depthwise separable convolutions
- **Channels**: 1→32→64→128→256
- **Parameters**: ~1.7M
- **Accuracy**: 85% validation
- **Focus**: Lightweight design for web deployment

#### **Version 2.0 (Enhanced Capacity)**

- **Added**: 5th convolutional layer (256→512 channels)
- **Enhanced**: Larger classifier (512→1024→3036)
- **Parameters**: ~3.8M
- **Expected**: +3-5% accuracy improvement
- **Focus**: Increased model capacity for complex patterns

#### **Version 2.1 (With Attention)**

- **Added**: SENet-style channel attention modules
- **Attention Points**: After conv3, conv4, conv5
- **Parameters**: ~3.9M
- **Expected**: **90-92% validation accuracy**
- **Focus**: Adaptive feature weighting for optimal kanji discrimination

### Accuracy Improvement Strategies

#### **Implemented Optimizations:**

1. **Enhanced Architecture**
   - 5th convolutional layer for deeper feature learning
   - Expanded classifier capacity (1024 hidden neurons)
   - **Impact**: +3-5% accuracy

2. **Channel Attention (SENet-Style)**
   - Adaptive feature recalibration per channel
   - Focus on discriminative stroke patterns
   - Noise suppression for cleaner representations
   - **Impact**: +2-3% accuracy

3. **Advanced Training Techniques**
   - Label smoothing for better generalization
   - Cosine annealing with warmup for optimal convergence
   - Progressive training strategy for large class sets
   - **Impact**: Stable training and convergence

#### **Potential Future Improvements:**

1. **Data Augmentation Enhancement**
   - Geometric transformations (rotation, perspective)
   - Multi-scale training for robustness
   - **Potential**: +2-4% accuracy

2. **Ensemble Methods**
   - Multiple model architectures
   - Voting mechanisms for final predictions
   - **Potential**: +3-6% accuracy

3. **Advanced Attention Mechanisms**
   - Spatial attention modules
   - Self-attention for sequence modeling
   - **Potential**: +1-3% accuracy

## Model File Naming Convention

All generated model files follow a consistent naming pattern that includes:

- **Dataset**: `etl9g` (always included)
- **Image Size**: `64x64` (training resolution)
- **Classes**: `3036classes` (total number of character classes)
- **Backend**: `tract`, `ort-tract`, or `strict` (for ONNX models)
- **Format**: `.onnx`, `.safetensors`, `.json` (file extension indicates format)

### Examples:

| File                                                     | Purpose                                 |
| -------------------------------------------------------- | --------------------------------------- |
| `kanji_model_etl9g_64x64_3036classes_tract.onnx`         | ONNX model for Tract backend            |
| `kanji_model_etl9g_64x64_3036classes.safetensors`        | SafeTensors model format                |
| `kanji_etl9g_enhanced_mapping.json`                      | Enhanced character mapping with Unicode |
| `kanji_model_etl9g_64x64_3036classes_tract_mapping.json` | Basic class-to-JIS mapping              |

This naming ensures you can easily identify model configurations and avoid confusion during deployment.

## Integration with WASM

### Model Export

The trained model is exported to ONNX format optimized for web deployment:

```python
# Automatic ONNX export with consistent naming
torch.onnx.export(model, dummy_input, 'kanji_model_etl9g_64x64_3036classes_tract.onnx',
                 opset_version=12, optimize=True)
```

### Class Mapping Integration

The enhanced character mapping provides comprehensive character information:

```json
{
  "model_info": {
    "dataset": "ETL9G",
    "total_classes": 3036,
    "description": "Enhanced character mapping with Unicode characters and stroke counts"
  },
  "characters": {
    "0": {
      "character": "あ",
      "jis_code": "2422",
      "stroke_count": 3
    },
    "201": {
      "character": "田",
      "jis_code": "4544",
      "stroke_count": 5
    }
  },
  "statistics": {
    "total_characters": 3036,
    "hiragana_count": 71,
    "kanji_count": 2965,
    "average_stroke_count": 10.3
  }
}
```

## UV Package Management

This project uses [uv](https://docs.astral.sh/uv/) for fast and reliable Python package management. Here are common uv commands:

### Environment Management

```powershell
# Create project environment with dependencies
uv sync

# Add new dependency
uv add package-name

# Add development dependency
uv add --dev pytest

# Remove dependency
uv remove package-name

# Update all dependencies
uv lock --upgrade

# Activate virtual environment
.venv\Scripts\Activate.ps1
```

### Running Scripts

```powershell
# Run script without activating environment
uv run python scripts/train_etl9g_model.py --help

# Run with project scripts (defined in pyproject.toml)
uv run train-kanji --help
uv run prepare-dataset --help
uv run convert-to-onnx --help
uv run convert-to-safetensors --help
```

### PyTorch CUDA Installation

Due to PyTorch's special CUDA requirements, install with the configured index:

```powershell
# Install PyTorch with CUDA support
uv add torch torchvision --index pytorch

# Or manually specify the index URL
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

## Troubleshooting

### Memory Issues

```powershell
# Use smaller batch size (with uv)
uv run python scripts/train_etl9g_model.py --batch-size 32

# Limit training samples for testing (with uv)
uv run python scripts/train_etl9g_model.py --sample-limit 50000

# Use fewer worker processes (with uv)
uv run python scripts/prepare_etl9g_dataset.py --workers 2

# Or with activated virtual environment
python scripts/train_etl9g_model.py --batch-size 32
python scripts/train_etl9g_model.py --sample-limit 50000
python scripts/prepare_etl9g_dataset.py --workers 2
```

### Training Issues

- **Low Accuracy**: Increase epochs, check learning rate
- **Overfitting**: Reduce model size, increase dropout
- **Slow Training**: Use GPU, increase batch size
- **Memory Errors**: Reduce batch size, use sample limiting

### ONNX Export Issues

- **Large Model Size**: Check model architecture, use quantization
- **Compatibility**: Ensure opset_version=12 for web support
- **Runtime Errors**: Verify input/output tensor shapes

### Backend Compatibility Issues

- **`unsupported op_type GlobalAveragePool` (ORT-Tract)**:
  - ✅ **Fixed**: Converter automatically uses `--target-backend ort-tract` to replace with compatible `AveragePool`
  - The converter detects ORT-Tract backend and overrides `adaptive_avg` → `fixed_avg` pooling
- **Model Architecture Mismatch**:
  - ✅ **Fixed**: Converter now imports the correct architecture from `train_etl9g_model.py`
  - Ensures trained weights are compatible with ONNX export model

## Scripts Reference

The project includes several Python scripts organized in the `scripts/` folder. Each script serves a specific purpose in the kanji recognition training and deployment pipeline.

### Core Training Scripts

#### 1. `preflight_check.py` - System Verification

Verifies your system setup and requirements before training.

```powershell
python scripts/preflight_check.py
```

**Purpose**: Ensures all dependencies are installed and the system is ready for training.

**Checks**:
- Python packages availability
- ETL9G data files presence
- System resources (RAM, GPU)
- CUDA compatibility

#### 2. `prepare_etl9g_dataset.py` - Dataset Preparation

Converts ETL9G binary files to training-ready format.

```powershell
# Basic usage
python scripts/prepare_etl9g_dataset.py --etl-dir ETL9G --output-dir dataset --size 64

# With custom workers
python scripts/prepare_etl9g_dataset.py --etl-dir ETL9G --output-dir dataset --size 64 --workers 4
```

**Parameters**:
- `--etl-dir`: Directory containing ETL9G files
- `--output-dir`: Output directory for processed dataset  
- `--size`: Target image size (default: 64x64 pixels)
- `--workers`: Number of parallel workers (default: auto)

#### 3. `test_etl9g_setup.py` - Setup Verification

Tests the prepared dataset and model architecture.

```powershell
python scripts/test_etl9g_setup.py --data-dir dataset --test-model
```

**Purpose**: Validates dataset preparation and model architecture before full training.

**Outputs**:
- Dataset statistics and sample visualization
- Model architecture verification
- Character mapping analysis
- Estimated model size

#### 4. `train_etl9g_model.py` - Model Training

Main training script for the kanji recognition model.

```powershell
# Quick test run
python scripts/train_etl9g_model.py --data-dir dataset --epochs 2 --batch-size 32 --sample-limit 15000

# Full training
python scripts/train_etl9g_model.py --data-dir dataset --epochs 30 --batch-size 64

# With custom learning rate
python scripts/train_etl9g_model.py --data-dir dataset --epochs 30 --batch-size 64 --learning-rate 0.001
```

**Key Parameters**:
- `--data-dir`: Directory containing prepared dataset
- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Training batch size (default: 64)
- `--learning-rate`: Learning rate (default: 0.001)
- `--sample-limit`: Limit samples for testing (optional)

### Model Export Scripts

#### 5. `convert_to_onnx.py` - ONNX Model Export

Converts trained PyTorch model to ONNX format for web deployment.

```powershell
# Default tract backend
python scripts/convert_to_onnx.py --model-path best_kanji_model.pth

# Different backends
python scripts/convert_to_onnx.py --model-path best_kanji_model.pth --target-backend ort-tract
python scripts/convert_to_onnx.py --model-path best_kanji_model.pth --target-backend strict

# Custom pooling
python scripts/convert_to_onnx.py --model-path best_kanji_model.pth --pooling-type adaptive_max
```

**Parameters**:
- `--model-path`: Path to trained model (default: best_kanji_model.pth)
- `--target-backend`: Backend (tract, ort-tract, strict)
- `--pooling-type`: Pooling configuration (adaptive_avg, adaptive_max, etc.)
- `--image-size`: Image size used in training (default: 64)

#### 6. `convert_to_safetensors.py` - SafeTensors Export

Converts PyTorch model to secure SafeTensors format.

```powershell
# Basic conversion
python scripts/convert_to_safetensors.py --model-path best_kanji_model.pth

# With verification
python scripts/convert_to_safetensors.py --model-path best_kanji_model.pth --verify

# Custom output path
python scripts/convert_to_safetensors.py --model-path best_kanji_model.pth --output-path custom_model.safetensors
```

**Parameters**:
- `--model-path`: Path to trained model
- `--output-path`: Output SafeTensors file (auto-generated if not specified)
- `--verify`: Verify the converted model
- `--no-metadata`: Skip adding metadata

### Utility Scripts

#### 7. `generate_enhanced_mapping.py` - Character Mapping Enhancement

Creates enhanced character mapping with Unicode characters and stroke counts.

```powershell
python scripts/generate_enhanced_mapping.py
```

**Purpose**: Generates comprehensive character mappings for web deployment.

**Outputs**:
- Enhanced mapping with Unicode characters
- Stroke count information
- Character statistics

#### 8. `inspect_onnx_model.py` - ONNX Model Inspector

Inspects ONNX models to verify operations and compatibility.

```powershell
python scripts/inspect_onnx_model.py
```

**Purpose**: Analyzes ONNX models for backend compatibility and operations used.

**Features**:
- Lists all operations in the model
- Identifies problematic operations
- Shows pooling operation details
- Checks backend compatibility

#### 9. `measure_co2_emissions.py` - Carbon Footprint Measurement

Measures CO2 emissions during training (requires CodeCarbon).

```powershell
python scripts/measure_co2_emissions.py
```

**Purpose**: Environmental impact assessment of training process.

**Requirements**: CodeCarbon package installed

### Analysis and Comparison Scripts

#### 10. `onnx_operations_comparison.py` - Operations Analysis

Compares operations between different ONNX model variants.

```powershell
python scripts/onnx_operations_comparison.py
```

**Purpose**: Analyzes differences between tract, ort-tract, and strict backend models.

#### 11. `pooling_comparison.py` - Pooling Strategy Comparison

Compares different pooling strategies and their impact on model size.

```powershell
python scripts/pooling_comparison.py
```

**Purpose**: Helps choose optimal pooling configuration for deployment.

### Example and Integration Scripts

#### 12. `example_safetensors_usage.py` - SafeTensors Usage Example

Demonstrates how to load and use SafeTensors models.

```powershell
python scripts/example_safetensors_usage.py
```

**Purpose**: Example code for integrating SafeTensors models in applications.

#### 13. `codecarbon_integration_guide.py` - CodeCarbon Integration

Guide and example for integrating CodeCarbon emission tracking.

```powershell
python scripts/codecarbon_integration_guide.py
```

**Purpose**: Demonstrates environmental impact tracking setup.

## File Structure

```
training/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT license
├── venv/                       # Virtual environment (created by setup)
├── scripts/                    # Python scripts for training and deployment
│   ├── preflight_check.py          # System verification
│   ├── prepare_etl9g_dataset.py    # Data preparation
│   ├── test_etl9g_setup.py         # Setup testing
│   ├── train_etl9g_model.py        # Main training script
│   ├── convert_to_onnx.py          # ONNX export script
│   ├── convert_to_safetensors.py   # SafeTensors export script
│   ├── generate_enhanced_mapping.py # Enhanced character mapping
│   ├── inspect_onnx_model.py       # ONNX model inspector
│   ├── measure_co2_emissions.py    # Carbon footprint measurement
│   ├── onnx_operations_comparison.py # Operations analysis
│   ├── pooling_comparison.py       # Pooling strategy comparison
│   ├── example_safetensors_usage.py # SafeTensors usage example
│   └── codecarbon_integration_guide.py # CodeCarbon integration
├── ETL9G/                     # Raw dataset (user provided)
│   ├── ETL9G_01
│   ├── ETL9G_02
│   ├── ...
│   ├── ETL9G_50
│   └── ETL9INFO
└── dataset/                   # Processed dataset (generated)
    ├── etl9g_dataset_chunk_*.npz
    ├── metadata.json
    ├── character_mapping.json
    └── chunk_info.json
```

## Virtual Environment Management

### Daily Usage

```powershell
# Always activate before training (in PowerShell)
.\venv\Scripts\Activate.ps1

# Verify environment is active (you should see "(venv)" in prompt)
python preflight_check.py

# Run training commands...
python scripts/train_etl9g_model.py --data-dir dataset --epochs 30

# Deactivate when done (optional)
deactivate
```

### Troubleshooting Virtual Environment

```powershell
# If activation fails or permission issues:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1

# If environment is corrupted, remove it:
Remove-Item -Recurse -Force venv
# ... and then do the installation again from the top of the document
```

## Expected Results

### Enhanced Training Performance (v2.0)

- **Validation Accuracy**: **90-92%** (improved with channel attention)
- **Top-3 Accuracy**: **96-99%** (enhanced feature discrimination)
- **Training Stability**: Consistent convergence with attention-guided learning
- **Model Size**: ~15 MB (ONNX) - still web-deployment friendly

### Performance Improvements

#### **Accuracy Enhancements:**

- **Original Model**: 85% validation accuracy
- **With 5th Conv Layer**: +3-5% improvement (88-90%)
- **With Channel Attention**: +2-3% additional improvement (**90-92%**)
- **Total Improvement**: **+5-7% accuracy gain**

#### **Feature Learning Benefits:**

- **Better Stroke Discrimination**: Attention focuses on critical kanji features
- **Noise Suppression**: Reduced emphasis on irrelevant background patterns
- **Complex Pattern Recognition**: Enhanced capability for intricate character details
- **Adaptive Feature Weighting**: SENet-style recalibration for optimal representation

### Character Coverage

- ✅ Rice field kanji (田) - JIS 4544
- ✅ All JIS Level 1 kanji (2,965 characters)
- ✅ Complete Hiragana set (71 characters)
- ✅ Proper class indexing (0-3035)

### Web Deployment Ready

- ✅ ONNX model optimized for WASM
- ✅ Enhanced architecture with attention (still efficient for web)
- ✅ Direct class-to-JIS mapping
- ✅ Compatible with existing Rust/WASM code

## Next Steps

After successful training:

1. **Copy Model Files**: Move ONNX/SafeTensors files to your web assets:
   - `kanji_model_etl9g_64x64_3036classes_tract.onnx` → Main inference model
   - `kanji_etl9g_enhanced_mapping.json` → Character mappings with Unicode
2. **Update Rust Code**: Integrate enhanced character mappings
3. **Test Integration**: Verify model works with your WASM interface
4. **Performance Tuning**: Optimize inference speed if needed

**Training Workflow:**

- Train: `python scripts/train_etl9g_model.py --data-dir dataset --epochs 30`
- Export ONNX: `python scripts/convert_to_onnx.py --model-path best_kanji_model.pth`
- Export SafeTensors: `python scripts/convert_to_safetensors.py --model-path best_kanji_model.pth --verify`

## Support

For issues or questions:

- Check the troubleshooting section above
- Review training logs in `training_progress.json`
- Verify system requirements with `preflight_check.py`
- Test with sample data using `--sample-limit` parameter

## Model file size considerations

Great question! The image size parameter has a **significant multiplicative effect** on model size, particularly for the input layer. Here's the breakdown:

## Current Configuration (64×64 pixels)

Looking at the training command:

```powershell
python scripts/prepare_etl9g_dataset.py --etl-dir ETL9G --output-dir dataset --size 64
```

With 64×64 input images, the **first layer parameters** are:

- **Input size**: 64×64 = 4,096 pixels (flattened)
- **First conv layer**: 4,096 → 32 channels
- **Parameters**: 4,096 × 32 = **131,072 parameters**

## Impact of Different Image Sizes

| Image Size  | Input Pixels | First Layer Params | Relative Size | Use Case        |
| ----------- | ------------ | ------------------ | ------------- | --------------- |
| **32×32**   | 1,024        | 32,768             | **0.25x**     | Mobile/embedded |
| **48×48**   | 2,304        | 73,728             | **0.56x**     | Balanced        |
| **64×64**   | 4,096        | 131,072            | **1.0x**      | **Current**     |
| **96×96**   | 9,216        | 294,912            | **2.25x**     | High detail     |
| **128×128** | 16,384       | 524,288            | **4.0x**      | Maximum detail  |

## Model Size Estimates

For your LightweightKanjiNet architecture:

- **32×32**: ~2-4 MB ONNX model
- **48×48**: ~3-6 MB ONNX model
- **64×64**: ~5-10 MB ONNX model (**current**)
- **96×96**: ~12-20 MB ONNX model
- **128×128**: ~20-35 MB ONNX model

## Kanji Recognition Considerations

**For Japanese kanji specifically:**

- **32×32**: May lose stroke detail in complex kanji
- **48×48**: Good compromise for simpler kanji
- **64×64**: **Optimal for kanji** - preserves stroke details
- **96×96+**: Overkill for most kanji, larger file size

## WASM Deployment Impact

For web deployment, consider:

- **Network**: Larger models = longer download times
- **Memory**: Browser memory usage scales with model size
- **Performance**: Larger input tensors = slower inference

## Recommendation

Your **64×64 choice is excellent** for kanji recognition because:

1. **Stroke Preservation**: Maintains fine details needed for complex kanji like 田
2. **Web-Friendly**: 5-10 MB is acceptable for web deployment
3. **Performance**: Good balance of accuracy vs. speed
4. **ETL9G Match**: Works well with ETL9G's native resolution (128×127 → 64×64)

If you need a smaller model, **48×48** would be the next best option, reducing model size by ~50% while maintaining reasonable kanji detail.

## Training results

Here's a comprehensive comparison table of the training results from both runs:

## Training Comparison Table

### First Training Run (Limited Dataset - 15,000 samples)

| Epoch | Learning Rate | Train Loss | Train Acc | Val Loss | Val Acc | Best Model |
| ----- | ------------- | ---------- | --------- | -------- | ------- | ---------- |
| 1/2   | 0.001000      | 8.0224     | 0.05%     | 8.0200   | 0.00%   | -          |
| 2/2   | 0.000501      | 7.9967     | 0.10%     | 8.0216   | 0.00%   | -          |

**Final Results:** Best Val Acc: 0.00%, Test Acc: 0.00%

---

### Second Training Run (Full Dataset - 607,200 samples)

| Epoch | Learning Rate | Train Loss | Train Acc | Val Loss | Val Acc | Best Model |
| ----- | ------------- | ---------- | --------- | -------- | ------- | ---------- |
| 1/30  | 0.001000      | 8.0215     | 0.02%     | 8.0185   | 0.03%   | ✅ 0.03%   |
| 2/30  | 0.000997      | 7.9559     | 0.04%     | 7.7351   | 0.08%   | ✅ 0.08%   |
| 3/30  | 0.000989      | 7.3906     | 0.36%     | 6.6943   | 1.89%   | ✅ 1.89%   |
| 4/30  | 0.000976      | 6.8397     | 1.45%     | 6.2111   | 5.27%   | ✅ 5.27%   |
| 5/30  | 0.000957      | 6.5033     | 2.89%     | 5.7594   | 10.76%  | ✅ 10.76%  |
| 6/30  | 0.000933      | 6.2513     | 4.67%     | 5.5149   | 14.44%  | ✅ 14.44%  |
| 7/30  | 0.000905      | 6.0274     | 6.75%     | 5.1806   | 22.99%  | ✅ 22.99%  |
| 8/30  | 0.000872      | 5.8233     | 8.96%     | 4.8874   | 30.36%  | ✅ 30.36%  |
| 9/30  | 0.000835      | 5.5402     | 11.92%    | 4.5525   | 37.95%  | ✅ 37.95%  |
| 10/30 | 0.000794      | 5.2813     | 15.26%    | 4.2635   | 44.17%  | ✅ 44.17%  |
| 11/30 | 0.000750      | 5.0153     | 18.96%    | 3.9647   | 54.06%  | ✅ 54.06%  |
| 12/30 | 0.000704      | 4.8358     | 22.43%    | 3.8065   | 59.20%  | ✅ 59.20%  |
| 13/30 | 0.000655      | 4.7270     | 24.74%    | 3.6419   | 64.56%  | ✅ 64.56%  |
| 14/30 | 0.000604      | 4.6278     | 26.91%    | 3.6153   | 63.19%  | -          |
| 15/30 | 0.000553      | 4.4868     | 29.66%    | 3.3444   | 70.81%  | ✅ 70.81%  |
| 16/30 | 0.000501      | 4.3064     | 33.21%    | 3.1876   | 74.09%  | ✅ 74.09%  |
| 17/30 | 0.000448      | 4.1558     | 36.59%    | 3.0477   | 76.25%  | ✅ 76.25%  |
| 18/30 | 0.000397      | 4.0218     | 39.54%    | 2.9668   | 78.20%  | ✅ 78.20%  |
| 19/30 | 0.000346      | 3.9320     | 41.74%    | 2.8928   | 79.30%  | ✅ 79.30%  |
| 20/30 | 0.000297      | 3.8449     | 43.87%    | 2.8151   | 80.61%  | ✅ 80.61%  |
| 21/30 | 0.000251      | 3.7698     | 45.73%    | 2.7582   | 81.84%  | ✅ 81.84%  |
| 22/30 | 0.000207      | 3.7035     | 47.46%    | 2.7052   | 82.74%  | ✅ 82.74%  |
| 23/30 | 0.000166      | 3.6515     | 48.77%    | 2.6653   | 83.59%  | ✅ 83.59%  |
| 24/30 | 0.000129      | 3.6088     | 49.91%    | 2.6414   | 83.84%  | ✅ 83.84%  |
| 25/30 | 0.000096      | 3.5740     | 50.87%    | 2.6202   | 84.28%  | ✅ 84.28%  |
| 26/30 | 0.000068      | 3.5547     | 51.41%    | 2.6083   | 84.66%  | ✅ 84.66%  |
| 27/30 | 0.000044      | 3.5393     | 51.84%    | 2.6001   | 84.77%  | ✅ 84.77%  |
| 28/30 | 0.000025      | 3.5297     | 52.20%    | 2.5918   | 84.95%  | ✅ 84.95%  |
| 29/30 | 0.000012      | 3.5248     | 52.31%    | 2.5916   | 84.87%  | -          |
| 30/30 | 0.000004      | 3.5219     | 52.32%    | 2.5937   | 84.89%  | -          |

**Final Results:** Best Val Acc: **84.95%**, Test Acc: **85.11%** ✅

---

## Key Insights

### 📊 **Performance Comparison**

- **Limited dataset (15K samples)**: Complete failure (0% accuracy)
- **Full dataset (607K samples)**: Excellent success (85.11% accuracy)

### 📈 **Learning Progression**

- **Epochs 1-10**: Rapid initial learning (0.03% → 44.17%)
- **Epochs 11-20**: Strong improvement (54.06% → 80.61%)
- **Epochs 21-30**: Fine-tuning convergence (81.84% → 84.95%)

### 🎯 **Training Quality Indicators**

- **Consistent improvement** for first 28 epochs
- **Healthy train/val gap** (~32% gap, indicating good generalization)
- **No overfitting** (validation accuracy kept improving)
- **Smooth cosine annealing** (learning rate: 0.001 → 0.000004)

This demonstrates that **data quantity is critical** for deep learning success, especially with large classification problems like kanji recognition! 🎉

### 📁 **Generated Files (Ready for WASM Deployment):**

| File                                                     | Size   | Purpose                                   |
| -------------------------------------------------------- | ------ | ----------------------------------------- |
| **`kanji_model_etl9g_64x64_3036classes_tract.onnx`**     | 6.6 MB | 🎯 **Main ONNX model for web deployment** |
| **`kanji_etl9g_enhanced_mapping.json`**                  | 0.3 MB | 🗾 **Enhanced character mappings**        |
| **`kanji_model_etl9g_64x64_3036classes.safetensors`**    | 6.6 MB | 🔒 **Secure SafeTensors format**          |
| `kanji_model_etl9g_64x64_3036classes_tract_mapping.json` | 0.5 MB | Basic class-to-JIS mappings               |
| `best_kanji_model.pth`                                   | 6.6 MB | PyTorch checkpoint (training backup)      |
| `best_model_info.json`                                   | < 1 KB | Training metadata                         |
| `training_progress.json`                                 | 4 KB   | Complete training history                 |

### 🏆 **Final Model Performance:**

- **Test Accuracy**: **85.11%** ✅
- **Validation Accuracy**: **84.95%**
- **Model Parameters**: 1,735,527 (lightweight!)
- **Classes Covered**: 3,036 (complete ETL9G)
- **Training Time**: ~30 epochs on full dataset

## Development

Python source code is formatted with [`ruff`](https://github.com/astral-sh/ruff):

```powershell
ruff format
```

## License

MIT, see [LICENSE](./LICENSE)
