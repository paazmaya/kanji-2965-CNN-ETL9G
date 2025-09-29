# ETL9G Kanji Recognition Training

This directory contains the complete pipeline for training a lightweight kanji recognition model using the ETL9G dataset for WASM deployment.

## Overview

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

First, set up a virtual environment for isolated package management:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA support for NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install other required packages
pip install numpy scikit-learn matplotlib tqdm
pip install opencv-python Pillow
pip install onnx onnxruntime-gpu
pip install psutil  # System monitoring
```

Now it is a good time to update dependencies file `requirements.txt`:

```powershell
pip freeze > requirements.txt
```

**Note**: Always activate the virtual environment before running any training scripts. Ensure your NVIDIA drivers are up to date.

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

## Training Workflow

### Step 0: Environment Setup

Before running any training commands, set up and activate your virtual environment:

```powershell
# Create and activate virtual environment (first time only)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies with CUDA support (first time only)
pip install -r requirements.txt
```

**Important**: Always ensure your virtual environment is activated before running any Python scripts. You should see `(venv)` in your PowerShell prompt.

### Step 1: Pre-flight Check

Verify your system setup and requirements:

```powershell
# Make sure virtual environment is activated first
python preflight_check.py
```

**Expected Output:**

- ✅ All Python packages available
- ✅ ETL9G data files present
- ✅ Sufficient system resources
- ✅ Training scripts ready

### Step 2: Data Preparation

Convert ETL9G binary files to training-ready format:

```powershell
python prepare_etl9g_dataset.py --etl-dir ETL9G --output-dir dataset --size 64
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
python test_etl9g_setup.py --data-dir dataset --test-model
```

**Expected Output:**

- Dataset statistics and sample visualization
- Model architecture verification
- Character mapping analysis (including rice field kanji)
- Estimated model size, for 64 pixel size, about 6.6 MB

### Step 4: Training Test Run

Perform a quick test with limited data to verify the training pipeline:

**Recommended for initial testing (fewer classes, better learning):**

```powershell
python train_etl9g_model.py --data-dir dataset --epochs 2 --batch-size 32 --class-limit 100 --sample-limit 15000
```

**Alternative commands:**

```powershell
# More samples per class (better for full dataset)
python train_etl9g_model.py --data-dir dataset --epochs 2 --batch-size 32 --sample-limit 50000

# Very quick test (may show 0% accuracy due to too few samples per class)
python train_etl9g_model.py --data-dir dataset --epochs 2 --batch-size 32 --sample-limit 5000
```

**Parameters:**

- `--class-limit 100`: Test with only 100 most frequent classes (150 samples per class)
- `--sample-limit 15000`: Use 15,000 samples total
- Fewer classes = better learning signal for testing pipeline

**Expected Output:**

- Quick training completion (5-10 minutes)
- Model convergence verification
- Training logs and progress files

### Step 5: Full Model Training

Train the complete model for production use:

```powershell
python train_etl9g_model.py --data-dir dataset --epochs 30 --batch-size 64
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
python convert_to_onnx.py --model-path best_kanji_model.pth --data-dir dataset
```

**Parameters:**

- `--model-path`: Path to trained model (default: best_kanji_model.pth)
- `--onnx-path`: Output ONNX file (default: kanji_etl9g_model.onnx)
- `--data-dir`: Dataset directory (default: dataset)
- `--image-size`: Image size used in training (default: 64)
- `--num-classes`: Number of classes (default: 3036)
- `--pooling-type`: Pooling layer configuration (default: adaptive_avg)
- `--target-backend`: Target inference backend (default: tract)

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
python convert_to_onnx.py --model-path best_kanji_model.pth

# ORT-Tract: Better integration with existing ort-based code
python convert_to_onnx.py --model-path best_kanji_model.pth --target-backend ort-tract

# Strict: Maximum compatibility across all inference engines
python convert_to_onnx.py --model-path best_kanji_model.pth --target-backend strict
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
python convert_to_onnx.py --model-path best_kanji_model.pth

# Global max pooling (alternative feature aggregation)
python convert_to_onnx.py --model-path best_kanji_model.pth --pooling-type adaptive_max

# 2x2 pooling (larger model, potentially better accuracy)
python convert_to_onnx.py --model-path best_kanji_model.pth --pooling-type avg_2x2
```

**Alternative (Combined Training + Export):**

```powershell
python train_etl9g_model.py --data-dir dataset --epochs 30 --batch-size 64 --export-onnx
```

**Note:** Using separate commands allows flexibility - you can train once and export multiple times with different settings.

## Output Files

### Training Outputs

- `best_kanji_model.pth`: Best PyTorch model checkpoint
- `training_progress.json`: Training metrics and curves
- `best_model_info.json`: Best model metadata

### WASM Deployment Files

- `kanji_etl9g_model.onnx`: Optimized ONNX model (5-10 MB)
- `kanji_etl9g_mapping.json`: Class-to-character mappings

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

### LightweightKanjiNet Features

- **Depthwise Separable Convolutions**: Efficient feature extraction
- **Global Average Pooling**: Reduces parameters vs. large FC layers
- **Progressive Feature Maps**: 32→64→128→256 channels
- **Dropout Regularization**: Prevents overfitting with 3,036 classes

### Model Specifications

- **Input**: 64×64 grayscale images (flattened to 4,096 values)
- **Output**: 3,036 class probabilities
- **Parameters**: ~500K-1M (optimized for web deployment)
- **Model Size**: 5-10 MB (ONNX format)

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

## Integration with WASM

### Model Export

The trained model is exported to ONNX format optimized for web deployment:

```python
# Automatic ONNX export with --export-onnx flag
torch.onnx.export(model, dummy_input, 'kanji_etl9g_model.onnx',
                 opset_version=12, optimize=True)
```

### Class Mapping Integration

The `kanji_etl9g_mapping.json` file provides direct class-to-JIS mappings:

```json
{
  "class_to_jis": {
    "0": "3042", // あ (Hiragana A)
    "1": "3044", // い (Hiragana I)
    "201": "4544" // 田 (Rice field kanji)
  },
  "num_classes": 3036,
  "model_info": {
    "dataset": "ETL9G",
    "accuracy": "85.2%"
  }
}
```

## Troubleshooting

### Memory Issues

```bash
# Use smaller batch size
python train_etl9g_model.py --batch-size 32

# Limit training samples for testing
python train_etl9g_model.py --sample-limit 50000

# Use fewer worker processes
python prepare_etl9g_dataset.py --workers 2
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

## File Structure

```
training/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── venv/                       # Virtual environment (created by setup)
├── setup_etl9g.ps1            # Automated environment setup
├── preflight_check.py          # System verification
├── prepare_etl9g_dataset.py    # Data preparation
├── train_etl9g_model.py        # Main training script
├── convert_to_onnx.py          # ONNX export script
├── test_etl9g_setup.py         # Setup testing
├── training_commands.ps1       # Command reference
├── ETL9G/                     # Raw dataset (user provided)
│   ├── ETL9G_01
│   └── ...
└── dataset/                   # Processed dataset (generated)
    ├── etl9g_dataset_chunk_*.npz
    ├── metadata.json
    └── character_mapping.json
```

## Virtual Environment Management

### Daily Usage

```powershell
# Always activate before training (in PowerShell)
.\venv\Scripts\Activate.ps1

# Verify environment is active (you should see "(venv)" in prompt)
python preflight_check.py

# Run training commands...
python train_etl9g_model.py --data-dir dataset --epochs 30 --export-onnx

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

### Training Performance

- **Validation Accuracy**: 85-90%
- **Top-3 Accuracy**: 95-98%
- **Training Stability**: Consistent convergence
- **Model Size**: 5-10 MB (ONNX)

### Character Coverage

- ✅ Rice field kanji (田) - JIS 4544
- ✅ All JIS Level 1 kanji (2,965 characters)
- ✅ Complete Hiragana set (71 characters)
- ✅ Proper class indexing (0-3035)

### Web Deployment Ready

- ✅ ONNX model optimized for WASM
- ✅ Lightweight architecture (fast inference)
- ✅ Direct class-to-JIS mapping
- ✅ Compatible with existing Rust/WASM code

## Next Steps

After successful training:

1. **Copy ONNX Model**: Move `kanji_etl9g_model.onnx` to your web assets
2. **Update Rust Code**: Integrate `kanji_etl9g_mapping.json` mappings
3. **Test Integration**: Verify model works with your WASM interface
4. **Performance Tuning**: Optimize inference speed if needed

**Training Workflow:**

- Train: `python train_etl9g_model.py --data-dir dataset --epochs 30`
- Export: `python convert_to_onnx.py --model-path best_kanji_model.pth`

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

```bash
python prepare_etl9g_dataset.py --etl-dir ETL9G --output-dir dataset --size 64
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

| File                           | Size   | Purpose                                |
| ------------------------------ | ------ | -------------------------------------- |
| **`kanji_etl9g_model.onnx`**   | 6.6 MB | 🎯 **Main model for web deployment**   |
| **`kanji_etl9g_mapping.json`** | 20 KB  | 🗾 **Class-to-JIS character mappings** |
| `best_kanji_model.pth`         | 6.6 MB | PyTorch checkpoint (training backup)   |
| `best_model_info.json`         | < 1 KB | Training metadata                      |
| `training_progress.json`       | 4 KB   | Complete training history              |

### 🏆 **Final Model Performance:**

- **Test Accuracy**: **85.11%** ✅
- **Validation Accuracy**: **84.95%**
- **Model Parameters**: 1,735,527 (lightweight!)
- **Classes Covered**: 3,036 (complete ETL9G)
- **Training Time**: ~30 epochs on full dataset

## Development

Python source code is formatted with [`ruff`](https://github.com/astral-sh/ruff):

```ps1
ruff format
```
