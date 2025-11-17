# ETL9G Kanji Recognition Training

This project trains multiple neural network architectures for Japanese kanji character recognition using the ETL9G dataset (3,036 classes, 607,200 samples).

**🎯 Status**: ✅ Complete | 5 architectures trained | HierCode: 95.56% | RNN: 98.4% CNN: 97.18% | INT8 quantized (1.67 MB) | Production-ready deployment

## 📚 Documentation

| Document                                 | Purpose                                                                           |
| ---------------------------------------- | --------------------------------------------------------------------------------- |
| [**PROJECT_DIARY.md**](PROJECT_DIARY.md) | Complete project history, all training phases, research references, key learnings |
| [**RESEARCH.md**](RESEARCH.md)           | Research findings, architecture comparisons, citations                            |
| [**model-card.md**](model-card.md)       | HuggingFace model card with carbon footprint analysis                             |

## 🚀 Quick Start

### Setup

```powershell
# Install dependencies with uv
uv pip install -r requirements.txt

# Verify environment
uv run python scripts/preflight_check.py
```

### Training

```powershell
# CNN (fast baseline, 97.18% accuracy)
python scripts/train_etl9g_model.py --data-dir dataset

# RNN (best accuracy, 98.4%)
python scripts/train_radical_rnn.py --data-dir dataset

# HierCode (recommended, 95.56% + quantizable)
python scripts/train_hiercode.py --data-dir dataset --epochs 30 --checkpoint-dir models/checkpoints

# With checkpoint resume (crash-safe)
python scripts/train_hiercode.py --data-dir dataset --resume-from models/checkpoints/checkpoint_epoch_015.pt --epochs 30

# QAT (lightweight deployment, 1.7 MB)
python scripts/train_qat.py --data-dir dataset --checkpoint-dir models/checkpoints
```

### Deployment

```powershell
# Export to ONNX
python scripts/convert_to_onnx.py --model-path models/best_kanji_model.pth

# Export to SafeTensors
python scripts/convert_to_safetensors.py --model-path models/best_kanji_model.pth

# Quantize INT8 PyTorch to ultra-lightweight ONNX
python scripts/convert_int8_pytorch_to_quantized_onnx.py --model-path models/quantized_hiercode_int8.pth
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
│   ├── train_etl9g_model.py  ← CNN baseline
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
├── dataset/                  ← ETL9G dataset (preprocessed)
│   ├── etl9g_dataset_chunk_*.npz
│   ├── character_mapping.json
│   └── metadata.json
│
└── ETL9G/                    ← Raw ETL9G files (download separately)
    ├── ETL9G_01 - ETL9G_50
    └── ETL9GINFO
```

## 📊 Results Comparison

| Architecture           | Accuracy | Model Size    | Speed    | Format       | Deployment  | Status     |
| ---------------------- | -------- | ------------- | -------- | ------------ | ----------- | ---------- |
| **CNN**                | 97.18%   | 6.6 MB        | ⚡⚡⚡   | PyTorch      | Python/ONNX | ✅ Prod    |
| **RNN**                | 98.4%    | 23 MB         | ⚡⚡     | PyTorch      | Python/ONNX | ✅ Prod    |
| **HierCode**           | 95.56%   | 2.1 MB (INT8) | ⚡⚡⚡   | PyTorch/ONNX | Python/ONNX | ✅ Prod    |
| **HierCode INT8 ONNX** | 95.56%   | **1.67 MB**   | ⚡⚡⚡   | ONNX         | Edge/Mobile | ✅ Prod    |
| **QAT**                | 62%      | 1.7 MB        | ⚡⚡⚡⚡ | ONNX         | Embedded    | ✅ Done    |
| **ViT**                | —        | —             | —        | —            | —           | 📋 Planned |

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

**Q: How do I handle training crashes?**  
A: All scripts have checkpoint/resume system built in. Use `--resume-from checkpoint.pt` flag.

**Q: How do I deploy to edge/mobile?**  
A: Use `hiercode_int8_quantized_quantized_int8_onnx_opset14.onnx` (1.67 MB). Supports ONNX Runtime, TensorRT, CoreML, TVM.

**Q: Can I use pre-trained models?**  
A: Models in `models/` are ready to use. Load with `torch.load()` or `ort.InferenceSession()` for ONNX.

**Q: How do I add new training approaches?**  
A: See `scripts/optimization_config.py` for unified config system. Inherit from `OptimizationConfig` class.

## 📝 System Requirements

- **OS**: Windows 11, Linux, or macOS
- **Python**: 3.11+ (tested with 3.13)
- **GPU**: NVIDIA GPU with CUDA 11.8+ recommended
- **RAM**: 8+ GB (16+ GB recommended)
- **Storage**: 15+ GB free space

## 🚀 Next Steps

1. **Read** [PROJECT_DIARY.md](PROJECT_DIARY.md) for complete project overview
2. **Setup** environment with `uv pip install -r requirements.txt`
3. **Train** model with `uv run python scripts/train_etl9g_model.py --data-dir dataset`
4. **Export** to ONNX for deployment with `uv run python scripts/convert_to_onnx.py`

---

## 📋 Project Summary

**Goal**: Multi-architecture platform for Japanese kanji recognition (3,036 characters)  
**Dataset**: ETL9G (607,200 samples, 64×64 grayscale images)  
**Best Result**: RNN 98.4% accuracy | HierCode 95.56% accuracy at 1.67 MB (quantized ONNX)  
**Key Achievement**: 82% size reduction while maintaining 95.56% accuracy

**Repository**: https://github.com/paazmaya/kanji-2965-CNN-ETL9G  
**Owner**: Jukka Paazmaya (@paazmaya)  
**License**: MIT (see LICENSE file)  
**Last Updated**: November 16, 2025
