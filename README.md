# ETL9G Kanji Recognition Training

This project trains multiple neural network architectures for Japanese kanji character recognition using the ETL9G dataset (3,036 classes, 607,200 samples).

**🎯 Status**: Comprehensive multi-architecture research platform with 97.18% CNN accuracy and 98.4% RNN accuracy.

## 📚 Documentation

**Start here**: [**PROJECT_DIARY.md**](PROJECT_DIARY.md) - Complete project history, all training approaches, results, and research references.

### What's in PROJECT_DIARY.md:
- 📖 Project origin story (how we evolved from CNN-only to multi-architecture platform)
- 🎯 5 training approaches (CNN, RNN, QAT, HierCode, ViT) with results
- 📊 Performance metrics and comparisons
- 🔄 Checkpoint/Resume system for crash recovery
- 📚 Complete arXiv paper references and GitHub project links
- 🚀 Next steps and expansion plans (multi-dataset, optimization strategies)
- 🏆 Key learnings and technical insights

## Quick Commands

### Setup

```powershell
# Install dependencies
pip install -r requirements.txt

# Verify environment
python scripts/preflight_check.py
```

### Training

```powershell
# CNN Training (baseline)
python scripts/train_etl9g_model.py --data-dir dataset

# RNN Training (radical-based, higher accuracy)
python scripts/train_radical_rnn.py --data-dir dataset

# QAT Training (quantization-aware for deployment)
python scripts/train_qat.py --data-dir dataset

# Resume training with checkpoints
python scripts/train_qat.py --data-dir dataset --resume-from models/checkpoints/checkpoint_epoch_004.pt
```

### Evaluation & Export

```powershell
# Evaluate model
python scripts/train_etl9g_model.py --data-dir dataset --evaluate

# Export to ONNX
python scripts/convert_to_onnx.py --model-path models/best_kanji_model.pth

# Export to SafeTensors
python scripts/convert_to_safetensors.py --model-path models/best_kanji_model.pth
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

## 📊 Key Results

| Model | Accuracy | Size | Speed | Status |
|-------|----------|------|-------|--------|
| CNN | 97.18% | 6.6 MB | Fast | ✅ Production |
| RNN (Radical) | 98.4% | ~23 MB | Medium | ✅ Trained |
| QAT | 96.5-97% | 1.7 MB | 2-4x | �� In Progress |
| ViT | TBD | TBD | TBD | 📋 Planned |
| HierCode | TBD | TBD | TBD | 📋 Planned |

## 🔗 Key Links

- **arXiv Papers**: See PROJECT_DIARY.md § Research References
- **GitHub Projects**: See PROJECT_DIARY.md § GitHub Projects
- **ETL9G Dataset**: http://etlcdb.db.aist.go.jp/download-links/
- **Model Card**: [model-card.md](model-card.md)
- **Research Notes**: [RESEARCH.md](RESEARCH.md)

## ❓ Common Questions

**Q: Where do I find all the project documentation?**  
A: [PROJECT_DIARY.md](PROJECT_DIARY.md) - Single source of truth for all project info.

**Q: How do I train the model?**  
A: See Quick Commands above, or detailed setup in PROJECT_DIARY.md § Project Phases.

**Q: What should I do if training crashes?**  
A: Use the checkpoint/resume system (added to train_qat.py). See PROJECT_DIARY.md § Checkpoint/Resume System.

**Q: How do I deploy the model?**  
A: Export to ONNX with `convert_to_onnx.py`. See Quick Commands above.

**Q: What's the best architecture to use?**  
A: CNN for speed (97.18% accuracy), RNN for accuracy (98.4%), QAT for deployment (1.7 MB). See PROJECT_DIARY.md § Key Results.

## 📝 System Requirements

- **OS**: Windows 11, Linux, or macOS
- **Python**: 3.11+ (tested with 3.13)
- **GPU**: NVIDIA GPU with CUDA 11.8+ recommended
- **RAM**: 8+ GB (16+ GB recommended)
- **Storage**: 15+ GB free space

## 🚀 Next Steps

1. **Read** [PROJECT_DIARY.md](PROJECT_DIARY.md) for complete project overview
2. **Setup** environment with `pip install -r requirements.txt`
3. **Train** model with `python scripts/train_etl9g_model.py --data-dir dataset`
4. **Export** to ONNX for deployment with `python scripts/convert_to_onnx.py`

---

**Repository**: https://github.com/paazmaya/kanji-2965-CNN-ETL9G  
**Owner**: Jukka Paazmaya (@paazmaya)  
**Last Updated**: November 15, 2025  
**License**: See LICENSE file
