# RNN-based Kanji Recognition Models

This directory contains RNN-based implementations for kanji recognition as an alternative to the CNN approach.

## Quick Start

1. **Train a basic RNN model:**
```bash
python scripts/rnn/train_rnn_model.py --data-dir dataset --model-type basic_rnn --epochs 50
```

2. **Train a stroke-based RNN model:**
```bash
python scripts/rnn/train_rnn_model.py --data-dir dataset --model-type stroke_rnn --epochs 50
```

3. **Train a hybrid CNN-RNN model:**
```bash
python scripts/rnn/train_rnn_model.py --data-dir dataset --model-type hybrid_cnn_rnn --epochs 50
```

4. **Evaluate trained models:**
```bash
python scripts/rnn/evaluate_rnn_models.py --data-dir dataset --model-dir models
```

5. **Perform inference:**
```bash
python scripts/rnn/deploy_rnn_model.py --model-path models/rnn/best_hybrid_cnn_rnn_model.pth --model-type hybrid_cnn_rnn --image path/to/kanji.png
```

## Model Types

### 1. Basic RNN (`basic_rnn`)
- Converts kanji images to spatial sequences using grid-based scanning
- Uses bidirectional LSTM with attention mechanism
- Best for: Learning sequential spatial patterns in kanji structure

### 2. Stroke-based RNN (`stroke_rnn`)
- Extracts stroke sequences using contour detection and ordering
- Processes temporal stroke information with LSTM
- Best for: Understanding stroke order and writing patterns

### 3. Radical-based RNN (`radical_rnn`)
- Decomposes kanji into radical components using structural analysis
- Processes radical sequences to understand character composition
- Best for: Leveraging semantic meaning and character structure

### 4. Hybrid CNN-RNN (`hybrid_cnn_rnn`)
- Combines spatial CNN features with temporal RNN processing
- Uses CNN backbone with LSTM head for sequence modeling
- Best for: Balanced spatial and temporal feature learning

## Architecture Benefits

### Parameter Efficiency
- **13-91% reduction** in parameters compared to pure CNN approaches
- Efficient sequence modeling reduces computational requirements
- Suitable for deployment on resource-constrained devices

### Sequential Understanding
- Natural modeling of stroke order and writing patterns
- Better handling of character variations and styles
- Improved generalization to unseen writing styles

### Structural Awareness
- Radical-based models understand semantic components
- Better handling of complex characters with multiple radicals
- Improved performance on rare or complex kanji

## Training Options

### Basic Training
```bash
python scripts/rnn/train_rnn_model.py \
    --data-dir dataset \
    --model-type basic_rnn \
    --batch-size 32 \
    --epochs 50 \
    --lr 0.001
```

### Advanced Training with Hyperparameter Tuning
```bash
python scripts/rnn/train_rnn_model.py \
    --data-dir dataset \
    --model-type hybrid_cnn_rnn \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.0005 \
    --hidden-size 256 \
    --num-layers 3 \
    --dropout 0.3 \
    --patience 10 \
    --save-dir models/rnn
```

## Evaluation and Comparison

### Single Model Evaluation
```bash
python scripts/rnn/evaluate_rnn_models.py \
    --data-dir dataset \
    --model-dir models/rnn \
    --batch-size 32 \
    --save-dir evaluation_results
```

### Model Comparison
```bash
# Compare multiple models
python scripts/rnn/deploy_rnn_model.py \
    --compare-models \
    --model-config model_comparison_config.json \
    --image test_kanji.png
```

Create `model_comparison_config.json`:
```json
[
    {
        "name": "Basic RNN",
        "path": "models/rnn/best_basic_rnn_model.pth",
        "type": "basic_rnn"
    },
    {
        "name": "Stroke RNN",
        "path": "models/rnn/best_stroke_rnn_model.pth",
        "type": "stroke_rnn"
    },
    {
        "name": "Hybrid CNN-RNN",
        "path": "models/rnn/best_hybrid_cnn_rnn_model.pth",
        "type": "hybrid_cnn_rnn"
    }
]
```

## Performance Monitoring

Training progress is saved in `training_progress.json` and includes:
- Loss curves (training and validation)
- Accuracy metrics (top-1 and top-5)
- Learning rate schedule
- Model checkpoints and best model selection

Visualize training progress:
```python
import json
import matplotlib.pyplot as plt

with open('training_progress.json', 'r') as f:
    progress = json.load(f)

plt.plot(progress['train_loss'], label='Training Loss')
plt.plot(progress['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
```

## Files Overview

- `__init__.py` - Package initialization and exports
- `rnn_model.py` - RNN model architectures (4 different types)
- `data_processor.py` - Data preprocessing for different sequence types
- `train_rnn_model.py` - Training script with full pipeline
- `evaluate_rnn_models.py` - Model evaluation and comparison tools
- `deploy_rnn_model.py` - Inference and deployment utilities
- `README.md` - This documentation

## Research Background

RNN models for kanji recognition offer several advantages:

1. **Sequential Nature**: Kanji characters have inherent stroke order and structure that RNNs can model naturally
2. **Parameter Efficiency**: Studies show 90% vocabulary reduction and 13-91% parameter reduction compared to CNNs
3. **Temporal Modeling**: Better handling of writing variations and stroke patterns
4. **Structural Understanding**: Radical-based decomposition aligns with how humans understand kanji

For detailed research background and methodology comparison, see the main project documentation.

## Troubleshooting

### Memory Issues
- Reduce batch size: `--batch-size 16`
- Use gradient accumulation: `--accumulate-grad-batches 2`
- Enable mixed precision training (requires compatible hardware)

### Slow Training
- Increase batch size if memory allows
- Use multiple workers: `--num-workers 4`
- Consider distributed training for very large datasets

### Poor Performance
- Try different model types for your specific use case
- Adjust hyperparameters (learning rate, hidden size, dropout)
- Ensure proper data preprocessing and augmentation
- Check for data quality issues

### CUDA Out of Memory
```bash
# Reduce memory usage
python scripts/rnn/train_rnn_model.py \
    --data-dir dataset \
    --model-type basic_rnn \
    --batch-size 8 \
    --accumulate-grad-batches 4
```

For additional support, refer to the main project documentation or create an issue in the repository.