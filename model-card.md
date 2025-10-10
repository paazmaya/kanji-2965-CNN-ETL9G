---
license: mit
language:
  - ja
tags:
  - image-classification
  - kanji-recognition
  - japanese
  - computer-vision
  - pytorch
  - etl9g
  - cnn
  - channel-attention
  - safetensors
  - onnx
datasets:
  - ETL9G
metrics:
  - accuracy
model-index:
  - name: ETL9G Kanji Recognition Model
    results:
      - task:
          type: image-classification
          name: Image Classification
        dataset:
          type: ETL9G
          name: ETL Character Database 9G
          split: test
        metrics:
          - type: accuracy
            value: 97.18
            name: Validation Accuracy
          - type: accuracy
            value: 93.51
            name: Training Accuracy
library_name: pytorch
pipeline_tag: image-classification
---

# ETL9G Kanji Recognition Model

A lightweight CNN model with SENet-style channel attention for recognizing Japanese kanji characters, trained on the ETL9G dataset.

## Model Description

This model is a **5-layer Convolutional Neural Network (CNN)** with **channel attention mechanisms** designed for high-accuracy Japanese kanji character recognition. It uses depthwise separable convolutions for efficiency while maintaining strong performance through strategic attention modules.

### Key Features

- **3,036 character classes**: Complete coverage of JIS Level 1 Kanji (2,965) + Hiragana (71)
- **Channel Attention**: 3 SENet-style attention modules for adaptive feature weighting
- **Lightweight Design**: ~15MB model size, suitable for web deployment
- **High Accuracy**: 97.18% validation accuracy
- **Multi-Format Support**: Available in PyTorch (.pth), SafeTensors, and ONNX formats
- **Cross-Platform**: Compatible with multiple inference backends

### Architecture Details

```
Input (64×64 grayscale) → Conv1 (1→32) → Conv2 (32→64) → Conv3 (64→128) + Attention
                       → Conv4 (128→256) + Attention → Conv5 (256→512) + Attention
                       → Global Average Pool → Classifier (3,036 classes)
```

**Channel Progression**: 1 → 32 → 64 → 128 → 256 → 512
**Total Parameters**: 1,735,527 (~1.7M parameters)
**Model Size**: ~14.7 MB

## Intended Use

### Primary Use Cases

- **Japanese Text Recognition**: OCR systems processing handwritten or printed kanji
- **Educational Applications**: Kanji learning and practice tools
- **Document Processing**: Automated Japanese document analysis
- **Mobile Applications**: On-device kanji recognition
- **Web Applications**: Browser-based Japanese text recognition

### Direct Use

The model accepts 64×64 pixel grayscale images of individual kanji characters and outputs classification scores across 3,036 possible characters.

### Downstream Use

This model can be integrated into:

- OCR pipelines for Japanese text
- Educational software for kanji learning
- Document digitization systems
- Handwriting recognition applications

## Training Data

### Dataset: ETL Character Database 9G (ETL9G)

- **Source**: National Institute of Advanced Industrial Science and Technology (AIST)
- **Total Samples**: 607,200 character images
- **Writers**: 4,000 different individuals
- **Character Classes**: 3,036 (2,965 JIS Level 1 Kanji + 71 Hiragana)
- **Image Format**: 128×127 pixels, 16 grayscale levels
- **Split**: 80% training, 20% validation

**Dataset URL**: http://etlcdb.db.aist.go.jp/

### Preprocessing

- **Resize**: 128×127 → 64×64 pixels
- **Normalization**: Pixel values normalized to [-1, 1] range
- **Format**: Single-channel grayscale input

## Training Procedure

### Training Hyperparameters

- **Epochs**: 30 (early stopping at epoch 27)
- **Batch Size**: 64
- **Learning Rate**: Adaptive (started at 0.001, final: 2.54e-05)
- **Optimizer**: Adam with weight decay
- **Loss Function**: CrossEntropyLoss
- **Hardware**: NVIDIA GPU with CUDA support

### Training Results

| Metric                        | Value    |
| ----------------------------- | -------- |
| **Final Validation Accuracy** | 97.18%   |
| **Final Training Accuracy**   | 93.51%   |
| **Validation Loss**           | 1.512    |
| **Training Epochs**           | 27/30    |
| **Learning Rate (final)**     | 2.54e-05 |

## Evaluation

### Testing Data, Factors & Metrics

**Testing Data**: 20% validation split from ETL9G dataset (~121,440 samples)

**Evaluation Metrics**:

- **Accuracy**: Primary metric for character classification
- **Loss**: Cross-entropy loss for training monitoring

**Performance**:

- Achieved **97.18% validation accuracy**
- Strong generalization with 3.67% accuracy gap (train vs validation)
- Consistent performance across different character types

## Model Formats

### Available Formats

1. **PyTorch (.pth)**: Native PyTorch format for fine-tuning
2. **SafeTensors**: Secure, memory-efficient format for deployment
3. **ONNX**: Cross-platform inference with multiple backend support

### ONNX Backend Compatibility

| Backend          | Support       | Model Variant  |
| ---------------- | ------------- | -------------- |
| **ONNX Runtime** | ✅ Full       | Standard ONNX  |
| **Direct Tract** | ✅ Full       | tract.onnx     |
| **ORT-Tract**    | ✅ Full       | ort-tract.onnx |
| **Strict Mode**  | ✅ Universal  | strict.onnx    |
| **WebAssembly**  | ✅ Compatible | strict.onnx    |
| **Mobile/Edge**  | ✅ Optimized  | strict.onnx    |

## Environmental Impact

- **Training Time**: ~2-3 hours on modern GPU
- **Model Size**: 6.62-14.7 MB (format dependent)
- **Inference Speed**: Optimized for real-time applications
- **Energy Efficiency**: Lightweight architecture suitable for edge deployment

## Technical Specifications

### Model Architecture

```python
class LightweightKanjiNet(nn.Module):
    # 5-layer CNN with channel attention
    # Depthwise separable convolutions
    # 3 SENet-style attention modules
    # Global average pooling
    # Final classifier layer (3,036 classes)
```

### Input Requirements

- **Format**: Single-channel (grayscale) image
- **Size**: 64×64 pixels
- **Data Type**: Float32
- **Range**: [-1.0, 1.0] (normalized)
- **Shape**: (batch_size, 1, 64, 64)

### Output

- **Format**: Logits (pre-softmax scores)
- **Shape**: (batch_size, 3036)
- **Classes**: 3,036 possible kanji/hiragana characters
- **Mapping**: Available in accompanying JSON files

## Usage Examples

### PyTorch

```python
import torch
from safetensors.torch import load_file

# Load model
model_weights = load_file("kanji_model_etl9g_64x64_3036classes.safetensors")
model.load_state_dict(model_weights)
model.eval()

# Inference
with torch.no_grad():
    output = model(input_tensor)  # shape: (batch, 3036)
    predicted_class = torch.argmax(output, dim=1)
```

### ONNX Runtime

```python
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("kanji_model_etl9g_64x64_3036classes_strict.onnx")

# Inference
outputs = session.run(None, {"input": input_array})
predictions = outputs[0]
```

## Limitations and Bias

### Known Limitations

1. **Single Character Only**: Designed for individual character recognition, not full text
2. **Fixed Input Size**: Requires 64×64 pixel input images
3. **Grayscale Only**: Does not process color images
4. **Writing Style**: Trained primarily on handwritten samples from ETL9G dataset
5. **Character Set**: Limited to JIS Level 1 Kanji + basic Hiragana

### Potential Biases

- **Writer Demographics**: Reflects the demographic distribution of ETL9G dataset writers
- **Writing Styles**: May be biased toward specific handwriting patterns in the training data
- **Character Frequency**: Some characters may be better recognized due to training data distribution

### Recommendations

- Pre-process images to match training data characteristics
- Consider ensemble methods for critical applications
- Validate performance on your specific use case
- Monitor for drift when deployed in production

## Carbon Footprint

<!-- This template follows the Hugging Face guidelines: https://huggingface.co/docs/hub/model-cards-co2 -->

### Training

- **Hardware Type**: NVIDIA GeForce RTX 4070 Ti
- **Hours used**: 2.5 hours
- **Cloud Provider**: N/A (Local training)
- **Compute Region**: Local development machine
- **Carbon Emitted**: 0.594 kg CO2 (estimated)

### Methodology

This carbon footprint estimate is calculated using:

1. **System Specifications**:
   - CPU: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel (24 cores)
   - GPU: NVIDIA GeForce RTX 4070 Ti (12GB VRAM)
   - Training Duration: 2.5 hours
   - Total Power Consumption: 500 W

2. **Power Consumption Estimates**:
   - Based on typical hardware power draw during ML training
   - Includes CPU (150W), GPU (250W), and system overhead (100W)
   - Total Energy: 1.25 kWh

3. **Emission Factors**:
   - Global Average: 0.594 kg CO2 (475g CO2/kWh)
   - USA Grid: 0.483 kg CO2 (386g CO2/kWh)
   - European Grid: 0.345 kg CO2 (276g CO2/kWh)
   - With Renewable Energy: 0.051 kg CO2 (41g CO2/kWh)

**Note**: These are estimates based on system specifications. For precise measurements, use tools like [CodeCarbon](https://codecarbon.io/) during actual training. #2

### Environmental Impact Comparison

| Reference Point          | CO2 Emissions |
| ------------------------ | ------------- |
| **This Model Training**  | 0.594 kg CO2  |
| Smartphone usage (1 day) | 0.30 kg CO2   |
| Car travel (3.5 km)      | 0.594 kg CO2  |
| Tree absorption (1 year) | 21.8 kg CO2   |

### Recommendations

- **Renewable Energy**: Using renewable energy could reduce emissions by ~91% (to 0.051 kg CO2)
- **Model Efficiency**: The lightweight architecture minimizes computational requirements
- **Transfer Learning**: Consider fine-tuning pre-trained models for similar tasks
- **Hyperparameter Optimization**: Efficient training reduced epochs from 30 to 27

### Daily Inference Impact

For reference, typical daily inference usage:

- **Processing**: 10,000 images/day
- **Energy Consumption**: 0.0014 kWh/day
- **CO2 Emissions**: 0.66 g CO2/day
- **Annual Impact**: 241 g CO2/year

---

_Carbon footprint measured on September 30, 2025_

## Citation

### BibTeX

```bibtex
@software{kanji_recognition_etl9g_2025,
  title={ETL9G Kanji Recognition Model with Channel Attention},
  author={Jukka Paasonen},
  year={2025},
  url={https://github.com/paazmaya/kanji-2965-CNN-ETL9G},
  license={MIT}
}
```

### Dataset Citation

```bibtex
@dataset{etl9g_dataset,
  title={ETL Character Database},
  author={National Institute of Advanced Industrial Science and Technology},
  url={http://etlcdb.db.aist.go.jp/},
  note={ETL9G: JIS Level 1 Kanji and Hiragana Characters}
}
```

## Model Card Authors

- **Jukka Paasonen** - Model development and training

## Model Card Contact

For questions about this model, please open an issue in the [GitHub repository](https://github.com/paazmaya/kanji-2965-CNN-ETL9G).

---

**Last Updated**: September 30, 2025  
**Model Version**: v2.1  
**License**: MIT
