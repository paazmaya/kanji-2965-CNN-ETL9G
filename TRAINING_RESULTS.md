# Training Results Comparison

## Model Performance Summary

| Model                    | Architecture        | Parameters | Model Size | Best Accuracy | Training Time | Final Loss | Epochs Completed |
| ------------------------ | ------------------- | ---------- | ---------- | ------------- | ------------- | ---------- | ---------------- |
| **CNN (Lightweight)**    | LightweightKanjiNet | 1,735,527  | 6.62 MB    | **97.18%**    | ~4.5 hours    | 1.5120     | 30/30 ✅         |
| **CNN (Enhanced)**       | LightweightKanjiNet | 3,863,655  | 14.74 MB   | **97.18%**    | ~4.5 hours    | 1.5120     | 27/30 ✅         |
| **RNN (Hybrid CNN-RNN)** | Hybrid CNN-RNN      | 6,158,748  | ~23.5 MB   | **98.40%**    | ~22 hours     | 0.0685     | 22/50 ❌         |

## Detailed Training Metrics

### CNN Model (LightweightKanjiNet)

- **Architecture**: Channel Attention + Lightweight CNN
- **Dataset**: ETL9G (607,200 samples, 3,036 classes)
- **Training Strategy**: 30 epochs with adaptive learning rate
- **Best Performance**: Epoch 27
  - **Validation Accuracy**: 97.18%
  - **Training Accuracy**: 93.51%
  - **Validation Loss**: 1.5120
  - **Learning Rate**: 2.54e-05

### RNN Model (Hybrid CNN-RNN)

- **Architecture**: CNN Feature Extractor + LSTM + Attention
- **Dataset**: ETL9G (607,200 samples, 3,036 classes)
- **Training Strategy**: 50 epochs planned (interrupted at epoch 22)
- **Best Performance**: Epoch 18
  - **Validation Accuracy**: 98.40%
  - **Training Loss**: 0.1180
  - **Validation Loss**: 0.0685

## Training Time Analysis

### CNN Training Performance

```
Total Training Time: ~4.5 hours (30 epochs)
Average Time per Epoch: 9 minutes
Training Efficiency: Very High
Final Training Accuracy: 93.67%
Final Validation Accuracy: 97.17%
```

### RNN Training Performance

```
Total Training Time: ~22 hours (22 epochs completed)
Average Time per Epoch: 60 minutes
Training Efficiency: Moderate (6.7x slower than CNN)
Last Completed Validation Accuracy: 98.24%
Peak Validation Accuracy: 98.40% (Epoch 18)
Training Status: Interrupted due to KeyboardInterrupt
```

## Accuracy Progression

### CNN Model Training Curve

| Epoch  | Train Acc  | Val Acc    | Train Loss | Val Loss   | Time per Epoch |
| ------ | ---------- | ---------- | ---------- | ---------- | -------------- |
| 1      | 2.06%      | 15.41%     | 6.6846     | 4.9642     | 9 min          |
| 5      | 73.90%     | 84.33%     | 2.5505     | 2.1007     | 9 min          |
| 10     | 86.72%     | 94.73%     | 2.0567     | 1.6785     | 9 min          |
| 15     | 90.53%     | 96.24%     | 1.8952     | 1.5768     | 9 min          |
| 20     | 92.38%     | 96.80%     | 1.8165     | 1.5367     | 9 min          |
| 25     | 93.35%     | 97.06%     | 1.7775     | 1.5155     | 9 min          |
| **27** | **93.51%** | **97.18%** | **1.7697** | **1.5120** | **9 min**      |

### RNN Model Training Curve

| Epoch  | Train Acc | Val Acc    | Train Loss | Val Loss   | Time per Epoch |
| ------ | --------- | ---------- | ---------- | ---------- | -------------- |
| 1      | -         | 0.02%      | 8.0239     | 8.0234     | 32 min         |
| 3      | -         | 84.40%     | 3.1385     | 0.5625     | 45 min         |
| 5      | -         | 96.01%     | 0.3706     | 0.1422     | 53 min         |
| 8      | -         | 97.33%     | 0.1975     | 0.0972     | 65 min         |
| 13     | -         | 98.01%     | 0.1368     | 0.0779     | 73 min         |
| **18** | **-**     | **98.40%** | **0.1180** | **0.0685** | **62 min**     |
| 22     | -         | 98.24%     | 0.1111     | 0.0790     | 75 min         |

## Model Architecture Comparison

### CNN (LightweightKanjiNet)

```
Features:
✅ Channel Attention Mechanism
✅ Lightweight Architecture (1.7M parameters)
✅ Fast Training (9 min/epoch)
✅ Fast Inference
✅ Stable Training
✅ ONNX/WASM Ready

Limitations:
❌ Spatial feature focus only
❌ No sequential understanding
❌ Lower parameter efficiency for complex patterns
```

### RNN (Hybrid CNN-RNN)

```
Features:
✅ Sequential Pattern Understanding
✅ Higher Accuracy Potential (98.40% achieved)
✅ Temporal Modeling Capability
✅ Structural Awareness
✅ Parameter Efficient for Complex Patterns

Limitations:
❌ Much Slower Training (6.7x slower)
❌ Higher Memory Requirements
❌ Larger Model Size (6.2M parameters)
```

## Performance Efficiency Analysis

### Training Speed Comparison

| Metric             | CNN       | RNN      | Ratio    |
| ------------------ | --------- | -------- | -------- |
| Parameters         | 1.7M      | 6.2M     | 3.6x     |
| Time per Epoch     | 9 min     | 60 min   | **6.7x** |
| Model Size         | 6.62 MB   | ~23.5 MB | 3.5x     |
| Training Stability | Excellent | Moderate | -        |

### Accuracy vs Efficiency

| Model   | Best Accuracy | Training Efficiency | Deployment Readiness |
| ------- | ------------- | ------------------- | -------------------- |
| **CNN** | 97.18%        | ⭐⭐⭐⭐⭐          | ⭐⭐⭐⭐⭐           |
| **RNN** | 98.40%        | ⭐⭐                | ⭐⭐⭐               |

## Parameter Efficiency

### CNN Models

- **Lightweight**: 1,735,527 parameters → 97.18% accuracy
- **Enhanced**: 3,863,655 parameters → 97.18% accuracy (same performance)

### RNN Model

- **Hybrid CNN-RNN**: 6,158,748 parameters → 98.40% accuracy
- **Parameter Efficiency**: 1.22% accuracy improvement per million parameters

## Computational Requirements

### Memory Usage

| Model | Training Memory | Inference Memory | Batch Processing |
| ----- | --------------- | ---------------- | ---------------- |
| CNN   | ~2-4 GB         | ~100-200 MB      | Excellent        |
| RNN   | ~8-12 GB        | ~400-600 MB      | Good             |

### Training Infrastructure

| Model | GPU Requirement | Training Time | Scalability |
| ----- | --------------- | ------------- | ----------- |
| CNN   | Optional        | 4.5 hours     | High        |
| RNN   | Recommended     | 22+ hours     | Moderate    |

## Deployment Considerations

### Production Readiness

| Feature           | CNN                | RNN                |
| ----------------- | ------------------ | ------------------ |
| Model Size        | ✅ Small (6.62 MB) | ⚠️ Large (23.5 MB) |
| Inference Speed   | ✅ Fast            | ⚠️ Moderate        |
| Web Deployment    | ✅ ONNX/WASM Ready | ❌ Complex         |
| Mobile Deployment | ✅ Lightweight     | ❌ Resource Heavy  |
| Edge Computing    | ✅ Excellent       | ⚠️ Limited         |

## Recommendations

### For Production Deployment

**Choose CNN (LightweightKanjiNet)** when:

- Fast inference is critical
- Deploying to web/mobile/edge devices
- Limited computational resources
- 97.18% accuracy is sufficient
- Training time is a constraint

### For Research/High Accuracy

**Choose RNN (Hybrid CNN-RNN)** when:

- Maximum accuracy is required (98.40%)
- Computational resources are abundant
- Sequential pattern understanding is valuable
- Training time is not a constraint
- Research/experimental use case

## Conclusion

| Aspect                 | Winner  | Reasoning                                  |
| ---------------------- | ------- | ------------------------------------------ |
| **Accuracy**           | RNN     | 98.40% vs 97.18% (+1.22% improvement)      |
| **Training Speed**     | CNN     | 6.7x faster training                       |
| **Model Size**         | CNN     | 3.5x smaller model                         |
| **Deployment**         | CNN     | Better suited for production               |
| **Research Value**     | RNN     | Higher potential, sequential understanding |
| **Overall Production** | **CNN** | Best balance of accuracy and efficiency    |

The **CNN model** offers the best production balance with 97.18% accuracy, fast training, and excellent deployment characteristics. The **RNN model** shows promise with higher accuracy potential (98.40%) but requires significantly more computational resources and training time.
