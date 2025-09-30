## Carbon Footprint

<!-- This template follows the Hugging Face guidelines: https://huggingface.co/docs/hub/model-cards-co2 -->

### Training

- **Hardware Type**: NVIDIA GeForce RTX 4070 Ti
- **Hours used**: 2.5 hours
- **Cloud Provider**: N/A (Local training)
- **Compute Region**: Local development machine
- **Carbon Emitted**: 0.593750 kg CO2 (estimated)

### Methodology

This carbon footprint estimate is calculated using:

1. **System Specifications**:
   - CPU: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel (24 cores)
   - GPU: NVIDIA GeForce RTX 4070 Ti
   - Training Duration: 2.5 hours
   - Total Power Consumption: 500 W

2. **Power Consumption Estimates**:
   - Based on typical hardware power draw during ML training
   - Includes CPU, GPU, and system overhead
   - Total Energy: 1.250 kWh

3. **Emission Factors**:
   - Global Average: 0.593750 kg CO2
   - USA Grid: 0.482500 kg CO2
   - European Grid: 0.345000 kg CO2

**Note**: These are estimates based on system specifications. For precise measurements, use tools like [CodeCarbon](https://codecarbon.io/) during actual training.

### Recommendations

- Consider using renewable energy sources for training
- Implement model efficiency techniques to reduce training time
- Use transfer learning when possible to reduce computational requirements
- Monitor and optimize hyperparameters to minimize training iterations

### Daily Inference Impact

For reference, daily inference usage:

- Processing 10,000 images/day
- Energy consumption: 0.001389 kWh/day
- CO2 emissions: 0.660 g CO2/day

---

_Carbon footprint measured on 2025-09-30_
