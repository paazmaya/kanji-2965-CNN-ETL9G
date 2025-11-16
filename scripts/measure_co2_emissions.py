#!/usr/bin/env python3
"""
CO2 Emissions Measurement for Kanji Recognition Model Training
Following Hugging Face guidelines: https://huggingface.co/docs/hub/model-cards-co2

This script measures and tracks CO2 emissions during model training and inference.
"""

import json
import platform
from datetime import datetime
from pathlib import Path

import psutil
import torch

# Try to import CodeCarbon for CO2 tracking
try:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker

    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("⚠️  CodeCarbon not installed. Install with: uv pip install codecarbon")


def get_system_info():
    """Get detailed system information for CO2 calculation"""
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }

    # GPU information
    if torch.cuda.is_available():
        info["gpu_available"] = True
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )
        info["cuda_version"] = torch.version.cuda
    else:
        info["gpu_available"] = False

    return info


def estimate_training_emissions():
    """Estimate CO2 emissions for model training based on system specs and training time"""

    print("🌍 CO2 Emissions Estimation for Kanji Recognition Model")
    print("=" * 60)
    print()

    # Get system information
    sys_info = get_system_info()

    print("📊 System Configuration:")
    print(f"   - Platform: {sys_info['platform']}")
    print(f"   - CPU: {sys_info['processor']} ({sys_info['cpu_count']} cores)")
    print(f"   - Memory: {sys_info['memory_total_gb']} GB")
    if sys_info["gpu_available"]:
        print(f"   - GPU: {sys_info['gpu_name']} ({sys_info['gpu_memory_gb']} GB)")
    else:
        print("   - GPU: Not available")
    print()

    # Training parameters from our model
    training_params = {
        "epochs": 30,
        "actual_epochs": 27,  # Early stopping
        "batch_size": 64,
        "dataset_size": 607200,  # ETL9G total samples
        "training_split": 0.8,
        "estimated_training_time_hours": 2.5,  # Based on our training
        "model_parameters": 1735527,
        "model_size_mb": 6.62,
    }

    print("🏋️ Training Configuration:")
    for key, value in training_params.items():
        if isinstance(value, float):
            print(f"   - {key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"   - {key.replace('_', ' ').title()}: {value:,}")
    print()

    # Power consumption estimates (typical values)
    power_estimates = {
        "cpu_idle_watts": 50,
        "cpu_training_watts": 150,
        "gpu_idle_watts": 30 if sys_info["gpu_available"] else 0,
        "gpu_training_watts": 250 if sys_info["gpu_available"] else 0,  # Assuming mid-range GPU
        "system_overhead_watts": 100,  # PSU, cooling, etc.
    }

    # Calculate total power consumption
    total_training_watts = (
        power_estimates["cpu_training_watts"]
        + power_estimates["gpu_training_watts"]
        + power_estimates["system_overhead_watts"]
    )

    total_training_kwh = (
        total_training_watts * training_params["estimated_training_time_hours"]
    ) / 1000

    print("⚡ Power Consumption Estimates:")
    print(f"   - CPU Training Power: {power_estimates['cpu_training_watts']} W")
    if sys_info["gpu_available"]:
        print(f"   - GPU Training Power: {power_estimates['gpu_training_watts']} W")
    print(f"   - System Overhead: {power_estimates['system_overhead_watts']} W")
    print(f"   - Total Training Power: {total_training_watts} W")
    print(f"   - Total Energy Consumption: {total_training_kwh:.3f} kWh")
    print()

    # CO2 emission factors (kg CO2 per kWh)
    # These vary by country/region - using global averages
    emission_factors = {
        "global_average": 0.475,  # kg CO2/kWh
        "usa": 0.386,
        "europe": 0.276,
        "china": 0.681,
        "renewable": 0.041,  # If using renewable energy
    }

    print("🌍 CO2 Emission Estimates by Region:")
    for region, factor in emission_factors.items():
        co2_kg = total_training_kwh * factor
        print(
            f"   - {region.replace('_', ' ').title()}: {co2_kg:.6f} kg CO2 ({co2_kg * 1000:.3f} g CO2)"
        )
    print()

    # Additional metrics
    inference_power_watts = 50  # Much lower for inference
    inference_time_ms = 10  # ~10ms per image
    images_per_day = 10000  # Example usage

    daily_inference_kwh = (
        inference_power_watts * (images_per_day * inference_time_ms / 1000) / 3600
    ) / 1000
    daily_inference_co2 = daily_inference_kwh * emission_factors["global_average"]

    print("📱 Inference Emissions (Example Usage):")
    print(f"   - Power per inference: ~{inference_power_watts} W")
    print(f"   - Time per inference: ~{inference_time_ms} ms")
    print(f"   - Daily usage: {images_per_day:,} images")
    print(f"   - Daily energy: {daily_inference_kwh:.6f} kWh")
    print(f"   - Daily CO2: {daily_inference_co2 * 1000:.3f} g CO2")
    print()

    # Create summary report
    report = {
        "model_info": {
            "name": "ETL9G Kanji Recognition Model",
            "version": "v2.1",
            "parameters": training_params["model_parameters"],
            "size_mb": training_params["model_size_mb"],
        },
        "system_info": sys_info,
        "training": {
            "duration_hours": training_params["estimated_training_time_hours"],
            "energy_kwh": total_training_kwh,
            "power_watts": total_training_watts,
            "co2_emissions_kg": {
                region: total_training_kwh * factor for region, factor in emission_factors.items()
            },
        },
        "inference_daily_example": {
            "images_processed": images_per_day,
            "energy_kwh": daily_inference_kwh,
            "co2_emissions_g": daily_inference_co2 * 1000,
        },
        "methodology": "Estimated based on system specifications and typical power consumption",
        "measurement_date": datetime.now().isoformat(),
    }

    # Save report
    report_file = "co2_emissions_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"📄 Detailed report saved to: {report_file}")

    return report


def setup_codecarbon_tracking():
    """Set up CodeCarbon for actual measurement during training"""

    if not CODECARBON_AVAILABLE:
        print("❌ CodeCarbon not available. Install with:")
        print("   uv pip install codecarbon")
        return None

    print("🔧 Setting up CodeCarbon tracking...")

    # Create CodeCarbon configuration
    config = {
        "project_name": "kanji-recognition-etl9g",
        "measure_power_secs": 15,  # Measure every 15 seconds
        "save_to_file": True,
        "output_dir": "./emissions/",
        "country_iso_code": "USA",  # Change to your country
        "region": "us-east-1",  # Change to your region
        "cloud_provider": None,  # Set if using cloud
        "cloud_region": None,
        "gpu_ids": [0] if torch.cuda.is_available() else None,
    }

    # Create emissions directory
    Path("./emissions/").mkdir(exist_ok=True)

    # Save configuration
    with open("./emissions/codecarbon_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("✅ CodeCarbon configuration saved to ./emissions/codecarbon_config.json")
    print("💡 To track training emissions, modify your training script:")
    print("""
    from codecarbon import EmissionsTracker
    
    # Start tracking at beginning of training
    tracker = EmissionsTracker(
        project_name="kanji-recognition-etl9g",
        output_dir="./emissions/"
    )
    tracker.start()
    
    # Your training code here...
    
    # Stop tracking at end of training
    emissions = tracker.stop()
    print(f"Training emissions: {emissions:.6f} kg CO2")
    """)

    return config


def create_carbon_footprint_section():
    """Create the carbon footprint section for the model card"""

    # Run estimation
    report = estimate_training_emissions()

    # Create model card section
    carbon_section = f"""
## Carbon Footprint

<!-- This template follows the Hugging Face guidelines: https://huggingface.co/docs/hub/model-cards-co2 -->

### Training

- **Hardware Type**: {report["system_info"].get("gpu_name", "CPU-only")}
- **Hours used**: {report["training"]["duration_hours"]} hours
- **Cloud Provider**: N/A (Local training)
- **Compute Region**: Local development machine
- **Carbon Emitted**: {report["training"]["co2_emissions_kg"]["global_average"]:.6f} kg CO2 (estimated)

### Methodology

This carbon footprint estimate is calculated using:

1. **System Specifications**:
   - CPU: {report["system_info"]["processor"]} ({report["system_info"]["cpu_count"]} cores)
   - GPU: {report["system_info"].get("gpu_name", "Not available")}
   - Training Duration: {report["training"]["duration_hours"]} hours
   - Total Power Consumption: {report["training"]["power_watts"]} W

2. **Power Consumption Estimates**:
   - Based on typical hardware power draw during ML training
   - Includes CPU, GPU, and system overhead
   - Total Energy: {report["training"]["energy_kwh"]:.3f} kWh

3. **Emission Factors**:
   - Global Average: {report["training"]["co2_emissions_kg"]["global_average"]:.6f} kg CO2
   - USA Grid: {report["training"]["co2_emissions_kg"]["usa"]:.6f} kg CO2
   - European Grid: {report["training"]["co2_emissions_kg"]["europe"]:.6f} kg CO2

**Note**: These are estimates based on system specifications. For precise measurements, use tools like [CodeCarbon](https://codecarbon.io/) during actual training.

### Recommendations

- Consider using renewable energy sources for training
- Implement model efficiency techniques to reduce training time
- Use transfer learning when possible to reduce computational requirements
- Monitor and optimize hyperparameters to minimize training iterations

### Daily Inference Impact

For reference, daily inference usage:
- Processing {report["inference_daily_example"]["images_processed"]:,} images/day
- Energy consumption: {report["inference_daily_example"]["energy_kwh"]:.6f} kWh/day  
- CO2 emissions: {report["inference_daily_example"]["co2_emissions_g"]:.3f} g CO2/day

---

*Carbon footprint measured on {report["measurement_date"][:10]}*
"""

    # Save to file
    with open("carbon_footprint_section.md", "w") as f:
        f.write(carbon_section)

    print("\n📝 Carbon footprint section saved to: carbon_footprint_section.md")
    print("💡 You can copy this section into your model-card.md file")

    return carbon_section


def main():
    """Main function to run CO2 measurement and reporting"""

    print("🌱 Kanji Recognition Model - CO2 Emissions Assessment")
    print("=" * 70)
    print()

    # 1. Estimate emissions from completed training
    print("📊 Step 1: Estimating emissions from completed training...")
    estimate_training_emissions()
    print()

    # 2. Set up CodeCarbon for future training
    print("🔧 Step 2: Setting up CodeCarbon for precise measurement...")
    setup_codecarbon_tracking()
    print()

    # 3. Create model card section
    print("📝 Step 3: Creating model card carbon footprint section...")
    create_carbon_footprint_section()
    print()

    print("✅ CO2 assessment complete!")
    print("\n📋 Next Steps:")
    print("1. Review the generated carbon footprint section")
    print("2. Add it to your model-card.md file")
    print("3. Install CodeCarbon for future training runs")
    print("4. Consider using renewable energy for training")


if __name__ == "__main__":
    main()
