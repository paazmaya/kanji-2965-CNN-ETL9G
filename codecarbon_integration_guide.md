# CodeCarbon Integration Guide

```python
# Add these imports at the top of train_etl9g_model.py
from codecarbon import EmissionsTracker
import json
from datetime import datetime

def main():
    """Enhanced training function with CO2 tracking"""

    # Initialize emissions tracker BEFORE any GPU operations
    tracker = EmissionsTracker(
        project_name="kanji-recognition-etl9g",
        output_dir="./emissions/",
        output_file="training_emissions.csv",
        save_to_file=True,
        save_to_api=False,  # Set to True if using CodeCarbon API
        country_iso_code="USA",  # Change to your country
        region="us-east-1",  # Change to your region
        measure_power_secs=15,  # Measure every 15 seconds
        tracking_mode="machine",  # Track entire machine
        log_level="info"
    )

    print("🌱 Starting CO2 emissions tracking...")
    tracker.start()

    try:
        # Your existing training code here...
        # parse arguments, setup data loaders, model, etc.

        # Training loop with emissions context
        for epoch in range(args.epochs):
            # Training code...
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            # Log progress including current emissions
            current_emissions = tracker.final_emissions_data
            print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, "
                  f"Val Acc: {val_acc:.2f}%, Current CO2: {current_emissions.emissions:.6f} kg")

            # Early stopping, model saving, etc...

    finally:
        # Always stop tracking, even if training fails
        final_emissions = tracker.stop()

        print(f"\n🌍 Training completed!")
        print(f"Total CO2 emissions: {final_emissions:.6f} kg CO2")

        # Save detailed emissions report
        emissions_report = {
            "model_name": "kanji-recognition-etl9g",
            "training_date": datetime.now().isoformat(),
            "total_emissions_kg": final_emissions,
            "training_duration_hours": tracker.final_emissions_data.duration.total_seconds() / 3600,
            "energy_consumed_kwh": tracker.final_emissions_data.energy_consumed,
            "country": "USA",  # Your country
            "grid_carbon_intensity": tracker.final_emissions_data.country_iso_code,
            "hardware": {
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only",
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }

        with open("./emissions/training_emissions_report.json", "w") as f:
            json.dump(emissions_report, f, indent=2, default=str)

        print(f"📄 Detailed emissions report saved to: ./emissions/training_emissions_report.json")

if __name__ == "__main__":
    main()
```

## Configuration Options

### Country Codes

Common country codes for emissions factors:

- USA: United States
- FRA: France
- DEU: Germany
- JPN: Japan
- GBR: United Kingdom
- CAN: Canada
- AUS: Australia

### Cloud Regions

If training on cloud platforms:

- AWS: us-east-1, eu-west-1, etc.
- Azure: eastus, westeurope, etc.
- GCP: us-central1, europe-west1, etc.

### Tracking Modes

- `machine`: Track entire machine power consumption (recommended)
- `process`: Track only the Python process (less accurate)

## Output Files

1. **training_emissions.csv**: Timestamped measurements during training
2. **training_emissions_report.json**: Summary with total emissions and metadata

## Best Practices

1. Start tracking before any GPU initialization
2. Use try/finally to ensure tracker.stop() is called
3. Configure your actual country/region for accurate emissions factors
4. Save detailed reports for model card documentation
5. Monitor energy consumption patterns to optimize training efficiency
