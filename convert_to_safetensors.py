#!/usr/bin/env python3
"""
Convert PyTorch Kanji Model to SafeTensors Format
Provides secure, fast loading format with embedded metadata.
"""

import argparse
from pathlib import Path
import torch
from safetensors.torch import save_file
import json
from train_etl9g_model import LightweightKanjiNet


def convert_to_safetensors(
    model_path="best_kanji_model.pth",
    output_path=None,
    include_metadata=True,
):
    """Convert PyTorch model to SafeTensors format.

    SafeTensors provides secure, efficient model weight storage with metadata
    """

    # ETL9G dataset has exactly 3,036 character classes (fixed)
    NUM_CLASSES = 3036

    # Generate default filename if not provided
    if output_path is None:
        output_path = generate_output_filename("kanji_model", 64, ".safetensors")

    print("🔄 Converting PyTorch model to SafeTensors format...")
    print(f"📁 Input: {model_path}")
    print(f"📁 Output: {output_path}")
    print(f"Classes: {NUM_CLASSES} (ETL9G dataset)")

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_conversion(model_path, image_size=64)

    # Extract state dict
    state_dict = model.state_dict()

    # Convert tensors to CPU and ensure they're contiguous
    cpu_state_dict = {}
    for key, tensor in state_dict.items():
        cpu_state_dict[key] = tensor.cpu().contiguous()

    # Generate metadata if requested
    metadata = {}
    if include_metadata:
        metadata = extract_model_metadata(model, model_path)

    # Save as SafeTensors
    try:
        save_file(cpu_state_dict, output_path, metadata=metadata)
        print("✅ SafeTensors model saved successfully!")

        # Save companion info file
        info_path = str(output_path).replace(".safetensors", "_info.json")
        model_info = {
            "format": "safetensors",
            "model_path": str(model_path),
            "safetensors_path": str(output_path),
            "metadata": metadata,
            "architecture": {
                "type": "LightweightKanjiNet",
                "num_classes": NUM_CLASSES,
                "image_size": 64,
            },
        }

        with open(info_path, "w") as f:
            json.dump(model_info, f, indent=2)

        print(f"📄 Model info saved: {info_path}")
        return output_path

    except Exception as e:
        print(f"❌ Error saving SafeTensors: {e}")
        return None


def load_model_for_conversion(model_path, image_size=64):
    """Load the trained PyTorch model for conversion."""
    # ETL9G dataset has exactly 3,036 character classes (fixed)
    NUM_CLASSES = 3036

    print(f"📁 Loading model from: {model_path}")

    # Create model instance with same architecture as training
    model = LightweightKanjiNet(num_classes=NUM_CLASSES)  # Load the trained weights
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("✅ Loaded model weights from checkpoint")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded model weights directly")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    model.eval()
    return model


def extract_model_metadata(model_path, model, image_size):
    """Extract metadata about the model for SafeTensors header."""
    # ETL9G dataset has exactly 3,036 character classes (fixed)
    NUM_CLASSES = 3036

    metadata = {
        "framework": "pytorch",
        "model_type": "image_classification",
        "architecture": "LightweightKanjiNet",
        "task": "kanji_recognition",
        "dataset": "ETL9G",
        "num_classes": str(NUM_CLASSES),
        "input_size": f"{image_size}x{image_size}",
        "color_channels": "1",
        "format_version": "safetensors_v0.4.0",
    }

    # Add training info if available
    try:
        if Path("best_model_info.json").exists():
            with open("best_model_info.json", "r") as f:
                training_info = json.load(f)
                metadata.update(
                    {
                        "accuracy": str(training_info.get("accuracy", "unknown")),
                        "loss": str(training_info.get("loss", "unknown")),
                        "epoch": str(training_info.get("epoch", "unknown")),
                        "learning_rate": str(
                            training_info.get("learning_rate", "unknown")
                        ),
                    }
                )
    except Exception as e:
        print(f"⚠️  Could not load training metadata: {e}")

    # Add model architecture details
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metadata.update(
        {
            "total_parameters": str(total_params),
            "trainable_parameters": str(trainable_params),
            "model_size_mb": f"{total_params * 4 / (1024 * 1024):.2f}",  # Assuming float32
        }
    )

    return metadata


def generate_output_filename(base_name, image_size, suffix):
    """Generate consistent filename with configuration details."""
    return f"{base_name}_etl9g_{image_size}x{image_size}_3036classes{suffix}"


def convert_to_safetensors(
    model_path="best_kanji_model.pth",
    output_path=None,
    num_classes=3036,
    image_size=64,
    include_metadata=True,
):
    """Convert PyTorch model to SafeTensors format."""

    # Generate default filename if not provided
    if output_path is None:
        output_path = generate_output_filename(
            "kanji_model", image_size, ".safetensors"
        )

    print("🔄 Converting PyTorch model to SafeTensors...")
    print(f"Input model: {model_path}")
    print(f"Output path: {output_path}")
    print(f"Classes: {NUM_CLASSES} (ETL9G dataset)")
    print(f"Image size: {image_size}x{image_size}")

    # Load the model
    model = load_model_for_conversion(model_path, image_size)
    if model is None:
        return False

    # Get model state dict (weights and biases)
    state_dict = model.state_dict()

    print(f"📊 Model layers found: {len(state_dict)}")
    for name, tensor in state_dict.items():
        print(f"   {name}: {tensor.shape} ({tensor.dtype})")

    # Prepare metadata
    metadata = {}
    if include_metadata:
        metadata = extract_model_metadata(model_path, model, image_size)
        print(f"📝 Added metadata: {len(metadata)} fields")

    # Convert and save to SafeTensors
    try:
        save_file(state_dict, output_path, metadata=metadata)
        print("✅ SafeTensors conversion successful!")

        # Verify file was created and get size
        output_file = Path(output_path)
        if output_file.exists():
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"📁 Output file: {output_path} ({file_size_mb:.2f} MB)")

            # Create companion info file
            info_path = output_path.replace(".safetensors", "_info.json")
            model_info = {
                "model_file": output_path,
                "format": "safetensors",
                "architecture": "LightweightKanjiNet",
                "dataset": "ETL9G",
                "num_classes": NUM_CLASSES,
                "input_size": [
                    1,
                    1,
                    image_size,
                    image_size,
                ],  # [batch, channels, height, width]
                "preprocessing": {
                    "normalize": True,
                    "mean": [0.5],
                    "std": [0.5],
                    "resize": [image_size, image_size],
                },
                "metadata": metadata,
            }

            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            print(f"📋 Model info saved: {info_path}")

            return output_path
        else:
            print("❌ Output file not created")
            return None

    except Exception as e:
        print(f"❌ SafeTensors conversion failed: {e}")
        return None


def verify_safetensors_model(safetensors_path):
    """Verify the SafeTensors model can be loaded correctly."""
    try:
        from safetensors.torch import load_file

        print(f"🔍 Verifying SafeTensors model: {safetensors_path}")

        # Load the SafeTensors file
        state_dict = load_file(safetensors_path)

        print("✅ Successfully loaded SafeTensors model")
        print(f"📊 Layers: {len(state_dict)}")

        # Check tensor shapes and types
        total_params = 0
        for name, tensor in state_dict.items():
            total_params += tensor.numel()
            print(f"   {name}: {tensor.shape} ({tensor.dtype})")

        print(f"📈 Total parameters: {total_params:,}")

        # Try to load into model architecture
        # ETL9G dataset has exactly 3,036 character classes (fixed)
        model = LightweightKanjiNet(num_classes=3036)
        model.load_state_dict(state_dict)
        model.eval()

        print("✅ Model architecture compatibility verified")

        # Test forward pass
        test_input = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            output = model(test_input)
            print(f"✅ Forward pass test successful: output shape {output.shape}")

        return True

    except Exception as e:
        print(f"❌ SafeTensors verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch kanji model to SafeTensors"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="best_kanji_model.pth",
        help="Path to the trained PyTorch model",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for SafeTensors model (auto-generated if not specified)",
    )
    parser.add_argument(
        "--image-size", type=int, default=64, help="Input image size (square)"
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip adding metadata to SafeTensors file",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify the converted SafeTensors model"
    )

    args = parser.parse_args()

    print("🚀 SafeTensors Conversion Tool")
    print("=" * 50)
    print(f"PyTorch model: {args.model_path}")
    print(f"SafeTensors output: {args.output_path}")
    print("Classes: 3036 (ETL9G dataset)")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Include metadata: {not args.no_metadata}")
    print()

    # Convert to SafeTensors
    output_path = convert_to_safetensors(
        args.model_path,
        args.output_path,
        include_metadata=not args.no_metadata,
    )

    if output_path:
        print("\n🎉 SafeTensors conversion completed!")

        print("\n📋 **SafeTensors Benefits:**")
        print("   ✅ Secure - No arbitrary code execution")
        print("   ✅ Fast loading - Memory-mapped access")
        print("   ✅ Cross-platform - Language agnostic")
        print("   ✅ Metadata support - Rich model information")
        print("   ✅ Integrity checks - Built-in validation")

        print("\n📂 **Usage in deployment:**")
        print("   ```python")
        print("   from safetensors.torch import load_file")
        print(f"   state_dict = load_file('{output_path}')")
        print("   model.load_state_dict(state_dict)")
        print("   ```")

        # Verify if requested
        if args.verify:
            print("\n" + "=" * 50)
            verify_safetensors_model(output_path)

        print("\nFiles ready for deployment:")
        print(f"  - {output_path}")
        print(f"  - {output_path.replace('.safetensors', '_info.json')}")

    else:
        print("❌ SafeTensors conversion failed")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
