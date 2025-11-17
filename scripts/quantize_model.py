#!/usr/bin/env python3
"""
Post-Training INT8 Quantization for Kanji Models
Converts trained PyTorch models to INT8 for efficient deployment
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from optimization_config import (
    create_data_loaders,
    load_chunked_dataset,
)
from train_cnn_model import LightweightKanjiNet
from train_hiercode import HierCodeClassifier


def quantize_model_int8(model: nn.Module, model_name: str = "quantized"):
    """
    Convert model to INT8 using PyTorch quantization.

    This uses Post-Training Quantization (PTQ):
    - No retraining required
    - Converts weights to INT8
    - Reduces model size by ~4x
    - Minimal accuracy loss (typically <1%)
    """
    print("\n" + "=" * 70)
    print("POST-TRAINING INT8 QUANTIZATION")
    print("=" * 70)

    # Move to CPU for quantization (INT8 primarily for CPU)
    model = model.cpu()
    model.eval()

    print("\n📊 Original Model:")
    print(f"  Size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # Calculate original size
    original_state = model.state_dict()
    original_size = sum(
        v.numel() * v.element_size() for v in original_state.values() if v is not None
    )
    print(f"  Weight size: {original_size / 1e6:.2f} MB")

    # Set model to eval mode
    model.eval()

    # Insert quantization stubs
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)

    print("\n🔄 Prepared model for quantization")

    # Calibrate with representative data (optional but recommended)
    # For better accuracy, use actual dataset
    print("   Note: Skipping calibration (static quantization)")

    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)

    print("\n✅ Model converted to INT8")

    # Calculate quantized size
    quantized_state = model.state_dict()
    quantized_size = sum(
        v.numel() * v.element_size() for v in quantized_state.values() if v is not None
    )
    print(f"  Quantized size: {quantized_size / 1e6:.2f} MB")
    print(f"  Size reduction: {original_size / quantized_size:.2f}x")
    print(f"  Space saved: {(original_size - quantized_size) / 1e6:.2f} MB")

    return model, original_size, quantized_size


def quantize_with_calibration(
    model: nn.Module,
    train_loader,
    device: str = "cpu",
    model_name: str = "quantized",
):
    """
    Quantize model with calibration using actual training data.

    This improves quantization accuracy by:
    - Using real data distribution for calibration
    - Computing optimal scale factors
    - Reducing accuracy loss from ~2-3% to <1%
    """
    print("\n" + "=" * 70)
    print("POST-TRAINING INT8 QUANTIZATION WITH CALIBRATION")
    print("=" * 70)

    model = model.to(device)
    model.eval()

    print("\n📊 Original Model:")
    print(f"  Size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    original_state = model.state_dict()
    original_size = sum(
        v.numel() * v.element_size() for v in original_state.values() if isinstance(v, torch.Tensor)
    )
    print(f"  Weight size: {original_size / 1e6:.2f} MB")

    # Prepare for quantization
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)

    print("\n🔄 Calibrating on training data...")

    # Calibration: Run model on subset of training data
    num_batches = min(100, len(train_loader))  # Use 100 batches for calibration
    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):
            if i >= num_batches:
                break

            images = images.to(device)

            # Forward pass to collect activation statistics
            if hasattr(model, "forward"):
                _ = model(images)

            if (i + 1) % 25 == 0:
                print(f"  Calibrated on {i + 1}/{num_batches} batches")

    print("  ✓ Calibration complete")

    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)

    print("\n✅ Model converted to INT8 (with calibration)")

    quantized_state = model.state_dict()
    quantized_size = sum(
        v.numel() * v.element_size()
        for v in quantized_state.values()
        if isinstance(v, torch.Tensor)
    )
    print(f"  Quantized size: {quantized_size / 1e6:.2f} MB")
    print(f"  Size reduction: {original_size / quantized_size:.2f}x")

    return model, original_size, quantized_size


def evaluate_quantized_model(model, test_loader, criterion, device: str = "cpu"):
    """Evaluate quantized model accuracy"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    print("\n🧪 Evaluating quantized model...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)

    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Loss: {avg_loss:.4f}")

    return accuracy, avg_loss


def main():
    parser = argparse.ArgumentParser(
        description="Post-Training INT8 Quantization for Kanji Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize HierCode model
  python quantize_model.py --model-path models/best_kanji_model.pth --model-type hiercode

  # Quantize with calibration
  python quantize_model.py --model-path models/best_kanji_model.pth --model-type cnn --calibrate

  # Evaluate quantized model
  python quantize_model.py --model-path models/best_kanji_model.pth --evaluate
        """,
    )

    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="hiercode",
        choices=["hiercode", "cnn", "rnn"],
        help="Model architecture type (default: hiercode)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset",
        help="Dataset directory (default: dataset)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Use calibration with training data for better accuracy",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate quantized model on test set",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for quantization (default: cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for quantized model (default: auto-generated)",
    )

    args = parser.parse_args()

    # Load model
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    print(f"📂 Loading: {model_path}")

    # Load config
    config_path = model_path.parent / f"{args.model_type}_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
            print(f"📋 Config: {config_dict}")
    else:
        print(f"⚠️  Config not found: {config_path}")
        config_dict = {"num_classes": 3036}

    num_classes = config_dict.get("num_classes", 3036)

    # Create config based on model type
    if args.model_type == "hiercode":
        from optimization_config import HierCodeConfig

        config = HierCodeConfig(num_classes=num_classes)
    else:
        from optimization_config import OptimizationConfig

        config = OptimizationConfig(num_classes=num_classes)

    # Load model based on type
    if args.model_type == "hiercode":
        model = HierCodeClassifier(num_classes=num_classes, config=config)
    elif args.model_type == "cnn":
        model = LightweightKanjiNet(num_classes=num_classes)
    else:
        print(f"❌ Unknown model type: {args.model_type}")
        return

    # Load state dict with flexible key matching
    checkpoint = torch.load(model_path, map_location="cpu")
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError:
        # Try loading with strict=False for compatibility
        print("⚠️  Strict loading failed, trying flexible loading...")
        model.load_state_dict(checkpoint, strict=False)
        print("✓ Model loaded with some keys skipped (compatibility mode)")
    else:
        print("✓ Model loaded successfully")

    # Quantize
    if args.calibrate:
        # Load dataset for calibration
        print("\n📂 Loading dataset for calibration...")
        X, y = load_chunked_dataset(args.data_dir)
        train_loader, _, test_loader = create_data_loaders(X, y, config)

        quantized_model, orig_size, quant_size = quantize_with_calibration(
            model, train_loader, device=args.device, model_name=args.model_type
        )
    else:
        quantized_model, orig_size, quant_size = quantize_model_int8(
            model, model_name=args.model_type
        )

    # Save quantized model
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = model_path.parent / f"quantized_{args.model_type}_int8.pth"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized_model.state_dict(), output_path)
    print(f"\n✅ Quantized model saved: {output_path}")

    # Evaluate if requested
    if args.evaluate:
        print("\n📂 Loading test set...")
        X, y = load_chunked_dataset(args.data_dir)
        _, _, test_loader = create_data_loaders(X, y, config)

        criterion = nn.CrossEntropyLoss()
        accuracy, loss = evaluate_quantized_model(
            quantized_model, test_loader, criterion, device=args.device
        )

        # Save results
        results = {
            "model_type": args.model_type,
            "original_size_mb": orig_size / 1e6,
            "quantized_size_mb": quant_size / 1e6,
            "size_reduction": orig_size / quant_size,
            "quantized_accuracy": accuracy,
            "quantized_loss": loss,
            "calibrated": args.calibrate,
        }

        results_path = output_path.parent / f"quantization_results_{args.model_type}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n📊 Results saved: {results_path}")
        print("\n✅ Quantization Summary:")
        print(f"  Original: {orig_size / 1e6:.2f} MB")
        print(f"  Quantized: {quant_size / 1e6:.2f} MB")
        print(f"  Reduction: {orig_size / quant_size:.2f}x")
        print(f"  Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
