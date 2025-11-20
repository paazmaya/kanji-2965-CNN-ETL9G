#!/usr/bin/env python3
"""
Export HierCode Model to ONNX Format
Converts PyTorch model to ONNX for cross-platform deployment
"""

import argparse
import json
from pathlib import Path

import torch
from optimization_config import HierCodeConfig
from train_hiercode import HierCodeClassifier


def export_to_onnx(
    model_path: str,
    output_path: str = None,
    opset_version: int = 14,
    model_type: str = "hiercode",
):
    """Export model to ONNX format"""
    print("\n" + "=" * 70)
    print("EXPORTING TO ONNX")
    print("=" * 70)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    print(f"\n📂 Loading model: {model_path}")

    # Load config
    config_path = model_path.parent / f"{model_type}_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
    else:
        config_dict = {"num_classes": 3036}

    num_classes = config_dict.get("num_classes", 3036)

    # Create model
    if model_type == "hiercode":
        config = HierCodeConfig(num_classes=num_classes)
        model = HierCodeClassifier(num_classes=num_classes, config=config)
    else:
        print(f"❌ Unknown model type: {model_type}")
        return

    # Load weights
    checkpoint = torch.load(model_path, map_location="cpu")

    # Check if this is a quantized model
    is_quantized = any(
        "qint" in str(v.dtype) or "quint" in str(v.dtype)
        for v in checkpoint.values()
        if isinstance(v, torch.Tensor)
    )

    if is_quantized:
        print("ℹ️  Loading quantized model (INT8)")
        # For quantized models, load directly without strict checking
        model.load_state_dict(checkpoint, strict=False)
    else:
        print("ℹ️  Loading standard model")
        model.load_state_dict(checkpoint, strict=False)

    model.eval()

    print("✓ Model loaded successfully")

    # Generate output path if not specified
    if output_path is None:
        suffix = "_int8" if "quantized" in model_path.name else ""
        output_path = model_path.parent / f"hiercode{suffix}_opset{opset_version}.onnx"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n🔧 Exporting to ONNX...")
    print(f"  Model type: {model_type}")
    print(f"  Opset version: {opset_version}")
    print("  Input: (batch_size, 1, 64, 64)")
    print(f"  Output: (batch_size, {num_classes})")

    # Create dummy input
    dummy_input = torch.randn(1, 1, 64, 64)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input_image"],
        output_names=["logits"],
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
        export_params=True,
    )

    print("\n✅ Model exported to ONNX")
    print(f"  Path: {output_path}")

    # Check file size
    file_size = output_path.stat().st_size
    print(f"  Size: {file_size / 1e6:.2f} MB")

    # Create info file
    info = {
        "model_type": model_type,
        "num_classes": num_classes,
        "input_shape": [1, 1, 64, 64],
        "input_names": ["input_image"],
        "output_names": ["logits"],
        "opset_version": opset_version,
        "file_size_mb": file_size / 1e6,
        "from_pytorch": str(model_path),
    }

    info_path = output_path.with_suffix(".json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n📋 Info file saved: {info_path}")

    return output_path, info


def verify_onnx(onnx_path: str):
    """Verify ONNX model is valid"""
    print("\n🔍 Verifying ONNX model...")

    try:
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print("  ✓ ONNX model is valid")
        return True
    except ImportError:
        print("  ⚠️  onnx package not installed, skipping verification")
        print("     Install with: uv pip install onnx")
        return False
    except Exception as e:
        print(f"  ❌ ONNX validation failed: {e}")
        return False


def test_inference(onnx_path: str, num_samples: int = 5):
    """Test ONNX model inference"""
    print("\n🧪 Testing ONNX inference...")

    try:
        import onnxruntime as ort

        # Use GPU providers if available, fallback to CPU
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            ("CPUExecutionProvider", {}),
        ]

        sess = ort.InferenceSession(str(onnx_path), providers=providers)
        provider_used = sess.get_providers()[0]
        print(f"  ✓ ONNX Runtime session created (provider: {provider_used})")

        # Get input/output info
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        print(f"  Input: {input_name}")
        print(f"  Output: {output_name}")

        # Test inference
        print(f"\n  Running {num_samples} inference tests...")
        import time

        times = []
        for i in range(num_samples):
            test_input = (torch.randn(1, 1, 64, 64).numpy()).astype("float32")

            start = time.time()
            sess.run([output_name], {input_name: test_input})
            elapsed = time.time() - start
            times.append(elapsed * 1000)  # Convert to ms

            print(f"    Sample {i + 1}: {elapsed * 1000:.2f} ms")

        avg_time = sum(times) / len(times)
        print(f"\n  ✓ Average inference time: {avg_time:.2f} ms")
        print(f"  Throughput: {1000 / avg_time:.1f} samples/sec")

        return True

    except ImportError:
        print("  ⚠️  onnxruntime package not installed, skipping inference test")
        print("     Install with: uv pip install onnxruntime-gpu")
        return False
    except Exception as e:
        print(f"  ❌ Inference test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export HierCode Model to ONNX Format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export HierCode model
  python export_to_onnx.py --model-path training/hiercode/hiercode_model_best.pth

  # Export quantized model
  python export_to_onnx.py --model-path training/hiercode/quantized_hiercode_int8.pth

  # Export with specific opset version
  python export_to_onnx.py --model-path training/hiercode/hiercode_model_best.pth --opset 12

  # Export and test inference
  python export_to_onnx.py --model-path training/hiercode/hiercode_model_best.pth --test-inference
        """,
    )

    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for ONNX model (default: auto-generated)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify ONNX model validity",
    )
    parser.add_argument(
        "--test-inference",
        action="store_true",
        help="Test ONNX model inference",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="hiercode",
        help="Model type (default: hiercode)",
    )

    args = parser.parse_args()

    # Export
    onnx_path, info = export_to_onnx(
        args.model_path,
        output_path=args.output,
        opset_version=args.opset,
        model_type=args.model_type,
    )

    # Verify if requested
    if args.verify:
        verify_onnx(onnx_path)

    # Test inference if requested
    if args.test_inference:
        test_inference(onnx_path)

    print("\n" + "=" * 70)
    print("✅ EXPORT COMPLETE")
    print("=" * 70)
    print(f"\nONNX Model: {onnx_path}")
    print("\nDeployment Options:")
    print("  1. Python: uv pip install onnxruntime-gpu")
    print("  2. Web: ONNX.js + WebAssembly")
    print("  3. Mobile: ONNX Mobile Runtime (iOS/Android)")
    print("  4. Edge: TensorRT (NVIDIA), TVM, TensorFlow Lite")
    print()


if __name__ == "__main__":
    main()
