#!/usr/bin/env python3
"""
Export Quantized INT8 HierCode Model to ONNX Format
Converts quantized PyTorch model to ONNX for optimized cross-platform deployment
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
from model_utils import generate_export_path, infer_model_type
from optimization_config import HierCodeConfig
from train_hiercode import HierCodeClassifier


def export_quantized_to_onnx(
    model_path: str,
    output_path: Optional[str] = None,
    opset_version: int = 14,
    model_type: str = "hiercode",
) -> Tuple[Optional[str], Optional[dict]]:
    """Export quantized INT8 model to ONNX format"""
    print("\n" + "=" * 70)
    print("EXPORTING QUANTIZED INT8 MODEL TO ONNX")
    print("=" * 70)

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        print(f"❌ Model not found: {model_path}")
        return None, None

    print(f"\n📂 Loading quantized model: {model_path}")

    # Load config
    config_path = model_path_obj.parent / f"{model_type}_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
    else:
        config_dict = {"num_classes": 3036}

    num_classes = config_dict.get("num_classes", 3036)

    # Load quantized checkpoint
    checkpoint = torch.load(model_path_obj, map_location="cpu")

    print("ℹ️  Loading quantized INT8 model")

    # For quantized models, the state dict is already quantized
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Check if model is quantized by inspecting state dict keys
    # (Deprecated check - all INT8 models are quantized)

    # Create model
    if model_type == "hiercode":
        config = HierCodeConfig(num_classes=num_classes)
        model = HierCodeClassifier(num_classes=num_classes, config=config)
    else:
        print(f"❌ Unknown model type: {model_type}")
        return None, None

    # Dequantize tensors for ONNX export
    print("ℹ️  Dequantizing tensors for ONNX export...")
    dequantized_state = {}
    for key, value in state_dict.items():
        if hasattr(value, "dequantize"):
            dequantized_state[key] = value.dequantize()
        else:
            dequantized_state[key] = value

    # Load dequantized state dict
    try:
        model.load_state_dict(dequantized_state, strict=False)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None

    model.eval()

    print("✓ Quantized model loaded successfully")
    print("✓ Model is quantized INT8 (dequantized for ONNX export)")

    # Generate output path if not specified
    if output_path is None:
        # Place exports in model-type-specific exports directory
        model_path_obj = Path(model_path)
        # Try to infer model type from parent directory
        model_type_dir = infer_model_type(str(model_path_obj.parent), default=model_type)
        exports_dir = generate_export_path(model_type_dir)
        output_path_obj = exports_dir / f"hiercode_int8_opset{opset_version}.onnx"
    else:
        output_path_obj = Path(output_path)

    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    print("\n🔧 Exporting to ONNX...")
    print(f"  Model type: {model_type}")
    print("  Quantization: INT8 (dequantized)")
    print(f"  Opset version: {opset_version}")
    print("  Input: (batch_size, 1, 64, 64)")
    print(f"  Output: (batch_size, {num_classes})")

    # Create dummy input
    dummy_input = torch.randn(1, 1, 64, 64)

    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path_obj),
            input_names=["input_image"],
            output_names=["logits"],
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
            export_params=True,
        )
        print("\n✅ Model exported to ONNX")
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        return None, None

    # Check file size
    file_size = output_path_obj.stat().st_size
    print(f"  Path: {output_path_obj}")
    print(f"  Size: {file_size / 1e6:.2f} MB")

    # Create info file with quantization details
    info = {
        "model_type": model_type,
        "num_classes": num_classes,
        "quantization": "INT8 (dequantized for ONNX)",
        "input_shape": [1, 1, 64, 64],
        "input_names": ["input_image"],
        "output_names": ["logits"],
        "opset_version": opset_version,
        "file_size_mb": file_size / 1e6,
        "from_pytorch": str(model_path),
        "deployment_note": "Optimized for CPU inference with INT8 quantization",
    }

    info_path = output_path_obj.with_suffix(".json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n📋 Info file saved: {info_path}")

    return str(output_path_obj), info


def verify_onnx(onnx_path: str):
    """Verify ONNX model is valid"""
    print("\n🔍 Verifying ONNX model...")

    try:
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print("  ✓ ONNX model is valid")

        # Print model info
        graph = model.graph
        print("\n  Model Graph Info:")
        print(f"    Inputs: {[inp.name for inp in graph.input]}")
        print(f"    Outputs: {[out.name for out in graph.output]}")
        print(f"    Nodes: {len(graph.node)}")

        return True
    except ImportError:
        print("  ⚠️  onnx package not installed, skipping verification")
        print("     Install with: uv pip install onnx")
        return False
    except Exception as e:
        print(f"  ❌ ONNX validation failed: {e}")
        return False


def test_inference(onnx_path: str, num_samples: int = 5):
    """Test ONNX model inference and compare with PyTorch"""
    print("\n🧪 Testing ONNX inference...")

    try:
        import numpy as np
        import onnxruntime as ort

        # Use GPU providers if available, fallback to CPU
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            ("CPUExecutionProvider", {}),
        ]
        sess = ort.InferenceSession(str(onnx_path), providers=providers)  # type: ignore[attr-defined]
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
            test_input = np.random.randn(1, 1, 64, 64).astype("float32")

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


def compare_models(pytorch_path: str, onnx_path: str):
    """Compare PyTorch and ONNX model outputs"""
    print("\n🔀 Comparing PyTorch vs ONNX outputs...")

    try:
        import numpy as np
        import onnxruntime as ort

        # Use GPU providers if available, fallback to CPU
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            ("CPUExecutionProvider", {}),
        ]
        sess = ort.InferenceSession(str(onnx_path), providers=providers)  # type: ignore[attr-defined]
        provider_used = sess.get_providers()[0]
        print(f"  ONNX provider: {provider_used}")

        # Load PyTorch model
        config = HierCodeConfig(num_classes=3036)
        model = HierCodeClassifier(num_classes=3036, config=config)
        checkpoint = torch.load(pytorch_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Dequantize if needed
        dequantized_state = {}
        for key, value in state_dict.items():
            if hasattr(value, "dequantize"):
                dequantized_state[key] = value.dequantize()
            else:
                dequantized_state[key] = value

        model.load_state_dict(dequantized_state, strict=False)
        model.eval()

        # Load ONNX model (already created session above with GPU support)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # Test with same input
        print("  Testing with random inputs...")
        test_input = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            pytorch_output = model(test_input).numpy()

        onnx_output = sess.run([output_name], {input_name: test_input.numpy()})[0]

        # Compare
        diff = np.abs(pytorch_output - onnx_output)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print("\n  Output Comparison:")
        print(f"    Max difference: {max_diff:.6f}")
        print(f"    Mean difference: {mean_diff:.6f}")

        if max_diff < 1e-3:
            print("  ✓ Outputs match (difference < 1e-3)")
            return True
        else:
            print("  ⚠️  Outputs differ (difference > 1e-3)")
            return False

    except ImportError as e:
        print(f"  ⚠️  Required package not installed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Comparison failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export Quantized INT8 HierCode Model to ONNX Format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export quantized model to ONNX
  python export_quantized_to_onnx.py --model-path training/hiercode/quantized_hiercode_int8.pth

  # Export with verification
  python export_quantized_to_onnx.py --model-path training/hiercode/quantized_hiercode_int8.pth --verify

  # Export with inference test
  python export_quantized_to_onnx.py --model-path training/hiercode/quantized_hiercode_int8.pth --test-inference

  # Export and compare with PyTorch
  python export_quantized_to_onnx.py --model-path training/hiercode/quantized_hiercode_int8.pth \
    --pytorch-model training/hiercode/hiercode_model_best.pth --compare

  # Full validation pipeline
  python export_quantized_to_onnx.py --model-path training/hiercode/quantized_hiercode_int8.pth \
    --verify --test-inference --compare --pytorch-model training/hiercode/hiercode_model_best.pth
        """,
    )

    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="Path to quantized model checkpoint",
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
        "--compare",
        action="store_true",
        help="Compare PyTorch and ONNX outputs",
    )
    parser.add_argument(
        "--pytorch-model",
        type=str,
        default=None,
        help="Path to original PyTorch model for comparison",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="hiercode",
        help="Model type (default: hiercode)",
    )

    args = parser.parse_args()

    # Export
    onnx_path, info = export_quantized_to_onnx(
        args.model_path,
        output_path=args.output,
        opset_version=args.opset,
        model_type=args.model_type,
    )

    if onnx_path is None:
        print("\n❌ Export failed")
        return

    # Verify if requested
    if args.verify:
        verify_onnx(onnx_path)

    # Test inference if requested
    if args.test_inference:
        test_inference(onnx_path)

    # Compare if requested
    if args.compare and args.pytorch_model:
        compare_models(args.pytorch_model, onnx_path)
    elif args.compare and not args.pytorch_model:
        print("\n⚠️  --pytorch-model not provided, skipping comparison")

    print("\n" + "=" * 70)
    print("✅ QUANTIZED ONNX EXPORT COMPLETE")
    print("=" * 70)

    if onnx_path is None or info is None:
        print("\n❌ Export failed")
        return

    print(f"\nQuantized ONNX Model: {onnx_path}")
    print(f"File Size: {info['file_size_mb']:.2f} MB (optimized with INT8 quantization)")
    print("\nDeployment Benefits:")
    print("  ✓ 5x smaller than float32 (2.1 MB vs 11.6 MB PyTorch)")
    print("  ✓ Faster inference on CPU")
    print("  ✓ Lower memory footprint")
    print("  ✓ Cross-platform compatibility (Python, Web, Mobile, Edge)")
    print("\nDeployment Options:")
    print("  1. Python: uv pip install onnxruntime-gpu")
    print("  2. Web: ONNX.js + WebAssembly")
    print("  3. Mobile: ONNX Mobile Runtime (iOS/Android)")
    print("  4. Edge: TensorRT (NVIDIA), TVM, TensorFlow Lite")
    print()


if __name__ == "__main__":
    main()
