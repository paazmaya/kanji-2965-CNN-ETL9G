#!/usr/bin/env python3
"""
ONNX Conversion for ETL9G Kanji Recognition Model
Converts trained PyTorch model to ONNX format with backend-specific optimizations
"""

import torch
import torch.nn as nn
import argparse
import json
from pathlib import Path

# Import the correct model architecture from training
from train_etl9g_model import LightweightKanjiNet as OriginalLightweightKanjiNet

try:
    import onnxruntime

    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False


class LightweightKanjiNet(OriginalLightweightKanjiNet):
    """Extended LightweightKanjiNet with configurable pooling for different backends."""

    def __init__(
        self, num_classes: int, image_size: int = 64, pooling_type: str = "adaptive_avg"
    ):
        # Initialize the original model first
        super().__init__(num_classes, image_size)

        # Override the pooling layer based on target backend compatibility
        if pooling_type == "adaptive_avg":
            self.global_pool = nn.AdaptiveAvgPool2d(
                1
            )  # Original: GlobalAveragePool in ONNX
            print(f"🔍 Using AdaptiveAvgPool2d(1) -> GlobalAveragePool in ONNX")
        elif pooling_type == "adaptive_max":
            self.global_pool = nn.AdaptiveMaxPool2d(
                1
            )  # Alternative: GlobalMaxPool in ONNX
            print(f"🔍 Using AdaptiveMaxPool2d(1) -> GlobalMaxPool in ONNX")
        elif pooling_type == "fixed_avg":
            self.global_pool = nn.AvgPool2d(
                kernel_size=4, stride=1, padding=0
            )  # Compatible: AveragePool in ONNX
            print(f"🔍 Using AvgPool2d(4) -> AveragePool in ONNX")
        elif pooling_type == "fixed_max":
            self.global_pool = nn.MaxPool2d(
                kernel_size=4, stride=1, padding=0
            )  # Compatible: MaxPool in ONNX
            print(f"🔍 Using MaxPool2d(4) -> MaxPool in ONNX")
        else:
            print(f"🔍 Using original pooling: {self.global_pool}")

        # Note: For the trained model, the input after conv4 is 4x4, so both adaptive and fixed pooling
        # with kernel_size=4 will produce 1x1 output, maintaining the same classifier input size (256)


def export_to_onnx(
    model_path,
    onnx_path,
    image_size,
    num_classes,
    pooling_type="adaptive_avg",
    target_backend="tract",
):
    """Export PyTorch model to ONNX with backend-specific optimizations

    Backend Compatibility Analysis:
    ===============================

    Sonos Tract vs ORT-Tract (https://ort.pyke.io/backends/tract) Comparison:
    -----------------------------------
    1. **Sonos Tract (Direct)**:
       - Full Tract library with ~85% ONNX operator coverage
       - Supports opset versions 9-18
       - Direct access to all Tract optimizations
       - Designed for embedded/edge deployment

    2. **ORT-Tract (via ort crate)**:
       - ONNX Runtime API wrapper around Tract
       - Same underlying engine but through ort interface
       - Limited to ort-supported APIs (tensor operations only)
       - More constrained feature set for compatibility

    Model Architecture -> ONNX Operators -> Backend Support:
    -------------------------------------------------------
    ✅ **UNIVERSALLY SUPPORTED** (Both Tract & ORT-Tract):
    - Depthwise Convolution -> Conv (with groups) -> ✅ Full Support
    - Pointwise Convolution -> Conv (1x1) -> ✅ Full Support
    - Batch Normalization -> BatchNormalization -> ✅ Full Support
    - ReLU Activations -> Relu -> ✅ Full Support
    - Fixed Pooling -> AveragePool/MaxPool -> ✅ Full Support

    ⚠️  **TRACT ONLY** (Direct Tract, NOT ORT-Tract):
    - Global Pooling -> GlobalAveragePool/GlobalMaxPool -> ❌ ORT-Tract Unsupported

    Export Method Selection:
    -----------------------
    - `tract`: Uses dynamo=True (modern torch.export) + opset 12
    - `ort-tract`: Uses dynamo=False (legacy TorchScript) + opset 11
    - `strict`: Uses dynamo=False (legacy TorchScript) + opset 11

    The new dynamo=True export (PyTorch 2.9+) provides better optimization
    and support for advanced features, but we use it selectively for compatibility.
    =========================
    """

    # Validate inputs
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return None

    if not ONNXRUNTIME_AVAILABLE:
        print(
            "⚠️  ONNX/ONNXRuntime not available. Install with: pip install onnx onnxruntime"
        )
        return None

    print(f"Loading model from {model_path}...")

    # Choose export method and opset version based on target backend
    if target_backend == "ort-tract":
        # ORT-Tract: Use more conservative settings for API compatibility
        opset_version = 11  # More conservative for ort compatibility
        use_dynamo = False  # Conservative export for better compatibility
        # Override pooling for ORT-Tract compatibility - GlobalAveragePool not supported
        if pooling_type in ["adaptive_avg", "adaptive_max"]:
            original_pooling = pooling_type
            pooling_type = (
                "fixed_avg" if pooling_type == "adaptive_avg" else "fixed_max"
            )
            print(
                f"⚠️  ORT-Tract: Overriding {original_pooling} -> {pooling_type} (GlobalAveragePool unsupported)"
            )
        print(f"📋 Configuring for ORT-Tract backend (ONNX Runtime API)")
    elif target_backend == "strict":
        # Strict mode: Maximum compatibility - also avoid GlobalAveragePool
        opset_version = 11
        use_dynamo = False  # Legacy export for maximum compatibility
        if pooling_type in ["adaptive_avg", "adaptive_max"]:
            original_pooling = pooling_type
            pooling_type = (
                "fixed_avg" if pooling_type == "adaptive_avg" else "fixed_max"
            )
            print(
                f"⚠️  Strict mode: Overriding {original_pooling} -> {pooling_type} (Maximum compatibility)"
            )
        print(f"📋 Configuring for strict compatibility mode")
    else:
        # Direct Tract: Use newer opset for better optimization
        opset_version = 12  # Newer opset for direct tract
        use_dynamo = True  # Modern export for better optimization
        print(f"📋 Configuring for direct Sonos Tract backend")

    # Initialize model architecture with proper pooling
    model = LightweightKanjiNet(num_classes, image_size, pooling_type)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to("cpu")  # Explicitly move model to CPU
    model.eval()

    # Create dummy input for ONNX export
    # The trained model expects 2D image input (1, 1, 64, 64), not flattened
    dummy_input = torch.randn(1, 1, image_size, image_size)

    print("Exporting to ONNX format...")

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,  # Both backends benefit from constant folding ✅
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            if target_backend != "strict"
            else None,  # Disable dynamic axes for ultra-conservative mode
            dynamo=use_dynamo,
            verbose=False,
        )
        print(
            f"✅ Export successful using {'torch.export (dynamo)' if use_dynamo else 'TorchScript'} method"
        )
    except Exception as e:
        if use_dynamo:
            print(f"⚠️  New dynamo export failed: {e}")
            print("🔄 Falling back to legacy TorchScript export...")
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
                if target_backend != "strict"
                else None,
                verbose=False,
            )
            print("✅ Legacy TorchScript export completed")
        else:
            raise

    # Optimize ONNX model
    print("Optimizing ONNX model...")

    try:
        # Basic optimization using session options
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Test inference to ensure model works
        session = onnxruntime.InferenceSession(str(onnx_path), sess_options)

        # Get model info
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape

        # Test inference
        test_input = torch.randn(1, 1, image_size, image_size).numpy()
        outputs = session.run(None, {"input": test_input})

        print(f"✅ ONNX model exported successfully to {onnx_path}")
        print(f"📁 Model size: {Path(onnx_path).stat().st_size / (1024 * 1024):.1f} MB")
        print(f"Input shape: {input_shape}")
        print(f"Output shape: {output_shape}")

    except Exception as e:
        print(f"⚠️  ONNX validation warning: {e}")
        print("Model exported but validation failed. Check ONNX compatibility.")

    # Create character mapping for inference
    mapping_path = onnx_path.replace(".onnx", "_mapping.json")

    # Basic character mapping (this should ideally be loaded from training data)
    mapping = {
        "num_classes": num_classes,
        "image_size": image_size,
        "pooling_type": pooling_type,
        "target_backend": target_backend,
        "opset_version": opset_version,
        "export_method": "torch.export" if use_dynamo else "TorchScript",
    }

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"📄 Character mapping saved to {mapping_path}")

    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch kanji model to ONNX")
    parser.add_argument(
        "--model-path",
        type=str,
        default="best_kanji_model.pth",
        help="Path to the trained PyTorch model",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="kanji_etl9g_model.onnx",
        help="Output path for ONNX model",
    )
    parser.add_argument(
        "--image-size", type=int, default=64, help="Image size used in training"
    )
    parser.add_argument(
        "--num-classes", type=int, default=3036, help="Number of classes"
    )
    parser.add_argument(
        "--pooling-type",
        type=str,
        default="adaptive_avg",
        choices=["adaptive_avg", "adaptive_max", "fixed_avg", "fixed_max"],
        help="Pooling layer configuration",
    )
    parser.add_argument(
        "--target-backend",
        type=str,
        default="tract",
        choices=["tract", "ort-tract", "strict"],
        help="Target inference backend",
    )

    args = parser.parse_args()

    print("🔄 Starting ONNX conversion...")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.onnx_path}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Classes: {args.num_classes}")
    print(f"Pooling: {args.pooling_type}")
    print(f"Backend: {args.target_backend}")

    result = export_to_onnx(
        args.model_path,
        args.onnx_path,
        args.image_size,
        args.num_classes,
        args.pooling_type,
        args.target_backend,
    )

    if result:
        print("\n🎉 ONNX conversion completed!")
        print("Backend compatibility status:")
        if args.target_backend == "tract":
            print("📋 **Direct Tract** - Maximum performance")
            print("   ✅ Full Tract inference engine")
            print("   ✅ All pooling types supported")
            print("   ✅ Latest opset features")
            print("   📜 Modern torch.export for optimization")
        elif args.target_backend == "ort-tract":
            print("📋 **ORT-Tract** - ONNX Runtime API compatibility")
            print("   ✅ Same Tract engine with ort wrapper")
            print("   🔷 Tensor operations only (no sequences/maps)")
            print("   ✅ Good API compatibility")
            print("   ⚠️  Opset 11 for conservative compatibility")
            print("   📜 Legacy TorchScript export for compatibility")
        else:  # strict
            print("📋 **Strict** - Universal compatibility")
            print("   ✅ Maximum compatibility across engines")
            print("   ⚠️  Conservative feature set")
            print("   ⚠️  Basic pooling operations only")
            print("   ⚠️  Opset 11 for maximum compatibility")
            print("   📜 Legacy TorchScript export")

        print("Files ready for deployment:")
        print(
            f"  - {args.onnx_path} ({Path(args.onnx_path).stat().st_size / (1024 * 1024):.1f} MB)"
        )
        print(f"  - {args.onnx_path.replace('.onnx', '_mapping.json')}")
    else:
        print("❌ ONNX conversion failed")


if __name__ == "__main__":
    main()
