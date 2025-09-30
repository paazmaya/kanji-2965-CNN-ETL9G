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
            print("🔍 Using AdaptiveAvgPool2d(1) -> GlobalAveragePool in ONNX")
        elif pooling_type == "adaptive_max":
            self.global_pool = nn.AdaptiveMaxPool2d(
                1
            )  # Alternative: GlobalMaxPool in ONNX
            print("🔍 Using AdaptiveMaxPool2d(1) -> GlobalMaxPool in ONNX")
        elif pooling_type == "fixed_avg":
            self.global_pool = nn.AvgPool2d(
                kernel_size=4, stride=1, padding=0
            )  # Compatible: AveragePool in ONNX
            print("🔍 Using AvgPool2d(4) -> AveragePool in ONNX")
        elif pooling_type == "fixed_max":
            self.global_pool = nn.MaxPool2d(
                kernel_size=4, stride=1, padding=0
            )  # Compatible: MaxPool in ONNX
            print("🔍 Using MaxPool2d(4) -> MaxPool in ONNX")
        else:
            print(f"🔍 Using original pooling: {self.global_pool}")

        # Note: For the trained model, the input after conv4 is 4x4, so both adaptive and fixed pooling
        # with kernel_size=4 will produce 1x1 output, maintaining the same classifier input size (256)


def generate_output_filename(base_name, image_size, backend, suffix):
    """Generate consistent filename with configuration details."""
    return f"{base_name}_etl9g_{image_size}x{image_size}_3036classes_{backend}{suffix}"


def export_to_onnx(
    model_path,
    onnx_path,
    image_size,
    pooling_type="adaptive_avg",
    target_backend="tract",
):
    """Export PyTorch model to ONNX with backend-specific optimizations

    Auto-generates filename if onnx_path is None.

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

    # Generate default filename if not provided
    if onnx_path is None:
        onnx_path = generate_output_filename(
            "kanji_model", image_size, target_backend, ".onnx"
        )

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
        print("📋 Configuring for ORT-Tract backend (ONNX Runtime API)")
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
        print("📋 Configuring for strict compatibility mode")
    else:
        # Direct Tract: Use newer opset for better optimization
        opset_version = 12  # Newer opset for direct tract
        use_dynamo = True  # Modern export for better optimization
        print("📋 Configuring for direct Sonos Tract backend")

    # Initialize model architecture with proper pooling
    # ETL9G dataset has exactly 3,036 character classes (fixed)
    NUM_CLASSES = 3036
    model = LightweightKanjiNet(NUM_CLASSES, image_size, pooling_type)

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

    # Create enhanced character mapping for inference
    mapping_path = onnx_path.replace(".onnx", "_mapping.json")

    # Load the character mapping from dataset
    try:
        with open("dataset/character_mapping.json", "r", encoding="utf-8") as f:
            char_details = json.load(f)
        print("📚 Loaded character details from dataset")
    except FileNotFoundError:
        print("⚠️  Character mapping not found, using basic mapping")
        char_details = {}

    # Create comprehensive mapping
    mapping = create_enhanced_character_mapping(
        char_details,
        {
            "image_size": image_size,
            "pooling_type": pooling_type,
            "target_backend": target_backend,
            "opset_version": opset_version,
            "export_method": "torch.export" if use_dynamo else "TorchScript",
        },
    )

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"📄 Enhanced character mapping saved to {mapping_path}")

    return onnx_path


def create_enhanced_character_mapping(char_details, model_info):
    """Create enhanced character mapping for ETL9G dataset (3,036 classes)"""
    NUM_CLASSES = 3036

    def jis_to_unicode(jis_code):
        """Convert JIS X 0208 code to Unicode character"""
        try:
            # JIS X 0208 to Unicode conversion
            jis_int = int(jis_code, 16)

            # ETL9G uses JIS X 0208 encoding
            # Convert from JIS X 0208 area/code to Unicode

            # Extract area (high byte) and code (low byte)
            area = (jis_int >> 8) & 0xFF
            code = jis_int & 0xFF

            # Hiragana (area 24)
            if area == 0x24:
                # JIS X 0208 Hiragana starts at 24-21 (0x2421)
                # Unicode Hiragana starts at U+3041
                if 0x21 <= code <= 0x73:  # Valid Hiragana range
                    unicode_val = 0x3041 + (code - 0x21)
                    return chr(unicode_val)

            # Katakana (area 25)
            elif area == 0x25:
                # JIS X 0208 Katakana starts at 25-21 (0x2521)
                # Unicode Katakana starts at U+30A1
                if 0x21 <= code <= 0x76:  # Valid Katakana range
                    unicode_val = 0x30A1 + (code - 0x21)
                    return chr(unicode_val)

            # Hiragana (area 30 in some encodings)
            elif area == 0x30:
                # Alternative Hiragana encoding
                if 0x21 <= code <= 0x7E:
                    unicode_val = 0x3041 + (code - 0x21)
                    if unicode_val <= 0x3096:  # Valid Hiragana range
                        return chr(unicode_val)

            # Kanji Level 1 (areas 30-4F in JIS X 0208)
            elif 0x30 <= area <= 0x4F:
                # This is a complex mapping, using simplified approach
                # JIS Level 1 Kanji to Unicode CJK mapping
                jis_linear = ((area - 0x30) * 94) + (code - 0x21)
                unicode_val = 0x4E00 + jis_linear  # Start of CJK Unified Ideographs

                # Keep within reasonable CJK range
                if unicode_val > 0x9FAF:
                    unicode_val = 0x4E00 + (jis_linear % (0x9FAF - 0x4E00))

                return chr(unicode_val)

            # Fallback: return JIS code representation
            return f"[JIS:{jis_code}]"

        except (ValueError, OverflowError):
            return f"[JIS:{jis_code}]"

    def estimate_stroke_count(jis_code, character):
        """Estimate stroke count based on JIS code and character"""
        try:
            jis_int = int(jis_code, 16)
            area = (jis_int >> 8) & 0xFF
            code = jis_int & 0xFF

            # Hiragana typically have 1-4 strokes
            if area == 0x24 or area == 0x30:
                # Simple estimation based on position
                return min(4, max(1, (code - 0x21) % 4 + 1))

            # Katakana typically have 1-5 strokes
            elif area == 0x25:
                return min(5, max(1, (code - 0x21) % 5 + 1))

            # Kanji stroke count estimation
            elif 0x30 <= area <= 0x4F:
                # JIS ordering roughly follows stroke count/complexity
                jis_linear = ((area - 0x30) * 94) + (code - 0x21)

                # Rough stroke count estimation
                if jis_linear < 200:
                    return min(8, max(1, jis_linear // 25 + 1))  # 1-8 strokes
                elif jis_linear < 800:
                    return min(12, max(8, (jis_linear - 200) // 75 + 8))  # 8-12 strokes
                elif jis_linear < 1500:
                    return min(
                        16, max(12, (jis_linear - 800) // 100 + 12)
                    )  # 12-16 strokes
                else:
                    return min(
                        25, max(16, (jis_linear - 1500) // 150 + 16)
                    )  # 16+ strokes

            return 1  # Default fallback

        except ValueError:
            return 1

    # Create class-to-JIS mapping
    class_to_jis = {}
    jis_to_class = {}

    for jis_code, details in char_details.items():
        class_idx = details["class_idx"]
        class_to_jis[str(class_idx)] = jis_code
        jis_to_class[jis_code] = class_idx

    # Create comprehensive character mapping
    characters = {}

    for class_idx in range(NUM_CLASSES):
        class_str = str(class_idx)

        if class_str in class_to_jis:
            jis_code = class_to_jis[class_str]
            details = char_details[jis_code]

            # Get the actual character
            character = jis_to_unicode(jis_code)
            stroke_count = estimate_stroke_count(jis_code, character)

            characters[class_str] = {
                "jis_code": jis_code,
                "character": character,
                "reading": details.get("ascii_reading", ""),
                "stroke_count": stroke_count,
                "sample_count": details.get("sample_count", 0),
            }
        else:
            # Fallback for missing mappings
            characters[class_str] = {
                "jis_code": "unknown",
                "character": f"[{class_idx}]",
                "reading": "",
                "stroke_count": 1,
                "sample_count": 0,
            }

    # Complete mapping structure
    mapping = {
        "model_info": model_info,
        "num_classes": NUM_CLASSES,
        "class_to_jis": class_to_jis,
        "characters": characters,
        "statistics": {
            "total_characters": len(characters),
            "hiragana_count": len(
                [c for c in characters.values() if c["jis_code"].startswith("30")]
            ),
            "kanji_count": len(
                [
                    c
                    for c in characters.values()
                    if c["jis_code"].startswith(
                        (
                            "31",
                            "32",
                            "33",
                            "34",
                            "35",
                            "36",
                            "37",
                            "38",
                            "39",
                            "3A",
                            "3B",
                            "3C",
                            "3D",
                            "3E",
                            "3F",
                            "40",
                            "41",
                            "42",
                            "43",
                            "44",
                        )
                    )
                ]
            ),
            "average_stroke_count": sum(c["stroke_count"] for c in characters.values())
            / len(characters)
            if characters
            else 0,
        },
    }

    return mapping


def create_character_mapping(data_dir, image_size, accuracy):
    """Legacy function for backward compatibility - ETL9G has 3,036 classes"""
    NUM_CLASSES = 3036
    try:
        with open(f"{data_dir}/character_mapping.json", "r", encoding="utf-8") as f:
            char_details = json.load(f)

        model_info = {
            "image_size": image_size,
            "accuracy": accuracy,
            "pooling_type": "adaptive_avg",
            "target_backend": "tract",
            "opset_version": 12,
            "export_method": "TorchScript",
        }

        mapping = create_enhanced_character_mapping(char_details, model_info)

        with open("kanji_etl9g_mapping.json", "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

        print("📄 Enhanced character mapping created: kanji_etl9g_mapping.json")

    except FileNotFoundError:
        print("⚠️  Character mapping file not found in dataset directory")


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
        default=None,
        help="Output path for ONNX model (auto-generated if not specified)",
    )
    parser.add_argument(
        "--image-size", type=int, default=64, help="Image size used in training"
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
    if args.onnx_path is None:
        onnx_filename = generate_output_filename(
            "kanji_model",
            args.image_size,
            args.target_backend,
            ".onnx",
        )
        print(f"Output: {onnx_filename} (auto-generated)")
    else:
        onnx_filename = args.onnx_path
        print(f"Output: {onnx_filename}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print("Classes: 3036 (ETL9G dataset)")
    print(f"Pooling: {args.pooling_type}")
    print(f"Backend: {args.target_backend}")

    result = export_to_onnx(
        args.model_path,
        onnx_filename,
        args.image_size,
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
            f"  - {onnx_filename} ({Path(onnx_filename).stat().st_size / (1024 * 1024):.1f} MB)"
        )
        print(f"  - {onnx_filename.replace('.onnx', '_mapping.json')}")
    else:
        print("❌ ONNX conversion failed")


if __name__ == "__main__":
    main()
