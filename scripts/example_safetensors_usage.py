#!/usr/bin/env python3
"""
Example: Using SafeTensors Kanji Model for Inference
Demonstrates how to load and use the SafeTensors model for kanji recognition
"""

import json

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

try:
    from train_cnn_model import LightweightKanjiNet
except ImportError:
    # Handle case when running from scripts directory
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent))
    from train_cnn_model import LightweightKanjiNet


def load_safetensors_model(
    safetensors_path="models/kanji_etl9g_model_64x64.safetensors",
    info_path="kanji_etl9g_model_64x64_info.json",
):
    """Load model from SafeTensors format."""
    print(f"📁 Loading SafeTensors model: {safetensors_path}")

    # Load model info
    with open(info_path) as f:
        model_info = json.load(f)

    num_classes = model_info["num_classes"]
    print(
        f"📊 Model info: {num_classes} classes, {model_info['metadata']['total_parameters']} parameters"
    )

    # Create model architecture
    model = LightweightKanjiNet(num_classes=num_classes)

    # Load weights from SafeTensors
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict)
    model.eval()

    print("✅ SafeTensors model loaded successfully")
    return model, model_info


def preprocess_image(image_path, target_size=64):
    """Preprocess image for model inference."""
    # Load and convert to grayscale
    image = Image.open(image_path).convert("L")

    # Resize to target size
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize
    image_array = np.array(image, dtype=np.float32) / 255.0

    # Normalize to [-1, 1] (same as training)
    image_array = (image_array - 0.5) / 0.5

    # Add batch and channel dimensions
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)

    return image_tensor


def predict_kanji(
    model,
    image_tensor,
    character_mapping_path="kanji_etl9g_mapping.json",
    top_k=5,
):
    """Predict kanji character from preprocessed image."""

    # Load character mapping
    with open(character_mapping_path, encoding="utf-8") as f:
        mapping_data = json.load(f)

    characters = mapping_data["characters"]

    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)

        predictions = []
        for i in range(top_k):
            class_idx = str(top_indices[0][i].item())
            confidence = top_probs[0][i].item()

            if class_idx in characters:
                char_info = characters[class_idx]
                predictions.append(
                    {
                        "character": char_info["character"],
                        "jis_code": char_info["jis_code"],
                        "stroke_count": char_info["stroke_count"],
                        "confidence": confidence,
                        "class_index": int(class_idx),
                    }
                )
            else:
                predictions.append(
                    {
                        "character": f"[Class {class_idx}]",
                        "jis_code": "unknown",
                        "stroke_count": 0,
                        "confidence": confidence,
                        "class_index": int(class_idx),
                    }
                )

    return predictions


def main():
    """Example usage of SafeTensors kanji model."""
    print("🚀 SafeTensors Kanji Recognition Example")
    print("=" * 50)

    try:
        # Load the SafeTensors model
        model, model_info = load_safetensors_model()

        print("\n📋 Model Metadata:")
        metadata = model_info["metadata"]
        print(f"   Dataset: {metadata['dataset']}")
        print(f"   Architecture: {metadata['architecture']}")
        total_params = int(metadata["total_parameters"])
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Model Size: {metadata['model_size_mb']} MB")
        print(f"   Training Epoch: {metadata['epoch']}")

        # Example with a synthetic image (since we don't have actual kanji images)
        print("\n🖼️  Creating synthetic test image...")
        test_image = torch.randn(1, 1, 64, 64)  # Random noise as example

        # Run prediction
        print("🔍 Running inference...")
        predictions = predict_kanji(model, test_image, top_k=3)

        print("\n🎯 Top 3 Predictions:")
        for i, pred in enumerate(predictions, 1):
            char = pred["character"]
            confidence = pred["confidence"] * 100
            strokes = pred["stroke_count"]
            jis = pred["jis_code"]
            print(f"   {i}. {char} ({confidence:.1f}% confidence, {strokes} strokes, JIS: {jis})")

        print("\n✅ SafeTensors inference example completed!")

        print("\n📂 **Integration Tips:**")
        print("   - Use safetensors.torch.load_file() for fast loading")
        print("   - Model weights are memory-mapped for efficiency")
        print("   - No pickle dependency - secure deserialization")
        print("   - Cross-platform compatibility guaranteed")
        print("   - Rich metadata embedded in the file")

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("💡 Make sure to run convert_to_safetensors.py first")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
