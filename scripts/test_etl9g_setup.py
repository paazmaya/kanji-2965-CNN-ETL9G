#!/usr/bin/env python3
"""
Quick test script to verify ETL9G data preparation and training setup
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analyze_etl9g_data(data_dir):
    """Analyze prepared ETL9G dataset"""
    data_path = Path(data_dir)

    # Check if data exists
    if not data_path.exists():
        print(f"Error: Data directory {data_path} does not exist")
        return

    print("=== ETL9G Dataset Analysis ===")

    # Load metadata
    metadata_path = data_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        print(f"Classes: {metadata['num_classes']}")
        print(f"Total samples: {metadata['total_samples']}")
        print(f"Target size: {metadata['target_size']}x{metadata['target_size']}")
        print(f"Files processed: {metadata['dataset_info']['files_processed']}")
        print(f"Avg samples per class: {metadata['dataset_info']['avg_samples_per_class']:.1f}")

    # Check dataset files
    chunk_info_path = data_path / "chunk_info.json"
    if chunk_info_path.exists():
        with open(chunk_info_path) as f:
            chunk_info = json.load(f)
        print("\nDataset is chunked:")
        print(f"  Total samples: {chunk_info['total_samples']}")
        print(f"  Chunk size: {chunk_info['chunk_size']}")
        print(f"  Number of chunks: {chunk_info['num_chunks']}")

        # Verify all chunk files exist
        missing_chunks = []
        for i in range(chunk_info["num_chunks"]):
            chunk_file = data_path / f"etl9g_dataset_chunk_{i:02d}.npz"
            if not chunk_file.exists():
                missing_chunks.append(i)

        if missing_chunks:
            print(f"  Warning: Missing chunks: {missing_chunks}")
        else:
            print("  All chunk files present ✓")

    # Load a sample to verify data format
    try:
        # Try loading first chunk or main file
        chunk_file = data_path / "etl9g_dataset_chunk_00.npz"
        main_file = data_path / "etl9g_dataset.npz"

        if chunk_file.exists():
            data = np.load(chunk_file)
        elif main_file.exists():
            data = np.load(main_file)
        else:
            print("No dataset files found!")
            return

        X_sample = data["X"][:10]  # First 10 samples
        y_sample = data["y"][:10]

        print("\nSample verification:")
        print(f"  X shape: {X_sample.shape}")
        print(f"  y shape: {y_sample.shape}")
        print(f"  X data type: {X_sample.dtype}")
        print(f"  y data type: {y_sample.dtype}")
        print(f"  X range: [{X_sample.min():.3f}, {X_sample.max():.3f}]")
        print(f"  Sample classes: {np.unique(y_sample)}")

        # Show a sample image
        if len(X_sample) > 0:
            img_size = int(np.sqrt(X_sample.shape[1]))
            sample_img = X_sample[0].reshape(img_size, img_size)

            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(sample_img, cmap="gray")
            plt.title(f"Sample Image (Class {y_sample[0]})")
            plt.axis("off")

            # Show class distribution (sample)
            plt.subplot(1, 2, 2)
            class_counts = np.bincount(y_sample)
            plt.bar(range(len(class_counts)), class_counts)
            plt.title("Sample Class Distribution")
            plt.xlabel("Class Index")
            plt.ylabel("Count")

            plt.tight_layout()
            plt.savefig(data_path / "dataset_sample.png", dpi=150, bbox_inches="tight")
            plt.show()
            print(f"Sample visualization saved to: {data_path / 'dataset_sample.png'}")

    except Exception as e:
        print(f"Error loading sample data: {e}")

    # Character mapping analysis
    char_mapping_path = data_path / "character_mapping.json"
    if char_mapping_path.exists():
        with open(char_mapping_path, encoding="utf-8") as f:
            char_mapping = json.load(f)

        print(f"\nCharacter mapping available: {len(char_mapping)} entries")

        # Check for rice field kanji
        rice_field_jis = "4544"  # Rice field kanji JIS code
        if rice_field_jis in char_mapping:
            rice_info = char_mapping[rice_field_jis]
            print("Rice field kanji (田) found:")
            print(f"  JIS: {rice_field_jis}")
            print(f"  Class: {rice_info['class_idx']}")
            print(f"  Samples: {rice_info['sample_count']}")
        else:
            print("Rice field kanji (田) not found in mapping")

    print("\n=== Analysis Complete ===")


def test_model_architecture():
    """Test the model architecture without training"""
    try:
        import os
        import sys

        import torch

        # Add current directory to path to import model
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from train_cnn_model import LightweightKanjiNet

        print("\n=== Model Architecture Test ===")

        # Create model
        num_classes = 3036  # ETL9G classes
        image_size = 64
        model = LightweightKanjiNet(num_classes, image_size)

        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model created successfully:")
        print(f"  Input size: {image_size}x{image_size}")
        print(f"  Output classes: {num_classes}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Test forward pass
        batch_size = 4
        test_input = torch.randn(batch_size, image_size * image_size)

        model.eval()
        with torch.no_grad():
            output = model(test_input)

        print("Forward pass test:")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

        # Memory usage estimation
        model_size_mb = sum(p.numel() * 4 for p in model.parameters()) / (
            1024 * 1024
        )  # 4 bytes per float32
        print(f"Estimated model size: {model_size_mb:.1f} MB")

        print("Model architecture test passed ✓")

    except ImportError as e:
        print(f"PyTorch not available: {e}")
    except Exception as e:
        print(f"Model test failed: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test ETL9G dataset and training setup")
    parser.add_argument("--data-dir", default="dataset", help="Dataset directory to analyze")
    parser.add_argument("--test-model", action="store_true", help="Test model architecture")

    args = parser.parse_args()

    if args.data_dir:
        analyze_etl9g_data(args.data_dir)

    if args.test_model:
        test_model_architecture()


if __name__ == "__main__":
    main()
