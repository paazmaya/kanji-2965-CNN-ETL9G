"""
RNN Model Deployment and Inference

Provides tools for deploying trained RNN models and performing inference on new kanji images.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from data_processor import (
    RadicalSequenceProcessor,
    SpatialSequenceProcessor,
    StrokeSequenceProcessor,
)

# Import RNN components
from rnn_model import create_rnn_model


class KanjiRNNInference:
    """Inference engine for trained RNN kanji recognition models."""

    def __init__(self, model_path: Path, model_type: str, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        # Load model
        self.model, self.character_mapping = self._load_model(model_path)

        # Initialize data processors
        self.processors = self._initialize_processors()

        print(f"Loaded {model_type} model on {self.device}")
        print(f"Model supports {len(self.character_mapping)} characters")

    def _load_model(self, model_path: Path) -> Tuple[nn.Module, Dict[str, int]]:
        """Load trained model and character mapping."""
        checkpoint = torch.load(model_path, map_location=self.device)

        # Load character mapping
        character_mapping = checkpoint.get("character_mapping", {})
        num_classes = len(character_mapping)

        # Create model
        model_config = checkpoint.get("model_config", {})
        model = create_rnn_model(self.model_type, num_classes=num_classes, **model_config)

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        return model, character_mapping

    def _initialize_processors(self) -> Dict[str, Any]:
        """Initialize data processors based on model type."""
        processors = {}

        if self.model_type == "stroke_rnn":
            processors["stroke"] = StrokeSequenceProcessor()
        elif self.model_type == "radical_rnn":
            processors["radical"] = RadicalSequenceProcessor()
        elif self.model_type in ["basic_rnn", "hybrid_cnn_rnn"]:
            processors["spatial"] = SpatialSequenceProcessor()

        return processors

    def preprocess_image(self, image_path: Path) -> Dict[str, torch.Tensor]:
        """Preprocess image for model input."""
        # Load image
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        else:
            image = image_path  # Assume it's already a numpy array

        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Resize to standard size (64x64)
        image = cv2.resize(image, (64, 64))

        # Normalize
        image = image.astype(np.float32) / 255.0

        # Process based on model type
        if self.model_type == "stroke_rnn":
            stroke_sequence = self.processors["stroke"].extract_stroke_sequence(image)
            return {
                "stroke_sequences": torch.tensor(stroke_sequence, dtype=torch.float32).unsqueeze(0),
                "stroke_lengths": torch.tensor([len(stroke_sequence)], dtype=torch.long),
            }

        elif self.model_type == "radical_rnn":
            radical_sequence = self.processors["radical"].extract_radical_sequence(image)
            return {
                "radical_sequences": torch.tensor(radical_sequence, dtype=torch.float32).unsqueeze(
                    0
                ),
                "radical_lengths": torch.tensor([len(radical_sequence)], dtype=torch.long),
            }

        elif self.model_type == "basic_rnn":
            spatial_sequence = self.processors["spatial"].extract_spatial_sequence(image)
            return {"sequences": torch.tensor(spatial_sequence, dtype=torch.float32).unsqueeze(0)}

        elif self.model_type == "hybrid_cnn_rnn":
            # For hybrid model, use image directly
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            return {"images": image_tensor}

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def predict(self, image_path: Path, top_k: int = 5) -> List[Dict[str, Any]]:
        """Predict kanji character from image."""
        # Preprocess image
        inputs = self.preprocess_image(image_path)

        # Move to device
        for key, value in inputs.items():
            inputs[key] = value.to(self.device)

        # Inference
        start_time = time.time()
        with torch.no_grad():
            if self.model_type == "stroke_rnn":
                outputs = self.model(inputs["stroke_sequences"], inputs["stroke_lengths"])
            elif self.model_type == "radical_rnn":
                outputs = self.model(inputs["radical_sequences"], inputs["radical_lengths"])
            elif self.model_type == "basic_rnn":
                outputs = self.model(inputs["sequences"])
            elif self.model_type == "hybrid_cnn_rnn":
                outputs = self.model(inputs["images"])

        inference_time = time.time() - start_time

        # Get top-k predictions
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(
            probabilities, k=min(top_k, len(self.character_mapping)), dim=1
        )

        # Convert to results
        results = []
        char_to_idx = {v: k for k, v in self.character_mapping.items()}

        for i in range(top_probs.size(1)):
            prob = top_probs[0, i].item()
            idx = top_indices[0, i].item()
            character = char_to_idx.get(idx, f"UNKNOWN_{idx}")

            results.append(
                {
                    "character": character,
                    "probability": prob,
                    "confidence": prob * 100,
                    "rank": i + 1,
                }
            )

        return results, inference_time

    def predict_batch(
        self, image_paths: List[Path], top_k: int = 5
    ) -> List[Tuple[List[Dict], float]]:
        """Predict multiple images in batch."""
        results = []

        for image_path in image_paths:
            try:
                prediction, inference_time = self.predict(image_path, top_k)
                results.append((prediction, inference_time))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append(([], 0.0))

        return results


class ModelComparisonInference:
    """Compare predictions from multiple RNN models."""

    def __init__(self, model_configs: List[Dict[str, str]], device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}

        # Load all models
        for config in model_configs:
            model_name = config["name"]
            model_path = Path(config["path"])
            model_type = config["type"]

            try:
                self.models[model_name] = KanjiRNNInference(model_path, model_type, self.device)
                print(f"Loaded {model_name}")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")

    def compare_predictions(self, image_path: Path, top_k: int = 3) -> Dict[str, Any]:
        """Compare predictions from all loaded models."""
        results = {}

        for model_name, model in self.models.items():
            try:
                predictions, inference_time = model.predict(image_path, top_k)
                results[model_name] = {
                    "predictions": predictions,
                    "inference_time": inference_time,
                    "top_prediction": predictions[0] if predictions else None,
                }
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                results[model_name] = {
                    "predictions": [],
                    "inference_time": 0.0,
                    "top_prediction": None,
                    "error": str(e),
                }

        return results

    def consensus_prediction(self, image_path: Path, top_k: int = 3) -> Dict[str, Any]:
        """Generate consensus prediction from all models."""
        all_predictions = self.compare_predictions(image_path, top_k)

        # Aggregate predictions
        character_votes = {}
        total_confidence = 0
        valid_models = 0

        for model_name, result in all_predictions.items():
            if result["top_prediction"]:
                char = result["top_prediction"]["character"]
                conf = result["top_prediction"]["confidence"]

                if char not in character_votes:
                    character_votes[char] = {"votes": 0, "total_confidence": 0, "models": []}

                character_votes[char]["votes"] += 1
                character_votes[char]["total_confidence"] += conf
                character_votes[char]["models"].append(model_name)

                total_confidence += conf
                valid_models += 1

        # Find consensus
        if character_votes:
            best_char = max(
                character_votes.keys(),
                key=lambda x: (character_votes[x]["votes"], character_votes[x]["total_confidence"]),
            )

            consensus = {
                "character": best_char,
                "votes": character_votes[best_char]["votes"],
                "confidence": character_votes[best_char]["total_confidence"]
                / character_votes[best_char]["votes"],
                "agreement": character_votes[best_char]["votes"] / valid_models,
                "supporting_models": character_votes[best_char]["models"],
                "all_predictions": all_predictions,
            }
        else:
            consensus = {
                "character": None,
                "votes": 0,
                "confidence": 0,
                "agreement": 0,
                "supporting_models": [],
                "all_predictions": all_predictions,
            }

        return consensus


def main():
    parser = argparse.ArgumentParser(description="RNN Model Inference")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["basic_rnn", "stroke_rnn", "radical_rnn", "hybrid_cnn_rnn"],
        help="Type of RNN model",
    )
    parser.add_argument("--image", type=str, help="Path to image file for single prediction")
    parser.add_argument("--image-dir", type=str, help="Directory of images for batch prediction")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions to return")
    parser.add_argument("--compare-models", action="store_true", help="Compare multiple models")
    parser.add_argument("--model-config", type=str, help="JSON config file for model comparison")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.compare_models and args.model_config:
        # Model comparison mode
        with open(args.model_config) as f:
            model_configs = json.load(f)

        comparator = ModelComparisonInference(model_configs, device)

        if args.image:
            print(f"\nComparing models on: {args.image}")
            consensus = comparator.consensus_prediction(Path(args.image), args.top_k)

            print(f"\nConsensus Prediction: {consensus['character']}")
            print(f"Agreement: {consensus['agreement'] * 100:.1f}%")
            print(f"Confidence: {consensus['confidence']:.1f}%")
            print(f"Supporting models: {', '.join(consensus['supporting_models'])}")

            print("\nIndividual Model Predictions:")
            for model_name, result in consensus["all_predictions"].items():
                if result["top_prediction"]:
                    pred = result["top_prediction"]
                    print(
                        f"{model_name}: {pred['character']} ({pred['confidence']:.1f}%) "
                        f"[{result['inference_time'] * 1000:.1f}ms]"
                    )

    else:
        # Single model mode
        if not args.model_path or not args.model_type:
            print("Error: --model-path and --model-type are required for single model inference")
            return

        inference_engine = KanjiRNNInference(Path(args.model_path), args.model_type, device)

        if args.image:
            # Single image prediction
            print(f"Predicting: {args.image}")
            predictions, inference_time = inference_engine.predict(Path(args.image), args.top_k)

            print(f"\nTop {len(predictions)} predictions:")
            for pred in predictions:
                print(f"{pred['rank']}. {pred['character']} - {pred['confidence']:.1f}%")

            print(f"\nInference time: {inference_time * 1000:.1f}ms")

        elif args.image_dir:
            # Batch prediction
            image_dir = Path(args.image_dir)
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                image_files.extend(image_dir.glob(ext))

            if not image_files:
                print(f"No images found in {image_dir}")
                return

            print(f"Processing {len(image_files)} images...")
            results = inference_engine.predict_batch(image_files, args.top_k)

            total_time = sum(time for _, time in results)
            avg_time = total_time / len(results) if results else 0

            print("\nBatch Results:")
            print(f"Total time: {total_time * 1000:.1f}ms")
            print(f"Average time per image: {avg_time * 1000:.1f}ms")

            # Show sample predictions
            for i, (predictions, inference_time) in enumerate(results[:5]):
                if predictions:
                    print(
                        f"{image_files[i].name}: {predictions[0]['character']} ({predictions[0]['confidence']:.1f}%)"
                    )


if __name__ == "__main__":
    main()
