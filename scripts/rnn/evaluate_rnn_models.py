"""
RNN Model Evaluation and Comparison

Provides tools for evaluating and comparing RNN models with the existing CNN model.
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

# Import RNN components
from train_rnn_model import RNNKanjiDataset, collate_fn_factory
from rnn_model import create_rnn_model

# Import existing CNN model for comparison
import sys

sys.path.append(str(Path(__file__).parent.parent))


class ModelEvaluator:
    """Evaluates and compares different kanji recognition models."""

    def __init__(self, device: torch.device):
        self.device = device
        self.results = {}

    def load_rnn_model(self, model_path: Path, model_type: str, num_classes: int) -> nn.Module:
        """Load a trained RNN model."""
        checkpoint = torch.load(model_path, map_location=self.device)

        # Create model with same config
        model_config = checkpoint.get("model_config", {})
        model = create_rnn_model(model_type, num_classes=num_classes, **model_config)

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        return model

    def evaluate_model(
        self, model: nn.Module, test_loader: DataLoader, model_type: str, model_name: str
    ) -> Dict[str, Any]:
        """Evaluate a model on test data."""
        model.eval()

        all_predictions = []
        all_labels = []
        all_characters = []
        inference_times = []

        with torch.no_grad():
            for batch in test_loader:
                start_time = time.time()

                # Forward pass based on model type
                if model_type == "basic_rnn":
                    outputs = model(batch["sequences"].to(self.device))
                elif model_type == "stroke_rnn":
                    outputs = model(
                        batch["stroke_sequences"].to(self.device),
                        batch["stroke_lengths"].to(self.device),
                    )
                elif model_type == "radical_rnn":
                    outputs = model(
                        batch["radical_sequences"].to(self.device),
                        batch["radical_lengths"].to(self.device),
                    )
                elif model_type == "hybrid_cnn_rnn":
                    outputs = model(batch["images"].to(self.device))
                else:
                    outputs = model(batch["images"].to(self.device))  # Default CNN

                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch["labels"].numpy())
                all_characters.extend(batch["characters"])

        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        avg_inference_time = np.mean(inference_times)

        # Top-k accuracy
        top5_correct = 0
        with torch.no_grad():
            for batch in test_loader:
                if model_type == "basic_rnn":
                    outputs = model(batch["sequences"].to(self.device))
                elif model_type == "stroke_rnn":
                    outputs = model(
                        batch["stroke_sequences"].to(self.device),
                        batch["stroke_lengths"].to(self.device),
                    )
                elif model_type == "radical_rnn":
                    outputs = model(
                        batch["radical_sequences"].to(self.device),
                        batch["radical_lengths"].to(self.device),
                    )
                elif model_type == "hybrid_cnn_rnn":
                    outputs = model(batch["images"].to(self.device))
                else:
                    outputs = model(batch["images"].to(self.device))

                _, top5_pred = outputs.topk(5, dim=1)
                labels = batch["labels"].to(self.device).unsqueeze(1)
                top5_correct += (top5_pred == labels).any(dim=1).sum().item()

        top5_accuracy = top5_correct / len(all_labels)

        results = {
            "model_name": model_name,
            "model_type": model_type,
            "accuracy": accuracy,
            "top5_accuracy": top5_accuracy,
            "avg_inference_time": avg_inference_time,
            "total_samples": len(all_labels),
            "predictions": all_predictions,
            "labels": all_labels,
            "characters": all_characters,
        }

        return results

    def compare_models(
        self, model_results: List[Dict[str, Any]], save_dir: Path = Path("evaluation_results")
    ):
        """Compare multiple models and generate reports."""
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create comparison table
        comparison_data = []
        for result in model_results:
            comparison_data.append(
                {
                    "Model": result["model_name"],
                    "Type": result["model_type"],
                    "Accuracy (%)": f"{result['accuracy'] * 100:.2f}",
                    "Top-5 Accuracy (%)": f"{result['top5_accuracy'] * 100:.2f}",
                    "Avg Inference Time (ms)": f"{result['avg_inference_time'] * 1000:.2f}",
                }
            )

        # Save comparison table
        comparison_df = pd.DataFrame(comparison_data) if "pandas" in globals() else None

        # Generate plots
        self._plot_accuracy_comparison(model_results, save_dir)
        self._plot_inference_time_comparison(model_results, save_dir)

        # Generate detailed reports for each model
        for result in model_results:
            self._generate_detailed_report(result, save_dir)

        return comparison_data

    def _plot_accuracy_comparison(self, model_results: List[Dict], save_dir: Path):
        """Plot accuracy comparison between models."""
        model_names = [r["model_name"] for r in model_results]
        accuracies = [r["accuracy"] * 100 for r in model_results]
        top5_accuracies = [r["top5_accuracy"] * 100 for r in model_results]

        x = np.arange(len(model_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, accuracies, width, label="Top-1 Accuracy", alpha=0.8)
        ax.bar(x + width / 2, top5_accuracies, width, label="Top-5 Accuracy", alpha=0.8)

        ax.set_xlabel("Models")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Model Accuracy Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / "accuracy_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_inference_time_comparison(self, model_results: List[Dict], save_dir: Path):
        """Plot inference time comparison between models."""
        model_names = [r["model_name"] for r in model_results]
        inference_times = [r["avg_inference_time"] * 1000 for r in model_results]  # Convert to ms

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(model_names, inference_times, alpha=0.7)

        # Color bars based on performance
        for i, bar in enumerate(bars):
            if inference_times[i] < 10:  # Fast
                bar.set_color("green")
            elif inference_times[i] < 50:  # Medium
                bar.set_color("orange")
            else:  # Slow
                bar.set_color("red")

        ax.set_xlabel("Models")
        ax.set_ylabel("Average Inference Time (ms)")
        ax.set_title("Model Inference Time Comparison")
        plt.xticks(rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / "inference_time_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_detailed_report(self, result: Dict[str, Any], save_dir: Path):
        """Generate detailed report for a single model."""
        model_name = result["model_name"].replace(" ", "_").lower()

        # Classification report
        y_true = result["labels"]
        y_pred = result["predictions"]

        # Save detailed metrics
        report_data = {
            "model_name": result["model_name"],
            "model_type": result["model_type"],
            "accuracy": result["accuracy"],
            "top5_accuracy": result["top5_accuracy"],
            "avg_inference_time_ms": result["avg_inference_time"] * 1000,
            "total_samples": result["total_samples"],
        }

        report_path = save_dir / f"{model_name}_detailed_report.json"
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate and Compare RNN Models")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Directory containing test dataset"
    )
    parser.add_argument(
        "--model-dir", type=str, default="models", help="Directory containing trained models"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument(
        "--sample-limit", type=int, default=None, help="Limit number of test samples"
    )
    parser.add_argument(
        "--save-dir", type=str, default="evaluation_results", help="Directory to save results"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Find all model files
    model_dir = Path(args.model_dir)
    rnn_model_dir = model_dir / "rnn"

    model_files = []
    if rnn_model_dir.exists():
        model_files.extend(list(rnn_model_dir.glob("best_*_model.pth")))

    if not model_files:
        print("No trained models found!")
        return

    # Create test dataset
    print("Loading test dataset...")

    evaluator = ModelEvaluator(device)
    all_results = []

    for model_file in model_files:
        # Extract model type from filename
        model_type = model_file.stem.replace("best_", "").replace("_model", "")

        print(f"\nEvaluating {model_type} model...")

        # Create dataset for this model type
        test_dataset = RNNKanjiDataset(data_dir=args.data_dir, model_type=model_type)

        if args.sample_limit:
            test_dataset.data = test_dataset.data[: args.sample_limit]

        # Create data loader
        collate_fn = collate_fn_factory(model_type)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
        )

        # Load and evaluate model
        try:
            model = evaluator.load_rnn_model(model_file, model_type, test_dataset.num_classes)
            result = evaluator.evaluate_model(
                model, test_loader, model_type, f"{model_type.upper()} Model"
            )
            all_results.append(result)

            print(f"Accuracy: {result['accuracy'] * 100:.2f}%")
            print(f"Top-5 Accuracy: {result['top5_accuracy'] * 100:.2f}%")
            print(f"Avg Inference Time: {result['avg_inference_time'] * 1000:.2f}ms")

        except Exception as e:
            print(f"Error evaluating {model_type}: {e}")

    if all_results:
        # Compare all models
        print("\nGenerating comparison reports...")
        save_dir = Path(args.save_dir)
        comparison_data = evaluator.compare_models(all_results, save_dir)

        print("\nModel Comparison Summary:")
        for data in comparison_data:
            print(
                f"{data['Model']}: {data['Accuracy (%)']}% accuracy, {data['Avg Inference Time (ms)']}ms inference"
            )

        print(f"\nDetailed results saved to {save_dir}")
    else:
        print("No models were successfully evaluated.")


if __name__ == "__main__":
    main()
