"""
Training Script for RNN-based Kanji Recognition Models

Supports training various RNN architectures:
1. Basic RNN
2. Stroke-based RNN
3. Radical-based RNN
4. Hybrid CNN-RNN
"""

import argparse
import json
import logging

# Import existing data utilities (assuming they exist in parent directory)
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from data_processor import (
    RadicalSequenceProcessor,
    SpatialSequenceProcessor,
    StrokeSequenceProcessor,
)

# Import our RNN models and data processors
from rnn_model import create_rnn_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from train_etl9g_model import load_chunked_dataset


class RNNKanjiDataset(Dataset):
    """Dataset class for RNN-based kanji recognition."""

    def __init__(
        self, data_dir: Path, model_type: str = "basic_rnn", image_size: int = 64, transform=None
    ):
        self.data_dir = Path(data_dir)
        self.model_type = model_type
        self.image_size = image_size
        self.transform = transform

        # Load metadata and character mapping
        with open(self.data_dir / "metadata.json", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.num_classes = self.metadata["num_classes"]
        self.class_to_jis = self.metadata["class_to_jis"]

        # Initialize data processors based on model type
        self.stroke_processor = None
        self.radical_processor = None
        self.spatial_processor = None

        if model_type == "stroke_rnn":
            self.stroke_processor = StrokeSequenceProcessor()
        elif model_type == "radical_rnn":
            self.radical_processor = RadicalSequenceProcessor()
        elif model_type in ["basic_rnn", "hybrid_cnn_rnn"]:
            self.spatial_processor = SpatialSequenceProcessor()

        # Load dataset
        print(f"Loading dataset from {self.data_dir}...")
        self.X, self.y = load_chunked_dataset(self.data_dir)
        print(f"Loaded {len(self.X)} samples with {self.num_classes} classes")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get image and label
        image = self.X[idx].reshape(self.image_size, self.image_size)  # Reshape from flattened
        label = self.y[idx]

        # Get character from class index
        character = self.class_to_jis.get(str(label), f"UNK_{label}")

        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)

        # Prepare data based on model type
        if self.model_type == "basic_rnn":
            # Convert to spatial sequence using grid-based processor
            if self.spatial_processor:
                sequence = self.spatial_processor.extract_spatial_sequence(image)
                return {
                    "sequences": torch.tensor(sequence, dtype=torch.float32).unsqueeze(0),
                    "labels": torch.tensor(label, dtype=torch.long),
                    "characters": character,
                }
            else:
                # Fallback: treat flattened image as sequence
                sequence = torch.tensor(image.flatten(), dtype=torch.float32)
                return {
                    "sequences": sequence.unsqueeze(0),
                    "labels": torch.tensor(label, dtype=torch.long),
                    "characters": character,
                }

        elif self.model_type == "stroke_rnn":
            stroke_sequence = self.stroke_processor.extract_stroke_sequence(image)
            return {
                "stroke_sequences": torch.tensor(stroke_sequence, dtype=torch.float32),
                "stroke_lengths": torch.tensor(len(stroke_sequence), dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.long),
                "characters": character,
            }

        elif self.model_type == "radical_rnn":
            radical_sequence = self.radical_processor.extract_radical_sequence(image)
            return {
                "radical_sequences": torch.tensor(radical_sequence, dtype=torch.float32),
                "radical_lengths": torch.tensor(len(radical_sequence), dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.long),
                "characters": character,
            }

        elif self.model_type == "hybrid_cnn_rnn":
            # For hybrid model, return image directly
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dim
            return {
                "images": image_tensor,
                "labels": torch.tensor(label, dtype=torch.long),
                "characters": character,
            }

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


def collate_fn_factory(model_type: str):
    """Factory function to create collate functions for different model types."""

    def basic_rnn_collate(batch):
        sequences = torch.stack([item["sequences"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        characters = [item["characters"] for item in batch]
        return {"sequences": sequences, "labels": labels, "characters": characters}

    def stroke_rnn_collate(batch):
        stroke_sequences = torch.stack([item["stroke_sequences"] for item in batch])
        stroke_lengths = torch.stack([item["stroke_lengths"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        characters = [item["characters"] for item in batch]
        return {
            "stroke_sequences": stroke_sequences,
            "stroke_lengths": stroke_lengths,
            "labels": labels,
            "characters": characters,
        }

    def radical_rnn_collate(batch):
        radical_sequences = torch.stack([item["radical_sequences"] for item in batch])
        radical_lengths = torch.stack([item["radical_lengths"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        characters = [item["characters"] for item in batch]
        return {
            "radical_sequences": radical_sequences,
            "radical_lengths": radical_lengths,
            "labels": labels,
            "characters": characters,
        }

    def hybrid_cnn_rnn_collate(batch):
        images = torch.stack([item["images"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        characters = [item["characters"] for item in batch]
        return {"images": images, "labels": labels, "characters": characters}

    collate_functions = {
        "basic_rnn": basic_rnn_collate,
        "stroke_rnn": stroke_rnn_collate,
        "radical_rnn": radical_rnn_collate,
        "hybrid_cnn_rnn": hybrid_cnn_rnn_collate,
    }

    return collate_functions[model_type]


class RNNTrainer:
    """Trainer class for RNN models."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        model_type: str,
        save_dir: Path = Path("models/rnn"),
    ):
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def train_epoch(
        self, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc="Training")

        for batch in progress_bar:
            optimizer.zero_grad()

            # Forward pass based on model type
            if self.model_type == "basic_rnn":
                outputs = self.model(batch["sequences"].to(self.device))
                labels = batch["labels"].to(self.device)

            elif self.model_type == "stroke_rnn":
                outputs = self.model(
                    batch["stroke_sequences"].to(self.device),
                    batch["stroke_lengths"].to(self.device),
                )
                labels = batch["labels"].to(self.device)

            elif self.model_type == "radical_rnn":
                outputs = self.model(
                    batch["radical_sequences"].to(self.device),
                    batch["radical_lengths"].to(self.device),
                )
                labels = batch["labels"].to(self.device)

            elif self.model_type == "hybrid_cnn_rnn":
                outputs = self.model(batch["images"].to(self.device))
                labels = batch["labels"].to(self.device)

            # Compute loss and backpropagate
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / num_batches

    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Forward pass based on model type
                if self.model_type == "basic_rnn":
                    outputs = self.model(batch["sequences"].to(self.device))
                    labels = batch["labels"].to(self.device)

                elif self.model_type == "stroke_rnn":
                    outputs = self.model(
                        batch["stroke_sequences"].to(self.device),
                        batch["stroke_lengths"].to(self.device),
                    )
                    labels = batch["labels"].to(self.device)

                elif self.model_type == "radical_rnn":
                    outputs = self.model(
                        batch["radical_sequences"].to(self.device),
                        batch["radical_lengths"].to(self.device),
                    )
                    labels = batch["labels"].to(self.device)

                elif self.model_type == "hybrid_cnn_rnn":
                    outputs = self.model(batch["images"].to(self.device))
                    labels = batch["labels"].to(self.device)

                # Compute loss and accuracy
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
    ):
        """Full training loop."""
        # Setup optimizer and criterion
        optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        criterion = nn.CrossEntropyLoss()

        self.logger.info(f"Starting training for {epochs} epochs...")
        self.logger.info(f"Model type: {self.model_type}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)

            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)

            # Update learning rate
            scheduler.step(val_acc)

            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(f"best_{self.model_type}_model.pth", epoch, val_acc)

            epoch_time = time.time() - start_time

            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"Time: {epoch_time:.1f}s"
            )

        # Save final model and training history
        self.save_model(f"final_{self.model_type}_model.pth", epochs, self.best_val_acc)
        self.save_training_history()
        self.plot_training_curves()

    def save_model(self, filename: str, epoch: int, val_acc: float):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_accuracy": val_acc,
            "model_type": self.model_type,
            "model_config": self.model.__dict__ if hasattr(self.model, "__dict__") else {},
        }

        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        self.logger.info(f"Model saved to {save_path}")

    def save_training_history(self):
        """Save training history."""
        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "best_val_acc": self.best_val_acc,
            "model_type": self.model_type,
        }

        history_path = self.save_dir / f"{self.model_type}_training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        self.logger.info(f"Training history saved to {history_path}")

    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curves
        ax1.plot(self.train_losses, label="Train Loss")
        ax1.plot(self.val_losses, label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"{self.model_type.upper()} - Loss Curves")
        ax1.legend()
        ax1.grid(True)

        # Accuracy curve
        ax2.plot(self.val_accuracies, label="Val Accuracy", color="green")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title(f"{self.model_type.upper()} - Validation Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        plot_path = self.save_dir / f"{self.model_type}_training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Training curves saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Train RNN-based Kanji Recognition Models")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Directory containing ETL9G dataset"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="hybrid_cnn_rnn",
        choices=["basic_rnn", "stroke_rnn", "radical_rnn", "hybrid_cnn_rnn"],
        help="Type of RNN model to train",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--hidden-size", type=int, default=256, help="RNN hidden size")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of RNN layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument(
        "--sample-limit", type=int, default=None, help="Limit number of samples for testing"
    )
    parser.add_argument(
        "--save-dir", type=str, default="models/rnn", help="Directory to save models"
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    print(f"Loading dataset from {args.data_dir}...")
    dataset = RNNKanjiDataset(data_dir=args.data_dir, model_type=args.model_type)

    # Limit samples if specified
    if args.sample_limit:
        dataset.X = dataset.X[: args.sample_limit]
        dataset.y = dataset.y[: args.sample_limit]
        print(f"Limited dataset to {len(dataset.X)} samples")

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    collate_fn = collate_fn_factory(args.model_type)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Disable multiprocessing to avoid pickle issues
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    model_kwargs = {
        "num_classes": dataset.num_classes,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }

    if args.model_type == "radical_rnn":
        model_kwargs["radical_vocab_size"] = dataset.radical_processor.vocab_size

    model = create_rnn_model(args.model_type, **model_kwargs)
    print(
        f"Created {args.model_type} model with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Create trainer and train
    trainer = RNNTrainer(model, device, args.model_type, Path(args.save_dir))
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"Training completed! Best validation accuracy: {trainer.best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
