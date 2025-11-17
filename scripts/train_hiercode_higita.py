#!/usr/bin/env python3
"""
HierCode Training with Optional Hi-GITA Enhancement
=====================================================

Trains HierCode model with optional Hi-GITA improvements:
- Multi-granularity image encoding (strokes → radicals → character)
- Contrastive image-text alignment
- Fine-grained fusion modules

Usage:
    python scripts/train_hiercode_higita.py --data-dir dataset --use-higita
    python scripts/train_hiercode_higita.py --data-dir dataset  # Standard HierCode

Author: Enhancement for kanji-2965-CNN-ETL9G
Date: November 17, 2025
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from hiercode_higita_enhancement import (
    FineGrainedContrastiveLoss,
    HierCodeWithHiGITA,
    MultiGranularityTextEncoder,
)


class HiGITAConfig:
    """Hi-GITA enhancement configuration"""

    def __init__(self):
        self.use_higita = True
        self.stroke_dim = 128
        self.radical_dim = 256
        self.character_dim = 512
        self.num_radicals = 16

        # Contrastive learning
        self.contrastive_weight = 0.5  # Balance classification + contrastive
        self.temperature = 0.07

        # Text encoder
        self.num_strokes = 20
        self.num_hierarcical_radicals = 214

        # Training
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 30
        self.warmup_epochs = 2
        self.weight_decay = 1e-5

        # Checkpoint
        self.checkpoint_dir = "models/checkpoints_higita"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def load_etl9g_dataset(data_dir: str, limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load ETL9G dataset from preprocessed chunks"""
    print(f"Loading ETL9G dataset from {data_dir}...")

    data_dir = Path(data_dir)
    chunk_files = sorted(data_dir.glob("etl9g_dataset_chunk_*.npz"))

    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {data_dir}")

    images_list = []
    labels_list = []
    total_samples = 0

    for chunk_file in chunk_files:
        chunk = np.load(chunk_file)
        images = chunk["images"]  # (N, 64, 64)
        labels = chunk["labels"]  # (N,)

        # Load subset if limit specified
        if limit and total_samples + len(images) > limit:
            remaining = limit - total_samples
            images = images[:remaining]
            labels = labels[:remaining]

        images_list.append(images)
        labels_list.append(labels)

        total_samples += len(images)
        print(f"  Loaded {chunk_file.name}: {len(images)} samples (total: {total_samples})")

        if limit and total_samples >= limit:
            break

    images = np.concatenate(images_list)
    labels = np.concatenate(labels_list)

    # Normalize to [0, 1]
    images = images.astype(np.float32) / 255.0

    # Add channel dimension if needed
    if len(images.shape) == 3:
        images = images[:, np.newaxis, :, :]  # (N, 1, 64, 64)

    print(f"✅ Loaded {len(images)} images with {len(np.unique(labels))} classes")
    return images, labels


def create_synthetic_text_data(
    labels: np.ndarray, num_samples: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic text representations (stroke and radical codes)

    For real application, this would come from CJK radical decomposition database.
    Currently generates random representations for proof-of-concept.
    """
    if num_samples is None:
        num_samples = len(labels)

    # Synthetic stroke codes (each character has ~5-15 strokes)
    stroke_lengths = np.random.randint(5, 16, num_samples)
    max_strokes = max(stroke_lengths)
    stroke_codes = np.zeros((num_samples, max_strokes), dtype=np.int64)
    for i, length in enumerate(stroke_lengths):
        stroke_codes[i, :length] = np.random.randint(0, 20, length)

    # Synthetic radical codes (each character has ~1-6 radicals)
    radical_lengths = np.random.randint(1, 7, num_samples)
    max_radicals = max(radical_lengths)
    radical_codes = np.zeros((num_samples, max_radicals), dtype=np.int64)
    for i, length in enumerate(radical_lengths):
        radical_codes[i, :length] = np.random.randint(0, 214, length)

    return torch.from_numpy(stroke_codes), torch.from_numpy(radical_codes)


def train_epoch(
    model: HierCodeWithHiGITA,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    args: argparse.Namespace,
    config: HiGITAConfig,
    text_encoder: Optional[MultiGranularityTextEncoder] = None,
    contrastive_loss_fn: Optional[FineGrainedContrastiveLoss] = None,
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()

    total_loss = 0
    classification_loss = 0
    contrastive_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        output = model(images)
        logits = output["logits"]

        # Classification loss
        ce_loss = criterion(logits, labels)

        # Optionally add contrastive loss
        total_batch_loss = ce_loss
        if args.use_higita and text_encoder and contrastive_loss_fn:
            # Generate synthetic text for this batch
            stroke_codes, radical_codes = create_synthetic_text_data(
                labels.cpu().numpy(), len(labels)
            )
            stroke_codes = stroke_codes.to(device)
            radical_codes = radical_codes.to(device)

            # Get text encodings
            text_output = text_encoder(stroke_codes, radical_codes)

            # Compute contrastive loss
            losses_dict = contrastive_loss_fn(output["features"], text_output)
            con_loss = losses_dict["total_loss"]

            # Combined loss
            total_batch_loss = ce_loss + config.contrastive_weight * con_loss
            contrastive_loss += con_loss.item()

        # Backward pass
        optimizer.zero_grad()
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        total_loss += total_batch_loss.item()
        classification_loss += ce_loss.item()

        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Progress
        if (batch_idx + 1) % 10 == 0:
            batch_acc = 100 * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            print(
                f"  Epoch {epoch} [{batch_idx + 1:3d}/{len(train_loader):3d}] "
                f"Loss: {avg_loss:.4f} | Acc: {batch_acc:.2f}%"
            )

    return {
        "total_loss": total_loss / len(train_loader),
        "ce_loss": classification_loss / len(train_loader),
        "contrastive_loss": contrastive_loss / len(train_loader) if args.use_higita else 0,
        "accuracy": 100 * correct / total,
    }


def validate(
    model: HierCodeWithHiGITA,
    val_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Validate model"""
    model.eval()

    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            logits = output["logits"]

            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return {
        "loss": total_loss / len(val_loader),
        "accuracy": 100 * correct / total,
    }


def main():
    parser = argparse.ArgumentParser(description="Train HierCode with optional Hi-GITA enhancement")
    parser.add_argument("--data-dir", required=True, help="Dataset directory")
    parser.add_argument("--use-higita", action="store_true", help="Enable Hi-GITA enhancement")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--limit-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument(
        "--checkpoint-dir", default="models/checkpoints_higita", help="Checkpoint directory"
    )
    parser.add_argument("--resume-from", help="Resume from checkpoint")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    # Setup
    config = HiGITAConfig()
    config.use_higita = args.use_higita
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.epochs = args.epochs
    config.checkpoint_dir = args.checkpoint_dir

    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"🔧 Device: {device}")
    print(f"🔧 Hi-GITA enhancement: {'✅ ENABLED' if args.use_higita else '❌ DISABLED'}")

    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset
    images, labels = load_etl9g_dataset(args.data_dir, limit=args.limit_samples)

    # Split into train/val
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(images))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.1, stratify=labels, random_state=42
    )

    train_images = torch.from_numpy(images[train_indices])
    train_labels = torch.from_numpy(labels[train_indices]).long()
    val_images = torch.from_numpy(images[val_indices])
    val_labels = torch.from_numpy(labels[val_indices]).long()

    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    print(f"✅ Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # Create model
    num_classes = len(np.unique(labels))
    model = HierCodeWithHiGITA(
        num_classes=num_classes,
        use_higita_enhancement=args.use_higita,
        stroke_dim=config.stroke_dim,
        radical_dim=config.radical_dim,
        character_dim=config.character_dim,
    )
    model = model.to(device)

    # Optional: Load text encoder and contrastive loss for Hi-GITA training
    text_encoder = None
    contrastive_loss_fn = None
    if args.use_higita:
        text_encoder = MultiGranularityTextEncoder().to(device)
        contrastive_loss_fn = FineGrainedContrastiveLoss()

    # Optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    # Training loop
    best_val_acc = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{config.epochs}")
        print(f"{'=' * 60}")

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args,
            config,
            text_encoder,
            contrastive_loss_fn,
        )

        val_metrics = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"\n📊 Train - Loss: {train_metrics['total_loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}%"
        )
        print(f"📊 Val   - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.2f}%")

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        # Save checkpoint
        checkpoint_path = Path(config.checkpoint_dir) / f"checkpoint_higita_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history,
                "config": config.to_dict(),
            },
            checkpoint_path,
        )

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_path = Path(config.checkpoint_dir) / "best_hiercode_higita.pth"
            torch.save(model.state_dict(), best_path)
            print(f"💾 Best model saved to {best_path} (Acc: {best_val_acc:.2f}%)")

    # Save final history
    history_path = Path(config.checkpoint_dir) / "training_history_higita.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("✅ Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Model saved to: {config.checkpoint_dir}")
    print(f"   History saved to: {history_path}")


if __name__ == "__main__":
    main()
