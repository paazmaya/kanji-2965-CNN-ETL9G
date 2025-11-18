#!/usr/bin/env python3
"""
HierCode Training with Optional Hi-GITA Enhancement
=====================================================

Trains HierCode model with optional Hi-GITA improvements:
- Multi-granularity image encoding (strokes → radicals → character)
- Contrastive image-text alignment
- Fine-grained fusion modules

Features:
- Automatic checkpoint management with resume from latest checkpoint
- Dataset auto-detection (combined_all_etl, etl9g, etl8g, etl7, etl6, etl1)
- NVIDIA GPU required with CUDA optimizations enabled

Usage:
    python scripts/train_hiercode_higita.py --data-dir dataset --use-higita
    python scripts/train_hiercode_higita.py --data-dir dataset  # Standard HierCode

Author: Enhancement for kanji-2965-CNN-ETL9G
Date: November 17, 2025
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from checkpoint_manager import CheckpointManager, setup_checkpoint_arguments
from optimization_config import (
    HierCodeConfig,
    create_data_loaders,
    get_optimizer,
    get_scheduler,
    load_chunked_dataset,
    save_config,
    verify_and_setup_gpu,
import torch.nn as nn
import torch.optim as optim
from checkpoint_manager import CheckpointManager, setup_checkpoint_arguments
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from hiercode_higita_enhancement import (
    FineGrainedContrastiveLoss,
    HierCodeWithHiGITA,
    MultiGranularityTextEncoder,
)

logger = logging.getLogger(__name__)


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
        self.checkpoint_dir = "training/hiercode_higita/checkpoints"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def load_etl9g_dataset(data_dir: str, limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load ETLCDB dataset from preprocessed chunks.
    Auto-detects: combined_all_etl > etl9g > etl8g > etl7 > etl6 > etl1"""
    logger.info(f"Loading dataset from {data_dir}...")

    data_dir = Path(data_dir)

    # Priority order for dataset selection
    dataset_priority = [
        "combined_all_etl",
        "etl9g",
        "etl8g",
        "etl7",
        "etl6",
        "etl1",
    ]

    # Find the best available dataset
    selected_dataset = None
    for dataset_name in dataset_priority:
        dataset_subdir = data_dir / dataset_name
        if dataset_subdir.exists():
            chunk_files = sorted(dataset_subdir.glob(f"{dataset_name}_dataset_chunk_*.npz"))
            if not chunk_files:
                # Try without "_dataset" part
                chunk_files = sorted(dataset_subdir.glob(f"{dataset_name}_chunk_*.npz"))
            if chunk_files:
                selected_dataset = dataset_name
                logger.debug(f"🔍 Auto-detected dataset: {dataset_name}")
                break

    if selected_dataset is None:
        # Check legacy flat structure
        chunk_files = sorted(data_dir.glob("etl9g_dataset_chunk_*.npz"))
        if chunk_files:
            selected_dataset = "legacy"
            logger.debug("🔍 Auto-detected legacy dataset structure")
        else:
            raise FileNotFoundError(f"No chunk files found in {data_dir}")

    if selected_dataset == "legacy":
        chunk_files = sorted(data_dir.glob("etl9g_dataset_chunk_*.npz"))
    else:
        dataset_subdir = data_dir / selected_dataset
        chunk_files = sorted(dataset_subdir.glob(f"{selected_dataset}_dataset_chunk_*.npz"))
        if not chunk_files:
            chunk_files = sorted(dataset_subdir.glob(f"{selected_dataset}_chunk_*.npz"))

    images_list = []
    labels_list = []
    total_samples = 0

    for chunk_file in chunk_files:
        chunk = np.load(chunk_file)

        # Handle both key formats: (images, labels) or (X, y)
        if "images" in chunk:
            images = chunk["images"]  # (N, 64, 64)
            labels = chunk["labels"]  # (N,)
        else:
            images = chunk["X"]  # Flattened (N, 4096)
            labels = chunk["y"]  # (N,)
            # Reshape if flattened
            if images.ndim == 2 and images.shape[1] == 4096:
                images = images.reshape(-1, 64, 64)

        # Load subset if limit specified
        if limit and total_samples + len(images) > limit:
            remaining = limit - total_samples
            images = images[:remaining]
            labels = labels[:remaining]

        images_list.append(images)
        labels_list.append(labels)

        total_samples += len(images)
        logger.debug(f"  Loaded {chunk_file.name}: {len(images)} samples (total: {total_samples})")

        if limit and total_samples >= limit:
            break

    images = np.concatenate(images_list)
    labels = np.concatenate(labels_list)

    # Normalize to [0, 1]
    images = images.astype(np.float32) / 255.0

    # Add channel dimension if needed
    if len(images.shape) == 3:
        images = images[:, np.newaxis, :, :]  # (N, 1, 64, 64)

    logger.info(f"✅ Loaded {len(images)} images with {len(np.unique(labels))} classes")
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
            logger.info(
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
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")

    # Add checkpoint management arguments
    setup_checkpoint_arguments(parser, "hiercode_higita")

    args = parser.parse_args()

    # ========== VERIFY GPU ==========
    verify_and_setup_gpu()

    # Setup
    config = HiGITAConfig()
    config.use_higita = args.use_higita
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.epochs = args.epochs
    config.checkpoint_dir = args.checkpoint_dir

    device = "cuda"
    logger.info(f"🔧 Device: {device}")
    logger.info(f"🔧 Hi-GITA enhancement: {'✅ ENABLED' if args.use_higita else '❌ DISABLED'}")

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

    logger.info(f"✅ Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

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

    # ========== INITIALIZE CHECKPOINT MANAGER ==========
    checkpoint_manager = CheckpointManager(config.checkpoint_dir, "hiercode_higita")

    # Training loop
    best_val_acc = 0
    history = []

    # Resume from checkpoint using unified DRY method
    start_epoch, best_metrics = checkpoint_manager.load_checkpoint_for_training(
        model,
        optimizer,
        scheduler,
        device,
        resume_from=args.resume_from,
        args_no_checkpoint=args.no_checkpoint,
    )
    best_val_acc = best_metrics.get("val_accuracy", 0.0)
    start_epoch = max(start_epoch, 1)  # Epoch numbering starts at 1

    for epoch in range(start_epoch, config.epochs + 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Epoch {epoch}/{config.epochs}")
        logger.info(f"{'=' * 60}")

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

        logger.info(
            f"\n📊 Train - Loss: {train_metrics['total_loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}%"
        )
        logger.info(
            f"📋 Val   - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.2f}%"
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        # Save checkpoint after each epoch for resuming later
        checkpoint_manager.save_checkpoint(
            epoch, model, optimizer, scheduler, {"val_accuracy": val_metrics["accuracy"]}
        )

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_path = Path(config.checkpoint_dir) / "best_hiercode_higita.pth"
            torch.save(model.state_dict(), best_path)
            logger.info(f"💾 Best model saved to {best_path} (Acc: {best_val_acc:.2f}%)")

    # Save final history
    history_path = Path(config.checkpoint_dir) / "training_history_higita.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    logger.info(f"\n{'=' * 60}")
    logger.info("✅ Training complete!")
    logger.info(f"   Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"   Model saved to: {config.checkpoint_dir}")
    logger.info(f"   History saved to: {history_path}")


if __name__ == "__main__":
    main()
