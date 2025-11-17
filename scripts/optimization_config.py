"""
Unified configuration and utilities for all optimization approaches.
Centralizes common parameters and utilities to reduce code duplication.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ============================================================================
# SHARED CONFIGURATION
# ============================================================================


@dataclass
class OptimizationConfig:
    """
    Unified configuration for all optimization approaches.
    Each approach inherits from this and adds approach-specific parameters.
    """

    # ========== DATASET PARAMETERS ==========
    data_dir: str = "dataset"
    image_size: int = 64  # Input: 64x64 pixels
    num_classes: int = 3036  # Kanji classes

    # ========== TRAINING HYPERPARAMETERS ==========
    epochs: int = 30  # Complete passes through dataset
    batch_size: int = 64  # Samples per batch
    learning_rate: float = 0.001  # Initial learning rate
    weight_decay: float = 1e-5  # L2 regularization coefficient

    # ========== TRAIN/VAL/TEST SPLIT ==========
    val_split: float = 0.1  # 10% validation
    test_split: float = 0.1  # 10% test
    random_seed: int = 42  # For reproducibility

    # ========== DATA AUGMENTATION PARAMETERS ==========
    augment_enabled: bool = True  # Enable/disable augmentation
    augment_probability: float = 0.3  # 30% of samples augmented
    augment_noise_level: float = 0.05  # 5% Gaussian noise

    # ========== OPTIMIZATION ALGORITHM PARAMETERS ==========
    optimizer: str = "adamw"  # adamw or sgd
    scheduler: str = "cosine"  # cosine or step
    scheduler_T_max: int = 30  # For cosine annealing (usually = epochs)

    # ========== DEVICE & LOGGING ==========
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 100  # Batches between logs

    # ========== OUTPUT PATHS ==========
    model_dir: str = "models"
    results_dir: str = "results"

    def to_dict(self) -> Dict:
        """Convert config to dictionary for logging/saving"""
        return {
            "data_dir": self.data_dir,
            "image_size": self.image_size,
            "num_classes": self.num_classes,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }


@dataclass
class QATConfig(OptimizationConfig):
    """
    Configuration for Quantization-Aware Training (QAT).

    Key parameters:
    - qat_backend: Which quantization backend to use (fbgemm for CPU, qnnpack for mobile)
    - qat_bits: Bit width (8 for INT8 standard)
    - qat_epochs: Usually fewer than full training (fine-tuning phase)
    """

    # ========== QAT SPECIFIC PARAMETERS ==========
    qat_backend: str = "fbgemm"  # fbgemm (Intel CPU), qnnpack (mobile), x86 (server)
    qat_bits: int = 8  # INT8 quantization
    qat_calibration_batches: int = 32  # Batches for calibration phase
    qat_freeze_bn: bool = True  # Freeze batch norm statistics
    qat_start_epoch: int = 5  # Start QAT after warming up model

    # Fine-tuning specific
    qat_fine_tune_lr: float = 0.00001  # Reduced learning rate for QAT phase

    def to_dict(self) -> Dict:
        """Include QAT-specific params in config dict"""
        config = super().to_dict()
        config.update(
            {
                "qat_backend": self.qat_backend,
                "qat_bits": self.qat_bits,
                "qat_calibration_batches": self.qat_calibration_batches,
                "qat_freeze_bn": self.qat_freeze_bn,
                "qat_start_epoch": self.qat_start_epoch,
            }
        )
        return config


@dataclass
class RadicalRNNConfig(OptimizationConfig):
    """
    Configuration for Radical RNN / Radical Decomposition approach.

    Key parameters:
    - radical_vocab_size: Number of unique radicals
    - radical_embedding_dim: Dimension of radical embeddings
    - rnn_hidden_size: RNN hidden state dimension
    - rnn_num_layers: Number of RNN layers
    """

    # ========== RADICAL DECOMPOSITION PARAMETERS ==========
    radical_vocab_size: int = 500  # Estimated unique radicals in kanji
    radical_embedding_dim: int = 128  # Dimension for radical embeddings
    radical_encoding_type: str = "binary_tree"  # binary_tree, one_hot, or learned

    # ========== RNN PARAMETERS ==========
    rnn_type: str = "lstm"  # lstm or gru
    rnn_hidden_size: int = 256  # RNN hidden dimension
    rnn_num_layers: int = 2  # Number of stacked RNN layers
    rnn_dropout: float = 0.3  # Dropout in RNN

    # ========== CNN BACKBONE FOR RADICAL EXTRACTION ==========
    cnn_channels: Tuple[int, ...] = (32, 64, 128)  # Channel progression

    def to_dict(self) -> Dict:
        config = super().to_dict()
        config.update(
            {
                "radical_vocab_size": self.radical_vocab_size,
                "radical_embedding_dim": self.radical_embedding_dim,
                "rnn_type": self.rnn_type,
                "rnn_hidden_size": self.rnn_hidden_size,
                "rnn_num_layers": self.rnn_num_layers,
            }
        )
        return config


@dataclass
class HierCodeConfig(OptimizationConfig):
    """
    Configuration for HierCode: Hierarchical Codebook approach.
    Based on arXiv:2403.13761

    Key parameters:
    - codebook_size: Total number of codewords (hierarchical codebook)
    - codebook_dim: Dimension of each codeword
    - hierarch_depth: Depth of binary tree hierarchy
    - prototype_learning: Enable prototype learning
    """

    # ========== HIERCODE SPECIFIC PARAMETERS ==========
    # HierCode uses hierarchical binary tree encoding and multi-hot representation
    # NOTE: codebook_total_size MUST be <= 2^hierarch_depth (number of leaf nodes)

    codebook_total_size: int = 1024  # Total codebook entries (must be <= 2^hierarch_depth)
    codebook_dim: int = 128  # Dimension of codebook vectors
    hierarch_depth: int = 10  # Depth of binary tree (2^10 = 1024 leaf nodes)

    # Multi-hot encoding parameters
    multi_hot_k: int = 5  # Number of active codewords per character (multi-hot)
    temperature: float = 0.1  # Gumbel-softmax temperature

    # Feature extraction backbone
    backbone_type: str = "lightweight_cnn"  # lightweight_cnn or vit_small
    backbone_output_dim: int = 256  # Output dimension from backbone

    # Prototype learning
    enable_prototype_learning: bool = True
    prototype_learning_weight: float = 0.1  # Loss weight for prototype learning

    # Zero-shot learning parameters
    enable_zero_shot: bool = True
    zero_shot_radical_aware: bool = True  # Use radical decomposition for zero-shot

    def to_dict(self) -> Dict:
        config = super().to_dict()
        config.update(
            {
                "codebook_total_size": self.codebook_total_size,
                "codebook_dim": self.codebook_dim,
                "hierarch_depth": self.hierarch_depth,
                "multi_hot_k": self.multi_hot_k,
                "enable_prototype_learning": self.enable_prototype_learning,
                "enable_zero_shot": self.enable_zero_shot,
            }
        )
        return config


@dataclass
class ViTConfig(OptimizationConfig):
    """
    Configuration for Vision Transformer (ViT) approach.
    Uses T2T-ViT concepts for efficiency.

    Key parameters:
    - patch_size: Size of image patches
    - embedding_dim: Transformer embedding dimension
    - num_heads: Number of attention heads
    - num_layers: Number of transformer blocks
    """

    # ========== VISION TRANSFORMER PARAMETERS ==========
    # T2T-ViT: Tokens-to-Token ViT for efficiency

    patch_size: int = 8  # Divide 64x64 image into 8x8 patches = 64 tokens
    embedding_dim: int = 256  # Transformer embedding dimension
    num_heads: int = 8  # Multi-head attention heads
    num_transformer_layers: int = 12  # Number of transformer blocks
    mlp_dim: int = 1024  # MLP hidden dimension

    # T2T progressive tokenization
    use_tokens_to_tokens: bool = True  # Progressive tokenization for efficiency
    t2t_kernel_sizes: Tuple[int, ...] = (3, 3, 3)  # Progressive kernel sizes

    # Efficiency parameters
    dropout: float = 0.1
    attention_dropout: float = 0.0

    def to_dict(self) -> Dict:
        config = super().to_dict()
        config.update(
            {
                "patch_size": self.patch_size,
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "num_transformer_layers": self.num_transformer_layers,
                "use_tokens_to_tokens": self.use_tokens_to_tokens,
            }
        )
        return config


# ============================================================================
# SHARED DATASET CLASS
# ============================================================================


class ETL9GDataset(Dataset):
    """Efficient dataset for ETL9G with optional augmentation"""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        config: Optional[OptimizationConfig] = None,
    ):
        """
        Args:
            X: Image data (N, 64*64)
            y: Labels (N,)
            augment: Enable augmentation
            config: Configuration with augmentation parameters
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment

        # Use provided config or default
        if config is None:
            config = OptimizationConfig()
        self.config = config

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.X[idx]
        label = self.y[idx]

        if self.augment and torch.rand(1) < self.config.augment_probability:
            # Gaussian noise augmentation
            noise = torch.randn_like(image) * self.config.augment_noise_level
            image = torch.clamp(image + noise, 0, 1)

        return image, label


# ============================================================================
# SHARED UTILITY FUNCTIONS
# ============================================================================


def load_chunked_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from chunks if available, otherwise load single file.
    Auto-detects: combined_all_etl > etl9g > etl8g > etl7 > etl6 > etl1

    Args:
        data_dir: Directory containing dataset chunks

    Returns:
        X: Image data (N, 4096)
        y: Labels (N,)
    """
    data_path = Path(data_dir)

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
        chunk_info_path = data_path / dataset_name / "chunk_info.json"
        if chunk_info_path.exists():
            selected_dataset = dataset_name
            print(f"🔍 Auto-detected dataset: {dataset_name}")
            break

    if selected_dataset is None:
        # Check for legacy flat structure (backwards compatibility)
        chunk_info_path = data_path / "chunk_info.json"
        if chunk_info_path.exists():
            selected_dataset = "legacy"
            print("🔍 Auto-detected legacy dataset structure")
        else:
            raise FileNotFoundError(
                f"No dataset found in {data_path}. Available datasets should have chunk_info.json"
            )

    if selected_dataset == "legacy":
        # Legacy flat structure
        print("Loading legacy chunked dataset...")
        with open(data_path / "chunk_info.json") as f:
            chunk_info = json.load(f)

        all_X = []
        all_y = []

        for i in range(chunk_info["num_chunks"]):
            chunk_file = data_path / f"etl9g_dataset_chunk_{i:02d}.npz"
            if chunk_file.exists():
                chunk = np.load(chunk_file)
                all_X.append(chunk["X"])
                all_y.append(chunk["y"])
                print(f"  Loaded chunk {i + 1}/{chunk_info['num_chunks']}")

        X = np.concatenate(all_X, axis=0) if all_X else np.array([])
        y = np.concatenate(all_y, axis=0) if all_y else np.array([])
        print(f"✓ Total samples loaded: {len(X)}")
        return X, y
    else:
        # New directory structure
        dataset_dir = data_path / selected_dataset
        chunk_info_path = dataset_dir / "chunk_info.json"

        if chunk_info_path.exists():
            print(f"Loading {selected_dataset} dataset from chunks...")
            with open(chunk_info_path) as f:
                chunk_info = json.load(f)

            all_X = []
            all_y = []

            for i in range(chunk_info["num_chunks"]):
                chunk_file = dataset_dir / f"{selected_dataset}_chunk_{i:02d}.npz"
                if chunk_file.exists():
                    chunk = np.load(chunk_file)
                    all_X.append(chunk["X"])
                    all_y.append(chunk["y"])
                    print(f"  Loaded chunk {i + 1}/{chunk_info['num_chunks']}")
                else:
                    print(f"  ⚠️  Missing chunk {i:02d}, continuing...")

            X = np.concatenate(all_X, axis=0) if all_X else np.array([])
            y = np.concatenate(all_y, axis=0) if all_y else np.array([])
            print(f"✓ Total samples loaded: {len(X)}")
            return X, y
        else:
            # Try single file
            single_file = dataset_dir / f"{selected_dataset}_dataset.npz"
            if single_file.exists():
                print(f"Loading {selected_dataset} dataset from single file...")
                dataset = np.load(single_file)
                print(f"✓ Loaded {len(dataset['X'])} samples")
                return dataset["X"], dataset["y"]
            else:
                raise FileNotFoundError(
                    f"No valid dataset files found in {dataset_dir}"
                )


def create_data_loaders(
    X: np.ndarray, y: np.ndarray, config: OptimizationConfig, sample_limit: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data loaders with proper splitting.

    Args:
        X: Image data
        y: Labels
        config: Optimization configuration
        sample_limit: Optional limit on number of samples

    Returns:
        train_loader, val_loader, test_loader
    """
    from sklearn.model_selection import train_test_split

    if sample_limit:
        X = X[:sample_limit]
        y = y[:sample_limit]
        print(f"Using {sample_limit} samples")

    print(f"Dataset: {len(X)} samples, {len(np.unique(y))} classes")

    # Train/val/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.test_split, random_state=config.random_seed, stratify=y
    )

    val_size = config.val_split / (1.0 - config.test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=config.random_seed, stratify=y_temp
    )

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create datasets
    train_dataset = ETL9GDataset(X_train, y_train, augment=config.augment_enabled, config=config)
    val_dataset = ETL9GDataset(X_val, y_val, augment=False, config=config)
    test_dataset = ETL9GDataset(X_test, y_test, augment=False, config=config)

    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def save_config(config: OptimizationConfig, output_dir: str, name: str = "config.json"):
    """Save configuration to JSON for reproducibility"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config_path = Path(output_dir) / name
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"✓ Configuration saved to {config_path}")


def get_optimizer(model: nn.Module, config: OptimizationConfig):
    """
    Create optimizer based on config.

    Supports: adamw (default), sgd
    """
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    elif config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def get_scheduler(optimizer, config: OptimizationConfig):
    """
    Create learning rate scheduler based on config.

    Supports: cosine (default), step
    """
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.scheduler_T_max)
    elif config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")
