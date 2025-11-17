#!/usr/bin/env python3
"""
Lightweight Kanji Recognition Model for ETL9G Dataset
Optimized for ONNX/WASM deployment with 3,036 character classes

Features:
- Automatic checkpoint management with resume from latest checkpoint
- Dataset auto-detection (combined_all_etl, etl9g, etl8g, etl7, etl6, etl1)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from checkpoint_manager import CheckpointManager, setup_checkpoint_arguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ETL9GDataset(Dataset):
    """Efficient dataset for ETL9G with memory management"""

    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        if self.augment and torch.rand(1) < 0.3:
            # =========================
            # DATA AUGMENTATION ALGORITHMS - ADJUSTABLE
            # =========================
            # Current: Simple Gaussian noise augmentation
            # Alternatives: rotation, elastic deformation, stroke width variation, shearing
            # Probability: 0.3 (30% of training samples)
            # Noise level: 0.05 (5% of pixel intensity range)
            noise = torch.randn_like(image) * 0.05
            image = torch.clamp(image + noise, 0, 1)

            # Other augmentation options (commented out):
            # image = transforms.RandomRotation(degrees=15)(image.reshape(64, 64)).flatten()
            # image = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))(image.reshape(64, 64)).flatten()

        return image, label


class ChannelAttention(nn.Module):
    """SENet-style Channel Attention Module for feature recalibration"""

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # =========================
        # CHANNEL ATTENTION ALGORITHMS - SENet STYLE
        # =========================
        # Purpose: Adaptively recalibrate channel-wise feature responses
        # Process: Global Pool -> FC -> ReLU -> FC -> Sigmoid -> Scale
        # Reduction ratio: 16 (standard SENet configuration)

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze: H×W -> 1×1
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),  # Reduction layer
            nn.ReLU(inplace=True),  # Non-linearity
            nn.Linear(in_channels // reduction, in_channels, bias=False),  # Excitation layer
            nn.Sigmoid(),  # Gating function (0-1 scale per channel)
        )

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        b, c, _, _ = x.size()

        # Squeeze: Global spatial information into channel descriptor
        y = self.global_pool(x).view(b, c)  # (batch, channels)

        # Excitation: Generate channel-wise attention weights
        y = self.fc(y).view(b, c, 1, 1)  # (batch, channels, 1, 1)

        # Scale: Apply attention weights to input features
        return x * y.expand_as(x)  # Element-wise multiplication


class LightweightKanjiNet(nn.Module):
    """Lightweight CNN optimized for web deployment with Channel Attention"""

    def __init__(self, num_classes: int, image_size: int = 64):
        super(LightweightKanjiNet, self).__init__()

        self.image_size = image_size
        self.num_classes = num_classes

        # =========================
        # CNN ARCHITECTURE WITH CHANNEL ATTENTION
        # =========================
        # Current: Depthwise separable convolutions + SENet-style channel attention
        # Attention placement: After conv3, conv4, conv5 (deeper layers benefit most)
        # Channel progression: 1 -> 32 -> 64 -> 128 -> 256 -> 512 (increased capacity)
        # Stride pattern: 2 for downsampling layers, 1 for feature refinement
        self.conv1 = self._depthwise_separable_conv(1, 32, stride=2)  # 64x64 -> 32x32
        self.conv2 = self._depthwise_separable_conv(32, 64, stride=2)  # 32x32 -> 16x16
        self.conv3 = self._depthwise_separable_conv(64, 128, stride=2)  # 16x16 -> 8x8
        self.conv4 = self._depthwise_separable_conv(128, 256, stride=2)  # 8x8 -> 4x4
        self.conv5 = self._depthwise_separable_conv(
            256, 512, stride=1
        )  # 4x4 -> 4x4 (feature refinement)

        # =========================
        # CHANNEL ATTENTION MODULES - SENet INTEGRATION
        # =========================
        # Apply attention after deeper layers for maximum impact
        # Reduction ratio: 16 (balances effectiveness vs efficiency)
        self.attention3 = ChannelAttention(
            128, reduction=8
        )  # After conv3 (smaller reduction for fewer channels)
        self.attention4 = ChannelAttention(256, reduction=16)  # After conv4 (standard reduction)
        self.attention5 = ChannelAttention(512, reduction=16)  # After conv5 (standard reduction)

        # Alternative architectures (commented out):
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # Regular convolution
        # self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)  # Attention mechanism

        # =========================
        # POOLING ALGORITHM - ADJUSTABLE
        # =========================
        # Current: Global Average Pooling (reduces parameters vs large FC layers)
        # Alternatives: AdaptiveMaxPool2d, regular pooling + flatten
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # =========================
        # CLASSIFIER HEAD ALGORITHMS - ADJUSTABLE
        # =========================
        # Two-layer MLP with increased capacity for 512 input features
        # Hidden layer size: 1024 (larger intermediate representation for complex patterns)
        # Dropout rates: 0.3 and 0.2 (prevent overfitting with larger model)
        # Activation: ReLU (could use GELU, Swish, etc.)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # First dropout layer
            nn.Linear(512, 1024),  # Hidden layer (512->1024)
            nn.ReLU(inplace=True),  # Activation function
            nn.Dropout(0.2),  # Second dropout layer
            nn.Linear(1024, num_classes),  # Output layer
        )

        # Alternative classifier options (commented out):
        # Single layer: nn.Linear(512, num_classes)
        # Original smaller: nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, num_classes)
        # Larger hidden: nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, num_classes)
        # Different activation: nn.GELU(), nn.Swish()

        # Initialize weights
        self._initialize_weights()

    def _depthwise_separable_conv(self, in_channels, out_channels, stride=1):
        """Efficient depthwise separable convolution block"""
        # =========================
        # CONVOLUTION BLOCK ALGORITHMS - ADJUSTABLE
        # =========================
        # Current: Depthwise Separable Convolution (MobileNet-style)
        # Benefits: Fewer parameters, faster inference for mobile/web deployment
        # Components: Depthwise conv -> BatchNorm -> ReLU -> Pointwise conv -> BatchNorm -> ReLU
        # Kernel size: 3x3 for spatial features, 1x1 for channel mixing
        return nn.Sequential(
            # Depthwise convolution (spatial filtering)
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),  # groups=in_channels makes it depthwise
            nn.BatchNorm2d(in_channels),  # Normalization for training stability
            nn.ReLU(inplace=True),  # Activation function
            # Pointwise convolution (channel mixing)
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            ),  # 1x1 conv for channels
            nn.BatchNorm2d(out_channels),  # Normalization
            nn.ReLU(inplace=True),  # Activation function
        )

        # Alternative convolution blocks (commented out):
        # Regular convolution: nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # ResNet block: with skip connections
        # SENet block: with squeeze-and-excitation attention

    def _initialize_weights(self):
        """Initialize network weights"""
        # =========================
        # WEIGHT INITIALIZATION ALGORITHMS - ADJUSTABLE
        # =========================
        # Current: Kaiming (He) Normal initialization for conv layers
        # Purpose: Maintains variance in deeper networks with ReLU activations
        # Formula: N(0, √(2/fan_in)) for ReLU networks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )  # He initialization for ReLU
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # BatchNorm scale parameter
                nn.init.constant_(m.bias, 0)  # BatchNorm shift parameter
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # Small normal distribution for linear layers
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # Zero bias initialization

        # Alternative initialization methods (commented out):
        # Xavier/Glorot uniform: nn.init.xavier_uniform_(m.weight)
        # Xavier/Glorot normal: nn.init.xavier_normal_(m.weight)
        # Kaiming uniform: nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        # Orthogonal: nn.init.orthogonal_(m.weight)

    def forward(self, x):
        # Reshape flattened input to 2D image
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.image_size, self.image_size)

        # Feature extraction with Channel Attention
        x = self.conv1(x)  # 64x64 -> 32x32, channels: 1->32
        x = self.conv2(x)  # 32x32 -> 16x16, channels: 32->64

        x = self.conv3(x)  # 16x16 -> 8x8, channels: 64->128
        x = self.attention3(x)  # Apply channel attention to refine 128 features

        x = self.conv4(x)  # 8x8 -> 4x4, channels: 128->256
        x = self.attention4(x)  # Apply channel attention to refine 256 features

        x = self.conv5(x)  # 4x4 -> 4x4, channels: 256->512 (feature refinement)
        x = self.attention5(x)  # Apply channel attention to refine 512 features

        # Global pooling and classification
        x = self.global_pool(x)  # 4x4 -> 1x1
        x = x.view(batch_size, -1)  # Flatten to 512 features
        x = self.classifier(x)

        return x


class ProgressiveTrainer:
    """Progressive training strategy for large character sets"""

    def __init__(self, model, device, num_classes):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def train_epoch(self, dataloader, optimizer, criterion, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            # Update progress bar
            if batch_idx % 100 == 0:
                pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Acc": f"{100.0 * correct / total:.1f}%",
                    }
                )

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def validate(self, dataloader, criterion):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(dataloader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def train(
        self,
        train_loader,
        val_loader,
        epochs,
        learning_rate=0.001,
        checkpoint_manager=None,
        start_epoch=0,
    ):
        """
        Progressive training with learning rate scheduling and checkpoint management.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Total number of epochs to train
            learning_rate: Initial learning rate
            checkpoint_manager: Optional CheckpointManager for saving/resuming checkpoints
            start_epoch: Epoch to start training from (default: 0, use >0 when resuming)
        """

        # =========================
        # TRAINING LOSS FUNCTION - ADJUSTABLE
        # =========================
        # Current: CrossEntropyLoss with label smoothing = 0.1
        # Purpose: Label smoothing reduces overconfidence on large datasets (3,036 classes)
        # Effect: Smooths target distribution, prevents overfitting to training labels
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 0.1 = 10% smoothing

        # Alternative loss functions (commented out):
        # Standard cross-entropy: nn.CrossEntropyLoss()
        # Focal loss: FocalLoss(alpha=1, gamma=2) for class imbalance
        # Label smoothing values: 0.05 (light), 0.1 (moderate), 0.2 (heavy)

        # =========================
        # OPTIMIZER ALGORITHM - ADJUSTABLE
        # =========================
        # Current: AdamW with weight_decay=1e-4, lr=0.001
        # Purpose: Adam with decoupled weight decay, better generalization than Adam
        # Benefits: Adaptive learning rates per parameter + L2 regularization
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Alternative optimizers (commented out):
        # SGD with momentum: optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        # Standard Adam: optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        # RMSprop: optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-4)

        # =========================
        # LEARNING RATE SCHEDULER - ADJUSTABLE
        # =========================
        # Current: CosineAnnealingLR with T_max=epochs, eta_min=1e-6
        # Purpose: Smooth learning rate decay from initial LR to minimum LR
        # Pattern: Cosine curve, allows model to fine-tune in later epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        # Alternative schedulers (commented out):
        # Step decay: optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        # Exponential decay: optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        # Reduce on plateau: optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        # Warmup + cosine: Custom warmup for first few epochs then cosine annealing

        # =========================
        # EARLY STOPPING PARAMETERS - ADJUSTABLE
        # =========================
        best_val_acc = 0
        patience_counter = 0
        max_patience = 15  # Stop training if no improvement for 15 epochs

        # Alternative early stopping strategies:
        # Shorter patience: max_patience = 10 (faster stopping)
        # Longer patience: max_patience = 20 (more training)
        # Loss-based stopping: Track validation loss instead of accuracy

        # Create progress log
        progress_log = {
            "epochs": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }

        for epoch in range(start_epoch, epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Update learning rate
            scheduler.step()

            # Log progress
            progress_log["epochs"].append(epoch + 1)
            progress_log["train_loss"].append(train_loss)
            progress_log["train_acc"].append(train_acc)
            progress_log["val_loss"].append(val_loss)
            progress_log["val_acc"].append(val_acc)
            progress_log["learning_rate"].append(optimizer.param_groups[0]["lr"])

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save progress periodically
            if (epoch + 1) % 5 == 0 or epoch == 0:
                with open("models/training_progress.json", "w") as f:
                    json.dump(progress_log, f, indent=2)

            # Save checkpoint after each epoch if checkpoint manager is provided
            if checkpoint_manager:
                is_best = val_acc > best_val_acc
                checkpoint_manager.save_checkpoint(
                    epoch,
                    self.model,
                    optimizer,
                    scheduler,
                    metrics={
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    is_best=is_best,
                )
                # Clean up old checkpoints
                checkpoint_manager.cleanup_old_checkpoints(keep_last_n=5)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), "models/best_kanji_model.pth")
                patience_counter = 0
                print(f"New best model saved! Accuracy: {best_val_acc:.2f}%")

                # Save model info
                model_info = {
                    "epoch": epoch + 1,
                    "val_accuracy": best_val_acc,
                    "val_loss": val_loss,
                    "train_accuracy": train_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                with open("models/best_model_info.json", "w") as f:
                    json.dump(model_info, f, indent=2)
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")

        # Save final progress
        with open("models/training_progress.json", "w") as f:
            json.dump(progress_log, f, indent=2)

        return best_val_acc


def create_balanced_loaders(X, y, batch_size, test_size=0.15, val_size=0.15):
    """Create balanced data loaders with stratification"""

    print("Creating stratified splits...")

    # Check if we have enough samples per class for stratified splitting
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_samples = np.min(class_counts)

    print(f"Classes: {len(unique_classes)}, Min samples per class: {min_samples}")

    if min_samples < 2:
        print(
            "⚠️  Warning: Some classes have only 1 sample. Using non-stratified splitting for small datasets."
        )
        # Use simple random splitting when stratification isn't possible

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Second split: train vs val
        val_size_adj = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adj, random_state=42
        )
    else:
        # Use stratified splitting when we have enough samples
        print("✅ Using stratified splitting (recommended for balanced training)")

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # Second split: train vs val
        val_size_adj = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adj, stratify=y_temp, random_state=42
        )

    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Create datasets with augmentation for training
    train_dataset = ETL9GDataset(X_train, y_train, augment=True)
    val_dataset = ETL9GDataset(X_val, y_val, augment=False)
    test_dataset = ETL9GDataset(X_test, y_test, augment=False)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def load_chunked_dataset(data_dir):
    """Load dataset from chunks if available, otherwise load single file.
    Auto-detects: combined_all_etl > etl9g > etl8g > etl7 > etl6 > etl1"""
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
            else:
                print(f"Warning: Missing chunk file {chunk_file}")

        X = np.concatenate(all_X, axis=0) if all_X else np.array([])
        y = np.concatenate(all_y, axis=0) if all_y else np.array([])
        print(f"Total samples loaded: {len(X)}")
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
                    print(f"Warning: Missing chunk file {chunk_file}")

            X = np.concatenate(all_X, axis=0) if all_X else np.array([])
            y = np.concatenate(all_y, axis=0) if all_y else np.array([])
            print(f"Total samples loaded: {len(X)}")
            return X, y
        else:
            # Try single file
            single_file = dataset_dir / f"{selected_dataset}_dataset.npz"
            if single_file.exists():
                print(f"Loading {selected_dataset} dataset from single file...")
                dataset = np.load(single_file)
                print(f"Total samples loaded: {len(dataset['X'])}")
                return dataset["X"], dataset["y"]
            else:
                raise FileNotFoundError(f"No valid dataset files found in {dataset_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Lightweight Kanji Model for ETL9G")
    parser.add_argument("--data-dir", required=True, help="Directory containing ETL9G dataset")

    # =========================
    # TRAINING HYPERPARAMETERS - ADJUSTABLE
    # =========================
    # Current default values chosen for balanced training on ETL9G dataset

    # Epochs: Number of complete passes through dataset
    # Current: 30 (moderate training, prevents overfitting on large dataset)
    # Alternatives: 20 (faster), 50 (longer training), 100 (extensive)
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs (default: 30)")

    # Batch size: Number of samples processed together
    # Current: 64 (good balance for memory and convergence)
    # GPU memory dependent: 32 (low memory), 128 (high memory), 256 (very high memory)
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")

    # Learning rate: Step size for parameter updates
    # Current: 0.001 (moderate rate, works well with AdamW + cosine scheduling)
    # Alternatives: 0.0001 (conservative), 0.01 (aggressive), 0.005 (moderate-high)
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")

    # Image size: Input image dimensions (square)
    # Current: 64x64 (good balance for kanji detail and computational efficiency)
    # Alternatives: 32x32 (faster, less detail), 128x128 (slower, more detail)
    parser.add_argument("--image-size", type=int, default=64, help="Image size")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit samples for testing (e.g., 50000)",
    )

    # Add checkpoint management arguments
    setup_checkpoint_arguments(parser, "cnn")

    args = parser.parse_args()

    # Load dataset
    data_path = Path(args.data_dir)
    print("Loading dataset (auto-detecting best available)...")
    X, y = load_chunked_dataset(args.data_dir)

    with open(data_path / "metadata.json") as f:
        metadata = json.load(f)

    # ETL9G dataset has exactly 3,036 character classes (fixed)
    num_classes = 3036
    print(f"Dataset loaded: {X.shape}, {num_classes} classes")

    # Optional: Limit samples for faster testing/debugging
    if args.sample_limit and len(X) > args.sample_limit:
        print(f"Limiting dataset to {args.sample_limit} samples for testing...")
        indices = np.random.choice(len(X), args.sample_limit, replace=False)
        X = X[indices]
        y = y[indices]

    print(f"Dataset loaded: {X.shape}, {num_classes} classes")
    print(f"Memory usage: {X.nbytes / (1024**3):.1f} GB")

    # Create data loaders
    train_loader, val_loader, test_loader = create_balanced_loaders(X, y, args.batch_size)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LightweightKanjiNet(num_classes, args.image_size)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create models directory if it doesn't exist
    Path("models").mkdir(exist_ok=True)

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(args.checkpoint_dir, "cnn")

    # Initialize trainer
    trainer = ProgressiveTrainer(model, device, num_classes)

    # Check for existing checkpoint and resume if available
    start_epoch = 0
    best_val_acc = 0
    checkpoint_manager = CheckpointManager(args.checkpoint_dir, "cnn")

    # Use unified checkpoint loading (DRY pattern)
    start_epoch, best_metrics = checkpoint_manager.load_checkpoint_for_training(
        model,
        optimizer,
        scheduler,
        device,
        resume_from=args.resume_from,
        args_no_checkpoint=args.no_checkpoint,
    )
    best_val_acc = best_metrics.get("val_accuracy", 0.0)

    # Train model
    print("\nStarting training...")
    trainer.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_manager=checkpoint_manager,
        start_epoch=start_epoch,
    )

    # Test final model
    test_loss, test_acc = trainer.validate(test_loader, nn.CrossEntropyLoss())
    print(f"\nFinal test accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
