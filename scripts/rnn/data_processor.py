"""
Data Processing for RNN-based Kanji Recognition

Handles conversion of kanji images to various sequential representations:
1. Stroke sequences (for stroke-based RNN)
2. Radical sequences (for radical-based RNN)
3. Spatial sequences (for hybrid models)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


class StrokeSequenceProcessor:
    """Processes kanji images to extract stroke sequences."""

    def __init__(self, max_strokes: int = 30, stroke_features: int = 8):
        self.max_strokes = max_strokes
        self.stroke_features = stroke_features
        self.stroke_extractor = None

    def extract_strokes_from_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract stroke sequences from a kanji image.

        This is a simplified implementation. In practice, you might want to use
        more sophisticated stroke extraction techniques.

        Args:
            image: Grayscale kanji image (H, W)

        Returns:
            List of stroke dictionaries with features
        """
        strokes = []

        # Simple contour-based stroke extraction
        # In practice, you'd use more sophisticated methods
        contours, _ = cv2.findContours(
            (255 - image).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for i, contour in enumerate(contours[: self.max_strokes]):
            if len(contour) < 3:  # Skip very small contours
                continue

            # Extract stroke features
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            # Centroid
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]

            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Stroke features: [cx, cy, x, y, w, h, area, perimeter]
            stroke_features = [
                cx / image.shape[1],  # Normalized centroid x
                cy / image.shape[0],  # Normalized centroid y
                x / image.shape[1],  # Normalized bbox x
                y / image.shape[0],  # Normalized bbox y
                w / image.shape[1],  # Normalized width
                h / image.shape[0],  # Normalized height
                cv2.contourArea(contour) / (image.shape[0] * image.shape[1]),  # Normalized area
                cv2.arcLength(contour, True)
                / (image.shape[0] + image.shape[1]),  # Normalized perimeter
            ]

            strokes.append({"features": stroke_features, "order": i, "contour": contour})

        return strokes

    def process_image_to_sequence(self, image: np.ndarray) -> Tuple[torch.Tensor, int]:
        """
        Convert kanji image to stroke sequence tensor.

        Args:
            image: Grayscale kanji image

        Returns:
            Tuple of (stroke_sequence_tensor, actual_stroke_count)
        """
        strokes = self.extract_strokes_from_image(image)

        # Create padded sequence tensor
        sequence = torch.zeros(self.max_strokes, self.stroke_features)
        actual_length = min(len(strokes), self.max_strokes)

        for i, stroke in enumerate(strokes[: self.max_strokes]):
            sequence[i] = torch.tensor(stroke["features"][: self.stroke_features])

        return sequence, actual_length

    def process_batch(self, images: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a batch of images to stroke sequences.

        Args:
            images: List of grayscale kanji images

        Returns:
            Tuple of (batch_sequences, batch_lengths)
        """
        batch_size = len(images)
        batch_sequences = torch.zeros(batch_size, self.max_strokes, self.stroke_features)
        batch_lengths = torch.zeros(batch_size, dtype=torch.long)

        for i, image in enumerate(images):
            sequence, length = self.process_image_to_sequence(image)
            batch_sequences[i] = sequence
            batch_lengths[i] = length

        return batch_sequences, batch_lengths


class RadicalSequenceProcessor:
    """Processes kanji characters to extract radical sequences."""

    def __init__(self, radical_vocab_path: Optional[Path] = None, max_radicals: int = 10):
        self.max_radicals = max_radicals
        self.radical_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_radical = {0: "<PAD>", 1: "<UNK>"}
        self.radical_decomposition = {}

        if radical_vocab_path and radical_vocab_path.exists():
            self.load_radical_vocab(radical_vocab_path)
        else:
            self._create_default_radical_vocab()

    def _create_default_radical_vocab(self):
        """Create a default radical vocabulary with common radicals."""
        # Common Japanese radicals (simplified set)
        common_radicals = [
            "人",
            "口",
            "日",
            "月",
            "木",
            "水",
            "火",
            "土",
            "金",
            "心",
            "手",
            "目",
            "耳",
            "足",
            "言",
            "糸",
            "竹",
            "米",
            "車",
            "食",
            "魚",
            "鳥",
            "馬",
            "豕",
            "犬",
            "虫",
            "艸",
            "彳",
            "亻",
            "氵",
            "扌",
            "忄",
            "礻",
            "衤",
            "訁",
            "辶",
            "廴",
            "广",
            "疒",
            "厂",
        ]

        for i, radical in enumerate(common_radicals, 2):  # Start from 2 (after PAD and UNK)
            self.radical_to_idx[radical] = i
            self.idx_to_radical[i] = radical

    def load_radical_vocab(self, vocab_path: Path):
        """Load radical vocabulary from file."""
        with open(vocab_path, encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.radical_to_idx = vocab_data["radical_to_idx"]
        self.idx_to_radical = {int(k): v for k, v in vocab_data["idx_to_radical"].items()}
        self.radical_decomposition = vocab_data.get("radical_decomposition", {})

    def save_radical_vocab(self, vocab_path: Path):
        """Save radical vocabulary to file."""
        vocab_data = {
            "radical_to_idx": self.radical_to_idx,
            "idx_to_radical": self.idx_to_radical,
            "radical_decomposition": self.radical_decomposition,
        }

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    def decompose_kanji(self, kanji_char: str) -> List[str]:
        """
        Decompose a kanji character into radicals.

        This is a simplified implementation. In practice, you'd use
        a comprehensive radical decomposition database.

        Args:
            kanji_char: Single kanji character

        Returns:
            List of radical characters
        """
        if kanji_char in self.radical_decomposition:
            return self.radical_decomposition[kanji_char]

        # Simplified decomposition (just return the character itself)
        # In practice, implement proper radical decomposition
        return [kanji_char] if kanji_char in self.radical_to_idx else ["<UNK>"]

    def kanji_to_radical_sequence(self, kanji_char: str) -> Tuple[torch.Tensor, int]:
        """
        Convert kanji character to radical sequence tensor.

        Args:
            kanji_char: Single kanji character

        Returns:
            Tuple of (radical_sequence_tensor, actual_radical_count)
        """
        radicals = self.decompose_kanji(kanji_char)

        # Convert radicals to indices
        radical_indices = []
        for radical in radicals[: self.max_radicals]:
            idx = self.radical_to_idx.get(radical, self.radical_to_idx["<UNK>"])
            radical_indices.append(idx)

        # Create padded sequence
        sequence = torch.zeros(self.max_radicals, dtype=torch.long)
        actual_length = len(radical_indices)

        for i, idx in enumerate(radical_indices):
            sequence[i] = idx

        return sequence, actual_length

    def process_batch(self, kanji_chars: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a batch of kanji characters to radical sequences.

        Args:
            kanji_chars: List of kanji characters

        Returns:
            Tuple of (batch_sequences, batch_lengths)
        """
        batch_size = len(kanji_chars)
        batch_sequences = torch.zeros(batch_size, self.max_radicals, dtype=torch.long)
        batch_lengths = torch.zeros(batch_size, dtype=torch.long)

        for i, kanji_char in enumerate(kanji_chars):
            sequence, length = self.kanji_to_radical_sequence(kanji_char)
            batch_sequences[i] = sequence
            batch_lengths[i] = length

        return batch_sequences, batch_lengths

    @property
    def vocab_size(self) -> int:
        """Return the size of the radical vocabulary."""
        return len(self.radical_to_idx)


class SpatialSequenceProcessor:
    """Processes kanji images into spatial sequences for hybrid models."""

    def __init__(self, grid_size: int = 8, feature_dim: int = 128):
        self.grid_size = grid_size
        self.feature_dim = feature_dim

    def image_to_spatial_sequence(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert kanji image to spatial sequence by dividing into grid.

        Args:
            image: Grayscale kanji image

        Returns:
            Spatial sequence tensor of shape (grid_size^2, feature_dim)
        """
        h, w = image.shape
        grid_h, grid_w = h // self.grid_size, w // self.grid_size

        spatial_features = []

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Extract grid cell
                start_h, end_h = i * grid_h, (i + 1) * grid_h
                start_w, end_w = j * grid_w, (j + 1) * grid_w
                cell = image[start_h:end_h, start_w:end_w]

                # Extract features from cell
                features = self._extract_cell_features(cell, i, j)
                spatial_features.append(features)

        return torch.tensor(spatial_features, dtype=torch.float32)

    def _extract_cell_features(self, cell: np.ndarray, row: int, col: int) -> List[float]:
        """Extract features from a single grid cell."""
        features = []

        # Position features
        features.extend([row / self.grid_size, col / self.grid_size])

        # Statistical features
        features.extend([np.mean(cell), np.std(cell), np.min(cell), np.max(cell), np.median(cell)])

        # Gradient features
        grad_x = np.gradient(cell, axis=1)
        grad_y = np.gradient(cell, axis=0)
        features.extend(
            [np.mean(np.abs(grad_x)), np.mean(np.abs(grad_y)), np.std(grad_x), np.std(grad_y)]
        )

        # Histogram features (simplified)
        hist, _ = np.histogram(cell.flatten(), bins=8, range=(0, 255))
        hist = hist / (hist.sum() + 1e-8)  # Normalize
        features.extend(hist.tolist())

        # Pad or truncate to feature_dim
        if len(features) < self.feature_dim:
            features.extend([0.0] * (self.feature_dim - len(features)))
        else:
            features = features[: self.feature_dim]

        return features


if __name__ == "__main__":
    # Test data processors
    print("Testing RNN data processors...")

    # Test stroke processor
    stroke_processor = StrokeSequenceProcessor()
    dummy_image = np.random.randint(0, 255, (64, 64)).astype(np.uint8)
    stroke_seq, stroke_len = stroke_processor.process_image_to_sequence(dummy_image)
    print(f"Stroke sequence shape: {stroke_seq.shape}, length: {stroke_len}")

    # Test radical processor
    radical_processor = RadicalSequenceProcessor()
    radical_seq, radical_len = radical_processor.kanji_to_radical_sequence("田")
    print(f"Radical sequence shape: {radical_seq.shape}, length: {radical_len}")
    print(f"Radical vocab size: {radical_processor.vocab_size}")

    # Test spatial processor
    spatial_processor = SpatialSequenceProcessor()
    spatial_seq = spatial_processor.image_to_spatial_sequence(dummy_image)
    print(f"Spatial sequence shape: {spatial_seq.shape}")
