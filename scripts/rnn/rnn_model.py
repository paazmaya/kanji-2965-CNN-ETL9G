"""
RNN Model Architectures for Kanji Recognition

Implements various RNN-based approaches:
1. Basic RNN for sequence classification
2. Stroke-based RNN for temporal stroke processing
3. Radical-based RNN for structural decomposition
4. Hybrid CNN-RNN for spatial-temporal features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KanjiRNN(nn.Module):
    """Base RNN model for kanji recognition."""

    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 3036,
        rnn_type: str = "LSTM",
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super(KanjiRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # RNN layer
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        # Output dimensions after RNN
        rnn_output_size = hidden_size * (2 if bidirectional else 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # RNN forward pass
        rnn_out, _ = self.rnn(x)

        # Use the last output (or mean pooling)
        # For bidirectional, we get outputs from both directions
        if self.bidirectional:
            # Mean pooling over sequence length
            pooled = torch.mean(rnn_out, dim=1)
        else:
            # Take last output
            pooled = rnn_out[:, -1, :]

        # Classification
        output = self.classifier(pooled)
        return output


class StrokeBasedRNN(nn.Module):
    """RNN that processes kanji as sequences of strokes."""

    def __init__(
        self,
        stroke_features: int = 8,  # x, y, dx, dy, pressure, timestamp, etc.
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 3036,
        max_strokes: int = 30,
        dropout: float = 0.3,
    ):
        super(StrokeBasedRNN, self).__init__()

        self.stroke_features = stroke_features
        self.hidden_size = hidden_size
        self.max_strokes = max_strokes

        # Stroke embedding
        self.stroke_embedding = nn.Linear(stroke_features, hidden_size // 2)

        # Bi-directional LSTM for stroke sequences
        self.stroke_lstm = nn.LSTM(
            input_size=hidden_size // 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )

        # Attention mechanism for stroke importance
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, num_heads=8, dropout=dropout, batch_first=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, stroke_sequences: torch.Tensor, stroke_lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for stroke-based recognition

        Args:
            stroke_sequences: (batch_size, max_strokes, stroke_features)
            stroke_lengths: (batch_size,) actual number of strokes per sample

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size = stroke_sequences.size(0)

        # Embed stroke features
        embedded_strokes = self.stroke_embedding(stroke_sequences)
        embedded_strokes = F.relu(embedded_strokes)

        # Pack padded sequences for efficient RNN processing
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embedded_strokes, stroke_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward pass
        packed_output, _ = self.stroke_lstm(packed_input)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Self-attention over stroke sequence
        attended_output, _ = self.attention(lstm_output, lstm_output, lstm_output)

        # Global max pooling over stroke dimension
        pooled_output = torch.max(attended_output, dim=1)[0]

        # Classification
        output = self.classifier(pooled_output)
        return output


class RadicalRNN(nn.Module):
    """RNN that processes kanji as sequences of radicals."""

    def __init__(
        self,
        radical_vocab_size: int = 500,  # Number of unique radicals
        radical_embed_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 3036,
        max_radicals: int = 10,
        dropout: float = 0.3,
    ):
        super(RadicalRNN, self).__init__()

        self.radical_vocab_size = radical_vocab_size
        self.radical_embed_dim = radical_embed_dim
        self.hidden_size = hidden_size
        self.max_radicals = max_radicals

        # Radical embedding layer
        self.radical_embedding = nn.Embedding(
            num_embeddings=radical_vocab_size, embedding_dim=radical_embed_dim, padding_idx=0
        )

        # Bi-directional LSTM for radical sequences
        self.radical_lstm = nn.LSTM(
            input_size=radical_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(
        self, radical_sequences: torch.Tensor, radical_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for radical-based recognition

        Args:
            radical_sequences: (batch_size, max_radicals) radical indices
            radical_lengths: (batch_size,) actual number of radicals per sample

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Embed radical sequences
        embedded_radicals = self.radical_embedding(radical_sequences)

        # Pack padded sequences
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embedded_radicals, radical_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward pass
        packed_output, _ = self.radical_lstm(packed_input)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Use mean pooling over radical sequence
        # Create mask for padding positions
        batch_size, max_len = radical_sequences.size()
        mask = torch.arange(max_len).expand(batch_size, max_len).to(radical_sequences.device)
        mask = mask < radical_lengths.unsqueeze(1)

        # Apply mask and compute mean
        masked_output = lstm_output * mask.unsqueeze(-1).float()
        lengths_expanded = radical_lengths.unsqueeze(-1).unsqueeze(-1).float()
        pooled_output = masked_output.sum(dim=1) / lengths_expanded

        # Classification
        output = self.classifier(pooled_output)
        return output


class HybridCNNRNN(nn.Module):
    """Hybrid model combining CNN spatial features with RNN temporal processing."""

    def __init__(
        self,
        image_size: int = 64,
        cnn_features: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 3036,
        dropout: float = 0.3,
    ):
        super(HybridCNNRNN, self).__init__()

        self.image_size = image_size
        self.cnn_features = cnn_features
        self.hidden_size = hidden_size

        # CNN feature extractor (simplified ResNet-style)
        self.cnn_backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> 4x4
        )

        # Feature projection
        self.feature_projection = nn.Linear(256 * 4 * 4, cnn_features)

        # RNN for processing spatial features as sequences
        # We can treat spatial regions as sequence elements
        self.spatial_rnn = nn.LSTM(
            input_size=256,  # Features per spatial location
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )

        # Classification head combining both CNN and RNN features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(cnn_features + hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for hybrid CNN-RNN

        Args:
            x: Input tensor of shape (batch_size, 1, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # CNN feature extraction
        cnn_features = self.cnn_backbone(x)  # (batch_size, 256, 4, 4)

        # Global CNN features
        global_features = cnn_features.view(batch_size, -1)  # (batch_size, 256*4*4)
        global_features = self.feature_projection(global_features)  # (batch_size, cnn_features)

        # Prepare spatial features for RNN
        # Treat each spatial location as a sequence element
        spatial_features = cnn_features.view(batch_size, 256, -1)  # (batch_size, 256, 16)
        spatial_features = spatial_features.transpose(1, 2)  # (batch_size, 16, 256)

        # RNN processing of spatial sequence
        rnn_output, _ = self.spatial_rnn(spatial_features)

        # Pool RNN output
        rnn_pooled = torch.mean(rnn_output, dim=1)  # (batch_size, hidden_size*2)

        # Combine CNN and RNN features
        combined_features = torch.cat([global_features, rnn_pooled], dim=1)

        # Classification
        output = self.classifier(combined_features)
        return output


def create_rnn_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create RNN models

    Args:
        model_type: Type of RNN model to create
        **kwargs: Model-specific parameters

    Returns:
        RNN model instance
    """
    models = {
        "basic_rnn": KanjiRNN,
        "stroke_rnn": StrokeBasedRNN,
        "radical_rnn": RadicalRNN,
        "hybrid_cnn_rnn": HybridCNNRNN,
    }

    if model_type not in models:
        available = ", ".join(models.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")

    return models[model_type](**kwargs)


if __name__ == "__main__":
    # Test model creation
    print("Testing RNN model creation...")

    # Test basic RNN
    model = create_rnn_model("basic_rnn", num_classes=3036)
    print(f"Basic RNN created: {sum(p.numel() for p in model.parameters())} parameters")

    # Test hybrid CNN-RNN
    hybrid_model = create_rnn_model("hybrid_cnn_rnn", num_classes=3036)
    print(f"Hybrid CNN-RNN created: {sum(p.numel() for p in hybrid_model.parameters())} parameters")

    # Test with dummy input
    dummy_input = torch.randn(2, 1, 64, 64)
    output = hybrid_model(dummy_input)
    print(f"Output shape: {output.shape}")
