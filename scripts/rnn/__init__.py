"""
RNN-based Kanji Recognition Models

This module implements various RNN approaches for kanji character recognition,
including stroke-based, radical-based, and hybrid CNN-RNN architectures.
"""

__version__ = "1.0.0"
__author__ = "Kanji Recognition Project"

from .rnn_model import KanjiRNN, StrokeBasedRNN, RadicalRNN, HybridCNNRNN, create_rnn_model
from .data_processor import (
    StrokeSequenceProcessor,
    RadicalSequenceProcessor,
    SpatialSequenceProcessor,
)
from .train_rnn_model import RNNTrainer, RNNKanjiDataset, collate_fn_factory
from .deploy_rnn_model import KanjiRNNInference, ModelComparisonInference

__all__ = [
    "KanjiRNN",
    "StrokeBasedRNN",
    "RadicalRNN",
    "HybridCNNRNN",
    "create_rnn_model",
    "StrokeSequenceProcessor",
    "RadicalSequenceProcessor",
    "SpatialSequenceProcessor",
    "RNNTrainer",
    "RNNKanjiDataset",
    "collate_fn_factory",
    "KanjiRNNInference",
    "ModelComparisonInference",
]
