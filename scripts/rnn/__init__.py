"""
RNN-based Kanji Recognition Models

This module implements various RNN approaches for kanji character recognition,
including stroke-based, radical-based, and hybrid CNN-RNN architectures.
"""

__version__ = "1.0.0"
__author__ = "Kanji Recognition Project"

from .data_processor import (
    RadicalSequenceProcessor,
    SpatialSequenceProcessor,
    StrokeSequenceProcessor,
)
from .deploy_rnn_model import KanjiRNNInference, ModelComparisonInference
from .rnn_model import HybridCNNRNN, KanjiRNN, RadicalRNN, StrokeBasedRNN, create_rnn_model
from .train_rnn_model import RNNKanjiDataset, RNNTrainer, collate_fn_factory

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
