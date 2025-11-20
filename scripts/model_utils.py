"""
Utility functions for model type detection and export path management.
"""

from pathlib import Path

# Supported model types in the training structure
SUPPORTED_MODEL_TYPES = ["cnn", "rnn", "qat", "vit", "hiercode", "hiercode_higita"]


def infer_model_type(base_name_or_path, default="cnn"):
    """
    Infer model type from a base name or file path.

    Checks for model type keywords in the input string (case-insensitive).
    Checks for "higita" before "hiercode" to correctly identify hiercode_higita.

    Args:
        base_name_or_path: String containing model name or file path
        default: Default model type if no match found (default: "cnn")

    Returns:
        str: One of SUPPORTED_MODEL_TYPES

    Examples:
        >>> infer_model_type("hiercode_higita_model.pth")
        'hiercode_higita'
        >>> infer_model_type("training/hiercode/checkpoints/best.pth")
        'hiercode'
        >>> infer_model_type("cnn_checkpoint.pt")
        'cnn'
        >>> infer_model_type("unknown_model.pth")
        'cnn'
    """
    input_str = str(base_name_or_path).lower()

    # Check for higita first (before hiercode) to catch hiercode_higita correctly
    if "higita" in input_str:
        return "hiercode_higita"
    elif "hiercode" in input_str:
        return "hiercode"
    elif "rnn" in input_str:
        return "rnn"
    elif "vit" in input_str:
        return "vit"
    elif "qat" in input_str:
        return "qat"
    elif "cnn" in input_str:
        return "cnn"

    return default


def generate_export_path(model_type):
    """
    Generate the export directory path for a given model type.

    Creates the directory if it doesn't exist.

    Args:
        model_type: One of SUPPORTED_MODEL_TYPES

    Returns:
        Path: The export directory path (training/{model_type}/exports/)

    Examples:
        >>> path = generate_export_path("cnn")
        >>> str(path)
        'training/cnn/exports'
        >>> path.exists()
        True
    """
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}. Supported: {SUPPORTED_MODEL_TYPES}")

    export_dir = Path(f"training/{model_type}/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir
