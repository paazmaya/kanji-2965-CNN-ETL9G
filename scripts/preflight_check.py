#!/usr/bin/env python3
"""
Pre-flight check for ETL9G training setup
Run this before starting actual training
"""

import sys
from pathlib import Path


def check_virtual_environment():
    """Check if virtual environment is active"""
    print("=== Checking Virtual Environment ===")

    # Check if we're in a virtual environment
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )

    if in_venv:
        venv_path = sys.prefix
        print(f"✓ Virtual environment active: {venv_path}")
        return True
    else:
        print("✗ No virtual environment detected")
        print("Please activate your virtual environment:")
        print("  Windows: .\\venv\\Scripts\\Activate.ps1")
        print("  Linux/Mac: source venv/bin/activate")
        return False


def check_gpu_availability():
    """Check NVIDIA GPU and CUDA availability"""
    print("\n=== Checking GPU and CUDA ===")

    try:
        import torch

        # Check CUDA availability
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            gpu_count = torch.cuda.device_count()
            print(f"✓ CUDA available with {gpu_count} GPU(s)")

            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

            # Test GPU memory allocation
            try:
                test_tensor = torch.randn(1000, 1000).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                print("✓ GPU memory allocation test passed")
            except Exception as e:
                print(f"⚠ GPU memory test failed: {e}")

            return True
        else:
            print("✗ CUDA not available")
            print("Please check:")
            print("  - NVIDIA GPU drivers are installed")
            print("  - PyTorch with CUDA support is installed")
            print("  - uv pip install torch --index-url https://download.pytorch.org/whl/cu129")
            return False

    except ImportError:
        print("✗ PyTorch not available")
        return False


def check_requirements():
    """Check if all required packages are available"""
    print("=== Checking Requirements ===")

    required_packages = [
        "torch",
        "torchvision",
        "numpy",
        "sklearn",
        "matplotlib",
        "tqdm",
        "cv2",
        "PIL",
        "onnx",
        "onnxruntime",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == "cv2":
                import cv2
            elif package == "PIL":
                import PIL
            elif package == "sklearn":
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)

    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: uv pip install " + " ".join(missing_packages))
        return False

    print("All packages available ✓")
    return True


def check_data_files():
    """Check if ETL9G data files are available"""
    print("\n=== Checking Data Files ===")

    etl9g_dir = Path("ETL9G")
    if not etl9g_dir.exists():
        print(f"✗ ETL9G directory not found: {etl9g_dir}")
        return False

    # Check for ETL9G files
    etl_files = list(etl9g_dir.glob("ETL9G_*"))
    etl_files = [f for f in etl_files if f.is_file() and "INFO" not in f.name]

    print(f"Found {len(etl_files)} ETL9G files")

    if len(etl_files) < 50:
        print(f"✗ Expected 50 ETL9G files, found {len(etl_files)}")
        return False

    # Check file sizes (should be around 99MB each)
    sample_file = etl_files[0]
    file_size_mb = sample_file.stat().st_size / (1024 * 1024)
    print(f"Sample file size: {file_size_mb:.1f} MB")

    if file_size_mb < 90 or file_size_mb > 110:
        print(f"✗ Unexpected file size: {file_size_mb:.1f} MB (expected ~99 MB)")
        return False

    print("✓ ETL9G data files look good")
    return True


def check_system_resources():
    """Check system memory and disk space"""
    print("\n=== Checking System Resources ===")

    try:
        import psutil

        # Memory check
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)

        print(f"Total RAM: {memory_gb:.1f} GB")
        print(f"Available RAM: {available_gb:.1f} GB")

        if available_gb < 8:
            print(
                "⚠ Warning: Less than 8 GB RAM available. Consider using --sample-limit for testing."
            )
        else:
            print("✓ Sufficient RAM available")

        # Disk space check
        disk = psutil.disk_usage(".")
        free_gb = disk.free / (1024**3)

        print(f"Free disk space: {free_gb:.1f} GB")

        if free_gb < 10:
            print("⚠ Warning: Less than 10 GB free disk space")
        else:
            print("✓ Sufficient disk space")

    except ImportError:
        print("psutil not available - cannot check system resources")
        print("Install with: uv pip install psutil")


def check_training_scripts():
    """Check if all training scripts are present"""
    print("\n=== Checking Training Scripts ===")

    required_files = [
        "scripts/prepare_etl9g_dataset.py",
        "scripts/train_etl9g_model.py",
        "scripts/test_etl9g_setup.py",
    ]

    all_present = True

    for filename in required_files:
        if Path(filename).exists():
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} - MISSING")
            all_present = False

    return all_present


def estimate_training_time():
    """Estimate training time based on system"""
    print("\n=== Training Time Estimates ===")

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU available: {gpu_name}")
            print("Estimated training time: 2-4 hours for 30 epochs")
        else:
            print("No GPU available - using CPU")
            print("Estimated training time: 8-12 hours for 30 epochs")
            print("Recommendation: Use --sample-limit 50000 for testing")

    except ImportError:
        print("PyTorch not available")


def main():
    """Run all checks"""
    print("ETL9G Training Setup Check")
    print("=" * 50)

    checks_passed = 0
    total_checks = 0

    # Virtual environment check
    total_checks += 1
    if check_virtual_environment():
        checks_passed += 1

    # GPU and CUDA check
    total_checks += 1
    if check_gpu_availability():
        checks_passed += 1

    # Requirements check
    total_checks += 1
    if check_requirements():
        checks_passed += 1

    # Data files check
    total_checks += 1
    if check_data_files():
        checks_passed += 1

    # System resources
    check_system_resources()

    # Training scripts check
    total_checks += 1
    if check_training_scripts():
        checks_passed += 1

    # Training time estimate
    estimate_training_time()

    print("\n=== Summary ===")
    print(f"Checks passed: {checks_passed}/{total_checks}")

    if checks_passed == total_checks:
        print("✓ Setup looks good! You can proceed with training.")
        print("\nRecommended workflow:")
        print(
            "1. python scripts/prepare_etl9g_dataset.py --etl-dir ETL9G --output-dir dataset --size 64"
        )
        print("2. python scripts/test_etl9g_setup.py --data-dir dataset --test-model")
        print("3. python scripts/train_etl9g_model.py --data-dir dataset --epochs 30")
        print("4. python scripts/convert_to_onnx.py --model-path best_kanji_model.pth")
        print("5. python scripts/convert_to_safetensors.py --model-path best_kanji_model.pth")
    else:
        print("✗ Some checks failed. Please resolve issues before training.")

    return checks_passed == total_checks


if __name__ == "__main__":
    main()
