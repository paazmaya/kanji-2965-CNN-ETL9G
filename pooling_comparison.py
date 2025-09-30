#!/usr/bin/env python3
"""
GlobalAveragePool vs AveragePool - Demonstration of the differences
"""

import torch
import torch.nn as nn
import numpy as np


def demonstrate_pooling_differences():
    """Demonstrate the differences between GlobalAveragePool and AveragePool"""

    print("🏊 GlobalAveragePool vs AveragePool Comparison")
    print("=" * 60)
    print()

    # Create sample input tensor: (batch=1, channels=128, height=8, width=8)
    sample_input = torch.randn(1, 128, 8, 8)
    print(f"📥 Input tensor shape: {sample_input.shape}")
    print(f"   - Batch size: {sample_input.shape[0]}")
    print(f"   - Channels: {sample_input.shape[1]}")
    print(f"   - Height: {sample_input.shape[2]}")
    print(f"   - Width: {sample_input.shape[3]}")
    print()

    # 1. GlobalAveragePool (AdaptiveAvgPool2d(1))
    print("🌍 GlobalAveragePool (AdaptiveAvgPool2d(1)):")
    print("   - Reduces spatial dimensions to 1x1 REGARDLESS of input size")
    print("   - Output size is always (batch, channels, 1, 1)")
    print("   - Averages over ALL spatial locations")

    global_pool = nn.AdaptiveAvgPool2d(1)
    global_output = global_pool(sample_input)
    print(f"   - Input: {sample_input.shape} → Output: {global_output.shape}")
    print(f"   - Computation: Average over {8 * 8} = 64 spatial locations")
    print()

    # 2. AveragePool with same effect
    print("🎯 AveragePool (with 8x8 kernel to match GlobalAveragePool):")
    print("   - Uses FIXED kernel size")
    print("   - Must specify exact kernel size that matches input spatial dimensions")
    print("   - Only works if kernel size matches input size exactly")

    avg_pool_8x8 = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
    avg_output_8x8 = avg_pool_8x8(sample_input)
    print(f"   - Input: {sample_input.shape} → Output: {avg_output_8x8.shape}")
    print(f"   - Computation: Average over 8×8 = 64 spatial locations")
    print()

    # Verify they produce the same result
    print("🔍 Verification - Are outputs identical?")
    are_identical = torch.allclose(global_output, avg_output_8x8, atol=1e-6)
    print(f"   - Results identical: {'✅ YES' if are_identical else '❌ NO'}")
    print(
        f"   - Max difference: {torch.max(torch.abs(global_output - avg_output_8x8)).item():.2e}"
    )
    print()

    # 3. Show what happens with wrong kernel size
    print("⚠️  AveragePool with WRONG kernel size:")
    avg_pool_4x4 = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
    avg_output_4x4 = avg_pool_4x4(sample_input)
    print(f"   - Input: {sample_input.shape} → Output: {avg_output_4x4.shape}")
    print("   - Result: 5x5 output instead of 1x1!")
    print("   - This would break channel attention that expects 1x1 output")
    print()

    # 4. Different input size demonstration
    print("🔄 Different input size demonstration:")
    different_input = torch.randn(1, 128, 4, 4)  # 4x4 instead of 8x8
    print(f"📥 New input shape: {different_input.shape}")

    # GlobalAveragePool adapts automatically
    global_output_4x4 = global_pool(different_input)
    print(
        f"   - GlobalAveragePool: {different_input.shape} → {global_output_4x4.shape} ✅"
    )

    # AveragePool with 8x8 kernel fails on 4x4 input
    print("   - AveragePool(8x8) on 4x4 input: WOULD FAIL ❌")
    print("   - Reason: Kernel (8x8) larger than input (4x4)")

    # But AveragePool with 4x4 kernel works
    avg_pool_4x4_correct = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
    avg_output_4x4_correct = avg_pool_4x4_correct(different_input)
    print(
        f"   - AveragePool(4x4) on 4x4 input: {different_input.shape} → {avg_output_4x4_correct.shape} ✅"
    )
    print()


def explain_onnx_differences():
    """Explain why this matters for ONNX export and backend compatibility"""

    print("🔧 ONNX Export & Backend Compatibility")
    print("=" * 60)
    print()

    print("📋 In PyTorch:")
    print("   - AdaptiveAvgPool2d(1) = Smart pooling that adapts to any input size")
    print("   - AvgPool2d(kernel_size) = Fixed pooling with specified kernel")
    print()

    print("📋 In ONNX:")
    print("   - AdaptiveAvgPool2d(1) → GlobalAveragePool operator")
    print("   - AvgPool2d(kernel_size) → AveragePool operator")
    print()

    print("🎯 Backend Support:")
    print("   ✅ GlobalAveragePool:")
    print("      - Supported: PyTorch, TensorFlow, ONNX Runtime")
    print("      - ❌ NOT Supported: Tract (and ORT-Tract wrapper)")
    print()
    print("   ✅ AveragePool:")
    print("      - Supported: ALL backends including Tract")
    print("      - Universal compatibility")
    print()

    print("⚡ The Trade-off:")
    print("   - GlobalAveragePool: Flexible but limited backend support")
    print("   - AveragePool: Requires exact kernel size but universal support")
    print()

    print("🛠️ Our Solution:")
    print("   1. Calculate exact spatial dimensions at each layer")
    print("   2. Replace AdaptiveAvgPool2d with correctly-sized AvgPool2d")
    print("   3. Maintain same functionality with broader compatibility")
    print()

    spatial_sizes = {
        "Input": "64×64",
        "After conv1 (stride=2)": "32×32",
        "After conv2 (stride=2)": "16×16",
        "After conv3 (stride=2)": "8×8  ← attention3 uses AvgPool2d(8)",
        "After conv4 (stride=2)": "4×4  ← attention4 uses AvgPool2d(4)",
        "After conv5 (stride=1)": "4×4  ← attention5 uses AvgPool2d(4), main pool uses AvgPool2d(4)",
    }

    print("📐 Spatial size calculation for our model:")
    for stage, size in spatial_sizes.items():
        print(f"   - {stage}: {size}")
    print()


def show_onnx_operation_details():
    """Show the actual ONNX operation differences"""

    print("🔍 ONNX Operation Details")
    print("=" * 60)
    print()

    print("📋 GlobalAveragePool ONNX Operation:")
    print("   - Opset: 1+")
    print("   - Inputs: X (tensor)")
    print("   - Outputs: Y (tensor)")
    print("   - Attributes: None")
    print("   - Behavior: Y = global_average_pool(X)")
    print("   - Output shape: (N, C, 1, 1) regardless of input spatial size")
    print()

    print("📋 AveragePool ONNX Operation:")
    print("   - Opset: 1+")
    print("   - Inputs: X (tensor)")
    print("   - Outputs: Y (tensor)")
    print("   - Attributes:")
    print("     * kernel_shape: [int, int] - Size of pooling kernel")
    print("     * strides: [int, int] - Stride of pooling")
    print("     * pads: [int, int, int, int] - Padding")
    print("     * ceil_mode: bool - Ceiling or floor for output shape")
    print("   - Behavior: Y = average_pool(X, kernel_shape, strides, pads)")
    print("   - Output shape: Depends on kernel_shape and input size")
    print()

    print("🎯 Why Tract doesn't support GlobalAveragePool:")
    print("   - Tract focuses on deterministic, explicit operations")
    print("   - GlobalAveragePool behavior depends on runtime input shape")
    print("   - AveragePool has explicit, compile-time determined behavior")
    print("   - This makes AveragePool more suitable for embedded/edge deployment")


if __name__ == "__main__":
    demonstrate_pooling_differences()
    explain_onnx_differences()
    show_onnx_operation_details()
