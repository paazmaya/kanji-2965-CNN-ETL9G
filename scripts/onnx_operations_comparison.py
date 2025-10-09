#!/usr/bin/env python3
"""
Real ONNX Operation Comparison - Show actual operations from our models
"""

import os

import onnx


def show_real_onnx_operations():
    """Show the actual ONNX operations from our converted models"""

    print("🔍 Real ONNX Operations in Our Models")
    print("=" * 60)
    print()

    models = {
        "Direct Tract (GlobalAveragePool)": "models/kanji_model_etl9g_64x64_3036classes_tract.onnx",
        "ORT-Tract (Fixed AveragePool)": "models/kanji_model_etl9g_64x64_3036classes_ort-tract.onnx",
    }

    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"❌ {model_name}: File not found")
            continue

        print(f"📋 {model_name}:")
        print(f"   File: {model_path}")

        try:
            model = onnx.load(model_path)

            # Find pooling operations
            pooling_ops = []
            for node in model.graph.node:
                if "pool" in node.op_type.lower():
                    pooling_ops.append(node)

            print(f"   Pooling operations found: {len(pooling_ops)}")

            for i, node in enumerate(pooling_ops, 1):
                print(f"   {i}. {node.op_type}:")
                print(f"      - Name: {node.name}")

                if node.op_type == "GlobalAveragePool":
                    print("      - Attributes: None (global pooling)")
                    print("      - Behavior: Averages over ALL spatial dimensions")
                    print("      - Output: Always (N, C, 1, 1)")

                elif node.op_type == "AveragePool":
                    for attr in node.attribute:
                        if attr.name == "kernel_shape":
                            kernel = list(attr.ints)
                            print(f"      - Kernel shape: {kernel}")
                        elif attr.name == "strides":
                            strides = list(attr.ints)
                            print(f"      - Strides: {strides}")
                        elif attr.name == "pads":
                            pads = list(attr.ints)
                            print(f"      - Padding: {pads}")
                    print("      - Behavior: Fixed kernel size pooling")
                    print("      - Output: Depends on input size and kernel")

                print()

        except Exception as e:
            print(f"   ❌ Error loading model: {e}")

        print("-" * 40)
        print()


def explain_compatibility_impact():
    """Explain the practical impact of the differences"""

    print("💡 Practical Impact Summary")
    print("=" * 60)
    print()

    print("🎯 Why This Matters:")
    print()

    print("1️⃣ **Deployment Flexibility:**")
    print("   - GlobalAveragePool: Limited to ONNX Runtime, PyTorch")
    print("   - AveragePool: Works everywhere (Tract, ORT-Tract, WASM, mobile)")
    print()

    print("2️⃣ **Performance:**")
    print("   - Both operations do the SAME mathematical computation")
    print("   - AveragePool may be slightly faster (no runtime shape checking)")
    print("   - Memory usage identical")
    print()

    print("3️⃣ **Model Size:**")
    print("   - GlobalAveragePool: Smaller ONNX (no kernel_shape attributes)")
    print("   - AveragePool: Slightly larger (explicit kernel_shape stored)")
    print("   - Difference: Negligible (few bytes per operation)")
    print()

    print("4️⃣ **Debugging:**")
    print("   - GlobalAveragePool: 'Magic' - behavior depends on input")
    print("   - AveragePool: Explicit - you can see exact kernel size")
    print("   - AveragePool is easier to debug and understand")
    print()

    print("5️⃣ **Edge Cases:**")
    print("   - GlobalAveragePool: Always works, adapts to any input size")
    print("   - AveragePool: Fails if kernel > input size")
    print("   - Our fix: Pre-calculate sizes, so AveragePool always works")
    print()


if __name__ == "__main__":
    show_real_onnx_operations()
    explain_compatibility_impact()
