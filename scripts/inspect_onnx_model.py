#!/usr/bin/env python3
"""
ONNX Model Inspector - Check what operations are used in ONNX models
"""

import os
from collections import Counter

import onnx


def inspect_onnx_model(model_path):
    """Inspect ONNX model and show all operations used"""
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    try:
        # Load the ONNX model
        model = onnx.load(model_path)

        print(f"🔍 Inspecting: {os.path.basename(model_path)}")
        print(f"📁 File size: {os.path.getsize(model_path) / (1024 * 1024):.1f} MB")
        print(f"📊 ONNX version: {model.opset_import[0].version}")
        print()

        # Get all operations used in the model
        operations = []
        pooling_ops = []

        for node in model.graph.node:
            operations.append(node.op_type)
            if "pool" in node.op_type.lower():
                pooling_ops.append(
                    {
                        "op": node.op_type,
                        "name": node.name,
                        "inputs": list(node.input),
                        "outputs": list(node.output),
                        "attributes": {attr.name: attr for attr in node.attribute},
                    }
                )

        # Count operations
        op_counts = Counter(operations)

        print("🔧 Operations used in model:")
        for op, count in sorted(op_counts.items()):
            emoji = "🔴" if "GlobalAveragePool" in op else "🟢" if "pool" in op.lower() else "⚪"
            print(f"  {emoji} {op}: {count}")

        print()

        # Check specifically for pooling operations
        if pooling_ops:
            print("🏊 Pooling operations details:")
            for i, pool_op in enumerate(pooling_ops, 1):
                print(f"  {i}. {pool_op['op']} (name: {pool_op['name']})")
                if pool_op["attributes"]:
                    for attr_name, attr in pool_op["attributes"].items():
                        if attr.type == onnx.AttributeProto.INTS:
                            value = list(attr.ints)
                        elif attr.type == onnx.AttributeProto.INT:
                            value = attr.i
                        elif attr.type == onnx.AttributeProto.STRING:
                            value = attr.s.decode("utf-8")
                        else:
                            value = str(attr)
                        print(f"     - {attr_name}: {value}")
                print()

        # Check for problematic operations
        problematic_ops = [op for op in operations if op in ["GlobalAveragePool", "GlobalMaxPool"]]
        if problematic_ops:
            print("🚨 PROBLEMATIC OPERATIONS FOUND:")
            for op in set(problematic_ops):
                print(f"  ❌ {op} - May not be supported by all backends")
        else:
            print("✅ No problematic global pooling operations found!")

        print()

    except Exception as e:
        print(f"❌ Error inspecting model: {e}")


def main():
    """Main function to inspect all ONNX models"""
    models_to_check = [
        "training/cnn/exports/kanji_model_etl9g_64x64_3036classes_tract.onnx",
        "training/cnn/exports/kanji_model_etl9g_64x64_3036classes_ort-tract.onnx",
        "training/cnn/exports/kanji_model_etl9g_64x64_3036classes_strict.onnx",
    ]

    print("🔍 ONNX Model Inspector")
    print("=" * 50)
    print()

    for model_file in models_to_check:
        if os.path.exists(model_file):
            inspect_onnx_model(model_file)
            print("-" * 50)
            print()
        else:
            print(f"⚠️  Model not found: {model_file}")
            print()


if __name__ == "__main__":
    main()
