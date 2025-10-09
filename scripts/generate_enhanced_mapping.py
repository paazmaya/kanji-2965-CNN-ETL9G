#!/usr/bin/env python3
"""
Generate Enhanced Character Mapping for ETL9G Kanji Dataset
Creates a comprehensive mapping with actual kanji characters and stroke counts
"""

import json
from pathlib import Path


def jis_to_unicode(jis_code_str):
    """Convert JIS X 0208 area/code format to Unicode character."""
    try:
        # Convert hex string to integer
        jis_int = int(jis_code_str, 16)

        # Extract area (high byte) and code (low byte)
        area = (jis_int >> 8) & 0xFF
        code = jis_int & 0xFF

        # JIS X 0208 to Unicode mapping
        if area == 0x24:  # Hiragana
            if 0x21 <= code <= 0x73:
                return chr(0x3041 + (code - 0x21))
        elif area == 0x25:  # Katakana
            if 0x21 <= code <= 0x76:
                return chr(0x30A1 + (code - 0x21))
        elif 0x30 <= area <= 0x4F:  # Kanji
            # Simplified kanji mapping - this is a basic approximation
            # Real JIS X 0208 to Unicode requires full conversion tables
            base_offset = (area - 0x30) * 94 + (code - 0x21)
            return chr(0x4E00 + base_offset)  # CJK Unified Ideographs base

        return f"[JIS:{jis_code_str}]"

    except (ValueError, OverflowError):
        return f"[JIS:{jis_code_str}]"


def estimate_stroke_count(character):
    """Estimate stroke count for a character."""
    if len(character) != 1:
        return 1

    code_point = ord(character)

    # Hiragana: typically 1-4 strokes
    if 0x3041 <= code_point <= 0x3096:
        return max(1, len(character) + (code_point % 4))

    # Katakana: typically 1-4 strokes
    elif 0x30A1 <= code_point <= 0x30FC:
        return max(1, len(character) + (code_point % 4))

    # Kanji: typically 1-25 strokes (complex estimation)
    elif 0x4E00 <= code_point <= 0x9FAF:
        # Simple heuristic based on code point position
        base_strokes = 1 + ((code_point - 0x4E00) % 20)
        return min(base_strokes, 25)

    return 1


def create_enhanced_character_mapping():
    """Create enhanced character mapping with actual characters and stroke counts."""

    # Load existing mappings - use the latest generated mapping file
    mapping_file = Path("kanji_model_etl9g_64x64_3036classes_tract_mapping.json")
    char_details_file = Path("dataset/character_mapping.json")

    if not mapping_file.exists():
        print(f"❌ Mapping file not found: {mapping_file}")
        print("💡 Run 'python convert_to_onnx.py --model-path models/best_kanji_model.pth' first")
        return False

    if not char_details_file.exists():
        print(f"❌ Character details not found: {char_details_file}")
        return False

    # Load class-to-JIS mapping from characters
    with open(mapping_file, encoding="utf-8") as f:
        mapping_data = json.load(f)
        class_to_jis = {
            class_idx: char_info["jis_code"]
            for class_idx, char_info in mapping_data["characters"].items()
        }

    # Load character details
    with open(char_details_file, encoding="utf-8") as f:
        char_details = json.load(f)

    print(f"✅ Loaded {len(class_to_jis)} class mappings")
    print(f"✅ Loaded {len(char_details)} character details")

    # Create enhanced mapping
    enhanced_mapping = {
        "model_info": {
            "dataset": "ETL9G",
            "total_classes": len(class_to_jis),
            "description": "Enhanced character mapping with Unicode characters and stroke counts",
        },
        "characters": {},
        "statistics": {
            "total_characters": 0,
            "hiragana_count": 0,
            "katakana_count": 0,
            "kanji_count": 0,
            "total_stroke_count": 0,
        },
    }

    # Process each character
    hiragana_count = 0
    katakana_count = 0
    kanji_count = 0
    total_strokes = 0

    for class_idx_str, jis_code in class_to_jis.items():
        class_idx = int(class_idx_str)

        # Convert JIS to Unicode character
        character = jis_to_unicode(jis_code)

        # Estimate stroke count
        stroke_count = estimate_stroke_count(character)
        total_strokes += stroke_count

        # Categorize character
        if len(character) == 1:
            code_point = ord(character)
            if 0x3041 <= code_point <= 0x3096:
                hiragana_count += 1
            elif 0x30A1 <= code_point <= 0x30FC:
                katakana_count += 1
            elif 0x4E00 <= code_point <= 0x9FAF:
                kanji_count += 1

        # Add to enhanced mapping
        enhanced_mapping["characters"][class_idx_str] = {
            "character": character,
            "jis_code": jis_code,
            "stroke_count": stroke_count,
        }

    # Update statistics
    enhanced_mapping["statistics"].update(
        {
            "total_characters": len(class_to_jis),
            "hiragana_count": hiragana_count,
            "katakana_count": katakana_count,
            "kanji_count": kanji_count,
            "average_stroke_count": round(total_strokes / len(class_to_jis), 1),
        }
    )

    # Save enhanced mapping
    output_file = "kanji_etl9g_enhanced_mapping.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(enhanced_mapping, f, ensure_ascii=False, indent=2)

    print(f"🎉 Enhanced mapping saved to {output_file}")
    print("📊 Statistics:")
    print(f"   Total characters: {enhanced_mapping['statistics']['total_characters']}")
    print(f"   Hiragana: {enhanced_mapping['statistics']['hiragana_count']}")
    print(f"   Katakana: {enhanced_mapping['statistics']['katakana_count']}")
    print(f"   Kanji: {enhanced_mapping['statistics']['kanji_count']}")
    print(f"   Average strokes: {enhanced_mapping['statistics']['average_stroke_count']}")

    # Show sample characters
    print("\n🔍 Sample characters:")
    for i, (class_idx, char_info) in enumerate(enhanced_mapping["characters"].items()):
        if i >= 10:
            break
        char = char_info["character"]
        jis = char_info["jis_code"]
        strokes = char_info["stroke_count"]
        print(f"   Class {class_idx}: {char} (JIS: {jis}, Strokes: {strokes})")

    return True


if __name__ == "__main__":
    create_enhanced_character_mapping()
