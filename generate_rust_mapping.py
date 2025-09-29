#!/usr/bin/env python3
"""
Update Rust code with new model mappings after training
"""

import json
import argparse
from pathlib import Path


def generate_rust_mapping(class_mapping_file: str, output_file: str):
    """Generate Rust HashMap initialization code from class mapping"""

    with open(class_mapping_file, "r") as f:
        mapping = json.load(f)

    class_to_jis = mapping["class_to_jis"]
    num_classes = mapping["num_classes"]

    # Generate Rust HashMap initialization
    rust_code = f"""// Auto-generated class mapping from trained model
// {num_classes} classes total

use std::collections::HashMap;

pub fn create_class_to_jis_mapping() -> HashMap<usize, String> {{
    let mut map = HashMap::new();
"""

    for class_idx, jis_code in class_to_jis.items():
        rust_code += f'    map.insert({class_idx}, "{jis_code}".to_string());\n'

    rust_code += """    map
}

// Updated get_character_from_class_index method
impl KanjiClassifier {
    fn get_character_from_class_index(&self, class_index: usize) -> (String, String) {
        web_sys::console::log_1(&format!("[DEBUG] Looking up class_index {}", class_index).into());
        
        // Use the trained model's direct class mapping
        if let Some(jis_hex) = self.class_to_jis.get(&class_index) {
            web_sys::console::log_1(&format!("[DEBUG] Found class {} -> JIS {}", class_index, jis_hex).into());
            
            // Get character from JIS code
            if let Some(character) = self.jis_to_char.get(jis_hex) {
                return (character.clone(), jis_hex.clone());
            }
            
            // If no character mapping, return JIS code as fallback
            return (format!("JIS_{}", jis_hex), jis_hex.clone());
        }
        
        web_sys::console::log_1(&format!("[DEBUG] No mapping found for class_index {}", class_index).into());
        // Fallback for unmapped class indices
        (format!("Class_{}", class_index), format!("{:04X}", class_index))
    }
}
"""

    with open(output_file, "w") as f:
        f.write(rust_code)

    print(f"Rust mapping code generated: {output_file}")
    print(f"Classes: {num_classes}")


def main():
    parser = argparse.ArgumentParser(description="Generate Rust integration code")
    parser.add_argument(
        "--mapping-file", required=True, help="class_mapping.json from training"
    )
    parser.add_argument("--output", default="rust_mapping.rs", help="Output Rust file")

    args = parser.parse_args()
    generate_rust_mapping(args.mapping_file, args.output)


if __name__ == "__main__":
    main()
