#!/usr/bin/env python3
"""
ETL9G Dataset Preparation - Optimized for Large-Scale Kanji Recognition
Handles 3,036 character classes with 607,200 samples efficiently
"""

import os
import struct
import numpy as np
import json
from pathlib import Path
import argparse
from collections import defaultdict
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import sqlite3

class ETL9GProcessor:
    def __init__(self, etl_dir: str, output_dir: str):
        self.etl_dir = Path(etl_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ETL9G record format constants
        self.RECORD_SIZE = 8199
        self.IMAGE_WIDTH = 128
        self.IMAGE_HEIGHT = 127
        self.IMAGE_OFFSET = 64  # Gray-scale image starts at byte 65 (0-indexed: 64)
        self.IMAGE_SIZE_BYTES = 8128  # 16256 pixels * 4 bits / 8 bits per byte
        
        # Character mappings
        self.jis_to_class = {}
        self.class_to_jis = {}
        self.class_counter = 0
        
        # Statistics
        self.total_samples = 0
        self.samples_per_jis = defaultdict(int)
        
    def extract_record_info(self, record_data: bytes):
        """Extract information from a single ETL9G record"""
        try:
            # Serial Sheet Number (bytes 1-2, big-endian, 1-indexed in docs)
            serial = struct.unpack('>H', record_data[0:2])[0]
            
            # JIS code (bytes 3-4, big-endian for ETL9G)
            jis_code = struct.unpack('>H', record_data[2:4])[0]
            
            # ASCII Reading (bytes 5-12)
            ascii_reading = record_data[4:12].decode('ascii', errors='ignore').strip()
            
            # Serial Data Number (bytes 13-16)
            data_serial = struct.unpack('>I', record_data[12:16])[0]
            
            # Writer information
            writer_id = record_data[18]  # Age of writer
            
            # Gray-scale image data (bytes 65-8192, 4-bit packed)
            image_data = record_data[self.IMAGE_OFFSET:self.IMAGE_OFFSET + self.IMAGE_SIZE_BYTES]
            
            return {
                'serial': serial,
                'jis_code': jis_code,
                'ascii_reading': ascii_reading,
                'data_serial': data_serial,
                'writer_id': writer_id,
                'image_data': image_data
            }
        except Exception as e:
            print(f"Error extracting record: {e}")
            return None
    
    def unpack_4bit_image(self, packed_data: bytes) -> np.ndarray:
        """Convert 4-bit packed grayscale image to 2D array"""
        try:
            # Unpack 4-bit data
            unpacked = []
            for byte in packed_data:
                # Each byte contains two 4-bit pixels
                unpacked.append((byte >> 4) & 0xF)  # Upper 4 bits
                unpacked.append(byte & 0xF)         # Lower 4 bits
            
            # Take only the required number of pixels
            pixels = unpacked[:self.IMAGE_WIDTH * self.IMAGE_HEIGHT]
            
            # Reshape to image dimensions and convert to 0-255 range
            image = np.array(pixels, dtype=np.uint8).reshape(
                self.IMAGE_HEIGHT, self.IMAGE_WIDTH
            )
            
            # Convert from 16 levels (0-15) to 256 levels (0-255)
            return image * 17  # 15 * 17 = 255
            
        except Exception as e:
            print(f"Error unpacking image: {e}")
            return None
    
    def preprocess_image(self, image: np.ndarray, target_size: int = 64) -> np.ndarray:
        """Preprocess image for training with quality improvements"""
        # Apply slight Gaussian blur to reduce noise
        image_smooth = cv2.GaussianBlur(image, (3, 3), 0.5)
        
        # Resize to target size using area interpolation (better for downsampling)
        image_resized = cv2.resize(image_smooth, (target_size, target_size), 
                                 interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Optional: histogram equalization for better contrast
        # image_eq = cv2.equalizeHist((image_normalized * 255).astype(np.uint8))
        # image_normalized = image_eq.astype(np.float32) / 255.0
        
        return image_normalized
    
    def process_single_file(self, etl_file_path: Path, target_size: int = 64):
        """Process a single ETL9G file"""
        samples = []
        local_jis_to_class = {}
        local_class_counter = 0
        
        print(f"Processing {etl_file_path.name}...")
        
        with open(etl_file_path, 'rb') as f:
            record_count = 0
            
            while True:
                # Read one record
                record_data = f.read(self.RECORD_SIZE)
                if len(record_data) != self.RECORD_SIZE:
                    break
                
                # Extract record information
                record_info = self.extract_record_info(record_data)
                if record_info is None:
                    continue
                
                jis_code = record_info['jis_code']
                
                # Skip invalid JIS codes
                if jis_code == 0 or jis_code == 0xFFFF:
                    continue
                
                # Assign class index for new JIS code
                if jis_code not in local_jis_to_class:
                    local_jis_to_class[jis_code] = local_class_counter
                    local_class_counter += 1
                
                class_idx = local_jis_to_class[jis_code]
                
                # Unpack and preprocess image
                image = self.unpack_4bit_image(record_info['image_data'])
                if image is None:
                    continue
                
                processed_image = self.preprocess_image(image, target_size)
                
                # Store sample data
                samples.append({
                    'image': processed_image.flatten(),
                    'class_idx': class_idx,
                    'jis_code': jis_code,
                    'writer_id': record_info['writer_id'],
                    'ascii_reading': record_info['ascii_reading']
                })
                
                record_count += 1
        
        print(f"  Processed {etl_file_path.name}: {record_count} records, {local_class_counter} unique characters")
        return samples, local_jis_to_class
    
    def merge_class_mappings(self, all_local_mappings):
        """Merge class mappings from all files into global mapping"""
        global_jis_to_class = {}
        global_class_counter = 0
        
        # Collect all unique JIS codes
        all_jis_codes = set()
        for local_mapping in all_local_mappings:
            all_jis_codes.update(local_mapping.keys())
        
        # Sort JIS codes for consistent ordering
        sorted_jis_codes = sorted(all_jis_codes)
        
        # Assign global class indices
        for jis_code in sorted_jis_codes:
            global_jis_to_class[jis_code] = global_class_counter
            global_class_counter += 1
        
        return global_jis_to_class
    
    def process_all_files(self, target_size: int = 64, max_workers: int = None):
        """Process all ETL9G files with multiprocessing"""
        
        # Find all ETL9G files (exclude info files)
        etl_files = sorted([f for f in self.etl_dir.glob('ETL9G_*') if f.is_file() and 'INFO' not in f.name])
        print(f"Found {len(etl_files)} ETL9G files")
        
        if not etl_files:
            raise FileNotFoundError(f"No ETL9G files found in {self.etl_dir}")
        
        # Process files in parallel (conservative worker count for memory)
        if max_workers is None:
            max_workers = min(4, mp.cpu_count() // 2)  # Conservative for memory usage
        
        all_samples = []
        all_local_mappings = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file processing tasks
            future_to_file = {
                executor.submit(self.process_single_file, etl_file, target_size): etl_file 
                for etl_file in etl_files
            }
            
            # Collect results
            for future in tqdm(as_completed(future_to_file), total=len(etl_files), desc="Processing files"):
                etl_file = future_to_file[future]
                try:
                    samples, local_mapping = future.result()
                    all_samples.extend(samples)
                    all_local_mappings.append(local_mapping)
                except Exception as e:
                    print(f"Error processing {etl_file}: {e}")
        
        # Merge class mappings
        print("Merging class mappings...")
        global_jis_to_class = self.merge_class_mappings(all_local_mappings)
        
        # Update class indices in samples
        print("Updating class indices...")
        for sample in tqdm(all_samples, desc="Updating indices"):
            sample['class_idx'] = global_jis_to_class[sample['jis_code']]
        
        # Create final arrays
        print("Creating final dataset...")
        X = np.array([sample['image'] for sample in all_samples], dtype=np.float32)
        y = np.array([sample['class_idx'] for sample in all_samples], dtype=np.int32)
        
        # Create metadata
        class_to_jis = {class_idx: f"{jis_code:04X}" for jis_code, class_idx in global_jis_to_class.items()}
        
        # Statistics
        samples_per_class = defaultdict(int)
        for sample in all_samples:
            samples_per_class[sample['class_idx']] += 1
        
        metadata = {
            'num_classes': len(global_jis_to_class),
            'total_samples': len(all_samples),
            'target_size': target_size,
            'jis_to_class': {f"{k:04X}": v for k, v in global_jis_to_class.items()},
            'class_to_jis': class_to_jis,
            'samples_per_class': dict(samples_per_class),
            'dataset_info': {
                'source': 'ETL9G',
                'description': '3,036 JIS Level 1 Kanji + Hiragana characters',
                'files_processed': len(etl_files),
                'avg_samples_per_class': len(all_samples) / len(global_jis_to_class)
            }
        }
        
        print(f"\nDataset Summary:")
        print(f"  Total classes: {metadata['num_classes']}")
        print(f"  Total samples: {metadata['total_samples']}")
        print(f"  Image size: {target_size}x{target_size}")
        print(f"  Average samples per class: {metadata['dataset_info']['avg_samples_per_class']:.1f}")
        print(f"  Files processed: {len(etl_files)}")
        
        # Save dataset in chunks to handle large size
        print("Saving dataset...")
        chunk_size = 50000  # Process in chunks of 50k samples
        
        if len(all_samples) > chunk_size:
            # Save in multiple chunks for memory efficiency
            for i in range(0, len(all_samples), chunk_size):
                chunk_end = min(i + chunk_size, len(all_samples))
                chunk_X = np.array([sample['image'] for sample in all_samples[i:chunk_end]], dtype=np.float32)
                chunk_y = np.array([sample['class_idx'] for sample in all_samples[i:chunk_end]], dtype=np.int32)
                
                np.savez_compressed(
                    self.output_dir / f'etl9g_dataset_chunk_{i//chunk_size:02d}.npz',
                    X=chunk_X, y=chunk_y
                )
                print(f"  Saved chunk {i//chunk_size + 1}: samples {i}-{chunk_end-1}")
            
            # Save chunk info
            chunk_info = {
                'total_samples': len(all_samples),
                'chunk_size': chunk_size,
                'num_chunks': (len(all_samples) + chunk_size - 1) // chunk_size
            }
            with open(self.output_dir / 'chunk_info.json', 'w') as f:
                json.dump(chunk_info, f, indent=2)
        else:
            # Save as single file if manageable size
            X = np.array([sample['image'] for sample in all_samples], dtype=np.float32)
            y = np.array([sample['class_idx'] for sample in all_samples], dtype=np.int32)
            np.savez_compressed(
                self.output_dir / 'etl9g_dataset.npz',
                X=X, y=y
            )
        
        # Save metadata
        with open(self.output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save detailed character mapping for debugging
        with open(self.output_dir / 'character_mapping.json', 'w', encoding='utf-8') as f:
            char_details = {}
            for sample in all_samples[:1000]:  # Sample for details
                jis_hex = f"{sample['jis_code']:04X}"
                if jis_hex not in char_details:
                    char_details[jis_hex] = {
                        'class_idx': sample['class_idx'],
                        'ascii_reading': sample['ascii_reading'],
                        'sample_count': samples_per_class[sample['class_idx']]
                    }
            json.dump(char_details, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to {self.output_dir}")
        return X, y, metadata

def main():
    parser = argparse.ArgumentParser(description='Prepare ETL9G dataset for training')
    parser.add_argument('--etl-dir', required=True, help='Directory containing ETL9G files')
    parser.add_argument('--output-dir', required=True, help='Output directory for processed dataset')
    parser.add_argument('--size', type=int, default=64, help='Target image size (default: 64)')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')
    
    args = parser.parse_args()
    
    processor = ETL9GProcessor(args.etl_dir, args.output_dir)
    X, y, metadata = processor.process_all_files(target_size=args.size, max_workers=args.workers)
    
    print(f"ETL9G dataset preparation complete!")
    print(f"Shape: X={X.shape}, y={y.shape}")
    print(f"Ready for training with {metadata['num_classes']} character classes!")

if __name__ == "__main__":
    main()