#!/usr/bin/env python3
"""
Normalize filenames in the dataset directory.
Renames files to clean, indexed format: <class_name>_<index>.<ext>

Example:
  Before: N_456AlterneriaLeafSpot_original_ALS(135).jpg_d14dbbc4-cb5b-4403-a3db-5bfa8eb6f990.jpg
  After:  alternaria_leaf_spot_001.jpg

Usage:
  python normalize_filenames.py [--dry-run] [--dataset-path PATH]
"""

import os
import argparse
from pathlib import Path


def normalize_class_name(class_name):
    """Convert class name to lowercase with underscores."""
    return class_name.lower().replace(' ', '_').replace('-', '_')


def get_extension(filename):
    """Get file extension, preferring common image extensions."""
    # Handle double extensions like .jpg.jpg
    parts = filename.lower().split('.')
    # Common image extensions
    valid_exts = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff'}
    
    # Find the first valid extension from the end
    for ext in reversed(parts[1:]):
        if ext in valid_exts:
            return '.' + ext
    
    # Fallback to last extension
    return '.' + parts[-1] if len(parts) > 1 else '.jpg'


def rename_files_in_dataset(dataset_path, dry_run=False):
    """
    Rename all files in dataset to normalized format.
    
    Args:
        dataset_path: Path to dataset directory containing class folders
        dry_run: If True, print changes without applying them
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist")
        return
    
    stats = {'total': 0, 'renamed': 0, 'skipped': 0, 'errors': 0}
    
    # Iterate through class directories
    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        normalized_class = normalize_class_name(class_name)
        
        print(f"\nProcessing class: {class_name}")
        
        # Get all files in the class directory
        files = sorted([f for f in class_dir.iterdir() if f.is_file()])
        
        # Filter out hidden files and non-image files
        image_files = [f for f in files if not f.name.startswith('.')]
        
        print(f"  Found {len(image_files)} files")
        
        # Rename each file with zero-padded index
        index_width = len(str(len(image_files)))  # Dynamic width based on count
        
        for idx, old_path in enumerate(image_files, start=1):
            stats['total'] += 1
            
            # Get extension
            ext = get_extension(old_path.name)
            
            # Create new filename
            new_name = f"{normalized_class}_{idx:0{index_width}d}{ext}"
            new_path = class_dir / new_name
            
            # Skip if already correctly named
            if old_path.name == new_name:
                stats['skipped'] += 1
                continue
            
            # Check for name collision
            if new_path.exists():
                print(f"  ⚠ Collision: {new_name} already exists, skipping {old_path.name}")
                stats['errors'] += 1
                continue

            # Rename or print
            if dry_run:
                print(f"  {old_path.name} → {new_name}")
            else:
                try:
                    old_path.rename(new_path)
                    stats['renamed'] += 1
                    if idx <= 3:  # Show first 3 as examples
                        print(f"  ✓ {old_path.name} → {new_name}")
                except Exception as e:
                    print(f"  ✗ Error renaming {old_path.name}: {e}")
                    stats['errors'] += 1
        
        if not dry_run and stats['renamed'] > 3:
            print(f"  ... and {len(image_files) - 3} more files renamed")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Total files processed: {stats['total']}")
    print(f"  Files renamed: {stats['renamed']}")
    print(f"  Files skipped (already normalized): {stats['skipped']}")
    print(f"  Errors: {stats['errors']}")
    
    if dry_run:
        print("\n⚠ DRY RUN - no files were actually renamed")
        print("  Run without --dry-run to apply changes")


def main():
    parser = argparse.ArgumentParser(
        description="Normalize filenames in dataset directory"
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='/Users/thun/Desktop/cauliflower_training/dataset',
        help='Path to dataset directory (default: /Users/thun/Desktop/cauliflower_training/dataset)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print changes without applying them'
    )
    
    args = parser.parse_args()
    
    print("Dataset Filename Normalizer")
    print("="*60)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE (will rename files)'}")
    print("="*60)
    
    rename_files_in_dataset(args.dataset_path, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
