#!/usr/bin/env python3
"""
Add Stringency to Existing AR6 Files
====================================

This script adds proper stringency mapping to existing AR6 files that were
processed before the stringency mapping was implemented in the pipeline.
"""

import pandas as pd
import sys
from pathlib import Path
from stringency_mapper import add_stringency_to_dataframe

def update_file_with_stringency(filename: str, metadata_file_path: str = "AR6_Scenarios_Database_metadata_indicators_v1.1 2.xlsx"):
    """Update a single file with stringency mapping"""
    if not Path(filename).exists():
        print(f"⚠️  {filename} not found, skipping...")
        return
    
    print(f"\n📂 Processing {filename}...")
    
    # Load the file
    df = pd.read_csv(filename, low_memory=False)
    print(f"   Loaded {len(df):,} rows")
    
    # Check if stringency column already exists
    if 'stringency' in df.columns:
        print(f"   ⚠️  Stringency column already exists in {filename}")
        response = input("   Overwrite existing stringency column? (y/N): ").lower().strip()
        if response != 'y':
            print("   Skipped.")
            return
    
    # Add stringency mapping
    df = add_stringency_to_dataframe(df, metadata_file_path)
    
    # Save the updated file
    df.to_csv(filename, index=False)
    print(f"   ✅ Saved updated {filename} with stringency column")

def main():
    """Main function to update files with stringency"""
    print("🗺️  Adding Stringency to Existing AR6 Files")
    print("=" * 50)
    
    # List of files to update
    files_to_update = [
        "3_final_AR6_target_schema.csv",
        "4_final_AR6_gapfilled.csv", 
        "4_final_AR6_gapfilled_complete.csv"
    ]
    
    metadata_file = "AR6_Scenarios_Database_metadata_indicators_v1.1 2.xlsx"
    
    # Check if metadata file exists
    if not Path(metadata_file).exists():
        print(f"❌ Metadata file not found: {metadata_file}")
        print("   Please ensure the AR6 metadata Excel file is in the current directory")
        return
    
    print(f"📊 Using metadata file: {metadata_file}")
    print(f"🎯 Files to update: {len(files_to_update)}")
    
    # Process each file
    for filename in files_to_update:
        try:
            update_file_with_stringency(filename, metadata_file)
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
    
    print(f"\n✅ Stringency update complete!")
    print("\nUpdated files can now be used with proper stringency mapping for:")
    print("- Gap-filling by stringency category")  
    print("- Violin plots by stringency")
    print("- Analysis by climate ambition level")

if __name__ == "__main__":
    main()