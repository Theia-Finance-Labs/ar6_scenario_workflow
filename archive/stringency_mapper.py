#!/usr/bin/env python3
"""
AR6 Stringency Mapper Module
============================

This module provides functions to load and apply the official AR6 stringency 
mapping from the metadata Excel file to AR6 scenario data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def load_ar6_metadata_stringency_mapping(metadata_file_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load stringency mapping directly from AR6 metadata Excel file.
    
    Args:
        metadata_file_path: Path to the AR6 metadata Excel file. 
                           If None, uses default path.
    
    Returns:
        Dictionary mapping "Model|Scenario" -> "Stringency"
    """
    if metadata_file_path is None:
        metadata_file_path = "AR6_Scenarios_Database_metadata_indicators_v1.1 2.xlsx"
    
    print(f"📂 Loading AR6 stringency mapping from {metadata_file_path}...")
    
    try:
        # Read the metadata Excel file
        excel_file = pd.ExcelFile(metadata_file_path)
        sheet_names = excel_file.sheet_names
        
        # Find the metadata sheet (starts with 'meta_')
        meta_sheet = None
        for sheet in sheet_names:
            if sheet.lower().startswith('meta_'):
                meta_sheet = sheet
                break
        
        if meta_sheet is None:
            # Look for sheets containing 'meta' or 'indicator'
            for sheet in sheet_names:
                if 'meta' in sheet.lower() or 'indicator' in sheet.lower():
                    meta_sheet = sheet
                    break
        
        if meta_sheet is None:
            print(f"   ⚠️  No metadata sheet found in {metadata_file_path}")
            print(f"   Available sheets: {sheet_names}")
            return {}
        
        # Read the metadata sheet
        metadata_df = pd.read_excel(metadata_file_path, sheet_name=meta_sheet)
        print(f"   ✅ Loaded {len(metadata_df)} rows from sheet '{meta_sheet}'")
        
        # Extract Model, Scenario, and Category (stringency) columns
        # The typical column names in AR6 metadata are: 'Model', 'Scenario', 'Category'
        required_cols = ['Model', 'Scenario', 'Category']
        missing_cols = [col for col in required_cols if col not in metadata_df.columns]
        
        if missing_cols:
            print(f"   ⚠️  Missing required columns: {missing_cols}")
            print(f"   Available columns: {list(metadata_df.columns)}")
            return {}
        
        # Create the mapping dictionary
        mapping_dict = {}
        for _, row in metadata_df.iterrows():
            if pd.notna(row['Model']) and pd.notna(row['Scenario']) and pd.notna(row['Category']):
                key = f"{row['Model']}|{row['Scenario']}"
                mapping_dict[key] = row['Category']
        
        print(f"   ✅ Created stringency mapping for {len(mapping_dict)} model+scenario combinations")
        
        # Show stringency distribution
        stringency_values = list(mapping_dict.values())
        if stringency_values:
            from collections import Counter
            stringency_counts = Counter(stringency_values)
            print(f"   📊 Stringency distribution:")
            for stringency in sorted(stringency_counts.keys()):
                print(f"     {stringency}: {stringency_counts[stringency]} scenarios")
        
        return mapping_dict
        
    except Exception as e:
        print(f"   ❌ Error loading metadata file: {e}")
        return {}


def add_stringency_to_dataframe(df: pd.DataFrame, metadata_file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Add stringency column to AR6 dataframe using official metadata mapping.
    
    Args:
        df: DataFrame with 'scenario_provider' and 'scenario' columns
        metadata_file_path: Path to AR6 metadata Excel file
        
    Returns:
        DataFrame with added 'stringency' column
    """
    print("🗺️  Adding stringency column to dataframe...")
    
    # Load the mapping
    mapping_dict = load_ar6_metadata_stringency_mapping(metadata_file_path)
    
    if not mapping_dict:
        print("   ⚠️  No stringency mapping available, setting all to 'UNKNOWN'")
        df['stringency'] = 'UNKNOWN'
        return df
    
    # Apply the mapping
    def get_stringency(row):
        key = f"{row['scenario_provider']}|{row['scenario']}"
        return mapping_dict.get(key, 'UNKNOWN')
    
    df['stringency'] = df.apply(get_stringency, axis=1)
    
    # Report results
    stringency_counts = df['stringency'].value_counts()
    print(f"   📊 Stringency mapping results:")
    total_rows = len(df)
    unknown_count = stringency_counts.get('UNKNOWN', 0)
    mapped_count = total_rows - unknown_count
    
    print(f"     Successfully mapped: {mapped_count:,} ({mapped_count/total_rows*100:.1f}%)")
    print(f"     Unknown scenarios: {unknown_count:,} ({unknown_count/total_rows*100:.1f}%)")
    
    print(f"   📈 Stringency distribution:")
    for stringency in sorted(stringency_counts.index):
        count = stringency_counts[stringency]
        print(f"     {stringency}: {count:,} ({count/total_rows*100:.1f}%)")
    
    return df


def save_stringency_mapping_reference(output_path: str = "stringency_mapping_reference.csv", 
                                     metadata_file_path: Optional[str] = None) -> None:
    """
    Save the stringency mapping as a reference CSV file.
    
    Args:
        output_path: Path where to save the mapping CSV
        metadata_file_path: Path to AR6 metadata Excel file
    """
    print(f"💾 Saving stringency mapping reference to {output_path}...")
    
    mapping_dict = load_ar6_metadata_stringency_mapping(metadata_file_path)
    
    if not mapping_dict:
        print("   ⚠️  No mapping to save")
        return
    
    # Convert to DataFrame
    mapping_data = []
    for key, stringency in mapping_dict.items():
        model, scenario = key.split('|', 1)
        mapping_data.append({
            'model': model,
            'scenario': scenario,
            'stringency': stringency
        })
    
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv(output_path, index=False)
    
    print(f"   ✅ Saved {len(mapping_df)} mappings to {output_path}")


if __name__ == "__main__":
    # Test the mapping functionality
    print("🧪 Testing AR6 Stringency Mapper")
    print("=" * 50)
    
    # Save reference mapping
    save_stringency_mapping_reference()
    
    print("\n✅ Stringency mapper module ready for use!")