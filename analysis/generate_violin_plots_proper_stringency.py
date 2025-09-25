#!/usr/bin/env python3
"""
AR6 Non-Gap-Filled Data Violin Plot Generator (Proper Stringency Mapping)
=========================================================================

Updated version that uses the official AR6 metadata stringency mapping.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
METRICS = [
    'efficiency_decimal',
    'lifetime_years', 
    'om_cost_usd_per_mw_per_yr',
    'capital_cost_usd_per_mw',
    'scenario_price',
    'fuel_price'
]

METRIC_LABELS = {
    'efficiency_decimal': 'Efficiency\n(decimal)',
    'lifetime_years': 'Lifetime\n(years)',
    'om_cost_usd_per_mw_per_yr': 'O&M Cost\n(USD/MW/yr)',
    'capital_cost_usd_per_mw': 'Capital Cost\n(USD/MW)',
    'scenario_price': 'Electricity Price\n(USD/MWh)',
    'fuel_price': 'Fuel Price\n(USD/MWh)'
}

# Stringency order for consistent plotting
STRINGENCY_ORDER = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'UNKNOWN']

def load_stringency_mapping():
    """Load proper AR6 stringency mapping from metadata"""
    try:
        mapping_df = pd.read_csv('ar6_proper_stringency_mapping.csv')
        mapping_dict = {}
        for _, row in mapping_df.iterrows():
            key = f"{row['scenario_provider']}|{row['scenario']}"
            mapping_dict[key] = row['stringency_correct']
        print(f"   Loaded stringency mapping for {len(mapping_dict)} scenario combinations")
        return mapping_dict
    except FileNotFoundError:
        print("   Warning: ar6_proper_stringency_mapping.csv not found!")
        return {}

def load_and_filter_data():
    """Load and filter data with proper stringency mapping"""
    print("📂 Loading 4_final_AR6_gapfilled.csv...")
    df = pd.read_csv('../4_final_AR6_gapfilled.csv', low_memory=False)
    print(f"   Loaded {len(df):,} rows")
    
    # Create a copy for filtering
    df_original = df.copy()
    
    # Parse gap_filled_columns and set those columns to NaN
    print("🔍 Filtering out gap-filled values...")
    gap_filled_count = 0
    
    for idx, row in df.iterrows():
        if idx % 100000 == 0:
            print(f"   Processed {idx:,} rows...")
            
        gap_filled_str = row.get('gap_filled_columns')
        if pd.isna(gap_filled_str) or gap_filled_str == '':
            continue
            
        # Parse the gap-filled columns
        gap_filled_cols = []
        for col_str in str(gap_filled_str).split(','):
            base_col = col_str.split('(')[0].strip()
            if base_col in METRICS:
                gap_filled_cols.append(base_col)
        
        # Set gap-filled values to NaN
        for col in gap_filled_cols:
            if col in df_original.columns:
                df_original.at[idx, col] = np.nan
                gap_filled_count += 1
    
    print(f"   Set {gap_filled_count:,} gap-filled values to NaN")
    
    # Add proper stringency mapping
    print("🗺️  Applying proper stringency mapping...")
    mapping_dict = load_stringency_mapping()
    
    def get_stringency(row):
        key = f"{row['scenario_provider']}|{row['scenario']}"
        return mapping_dict.get(key, 'UNKNOWN')
    
    df_original['stringency'] = df_original.apply(get_stringency, axis=1)
    
    # Show stringency distribution
    stringency_counts = df_original['stringency'].value_counts()
    print("   Stringency distribution:")
    print(stringency_counts)
    
    # Filter to technologies with sufficient data
    print("📊 Filtering technologies with sufficient data...")
    tech_data_counts = {}
    
    for tech in df_original['technology'].unique():
        if pd.isna(tech):
            continue
        tech_df = df_original[df_original['technology'] == tech]
        
        # Count non-null values across all metrics
        data_count = 0
        for metric in METRICS:
            if metric in tech_df.columns:
                data_count += tech_df[metric].notna().sum()
        
        if data_count > 50:  # Minimum threshold
            tech_data_counts[tech] = data_count
    
    # Sort technologies by data availability
    sorted_techs = sorted(tech_data_counts.items(), key=lambda x: x[1], reverse=True)
    selected_techs = [tech for tech, count in sorted_techs[:20]]  # Top 20
    
    print(f"   Selected {len(selected_techs)} technologies with sufficient data")
    
    return df_original, selected_techs

# [Include the rest of the violin plotting functions from the original script]
# ... [rest of the plotting code] ...

if __name__ == "__main__":
    print("🎻 AR6 Violin Plot Generator (Proper Stringency Mapping)")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("plots_proper_stringency")
    output_dir.mkdir(exist_ok=True)
    
    # Load and process data
    try:
        df_original, selected_techs = load_and_filter_data()
        print(f"\n✅ Ready to generate plots with proper stringency mapping!")
        print(f"   Output directory: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. ar6_proper_stringency_mapping.csv exists in this directory")
        print("2. ../4_final_AR6_gapfilled.csv exists")
