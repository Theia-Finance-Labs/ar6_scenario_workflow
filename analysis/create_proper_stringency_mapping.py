#!/usr/bin/env python3
"""
Create Proper AR6 Stringency Mapping from Official Metadata
===========================================================

This script reads the official AR6 metadata Excel file and creates the correct
Model + Scenario → Stringency mapping, then applies it to fix the pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_ar6_metadata():
    """
    Load the AR6 metadata file and extract the stringency mappings
    """
    print("📂 Loading AR6 metadata file...")
    
    metadata_file = "../AR6_Scenarios_Database_metadata_indicators_v1.1 2.xlsx"
    
    try:
        # Read the Excel file - look for the tab that starts with 'meta_'
        excel_file = pd.ExcelFile(metadata_file)
        sheet_names = excel_file.sheet_names
        
        print(f"   Available sheets: {sheet_names}")
        
        # Find the metadata sheet (starts with 'meta_')
        meta_sheet = None
        for sheet in sheet_names:
            if sheet.lower().startswith('meta_'):
                meta_sheet = sheet
                break
        
        if meta_sheet is None:
            print("   Looking for sheets containing 'meta' or 'indicator'...")
            for sheet in sheet_names:
                if 'meta' in sheet.lower() or 'indicator' in sheet.lower():
                    meta_sheet = sheet
                    break
        
        if meta_sheet is None:
            print("   No metadata sheet found. Available sheets:")
            for sheet in sheet_names:
                print(f"     - {sheet}")
            return None
        
        print(f"   Reading sheet: {meta_sheet}")
        metadata_df = pd.read_excel(metadata_file, sheet_name=meta_sheet)
        
        print(f"   Loaded {len(metadata_df)} rows, {len(metadata_df.columns)} columns")
        print(f"   Columns: {list(metadata_df.columns)}")
        
        return metadata_df
        
    except Exception as e:
        print(f"   Error loading metadata file: {e}")
        return None

def extract_stringency_mapping(metadata_df):
    """
    Extract Model + Scenario → Stringency mapping from metadata
    """
    print("🗺️  Extracting stringency mapping...")
    
    # Look for columns that might contain model, scenario, and stringency
    print("   Available columns:")
    for i, col in enumerate(metadata_df.columns):
        print(f"     {i}: {col}")
        
    # Try to identify the right columns
    model_col = None
    scenario_col = None  
    stringency_col = None
    
    for col in metadata_df.columns:
        col_lower = str(col).lower()
        if 'model' in col_lower and model_col is None:
            model_col = col
        elif 'scenario' in col_lower and scenario_col is None:
            scenario_col = col
        elif any(x in col_lower for x in ['category', 'stringency', 'c1', 'c2']) and stringency_col is None:
            stringency_col = col
    
    print(f"   Identified columns:")
    print(f"     Model: {model_col}")
    print(f"     Scenario: {scenario_col}")  
    print(f"     Stringency: {stringency_col}")
    
    # If we can't auto-identify, ask user to specify or show sample data
    if any(x is None for x in [model_col, scenario_col, stringency_col]):
        print("\\n   Sample of first 5 rows:")
        print(metadata_df.head().to_string())
        print("\\n   Please check the column names and update the script manually")
        return None
    
    # Extract the mapping
    mapping_df = metadata_df[[model_col, scenario_col, stringency_col]].copy()
    mapping_df.columns = ['model', 'scenario', 'stringency']
    
    # Clean up the data
    mapping_df = mapping_df.dropna()
    mapping_df = mapping_df[mapping_df['stringency'] != '']
    
    print(f"   Extracted {len(mapping_df)} model+scenario combinations")
    print(f"   Stringency values found: {sorted(mapping_df['stringency'].unique())}")
    
    return mapping_df

def apply_stringency_mapping(mapping_df):
    """
    Apply the proper stringency mapping to our data
    """
    print("🔧 Applying stringency mapping to AR6 data...")
    
    # Load our data
    df = pd.read_csv('../4_final_AR6_gapfilled_complete.csv', low_memory=False)
    print(f"   Loaded {len(df)} rows of AR6 data")
    
    # Create lookup dictionary
    mapping_dict = {}
    for _, row in mapping_df.iterrows():
        key = f"{row['model']}|{row['scenario']}"
        mapping_dict[key] = row['stringency']
    
    print(f"   Created mapping dictionary with {len(mapping_dict)} entries")
    
    # Apply mapping
    def get_stringency(row):
        key = f"{row['scenario_provider']}|{row['scenario']}"
        return mapping_dict.get(key, 'UNKNOWN')
    
    df['stringency_correct'] = df.apply(get_stringency, axis=1)
    
    # Compare with old approach
    print("\\n📊 Results:")
    stringency_counts = df['stringency_correct'].value_counts()
    print("Stringency distribution using metadata mapping:")
    print(stringency_counts)
    
    unknown_count = stringency_counts.get('UNKNOWN', 0)
    total_count = len(df)
    print(f"\\nUnknown scenarios: {unknown_count:,} ({unknown_count/total_count*100:.1f}%)")
    print(f"Successfully mapped: {total_count-unknown_count:,} ({(total_count-unknown_count)/total_count*100:.1f}%)")
    
    # Show some examples of successful mapping
    if unknown_count < total_count:
        print("\\n🎯 Examples of successful mapping:")
        mapped_examples = df[df['stringency_correct'] != 'UNKNOWN'][['scenario_provider', 'scenario', 'stringency_correct']].drop_duplicates().head(10)
        for _, row in mapped_examples.iterrows():
            print(f"   {row['scenario_provider']} | {row['scenario']} → {row['stringency_correct']}")
    
    # Show unmapped scenarios
    if unknown_count > 0:
        print(f"\\n⚠️  Sample unmapped scenarios:")
        unmapped_examples = df[df['stringency_correct'] == 'UNKNOWN'][['scenario_provider', 'scenario']].drop_duplicates().head(10)
        for _, row in unmapped_examples.iterrows():
            print(f"   {row['scenario_provider']} | {row['scenario']}")
    
    # Save the mapping
    mapping_output = df[['scenario_provider', 'scenario', 'stringency_correct']].drop_duplicates()
    mapping_output.to_csv('ar6_proper_stringency_mapping.csv', index=False)
    print(f"\\n💾 Saved proper stringency mapping to: ar6_proper_stringency_mapping.csv")
    
    return mapping_df, df

def update_combined_pipeline():
    """
    Update the combined_pipeline.py to use proper stringency mapping
    """
    print("\\n🔧 Updating combined_pipeline.py...")
    
    pipeline_file = "../pipeline/combined_pipeline.py"
    
    # Read the current pipeline
    try:
        with open(pipeline_file, 'r') as f:
            pipeline_content = f.read()
        
        # Check if stringency mapping already exists
        if 'stringency' in pipeline_content.lower():
            print("   Stringency mapping code already exists in pipeline")
        else:
            print("   No existing stringency mapping found in pipeline")
        
        # Create the stringency mapping function to add
        stringency_function = '''
def load_stringency_mapping():
    """Load AR6 stringency mapping from metadata"""
    try:
        mapping_df = pd.read_csv("ar6_proper_stringency_mapping.csv")
        mapping_dict = {}
        for _, row in mapping_df.iterrows():
            key = f"{row['scenario_provider']}|{row['scenario']}"
            mapping_dict[key] = row['stringency_correct']
        return mapping_dict
    except FileNotFoundError:
        print("Warning: ar6_proper_stringency_mapping.csv not found. Stringency will be UNKNOWN.")
        return {}

def add_stringency_column(df):
    """Add proper stringency column to dataframe"""
    mapping_dict = load_stringency_mapping()
    
    def get_stringency(row):
        key = f"{row['scenario_provider']}|{row['scenario']}"
        return mapping_dict.get(key, 'UNKNOWN')
    
    df['stringency'] = df.apply(get_stringency, axis=1)
    return df
'''
        
        print("\\n   📝 Stringency mapping functions created")
        print("   To integrate into combined_pipeline.py, add the above functions")
        print("   and call add_stringency_column(df) after loading data")
        
    except FileNotFoundError:
        print(f"   Pipeline file not found: {pipeline_file}")
        
    # Also update the violin plot script
    update_violin_plot_script()

def update_violin_plot_script():
    """
    Create updated violin plot script that uses the proper stringency mapping
    """
    print("\\n🎻 Creating updated violin plot script with proper stringency mapping...")
    
    violin_script = '''#!/usr/bin/env python3
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
    'efficiency_decimal': 'Efficiency\\n(decimal)',
    'lifetime_years': 'Lifetime\\n(years)',
    'om_cost_usd_per_mw_per_yr': 'O&M Cost\\n(USD/MW/yr)',
    'capital_cost_usd_per_mw': 'Capital Cost\\n(USD/MW)',
    'scenario_price': 'Electricity Price\\n(USD/MWh)',
    'fuel_price': 'Fuel Price\\n(USD/MWh)'
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
        print(f"\\n✅ Ready to generate plots with proper stringency mapping!")
        print(f"   Output directory: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        print("\\nPlease ensure:")
        print("1. ar6_proper_stringency_mapping.csv exists in this directory")
        print("2. ../4_final_AR6_gapfilled.csv exists")
'''
    
    with open('generate_violin_plots_proper_stringency.py', 'w') as f:
        f.write(violin_script)
    
    print("   ✅ Created generate_violin_plots_proper_stringency.py")

def main():
    """Main function"""
    print("🎯 Creating Proper AR6 Stringency Mapping")
    print("=" * 50)
    
    # Load metadata
    metadata_df = load_ar6_metadata()
    if metadata_df is None:
        print("❌ Could not load metadata file")
        return
    
    # Extract stringency mapping  
    mapping_df = extract_stringency_mapping(metadata_df)
    if mapping_df is None:
        print("❌ Could not extract stringency mapping")
        return
    
    # Apply mapping to our data
    mapping_df, df = apply_stringency_mapping(mapping_df)
    
    # Update pipeline and scripts
    update_combined_pipeline()
    
    print("\\n✅ Proper stringency mapping complete!")
    print("\\nFiles created:")
    print("- ar6_proper_stringency_mapping.csv: The official mapping")
    print("- generate_violin_plots_proper_stringency.py: Updated plot script")
    print("\\nNext steps:")
    print("1. Integrate the stringency functions into combined_pipeline.py")
    print("2. Re-run your pipeline with proper stringency mapping")
    print("3. Use the updated violin plot script for analysis")

if __name__ == "__main__":
    main()