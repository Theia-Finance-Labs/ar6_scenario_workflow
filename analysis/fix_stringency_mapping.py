#!/usr/bin/env python3
"""
Fix Stringency Mapping for AR6 Scenarios
========================================

This script fixes the stringency mapping issue by creating a proper mapping from 
temperature/CO2 targets to stringency categories (C1-C8).
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_temperature_to_stringency_mapping():
    """
    Create mapping from temperature/CO2 targets to stringency categories
    Based on typical AR6 categorization:
    - C1: Below 1.5°C (< 400-500 ppm CO2)
    - C2: 1.5°C (500-600 ppm CO2)  
    - C3: Well below 2°C (600-800 ppm CO2)
    - C4: Below 2°C (800-1000 ppm CO2)
    - C5: Around 2°C (1000-1200 ppm CO2)
    - C6: 2-3°C (1200-1400 ppm CO2)
    - C7: 3-4°C (1400-1600 ppm CO2) 
    - C8: Above 4°C (>1600 ppm CO2)
    """
    
    # Temperature/CO2 target to stringency mapping
    temp_mapping = {
        '300': 'C1',  # Very ambitious
        '400': 'C1',  # Below 1.5°C
        '500': 'C2',  # 1.5°C
        '600': 'C2',  # 1.5°C
        '700': 'C3',  # Well below 2°C  
        '800': 'C3',  # Well below 2°C
        '900': 'C4',  # Below 2°C
        '1000': 'C4', # Below 2°C
        '1100': 'C5', # Around 2°C
        '1200': 'C5', # Around 2°C
        '1300': 'C6', # 2-3°C
        '1400': 'C6', # 2-3°C
        '1500': 'C7', # 3-4°C
        '1600': 'C7', # 3-4°C
        '1700': 'C8', # Above 4°C
        '1800': 'C8', # Above 4°C
        '1900': 'C8', # Above 4°C
        '2000': 'C8', # Above 4°C
    }
    
    return temp_mapping

def extract_stringency_from_scenario(scenario, temp_mapping):
    """
    Extract stringency from scenario name using temperature/CO2 target patterns
    """
    if pd.isna(scenario):
        return 'UNKNOWN'
    
    scenario_str = str(scenario)
    
    # First, try exact C1-C8 pattern (for scenarios that already have it)
    for stringency in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']:
        if stringency in scenario_str:
            return stringency
    
    # Then try temperature/CO2 target mapping
    # Look for numbers in the scenario name (prioritize longer matches)
    for temp_target in sorted(temp_mapping.keys(), key=len, reverse=True):
        if temp_target in scenario_str:
            return temp_mapping[temp_target]
    
    # Special cases for common AR6 scenario patterns
    if 'NPi' in scenario_str:  # No Policy Implementation
        return 'C8'  # Usually high emissions
    elif 'INDC' in scenario_str and any(x in scenario_str for x in ['2030', '2050']):
        # INDC scenarios are usually moderate ambition
        return 'C5'
    
    return 'UNKNOWN'

def analyze_stringency_mapping():
    """
    Analyze and fix stringency mapping in the AR6 data
    """
    print("🔧 Fixing AR6 Stringency Mapping")
    print("=" * 50)
    
    # Load data
    print("📂 Loading 4_final_AR6_gapfilled_complete.csv...")
    df = pd.read_csv('../4_final_AR6_gapfilled_complete.csv', low_memory=False)
    
    # Create temperature mapping
    temp_mapping = create_temperature_to_stringency_mapping()
    
    # Apply new stringency mapping
    print("🗺️  Applying improved stringency mapping...")
    df['stringency_fixed'] = df['scenario'].apply(
        lambda x: extract_stringency_from_scenario(x, temp_mapping)
    )
    
    # Compare old vs new mapping
    print("\\n📊 Comparison of mapping approaches:")
    
    # Old approach (simple C1-C8 substring search)
    df['stringency_old'] = 'UNKNOWN'
    for stringency in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']:
        mask = df['scenario'].str.contains(stringency, na=False)
        df.loc[mask, 'stringency_old'] = stringency
    
    print("Old approach (substring search):")
    old_counts = df['stringency_old'].value_counts()
    print(old_counts)
    
    print("\\nNew approach (temperature/CO2 target mapping):")
    new_counts = df['stringency_fixed'].value_counts()
    print(new_counts)
    
    print("\\n📈 Improvement:")
    old_unknown = old_counts.get('UNKNOWN', 0)
    new_unknown = new_counts.get('UNKNOWN', 0)
    improvement = old_unknown - new_unknown
    print(f"Reduced UNKNOWN scenarios by {improvement:,} ({improvement/old_unknown*100:.1f}%)")
    
    # Show examples of successful mapping
    print("\\n🎯 Examples of successful mapping:")
    successful_mapping = df[
        (df['stringency_old'] == 'UNKNOWN') & 
        (df['stringency_fixed'] != 'UNKNOWN')
    ][['scenario', 'stringency_fixed']].drop_duplicates().head(10)
    
    for _, row in successful_mapping.iterrows():
        print(f"  {row['scenario']} → {row['stringency_fixed']}")
    
    # Show remaining UNKNOWN cases
    if new_unknown > 0:
        print(f"\\n⚠️  Remaining {new_unknown:,} UNKNOWN scenarios:")
        remaining_unknown = df[df['stringency_fixed'] == 'UNKNOWN']['scenario'].unique()[:10]
        for scenario in remaining_unknown:
            print(f"  {scenario}")
    
    # Save corrected mapping for reference
    mapping_df = df[['scenario_provider', 'scenario', 'stringency_old', 'stringency_fixed']].drop_duplicates()
    mapping_df.to_csv('scenario_stringency_mapping.csv', index=False)
    print(f"\\n💾 Saved scenario mapping to: scenario_stringency_mapping.csv")
    
    return df, temp_mapping

def update_violin_plot_script(temp_mapping):
    """
    Create an updated violin plot script with correct stringency mapping
    """
    print("\\n🎻 Creating updated violin plot script...")
    
    # Read the original script
    with open('generate_violin_plots.py', 'r') as f:
        original_script = f.read()
    
    # Create the improved stringency mapping function
    mapping_function = f'''
def extract_stringency_from_scenario(scenario, temp_mapping):
    """Extract stringency from scenario name using temperature/CO2 target patterns"""
    if pd.isna(scenario):
        return 'UNKNOWN'
    
    scenario_str = str(scenario)
    
    # First, try exact C1-C8 pattern (for scenarios that already have it)
    for stringency in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']:
        if stringency in scenario_str:
            return stringency
    
    # Then try temperature/CO2 target mapping
    for temp_target in sorted(temp_mapping.keys(), key=len, reverse=True):
        if temp_target in scenario_str:
            return temp_mapping[temp_target]
    
    # Special cases
    if 'NPi' in scenario_str:
        return 'C8'
    elif 'INDC' in scenario_str and any(x in scenario_str for x in ['2030', '2050']):
        return 'C5'
    
    return 'UNKNOWN'

# Temperature/CO2 target to stringency mapping
TEMP_MAPPING = {temp_mapping}
'''
    
    # Replace the old stringency assignment with the new one
    old_pattern = '''    # Add stringency column (extract from scenario or use UNKNOWN)
    df_original['stringency'] = 'UNKNOWN'
    for stringency in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']:
        mask = df_original['scenario'].str.contains(stringency, na=False)
        df_original.loc[mask, 'stringency'] = stringency'''
    
    new_pattern = '''    # Add stringency column using improved mapping
    df_original['stringency'] = df_original['scenario'].apply(
        lambda x: extract_stringency_from_scenario(x, TEMP_MAPPING)
    )'''
    
    # Create updated script
    updated_script = mapping_function + "\\n" + original_script.replace(old_pattern, new_pattern)
    
    # Save updated script
    with open('generate_violin_plots_fixed.py', 'w') as f:
        f.write(updated_script)
    
    print("   ✅ Created generate_violin_plots_fixed.py with improved stringency mapping")

def main():
    """Main function"""
    df, temp_mapping = analyze_stringency_mapping()
    update_violin_plot_script(temp_mapping)
    
    print("\\n✅ Stringency mapping analysis complete!")
    print("\\nNext steps:")
    print("1. Review scenario_stringency_mapping.csv for the mapping results")
    print("2. Use generate_violin_plots_fixed.py for future plotting")
    print("3. If you have the AR6 metadata Excel file, we can create exact mappings")

if __name__ == "__main__":
    main()