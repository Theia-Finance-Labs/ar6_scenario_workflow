"""
AR6 Gap Analysis Script
=======================

This script analyzes gaps in the final AR6 dataset to understand why certain columns
aren't getting filled properly, especially for the required columns:
- scenario_pathway
- lifetime_years  
- efficiency_decimal (if Power or Renewables)
- scenario_price
- om_cost_usd_per_mw_per_yr
- capital_cost_usd_per_mw

Author: Energy Data Processing Pipeline
"""

import modin.pandas as pd
import numpy as np

print("=" * 80)
print("AR6 GAP ANALYSIS")
print("=" * 80)

# Load the final aggregated dataset
try:
    df = pd.read_csv("4_final_AR6_aggregated.csv")
    print(f"✅ Loaded final dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print("❌ File not found: 4_final_AR6_aggregated.csv")
    print("Please run step 4 first: python3 4_aggregateTechnologies.py")
    exit(1)

# Define the critical columns that need to be complete
critical_columns = [
    'scenario_pathway',
    'lifetime_years',
    'efficiency_decimal',
    'scenario_price',
    'om_cost_usd_per_mw_per_yr',
    'capital_cost_usd_per_mw'
]

# Filter to only include columns that exist
existing_critical_columns = [col for col in critical_columns if col in df.columns]
missing_columns = [col for col in critical_columns if col not in df.columns]

print(f"\n📋 Critical columns analysis:")
print(f"   Existing columns: {existing_critical_columns}")
if missing_columns:
    print(f"   Missing columns: {missing_columns}")

# Analyze gaps by column
print(f"\n🔍 Gap analysis by column:")
for col in existing_critical_columns:
    total_rows = len(df)
    missing_count = df[col].isna().sum()
    completeness = (total_rows - missing_count) / total_rows * 100
    
    print(f"   {col}:")
    print(f"     • Completeness: {completeness:.1f}% ({missing_count:,} missing)")
    
    # Analyze by sector
    if 'sector' in df.columns:
        print(f"     • By sector:")
        for sector in sorted(df['sector'].unique()):
            sector_data = df[df['sector'] == sector]
            sector_missing = sector_data[col].isna().sum()
            sector_total = len(sector_data)
            sector_completeness = (sector_total - sector_missing) / sector_total * 100
            print(f"       - {sector}: {sector_completeness:.1f}% ({sector_missing:,}/{sector_total:,})")

# Analyze gaps by technology
print(f"\n🔧 Gap analysis by technology:")
for col in existing_critical_columns:
    print(f"   {col} by technology:")
    tech_analysis = df.groupby('technology')[col].apply(lambda x: x.isna().sum()).sort_values(ascending=False)
    for tech, missing_count in tech_analysis.head(10).items():
        if missing_count > 0:
            tech_total = len(df[df['technology'] == tech])
            tech_completeness = (tech_total - missing_count) / tech_total * 100
            print(f"     • {tech}: {tech_completeness:.1f}% ({missing_count:,}/{tech_total:,})")

# Analyze gaps by scenario type
print(f"\n📊 Gap analysis by scenario type:")
for col in existing_critical_columns:
    print(f"   {col} by scenario_type:")
    scenario_analysis = df.groupby('scenario_type')[col].apply(lambda x: x.isna().sum())
    for scenario_type, missing_count in scenario_analysis.items():
        scenario_total = len(df[df['scenario_type'] == scenario_type])
        scenario_completeness = (scenario_total - missing_count) / scenario_total * 100
        print(f"     • {scenario_type}: {scenario_completeness:.1f}% ({missing_count:,}/{scenario_total:,})")

# Analyze gaps by geography
print(f"\n🌍 Gap analysis by geography:")
for col in existing_critical_columns:
    print(f"   {col} by scenario_geography:")
    geo_analysis = df.groupby('scenario_geography')[col].apply(lambda x: x.isna().sum()).sort_values(ascending=False)
    for geo, missing_count in geo_analysis.head(10).items():
        if missing_count > 0:
            geo_total = len(df[df['scenario_geography'] == geo])
            geo_completeness = (geo_total - missing_count) / geo_total * 100
            print(f"     • {geo}: {geo_completeness:.1f}% ({missing_count:,}/{geo_total:,})")

# Special analysis for efficiency_decimal (Power and Renewables only)
if 'efficiency_decimal' in df.columns:
    print(f"\n⚡ Efficiency analysis for Power and Renewables:")
    power_renewables = df[df['sector'].isin(['Power', 'Renewables'])]
    if len(power_renewables) > 0:
        missing_efficiency = power_renewables['efficiency_decimal'].isna().sum()
        total_pr = len(power_renewables)
        pr_completeness = (total_pr - missing_efficiency) / total_pr * 100
        print(f"   Power + Renewables efficiency completeness: {pr_completeness:.1f}% ({missing_efficiency:,}/{total_pr:,})")
        
        # By technology within Power/Renewables
        print(f"   By technology:")
        for tech in sorted(power_renewables['technology'].unique()):
            tech_data = power_renewables[power_renewables['technology'] == tech]
            tech_missing = tech_data['efficiency_decimal'].isna().sum()
            tech_total = len(tech_data)
            tech_completeness = (tech_total - tech_missing) / tech_total * 100
            print(f"     • {tech}: {tech_completeness:.1f}% ({tech_missing:,}/{tech_total:,})")

# Analyze rows that would be dropped with complete case filtering
print(f"\n🎯 Complete case analysis:")
complete_case_mask = df[existing_critical_columns].notna().all(axis=1)
complete_rows = complete_case_mask.sum()
total_rows = len(df)
dropped_rows = total_rows - complete_rows

print(f"   Rows with all critical columns complete: {complete_rows:,}/{total_rows:,} ({complete_rows/total_rows*100:.1f}%)")
print(f"   Rows that would be dropped: {dropped_rows:,} ({dropped_rows/total_rows*100:.1f}%)")

# Analyze what's causing the drops
print(f"\n🔍 Analysis of dropped rows:")
for col in existing_critical_columns:
    col_missing = df[col].isna().sum()
    print(f"   {col}: {col_missing:,} missing values")

# Show sample of incomplete rows
print(f"\n📋 Sample of incomplete rows:")
incomplete_rows = df[~complete_case_mask].head(10)
for idx, row in incomplete_rows.iterrows():
    missing_cols = [col for col in existing_critical_columns if pd.isna(row[col])]
    print(f"   Row {idx}: {row['sector']} - {row['technology']} - Missing: {missing_cols}")

print(f"\n" + "=" * 80)
print("GAP ANALYSIS COMPLETE")
print("=" * 80) 