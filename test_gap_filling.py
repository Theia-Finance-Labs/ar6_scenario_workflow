"""
Test different gap-filling strategies to improve coverage
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("TESTING GAP-FILLING STRATEGIES")
print("=" * 80)

# Load the data
df = pd.read_csv("4_final_AR6_aggregated.csv")
print(f"✅ Loaded dataset: {df.shape[0]:,} rows")

# Test different grouping strategies for cost columns
cost_columns = ['om_cost_usd_per_mw_per_yr', 'capital_cost_usd_per_mw']

print(f"\n🔍 Testing different grouping strategies for cost columns:")

# Strategy 1: Current approach (includes scenario_type)
print(f"\n📊 Strategy 1: Current (includes scenario_type)")
for col in cost_columns:
    if col in df.columns:
        initial_missing = df[col].isna().sum()
        print(f"   {col}: {initial_missing:,} missing initially")

# Strategy 2: Remove scenario_type from grouping
print(f"\n📊 Strategy 2: Remove scenario_type (technology + geography only)")
for col in cost_columns:
    if col in df.columns:
        initial_missing = df[col].isna().sum()
        
        # Test grouping by technology + geography only
        test_df = df.copy()
        medians = test_df.groupby(['technology', 'scenario_geography'])[col].median()
        
        # Count how many missing values could be filled
        fillable_count = 0
        for idx, row in test_df.iterrows():
            if pd.isna(row[col]):
                key = (row['technology'], row['scenario_geography'])
                if key in medians and pd.notna(medians[key]):
                    fillable_count += 1
        
        print(f"   {col}: {fillable_count:,} could be filled (vs {initial_missing:,} missing)")

# Strategy 3: Technology only
print(f"\n📊 Strategy 3: Technology only")
for col in cost_columns:
    if col in df.columns:
        initial_missing = df[col].isna().sum()
        
        # Test grouping by technology only
        test_df = df.copy()
        medians = test_df.groupby(['technology'])[col].median()
        
        # Count how many missing values could be filled
        fillable_count = 0
        for idx, row in test_df.iterrows():
            if pd.isna(row[col]):
                key = row['technology']
                if key in medians and pd.notna(medians[key]):
                    fillable_count += 1
        
        print(f"   {col}: {fillable_count:,} could be filled (vs {initial_missing:,} missing)")

# Strategy 4: Sector only
print(f"\n📊 Strategy 4: Sector only")
for col in cost_columns:
    if col in df.columns:
        initial_missing = df[col].isna().sum()
        
        # Test grouping by sector only
        test_df = df.copy()
        medians = test_df.groupby(['sector'])[col].median()
        
        # Count how many missing values could be filled
        fillable_count = 0
        for idx, row in test_df.iterrows():
            if pd.isna(row[col]):
                key = row['sector']
                if key in medians and pd.notna(medians[key]):
                    fillable_count += 1
        
        print(f"   {col}: {fillable_count:,} could be filled (vs {initial_missing:,} missing)")

# Test efficiency gap-filling without scenario_type
print(f"\n⚡ Testing efficiency gap-filling (technology only):")
if 'efficiency_decimal' in df.columns:
    initial_missing = df['efficiency_decimal'].isna().sum()
    
    # Test grouping by technology only
    test_df = df.copy()
    medians = test_df.groupby(['technology'])['efficiency_decimal'].median()
    
    # Count how many missing values could be filled
    fillable_count = 0
    for idx, row in test_df.iterrows():
        if pd.isna(row['efficiency_decimal']):
            key = row['technology']
            if key in medians and pd.notna(medians[key]):
                fillable_count += 1
    
    print(f"   efficiency_decimal: {fillable_count:,} could be filled (vs {initial_missing:,} missing)")

print(f"\n" + "=" * 80)
print("TESTING COMPLETE")
print("=" * 80) 