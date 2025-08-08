"""
Diagnose cost data issues and check for unit conversion problems
"""

import modin.pandas as pd
import numpy as np

print("=" * 80)
print("COST DATA DIAGNOSIS")
print("=" * 80)

# Load the final aggregated dataset
df = pd.read_csv("4_final_AR6_aggregated_complete.csv")
print(f"✅ Loaded complete dataset: {df.shape[0]:,} rows")

# Check cost data statistics
print(f"\n💰 COST DATA STATISTICS:")
print(f"=" * 40)

# OM Cost analysis
om_cost_stats = df['om_cost_usd_per_mw_per_yr'].describe()
print(f"📊 OM Cost (USD/MW/yr) statistics:")
print(f"   Min: {om_cost_stats['min']:,.0f}")
print(f"   Max: {om_cost_stats['max']:,.0f}")
print(f"   Mean: {om_cost_stats['mean']:,.0f}")
print(f"   Median: {om_cost_stats['50%']:,.0f}")
print(f"   Std: {om_cost_stats['std']:,.0f}")

# Capital Cost analysis
capital_cost_stats = df['capital_cost_usd_per_mw'].describe()
print(f"\n📊 Capital Cost (USD/MW) statistics:")
print(f"   Min: {capital_cost_stats['min']:,.0f}")
print(f"   Max: {capital_cost_stats['max']:,.0f}")
print(f"   Mean: {capital_cost_stats['mean']:,.0f}")
print(f"   Median: {capital_cost_stats['50%']:,.0f}")
print(f"   Std: {capital_cost_stats['std']:,.0f}")

# Check for extreme values
print(f"\n🚨 EXTREME VALUES ANALYSIS:")
print(f"=" * 40)

# Find very high values
om_cost_high = df[df['om_cost_usd_per_mw_per_yr'] > 100000]
capital_cost_high = df[df['capital_cost_usd_per_mw'] > 10000000]

print(f"📊 OM Cost > $100k/MW/yr: {len(om_cost_high):,} rows")
print(f"📊 Capital Cost > $10M/MW: {len(capital_cost_high):,} rows")

if len(om_cost_high) > 0:
    print(f"\n🔍 Sample of high OM costs:")
    high_om_sample = om_cost_high[['sector', 'technology', 'om_cost_usd_per_mw_per_yr']].head(10)
    for _, row in high_om_sample.iterrows():
        print(f"   {row['sector']} - {row['technology']}: ${row['om_cost_usd_per_mw_per_yr']:,.0f}/MW/yr")

if len(capital_cost_high) > 0:
    print(f"\n🔍 Sample of high Capital costs:")
    high_capital_sample = capital_cost_high[['sector', 'technology', 'capital_cost_usd_per_mw']].head(10)
    for _, row in high_capital_sample.iterrows():
        print(f"   {row['sector']} - {row['technology']}: ${row['capital_cost_usd_per_mw']:,.0f}/MW")

# Check by technology
print(f"\n📊 COST ANALYSIS BY TECHNOLOGY:")
print(f"=" * 40)

tech_cost_analysis = df.groupby('technology').agg({
    'om_cost_usd_per_mw_per_yr': ['mean', 'median', 'min', 'max'],
    'capital_cost_usd_per_mw': ['mean', 'median', 'min', 'max']
}).round(0)

for tech in sorted(df['technology'].unique()):
    tech_data = df[df['technology'] == tech]
    om_mean = tech_data['om_cost_usd_per_mw_per_yr'].mean()
    capital_mean = tech_data['capital_cost_usd_per_mw'].mean()
    print(f"   {tech}:")
    print(f"     • OM Cost: ${om_mean:,.0f}/MW/yr")
    print(f"     • Capital Cost: ${capital_mean:,.0f}/MW")

# Check source data from step 2
print(f"\n🔍 CHECKING SOURCE DATA FROM STEP 2:")
print(f"=" * 40)

try:
    step2_df = pd.read_csv("2_final_AR6_filtered.csv")
    print(f"✅ Loaded step 2 data: {step2_df.shape[0]:,} rows")
    
    # Check original cost columns
    if 'om_cost_usd_per_mw_per_yr' in step2_df.columns:
        step2_om_stats = step2_df['om_cost_usd_per_mw_per_yr'].describe()
        print(f"\n📊 Step 2 OM Cost statistics:")
        print(f"   Min: {step2_om_stats['min']:,.0f}")
        print(f"   Max: {step2_om_stats['max']:,.0f}")
        print(f"   Mean: {step2_om_stats['mean']:,.0f}")
        print(f"   Median: {step2_om_stats['50%']:,.0f}")
    
    if 'capital_cost_usd_per_mw' in step2_df.columns:
        step2_capital_stats = step2_df['capital_cost_usd_per_mw'].describe()
        print(f"\n📊 Step 2 Capital Cost statistics:")
        print(f"   Min: {step2_capital_stats['min']:,.0f}")
        print(f"   Max: {step2_capital_stats['max']:,.0f}")
        print(f"   Mean: {step2_capital_stats['mean']:,.0f}")
        print(f"   Median: {step2_capital_stats['50%']:,.0f}")
        
except FileNotFoundError:
    print("❌ Step 2 file not found")

# Check step 1 data for original units
print(f"\n🔍 CHECKING STEP 1 DATA FOR ORIGINAL UNITS:")
print(f"=" * 40)

try:
    step1_df = pd.read_csv("1_intermediate_AR6_scenario_formatting_ISO3.csv")
    print(f"✅ Loaded step 1 data: {step1_df.shape[0]:,} rows")
    
    # Check for cost-related columns
    cost_cols = [col for col in step1_df.columns if 'cost' in col.lower()]
    print(f"📋 Cost-related columns in step 1: {cost_cols}")
    
    # Check for unit columns
    unit_cols = [col for col in step1_df.columns if 'unit' in col.lower()]
    print(f"📋 Unit-related columns in step 1: {unit_cols}")
    
    if 'om_cost_unit' in step1_df.columns:
        print(f"\n📊 OM Cost units in step 1:")
        unit_counts = step1_df['om_cost_unit'].value_counts()
        for unit, count in unit_counts.items():
            print(f"   {unit}: {count:,} rows")
    
    if 'capital_cost_unit' in step1_df.columns:
        print(f"\n📊 Capital Cost units in step 1:")
        unit_counts = step1_df['capital_cost_unit'].value_counts()
        for unit, count in unit_counts.items():
            print(f"   {unit}: {count:,} rows")
            
except FileNotFoundError:
    print("❌ Step 1 file not found")

print(f"\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80) 