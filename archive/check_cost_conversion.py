"""
Check cost conversion logic and identify unit conversion issues
"""

import modin.pandas as pd
import numpy as np

print("=" * 80)
print("COST CONVERSION DIAGNOSIS")
print("=" * 80)

# Load step 1 data to see original units
print("🔍 CHECKING ORIGINAL COST DATA AND UNITS:")
print("=" * 50)

step1_df = pd.read_csv("1_intermediate_AR6_scenario_formatting_ISO3.csv")
print(f"✅ Loaded step 1 data: {step1_df.shape[0]:,} rows")

# Check OM cost data
print(f"\n📊 OM COST ANALYSIS:")
om_cost_data = step1_df[step1_df['om_cost'].notna()].copy()
print(f"   Total OM cost rows: {len(om_cost_data):,}")

if len(om_cost_data) > 0:
    print(f"   OM Cost units:")
    unit_counts = om_cost_data['om_cost_unit'].value_counts()
    for unit, count in unit_counts.items():
        print(f"     • {unit}: {count:,} rows")
    
    print(f"   OM Cost value statistics:")
    om_stats = om_cost_data['om_cost'].describe()
    print(f"     • Min: {om_stats['min']:,.2f}")
    print(f"     • Max: {om_stats['max']:,.2f}")
    print(f"     • Mean: {om_stats['mean']:,.2f}")
    print(f"     • Median: {om_stats['50%']:,.2f}")
    
    # Check for negative values
    negative_om = om_cost_data[om_cost_data['om_cost'] < 0]
    print(f"   Negative OM costs: {len(negative_om):,} rows")
    if len(negative_om) > 0:
        print(f"     • Min negative: {negative_om['om_cost'].min():,.2f}")
        print(f"     • Max negative: {negative_om['om_cost'].max():,.2f}")

# Check Capital cost data
print(f"\n📊 CAPITAL COST ANALYSIS:")
capital_cost_data = step1_df[step1_df['capital_cost'].notna()].copy()
print(f"   Total Capital cost rows: {len(capital_cost_data):,}")

if len(capital_cost_data) > 0:
    print(f"   Capital Cost units:")
    unit_counts = capital_cost_data['capital_cost_unit'].value_counts()
    for unit, count in unit_counts.items():
        print(f"     • {unit}: {count:,} rows")
    
    print(f"   Capital Cost value statistics:")
    capital_stats = capital_cost_data['capital_cost'].describe()
    print(f"     • Min: {capital_stats['min']:,.2f}")
    print(f"     • Max: {capital_stats['max']:,.2f}")
    print(f"     • Mean: {capital_stats['mean']:,.2f}")
    print(f"     • Median: {capital_stats['50%']:,.2f}")
    
    # Check for negative values
    negative_capital = capital_cost_data[capital_cost_data['capital_cost'] < 0]
    print(f"   Negative Capital costs: {len(negative_capital):,} rows")
    if len(negative_capital) > 0:
        print(f"     • Min negative: {negative_capital['capital_cost'].min():,.2f}")
        print(f"     • Max negative: {negative_capital['capital_cost'].max():,.2f}")

# Check step 2 data to see what happened during conversion
print(f"\n🔍 CHECKING STEP 2 CONVERSION RESULTS:")
print("=" * 50)

step2_df = pd.read_csv("2_final_AR6_filtered.csv")
print(f"✅ Loaded step 2 data: {step2_df.shape[0]:,} rows")

# Check cost columns in step 2
cost_cols = [col for col in step2_df.columns if 'cost' in col.lower()]
print(f"   Cost columns in step 2: {cost_cols}")

if 'om_cost_usd_per_mw_per_yr' in step2_df.columns:
    om_step2 = step2_df[step2_df['om_cost_usd_per_mw_per_yr'].notna()]
    print(f"\n📊 Step 2 OM Cost statistics:")
    om_step2_stats = om_step2['om_cost_usd_per_mw_per_yr'].describe()
    print(f"   • Min: {om_step2_stats['min']:,.2f}")
    print(f"   • Max: {om_step2_stats['max']:,.2f}")
    print(f"   • Mean: {om_step2_stats['mean']:,.2f}")
    print(f"   • Median: {om_step2_stats['50%']:,.2f}")
    
    # Check for extreme values
    extreme_om = om_step2[om_step2['om_cost_usd_per_mw_per_yr'] > 100000]
    print(f"   • OM Cost > $100k/MW/yr: {len(extreme_om):,} rows")
    
    negative_om_step2 = om_step2[om_step2['om_cost_usd_per_mw_per_yr'] < 0]
    print(f"   • Negative OM costs: {len(negative_om_step2):,} rows")

if 'capital_cost_usd_per_mw' in step2_df.columns:
    capital_step2 = step2_df[step2_df['capital_cost_usd_per_mw'].notna()]
    print(f"\n📊 Step 2 Capital Cost statistics:")
    capital_step2_stats = capital_step2['capital_cost_usd_per_mw'].describe()
    print(f"   • Min: {capital_step2_stats['min']:,.2f}")
    print(f"   • Max: {capital_step2_stats['max']:,.2f}")
    print(f"   • Mean: {capital_step2_stats['mean']:,.2f}")
    print(f"   • Median: {capital_step2_stats['50%']:,.2f}")
    
    # Check for extreme values
    extreme_capital = capital_step2[capital_step2['capital_cost_usd_per_mw'] > 10000000]
    print(f"   • Capital Cost > $10M/MW: {len(extreme_capital):,} rows")
    
    negative_capital_step2 = capital_step2[capital_step2['capital_cost_usd_per_mw'] < 0]
    print(f"   • Negative Capital costs: {len(negative_capital_step2):,} rows")

# Check by technology to see which ones have issues
print(f"\n🔍 CHECKING BY TECHNOLOGY:")
print("=" * 40)

if 'om_cost_usd_per_mw_per_yr' in step2_df.columns and 'Technology' in step2_df.columns:
    tech_om_analysis = step2_df.groupby('Technology')['om_cost_usd_per_mw_per_yr'].agg(['count', 'mean', 'min', 'max']).round(0)
    print(f"📊 OM Cost by technology (top 10 by count):")
    tech_om_analysis = tech_om_analysis.sort_values('count', ascending=False).head(10)
    for tech, row in tech_om_analysis.iterrows():
        print(f"   • {tech}: {row['count']:,} rows, mean=${row['mean']:,.0f}, min=${row['min']:,.0f}, max=${row['max']:,.0f}")

if 'capital_cost_usd_per_mw' in step2_df.columns and 'Technology' in step2_df.columns:
    tech_capital_analysis = step2_df.groupby('Technology')['capital_cost_usd_per_mw'].agg(['count', 'mean', 'min', 'max']).round(0)
    print(f"\n📊 Capital Cost by technology (top 10 by count):")
    tech_capital_analysis = tech_capital_analysis.sort_values('count', ascending=False).head(10)
    for tech, row in tech_capital_analysis.iterrows():
        print(f"   • {tech}: {row['count']:,} rows, mean=${row['mean']:,.0f}, min=${row['min']:,.0f}, max=${row['max']:,.0f}")

print(f"\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80) 