"""
Analyze which technologies have no cost data at all
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("ANALYZING MISSING TECHNOLOGIES")
print("=" * 80)

# Load the data
df = pd.read_csv("4_final_AR6_aggregated.csv")
print(f"✅ Loaded dataset: {df.shape[0]:,} rows")

# Analyze cost data by technology
cost_columns = ['om_cost_usd_per_mw_per_yr', 'capital_cost_usd_per_mw']

print(f"\n🔍 Cost data analysis by technology:")

for col in cost_columns:
    if col in df.columns:
        print(f"\n📊 {col}:")
        
        # Group by technology and count non-null values
        tech_analysis = df.groupby('technology')[col].apply(lambda x: x.notna().sum()).sort_values(ascending=False)
        
        # Show technologies with data
        techs_with_data = tech_analysis[tech_analysis > 0]
        print(f"   Technologies WITH data ({len(techs_with_data)}):")
        for tech, count in techs_with_data.head(10).items():
            total_tech_rows = len(df[df['technology'] == tech])
            completeness = count / total_tech_rows * 100
            print(f"     • {tech}: {count:,}/{total_tech_rows:,} ({completeness:.1f}%)")
        
        # Show technologies with NO data
        techs_without_data = tech_analysis[tech_analysis == 0]
        print(f"\n   Technologies WITHOUT data ({len(techs_without_data)}):")
        for tech in sorted(techs_without_data.index):
            total_tech_rows = len(df[df['technology'] == tech])
            print(f"     • {tech}: {total_tech_rows:,} rows")

# Analyze efficiency data by technology
print(f"\n⚡ Efficiency data analysis by technology:")
if 'efficiency_decimal' in df.columns:
    tech_analysis = df.groupby('technology')['efficiency_decimal'].apply(lambda x: x.notna().sum()).sort_values(ascending=False)
    
    # Show technologies with data
    techs_with_data = tech_analysis[tech_analysis > 0]
    print(f"   Technologies WITH efficiency data ({len(techs_with_data)}):")
    for tech, count in techs_with_data.head(10).items():
        total_tech_rows = len(df[df['technology'] == tech])
        completeness = count / total_tech_rows * 100
        print(f"     • {tech}: {count:,}/{total_tech_rows:,} ({completeness:.1f}%)")
    
    # Show technologies with NO data
    techs_without_data = tech_analysis[tech_analysis == 0]
    print(f"\n   Technologies WITHOUT efficiency data ({len(techs_without_data)}):")
    for tech in sorted(techs_without_data.index):
        total_tech_rows = len(df[df['technology'] == tech])
        print(f"     • {tech}: {total_tech_rows:,} rows")

# Analyze by sector
print(f"\n🏭 Cost data analysis by sector:")
for col in cost_columns:
    if col in df.columns:
        print(f"\n📊 {col} by sector:")
        sector_analysis = df.groupby('sector')[col].apply(lambda x: x.notna().sum())
        for sector, count in sector_analysis.items():
            total_sector_rows = len(df[df['sector'] == sector])
            completeness = count / total_sector_rows * 100
            print(f"     • {sector}: {count:,}/{total_sector_rows:,} ({completeness:.1f}%)")

print(f"\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80) 