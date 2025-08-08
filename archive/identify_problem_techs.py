"""
Identify which specific technologies have cost data issues
"""

import modin.pandas as pd
import numpy as np

print("=" * 80)
print("IDENTIFYING TECHNOLOGIES WITH COST DATA ISSUES")
print("=" * 80)

# Load the data
df = pd.read_csv("4_final_AR6_aggregated.csv")
print(f"✅ Loaded dataset: {df.shape[0]:,} rows")

# Define the critical columns
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
print(f"📋 Analyzing columns: {existing_critical_columns}")

print(f"\n🔍 TECHNOLOGIES WITH COST DATA ISSUES:")
print(f"=" * 60)

# Analyze each technology
tech_analysis = df.groupby('technology')[existing_critical_columns].apply(lambda x: x.notna().sum())
tech_totals = df.groupby('technology').size()

print(f"\n📊 Cost data completeness by technology:")
print(f"{'Technology':<30} {'OM Cost':<10} {'Capital Cost':<12} {'Efficiency':<10} {'Lifetime':<10} {'Pathway':<10} {'Price':<10} {'Total Rows':<12}")
print(f"{'-'*30} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

for tech in sorted(tech_analysis.index):
    tech_data = tech_analysis.loc[tech]
    total_rows = tech_totals[tech]
    
    om_cost_pct = (tech_data.get('om_cost_usd_per_mw_per_yr', 0) / total_rows * 100) if 'om_cost_usd_per_mw_per_yr' in tech_data else 0
    capital_cost_pct = (tech_data.get('capital_cost_usd_per_mw', 0) / total_rows * 100) if 'capital_cost_usd_per_mw' in tech_data else 0
    efficiency_pct = (tech_data.get('efficiency_decimal', 0) / total_rows * 100) if 'efficiency_decimal' in tech_data else 0
    lifetime_pct = (tech_data.get('lifetime_years', 0) / total_rows * 100) if 'lifetime_years' in tech_data else 0
    pathway_pct = (tech_data.get('scenario_pathway', 0) / total_rows * 100) if 'scenario_pathway' in tech_data else 0
    price_pct = (tech_data.get('scenario_price', 0) / total_rows * 100) if 'scenario_price' in tech_data else 0
    
    print(f"{tech:<30} {om_cost_pct:>8.1f}% {capital_cost_pct:>10.1f}% {efficiency_pct:>8.1f}% {lifetime_pct:>8.1f}% {pathway_pct:>8.1f}% {price_pct:>8.1f}% {total_rows:>10,}")

print(f"\n🚨 TECHNOLOGIES WITH ZERO COST DATA:")
print(f"=" * 40)

# Find technologies with zero cost data
zero_cost_techs = []
for tech in sorted(tech_analysis.index):
    tech_data = tech_analysis.loc[tech]
    total_rows = tech_totals[tech]
    
    om_cost_count = tech_data.get('om_cost_usd_per_mw_per_yr', 0)
    capital_cost_count = tech_data.get('capital_cost_usd_per_mw', 0)
    
    if om_cost_count == 0 and capital_cost_count == 0:
        zero_cost_techs.append(tech)
        print(f"• {tech}: {total_rows:,} rows")

print(f"\n📈 TECHNOLOGIES WITH COMPLETE COST DATA:")
print(f"=" * 40)

# Find technologies with complete cost data
complete_cost_techs = []
for tech in sorted(tech_analysis.index):
    tech_data = tech_analysis.loc[tech]
    total_rows = tech_totals[tech]
    
    om_cost_count = tech_data.get('om_cost_usd_per_mw_per_yr', 0)
    capital_cost_count = tech_data.get('capital_cost_usd_per_mw', 0)
    
    if om_cost_count == total_rows and capital_cost_count == total_rows:
        complete_cost_techs.append(tech)
        print(f"• {tech}: {total_rows:,} rows")

print(f"\n📊 SUMMARY:")
print(f"=" * 20)
print(f"• Total technologies: {len(tech_analysis)}")
print(f"• Technologies with zero cost data: {len(zero_cost_techs)}")
print(f"• Technologies with complete cost data: {len(complete_cost_techs)}")
print(f"• Technologies with partial cost data: {len(tech_analysis) - len(zero_cost_techs) - len(complete_cost_techs)}")

print(f"\n" + "=" * 80)
print("ANALYSIS COMPLETE - READY FOR DECISION")
print("=" * 80) 