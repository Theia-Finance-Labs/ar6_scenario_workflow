#!/usr/bin/env python3
"""
Map NGFS Scenarios to Technology Lookup Table
==============================================

This script simply maps NGFS scenarios to the existing AR6 technology lookup table
based on stringency levels. The technology_lookup_table.csv already has all the
data we need - we just need to know which stringency level each NGFS scenario
corresponds to.

Usage:
    python3 map_ngfs_to_lookup.py

Inputs:
    - NGFS_data/NGFS5_Scenarios_Database_R5_regions.csv: To get NGFS scenarios
    - NGFS_data/NGFS_stringency_mapping.csv: NGFS scenario to AR6 category mapping
    - technology_lookup_table.csv: Complete AR6 technology lookup (already has all params)

Outputs:
    - Prints the mapping for reference
"""

import pandas as pd

print("=" * 80)
print("NGFS TO AR6 TECHNOLOGY LOOKUP MAPPING")
print("=" * 80)

# Load NGFS scenarios
print("\n📂 Loading NGFS scenarios...")
ngfs_df = pd.read_csv("NGFS_data/NGFS5_Scenarios_Database_R5_regions.csv")
ngfs_scenarios = ngfs_df[['Model', 'Scenario']].drop_duplicates()
print(f"   Found {len(ngfs_scenarios)} unique Model-Scenario combinations")
print(f"   Scenarios: {sorted(ngfs_scenarios['Scenario'].unique())}")

# Load stringency mapping
print("\n📊 Loading NGFS stringency mapping...")
stringency_map = pd.read_csv("NGFS_data/NGFS_stringency_mapping.csv")
print(f"   Loaded {len(stringency_map)} mappings")
print("\nNGFS Scenario → AR6 Stringency Category:")
for _, row in stringency_map.iterrows():
    print(f"   {row['Scenario']:50s} → {row['Category']:4s} ({row['Category_name']})")

# Load AR6 technology lookup
print("\n📚 Loading AR6 technology lookup table...")
ar6_lookup = pd.read_csv("technology_lookup_table.csv")
print(f"   Shape: {ar6_lookup.shape}")
print(f"   Technologies: {ar6_lookup['technology'].nunique()}")
print(f"   Stringency levels: {sorted(ar6_lookup['stringency'].unique())}")
print(f"   Countries (iso2): {ar6_lookup['iso2'].nunique()}")
print(f"   Years: {ar6_lookup['year'].min()}-{ar6_lookup['year'].max()}")

print("\n" + "=" * 80)
print("USAGE INSTRUCTIONS")
print("=" * 80)
print("""
To use the technology lookup table for NGFS scenarios:

1. Match your NGFS scenario to its stringency level using the mapping above

2. Query the technology_lookup_table.csv with:
   - technology: The technology you need (e.g., 'SolarCap', 'WindCap', 'CoalCap')
   - year: The year you need (2023-2050)
   - iso2: The country code (use 'GLOBAL' if specific country not available)
   - stringency: The category from step 1 (use 'UNKNOWN' if not available)

3. The lookup table contains:
   - capital_cost_usd_per_mw
   - om_cost_usd_per_mw_per_yr
   - efficiency_decimal
   - lifetime_years
   - fuel_price_usd_per_mwh
   - electricity_price_usd_per_mwh

Example:
   For NGFS "Net Zero 2050" scenario (→ C1 stringency):
   - Query: technology='SolarCap', year=2030, iso2='USA', stringency='C1'
   - Fallback: technology='SolarCap', year=2030, iso2='GLOBAL', stringency='C1'
   - Fallback: technology='SolarCap', year=2030, iso2='GLOBAL', stringency='UNKNOWN'
""")

print("\n✅ The technology_lookup_table.csv is ready to use for NGFS scenarios!")
print("   Just map NGFS scenarios to their stringency levels using the mapping above.")
