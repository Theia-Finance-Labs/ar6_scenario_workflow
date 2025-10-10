#!/usr/bin/env python3
"""
NGFS Technology Lookup Table Generator
=======================================

This script creates technology lookup table for NGFS scenarios by:
1. Loading NGFS scenario data
2. Mapping NGFS scenarios to AR6 stringency categories (C1-C8)
3. Using AR6 technology lookup table to gap-fill technology parameters
4. Translating R5 regions to ISO3 country codes

Usage:
    python3 create_ngfs_technology_lookup.py

Inputs:
    - NGFS_data/NGFS5_Scenarios_Database_R5_regions.csv: NGFS scenario data
    - NGFS_data/r5_region_lookup.csv: R5 region to ISO3 mapping
    - NGFS_data/NGFS_stringency_mapping.csv: NGFS scenario to AR6 category mapping
    - technology_lookup_table.csv: AR6 technology lookup for gap-filling

Outputs:
    - technology_lookup_table_ngfs.csv: NGFS technology lookup table at ISO3 level
"""

import pandas as pd
import numpy as np

def load_r5_to_iso3_mapping():
    """Load R5 to ISO3 mapping"""
    print("🗺️  Loading R5 to ISO3 mapping...")

    r5_lookup = pd.read_csv("NGFS_data/r5_region_lookup.csv")

    # Build mapping dictionary
    r5_iso3_map = {}
    for _, row in r5_lookup.iterrows():
        region = row["R5 Region"]
        iso3_raw = row["ISO3"]
        if pd.notna(iso3_raw) and str(iso3_raw).strip() and str(iso3_raw) != 'N/A':
            # Handle pipe-separated ISO3 codes
            iso3_list = [p.strip() for p in str(iso3_raw).split("|") if p.strip() and p.strip() != 'N/A']
            r5_iso3_map[region] = iso3_list

    print(f"   Loaded mappings for {len(r5_iso3_map)} R5 regions")
    return r5_iso3_map

def load_ngfs_stringency_mapping():
    """Load NGFS scenario to stringency mapping"""
    print("📊 Loading NGFS stringency mapping...")

    stringency_map = pd.read_csv("NGFS_data/NGFS_stringency_mapping.csv")
    stringency_dict = dict(zip(stringency_map['Scenario'], stringency_map['Category']))

    print(f"   Loaded mappings for {len(stringency_dict)} NGFS scenarios")
    print(f"   Scenarios: {list(stringency_dict.keys())}")

    return stringency_dict

def load_ar6_technology_lookup():
    """Load AR6 technology lookup table"""
    print("📚 Loading AR6 technology lookup table...")

    ar6_lookup = pd.read_csv("technology_lookup_table.csv")
    print(f"   Loaded: {ar6_lookup.shape[0]:,} rows × {ar6_lookup.shape[1]} columns")
    print(f"   Technologies: {ar6_lookup['technology'].nunique()}")
    print(f"   Stringency levels: {sorted(ar6_lookup['stringency'].unique())}")
    print(f"   Years: {ar6_lookup['year'].min()}-{ar6_lookup['year'].max()}")

    return ar6_lookup

def load_ngfs_scenarios():
    """Load NGFS scenarios"""
    print("📂 Loading NGFS scenarios...")

    ngfs_df = pd.read_csv("NGFS_data/NGFS5_Scenarios_Database_R5_regions.csv")

    # Get unique scenario-region combinations
    ngfs_combos = ngfs_df[['Model', 'Scenario', 'Region']].drop_duplicates()
    print(f"   Loaded {len(ngfs_combos):,} unique Model-Scenario-Region combinations")
    print(f"   Scenarios: {sorted(ngfs_combos['Scenario'].unique())}")
    print(f"   Regions: {sorted(ngfs_combos['Region'].unique())}")

    return ngfs_combos

def create_ngfs_lookup_table():
    """Create NGFS technology lookup table"""
    print("=" * 80)
    print("NGFS TECHNOLOGY LOOKUP TABLE GENERATOR")
    print("=" * 80)

    # Load mappings
    r5_iso3_map = load_r5_to_iso3_mapping()
    stringency_dict = load_ngfs_stringency_mapping()
    ar6_lookup = load_ar6_technology_lookup()
    ngfs_scenarios = load_ngfs_scenarios()

    print("\n🔧 Creating NGFS technology lookup table...")

    # Add NGFS metadata to AR6 lookup
    # For each NGFS scenario-region, we'll duplicate the AR6 lookup entries

    # First, expand NGFS scenarios to ISO3 level
    expanded_scenarios = []
    for _, ngfs_row in ngfs_scenarios.iterrows():
        model = ngfs_row['Model']
        scenario = ngfs_row['Scenario']
        r5_region = ngfs_row['Region']

        # Get stringency for this scenario
        stringency = stringency_dict.get(scenario, 'UNKNOWN')

        # Get ISO3 codes for this R5 region
        iso3_codes = r5_iso3_map.get(r5_region, ['GLOBAL'])

        for iso3 in iso3_codes:
            expanded_scenarios.append({
                'model': model,
                'scenario': scenario,
                'r5_region': r5_region,
                'iso3': iso3,
                'stringency': stringency
            })

    expanded_df = pd.DataFrame(expanded_scenarios)
    print(f"   Expanded to {len(expanded_df):,} scenario-ISO3 combinations")

    # Now we'll use a vectorized merge approach
    # For each scenario-ISO3 combination, we try to match AR6 lookup entries

    all_lookups = []

    for stringency_level in expanded_df['stringency'].unique():
        print(f"\n   Processing stringency level: {stringency_level}")

        # Get scenarios for this stringency
        scenarios_for_stringency = expanded_df[expanded_df['stringency'] == stringency_level].copy()

        # Get AR6 data for this stringency (with GLOBAL as fallback)
        ar6_for_stringency = ar6_lookup[
            (ar6_lookup['stringency'] == stringency_level) |
            ((ar6_lookup['stringency'] == 'UNKNOWN') & (ar6_lookup['iso2'] == 'GLOBAL'))
        ].copy()

        if ar6_for_stringency.empty:
            print(f"      ⚠️ No AR6 data for {stringency_level}, using GLOBAL/UNKNOWN")
            ar6_for_stringency = ar6_lookup[
                (ar6_lookup['stringency'] == 'UNKNOWN') & (ar6_lookup['iso2'] == 'GLOBAL')
            ].copy()

        print(f"      AR6 entries: {len(ar6_for_stringency):,}")

        # For each scenario, merge with AR6 lookup
        for iso3 in scenarios_for_stringency['iso3'].unique():
            # Get AR6 data for this ISO3 or GLOBAL
            ar6_for_iso3 = ar6_for_stringency[
                (ar6_for_stringency['iso2'] == iso3) | (ar6_for_stringency['iso2'] == 'GLOBAL')
            ].copy()

            if ar6_for_iso3.empty:
                continue

            # Prefer specific ISO3 over GLOBAL
            # Group by technology and year, keep first (prefer non-GLOBAL)
            ar6_for_iso3 = ar6_for_iso3.sort_values('iso2', ascending=False)  # GLOBAL comes first, non-GLOBAL preferred
            ar6_for_iso3 = ar6_for_iso3.drop_duplicates(subset=['technology', 'year'], keep='last')

            # Get scenarios for this ISO3
            scenarios_for_iso3 = scenarios_for_stringency[scenarios_for_stringency['iso3'] == iso3]

            # Cross join with AR6 data
            scenarios_for_iso3['_key'] = 1
            ar6_for_iso3['_key'] = 1

            merged = scenarios_for_iso3.merge(ar6_for_iso3, on='_key').drop('_key', axis=1)

            # Rename iso2 to iso3_matched
            merged = merged.rename(columns={'iso2': 'iso3_matched'})

            all_lookups.append(merged)

    print(f"\n   Combining all lookups...")
    ngfs_lookup = pd.concat(all_lookups, ignore_index=True)

    # Summary statistics
    print("\n📊 Summary:")
    tech_params = [
        'capital_cost_usd_per_mw',
        'om_cost_usd_per_mw_per_yr',
        'efficiency_decimal',
        'lifetime_years',
        'fuel_price_usd_per_mwh',
        'electricity_price_usd_per_mwh'
    ]

    for param in tech_params:
        if param in ngfs_lookup.columns:
            filled = ngfs_lookup[param].notna().sum()
            total = len(ngfs_lookup)
            pct = (filled / total) * 100 if total > 0 else 0
            print(f"   {param}: {filled:,}/{total:,} ({pct:.1f}%) filled")

    return ngfs_lookup

def main():
    # Create NGFS lookup table
    ngfs_lookup = create_ngfs_lookup_table()

    if not ngfs_lookup.empty:
        # Save output
        output_file = "technology_lookup_table_ngfs.csv"
        ngfs_lookup.to_csv(output_file, index=False)
        print(f"\n💾 Saved NGFS lookup table: {output_file}")
        print(f"   Shape: {ngfs_lookup.shape}")

        print("\n✅ NGFS lookup table generation complete!")
        print(f"   📈 Models: {ngfs_lookup['model'].nunique()}")
        print(f"   🎯 Scenarios: {ngfs_lookup['scenario'].nunique()}")
        print(f"   🌍 Countries (ISO3): {ngfs_lookup['iso3'].nunique()}")
        print(f"   🔧 Technologies: {ngfs_lookup['technology'].nunique()}")
        print(f"   📅 Years: {ngfs_lookup['year'].min()}-{ngfs_lookup['year'].max()}")
    else:
        print("\n❌ Failed to generate NGFS lookup table")

if __name__ == "__main__":
    main()
