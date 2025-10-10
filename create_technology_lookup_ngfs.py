#!/usr/bin/env python3
"""
NGFS Technology Lookup Table Generator
=======================================

This script creates a comprehensive technology lookup table for NGFS scenarios
using the AR6 technology lookup table for gap-filling.

The NGFS data is at R5 regional level and needs to be translated to ISO3 level.
Missing technology parameters are gap-filled using the AR6 technology lookup table
based on stringency level (C1-C8) mapping.

Usage:
    python3 create_technology_lookup_ngfs.py

Inputs:
    - NGFS_data/NGFS5_Scenarios_Database_R5_regions.csv: NGFS scenario data
    - NGFS_data/r5_region_lookup.csv: R5 region to ISO3 mapping
    - NGFS_data/NGFS_stringency_mapping.csv: NGFS scenario to AR6 category mapping
    - technology_lookup_table.csv: AR6 technology lookup for gap-filling
    - ar6_variables_with_mapping.csv: Variable to sector/technology mapping

Outputs:
    - technology_lookup_table_ngfs.csv: NGFS technology lookup table
"""

import pandas as pd
import numpy as np
from typing import List, Dict

def load_ngfs_data() -> pd.DataFrame:
    """Load NGFS data and filter to relevant variables"""
    print("📂 Loading NGFS data...")

    # Load NGFS data
    ngfs_df = pd.read_csv("NGFS_data/NGFS5_Scenarios_Database_R5_regions.csv")
    print(f"   Original: {ngfs_df.shape[0]:,} rows × {ngfs_df.shape[1]} columns")
    print(f"   Scenarios: {ngfs_df['Scenario'].nunique()}")

    # Filter to relevant variables (technology costs, prices, etc.)
    target_variables = [
        "Capital Cost", "OM Cost", "Efficiency", "Lifetime",
        "Price"
    ]

    variable_mask = ngfs_df['Variable'].str.contains('|'.join(target_variables), na=False, case=False)
    filtered_df = ngfs_df[variable_mask].copy()
    print(f"   After variable filter: {filtered_df.shape[0]:,} rows")

    if filtered_df.empty:
        print("   ⚠️ No relevant variables found in NGFS data")
        return None

    # Load variable mapping
    try:
        mapping = pd.read_csv("ar6_variables_with_mapping.csv")
        filtered_df = filtered_df.merge(mapping, left_on="Variable", right_on="variable", how="inner")
        print(f"   After variable mapping: {filtered_df.shape[0]:,} rows")
    except FileNotFoundError:
        print("   ⚠️ Variable mapping not found, continuing without sector filtering")

    return filtered_df

def translate_r5_to_iso3(df: pd.DataFrame) -> pd.DataFrame:
    """Translate R5 regions to ISO3 country codes"""
    print("🗺️  Translating R5 regions to ISO3...")

    # Load R5 region lookup
    r5_lookup = pd.read_csv("NGFS_data/r5_region_lookup.csv")
    print(f"   Loaded R5 lookup: {len(r5_lookup)} regions")

    # Build mapping dictionary
    r5_iso3_map = {}
    for _, row in r5_lookup.iterrows():
        region = row["R5 Region"]
        iso3_raw = row["ISO3"]
        if pd.notna(iso3_raw) and str(iso3_raw).strip() and str(iso3_raw) != 'N/A':
            # Handle pipe-separated ISO3 codes
            iso3_list = [p.strip() for p in str(iso3_raw).split("|") if p.strip() and p.strip() != 'N/A']
            r5_iso3_map[region] = iso3_list

    print(f"   Created mappings for {len(r5_iso3_map)} regions")

    # Expand regions to individual ISO3 codes
    expanded_rows = []
    for _, row in df.iterrows():
        region = row['Region']
        iso3_codes = r5_iso3_map.get(region, ['GLOBAL'])

        for iso3 in iso3_codes:
            new_row = row.copy()
            new_row['iso3'] = iso3
            expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows)
    print(f"   Expanded to {len(expanded_df):,} rows with ISO3 codes")

    return expanded_df

def add_ngfs_stringency(df: pd.DataFrame) -> pd.DataFrame:
    """Add stringency categorization based on NGFS scenarios"""
    print("📊 Adding NGFS stringency categorization...")

    # Load NGFS stringency mapping
    stringency_map = pd.read_csv("NGFS_data/NGFS_stringency_mapping.csv")
    stringency_dict = dict(zip(stringency_map['Scenario'], stringency_map['Category']))

    df['stringency'] = df['Scenario'].map(stringency_dict).fillna("UNKNOWN")

    # Show stringency distribution
    stringency_counts = df['stringency'].value_counts()
    print(f"   Stringency distribution: {stringency_counts.to_dict()}")

    return df

def process_ngfs_data(ngfs_df: pd.DataFrame) -> pd.DataFrame:
    """Process NGFS data into melted format"""
    print("⚡ Processing NGFS data...")

    # Identify year columns
    base_id_cols = ["Model", "Scenario", "Region", "Variable", "Unit"]

    if 'Sector' in ngfs_df.columns and 'Technology' in ngfs_df.columns:
        id_cols = base_id_cols + ['Sector', 'Technology']
    else:
        id_cols = base_id_cols
        # Add dummy columns if not present
        ngfs_df['Sector'] = 'Unknown'
        ngfs_df['Technology'] = 'Unknown'
        id_cols = base_id_cols + ['Sector', 'Technology']

    # Get year columns (2020-2065 from NGFS data, but we want 2023-2050)
    all_cols = ngfs_df.columns.tolist()
    year_cols = []
    for col in all_cols:
        try:
            year = int(col)
            if 2023 <= year <= 2050:
                year_cols.append(str(year))
        except (ValueError, TypeError):
            continue

    print(f"   Year columns: {len(year_cols)} ({min(year_cols) if year_cols else 'none'}-{max(year_cols) if year_cols else 'none'})")

    # Melt the data
    print("   🔄 Melting data to long format...")
    melted = pd.melt(
        ngfs_df,
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="Year",
        value_name="Value"
    )

    melted["Year"] = pd.to_numeric(melted["Year"], errors="coerce")
    melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")

    # Filter to non-null values
    melted = melted[melted["Value"].notna()].copy()
    print(f"   After removing nulls: {len(melted):,} rows")

    # Translate R5 to ISO3
    melted = translate_r5_to_iso3(melted)

    # Add stringency
    melted = add_ngfs_stringency(melted)

    return melted

def load_ar6_lookup() -> pd.DataFrame:
    """Load AR6 technology lookup table for gap-filling"""
    print("📚 Loading AR6 technology lookup table for gap-filling...")

    ar6_lookup = pd.read_csv("technology_lookup_table.csv")
    print(f"   Loaded AR6 lookup: {ar6_lookup.shape[0]:,} rows × {ar6_lookup.shape[1]} columns")
    print(f"   Technologies: {ar6_lookup['technology'].nunique()}")
    print(f"   Stringency levels: {ar6_lookup['stringency'].nunique()}")

    return ar6_lookup

def gapfill_with_ar6_lookup(ngfs_data: pd.DataFrame, ar6_lookup: pd.DataFrame) -> pd.DataFrame:
    """Gap-fill missing NGFS technology data using AR6 lookup table"""
    print("🔧 Gap-filling NGFS data with AR6 lookup...")

    # Create a pivot of NGFS data if it has values
    # For now, let's assume NGFS data might be sparse or missing technology parameters
    # We'll use AR6 lookup to fill in technology characteristics

    # Define the technology parameters we expect
    tech_params = [
        'capital_cost_usd_per_mw',
        'om_cost_usd_per_mw_per_yr',
        'efficiency_decimal',
        'lifetime_years',
        'fuel_price_usd_per_mwh',
        'electricity_price_usd_per_mwh'
    ]

    # Get unique combinations from NGFS data
    ngfs_combos = ngfs_data[['Technology', 'Year', 'iso3', 'stringency']].drop_duplicates()
    print(f"   NGFS unique combinations: {len(ngfs_combos):,}")

    # For each combination, try to fill from AR6 lookup
    # Match on: technology, year, iso3 (or GLOBAL), stringency (or UNKNOWN)

    filled_rows = []

    for _, row in ngfs_combos.iterrows():
        tech = row['Technology']
        year = row['Year']
        iso3 = row['iso3']
        stringency = row['stringency']

        # Try exact match
        lookup_match = ar6_lookup[
            (ar6_lookup['technology'] == tech) &
            (ar6_lookup['year'] == year) &
            (ar6_lookup['iso3'] == iso3) &
            (ar6_lookup['stringency'] == stringency)
        ]

        # Fallback to GLOBAL if no ISO3 match
        if lookup_match.empty:
            lookup_match = ar6_lookup[
                (ar6_lookup['technology'] == tech) &
                (ar6_lookup['year'] == year) &
                (ar6_lookup['iso2'] == 'GLOBAL') &
                (ar6_lookup['stringency'] == stringency)
            ]

        # Fallback to UNKNOWN stringency if no match
        if lookup_match.empty:
            lookup_match = ar6_lookup[
                (ar6_lookup['technology'] == tech) &
                (ar6_lookup['year'] == year) &
                (ar6_lookup['iso2'] == 'GLOBAL') &
                (ar6_lookup['stringency'] == 'UNKNOWN')
            ]

        if not lookup_match.empty:
            # Use the first match
            match_row = lookup_match.iloc[0]
            filled_row = {
                'technology': tech,
                'year': year,
                'iso3': iso3,
                'stringency': stringency
            }

            # Copy all technology parameters
            for param in tech_params:
                if param in match_row and pd.notna(match_row[param]):
                    filled_row[param] = match_row[param]

            filled_rows.append(filled_row)

    filled_df = pd.DataFrame(filled_rows)
    print(f"   Gap-filled {len(filled_df):,} rows from AR6 lookup")

    # Summary of coverage
    for param in tech_params:
        if param in filled_df.columns:
            filled = filled_df[param].notna().sum()
            total = len(filled_df)
            pct = (filled / total) * 100 if total > 0 else 0
            print(f"   {param}: {filled:,}/{total:,} ({pct:.1f}%) filled")

    return filled_df

def create_ngfs_lookup_table() -> pd.DataFrame:
    """Main function to create NGFS lookup table"""
    print("=" * 80)
    print("NGFS TECHNOLOGY LOOKUP TABLE GENERATOR")
    print("=" * 80)

    # Load NGFS data
    ngfs_data = load_ngfs_data()
    if ngfs_data is None or ngfs_data.empty:
        print("❌ No NGFS data available")
        return pd.DataFrame()

    # Process NGFS data
    processed_ngfs = process_ngfs_data(ngfs_data)
    if processed_ngfs.empty:
        print("❌ No processed NGFS data")
        return pd.DataFrame()

    # Load AR6 lookup for gap-filling
    ar6_lookup = load_ar6_lookup()

    # Gap-fill NGFS data with AR6 lookup
    ngfs_lookup = gapfill_with_ar6_lookup(processed_ngfs, ar6_lookup)

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
    else:
        print("\n❌ Failed to generate NGFS lookup table")

if __name__ == "__main__":
    main()
