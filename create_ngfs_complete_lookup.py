#!/usr/bin/env python3
"""
NGFS Complete Technology Lookup Table Generator
=================================================

This script creates a complete technology lookup table for NGFS scenarios by:
1. Extracting NGFS Capital Cost and Price data (what NGFS provides)
2. Gap-filling missing parameters (OM Cost, Efficiency, Lifetime) from AR6 lookup based on stringency
3. Translating R5 regions to ISO3 country codes

Usage:
    python3 create_ngfs_complete_lookup.py

Inputs:
    - NGFS_data/NGFS5_Scenarios_Database_R5_regions.csv: NGFS scenario data
    - NGFS_data/r5_region_lookup.csv: R5 region to ISO3 mapping
    - NGFS_data/NGFS_stringency_mapping.csv: NGFS scenario to AR6 category mapping
    - technology_lookup_table.csv: AR6 technology lookup for gap-filling
    - ar6_variables_with_mapping.csv: Variable to technology mapping

Outputs:
    - technology_lookup_table_ngfs_complete.csv: Complete NGFS technology lookup
"""

import pandas as pd
import numpy as np

def load_ngfs_data():
    """Load and process NGFS data"""
    print("=" * 80)
    print("NGFS COMPLETE TECHNOLOGY LOOKUP GENERATOR")
    print("=" * 80)
    print("\n📂 Loading NGFS data...")

    df = pd.read_csv("NGFS_data/NGFS5_Scenarios_Database_R5_regions.csv")
    print(f"   Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Filter to relevant variables
    capital_mask = df['Variable'].str.contains('Capital Cost', case=False, na=False)
    price_mask = df['Variable'].str.contains('Price\\|Primary Energy|Price\\|Secondary Energy\\|Electricity', regex=True, case=False, na=False)

    relevant_df = df[capital_mask | price_mask].copy()
    print(f"   Filtered to {len(relevant_df):,} rows with Capital Cost and Price data")

    return relevant_df

def map_ngfs_variables_to_technologies(df):
    """Map NGFS variables to standard technology names"""
    print("\n🗺️  Mapping NGFS variables to technologies...")

    # Create mapping for NGFS Capital Cost variables to technology names
    variable_tech_mapping = {
        # Electricity generation
        'Capital Cost|Electricity|Biomass|w/ CCS': 'BiomassCap - w/ CCS',
        'Capital Cost|Electricity|Biomass|w/o CCS': 'BiomassCap - w/o CCS',
        'Capital Cost|Electricity|Coal|w/ CCS': 'CoalCap - w/ CCS',
        'Capital Cost|Electricity|Coal|w/o CCS': 'CoalCap - w/o CCS',
        'Capital Cost|Electricity|Gas|w/ CCS': 'GasCap - w/ CCS',
        'Capital Cost|Electricity|Gas|w/o CCS': 'GasCap - w/o CCS',
        'Capital Cost|Electricity|Geothermal': 'GeothermalCap',
        'Capital Cost|Electricity|Hydro': 'HydroCap',
        'Capital Cost|Electricity|Nuclear': 'NuclearCap',
        'Capital Cost|Electricity|Solar|CSP': 'SolarCap',
        'Capital Cost|Electricity|Solar|PV': 'SolarCap',
        'Capital Cost|Electricity|Wind|Offshore': 'WindCap',
        'Capital Cost|Electricity|Wind|Onshore': 'WindCap',

        # Prices
        'Price|Primary Energy|Biomass': 'biomass_price',
        'Price|Primary Energy|Coal': 'coal_price',
        'Price|Primary Energy|Gas': 'gas_price',
        'Price|Primary Energy|Oil': 'oil_price',
        'Price|Secondary Energy|Electricity': 'electricity_price',
    }

    df['technology'] = df['Variable'].map(variable_tech_mapping)

    # Filter to successfully mapped variables
    df = df[df['technology'].notna()].copy()
    print(f"   Mapped {len(df):,} rows to {df['technology'].nunique()} technologies")

    return df

def process_ngfs_to_long_format(df):
    """Process NGFS data to long format"""
    print("\n⚡ Converting to long format...")

    # Get year columns (2020-2050 for our use case)
    year_cols = [str(y) for y in range(2020, 2051, 5)]  # NGFS has 5-year intervals
    available_years = [c for c in year_cols if c in df.columns]

    print(f"   Year columns: {available_years}")

    # Melt to long format
    id_cols = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'technology']
    melted = pd.melt(
        df,
        id_vars=id_cols,
        value_vars=available_years,
        var_name='year',
        value_name='value'
    )

    melted['year'] = melted['year'].astype(int)
    melted['value'] = pd.to_numeric(melted['value'], errors='coerce')

    # Remove null values
    melted = melted[melted['value'].notna()].copy()
    print(f"   Melted to {len(melted):,} rows")

    return melted

def interpolate_to_yearly(df):
    """Interpolate 5-year data to yearly data (2023-2050)"""
    print("\n⏰ Interpolating to yearly data (2023-2050)...")

    target_years = list(range(2023, 2051))
    result_list = []

    grouping_cols = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'technology']

    for name, group in df.groupby(grouping_cols):
        # Create complete year range
        group_df = pd.DataFrame({'year': target_years})

        # Merge with actual data
        group_df = group_df.merge(group[['year', 'value']], on='year', how='left')

        # Linear interpolation
        group_df['value'] = group_df['value'].interpolate(method='linear', limit_direction='both')
        group_df['value'] = group_df['value'].ffill().bfill()

        # Add grouping columns
        for i, col in enumerate(grouping_cols):
            group_df[col] = name[i]

        result_list.append(group_df)

    result = pd.concat(result_list, ignore_index=True)
    result = result[result['value'].notna()].copy()

    print(f"   Interpolated to {len(result):,} yearly data points")
    return result

def translate_r5_to_iso3(df):
    """Translate R5 regions to ISO3"""
    print("\n🌍 Translating R5 to ISO3...")

    r5_lookup = pd.read_csv("NGFS_data/r5_region_lookup.csv")

    # Build mapping
    r5_iso3_map = {}
    for _, row in r5_lookup.iterrows():
        region = row["R5 Region"]
        iso3_raw = row["ISO3"]
        if pd.notna(iso3_raw) and str(iso3_raw).strip() and str(iso3_raw) != 'N/A':
            iso3_list = [p.strip() for p in str(iso3_raw).split("|") if p.strip() and p.strip() != 'N/A']
            r5_iso3_map[region] = iso3_list

    # Expand to ISO3
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

def add_stringency(df):
    """Add stringency categorization"""
    print("\n📊 Adding stringency...")

    stringency_map = pd.read_csv("NGFS_data/NGFS_stringency_mapping.csv")
    stringency_dict = dict(zip(stringency_map['Scenario'], stringency_map['Category']))

    df['stringency'] = df['Scenario'].map(stringency_dict).fillna('UNKNOWN')

    counts = df['stringency'].value_counts()
    print(f"   Stringency distribution: {counts.to_dict()}")

    return df

def pivot_ngfs_data(df):
    """Pivot NGFS data to have metric columns"""
    print("\n🔄 Pivoting NGFS data...")

    # Determine metric type from Variable
    def get_metric_type(row):
        var = row['Variable']
        if 'Capital Cost' in var:
            return 'capital_cost_ngfs'
        elif 'Price|Primary Energy|Biomass' in var:
            return 'biomass_price_ngfs'
        elif 'Price|Primary Energy|Coal' in var:
            return 'coal_price_ngfs'
        elif 'Price|Primary Energy|Gas' in var:
            return 'gas_price_ngfs'
        elif 'Price|Primary Energy|Oil' in var:
            return 'oil_price_ngfs'
        elif 'Price|Secondary Energy|Electricity' in var:
            return 'electricity_price_ngfs'
        return 'unknown'

    df['metric_type'] = df.apply(get_metric_type, axis=1)
    df = df[df['metric_type'] != 'unknown'].copy()

    # Pivot
    pivoted = df.pivot_table(
        index=['Model', 'Scenario', 'Region', 'iso3', 'technology', 'year', 'stringency'],
        columns='metric_type',
        values='value',
        aggfunc='median'
    ).reset_index()

    pivoted.columns.name = None

    print(f"   Pivoted to {len(pivoted):,} rows")

    # Convert units: Capital cost from USD2010/kW to USD/MW (*1000)
    if 'capital_cost_ngfs' in pivoted.columns:
        pivoted['capital_cost_usd_per_mw'] = pivoted['capital_cost_ngfs'] * 1000
        pivoted = pivoted.drop('capital_cost_ngfs', axis=1)
        print("   ✅ Converted capital cost: USD/kW → USD/MW")

    # Convert prices from USD/GJ to USD/MWh (*3.6)
    price_conversions = {
        'biomass_price_ngfs': 'fuel_price_usd_per_mwh',
        'coal_price_ngfs': 'fuel_price_usd_per_mwh',
        'gas_price_ngfs': 'fuel_price_usd_per_mwh',
        'oil_price_ngfs': 'fuel_price_usd_per_mwh',
        'electricity_price_ngfs': 'electricity_price_usd_per_mwh'
    }

    for old_col, new_col in price_conversions.items():
        if old_col in pivoted.columns:
            # For fuel prices, we need to map to the right technology
            if 'fuel_price' in new_col:
                # Store temporarily, will handle in gap-filling
                pass
            # Convert GJ to MWh
            pivoted[old_col] = pivoted[old_col] * 3.6

    return pivoted

def gapfill_with_ar6(ngfs_df):
    """Gap-fill NGFS data with AR6 lookup table - VECTORIZED"""
    print("\n🔧 Gap-filling with AR6 lookup table (vectorized)...")

    ar6_lookup = pd.read_csv("technology_lookup_table.csv")
    print(f"   AR6 lookup: {ar6_lookup.shape[0]:,} rows")

    # Rename iso2 to iso3 in AR6 for consistency
    ar6_lookup = ar6_lookup.rename(columns={'iso2': 'iso3'})

    # Try exact match first
    print("   Attempting exact ISO3 + stringency match...")
    merged = ngfs_df.merge(
        ar6_lookup,
        on=['technology', 'year', 'iso3', 'stringency'],
        how='left',
        suffixes=('_ngfs', '_ar6')
    )

    # Count matches
    exact_matches = merged['lifetime_years'].notna().sum()
    print(f"      Exact matches: {exact_matches:,}/{len(merged):,}")

    # For unmatched, try GLOBAL + stringency
    print("   Filling gaps with GLOBAL + stringency...")
    unmatched_mask = merged['lifetime_years'].isna()
    unmatched = merged[unmatched_mask].copy()

    if len(unmatched) > 0:
        ar6_global_stringency = ar6_lookup[ar6_lookup['iso3'] == 'GLOBAL'].copy()

        unmatched = unmatched.drop([c for c in unmatched.columns if c.endswith('_ar6')], axis=1)
        unmatched = unmatched.merge(
            ar6_global_stringency,
            on=['technology', 'year', 'stringency'],
            how='left',
            suffixes=('_ngfs', '_ar6')
        )

        # Update merged dataframe
        merged.loc[unmatched_mask, unmatched.columns] = unmatched.values

        global_matches = unmatched['lifetime_years'].notna().sum()
        print(f"      GLOBAL + stringency matches: {global_matches:,}")

    # For still unmatched, try GLOBAL + UNKNOWN
    print("   Filling remaining gaps with GLOBAL + UNKNOWN...")
    still_unmatched_mask = merged['lifetime_years'].isna()
    still_unmatched = merged[still_unmatched_mask].copy()

    if len(still_unmatched) > 0:
        ar6_global_unknown = ar6_lookup[
            (ar6_lookup['iso3'] == 'GLOBAL') &
            (ar6_lookup['stringency'] == 'UNKNOWN')
        ].copy()

        still_unmatched = still_unmatched.drop([c for c in still_unmatched.columns if c.endswith('_ar6')], axis=1)
        still_unmatched = still_unmatched.merge(
            ar6_global_unknown[['technology', 'year', 'om_cost_usd_per_mw_per_yr', 'efficiency_decimal',
                                 'lifetime_years', 'capital_cost_usd_per_mw', 'fuel_price_usd_per_mwh',
                                 'electricity_price_usd_per_mwh']],
            on=['technology', 'year'],
            how='left',
            suffixes=('_ngfs', '_ar6')
        )

        # Update merged dataframe
        for col in still_unmatched.columns:
            if col in merged.columns:
                merged.loc[still_unmatched_mask, col] = still_unmatched[col].values

        unknown_matches = still_unmatched['lifetime_years'].notna().sum()
        print(f"      GLOBAL + UNKNOWN matches: {unknown_matches:,}")

    # Combine NGFS and AR6 data
    print("\n   Combining NGFS and AR6 data...")

    # Use NGFS capital cost if available, otherwise AR6
    if 'capital_cost_usd_per_mw_ar6' in merged.columns:
        merged['capital_cost_usd_per_mw'] = merged['capital_cost_usd_per_mw_ngfs'].fillna(
            merged['capital_cost_usd_per_mw_ar6']
        )
        merged = merged.drop(['capital_cost_usd_per_mw_ngfs', 'capital_cost_usd_per_mw_ar6'], axis=1)
    elif 'capital_cost_usd_per_mw_ngfs' in merged.columns:
        merged = merged.rename(columns={'capital_cost_usd_per_mw_ngfs': 'capital_cost_usd_per_mw'})

    # For other parameters, use AR6 data
    ar6_params = ['om_cost_usd_per_mw_per_yr', 'efficiency_decimal', 'lifetime_years']
    for param in ar6_params:
        if f'{param}_ar6' in merged.columns:
            merged[param] = merged[f'{param}_ar6']
            merged = merged.drop(f'{param}_ar6', axis=1)

    # For prices, prefer NGFS, fallback to AR6
    price_params = ['fuel_price_usd_per_mwh', 'electricity_price_usd_per_mwh']
    for param in price_params:
        if f'{param}_ngfs' in merged.columns and f'{param}_ar6' in merged.columns:
            merged[param] = merged[f'{param}_ngfs'].fillna(merged[f'{param}_ar6'])
            merged = merged.drop([f'{param}_ngfs', f'{param}_ar6'], axis=1)
        elif f'{param}_ar6' in merged.columns:
            merged[param] = merged[f'{param}_ar6']
            merged = merged.drop(f'{param}_ar6', axis=1)
        elif f'{param}_ngfs' in merged.columns:
            merged = merged.rename(columns={f'{param}_ngfs': param})

    # Clean up extra columns
    merged = merged[[c for c in merged.columns if not c.endswith('_ngfs') and not c.endswith('_ar6')]]

    print(f"   Gap-filled to {len(merged):,} rows")

    # Summary
    print("\n📊 Coverage summary:")
    params = ['capital_cost_usd_per_mw', 'om_cost_usd_per_mw_per_yr', 'efficiency_decimal',
              'lifetime_years', 'fuel_price_usd_per_mwh', 'electricity_price_usd_per_mwh']

    for param in params:
        if param in merged.columns:
            filled = merged[param].notna().sum()
            total = len(merged)
            pct = (filled / total) * 100 if total > 0 else 0
            print(f"   {param}: {filled:,}/{total:,} ({pct:.1f}%)")

    return merged

def main():
    # Load NGFS data
    ngfs_data = load_ngfs_data()

    # Map to technologies
    ngfs_data = map_ngfs_variables_to_technologies(ngfs_data)

    # Process to long format
    ngfs_long = process_ngfs_to_long_format(ngfs_data)

    # Interpolate to yearly
    ngfs_yearly = interpolate_to_yearly(ngfs_long)

    # Translate R5 to ISO3
    ngfs_iso3 = translate_r5_to_iso3(ngfs_yearly)

    # Add stringency
    ngfs_with_stringency = add_stringency(ngfs_iso3)

    # Pivot to get metric columns
    ngfs_pivoted = pivot_ngfs_data(ngfs_with_stringency)

    # Gap-fill with AR6
    ngfs_complete = gapfill_with_ar6(ngfs_pivoted)

    # Save
    output_file = "technology_lookup_table_ngfs_complete.csv"
    ngfs_complete.to_csv(output_file, index=False)

    print(f"\n💾 Saved: {output_file}")
    print(f"   Shape: {ngfs_complete.shape}")
    print("\n✅ NGFS complete technology lookup generation finished!")

if __name__ == "__main__":
    main()
