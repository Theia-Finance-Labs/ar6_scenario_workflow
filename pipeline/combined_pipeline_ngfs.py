#!/usr/bin/env python3
"""
Combined NGFS Pipeline
======================

Processes NGFS scenario data using the same pipeline structure as AR6,
with gap-filling from the AR6 technology lookup table.

Key differences from AR6 pipeline:
- Loads NGFS CSV data instead of AR6 feather files
- Maps NGFS scenarios to AR6 stringency categories (C1-C8)
- Translates R5 regions to ISO3 country codes
- Uses technology_lookup_table.csv for gap-filling missing parameters

Outputs:
- 1_intermediate_NGFS_scenario_formatting.csv
- 2_final_NGFS_filtered.csv
- 3_final_NGFS_target_schema.csv
- 4_final_NGFS_gapfilled.csv
- 5_final_NGFS_complete_cases.csv
- 6_final_NGFS_viable_scenarios.csv
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import external step modules
# from step4_gapfill_simple import step4_gapfill_only
# from step5_complete_cases import step5_complete_cases
# from step6_scenario_tech_filter import step6_scenario_tech_filter

import pandas as pd
import numpy as np


# ================================
# Shared Utilities
# ================================

def print_banner(title: str) -> None:
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")


def memory_release(*objs) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()


# ================================
# Step 1: Load and Format NGFS Data
# ================================

def load_ngfs_stringency_mapping() -> Dict[str, str]:
    """Load NGFS scenario to AR6 stringency mapping"""
    for path in ["NGFS_data/NGFS_stringency_mapping.csv", "../NGFS_data/NGFS_stringency_mapping.csv"]:
        try:
            mapping = pd.read_csv(path)
            return dict(zip(mapping['Scenario'], mapping['Category']))
        except FileNotFoundError:
            continue
    print("⚠️ NGFS_stringency_mapping.csv not found, all scenarios will have UNKNOWN stringency")
    return {}


def load_r5_to_iso3_mapping() -> Dict[str, List[str]]:
    """Load R5 region to ISO3 country mapping"""
    for path in ["NGFS_data/r5_region_lookup.csv", "../NGFS_data/r5_region_lookup.csv"]:
        try:
            r5_lookup = pd.read_csv(path)
            r5_iso3_map = {}
            for _, row in r5_lookup.iterrows():
                region = row["R5 Region"]
                iso3_raw = row["ISO3"]
                if pd.notna(iso3_raw) and str(iso3_raw).strip() and str(iso3_raw) != 'N/A':
                    iso3_list = [p.strip() for p in str(iso3_raw).split("|")
                                if p.strip() and p.strip() != 'N/A']
                    r5_iso3_map[region] = iso3_list
            return r5_iso3_map
        except FileNotFoundError:
            continue
    print("⚠️ r5_region_lookup.csv not found, regions will not be mapped to countries")
    return {}


def map_ngfs_variable_to_technology(variable: str) -> Optional[str]:
    """Map NGFS variable names to standard technology names"""
    var_tech_map = {
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

        # Capacity variables
        'Capacity|Electricity|Biomass|w/ CCS': 'BiomassCap - w/ CCS',
        'Capacity|Electricity|Biomass|w/o CCS': 'BiomassCap - w/o CCS',
        'Capacity|Electricity|Coal|w/ CCS': 'CoalCap - w/ CCS',
        'Capacity|Electricity|Coal|w/o CCS': 'CoalCap - w/o CCS',
        'Capacity|Electricity|Gas|w/ CCS': 'GasCap - w/ CCS',
        'Capacity|Electricity|Gas|w/o CCS': 'GasCap - w/o CCS',
        'Capacity|Electricity|Geothermal': 'GeothermalCap',
        'Capacity|Electricity|Hydro': 'HydroCap',
        'Capacity|Electricity|Nuclear': 'NuclearCap',
        'Capacity|Electricity|Solar': 'SolarCap',
        'Capacity|Electricity|Wind': 'WindCap',

        # Energy variables - map to generic technology names for aggregation
        'Primary Energy': 'AllTech',
        'Secondary Energy|Electricity': 'AllTech',
        'Primary Energy|Biomass': 'BiomassCap - w/o CCS',
        'Primary Energy|Coal': 'CoalCap - w/o CCS',
        'Primary Energy|Gas': 'GasCap - w/o CCS',
        'Primary Energy|Oil': 'OilCap - w/o CCS',
        'Primary Energy|Nuclear': 'NuclearCap',
        'Primary Energy|Hydro': 'HydroCap',
        'Primary Energy|Solar': 'SolarCap',
        'Primary Energy|Wind': 'WindCap',
        'Primary Energy|Geothermal': 'GeothermalCap',
        'Primary Energy|Non-Biomass Renewables': 'AllRenewables',
        'Secondary Energy|Electricity|Biomass': 'BiomassCap - w/o CCS',
        'Secondary Energy|Electricity|Coal': 'CoalCap - w/o CCS',
        'Secondary Energy|Electricity|Gas': 'GasCap - w/o CCS',
        'Secondary Energy|Electricity|Oil': 'OilCap - w/o CCS',
        'Secondary Energy|Electricity|Nuclear': 'NuclearCap',
        'Secondary Energy|Electricity|Hydro': 'HydroCap',
        'Secondary Energy|Electricity|Solar': 'SolarCap',
        'Secondary Energy|Electricity|Wind': 'WindCap',
        'Secondary Energy|Electricity|Geothermal': 'GeothermalCap',
        'Secondary Energy|Electricity|Non-Biomass Renewables': 'AllRenewables',
    }
    return var_tech_map.get(variable)


def step1_load_ngfs() -> Optional[str]:
    """Load NGFS data and format it for the pipeline"""
    print_banner("STEP 1 — Loading and Processing NGFS Data")

    # Try both relative paths
    ngfs_df = None
    for path in ["NGFS_data/NGFS5_Scenarios_Database_R5_regions.csv",
                 "../NGFS_data/NGFS5_Scenarios_Database_R5_regions.csv"]:
        try:
            ngfs_df = pd.read_csv(path)
            print(f"   Loaded NGFS data from {path}: {ngfs_df.shape[0]:,} rows × {ngfs_df.shape[1]} columns")
            break
        except FileNotFoundError:
            continue

    if ngfs_df is None:
        print("❌ NGFS_data/NGFS5_Scenarios_Database_R5_regions.csv not found")
        return None

    # Load mappings
    stringency_map = load_ngfs_stringency_mapping()
    r5_iso3_map = load_r5_to_iso3_mapping()

    # Filter to relevant variables (Capital Cost, Capacity, Prices, Energy)
    relevant_patterns = [
        'Capital Cost',
        'Capacity|Electricity',
        'Capacity Additions|Electricity',
        'Price\\|Primary Energy',
        'Price\\|Secondary Energy\\|Electricity',
        'Secondary Energy\\|Electricity',
        'Primary Energy'
    ]

    mask = ngfs_df['Variable'].str.contains('|'.join(relevant_patterns), case=False, na=False, regex=True)
    ngfs_filtered = ngfs_df[mask].copy()
    print(f"   Filtered to relevant variables: {len(ngfs_filtered):,} rows")

    # Get year columns (use same range as AR6: 2023-2050, but only available years)
    year_cols = [str(y) for y in range(2023, 2051)]  # 2023, 2024, 2025, ..., 2050
    available_years = [c for c in year_cols if c in ngfs_filtered.columns]
    print(f"   Year columns: {available_years}")

    # Melt to long format
    id_cols = ['Model', 'Scenario', 'Region', 'Variable', 'Unit']
    melted = pd.melt(
        ngfs_filtered,
        id_vars=id_cols,
        value_vars=available_years,
        var_name='year',
        value_name='value'
    )

    melted['year'] = pd.to_numeric(melted['year'], errors='coerce')
    melted['value'] = pd.to_numeric(melted['value'], errors='coerce')
    melted = melted[melted['value'].notna()].copy()

    print(f"   Melted to {len(melted):,} rows")

    # Add stringency from mapping
    melted['stringency'] = melted['Scenario'].map(stringency_map).fillna('UNKNOWN')

    # Map technology
    melted['technology'] = melted['Variable'].apply(map_ngfs_variable_to_technology)

    # Add sector (all Power sector for now)
    melted['Sector'] = 'Power'

    # Rename columns to match AR6 format
    melted = melted.rename(columns={
        'Model': 'model',
        'Scenario': 'scenario',
        'Region': 'region',
        'Variable': 'variable',
        'Unit': 'unit'
    })

    # Expand R5 regions to ISO3 countries - VECTORIZED
    print("   Expanding R5 regions to ISO3 countries...")

    # Create a mapping DataFrame
    r5_mappings = []
    for r5_region, iso3_list in r5_iso3_map.items():
        for iso3 in iso3_list:
            r5_mappings.append({'region': r5_region, 'scenario_geography': iso3})

    # If no mappings, use GLOBAL
    if not r5_mappings:
        melted['scenario_geography'] = 'GLOBAL'
        melted['r5_region'] = melted['region']
        expanded_df = melted
    else:
        r5_map_df = pd.DataFrame(r5_mappings)

        # Vectorized merge instead of iterrows
        expanded_df = melted.merge(r5_map_df, on='region', how='left')

        # Fill missing with GLOBAL
        expanded_df['scenario_geography'] = expanded_df['scenario_geography'].fillna('GLOBAL')
        expanded_df['r5_region'] = expanded_df['region']

    print(f"   Expanded to {len(expanded_df):,} rows with ISO3 geography")

    # Save intermediate file
    output_file = "1_intermediate_NGFS_scenario_formatting.csv"
    print(f"   Writing {output_file}... (this may take a minute for {len(expanded_df):,} rows)")
    expanded_df.to_csv(output_file, index=False)
    print(f"✅ Wrote {output_file} | Shape: {expanded_df.shape}")

    memory_release(ngfs_df, ngfs_filtered, melted, expanded_df)
    return output_file


# ================================
# Step 2: Process NGFS Data
# ================================

def step2_process_ngfs() -> None:
    """Process NGFS data to extract metrics"""
    print_banner("STEP 2 — Processing NGFS Metrics")

    try:
        df = pd.read_csv("1_intermediate_NGFS_scenario_formatting.csv")
        print(f"   Loaded: {df.shape}")
    except FileNotFoundError:
        print("❌ Missing Step 1 output")
        return

    # Determine metric type from variable
    def get_metric_type(var: str) -> str:
        if 'Capital Cost' in var:
            return 'capital_cost'
        elif 'Capacity Additions' in var:
            return 'capacity_additions'
        elif 'Capacity|Electricity' in var and 'Additions' not in var:
            return 'capacity'
        elif 'Price|Primary Energy' in var:
            return 'primary_price'
        elif 'Price|Secondary Energy|Electricity' in var:
            return 'electricity_price'
        elif 'Secondary Energy|Electricity' in var:
            return 'secondary_energy'
        elif 'Primary Energy' in var and 'Price' not in var:
            return 'primary_energy'
        return 'other'

    df['metric_type'] = df['variable'].apply(get_metric_type)

    # Pivot to get metrics as columns
    grouping_cols = ['model', 'scenario', 'scenario_geography', 'year', 'Sector', 'technology', 'stringency']

    pivoted = df.pivot_table(
        index=grouping_cols,
        columns='metric_type',
        values='value',
        aggfunc='first'
    ).reset_index()

    pivoted.columns.name = None

    # Unit conversions
    if 'capital_cost' in pivoted.columns:
        pivoted['capital_cost_usd_per_mw'] = pivoted['capital_cost'] * 1000  # kW to MW
        pivoted = pivoted.drop('capital_cost', axis=1)
        print("   ✅ Converted capital cost: USD/kW → USD/MW")

    if 'capacity' in pivoted.columns:
        # Assume GW, convert to MW
        pivoted['capacity_mw'] = pivoted['capacity'] * 1000
        pivoted = pivoted.drop('capacity', axis=1)
        print("   ✅ Converted capacity: GW → MW")

    if 'capacity_additions' in pivoted.columns:
        pivoted['capacity_additions_mw_per_yr'] = pivoted['capacity_additions'] * 1000
        pivoted = pivoted.drop('capacity_additions', axis=1)
        print("   ✅ Converted capacity additions: GW/yr → MW/yr")

    # Energy conversions (EJ/yr to MWh/yr)
    for col in ['primary_energy', 'secondary_energy']:
        if col in pivoted.columns:
            pivoted[f'{col}_mwh_per_yr'] = pivoted[col] * (1e18 / 3.6e9)  # EJ to MWh
            pivoted = pivoted.drop(col, axis=1)
            print(f"   ✅ Converted {col}: EJ/yr → MWh/yr")

    # Calculate fuel intensity = Primary Energy / Secondary Energy
    if 'primary_energy_mwh_per_yr' in pivoted.columns and 'secondary_energy_mwh_per_yr' in pivoted.columns:
        print("🔥 Calculating fuel intensity (Primary Energy / Secondary Energy)...")
        
        # Initialize fuel_intensity column
        pivoted['fuel_intensity'] = np.nan
        
        # Create mask for valid calculations with data quality filters
        # Filter out extreme cases where secondary energy is too small relative to primary energy
        valid_mask = (
            pivoted['primary_energy_mwh_per_yr'].notna() & 
            pivoted['secondary_energy_mwh_per_yr'].notna() &
            (pivoted['primary_energy_mwh_per_yr'] > 0) & 
            (pivoted['secondary_energy_mwh_per_yr'] > 0) &
            # Data quality filter: secondary energy should be at least 1% of primary energy
            (pivoted['secondary_energy_mwh_per_yr'] >= pivoted['primary_energy_mwh_per_yr'] * 0.01) &
            # Additional filter: secondary energy should be at least 1000 MWh/yr
            (pivoted['secondary_energy_mwh_per_yr'] >= 1000)
        )
        
        if valid_mask.any():
            # Calculate fuel intensity: Primary Energy / Secondary Energy
            pivoted.loc[valid_mask, 'fuel_intensity'] = (
                pivoted.loc[valid_mask, 'primary_energy_mwh_per_yr'] / 
                pivoted.loc[valid_mask, 'secondary_energy_mwh_per_yr']
            )
            
            calculated_count = valid_mask.sum()
            print(f"   ✅ Calculated fuel_intensity for {calculated_count:,} entries")
            
            # Show statistics
            fuel_intensity_values = pivoted.loc[valid_mask, 'fuel_intensity']
            print(f"   📊 Fuel Intensity Statistics:")
            print(f"      Mean: {fuel_intensity_values.mean():.3f}")
            print(f"      Median: {fuel_intensity_values.median():.3f}")
            print(f"      Min: {fuel_intensity_values.min():.3f}")
            print(f"      Max: {fuel_intensity_values.max():.3f}")
            
            # Set fuel_intensity to 1.0 for renewable technologies (no fuel conversion loss)
            renewable_keywords = ['Solar', 'Wind', 'Hydro', 'Geothermal', 'Nuclear']
            is_renewable = pivoted['technology'].astype(str).apply(
                lambda x: any(kw in x for kw in renewable_keywords)
            )
            renewable_fuel_intensity_mask = is_renewable & valid_mask
            if renewable_fuel_intensity_mask.any():
                pivoted.loc[renewable_fuel_intensity_mask, 'fuel_intensity'] = 1.0
                renewable_count = renewable_fuel_intensity_mask.sum()
                print(f"   🌱 Set fuel_intensity=1.0 for {renewable_count:,} renewable entries (no conversion loss)")
            
            # Report filtered out cases
            filtered_out = (
                pivoted['primary_energy_mwh_per_yr'].notna() & 
                pivoted['secondary_energy_mwh_per_yr'].notna() &
                (pivoted['primary_energy_mwh_per_yr'] > 0) & 
                (pivoted['secondary_energy_mwh_per_yr'] > 0)
            ) & ~valid_mask
            filtered_count = filtered_out.sum()
            if filtered_count > 0:
                print(f"   ⚠️  Filtered out {filtered_count:,} cases with extreme energy ratios (data quality issues)")
        else:
            print("   ⚠️  No valid primary/secondary energy data for fuel intensity calculation")
    else:
        print("   ⚠️  Primary or secondary energy columns not found")

    # Price conversions (USD/GJ to USD/MWh)
    if 'electricity_price' in pivoted.columns:
        pivoted['scenario_price'] = pivoted['electricity_price'] * 3.6  # GJ to MWh
        pivoted = pivoted.drop('electricity_price', axis=1)
        print("   ✅ Converted electricity price: USD/GJ → USD/MWh")

    if 'primary_price' in pivoted.columns:
        pivoted['fuel_price'] = pivoted['primary_price'] * 3.6
        pivoted = pivoted.drop('primary_price', axis=1)
        print("   ✅ Converted fuel price: USD/GJ → USD/MWh")

    # Set renewables efficiency to 1.0
    renewable_keywords = ['Solar', 'Wind', 'Hydro', 'Geothermal', 'Nuclear']
    is_renewable = pivoted['technology'].astype(str).apply(
        lambda x: any(kw in x for kw in renewable_keywords)
    )
    pivoted['efficiency_decimal'] = np.where(is_renewable, 1.0, np.nan)
    print(f"   Set efficiency=1.0 for {is_renewable.sum()} renewable entries")

    # Set renewable fuel_price to 0
    pivoted.loc[is_renewable, 'fuel_price'] = 0.0
    print(f"   Set fuel_price=0 for {is_renewable.sum()} renewable entries")

    output_file = "2_final_NGFS_filtered.csv"
    pivoted.to_csv(output_file, index=False)
    print(f"✅ Wrote {output_file} | Shape: {pivoted.shape}")

    memory_release(df, pivoted)


# ================================
# Step 3: Finalize to Target Schema
# ================================

def step3_finalize_ngfs_schema() -> None:
    """Convert NGFS data to target schema"""
    print_banner("STEP 3 — Finalize NGFS Target Schema")

    try:
        df = pd.read_csv("2_final_NGFS_filtered.csv")
        print(f"   Loaded: {df.shape}")
    except FileNotFoundError:
        print("❌ Missing Step 2 output")
        return

    target = pd.DataFrame()
    target['scenario_provider'] = df['model']
    target['scenario'] = df['scenario']
    target['scenario_type'] = 'target'
    target['stringency'] = df['stringency']
    target['scenario_geography'] = df['scenario_geography']
    target['sector'] = df['Sector']
    target['technology'] = df['technology']
    target['scenario_year'] = df['year']

    # Technology type
    renewable_keywords = ['Solar', 'Wind', 'Hydro', 'Geothermal', 'Nuclear']
    target['technology_type'] = target['technology'].astype(str).apply(
        lambda x: 'greentech' if any(kw in x for kw in renewable_keywords) else 'carbontech'
    )

    # Price fields
    target['price_unit'] = 'USD/MWh'
    target['price_indicator'] = np.nan
    target['scenario_price'] = df['scenario_price'] if 'scenario_price' in df.columns else np.nan
    target['fuel_price'] = df['fuel_price'] if 'fuel_price' in df.columns else np.nan

    # Pathway (use secondary energy for Power sector)
    target['pathway_unit'] = 'MWh/yr'
    if 'secondary_energy_mwh_per_yr' in df.columns:
        target['scenario_pathway'] = df['secondary_energy_mwh_per_yr']
    elif 'capacity_mw' in df.columns:
        target['scenario_pathway'] = df['capacity_mw']
        target['pathway_unit'] = 'MW'
    else:
        target['scenario_pathway'] = np.nan

    # Capacity factor
    if 'secondary_energy_mwh_per_yr' in df.columns and 'capacity_mw' in df.columns:
        target['scenario_capacity_factor'] = (
            df['secondary_energy_mwh_per_yr'] / (df['capacity_mw'] * 8760)
        ).clip(0, 1)
    else:
        target['scenario_capacity_factor'] = np.nan

    # Fuel intensity (use calculated values from step2)
    if 'fuel_intensity' in df.columns:
        target['fuel_intensity'] = df['fuel_intensity']
    else:
        # Fallback: assume 1.0 for renewables, will be gap-filled for others
        is_renewable = target['technology'].astype(str).apply(
            lambda x: any(kw in x for kw in renewable_keywords)
        )
        target['fuel_intensity'] = np.where(is_renewable, 0.0, np.nan)

    # Country ISO2 list (already at ISO3 level from R5 expansion)
    target['country_iso2_list'] = target['scenario_geography']

    # Technology parameters (will be gap-filled in Step 4)
    for col in ['lifetime_years', 'efficiency_decimal', 'capacity_additions_mw_per_yr',
                'om_cost_usd_per_mw_per_yr', 'capital_cost_usd_per_mw']:
        if col in df.columns:
            target[col] = df[col]
        else:
            target[col] = np.nan

    # Carbon price (not in NGFS data)
    target['carbon_price_usd_per_tco2'] = np.nan

    output_file = "3_final_NGFS_target_schema.csv"
    target.to_csv(output_file, index=False)
    print(f"✅ Wrote {output_file} | Shape: {target.shape}")

    memory_release(df, target)


# ================================
# Step 4: Gap-fill using AR6 Technology Lookup
# ================================

def step4_gapfill_ngfs() -> None:
    """Gap-fill NGFS data using technology_lookup_table.csv"""
    print_banner("STEP 4 — Gap-filling with AR6 Technology Lookup Table")

    import pandas as pd
    import numpy as np
    import step4_gapfill_simple

    # Read NGFS target schema
    df = pd.read_csv("3_final_NGFS_target_schema.csv")
    print(f"   Loaded NGFS target schema: {df.shape}")

    # Read technology lookup table
    lookup_df = pd.read_csv("technology_lookup_table.csv")
    print(f"   Loaded technology lookup table: {lookup_df.shape}")

    # Check fuel_intensity status
    if 'fuel_intensity' in df.columns:
        fuel_available = df['fuel_intensity'].notna().sum()
        print(f"   Fuel intensity: {fuel_available} values already calculated from Primary/Secondary Energy")

    # Call the gap-fill function from step4
    print("   Starting gap-filling process...")
    filled_df = step4_gapfill_simple.ultra_fast_gap_fill(df, lookup_df)

    # Additional gap-filling for fuel_intensity if still missing
    if 'fuel_intensity' in filled_df.columns:
        missing_fuel_intensity = filled_df['fuel_intensity'].isna().sum()
        if missing_fuel_intensity > 0:
            print(f"   Gap-filling remaining {missing_fuel_intensity} fuel_intensity values...")
            
            # For remaining missing values, use technology-specific defaults
            tech_defaults = {
                'CoalCap - w/ CCS': 2.5,  # Typical coal efficiency ~40%
                'CoalCap - w/o CCS': 2.0,  # Typical coal efficiency ~50%
                'GasCap - w/ CCS': 1.8,   # Typical gas efficiency ~55%
                'GasCap - w/o CCS': 1.6,  # Typical gas efficiency ~62%
                'OilCap - w/ CCS': 2.2,   # Typical oil efficiency ~45%
                'OilCap - w/o CCS': 1.8,  # Typical oil efficiency ~55%
                'BiomassCap - w/ CCS': 2.0, # Typical biomass efficiency ~50%
                'BiomassCap - w/o CCS': 1.8, # Typical biomass efficiency ~55%
            }
            
            for tech, default_intensity in tech_defaults.items():
                tech_mask = (filled_df['technology'] == tech) & filled_df['fuel_intensity'].isna()
                filled_df.loc[tech_mask, 'fuel_intensity'] = default_intensity
                if tech_mask.sum() > 0:
                    print(f"     Set {tech}: {tech_mask.sum()} entries to {default_intensity}")

    # Save output
    filled_df.to_csv("4_final_NGFS_gapfilled.csv", index=False)
    print(f"✅ Wrote 4_final_NGFS_gapfilled.csv | Shape: {filled_df.shape}")

    # Print summary of filled values
    if 'fuel_intensity' in filled_df.columns:
        fuel_filled = filled_df['fuel_intensity'].notna().sum()
        print(f"   Fuel intensity: {fuel_filled} values available")
    
    if 'scenario_price' in filled_df.columns:
        price_filled = filled_df['scenario_price'].notna().sum()
        print(f"   Scenario price: {price_filled} values available")

    return filled_df


# ================================
# Step 5: Complete Cases Filtering
# ================================

def step5_ngfs_complete_cases() -> None:
    """Filter NGFS data to complete cases only"""
    print_banner("STEP 5 — NGFS Complete Cases Filtering")
    
    try:
        df = pd.read_csv("4_final_NGFS_gapfilled.csv")
        print(f"   Loaded: {df.shape}")
    except FileNotFoundError:
        print("❌ Missing Step 4 output")
        return
    
    # Define essential columns for complete cases
    essential_cols = [
        'scenario_provider', 'scenario', 'scenario_type', 'scenario_geography',
        'sector', 'technology', 'technology_type', 'scenario_year',
        'efficiency_decimal', 'lifetime_years', 'scenario_capacity_factor',
        'scenario_pathway', 'scenario_price', 'fuel_price', 'fuel_intensity',
        'capital_cost_usd_per_mw', 'om_cost_usd_per_mw_per_yr'
    ]
    
    # Filter to only include columns that exist in the dataset
    available_essential = [col for col in essential_cols if col in df.columns]
    print(f"   Essential columns available: {len(available_essential)}")
    
    # Filter to complete cases
    complete_mask = df[available_essential].notna().all(axis=1)
    complete_df = df[complete_mask].copy()
    
    print(f"   Complete cases: {len(complete_df):,} / {len(df):,} ({len(complete_df)/len(df)*100:.1f}%)")
    
    # Add EBITDA check column
    if 'scenario_price' in complete_df.columns and 'fuel_price' in complete_df.columns:
        # EBITDA = Revenue - Costs (simplified)
        complete_df['ebitda_check'] = (
            complete_df['scenario_price'].notna() & 
            complete_df['fuel_price'].notna() &
            (complete_df['scenario_price'] > complete_df['fuel_price'])
        )
        ebitda_count = complete_df['ebitda_check'].sum()
        print(f"   EBITDA positive cases: {ebitda_count:,} / {len(complete_df):,} ({ebitda_count/len(complete_df)*100:.1f}%)")
    else:
        complete_df['ebitda_check'] = False
        print("   ⚠️  Price data not available for EBITDA check")
    
    # Save output
    output_file = "5_final_NGFS_complete_cases.csv"
    complete_df.to_csv(output_file, index=False)
    print(f"✅ Wrote {output_file} | Shape: {complete_df.shape}")
    
    memory_release(df, complete_df)


# ================================
# Step 6: Scenario Technology Filtering
# ================================

def step6_ngfs_scenario_tech_filter() -> None:
    """Filter NGFS scenarios by technology viability"""
    print_banner("STEP 6 — NGFS Scenario Technology Filtering")
    
    try:
        df = pd.read_csv("5_final_NGFS_complete_cases.csv")
        print(f"   Loaded: {df.shape}")
    except FileNotFoundError:
        print("❌ Missing Step 5 output")
        return
    
    # Get Power sector technologies
    power_techs = df[df['sector'] == 'Power']['technology'].unique()
    print(f"   Power technologies: {len(power_techs)}")
    
    # Check scenario compliance
    def check_scenario_compliance(scenario_name):
        scenario_data = df[df['scenario'] == scenario_name]
        power_data = scenario_data[scenario_data['sector'] == 'Power']
        
        if len(power_data) == 0:
            return False
        
        # Check if all power technologies have at least one viable year
        tech_compliance = {}
        for tech in power_techs:
            tech_data = power_data[power_data['technology'] == tech]
            if len(tech_data) == 0:
                tech_compliance[tech] = False
            else:
                # Check if any year has positive EBITDA
                if 'ebitda_check' in tech_data.columns:
                    tech_compliance[tech] = tech_data['ebitda_check'].any()
                else:
                    tech_compliance[tech] = True  # Assume viable if no EBITDA data
        
        return all(tech_compliance.values())
    
    # Check all scenarios
    scenarios = df['scenario'].unique()
    viable_scenarios = []
    
    print("   Checking scenario compliance...")
    for i, scenario in enumerate(scenarios):
        if i % 10 == 0:
            print(f"     Progress: {i+1}/{len(scenarios)} scenarios")
        
        if check_scenario_compliance(scenario):
            viable_scenarios.append(scenario)
    
    print(f"   Viable scenarios: {len(viable_scenarios)} / {len(scenarios)} ({len(viable_scenarios)/len(scenarios)*100:.1f}%)")
    
    # Add viability flag
    df['scenario_viable'] = df['scenario'].isin(viable_scenarios)
    viable_count = df['scenario_viable'].sum()
    print(f"   Viable data rows: {viable_count:,} / {len(df):,} ({viable_count/len(df)*100:.1f}%)")
    
    # Save output
    output_file = "6_final_NGFS_viable_scenarios.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Wrote {output_file} | Shape: {df.shape}")
    
    memory_release(df)


# ================================
# Main Pipeline
# ================================

def main() -> None:
    print_banner("NGFS Combined Pipeline — Start")

    # Step 1: Load and format NGFS data
    step1_output = step1_load_ngfs()
    if not step1_output:
        print("❌ Failed at Step 1")
        return

    # Step 2: Process metrics
    step2_process_ngfs()

    # Step 3: Finalize schema
    step3_finalize_ngfs_schema()

    # Step 4: Gap-fill with AR6 lookup table
    step4_gapfill_ngfs()

    # Step 5: Complete Cases Filtering
    step5_ngfs_complete_cases()

    # Step 6: Scenario Technology Filtering
    step6_ngfs_scenario_tech_filter()

    print_banner("NGFS Combined Pipeline — Done")
    print("\nOutputs created:")
    print("  - 1_intermediate_NGFS_scenario_formatting.csv")
    print("  - 2_final_NGFS_filtered.csv")
    print("  - 3_final_NGFS_target_schema.csv")
    print("  - 4_final_NGFS_gapfilled.csv")
    print("  - 5_final_NGFS_complete_cases.csv")
    print("  - 6_final_NGFS_viable_scenarios.csv")


if __name__ == "__main__":
    main()
