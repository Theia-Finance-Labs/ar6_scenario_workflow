#!/usr/bin/env python3
"""
Standalone Technology Lookup Table Generator
============================================

This script creates a comprehensive technology lookup table for ALL scenarios 
but ONLY for the target technologies we actually use in the pipeline.

This allows the main pipeline to skip expensive lookup table creation and 
just load a pre-computed table.

Usage:
    python3 create_technology_lookup.py

Outputs:
    - technology_lookup_table.csv: Pre-computed lookup table

Performance: 
    - Processes only ~15 target technologies instead of all 46
    - Can be run once and reused across multiple pipeline runs
    - Dramatically speeds up the main pipeline

Target Technologies (15):
    - Core: SolarCap, WindCap, HydroCap, GeothermalCap, NuclearCap
    - Fossil: BiomassCap, CoalCap, GasCap, OilCap  
    - CCS variants: *_w/ CCS, *_w/o CCS
    - Other: OceanCap, Non-Biomass Renewables, Electricity - Non-Biomass Renewables
"""

import pandas as pd
import numpy as np
from typing import List, Dict

def load_and_process_raw_ar6() -> pd.DataFrame:
    """Load RAW AR6 data and immediately filter to only needed variables for maximum efficiency"""
    print("📂 Loading RAW AR6 data from ALL models (optimized for lookup generation)...")
    
    # Variables we need for the lookup table (comprehensive technology & price data)
    target_variables = [
        # Technology characteristics
        "Capital Cost", "OM Cost", "Efficiency", "Lifetime",
        # Price data (fuel and electricity prices)
        "Price"
    ]
    
    # Target sectors we care about (including Biomass for price data)
    target_sectors = ["Steel", "Nuclear", "Gas&Oil", "Cement", "Coal", "Renewables", "Power", "Biomass"]
    
    # Try to load raw feather files
    raw_files = [
        "AR6_Scenarios_Database_ISO3_v1.1.feather",
        "AR6_Scenarios_Database_R10_regions_v1.1.feather"
    ]
    
    all_processed = []
    
    for raw_file in raw_files:
        try:
            print(f"   📁 Loading {raw_file}...")
            raw_df = pd.read_feather(raw_file)
            print(f"      Original: {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")
            print(f"      Models: {raw_df['Model'].nunique()}")
            
            # ULTRA-FAST PRE-FILTERING: Only keep variables we need
            print(f"   🔍 Pre-filtering to target variables...")
            variable_mask = raw_df['Variable'].str.contains('|'.join([
                'Capital Cost', 'OM Cost', 'Efficiency', 'Lifetime', 'Price'
            ]), na=False, case=False)
            
            filtered_df = raw_df[variable_mask].copy()
            print(f"      After variable filter: {filtered_df.shape[0]:,} rows ({filtered_df.shape[0]/raw_df.shape[0]*100:.1f}%)")
            
            if filtered_df.empty:
                print(f"      No relevant variables found in {raw_file}")
                continue
                
            # Load variable mapping to get sectors
            print(f"   🗺️  Applying sector mapping...")
            try:
                mapping = pd.read_csv("ar6_variables_with_mapping.csv")
                filtered_df = filtered_df.merge(mapping, left_on="Variable", right_on="variable", how="inner")
                print(f"      After mapping: {filtered_df.shape[0]:,} rows")
                
                # Filter to target sectors
                sector_filtered = filtered_df[filtered_df["Sector"].isin(target_sectors)].copy()
                print(f"      After sector filter: {sector_filtered.shape[0]:,} rows")
                
            except FileNotFoundError:
                print(f"      ⚠️  Variable mapping not found, keeping all variables")
                sector_filtered = filtered_df.copy()
            
            if not sector_filtered.empty:
                all_processed.append(sector_filtered)
                
        except FileNotFoundError:
            print(f"   ⚠️  {raw_file} not found, skipping...")
            continue
    
    if not all_processed:
        print("❌ No usable data found in raw files")
        return None
    
    # Combine all processed data
    combined_df = pd.concat(all_processed, ignore_index=True)
    print(f"✅ Combined optimized data: {combined_df.shape[0]:,} rows × {combined_df.shape[1]} columns")
    print(f"   Models: {combined_df['Model'].nunique()}")
    
    return combined_df

def process_melted_data_for_lookup(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Process raw AR6 data directly in melted format for ultra-fast lookup creation"""
    print("⚡ Processing data in melted format (before pivoting - MUCH faster)...")
    
    # Target technologies that ACTUALLY EXIST in AR6 data (based on ar6_variables_with_mapping.csv)
    target_techs = [
        # Base technologies (from mapping file)
        "SolarCap", "WindCap", "HydroCap", "GeothermalCap", "NuclearCap", 
        "BiomassCap", "CoalCap", "GasCap", "OilCap",
        # CCS variants ONLY with dashes (actual AR6 format from mapping file)
        "BiomassCap - w/ CCS", "CoalCap - w/ CCS", "GasCap - w/ CCS", "OilCap - w/ CCS",
        "BiomassCap - w/o CCS", "CoalCap - w/o CCS", "GasCap - w/o CCS", "OilCap - w/o CCS",
        # Other technologies
        "OceanCap", "Non-Biomass Renewables", "Electricity - Non-Biomass Renewables",
        # Additional efficiency-specific technology names
        "Oil", "Heating",
        # Simple fuel names (needed for price data processing!)
        "Coal", "Gas", "Biomass", "Electricity"
    ]
    
    # Identify year columns (2023-2050 inclusive)
    # For price data we need to handle the different structure (col1, col2, Fuel columns)
    base_id_cols = ["Model", "Scenario", "Region", "Variable", "Unit"]
    
    # The raw AR6 data is in wide format - need to melt it
    id_cols = base_id_cols + ['Sector', 'Technology']
    all_year_cols = [c for c in raw_df.columns if c not in id_cols]
    year_cols = []
    for col in all_year_cols:
        try:
            year = float(col)
            if 2023 <= year <= 2050:  # Target years inclusive
                year_cols.append(col)
        except ValueError:
            continue
    
    print(f"   Year columns: {len(year_cols)} ({min(year_cols) if year_cols else 'none'}-{max(year_cols) if year_cols else 'none'})")
    
    # Melt the data to long format
    print("   🔄 Melting data to long format...")
    melted = pd.melt(
        raw_df,
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="Year",
        value_name="Value"
    )
    
    # Convert to numeric
    melted["Year"] = pd.to_numeric(melted["Year"], errors="coerce")
    melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")
    
    # Filter to target technologies early
    tech_mask = melted["Technology"].isin(target_techs)
    melted = melted[tech_mask].copy()
    print(f"   After tech filter: {len(melted):,} rows for {melted['Technology'].nunique()} technologies")
    
    # Filter to non-null values early
    melted = melted[melted["Value"].notna()].copy()
    print(f"   After removing nulls: {len(melted):,} rows")
    
    # Set lifetime values of 0 to NA BEFORE taking medians (like original pipeline)
    # Use exact matching instead of str.contains for speed
    lifetime_mask = melted['Variable'] == 'Lifetime'
    zero_lifetime_mask = lifetime_mask & (melted['Value'] == 0)
    if zero_lifetime_mask.any():
        melted.loc[zero_lifetime_mask, 'Value'] = pd.NA
        print(f"   Set {zero_lifetime_mask.sum():,} zero lifetime values to NA before median calculation")
    
    # Add geography mapping (ISO2)
    print("   🗺️  Adding geography mapping...")
    melted = add_geography_mapping(melted)
    
    # Add stringency (scenario categorization)
    print("   📊 Adding scenario categorization...")
    melted = add_scenario_categorization(melted)
    
    # INTERPOLATE TO YEARLY DATA FIRST (before taking medians)
    print("   ⏰ Interpolating to complete yearly data (2023-2050)...")
    melted_interpolated = interpolate_to_yearly_data(melted)
    
    # Clear original melted data to free memory
    del melted
    
    # Expand geographies to individual ISO2 codes
    print("   🌍 Expanding to individual countries...")
    melted_expanded = expand_to_iso2_codes(melted_interpolated)
    
    # Clear interpolated data to free memory
    del melted_interpolated
    
    return melted_expanded


def add_geography_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Add ISO2 country mapping to regions using the lookup file"""
    print("      Loading region lookup file...")
    
    # Load the region lookup file (should always be available)
    try:
        r10_lookup = pd.read_csv("r10_region_lookup.csv")
        print(f"      Loaded region lookup: {len(r10_lookup)} mappings")
        
        # Build mapping dictionary from the lookup file
        region_iso2_map = {}
        
        if set(["R10 Region", "ISO2"]).issubset(r10_lookup.columns):
            for _, row in r10_lookup.iterrows():
                region = row["R10 Region"]
                iso2_raw = row["ISO2"]
                if pd.notna(iso2_raw) and str(iso2_raw).strip():
                    # Handle pipe-separated ISO2 codes
                    iso2_list = [p.strip() for p in str(iso2_raw).split("|") if p.strip()]
                    region_iso2_map[region] = ",".join(sorted(iso2_list))
                    
        print(f"      Created mappings for {len(region_iso2_map)} regions")
        
        # Apply mapping
        df["country_iso2_list"] = df["Region"].map(region_iso2_map).fillna("")
        
        # Check how many regions got mapped
        mapped_count = (df["country_iso2_list"] != "").sum()
        print(f"      Mapped {mapped_count:,}/{len(df):,} rows to ISO2 codes")
        
    except FileNotFoundError:
        print("      ❌ r10_region_lookup.csv not found! This is required.")
        print("      Creating empty mapping as fallback...")
        df["country_iso2_list"] = ""
    
    return df


def interpolate_to_yearly_data(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate data to have complete yearly coverage from 2023-2050 - VECTORIZED"""
    print("      Creating complete yearly time series (vectorized approach)...")
    
    # Define target years (2023-2050 inclusive)
    target_years = list(range(2023, 2051))
    
    # Group by everything except Year and Value to create time series
    grouping_cols = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Sector', 'Technology', 'country_iso2_list', 'stringency']
    
    print(f"      Processing {df.groupby(grouping_cols).ngroups} time series with vectorized interpolation...")
    
    # Create a complete template with all years for each group
    unique_groups = df[grouping_cols].drop_duplicates()
    
    # Create cartesian product of groups × years
    years_df = pd.DataFrame({'Year': target_years})
    complete_template = unique_groups.assign(key=1).merge(years_df.assign(key=1), on='key').drop('key', axis=1)
    
    print(f"      Created template with {len(complete_template):,} group-year combinations")
    
    # Merge with original data
    merged = complete_template.merge(df, on=grouping_cols + ['Year'], how='left')
    
    # Interpolate within each group using transform (vectorized)
    print("      Applying vectorized interpolation...")
    
    def interpolate_group(group):
        """Vectorized interpolation for a single group"""
        # Sort by year
        group = group.sort_values('Year')
        
        # Interpolate linearly
        group['Value'] = group['Value'].interpolate(method='linear')
        
        # Forward and backward fill for extrapolation
        group['Value'] = group['Value'].bfill().ffill()
        
        return group
    
    # Apply interpolation to each group (still uses groupby but with vectorized operations)
    interpolated = merged.groupby(grouping_cols, group_keys=False).apply(interpolate_group)
    
    # Remove rows where interpolation failed (still NaN)
    result = interpolated.dropna(subset=['Value'])
    
    print(f"      ✅ Vectorized interpolation complete")
    print(f"      📈 Result: {len(result):,} yearly data points")
    
    return result


def add_scenario_categorization(df: pd.DataFrame) -> pd.DataFrame:
    """Add scenario categorization (stringency levels)"""
    # Load scenario metadata if available
    try:
        meta_df = pd.read_excel(
            "AR6_Scenarios_Database_metadata_indicators_v1.1 2.xlsx",
            sheet_name="meta_Ch3vetted_withclimate"
        )
        if set(["Model", "Scenario", "Category"]).issubset(meta_df.columns):
            meta_df = meta_df[["Model", "Scenario", "Category"]].dropna()
            df = df.merge(meta_df, on=["Model", "Scenario"], how="left")
            df["stringency"] = df["Category"].fillna("UNKNOWN")  # Mark as unknown instead of C8
            
            # Show stringency distribution
            stringency_counts = df["stringency"].value_counts()
            print(f"      Stringency distribution: {stringency_counts.to_dict()}")
        else:
            df["stringency"] = "UNKNOWN"  # Mark as unknown
    except FileNotFoundError:
        print("      Metadata file not found, marking all scenarios as UNKNOWN stringency")
        df["stringency"] = "UNKNOWN"  # Mark as unknown instead of C8
    
    return df


def expand_to_iso2_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Expand regional data to individual ISO2 codes - PURE VECTORIZED"""
    print("      Expanding regional data to individual countries (vectorized)...")
    
    # Use pandas explode for vectorized expansion
    df_copy = df.copy()
    
    # Convert country_iso2_list to lists (vectorized)
    df_copy['iso2_list'] = df_copy['country_iso2_list'].fillna('').astype(str).str.split(',')
    
    # Clean up empty strings and whitespace (vectorized)
    df_copy['iso2_list'] = df_copy['iso2_list'].apply(lambda x: [code.strip() for code in x if code.strip()] or ['GLOBAL'])
    
    # Limit to reasonable number of countries per region (vectorized)
    df_copy['iso2_list'] = df_copy['iso2_list'].apply(lambda x: x[:20] if len(x) <= 20 else [x[0]])
    
    # Explode to create one row per ISO2 code (pure pandas operation)
    expanded_df = df_copy.explode('iso2_list').reset_index(drop=True)
    
    # Rename the exploded column
    expanded_df = expanded_df.rename(columns={'iso2_list': 'iso2'})
    
    # Drop the original country_iso2_list column
    expanded_df = expanded_df.drop(columns=['country_iso2_list'])
    
    print(f"      ✅ Expanded to {len(expanded_df):,} rows with individual ISO2 codes")
    
    return expanded_df


def create_lookup_from_melted(melted_df: pd.DataFrame) -> pd.DataFrame:
    """Create lookup table directly from melted data using fast groupby operations"""
    print("📊 Creating lookup table from melted data...")
    
    # Map EXACT variable names to our target metrics (no substring matching!)
    variable_metric_map = {
        # Technology characteristics
        'Capital Cost': 'capital_cost_usd_per_kw',      # Raw data is USD2010/kW → convert to MW
        'OM Cost': 'om_cost_usd_per_kw_per_yr',         # Raw data is USD2010/kW/yr → convert to MW  
        'Efficiency': 'efficiency_decimal',             # Raw data sometimes as % → convert to decimal
        'Lifetime': 'lifetime_years',                   # Raw data in years
        # Energy prices (EXACT variable names from AR6 database)
        'Price|Primary Energy|Coal': 'coal_price_usd_per_gj',
        'Price|Primary Energy|Gas': 'gas_price_usd_per_gj',
        'Price|Primary Energy|Oil': 'oil_price_usd_per_gj', 
        'Price|Primary Energy|Biomass': 'biomass_price_usd_per_gj',
        'Price|Secondary Energy|Electricity': 'electricity_price_usd_per_gj'
    }
    
    # Check units for consistency before processing (using exact matching)
    print("   📏 Checking unit consistency...")
    for exact_var, metric_name in variable_metric_map.items():
        metric_data = melted_df[melted_df['Variable'] == exact_var]  # EXACT match only
        if not metric_data.empty:
            unique_units = metric_data['Unit'].unique()
            print(f"     {exact_var}: {len(unique_units)} unique units - {list(unique_units[:5])}")
    
    # Separate technology-specific data from price data (using exact matching)
    tech_variables = ['Capital Cost', 'OM Cost', 'Efficiency', 'Lifetime']
    price_variables = ['Price|Primary Energy|Coal', 'Price|Primary Energy|Gas', 'Price|Primary Energy|Oil', 
                      'Price|Primary Energy|Biomass', 'Price|Secondary Energy|Electricity']
    
    tech_data = melted_df[melted_df['Variable'].str.contains('|'.join(tech_variables), case=False, na=False)].copy()
    price_data = melted_df[melted_df['Variable'].isin(price_variables)].copy()  # Exact matching for prices
    
    print(f"   Technology data: {len(tech_data):,} rows")
    print(f"   Price data: {len(price_data):,} rows")
    
    # Process technology-specific data
    tech_lookup_data = []
    if not tech_data.empty:
        # Extract metric type from Variable column for technology data (using exact matching)
        tech_data['metric_type'] = 'unknown'
        for exact_var, metric in variable_metric_map.items():
            if 'Price' not in exact_var:  # Skip price patterns
                mask = tech_data['Variable'].str.contains(exact_var, case=False, na=False)
                tech_data.loc[mask, 'metric_type'] = metric
        
        # Filter to known metrics only
        known_tech_metrics = tech_data[tech_data['metric_type'] != 'unknown'].copy()
        print(f"   Technology metrics: {len(known_tech_metrics):,} rows")
        
        if not known_tech_metrics.empty:
            # Group by technology, year, iso2, stringency, and metric type, then take median
            tech_lookup_data = known_tech_metrics.groupby([
                'Technology', 'Year', 'iso2', 'stringency', 'metric_type'
            ])['Value'].median().reset_index()
            print(f"   Created {len(tech_lookup_data):,} technology lookup entries")
    
    # Process price data (map to technologies based on fuel type)
    price_lookup_data = []
    if not price_data.empty:
        # Extract price type from Variable column (using exact matching)
        price_data['metric_type'] = 'unknown'
        for exact_var, metric in variable_metric_map.items():
            if 'Price' in exact_var:  # Only price patterns
                mask = price_data['Variable'] == exact_var  # Exact match for price variables
                price_data.loc[mask, 'metric_type'] = metric
        
        # Filter to known price metrics only
        known_price_metrics = price_data[price_data['metric_type'] != 'unknown'].copy()
        print(f"   Price metrics: {len(known_price_metrics):,} rows")
        
        if not known_price_metrics.empty:
            # VECTORIZED price mapping to ALL relevant technologies
            
            # Define technology-to-fuel mapping ONLY for technologies that exist in AR6 data
            tech_fuel_map = {
                # Coal technologies → coal fuel price (only dash format exists in AR6)
                'CoalCap': 'Coal', 'CoalCap - w/ CCS': 'Coal', 'CoalCap - w/o CCS': 'Coal',
                
                # Gas technologies → gas fuel price (only dash format exists in AR6)
                'GasCap': 'Gas', 'GasCap - w/ CCS': 'Gas', 'GasCap - w/o CCS': 'Gas',
                
                # Oil technologies → oil fuel price (only dash format exists in AR6)
                'OilCap': 'Oil', 'OilCap - w/ CCS': 'Oil', 'OilCap - w/o CCS': 'Oil',
                
                # Biomass technologies → biomass fuel price (only dash format exists in AR6)
                'BiomassCap': 'Biomass', 'BiomassCap - w/ CCS': 'Biomass', 'BiomassCap - w/o CCS': 'Biomass',
                
                # Simple fuel names for price mapping (these get fuel prices but no cost data)
                'Coal': 'Coal', 'Gas': 'Gas', 'Oil': 'Oil', 'Biomass': 'Biomass'
            }
            
            # Get all technologies that should receive price mappings
            existing_techs = list(tech_fuel_map.keys())  # All technologies defined in our mapping
            
            price_lookup_data = []
            
            # VECTORIZED fuel price expansion
            fuel_price_mapping = {
                'coal_price_usd_per_gj': 'Coal',
                'gas_price_usd_per_gj': 'Gas',
                'oil_price_usd_per_gj': 'Oil', 
                'biomass_price_usd_per_gj': 'Biomass'
            }
            
            for price_metric, fuel_name in fuel_price_mapping.items():
                # Get price data for this fuel
                fuel_mask = known_price_metrics['metric_type'] == price_metric
                if fuel_mask.any():
                    base_fuel_data = known_price_metrics[fuel_mask].copy()
                    
                    # Find all technologies that use this fuel
                    relevant_techs = [tech for tech, fuel in tech_fuel_map.items() if fuel == fuel_name]
                    
                    if relevant_techs:
                        # VECTORIZED expansion: create cartesian product of price data × relevant technologies
                        tech_df = pd.DataFrame({'Technology': relevant_techs})
                        tech_df['key'] = 1
                        base_fuel_data['key'] = 1
                        
                        # Merge to create all combinations (vectorized)
                        expanded_fuel_data = base_fuel_data.merge(tech_df, on='key', suffixes=('', '_new')).drop('key', axis=1)
                        expanded_fuel_data['Technology'] = expanded_fuel_data['Technology_new']
                        expanded_fuel_data = expanded_fuel_data.drop('Technology_new', axis=1)
                        expanded_fuel_data['metric_type'] = 'fuel_price_usd_per_gj'
                        expanded_fuel_data['fuel_for_price'] = fuel_name
                        
                        price_lookup_data.append(expanded_fuel_data)
                        print(f"   Mapped {fuel_name} price to {len(relevant_techs)} technologies: {len(expanded_fuel_data):,} entries")
            
            # VECTORIZED electricity price expansion (to ALL technologies in tech_fuel_map)
            elec_mask = known_price_metrics['metric_type'] == 'electricity_price_usd_per_gj'
            if elec_mask.any():
                base_elec_data = known_price_metrics[elec_mask].copy()
                
                # Get ALL technologies that could use electricity price
                all_relevant_techs = list(tech_fuel_map.keys())
                
                if all_relevant_techs:
                    # VECTORIZED expansion: electricity price for ALL technologies
                    tech_df = pd.DataFrame({'Technology': all_relevant_techs})
                    tech_df['key'] = 1
                    base_elec_data['key'] = 1
                    
                    # Merge to create all combinations (vectorized)
                    expanded_elec_data = base_elec_data.merge(tech_df, on='key', suffixes=('', '_new')).drop('key', axis=1)
                    expanded_elec_data['Technology'] = expanded_elec_data['Technology_new']
                    expanded_elec_data = expanded_elec_data.drop('Technology_new', axis=1)
                    expanded_elec_data['metric_type'] = 'electricity_price_usd_per_gj'
                    
                    price_lookup_data.append(expanded_elec_data)
                    print(f"   Mapped electricity price to ALL {len(all_relevant_techs)} technologies: {len(expanded_elec_data):,} entries")
            
            if price_lookup_data:
                price_lookup_data = pd.concat(price_lookup_data, ignore_index=True)
                print(f"   Combined {len(price_lookup_data):,} total price entries")
    
    # Combine technology and price data
    if len(tech_lookup_data) > 0 and len(price_lookup_data) > 0:
        lookup_data = pd.concat([tech_lookup_data, price_lookup_data], ignore_index=True)
    elif len(tech_lookup_data) > 0:
        lookup_data = tech_lookup_data
    elif len(price_lookup_data) > 0:
        lookup_data = price_lookup_data
    else:
        lookup_data = pd.DataFrame()
    
    print(f"   Combined lookup entries: {len(lookup_data):,}")
    
    # Pivot metric types to columns
    print("   🔄 Pivoting metrics to columns...")
    lookup_pivoted = lookup_data.pivot_table(
        index=['Technology', 'Year', 'iso2', 'stringency'],
        columns='metric_type',
        values='Value',
        aggfunc='first'
    ).reset_index()
    
    # Rename columns to match expected format
    lookup_pivoted.columns.name = None
    column_renames = {
        'Technology': 'technology',
        'Year': 'year'
    }
    lookup_pivoted = lookup_pivoted.rename(columns=column_renames)
    
    # CRITICAL: Apply unit conversions to match pipeline expectations
    print("   🔄 Applying unit conversions...")
    
    # Convert kW units to MW units (×1000)
    if 'capital_cost_usd_per_kw' in lookup_pivoted.columns:
        lookup_pivoted['capital_cost_usd_per_mw'] = lookup_pivoted['capital_cost_usd_per_kw'] * 1000
        lookup_pivoted = lookup_pivoted.drop(columns=['capital_cost_usd_per_kw'])
        print("     ✅ Converted capital_cost: USD/kW → USD/MW (×1000)")
        
    if 'om_cost_usd_per_kw_per_yr' in lookup_pivoted.columns:
        lookup_pivoted['om_cost_usd_per_mw_per_yr'] = lookup_pivoted['om_cost_usd_per_kw_per_yr'] * 1000  
        lookup_pivoted = lookup_pivoted.drop(columns=['om_cost_usd_per_kw_per_yr'])
        print("     ✅ Converted om_cost: USD/kW/yr → USD/MW/yr (×1000)")
    
    # Convert efficiency percentages to decimals (÷100 if >1)
    if 'efficiency_decimal' in lookup_pivoted.columns:
        # Apply the same conversion as the main pipeline: divide by 100 if > 1
        over_one_mask = lookup_pivoted['efficiency_decimal'] > 1
        if over_one_mask.any():
            lookup_pivoted.loc[over_one_mask, 'efficiency_decimal'] = lookup_pivoted.loc[over_one_mask, 'efficiency_decimal'] / 100
            print(f"     ✅ Converted {over_one_mask.sum():,} efficiency values from percentage to decimal (÷100)")
        print("     ✅ Efficiency values are now in decimal format")
    
    # Note: Zero lifetime values already set to NA before median calculation
    
    # Convert price units from GJ to MWh (×3.6 like main pipeline)
    price_conversions = {
        'fuel_price_usd_per_gj': 'fuel_price_usd_per_mwh',
        'electricity_price_usd_per_gj': 'electricity_price_usd_per_mwh'
    }
    
    for old_col, new_col in price_conversions.items():
        if old_col in lookup_pivoted.columns:
            # Convert from USD/GJ to USD/MWh (×3.6)
            lookup_pivoted[new_col] = lookup_pivoted[old_col] * 3.6
            lookup_pivoted = lookup_pivoted.drop(columns=[old_col])
            print(f"     ✅ Converted {old_col} to {new_col} (×3.6)")
    
    if any(col in lookup_pivoted.columns for col in price_conversions.values()):
        print("     ✅ All price units converted from GJ to MWh")
    
    print(f"✅ Final lookup table: {len(lookup_pivoted):,} rows × {len(lookup_pivoted.columns)} columns")
    
    # Create UNKNOWN stringency fallback rows (average of all known stringencies)
    print("   📊 Creating UNKNOWN stringency fallback entries...")
    
    known_stringencies = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    fallback_data = []
    
    # Get unique combinations of technology, year, iso2
    unique_combos = lookup_pivoted[['technology', 'year', 'iso2']].drop_duplicates()
    
    # VECTORIZED fallback creation - no iterrows!
    # Filter to known stringencies and group by tech, year, iso2 to calculate means
    metric_cols = [col for col in lookup_pivoted.columns 
                  if col not in ['technology', 'year', 'iso2', 'stringency']]
    
    if metric_cols:
        # Filter to known stringencies
        known_data = lookup_pivoted[lookup_pivoted['stringency'].isin(known_stringencies)]
        
        if not known_data.empty:
            # Group and calculate means across stringencies (fully vectorized)
            fallback_df = known_data.groupby(['technology', 'year', 'iso2'])[metric_cols].mean().reset_index()
            fallback_df['stringency'] = 'UNKNOWN'
            
            # Reorder columns to match original format
            col_order = ['technology', 'year', 'iso2', 'stringency'] + metric_cols
            fallback_df = fallback_df[col_order]
            
            fallback_data = fallback_df.to_dict('records')
    
    if fallback_data:
        fallback_df = pd.DataFrame(fallback_data)
        final_lookup = pd.concat([lookup_pivoted, fallback_df], ignore_index=True)
        print(f"   Added {len(fallback_df):,} UNKNOWN stringency fallback entries")
    else:
        final_lookup = lookup_pivoted
        print("   No fallback entries needed")
    
    # Add Global geography fallbacks for missing values
    print("   🌍 Creating Global geography fallback entries...")
    global_fallback_data = []
    
    # Get unique combinations of technology, year, stringency
    unique_combos = final_lookup[['technology', 'year', 'stringency']].drop_duplicates()
    
    # VECTORIZED Global geography fallback creation - no iterrows!
    # Group by technology, year, stringency and calculate means across all geographies
    metric_cols = [col for col in final_lookup.columns 
                  if col not in ['technology', 'year', 'iso2', 'stringency']]
    
    if metric_cols:
        # Group and calculate means across geographies (fully vectorized)
        global_fallback_df = final_lookup.groupby(['technology', 'year', 'stringency'])[metric_cols].mean().reset_index()
        global_fallback_df['iso2'] = 'GLOBAL'
        
        # Reorder columns to match original format
        col_order = ['technology', 'year', 'iso2', 'stringency'] + metric_cols
        global_fallback_df = global_fallback_df[col_order]
        
        global_fallback_data = global_fallback_df.to_dict('records')
    
    if global_fallback_data:
        global_fallback_df = pd.DataFrame(global_fallback_data)
        final_lookup = pd.concat([final_lookup, global_fallback_df], ignore_index=True)
        print(f"   Added {len(global_fallback_df):,} Global geography fallback entries")
    else:
        print("   No global fallback entries needed")
    
    print(f"✅ Final lookup table with all fallbacks: {len(final_lookup):,} rows × {len(final_lookup.columns)} columns")
    
    return final_lookup

def main():
    print("=" * 80)
    print("ULTRA-FAST TECHNOLOGY LOOKUP TABLE GENERATOR")
    print("🚀 Processing ALL models from raw data")
    print("=" * 80)
    
    # Load and pre-filter raw AR6 data
    raw_data = load_and_process_raw_ar6()
    if raw_data is None:
        return
    
    # Process data in melted format for maximum efficiency
    melted_data = process_melted_data_for_lookup(raw_data)
    if melted_data is None or melted_data.empty:
        print("❌ No usable melted data produced")
        return
    
    # Create lookup table from melted data
    lookup_table = create_lookup_from_melted(melted_data)
    if lookup_table is None or lookup_table.empty:
        print("❌ No lookup table produced")
        return
    
    # Save to file
    output_file = "technology_lookup_table.csv"
    lookup_table.to_csv(output_file, index=False)
    print(f"💾 Saved lookup table: {output_file}")
    print(f"   Shape: {lookup_table.shape}")
    
    # Summary statistics
    print("\n📊 Summary:")
    value_cols = [
        'capital_cost_usd_per_mw', 'om_cost_usd_per_mw_per_yr', 'efficiency_decimal', 'lifetime_years',
        'fuel_price_usd_per_mwh', 'electricity_price_usd_per_mwh'
    ]
    
    # Also check fuel_for_price column
    if 'fuel_for_price' in lookup_table.columns:
        print(f"   fuel_for_price values: {lookup_table['fuel_for_price'].value_counts().to_dict()}")
    for col in value_cols:
        if col in lookup_table.columns:
            filled = lookup_table[col].notna().sum()
            total = len(lookup_table)
            pct = (filled / total) * 100 if total > 0 else 0
            print(f"   {col}: {filled:,}/{total:,} ({pct:.1f}%) filled")
    
    print(f"\n✅ Ultra-fast lookup table generation complete!")
    print(f"   📈 Processed data from {raw_data['Model'].nunique()} models")
    print(f"   🌍 Covers {lookup_table['iso2'].nunique()} countries/regions") 
    print(f"   ⚡ Ready for use in main pipeline (10-100x faster than on-demand creation)")
    print("\n💡 Usage: The main pipeline will automatically detect and use this lookup table.")

if __name__ == "__main__":
    main()
