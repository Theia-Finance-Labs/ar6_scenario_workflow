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
from pathlib import Path
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
    
    # Prefer Feather but accept the official wide CSVs. CSV input is streamed
    # because either v1.1 source can exceed the memory available on a laptop.
    stems = [
        "AR6_Scenarios_Database_ISO3_v1.1",
        "AR6_Scenarios_Database_R10_regions_v1.1",
    ]
    raw_files = []
    for stem in stems:
        candidates = [
            f"data/{stem}.feather",
            f"data/{stem}.csv",
            f"{stem}.feather",
            f"{stem}.csv",
        ]
        raw_files.append(next((path for path in candidates if Path(path).is_file()), None))

    missing_sources = [
        stem for stem, raw_file in zip(stems, raw_files) if raw_file is None
    ]
    if missing_sources:
        raise FileNotFoundError(
            "Both official AR6 v1.1 inputs are required to regenerate the lookup; "
            f"missing: {', '.join(missing_sources)}"
        )

    mapping = pd.read_csv("ar6_variables_with_mapping.csv")
    
    all_processed = []
    
    for raw_file in raw_files:
        print(f"   📁 Loading {raw_file}...")
        if raw_file.endswith(".csv"):
            source_chunks = pd.read_csv(raw_file, chunksize=100_000)
        else:
            source_chunks = [pd.read_feather(raw_file)]

        file_rows = 0
        for raw_df in source_chunks:
            file_rows += len(raw_df)
            
            # ULTRA-FAST PRE-FILTERING: Only keep variables we need
            variable_mask = raw_df['Variable'].str.contains('|'.join([
                'Capital Cost', 'OM Cost', 'Efficiency', 'Lifetime', 'Price'
            ]), na=False, case=False)
            
            filtered_df = raw_df[variable_mask].copy()
            
            if filtered_df.empty:
                continue
                
            # Load variable mapping to get sectors
            filtered_df = filtered_df.merge(
                mapping, left_on="Variable", right_on="variable", how="inner"
            )
            sector_filtered = filtered_df[
                filtered_df["Sector"].isin(target_sectors)
            ].copy()
            
            if not sector_filtered.empty:
                all_processed.append(sector_filtered)
        print(f"      Scanned {file_rows:,} raw rows")
    
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
    # Include ALL detailed variants that have cost/efficiency/lifetime data
    target_techs = [
        # Solar variants (detailed)
        "SolarCap - CSP", "SolarCap - PV", "SolarCap - Rooftop PV", "SolarCap - Utility PV",
        # Wind variants (detailed) 
        "WindCap", "WindCap - Offshore", "WindCap - Onshore",
        # Other renewables
        "HydroCap", "GeothermalCap", "NuclearCap",
        # Biomass variants
        "BiomassCap - w/ CCS", "BiomassCap - w/o CCS",
        # Coal variants
        "CoalCap - w/ CCS", "CoalCap - w/o CCS",
        # Gas variants
        "GasCap - w/ CCS", "GasCap - w/o CCS", 
        # Oil (no detailed variants with cost data found)
        "Oil",
        # Storage technologies
        "Storage - Battery Capacity", "Storage - Pumped Hydro Storage", "Pumped Hydro Storage",
        # Hydrogen
        "Hydrogen",
        # Additional fuel processing variants for completeness
        "Biomass - Gases", "Biomass - Gases - w/o CCS", "Biomass - Liquids - w/ CCS", "Biomass - Liquids - w/o CCS", 
        "Biomass - w/ CCS", "Biomass - w/o CCS",
        "Coal - Gases", "Coal - Gases - w/o CCS", "Coal - Liquids - w/ CCS", "Coal - Liquids - w/o CCS", "Coal - w/o CCS",
        "Gas - Synthetic", "Gas - w/ CCS", "Gas - w/o CCS",
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
    """Linear interpolation to create complete yearly time series (lifetimes use forward-fill)"""
    print("      Creating complete yearly time series (linear interpolation, lifetimes use forward-fill)...")
    
    # Define target years (2023-2050 inclusive)
    target_years = list(range(2023, 2051))
    
    # Get unique time series identifiers
    grouping_cols = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Sector', 'Technology', 'country_iso2_list', 'stringency']
    
    # Sort by year within each group for proper interpolation
    df_sorted = df.sort_values(grouping_cols + ['Year'])
    
    print(f"      Processing {len(df_sorted):,} input rows...")
    
    # Process each group separately to avoid memory issues
    result_list = []
    
    for name, group in df_sorted.groupby(grouping_cols):
        # Create complete year range for this group
        group_years = group['Year'].values
        min_year, max_year = group_years.min(), group_years.max()
        
        # Use FULL target range (2023-2050) to enable forward/backfill coverage
        target_years_group = target_years
        
        if len(target_years_group) > 0:
            # Create complete DataFrame for this group
            group_df = pd.DataFrame({
                'Year': target_years_group,
                'Value': np.nan
            })
            
            # Merge with actual data
            group_df = group_df.merge(group[['Year', 'Value']], on='Year', how='left', suffixes=('', '_actual'))
            group_df['Value'] = group_df['Value_actual']
            group_df = group_df.drop('Value_actual', axis=1)
            
            # Check if this is a lifetime variable - use forward-fill for lifetimes
            variable_name = name[3]  # Variable is the 4th element in grouping_cols
            is_lifetime = 'Lifetime' in str(variable_name)
            
            if is_lifetime:
                # For lifetime variables, use forward-fill (lifetimes don't change linearly)
                group_df['Value'] = group_df['Value'].ffill().bfill()
            else:
                # For other variables, use linear interpolation
                group_df['Value'] = group_df['Value'].interpolate(method='linear', limit_direction='both')
                
                # After interpolation, extend to full range with forward/backfill
                # This ensures data coverage for full 2023-2050 range
                group_df['Value'] = group_df['Value'].ffill().bfill()
            
            # Add grouping columns back
            for i, col in enumerate(grouping_cols):
                group_df[col] = name[i]
            
            # Only keep rows with valid values
            group_df = group_df.dropna(subset=['Value'])
            
            if len(group_df) > 0:
                result_list.append(group_df)
    
    if result_list:
        final_result = pd.concat(result_list, ignore_index=True)
        print(f"      ✅ Linear interpolation complete (lifetimes used forward-fill)")
        print(f"      📈 Result: {len(final_result):,} yearly data points")
        return final_result
    else:
        print("      ⚠️ No data after interpolation")
        return pd.DataFrame()


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
    """Expand regional data to individual ISO2 codes - MEMORY EFFICIENT"""
    print("      Expanding regional data to individual countries (memory efficient)...")
    
    # Memory efficient approach - process in chunks and only expand regions with reasonable country counts
    print(f"      Processing {len(df):,} rows...")
    
    # Vectorized preparation
    df['country_iso2_list'] = df['country_iso2_list'].fillna('')
    
    # For memory efficiency, limit expansion - max 5 countries per region
    # Split country codes and take first 5 only
    country_series = df['country_iso2_list'].astype(str).str.split(',')
    
    # Vectorized cleaning and limiting
    def clean_and_limit(iso_list):
        if not iso_list or iso_list == ['']:
            return ['GLOBAL']
        cleaned = [code.strip() for code in iso_list if code.strip()]
        if not cleaned:
            return ['GLOBAL']
        # Limit to max 5 countries for memory efficiency
        return cleaned[:5]
    
    print("      Cleaning and limiting country lists...")
    df['iso2_list'] = country_series.apply(clean_and_limit)
    
    # Use pandas explode which is optimized for this operation
    print("      Exploding to individual countries...")
    expanded_df = df.explode('iso2_list')
    
    # Rename and cleanup
    expanded_df = expanded_df.rename(columns={'iso2_list': 'iso2'})
    expanded_df = expanded_df.drop(columns=['country_iso2_list'])
    
    print(f"      ✅ Expanded to {len(expanded_df):,} rows with individual ISO2 codes")
    
    return expanded_df


def create_lookup_from_melted(melted_df: pd.DataFrame, group_name: str = "Unknown") -> pd.DataFrame:
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
            # Group by technology, year, iso2, stringency, and metric type, then take median AND track scenarios
            def median_with_scenarios(group):
                result = pd.Series({
                    'Value': group['Value'].median(),
                    'scenarios_used': ','.join(sorted(group['Scenario'].unique()))
                })
                return result
            
            tech_lookup_data = known_tech_metrics.groupby([
                'Technology', 'Year', 'iso2', 'stringency', 'metric_type'
            ]).apply(median_with_scenarios).reset_index()
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
            # SPEED-OPTIMIZED: First process price data with fuel names for fast filtering
            # Then map to actual power generation technologies during pivot phase
            
            # Step 1: Process fuel price data with scenario tracking (keep fuel names for speed)
            def median_with_scenarios_price(group):
                result = pd.Series({
                    'Value': group['Value'].median(),
                    'scenarios_used': ','.join(sorted(group['Scenario'].unique()))
                })
                return result
            
            # Group price data by fuel type (Technology field contains fuel names like 'Coal', 'Gas')
            fuel_price_data = known_price_metrics.groupby([
                'Technology', 'Year', 'iso2', 'stringency', 'metric_type'
            ]).apply(median_with_scenarios_price).reset_index()
            
            print(f"   Processed {len(fuel_price_data):,} fuel price entries")
            
            # Define technology-to-fuel mapping for actual power generation technologies
            tech_fuel_map = {
                # Coal technologies → coal fuel price
                'CoalCap': 'Coal', 'CoalCap - w/ CCS': 'Coal', 'CoalCap - w/o CCS': 'Coal',
                # Gas technologies → gas fuel price
                'GasCap': 'Gas', 'GasCap - w/ CCS': 'Gas', 'GasCap - w/o CCS': 'Gas',
                # Oil technologies → oil fuel price
                'OilCap': 'Oil', 'OilCap - w/ CCS': 'Oil', 'OilCap - w/o CCS': 'Oil',
                # Biomass technologies → biomass fuel price
                'BiomassCap': 'Biomass', 'BiomassCap - w/ CCS': 'Biomass', 'BiomassCap - w/o CCS': 'Biomass'
            }
            
            # Step 2: Map fuel prices to actual power generation technologies (VECTORIZED)
            price_lookup_data = []
            
            # Map specific fuel prices to technologies
            fuel_price_mapping = {
                'coal_price_usd_per_gj': 'Coal',
                'gas_price_usd_per_gj': 'Gas', 
                'oil_price_usd_per_gj': 'Oil',
                'biomass_price_usd_per_gj': 'Biomass'
            }
            
            for price_metric, fuel_name in fuel_price_mapping.items():
                # Get processed fuel price data for this fuel type
                fuel_mask = (fuel_price_data['metric_type'] == price_metric) & (fuel_price_data['Technology'] == fuel_name)
                if fuel_mask.any():
                    base_fuel_data = fuel_price_data[fuel_mask].copy()
                    
                    # Find all power generation technologies that use this fuel
                    relevant_techs = [tech for tech, fuel in tech_fuel_map.items() if fuel == fuel_name]
                    
                    if relevant_techs:
                        # VECTORIZED expansion: create cartesian product
                        tech_df = pd.DataFrame({'Technology': relevant_techs})
                        tech_df['key'] = 1
                        base_fuel_data['key'] = 1
                        
                        # Merge to create all combinations
                        expanded_fuel_data = base_fuel_data.merge(tech_df, on='key', suffixes=('_fuel', '_tech')).drop('key', axis=1)
                        expanded_fuel_data['Technology'] = expanded_fuel_data['Technology_tech']
                        expanded_fuel_data = expanded_fuel_data.drop(['Technology_fuel', 'Technology_tech'], axis=1)
                        expanded_fuel_data['metric_type'] = 'fuel_price_usd_per_gj'
                        expanded_fuel_data['fuel_for_price'] = fuel_name
                        
                        price_lookup_data.append(expanded_fuel_data)
                        print(f"   Mapped {fuel_name} price to {len(relevant_techs)} technologies: {len(expanded_fuel_data):,} entries")
            
            # Map electricity price to ALL power generation technologies
            elec_mask = (fuel_price_data['metric_type'] == 'electricity_price_usd_per_gj') & (fuel_price_data['Technology'] == 'Electricity')
            if elec_mask.any():
                base_elec_data = fuel_price_data[elec_mask].copy()
                
                # Get ALL power generation technologies (including renewables)
                all_power_techs = [
                    'SolarCap', 'WindCap', 'WindCap - Onshore', 'WindCap - Offshore', 'HydroCap', 'GeothermalCap', 'NuclearCap', 'OceanCap',
                    'CoalCap', 'CoalCap - w/ CCS', 'CoalCap - w/o CCS',
                    'GasCap', 'GasCap - w/ CCS', 'GasCap - w/o CCS', 
                    'OilCap', 'OilCap - w/ CCS', 'OilCap - w/o CCS',
                    'BiomassCap', 'BiomassCap - w/ CCS', 'BiomassCap - w/o CCS'
                ]
                
                if all_power_techs:
                    # VECTORIZED expansion: electricity price for ALL technologies
                    tech_df = pd.DataFrame({'Technology': all_power_techs})
                    tech_df['key'] = 1
                    base_elec_data['key'] = 1
                    
                    # Merge to create all combinations
                    expanded_elec_data = base_elec_data.merge(tech_df, on='key', suffixes=('_fuel', '_tech')).drop('key', axis=1)
                    expanded_elec_data['Technology'] = expanded_elec_data['Technology_tech']
                    expanded_elec_data = expanded_elec_data.drop(['Technology_fuel', 'Technology_tech'], axis=1)
                    expanded_elec_data['metric_type'] = 'electricity_price_usd_per_gj'
                    
                    price_lookup_data.append(expanded_elec_data)
                    print(f"   Mapped electricity price to ALL {len(all_power_techs)} technologies: {len(expanded_elec_data):,} entries")
            
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
    
    # Pivot metric types to columns (need to handle scenarios_used separately)
    print("   🔄 Pivoting metrics to columns...")
    
    # First pivot the values
    lookup_pivoted = lookup_data.pivot_table(
        index=['Technology', 'Year', 'iso2', 'stringency'],
        columns='metric_type',
        values='Value',
        aggfunc='first'
    ).reset_index()
    
    # Then pivot the scenarios_used (combine scenarios from different metrics)
    if 'scenarios_used' in lookup_data.columns:
        scenarios_pivoted = lookup_data.pivot_table(
            index=['Technology', 'Year', 'iso2', 'stringency'],
            columns='metric_type',
            values='scenarios_used',
            aggfunc='first'
        ).reset_index()
        
        # Combine all scenario lists into one column
        scenario_cols = [col for col in scenarios_pivoted.columns if col not in ['Technology', 'Year', 'iso2', 'stringency']]
        if scenario_cols:
            def combine_scenarios(row):
                all_scenarios = set()
                for col in scenario_cols:
                    if pd.notna(row[col]):
                        scenarios = str(row[col]).split(',')
                        all_scenarios.update([s.strip() for s in scenarios if s.strip()])
                return ','.join(sorted(all_scenarios))
            
            scenarios_pivoted['scenarios_used'] = scenarios_pivoted[scenario_cols].apply(combine_scenarios, axis=1)
            
            # Merge scenarios back to main lookup
            lookup_pivoted = lookup_pivoted.merge(
                scenarios_pivoted[['Technology', 'Year', 'iso2', 'stringency', 'scenarios_used']], 
                on=['Technology', 'Year', 'iso2', 'stringency'], 
                how='left'
            )
            print("     ✅ Added combined scenarios_used column")
    
    # Rename columns to match expected format
    lookup_pivoted.columns.name = None
    column_renames = {
        'Technology': 'technology',
        'Year': 'year'
    }
    lookup_pivoted = lookup_pivoted.rename(columns=column_renames)
    
    # CRITICAL: Aggregate detailed technology variants into base technologies
    print("   🔧 Aggregating detailed technology variants into base technologies...")
    
    # Aggregate data for technologies that now have multiple rows
    print("   📊 Aggregating duplicate technologies after mapping...")
    
    # Group by the new aggregated technology names and take median values
    id_cols = ['technology', 'year', 'iso2', 'stringency']
    metric_cols = [col for col in lookup_pivoted.columns if col not in id_cols + ['scenarios_used']]
    
    if metric_cols:
        # Check if we actually have duplicates to aggregate
        duplicate_count = lookup_pivoted.groupby(id_cols).size()
        has_duplicates = (duplicate_count > 1).any()
        
        if has_duplicates:
            print(f"      Found {(duplicate_count > 1).sum()} technology combinations with duplicates")
            
            # Aggregate numeric columns with median (fully vectorized)
            aggregated_metrics = lookup_pivoted.groupby(id_cols)[metric_cols].median().reset_index()
            
            # Aggregate scenarios_used by combining all scenarios (vectorized)
            if 'scenarios_used' in lookup_pivoted.columns:
                # Vectorized scenario aggregation
                scenarios_agg = (lookup_pivoted.groupby(id_cols)['scenarios_used']
                               .apply(lambda x: ','.join(sorted(set(','.join(x.dropna().astype(str)).split(',')))))
                               .reset_index())
                
                # Merge back
                lookup_pivoted = aggregated_metrics.merge(scenarios_agg, on=id_cols, how='left')
            else:
                lookup_pivoted = aggregated_metrics
        else:
            print("      No duplicate technologies found, skipping aggregation")
    else:
        print("      No metric columns found for aggregation")
    
    print(f"   ✅ Aggregated detailed variants into base technologies: {len(lookup_pivoted):,} rows")
    
    # Create base technologies from w/o CCS variants when base doesn't exist OR has no data
    print("   🔄 Creating base technologies from w/o CCS variants (default technology)...")
    
    base_tech_mappings = {
        'BiomassCap': 'BiomassCap - w/o CCS',
        'CoalCap': 'CoalCap - w/o CCS', 
        'GasCap': 'GasCap - w/o CCS',
        'OilCap': 'OilCap - w/o CCS'  # In case it exists
    }
    
    new_base_rows = []
    
    # Also create reverse mappings for completeness (OilCap -> OilCap - w/o CCS)
    reverse_mappings = {
        'OilCap - w/o CCS': 'OilCap'  # Create w/o CCS from base if base has data but w/o CCS doesn't
    }
    
    # Process base technology creation/replacement
    for base_tech, wo_ccs_variant in base_tech_mappings.items():
        # Check if base technology has actual data (not just empty rows)
        base_data = lookup_pivoted[lookup_pivoted['technology'] == base_tech]
        base_has_data = False
        if not base_data.empty:
            # Check if base tech has actual cost data (use available column names)
            cost_cols = [col for col in base_data.columns if 'capital_cost' in col or 'cost' in col]
            if cost_cols:
                base_has_data = base_data[cost_cols[0]].notna().any()
        
        # Get w/o CCS data 
        wo_ccs_data = lookup_pivoted[lookup_pivoted['technology'] == wo_ccs_variant]
        wo_ccs_has_data = False
        if not wo_ccs_data.empty:
            cost_cols = [col for col in wo_ccs_data.columns if 'capital_cost' in col or 'cost' in col]
            if cost_cols:
                wo_ccs_has_data = wo_ccs_data[cost_cols[0]].notna().any()
        
        # If base has no data but w/o CCS has data, replace base with w/o CCS data
        if not base_has_data and wo_ccs_has_data:
            print(f"     🔄 Replacing empty {base_tech} with data from {wo_ccs_variant}")
            
            # Remove existing empty base technology entries
            lookup_pivoted = lookup_pivoted[lookup_pivoted['technology'] != base_tech]
            
            # Copy the w/o CCS data and rename technology to base name
            base_data_new = wo_ccs_data.copy()
            base_data_new['technology'] = base_tech
            new_base_rows.append(base_data_new)
    
    # Process reverse mappings (create w/o CCS from base when needed)
    for wo_ccs_variant, base_tech in reverse_mappings.items():
        # Check if w/o CCS variant exists and has data
        wo_ccs_data = lookup_pivoted[lookup_pivoted['technology'] == wo_ccs_variant]
        wo_ccs_has_data = False
        if not wo_ccs_data.empty:
            cost_cols = [col for col in wo_ccs_data.columns if 'capital_cost' in col or 'cost' in col]
            if cost_cols:
                wo_ccs_has_data = wo_ccs_data[cost_cols[0]].notna().any()
        
        # Check if base has data
        base_data = lookup_pivoted[lookup_pivoted['technology'] == base_tech]
        base_has_data = False
        if not base_data.empty:
            cost_cols = [col for col in base_data.columns if 'capital_cost' in col or 'cost' in col]
            if cost_cols:
                base_has_data = base_data[cost_cols[0]].notna().any()
        
        # If base has data but w/o CCS doesn't, create w/o CCS from base
        if base_has_data and not wo_ccs_has_data:
            print(f"     ➕ Creating {wo_ccs_variant} from {base_tech} for completeness")
            
            # Remove existing empty w/o CCS entries if any
            lookup_pivoted = lookup_pivoted[lookup_pivoted['technology'] != wo_ccs_variant]
            
            # Copy base data and rename to w/o CCS
            wo_ccs_data_new = base_data.copy()
            wo_ccs_data_new['technology'] = wo_ccs_variant
            new_base_rows.append(wo_ccs_data_new)
    
    # Add new base technology rows
    if new_base_rows:
        combined_base = pd.concat(new_base_rows, ignore_index=True)
        lookup_pivoted = pd.concat([lookup_pivoted, combined_base], ignore_index=True)
        print(f"   ✅ Added/replaced {len(combined_base):,} base technology rows")
    
    # CRITICAL: Filter to only technologies that this group should handle
    print("   🔍 Filtering to only technologies relevant to this group...")
    
    # Define which technologies each group should handle
    group_tech_mapping = {
        "Renewables": ["SolarCap", "WindCap", "WindCap - Onshore", "WindCap - Offshore", "HydroCap", "GeothermalCap", "NuclearCap", "OceanCap"],
        "Coal": ["CoalCap", "CoalCap - w/ CCS", "CoalCap - w/o CCS"],
        "Gas": ["GasCap", "GasCap - w/ CCS", "GasCap - w/o CCS"],
        "Biomass": ["BiomassCap", "BiomassCap - w/ CCS", "BiomassCap - w/o CCS"],
        "Oil": ["OilCap", "OilCap - w/ CCS", "OilCap - w/o CCS"],
        "Storage_and_Other": ["Storage - Battery Capacity", "Storage - Pumped Hydro Storage", "Pumped Hydro Storage", "Hydrogen"]
    }
    
    # Use the explicitly provided group name
    if group_name in group_tech_mapping:
        current_group_techs = group_tech_mapping[group_name]
        print(f"     Processing {group_name} group technologies: {current_group_techs}")
        before_filter = len(lookup_pivoted)
        lookup_pivoted = lookup_pivoted[lookup_pivoted['technology'].isin(current_group_techs)].copy()
        after_filter = len(lookup_pivoted)
        print(f"     Filtered from {before_filter:,} to {after_filter:,} rows (removed {before_filter-after_filter:,} non-group entries)")
    else:
        # Fallback: remove only obvious fuel-only entries
        print(f"     Using fallback filtering for unknown group: {group_name}")
        fuel_only_techs = ["Coal", "Gas", "Biomass", "Oil", "Electricity"]
        before_filter = len(lookup_pivoted)
        lookup_pivoted = lookup_pivoted[~lookup_pivoted['technology'].isin(fuel_only_techs)].copy()
        after_filter = len(lookup_pivoted)
        print(f"     Filtered from {before_filter:,} to {after_filter:,} rows (removed {before_filter-after_filter:,} fuel-only entries)")
    
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
    
    # CRITICAL: Apply extreme value filtering after unit conversions
    print("   🚫 Applying extreme value filtering...")
    
    # Define bounds for extreme value filtering (user-specified)
    bounds = {
        'efficiency_decimal': (0.2, 1.0),
        # Preserve legitimate high-price nodes. Only the existing negative
        # price rule remains; there is no upper price cap.
        'fuel_price_usd_per_mwh': (0, None),
        'electricity_price_usd_per_mwh': (0, None),
        'capital_cost_usd_per_mw': (0, 1e7),
        'om_cost_usd_per_mw_per_yr': (0, 5e5)
    }
    
    # Count violations before filtering
    violations_before = 0
    for col, (min_val, max_val) in bounds.items():
        if col in lookup_pivoted.columns:
            out_of_bounds = lookup_pivoted[col] < min_val
            if max_val is not None:
                out_of_bounds = out_of_bounds | (lookup_pivoted[col] > max_val)
            violations = out_of_bounds.sum()
            violations_before += violations
            if violations > 0:
                upper = max_val if max_val is not None else "unbounded"
                print(f"     {col}: {violations:,} values outside bounds ({min_val}-{upper})")
    
    print(f"     Total violations before filtering: {violations_before:,}")
    
    # Apply filtering by setting out-of-bounds values to NaN
    filtered_count = 0
    for col, (min_val, max_val) in bounds.items():
        if col in lookup_pivoted.columns:
            # Create mask for out-of-bounds values
            out_of_bounds = lookup_pivoted[col] < min_val
            if max_val is not None:
                out_of_bounds = out_of_bounds | (lookup_pivoted[col] > max_val)
            count_filtered = out_of_bounds.sum()
            
            if count_filtered > 0:
                lookup_pivoted.loc[out_of_bounds, col] = np.nan
                filtered_count += count_filtered
                print(f"     ✅ Filtered {count_filtered:,} out-of-bounds values for {col}")
    
    if filtered_count > 0:
        print(f"     ✅ Total extreme values filtered: {filtered_count:,}")
    else:
        print("     ✅ No extreme values found to filter")
    
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
                  if col not in ['technology', 'year', 'iso2', 'stringency', 'scenarios_used']]
    
    if metric_cols:
        # Filter to known stringencies
        known_data = lookup_pivoted[lookup_pivoted['stringency'].isin(known_stringencies)]
        
        if not known_data.empty:
            # Group and calculate means across stringencies (fully vectorized) - only numeric columns
            fallback_df = known_data.groupby(['technology', 'year', 'iso2'])[metric_cols].mean().reset_index()
            fallback_df['stringency'] = 'UNKNOWN'
            
            # Handle scenarios_used separately - combine all scenarios from different stringencies (vectorized)
            if 'scenarios_used' in known_data.columns:
                # Vectorized scenario combination
                scenarios_df = (known_data.groupby(['technology', 'year', 'iso2'])['scenarios_used']
                              .apply(lambda x: ','.join(sorted(set(','.join(x.dropna().astype(str)).split(',')))))
                              .reset_index())
                fallback_df = fallback_df.merge(scenarios_df, on=['technology', 'year', 'iso2'], how='left')
            
            # Reorder columns to match original format
            all_cols = ['technology', 'year', 'iso2', 'stringency'] + metric_cols
            if 'scenarios_used' in fallback_df.columns:
                all_cols.append('scenarios_used')
            fallback_df = fallback_df[all_cols]
            
            fallback_data = fallback_df.to_dict('records')
    
    if fallback_data:
        fallback_df = pd.DataFrame(fallback_data)
        
        # Remove fallback entries that already exist to avoid duplicates
        existing_combos = set(lookup_pivoted[['technology', 'year', 'iso2', 'stringency']].apply(tuple, axis=1))
        fallback_combos = fallback_df[['technology', 'year', 'iso2', 'stringency']].apply(tuple, axis=1)
        new_fallback_mask = ~fallback_combos.isin(existing_combos)
        fallback_df = fallback_df[new_fallback_mask]
        
        if not fallback_df.empty:
            final_lookup = pd.concat([lookup_pivoted, fallback_df], ignore_index=True)
            print(f"   Added {len(fallback_df):,} new UNKNOWN stringency fallback entries")
        else:
            final_lookup = lookup_pivoted
            print("   All UNKNOWN fallback entries already exist, none added")
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
                  if col not in ['technology', 'year', 'iso2', 'stringency', 'scenarios_used']]
    
    if metric_cols:
        # Group and calculate means across geographies (fully vectorized) - only numeric columns
        global_fallback_df = final_lookup.groupby(['technology', 'year', 'stringency'])[metric_cols].mean().reset_index()
        global_fallback_df['iso2'] = 'GLOBAL'
        
        # Handle scenarios_used separately - combine all scenarios from different geographies (vectorized)
        if 'scenarios_used' in final_lookup.columns:
            # Vectorized scenario combination across geographies
            geo_scenarios_df = (final_lookup.groupby(['technology', 'year', 'stringency'])['scenarios_used']
                              .apply(lambda x: ','.join(sorted(set(','.join(x.dropna().astype(str)).split(',')))))
                              .reset_index())
            global_fallback_df = global_fallback_df.merge(geo_scenarios_df, on=['technology', 'year', 'stringency'], how='left')
        
        # Reorder columns to match original format
        all_cols = ['technology', 'year', 'iso2', 'stringency'] + metric_cols
        if 'scenarios_used' in global_fallback_df.columns:
            all_cols.append('scenarios_used')
        global_fallback_df = global_fallback_df[all_cols]
        
        global_fallback_data = global_fallback_df.to_dict('records')
    
    if global_fallback_data:
        global_fallback_df = pd.DataFrame(global_fallback_data)
        
        # Remove global fallback entries that already exist to avoid duplicates
        existing_combos = set(final_lookup[['technology', 'year', 'iso2', 'stringency']].apply(tuple, axis=1))
        global_fallback_combos = global_fallback_df[['technology', 'year', 'iso2', 'stringency']].apply(tuple, axis=1)
        new_global_mask = ~global_fallback_combos.isin(existing_combos)
        global_fallback_df = global_fallback_df[new_global_mask]
        
        if not global_fallback_df.empty:
            final_lookup = pd.concat([final_lookup, global_fallback_df], ignore_index=True)
            print(f"   Added {len(global_fallback_df):,} new Global geography fallback entries")
        else:
            print("   All Global fallback entries already exist, none added")
    else:
        print("   No global fallback entries needed")
    
    # FINAL COVERAGE GUARANTEE: Ensure ALL target technologies have UNKNOWN/GLOBAL entries
    print("   🎯 Ensuring complete coverage for all target technologies...")
    
    # Only check coverage for technologies that this group should handle
    if group_name in group_tech_mapping:
        target_power_technologies = group_tech_mapping[group_name]
    else:
        # Fallback for unknown groups
        target_power_technologies = [
            "SolarCap", "WindCap", "WindCap - Onshore", "WindCap - Offshore", "HydroCap", "GeothermalCap", "NuclearCap", 
            "BiomassCap", "CoalCap", "GasCap", "OilCap",
            "BiomassCap - w/ CCS", "CoalCap - w/ CCS", "GasCap - w/ CCS", "OilCap - w/ CCS",
            "BiomassCap - w/o CCS", "CoalCap - w/o CCS", "GasCap - w/o CCS", "OilCap - w/o CCS"
        ]
    
    target_years = list(range(2023, 2051))
    required_combinations = []
    
    for tech in target_power_technologies:
        for year in target_years:
            # Ensure UNKNOWN stringency + GLOBAL geography exists
            required_combinations.append({
                'technology': tech,
                'year': year,
                'iso2': 'GLOBAL', 
                'stringency': 'UNKNOWN'
            })
    
    # Check which combinations are missing
    required_df = pd.DataFrame(required_combinations)
    existing_combos = final_lookup[['technology', 'year', 'iso2', 'stringency']].drop_duplicates()
    
    # Find missing combinations
    merged = required_df.merge(existing_combos, on=['technology', 'year', 'iso2', 'stringency'], how='left', indicator=True)
    missing = merged[merged['_merge'] == 'left_only'][['technology', 'year', 'iso2', 'stringency']]
    
    if not missing.empty:
        print(f"   ⚠️  Found {len(missing):,} missing UNKNOWN/GLOBAL combinations, creating fallback entries...")
        
        # Create fallback entries with average values across all available data
        metric_cols = [col for col in final_lookup.columns 
                      if col not in ['technology', 'year', 'iso2', 'stringency', 'scenarios_used']]
        
        if metric_cols:
            # Calculate global average for each technology across all years/stringencies/geographies - only numeric columns
            tech_averages = final_lookup.groupby('technology')[metric_cols].mean().reset_index()
            
            # Handle scenarios separately - get all scenarios for each technology (vectorized)
            if 'scenarios_used' in final_lookup.columns:
                # Vectorized scenario combination across all data for each technology
                tech_scenarios = (final_lookup.groupby('technology')['scenarios_used']
                                .apply(lambda x: ','.join(sorted(set(','.join(x.dropna().astype(str)).split(',')))))
                                .reset_index())
                tech_averages = tech_averages.merge(tech_scenarios, on='technology', how='left')
            
            # Create missing entries
            missing_entries = []
            for _, row in missing.iterrows():
                tech = row['technology']
                year = row['year']
                
                # Get average values for this technology
                tech_avg = tech_averages[tech_averages['technology'] == tech]
                
                if not tech_avg.empty:
                    entry = {
                        'technology': tech,
                        'year': year,
                        'iso2': 'GLOBAL',
                        'stringency': 'UNKNOWN'
                    }
                    # Add averaged metric values (numeric columns only)
                    for col in metric_cols:
                        if col in tech_avg.columns:
                            entry[col] = tech_avg[col].iloc[0]
                        else:
                            entry[col] = None
                    
                    # Add scenarios_used if available
                    if 'scenarios_used' in tech_avg.columns:
                        entry['scenarios_used'] = tech_avg['scenarios_used'].iloc[0]
                    else:
                        entry['scenarios_used'] = ''
                    
                    missing_entries.append(entry)
            
            if missing_entries:
                missing_df = pd.DataFrame(missing_entries)
                final_lookup = pd.concat([final_lookup, missing_df], ignore_index=True)
                print(f"   ✅ Added {len(missing_df):,} missing coverage entries")
    else:
        print("   ✅ All target technologies have complete UNKNOWN/GLOBAL coverage")
    
    # Final verification
    coverage_check = final_lookup[
        (final_lookup['stringency'] == 'UNKNOWN') & 
        (final_lookup['iso2'] == 'GLOBAL') &
        (final_lookup['technology'].isin(target_power_technologies))
    ]
    
    unique_tech_coverage = coverage_check['technology'].nunique()
    print(f"   📊 Final coverage: {unique_tech_coverage}/{len(target_power_technologies)} target technologies have UNKNOWN/GLOBAL entries")
    
    if unique_tech_coverage < len(target_power_technologies):
        missing_techs = set(target_power_technologies) - set(coverage_check['technology'].unique())
        print(f"   ⚠️  Missing coverage for: {missing_techs}")
    
    print(f"✅ Final lookup table with guaranteed coverage: {len(final_lookup):,} rows × {len(final_lookup.columns)} columns")
    
    return final_lookup

def process_technology_group(raw_data: pd.DataFrame, tech_group_name: str, target_techs: list) -> pd.DataFrame:
    """Process a specific group of technologies to avoid memory issues"""
    print(f"\n{'='*60}")
    print(f"🔧 PROCESSING TECHNOLOGY GROUP: {tech_group_name}")
    print(f"Technologies: {target_techs}")
    print(f"{'='*60}")
    
    # Create a custom version of process_melted_data_for_lookup for this group
    melted_data = process_melted_data_for_lookup_group(raw_data, target_techs)
    if melted_data is None or melted_data.empty:
        print(f"❌ No usable melted data for {tech_group_name}")
        return pd.DataFrame()
    
    # Create lookup table from melted data
    lookup_table = create_lookup_from_melted(melted_data, tech_group_name)
    if lookup_table is None or lookup_table.empty:
        print(f"❌ No lookup table produced for {tech_group_name}")
        return pd.DataFrame()
    
    # Save intermediate file
    intermediate_file = f"temp_lookup_{tech_group_name.lower().replace(' ', '_')}.csv"
    lookup_table.to_csv(intermediate_file, index=False)
    print(f"💾 Saved intermediate lookup: {intermediate_file}")
    print(f"   Shape: {lookup_table.shape}")
    
    # Clear memory
    del melted_data
    
    return lookup_table

def process_melted_data_for_lookup_group(raw_df: pd.DataFrame, target_techs: list) -> pd.DataFrame:
    """Process raw AR6 data for a specific group of technologies"""
    print(f"⚡ Processing data for {len(target_techs)} technologies...")
    
    # Identify year columns (2023-2050 inclusive)
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
    
    # Set lifetime values of 0 to NA BEFORE taking medians
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

def main():
    print("=" * 80)
    print("CHUNKED TECHNOLOGY LOOKUP TABLE GENERATOR")
    print("🚀 Processing ALL models from raw data in technology groups")
    print("=" * 80)
    
    # Load and pre-filter raw AR6 data
    raw_data = load_and_process_raw_ar6()
    if raw_data is None:
        return
    
    # Define technology groups to process separately
    # Note: Electricity is included in ALL groups to ensure all technologies get electricity prices
    technology_groups = {
        "Renewables": [
            "SolarCap - CSP", "SolarCap - PV", "SolarCap - Rooftop PV", "SolarCap - Utility PV",
            "WindCap", "WindCap - Offshore", "WindCap - Onshore",
            "HydroCap", "GeothermalCap", "NuclearCap", "OceanCap",
            "Electricity"  # For electricity price data
        ],
        "Coal": [
            "CoalCap - w/ CCS", "CoalCap - w/o CCS",
            "Coal - Gases", "Coal - Gases - w/o CCS", "Coal - Liquids - w/ CCS", "Coal - Liquids - w/o CCS", "Coal - w/o CCS",
            "Coal",  # For coal price data
            "Electricity"  # For electricity price data
        ],
        "Gas": [
            "GasCap - w/ CCS", "GasCap - w/o CCS",
            "Gas - Synthetic", "Gas - w/ CCS", "Gas - w/o CCS",
            "Gas",  # For gas price data
            "Electricity"  # For electricity price data
        ],
        "Biomass": [
            "BiomassCap - w/ CCS", "BiomassCap - w/o CCS",
            "Biomass - Gases", "Biomass - Gases - w/o CCS", "Biomass - Liquids - w/ CCS", "Biomass - Liquids - w/o CCS", 
            "Biomass - w/ CCS", "Biomass - w/o CCS",
            "Biomass",  # For biomass price data
            "Electricity"  # For electricity price data
        ],
        "Oil": [
            "Oil",  # For oil price data and technology data
            "Electricity"  # For electricity price data
        ],
        "Storage_and_Other": [
            "Storage - Battery Capacity", "Storage - Pumped Hydro Storage", "Pumped Hydro Storage",
            "Hydrogen",
            "Electricity"  # For electricity price data
        ]
    }
    
    # Process each group separately
    all_lookup_tables = []
    
    for group_name, tech_list in technology_groups.items():
        print(f"\n🔄 Processing {group_name} group...")
        group_lookup = process_technology_group(raw_data, group_name, tech_list)
        
        if not group_lookup.empty:
            all_lookup_tables.append(group_lookup)
            print(f"✅ {group_name} completed: {len(group_lookup):,} rows")
        else:
            print(f"⚠️  {group_name} produced no data")
    
    # Combine all lookup tables
    if all_lookup_tables:
        print(f"\n🔗 Combining {len(all_lookup_tables)} technology group lookup tables...")
        combined_lookup = pd.concat(all_lookup_tables, ignore_index=True)
        
        # Save final combined file
        output_file = "technology_lookup_table.csv"
        combined_lookup.to_csv(output_file, index=False)
        print(f"💾 Saved final lookup table: {output_file}")
        print(f"   Shape: {combined_lookup.shape}")
        
        # Summary statistics
        print("\n📊 Summary:")
        value_cols = [
            'capital_cost_usd_per_mw', 'om_cost_usd_per_mw_per_yr', 'efficiency_decimal', 'lifetime_years',
            'fuel_price_usd_per_mwh', 'electricity_price_usd_per_mwh'
        ]
        
        for col in value_cols:
            if col in combined_lookup.columns:
                filled = combined_lookup[col].notna().sum()
                total = len(combined_lookup)
                pct = (filled / total) * 100 if total > 0 else 0
                print(f"   {col}: {filled:,}/{total:,} ({pct:.1f}%) filled")
        
        print(f"\n✅ Chunked lookup table generation complete!")
        print(f"   📈 Processed data from {raw_data['Model'].nunique()} models")
        print(f"   🌍 Covers {combined_lookup['iso2'].nunique()} countries/regions") 
        print(f"   🔧 Processed {len(technology_groups)} technology groups")
        print(f"   ⚡ Ready for use in main pipeline")
        
        print("\n📦 Retaining intermediate lookup files for reproducibility")
    else:
        print("❌ No lookup tables were generated")
    
    print("\n💡 Usage: The main pipeline will automatically detect and use this lookup table.")

if __name__ == "__main__":
    main()
