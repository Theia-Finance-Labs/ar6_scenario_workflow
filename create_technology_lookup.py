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
    
    # Variables we need for the lookup table (simplified to core metrics only)
    target_variables = [
        # Cost variables
        "Capital Cost", "OM Cost",
        # Efficiency variables  
        "Efficiency",
        # Lifetime
        "Lifetime"
    ]
    
    # Target sectors we care about
    target_sectors = ["Steel", "Nuclear", "Gas&Oil", "Cement", "Coal", "Renewables", "Power"]
    
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
                'Capital Cost', 'OM Cost', 'Efficiency', 'Lifetime'
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
    
    # Target technologies we care about
    target_techs = [
        "SolarCap", "WindCap", "HydroCap", "GeothermalCap", "NuclearCap", 
        "BiomassCap", "CoalCap", "GasCap", "OilCap",
        "BiomassCap_w/ CCS", "CoalCap_w/ CCS", "GasCap_w/ CCS", "OilCap_w/ CCS",
        "BiomassCap_w/o CCS", "CoalCap_w/o CCS", "GasCap_w/o CCS", "OilCap_w/o CCS",
        "OceanCap", "Non-Biomass Renewables", "Electricity - Non-Biomass Renewables"
    ]
    
    # Identify year columns (2023-2050 inclusive)
    id_cols = ["Model", "Scenario", "Region", "Variable", "Unit"]
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
        id_vars=id_cols + ['Sector', 'Technology'],  # Include mapped columns
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
    
    # Add geography mapping (ISO2)
    print("   🗺️  Adding geography mapping...")
    melted = add_geography_mapping(melted)
    
    # Add stringency (scenario categorization)
    print("   📊 Adding scenario categorization...")
    melted = add_scenario_categorization(melted)
    
    # INTERPOLATE TO YEARLY DATA FIRST (before taking medians)
    print("   ⏰ Interpolating to complete yearly data (2023-2050)...")
    melted_interpolated = interpolate_to_yearly_data(melted)
    
    # Expand geographies to individual ISO2 codes
    print("   🌍 Expanding to individual countries...")
    melted_expanded = expand_to_iso2_codes(melted_interpolated)
    
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
    """Interpolate data to have complete yearly coverage from 2023-2050"""
    print("      Creating complete yearly time series...")
    
    # Define target years (2023-2050 inclusive)
    target_years = list(range(2023, 2051))
    
    # Group by everything except Year and Value to create time series
    grouping_cols = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Sector', 'Technology', 'country_iso2_list', 'stringency']
    
    interpolated_groups = []
    total_groups = 0
    processed_groups = 0
    
    print(f"      Interpolating {df.groupby(grouping_cols).ngroups} time series...")
    
    for group_key, group_df in df.groupby(grouping_cols, dropna=False):
        total_groups += 1
        
        if total_groups % 10000 == 0:
            print(f"      Progress: {total_groups:,} groups processed")
        
        if group_df.empty:
            continue
            
        # Get available years and values
        year_value_pairs = group_df[['Year', 'Value']].dropna()
        if len(year_value_pairs) < 2:
            # Need at least 2 points for interpolation, skip this group
            continue
            
        available_years = year_value_pairs['Year'].values
        available_values = year_value_pairs['Value'].values
        
        # Create complete year range for this group
        complete_years_df = pd.DataFrame({'Year': target_years})
        
        # Add group metadata
        if isinstance(group_key, tuple):
            for i, col in enumerate(grouping_cols):
                complete_years_df[col] = group_key[i]
        else:
            complete_years_df[grouping_cols[0]] = group_key
        
        # Interpolate values using pandas
        complete_years_df = complete_years_df.set_index('Year')
        
        # Create a series with available data
        value_series = pd.Series(available_values, index=available_years)
        
        # Reindex to complete years and interpolate
        interpolated_series = value_series.reindex(target_years)
        interpolated_series = interpolated_series.interpolate(method='linear')
        
        # Forward fill and backward fill for edge extrapolation
        interpolated_series = interpolated_series.bfill().ffill()
        
        # Add interpolated values back to dataframe
        complete_years_df['Value'] = interpolated_series.values
        complete_years_df = complete_years_df.reset_index()
        
        # Only keep rows where we successfully interpolated
        complete_years_df = complete_years_df.dropna(subset=['Value'])
        
        if not complete_years_df.empty:
            interpolated_groups.append(complete_years_df)
            processed_groups += 1
    
    if not interpolated_groups:
        print("      ⚠️  No groups could be interpolated!")
        return df
    
    # Combine all interpolated groups
    result = pd.concat(interpolated_groups, ignore_index=True)
    
    print(f"      ✅ Interpolated {processed_groups:,} time series")
    print(f"      📈 Expanded from {len(df):,} to {len(result):,} yearly data points")
    
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
            df["stringency"] = df["Category"].fillna("C8")  # Default to least stringent
        else:
            df["stringency"] = "C8"  # Default
    except FileNotFoundError:
        df["stringency"] = "C8"  # Default
    
    return df


def expand_to_iso2_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Expand regional data to individual ISO2 codes"""
    print("      Expanding regional data to individual countries...")
    
    expanded_rows = []
    progress_count = 0
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        progress_count += 1
        if progress_count % 100000 == 0:
            print(f"      Progress: {progress_count:,}/{total_rows:,} ({progress_count/total_rows*100:.1f}%)")
            
        iso2_list_str = str(row['country_iso2_list'])
        if iso2_list_str and iso2_list_str != 'nan' and iso2_list_str.strip():
            iso2_codes = [code.strip() for code in iso2_list_str.split(',') if code.strip()]
            for iso2 in iso2_codes:
                new_row = row.copy()
                new_row['iso2'] = iso2
                expanded_rows.append(new_row)
        else:
            # Keep rows without ISO2 but mark them
            new_row = row.copy()
            new_row['iso2'] = 'UNKNOWN'
            expanded_rows.append(new_row)
    
    expanded_df = pd.DataFrame(expanded_rows)
    print(f"      Expanded to {len(expanded_df):,} rows with individual ISO2 codes")
    
    return expanded_df


def create_lookup_from_melted(melted_df: pd.DataFrame) -> pd.DataFrame:
    """Create lookup table directly from melted data using fast groupby operations"""
    print("📊 Creating lookup table from melted data...")
    
    # Map variable names to our target metrics (simplified)
    variable_metric_map = {
        'Capital Cost': 'capital_cost',
        'OM Cost': 'om_cost', 
        'Efficiency': 'efficiency',
        'Lifetime': 'lifetime'
    }
    
    # Extract metric type from Variable column
    melted_df['metric_type'] = 'unknown'
    for var_pattern, metric in variable_metric_map.items():
        mask = melted_df['Variable'].str.contains(var_pattern, case=False, na=False)
        melted_df.loc[mask, 'metric_type'] = metric
    
    # Filter to known metrics only
    known_metrics = melted_df[melted_df['metric_type'] != 'unknown'].copy()
    print(f"   Found {len(known_metrics):,} rows with target metrics")
    
    # Group by technology, year, iso2, stringency, and metric type, then take median
    print("   🔢 Computing medians by grouping dimensions...")
    lookup_data = known_metrics.groupby([
        'Technology', 'Year', 'iso2', 'stringency', 'metric_type'
    ])['Value'].median().reset_index()
    
    print(f"   Created {len(lookup_data):,} lookup entries")
    
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
    
    print(f"✅ Final lookup table: {len(lookup_pivoted):,} rows × {len(lookup_pivoted.columns)} columns")
    
    return lookup_pivoted

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
    value_cols = ['capital_cost', 'om_cost', 'efficiency', 'lifetime']
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
