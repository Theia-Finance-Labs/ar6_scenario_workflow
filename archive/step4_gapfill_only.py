"""
Step 4: Aggregate Technologies and Gap-Fill with Technology Lookup Table

This module handles the aggregation of technology data and gap-filling using
a technology lookup table with vectorized operations.

Refactored from combined_pipeline.py for better maintainability.
"""

import gc
import pandas as pd
import numpy as np
from typing import Dict


def print_banner(title: str) -> None:
    """Print a banner with the given title."""
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")


def memory_release(*objs) -> None:
    """Release memory by deleting objects and calling garbage collection."""
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()


def temporal_interpolation(
    df_in: pd.DataFrame, start_year: int = 2023, end_year: int = 2050
) -> pd.DataFrame:
    """
    Perform temporal interpolation to fill missing years in scenario data.

    - Linear interpolation for missing years between available data points
    - Constant extrapolation for years before first available year (extend backwards to start_year)
    - Constant extrapolation for years after last available year (extend forwards to end_year)

    Args:
        df_in: DataFrame with scenario_year column and value columns to interpolate
        start_year: First year to ensure coverage for (default 2023)
        end_year: Last year to ensure coverage for (default 2050)

    Returns:
        DataFrame with complete yearly data from start_year to end_year
    """
    if "scenario_year" not in df_in.columns:
        print("⚠️ No scenario_year column found, skipping temporal interpolation")
        return df_in

    # Identify numeric columns that should be interpolated
    target_numeric_cols = [
        "scenario_pathway",
        "scenario_price",
        "fuel_price",
        "scenario_capacity_factor",
        "efficiency_decimal",
        "lifetime_years",
        "capacity_additions_mw_per_yr",
        "om_cost_usd_per_mw_per_yr",
        "capital_cost_usd_per_mw",
        "carbon_price_usd_per_tco2",
    ]

    numeric_cols = []
    for col in target_numeric_cols:
        if col in df_in.columns:
            try:
                df_in[col] = pd.to_numeric(df_in[col], errors="coerce")
                if df_in[col].notna().any():
                    numeric_cols.append(col)
            except:
                continue

    if not numeric_cols:
        print("⚠️ No numeric columns found for interpolation")
        return df_in

    # Early exit if all years are already present
    available_years = sorted(df_in["scenario_year"].dropna().unique())
    expected_years = set(range(start_year, end_year + 1))
    missing_years = expected_years - set(available_years)

    if not missing_years:
        print(
            f"✅ All years {start_year}-{end_year} already present, skipping interpolation"
        )
        return df_in

    print(f"Missing years detected: {sorted(missing_years)}")

    # Group by essential identifier columns only
    essential_id_cols = [
        "scenario_provider",
        "scenario",
        "scenario_geography", 
        "sector",
        "technology",
    ]
    grouping_cols = [col for col in essential_id_cols if col in df_in.columns]

    print(
        f"Temporal interpolation: processing {len(numeric_cols)} numeric columns across {len(grouping_cols)} grouping dimensions"
    )

    # Generate all required years
    all_years = list(range(start_year, end_year + 1))
    interpolated_groups = []
    total_groups = 0
    processed_groups = 0

    # Check for any NaN values in grouping columns
    for col in grouping_cols:
        nan_count = df_in[col].isna().sum()
        if nan_count > 0:
            print(f"  WARNING: {col} has {nan_count} NaN values")

    try:
        grouped = df_in.groupby(grouping_cols, dropna=False)
        print(f"  DEBUG: Created groupby object with {grouped.ngroups} groups")

        for group_key, group_df in grouped:
            total_groups += 1

            if group_df.empty:
                continue

            # Get available years and sort
            available_years = sorted(group_df["scenario_year"].dropna().unique())
            if not available_years:
                continue

            # Check if this group needs interpolation
            group_missing = expected_years - set(available_years)
            if not group_missing:
                interpolated_groups.append(group_df)
                continue

            processed_groups += 1

            # Create complete year range for this group
            group_meta = {}
            if isinstance(group_key, tuple):
                for i, col in enumerate(grouping_cols):
                    group_meta[col] = group_key[i]
            else:
                group_meta[grouping_cols[0]] = group_key

            # Create DataFrame with all years for this group
            all_years_df = pd.DataFrame({"scenario_year": all_years})
            for col, val in group_meta.items():
                all_years_df[col] = val

            # Merge with existing data
            merge_cols = ["scenario_year"] + list(group_meta.keys())

            try:
                merged = all_years_df.merge(group_df, on=merge_cols, how="left")
            except Exception as e:
                print(f"  ERROR in merge for group {total_groups}: {e}")
                continue

            # Fill categorical/string columns with values from the group
            categorical_cols = [
                "scenario_type",
                "technology_type",
                "price_unit",
                "price_indicator",
                "fuel_for_price",
                "pathway_unit",
                "country_iso2_list",
                "stringency",
            ]

            for col in categorical_cols:
                if col in merged.columns:
                    non_null_values = merged[col].dropna()
                    if len(non_null_values) > 0:
                        fill_value = non_null_values.iloc[0]
                        merged[col] = merged[col].fillna(fill_value)

            # Interpolate each numeric column using vectorized operations
            for col in numeric_cols:
                if col not in merged.columns:
                    merged[col] = np.nan
                    continue

                # Get indices where we have valid data
                valid_mask = merged[col].notna()
                if not valid_mask.any():
                    continue

                valid_years = merged.loc[valid_mask, "scenario_year"].values
                valid_values = merged.loc[valid_mask, col].values

                if len(valid_values) == 1:
                    # Only one data point - constant extrapolation for all missing
                    merged[col] = merged[col].fillna(valid_values[0])
                else:
                    # Use pandas interpolate for the middle, manual extrapolation for edges
                    merged[col] = merged[col].interpolate(method="linear")

                    # Handle extrapolation for years before first valid point
                    first_valid_year = min(valid_years)
                    first_valid_value = valid_values[np.argmin(valid_years)]
                    before_mask = merged["scenario_year"] < first_valid_year
                    merged.loc[before_mask, col] = first_valid_value

                    # Handle extrapolation for years after last valid point
                    last_valid_year = max(valid_years)
                    last_valid_value = valid_values[np.argmax(valid_years)]
                    after_mask = merged["scenario_year"] > last_valid_year
                    merged.loc[after_mask, col] = last_valid_value

            interpolated_groups.append(merged)

    except Exception as e:
        print(f"  ERROR: Groupby operation failed: {e}")
        return df_in

    if not interpolated_groups:
        print("⚠️ No groups processed during temporal interpolation")
        return df_in

    result = pd.concat(interpolated_groups, ignore_index=True)

    # Summary statistics
    before_count = len(df_in)
    after_count = len(result)
    groups_with_interpolation = processed_groups
    groups_unchanged = total_groups - processed_groups

    print(f"Temporal interpolation completed:")
    print(f"  - {groups_with_interpolation} groups needed interpolation")
    print(f"  - {groups_unchanged} groups were already complete")
    print(f"  - Result: {after_count:,} rows (from {before_count:,})")
    print(f"  - Added {after_count - before_count:,} interpolated rows")

    return result


def gap_fill_from_technology_lookup(df: pd.DataFrame, lookup_df: pd.DataFrame) -> pd.DataFrame:
    """
    VECTORIZED gap-filling using technology lookup table based on technology, year, stringency and ISO2.
    
    For multiple countries in country_iso2_list, take median of lookup values.
    If still missing, fallback to "GLOBAL" geography.
    
    Args:
        df: Main dataframe with potential missing values
        lookup_df: Technology lookup table with reference values
        
    Returns:
        DataFrame with filled values and gap_filled_columns tracking
    """
    print("🔍 Starting VECTORIZED gap-filling using technology lookup table...")
    
    # Columns that can be gap-filled from lookup table
    gap_fillable_columns = [
        "efficiency_decimal",
        "lifetime_years", 
        "capital_cost_usd_per_mw",
        "om_cost_usd_per_mw_per_yr",
        "fuel_price",  # maps to fuel_price_usd_per_mwh in lookup
        "scenario_price",  # maps to electricity_price_usd_per_mwh in lookup
    ]
    
    # Mapping from main df columns to lookup df columns
    column_mapping = {
        "efficiency_decimal": "efficiency_decimal",
        "lifetime_years": "lifetime_years",
        "capital_cost_usd_per_mw": "capital_cost_usd_per_mw",
        "om_cost_usd_per_mw_per_yr": "om_cost_usd_per_mw_per_yr",
        "fuel_price": "fuel_price_usd_per_mwh",
        "scenario_price": "electricity_price_usd_per_mwh",
    }
    
    # Work on a copy
    df = df.copy()
    
    # Initialize tracking column
    if "gap_filled_columns" not in df.columns:
        df["gap_filled_columns"] = ""
    
    filled_counts = {col: 0 for col in gap_fillable_columns}
    
    # STEP 1: Prepare data for vectorized processing
    print("🔄 Preparing data for vectorized operations...")
    
    # Find rows that need gap-filling (have missing values in any gap-fillable column)
    missing_mask = df[gap_fillable_columns].isna().any(axis=1)
    df_to_fill = df[missing_mask].copy()
    
    if len(df_to_fill) == 0:
        print("✅ No missing values found - no gap-filling needed")
        return df
    
    print(f"📊 Processing {len(df_to_fill):,} rows with missing values")
    
    # OPTIMIZED: Use explode() method to expand country lists efficiently
    df_to_fill = df_to_fill.reset_index(drop=False)  # Keep original index
    df_to_fill["original_index"] = df_to_fill.index
    
        # STEP 2: ULTRA-OPTIMIZED APPROACH - No row expansion, direct vectorized operations
    print("⚡ Using direct vectorized matching - no row expansion")
    
    # Prepare lookup data (remove NaNs once)
    lookup_clean = lookup_df.dropna(subset=list(column_mapping.values())).copy()
    if len(lookup_clean) == 0:
        print("⚠️ No valid lookup data available")
        return df

    # STEP 3: Process each column with ultra-efficient vectorized operations
    print("🚀 Direct vectorized gap-filling for all columns...")
    
    lookup_cols_needed = [col for col in column_mapping.values() if col in lookup_clean.columns]
    
    # Process all columns at once using vectorized operations
    for df_col in gap_fillable_columns:
        if df_col not in df.columns:
            continue
            
        lookup_col = column_mapping[df_col]
        if lookup_col not in lookup_cols_needed:
            print(f"  ⚠️ Lookup column {lookup_col} not found, skipping {df_col}")
            continue
        
        print(f"  🔧 Processing: {df_col}")
        
        # Filter to rows that need this column filled AND have lookup data
        needs_filling = mega_merged[df_col].isna() & mega_merged[lookup_col].notna()
        
        if not needs_filling.any():
            print(f"    ✅ No fillable values for {df_col}")
            continue
        
        # Calculate medians by original_index using vectorized groupby
        fill_data = mega_merged[needs_filling]
        medians = fill_data.groupby("original_index")[lookup_col].median()
        
        # Apply medians back to main dataframe
        valid_medians = medians.dropna()
        if len(valid_medians) > 0:
            # Only fill if the original value is still missing
            indices_to_fill = valid_medians.index[df.loc[valid_medians.index, df_col].isna()]
            
            if len(indices_to_fill) > 0:
                df.loc[indices_to_fill, df_col] = valid_medians.loc[indices_to_fill]
                filled_counts[df_col] += len(indices_to_fill)
                
                # FULLY VECTORIZED tracking update - no apply()
                mask_to_update = df.index.isin(indices_to_fill)
                current_tracking = df.loc[mask_to_update, "gap_filled_columns"]
                # Use vectorized string operations instead of apply
                has_existing = current_tracking != ""
                # Update indices with existing tracking
                existing_indices = current_tracking[has_existing].index
                df.loc[existing_indices, "gap_filled_columns"] = current_tracking.loc[existing_indices] + f",{df_col}"
                # Update indices without existing tracking
                new_indices = current_tracking[~has_existing].index
                df.loc[new_indices, "gap_filled_columns"] = df_col
                
                print(f"    ✅ Filled {len(indices_to_fill):,} values for {df_col}")
    
    # STEP 3.5: Technology fallback for Cap technologies (CoalCap -> CoalCap - w/o CCS, etc.)
    print("🔧 Applying technology fallbacks for Cap technologies...")
    
    cap_tech_mapping = {
        "CoalCap": "CoalCap - w/o CCS",
        "BiomassCap": "BiomassCap - w/o CCS", 
        "GasCap": "GasCap - w/o CCS",
        "OilCap": "OilCap - w/o CCS"
    }
    
    # Process technology fallbacks for each column
    for df_col in gap_fillable_columns:
        if df_col not in df.columns:
            continue
            
        lookup_col = column_mapping[df_col]
        if lookup_col not in lookup_cols_needed:
            continue
        
        print(f"  🔧 Technology fallback for: {df_col}")
        
        # Find rows that still need filling after the main join
        still_missing = df[df_col].isna()
        missing_count = still_missing.sum()
        
        if missing_count == 0:
            print(f"    ✅ No missing values for {df_col}")
            continue
        
        # For each Cap technology, try its w/o CCS variant
        tech_filled_count = 0
        for base_tech, fallback_tech in cap_tech_mapping.items():
            # Find missing rows with the base technology
            base_tech_missing = still_missing & (df['technology'] == base_tech)
            base_missing_count = base_tech_missing.sum()
            
            if base_missing_count == 0:
                continue
                
            print(f"    🔄 {base_tech} -> {fallback_tech}: {base_missing_count} candidates")
            
            # Get the missing rows for this technology
            missing_base_df = df[base_tech_missing].copy()
            missing_base_df = missing_base_df.reset_index(drop=False)
            missing_base_df['original_index'] = missing_base_df.index
            
            # Expand countries for these missing rows
            missing_base_df['country_list_split'] = missing_base_df['country_iso2_list'].fillna('').astype(str).apply(
                lambda x: [c.strip() for c in x.split(',') if c.strip()] if x.strip() else ['GLOBAL']
            )
            
            missing_expanded = missing_base_df.explode('country_list_split').copy()
            missing_expanded['iso2_expanded'] = missing_expanded['country_list_split']
            missing_expanded = missing_expanded.drop('country_list_split', axis=1)
            
            # Create fallback lookup data with the fallback technology name
            fallback_lookup = lookup_clean[
                (lookup_clean['technology'] == fallback_tech) &
                (lookup_clean[lookup_col].notna())
            ][['technology', 'year', 'iso2', 'stringency', lookup_col]].copy()
            
            if len(fallback_lookup) == 0:
                print(f"      ⚠️ No fallback data available for {fallback_tech}")
                continue
            
            # Join with fallback data
            fallback_merged = missing_expanded.merge(
                fallback_lookup,
                left_on=['scenario_year', 'stringency', 'iso2_expanded'],
                right_on=['year', 'stringency', 'iso2'],
                how='left',
                suffixes=('', '_fallback')
            )
            
            # Calculate medians for this technology fallback
            if not fallback_merged.empty and lookup_col in fallback_merged.columns:
                has_fallback_data = fallback_merged[lookup_col].notna()
                
                if has_fallback_data.any():
                    fallback_data = fallback_merged[has_fallback_data]
                    tech_medians = fallback_data.groupby('original_index')[lookup_col].median()
                    
                    # Apply fallback values
                    valid_tech_medians = tech_medians.dropna()
                    if len(valid_tech_medians) > 0:
                        # Only fill if still missing in main dataframe
                        indices_to_fill = valid_tech_medians.index[df.loc[valid_tech_medians.index, df_col].isna()]
                        
                        if len(indices_to_fill) > 0:
                            df.loc[indices_to_fill, df_col] = valid_tech_medians.loc[indices_to_fill]
                            filled_counts[df_col] += len(indices_to_fill)
                            tech_filled_count += len(indices_to_fill)
                            
                            # Update tracking for technology fallbacks
                            mask_to_update = df.index.isin(indices_to_fill)
                            current_tracking = df.loc[mask_to_update, "gap_filled_columns"]
                            has_existing = current_tracking != ""
                            # Update indices with existing tracking
                            existing_indices = current_tracking[has_existing].index
                            df.loc[existing_indices, "gap_filled_columns"] = current_tracking.loc[existing_indices] + f",{df_col}(tech_fallback)"
                            # Update indices without existing tracking
                            new_indices = current_tracking[~has_existing].index
                            df.loc[new_indices, "gap_filled_columns"] = f"{df_col}(tech_fallback)"
                            
                            print(f"      ✅ Filled {len(indices_to_fill)} values using {fallback_tech}")
        
        if tech_filled_count > 0:
            print(f"    🎯 Technology fallback filled {tech_filled_count:,} total values for {df_col}")

    # STEP 4: Global fallback using single join
    print("🌍 Applying global fallbacks...")
    global_lookup = lookup_clean[lookup_clean["iso2"] == "GLOBAL"][join_cols].copy()
    
    if not global_lookup.empty:
        # Join with global data for any still-missing values
        global_merged = expanded_df.merge(
            global_lookup,
            left_on=["technology", "scenario_year", "stringency"],
            right_on=["technology", "year", "stringency"],
            how="left",
            suffixes=("", "_global")
        )
        
        # Apply global fallbacks for each column
        for df_col in gap_fillable_columns:
            if df_col not in df.columns:
                continue
                
            lookup_col = column_mapping[df_col]
            if lookup_col not in lookup_cols_needed:
                continue
            
            # Find rows that still need filling and have global data
            still_missing = df[df_col].isna()
            has_global_data = global_merged.groupby("original_index")[lookup_col].first().notna()
            
            # Get intersection of indices that need filling AND have global data
            fillable_indices = still_missing[still_missing].index.intersection(has_global_data[has_global_data].index)
            
            if len(fillable_indices) > 0:
                global_values = global_merged.groupby("original_index")[lookup_col].first()
                global_fills = global_values.loc[fillable_indices].dropna()
                
                if len(global_fills) > 0:
                    df.loc[global_fills.index, df_col] = global_fills
                    filled_counts[df_col] += len(global_fills)
                    
                    # FULLY VECTORIZED tracking for global fills - no apply()
                    mask_to_update = df.index.isin(global_fills.index)
                    current_tracking = df.loc[mask_to_update, "gap_filled_columns"]
                    # Use vectorized string operations instead of apply
                    has_existing = current_tracking != ""
                    # Update indices with existing tracking
                    existing_indices = current_tracking[has_existing].index
                    df.loc[existing_indices, "gap_filled_columns"] = current_tracking.loc[existing_indices] + f",{df_col}(global)"
                    # Update indices without existing tracking
                    new_indices = current_tracking[~has_existing].index
                    df.loc[new_indices, "gap_filled_columns"] = f"{df_col}(global)"
                    
                    print(f"    🌍 Global fallback filled {len(global_fills):,} values for {df_col}")
    
    # Print summary
    total_filled = sum(filled_counts.values())
    print(f"🎯 Vectorized gap-filling completed: {total_filled:,} total values filled")
    for col, count in filled_counts.items():
        if count > 0:
            print(f"  - {col}: {count:,} values")
    
    return df


def step4_gapfill_only() -> None:
    """
    Main step 4 function: gap-fill using lookup table (no aggregation).
    
    This function:
    1. Loads and processes the data
    2. Performs temporal interpolation
    3. Fixes data types
    4. Cleans zero values
    5. Applies gap-filling methodology using technology lookup table
    6. Updates scenario types
    7. Saves results
    """
    print_banner("STEP 4 — Gap-Fill with Technology Lookup Table")
    df = pd.read_csv("3_final_AR6_target_schema.csv")
    
    # Load technology lookup table
    try:
        lookup_df = pd.read_csv("technology_lookup_table.csv")
        print(f"✅ Loaded technology lookup table: {lookup_df.shape}")
    except FileNotFoundError:
        print("❌ Missing technology_lookup_table.csv — aborting step 4")
        return
    
    # Temporal interpolation BEFORE aggregation and gap-filling
    print_banner("STEP 4a — Temporal Interpolation")
    df = temporal_interpolation(df, start_year=2023, end_year=2050)

    # Fix column types before processing
    print("Fixing column data types...")
    
    # Numeric columns that should be float
    numeric_cols = [
        "scenario_pathway",
        "scenario_price",
        "fuel_price",
        "scenario_capacity_factor",
        "efficiency_decimal",
        "lifetime_years",
        "capacity_additions_mw_per_yr",
        "om_cost_usd_per_mw_per_yr",
        "capital_cost_usd_per_mw",
        "carbon_price_usd_per_tco2",
    ]

    for col in numeric_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                non_null = df[col].notna().sum()
                print(f"   {col}: converted to numeric, {non_null:,} non-null values")
            except Exception as e:
                print(f"   WARNING: Failed to convert {col} to numeric: {e}")

    # String columns that should be consistent
    string_cols = [
        "scenario_provider",
        "scenario",
        "scenario_type",
        "scenario_geography",
        "sector",
        "technology",
        "technology_type",
        "price_unit",
        "fuel_for_price",
        "pathway_unit",
        "country_iso2_list",
    ]

    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Year should be integer
    if "scenario_year" in df.columns:
        df["scenario_year"] = pd.to_numeric(
            df["scenario_year"], errors="coerce"
        ).astype("Int64")

    print(f"Data types fixed. Shape: {df.shape}")

    # No aggregation - work directly with individual records
    print("📋 Processing individual records without aggregation")
    processed_df = df.copy()

    print(f"Processing {len(processed_df):,} individual records")

    # Initialize gap-filled tracking column
    if "gap_filled_columns" not in processed_df.columns:
        processed_df["gap_filled_columns"] = ""
    else:
        processed_df["gap_filled_columns"] = (
            processed_df["gap_filled_columns"].fillna("").astype(str)
        )

    # Create stringency BEFORE gap-filling so it can be used in hierarchy
    processed_df["stringency"] = processed_df.get("scenario_type", np.nan)

    # Clean up zero values that should be treated as missing data
    print("🧹 Cleaning zero values that should be treated as missing...")
    zero_to_na_columns = ["lifetime_years", "efficiency_decimal"]

    for col in zero_to_na_columns:
        if col in processed_df.columns:
            zero_count = (processed_df[col] == 0).sum()
            if zero_count > 0:
                print(f"   {col}: converting {zero_count:,} zero values to NA")
                processed_df[col] = processed_df[col].replace(0, np.nan)
            else:
                print(f"   {col}: no zero values found")

    # NEW GAP-FILLING METHODOLOGY using technology lookup table
    print_banner("STEP 4b — Gap-Filling with Technology Lookup Table")
    processed_df = gap_fill_from_technology_lookup(processed_df, lookup_df)

    # Scenario type update based on stringency values
    if "scenario_type" in processed_df.columns:
        # Create baseline mask combining both WITCH and IMAGE baseline scenarios
        baseline_mask = (
            ((processed_df["scenario_provider"] == "WITCH 5.0") 
             & processed_df["scenario"].isin(["CO_CurPol", "EN_NoPolicy"]))
            | ((processed_df["scenario_provider"] == "IMAGE 3.2")
               & processed_df["scenario"].isin(["SSP1-baseline", "SSP2-baseline"]))
        )
        
        # Apply scenario types based on the combined mask
        processed_df.loc[baseline_mask, "scenario_type"] = "baseline"
        processed_df.loc[~baseline_mask, "scenario_type"] = "target"

    # CHECK FOR DUPLICATES based on key grouping columns
    print_banner("STEP 4c — Duplicate Detection")
    key_columns = [
        c for c in [
            "scenario_provider",
            "scenario",
            "scenario_type", 
            "scenario_geography",
            "sector",
            "technology",
            "scenario_year",
        ]
        if c in processed_df.columns
    ]
    
    # Check for duplicates
    duplicate_mask = processed_df.duplicated(subset=key_columns, keep=False)
    duplicate_count = duplicate_mask.sum()
    
    if duplicate_count > 0:
        print(f"⚠️  WARNING: Found {duplicate_count:,} duplicate records based on key grouping columns!")
        print(f"   Key columns: {key_columns}")
        
        # Show some examples of duplicates (FULLY VECTORIZED)
        duplicate_examples = processed_df[duplicate_mask][key_columns].head(5)
        print(f"   First few duplicate examples:")
        for i, (idx, row) in enumerate(duplicate_examples.iterrows()):
            key_values = [str(row[col]) for col in key_columns]
            print(f"     {i+1}. {' | '.join(key_values)}")
            if i >= 4:  # Show max 5 examples
                break
                
        print(f"   Consider reviewing data processing steps to understand why duplicates exist.")
    else:
        print(f"✅ No duplicates found based on key grouping columns: {key_columns}")

    # Write main output with columns ordering similar to original
    base_cols = [
        "scenario_provider",
        "scenario",
        "scenario_type",
        "scenario_geography",
        "sector",
        "technology",
        "technology_type",
        "price_unit",
        "price_indicator",
        "scenario_price",
        "fuel_price",
        "pathway_unit",
        "scenario_pathway",
        "scenario_capacity_factor",
        "scenario_year",
        "country_iso2_list",
        "stringency",
        "gap_filled_columns",  # Include gap-filling tracking
    ]
    extra_cols = [c for c in processed_df.columns if c not in base_cols]
    final_cols = [c for c in base_cols + extra_cols if c in processed_df.columns]
    out_file = "4_final_AR6_gapfilled.csv"
    processed_df[final_cols].to_csv(out_file, index=False)
    print(f"✅ Wrote {out_file} | Shape: {processed_df[final_cols].shape}")

    # Complete-case filtering
    critical = [
        c
        for c in [
            "scenario_pathway",
            "scenario_price",
            "om_cost_usd_per_mw_per_yr",
            "capital_cost_usd_per_mw",
        ]
        if c in processed_df.columns
    ]
    complete_mask = (
        processed_df[critical].notna().all(axis=1)
        if critical
        else pd.Series(True, index=processed_df.index)
    )
    if "efficiency_decimal" in processed_df.columns:
        # Define renewable technologies that should have efficiency data
        renewable_tech_keywords = [
            "Solar",
            "Wind",
            "Hydro",
            "Geothermal",
            "Nuclear",
            "Non-Biomass Renewables",
            "Electricity - Non-Biomass Renewables",
        ]

        # Create mask for renewable technologies
        is_renewable = (
            processed_df["technology"]
            .astype(str)
            .apply(lambda x: any(keyword in x for keyword in renewable_tech_keywords))
        )

        # Efficiency condition: renewable technologies OR legacy Renewables sector should have efficiency
        eff_cond = ~(
            (processed_df["sector"].isin(["Power", "Renewables"]) | is_renewable)
            & processed_df["efficiency_decimal"].isna()
        )
        complete_mask = complete_mask & eff_cond

    final_complete = processed_df.loc[complete_mask, final_cols].copy()
    complete_file = "4_final_AR6_gapfilled_complete.csv"
    final_complete.to_csv(complete_file, index=False)
    print(f"✅ Wrote {complete_file} | Shape: {final_complete.shape}")

    memory_release(df, lookup_df, processed_df, final_complete)


if __name__ == "__main__":
    step4_gapfill_only()
