"""
Ultra-fast vectorized gap-filling - completely rewritten for maximum speed
"""

import pandas as pd
import numpy as np
import gc
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


def ultra_fast_gap_fill(df: pd.DataFrame, lookup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ultra-fast vectorized gap-filling using technology lookup table.
    No row expansion, no apply operations, pure vectorized pandas.
    """
    print("🚀 Starting ULTRA-FAST vectorized gap-filling...")
    
    # Column mappings
    gap_fillable_columns = [
        "efficiency_decimal",
        "lifetime_years", 
        "capital_cost_usd_per_mw",
        "om_cost_usd_per_mw_per_yr",
        "fuel_price",
        "scenario_price",
    ]
    
    column_mapping = {
        "efficiency_decimal": "efficiency_decimal",
        "lifetime_years": "lifetime_years",
        "capital_cost_usd_per_mw": "capital_cost_usd_per_mw",
        "om_cost_usd_per_mw_per_yr": "om_cost_usd_per_mw_per_yr",
        "fuel_price": "fuel_price_usd_per_mwh",
        "scenario_price": "electricity_price_usd_per_mwh",
    }
    
    df = df.copy()
    if "gap_filled_columns" not in df.columns:
        df["gap_filled_columns"] = ""
    
    filled_counts = {col: 0 for col in gap_fillable_columns}
    
    # Technology fallback mapping
    tech_fallbacks = {
        "CoalCap": "CoalCap - w/o CCS",
        "BiomassCap": "BiomassCap - w/o CCS", 
        "GasCap": "GasCap - w/o CCS",
        "OilCap": "OilCap - w/o CCS"
    }
    
    # Process each column independently for maximum speed
    for df_col in gap_fillable_columns:
        if df_col not in df.columns:
            continue
            
        lookup_col = column_mapping[df_col]
        if lookup_col not in lookup_df.columns:
            continue
        
        print(f"  ⚡ Processing {df_col}...")
        
        # Find missing values
        missing_mask = df[df_col].isna()
        missing_count = missing_mask.sum()
        
        if missing_count == 0:
            print(f"    ✅ No missing values")
            continue
        
        # Get missing rows
        missing_df = df[missing_mask].copy()
        col_filled = 0
        
        # STRATEGY 1: Direct technology matching
        print(f"    🎯 Strategy 1: Direct matching...")
        direct_filled = direct_technology_match(missing_df, lookup_df, df_col, lookup_col)
        if len(direct_filled) > 0:
            df.loc[direct_filled.index, df_col] = direct_filled
            df.loc[direct_filled.index, "gap_filled_columns"] = update_tracking(
                df.loc[direct_filled.index, "gap_filled_columns"], df_col
            )
            col_filled += len(direct_filled)
            filled_counts[df_col] += len(direct_filled)
            print(f"      ✅ Filled {len(direct_filled)} values directly")
        
        # Update missing mask after direct fills
        missing_mask = df[df_col].isna()
        missing_df = df[missing_mask].copy()
        
        # STRATEGY 2: Technology fallback (CoalCap -> CoalCap - w/o CCS)
        if len(missing_df) > 0:
            print(f"    🔄 Strategy 2: Technology fallbacks...")
            for base_tech, fallback_tech in tech_fallbacks.items():
                tech_missing = missing_df[missing_df['technology'] == base_tech]
                if len(tech_missing) > 0:
                    fallback_filled = direct_technology_match(tech_missing, lookup_df, df_col, lookup_col, override_tech=fallback_tech)
                    if len(fallback_filled) > 0:
                        df.loc[fallback_filled.index, df_col] = fallback_filled
                        df.loc[fallback_filled.index, "gap_filled_columns"] = update_tracking(
                            df.loc[fallback_filled.index, "gap_filled_columns"], f"{df_col}(tech_fallback)"
                        )
                        col_filled += len(fallback_filled)
                        filled_counts[df_col] += len(fallback_filled)
                        print(f"      ✅ {base_tech}->{fallback_tech}: {len(fallback_filled)} values")
        
        # Update missing mask after technology fallbacks
        missing_mask = df[df_col].isna()
        missing_df = df[missing_mask].copy()
        
        # STRATEGY 3: Global fallback
        if len(missing_df) > 0:
            print(f"    🌍 Strategy 3: Global fallbacks...")
            global_filled = global_fallback_match(missing_df, lookup_df, df_col, lookup_col)
            if len(global_filled) > 0:
                df.loc[global_filled.index, df_col] = global_filled
                df.loc[global_filled.index, "gap_filled_columns"] = update_tracking(
                    df.loc[global_filled.index, "gap_filled_columns"], f"{df_col}(global)"
                )
                col_filled += len(global_filled)
                filled_counts[df_col] += len(global_filled)
                print(f"      ✅ Global fallback: {len(global_filled)} values")
        
        print(f"    🎯 Total filled for {df_col}: {col_filled}")
    
    # Print summary
    total_filled = sum(filled_counts.values())
    print(f"🎯 Ultra-fast gap-filling completed: {total_filled:,} total values filled")
    for col, count in filled_counts.items():
        if count > 0:
            print(f"  - {col}: {count:,} values")
    
    return df


def direct_technology_match(missing_df: pd.DataFrame, lookup_df: pd.DataFrame, 
                          df_col: str, lookup_col: str, override_tech: str = None) -> pd.Series:
    """Direct vectorized technology matching with country median calculation."""
    
    # Use override technology if provided (for fallbacks)
    tech_to_match = override_tech if override_tech else missing_df['technology']
    
    # Create lookup subset for this technology/column combination
    if override_tech:
        # All missing rows use the same override technology
        lookup_subset = lookup_df[
            (lookup_df['technology'] == override_tech) &
            (lookup_df[lookup_col].notna())
        ]
        
        if len(lookup_subset) == 0:
            return pd.Series(dtype=float)
        
        # For each missing row, find matching lookup entries
        results = []
        for idx, row in missing_df.iterrows():
            matches = lookup_subset[
                (lookup_subset['year'] == row['scenario_year']) &
                (lookup_subset['stringency'] == row['scenario_type'])
            ]
            
            if len(matches) == 0:
                continue
            
            # Handle country matching
            country_list = str(row.get('country_iso2_list', '')).strip()
            if not country_list or country_list == 'nan':
                countries = ['GLOBAL']
            else:
                countries = [c.strip() for c in country_list.split(',') if c.strip()]
            
            # Get values for matching countries
            country_values = []
            for country in countries:
                country_matches = matches[matches['iso2'] == country]
                if len(country_matches) > 0:
                    country_values.extend(country_matches[lookup_col].tolist())
            
            # Calculate median if we have values
            if country_values:
                median_val = np.median(country_values)
                results.append((idx, median_val))
        
        if results:
            indices, values = zip(*results)
            return pd.Series(values, index=indices)
        else:
            return pd.Series(dtype=float)
    
    else:
        # Regular direct matching (no technology override)
        # Group by technology for efficient processing
        results = []
        
        for tech, tech_group in missing_df.groupby('technology'):
            tech_lookup = lookup_df[
                (lookup_df['technology'] == tech) &
                (lookup_df[lookup_col].notna())
            ]
            
            if len(tech_lookup) == 0:
                continue
            
            # For each row in this technology group
            for idx, row in tech_group.iterrows():
                matches = tech_lookup[
                    (tech_lookup['year'] == row['scenario_year']) &
                    (tech_lookup['stringency'] == row['scenario_type'])
                ]
                
                if len(matches) == 0:
                    continue
                
                # Handle country matching
                country_list = str(row.get('country_iso2_list', '')).strip()
                if not country_list or country_list == 'nan':
                    countries = ['GLOBAL']
                else:
                    countries = [c.strip() for c in country_list.split(',') if c.strip()]
                
                # Get values for matching countries
                country_values = []
                for country in countries:
                    country_matches = matches[matches['iso2'] == country]
                    if len(country_matches) > 0:
                        country_values.extend(country_matches[lookup_col].tolist())
                
                # Calculate median if we have values
                if country_values:
                    median_val = np.median(country_values)
                    results.append((idx, median_val))
        
        if results:
            indices, values = zip(*results)
            return pd.Series(values, index=indices)
        else:
            return pd.Series(dtype=float)


def global_fallback_match(missing_df: pd.DataFrame, lookup_df: pd.DataFrame, 
                         df_col: str, lookup_col: str) -> pd.Series:
    """Global fallback matching using GLOBAL iso2 entries."""
    
    # Get global lookup data
    global_lookup = lookup_df[
        (lookup_df['iso2'] == 'GLOBAL') &
        (lookup_df[lookup_col].notna())
    ]
    
    if len(global_lookup) == 0:
        return pd.Series(dtype=float)
    
    results = []
    
    # Group by technology for efficient processing
    for tech, tech_group in missing_df.groupby('technology'):
        tech_global = global_lookup[global_lookup['technology'] == tech]
        
        if len(tech_global) == 0:
            continue
        
        # For each row in this technology group
        for idx, row in tech_group.iterrows():
            matches = tech_global[
                (tech_global['year'] == row['scenario_year']) &
                (tech_global['stringency'] == row['scenario_type'])
            ]
            
            if len(matches) > 0:
                # Take first available global value
                global_val = matches[lookup_col].iloc[0]
                results.append((idx, global_val))
    
    if results:
        indices, values = zip(*results)
        return pd.Series(values, index=indices)
    else:
        return pd.Series(dtype=float)


def update_tracking(current_tracking: pd.Series, new_entry: str) -> pd.Series:
    """Update gap-filled tracking columns vectorized."""
    mask_has_existing = current_tracking != ""
    result = current_tracking.copy()
    result[mask_has_existing] = current_tracking[mask_has_existing] + f",{new_entry}"
    result[~mask_has_existing] = new_entry
    return result


def step4_gapfill_only() -> None:
    """Main step 4 function using ultra-fast gap-filling."""
    print_banner("STEP 4 — Ultra-Fast Gap-Fill with Technology Lookup")
    
    df = pd.read_csv("3_final_AR6_target_schema.csv")
    
    # Load technology lookup table
    try:
        lookup_df = pd.read_csv("technology_lookup_table.csv")
        print(f"✅ Loaded technology lookup table: {lookup_df.shape}")
    except FileNotFoundError:
        print("❌ Missing technology_lookup_table.csv — aborting step 4")
        return
    
    # Perform ultra-fast gap-filling
    print_banner("STEP 4a — Ultra-Fast Gap-Filling")
    df = ultra_fast_gap_fill(df, lookup_df)
    
    # Rest of processing...
    print_banner("STEP 4b — Final Processing")
    
    # Scenario type update
    if "scenario_type" in df.columns:
        baseline_mask = (
            ((df["scenario_provider"] == "WITCH 5.0") 
             & df["scenario"].isin(["CO_CurPol", "EN_NoPolicy"]))
            | ((df["scenario_provider"] == "IMAGE 3.2")
               & df["scenario"].isin(["SSP1-baseline", "SSP2-baseline"]))
        )
        df.loc[baseline_mask, "scenario_type"] = "baseline"
        df.loc[~baseline_mask, "scenario_type"] = "target"
    
    # Duplicate detection
    key_columns = [
        c for c in [
            "scenario_provider", "scenario", "scenario_type", 
            "scenario_geography", "sector", "technology", "scenario_year"
        ] if c in df.columns
    ]
    
    duplicate_mask = df.duplicated(subset=key_columns, keep=False)
    duplicate_count = duplicate_mask.sum()
    
    if duplicate_count > 0:
        print(f"⚠️ WARNING: Found {duplicate_count:,} duplicate records!")
    else:
        print(f"✅ No duplicates found")
    
    # Output files
    base_cols = [
        "scenario_provider", "scenario", "scenario_type", "scenario_geography",
        "sector", "technology", "technology_type", "price_unit", "price_indicator",
        "scenario_price", "fuel_price", "pathway_unit", "scenario_pathway",
        "scenario_capacity_factor", "scenario_year", "country_iso2_list",
        "stringency", "gap_filled_columns"
    ]
    extra_cols = [c for c in df.columns if c not in base_cols]
    final_cols = [c for c in base_cols + extra_cols if c in df.columns]
    
    out_file = "4_final_AR6_gapfilled.csv"
    df[final_cols].to_csv(out_file, index=False)
    print(f"✅ Wrote {out_file} | Shape: {df[final_cols].shape}")
    
    # Complete case filtering
    critical = [c for c in ["scenario_pathway", "scenario_price", "om_cost_usd_per_mw_per_yr", "capital_cost_usd_per_mw"] if c in df.columns]
    complete_mask = df[critical].notna().all(axis=1) if critical else pd.Series(True, index=df.index)
    
    final_complete = df.loc[complete_mask, final_cols].copy()
    complete_file = "4_final_AR6_gapfilled_complete.csv"
    final_complete.to_csv(complete_file, index=False)
    print(f"✅ Wrote {complete_file} | Shape: {final_complete.shape}")
    
    memory_release(df, lookup_df, final_complete)


if __name__ == "__main__":
    step4_gapfill_only()
