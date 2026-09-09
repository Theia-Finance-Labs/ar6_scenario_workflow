"""
Ultra-fast vectorized gap-filling - completely rewritten for maximum speed
"""

import pandas as pd
import numpy as np
import gc
from typing import Dict

from fuel_price_validation import (
    FOSSIL_FUELS,
    assert_primary_fuel_price_provenance,
    canonical_primary_variable,
    classify_scenario_type,
    normalize_fuel,
    validate_fuel_price_quality,
)


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


def _combine_unique(values: pd.Series) -> str:
    parts = set()
    for value in values.dropna():
        parts.update(part.strip() for part in str(value).split(";") if part.strip())
    return ";".join(sorted(parts))


def _seed_fuel_price_provenance(df: pd.DataFrame) -> pd.DataFrame:
    """Initialize fixed lineage fields without inventing direct-source provenance."""
    df = df.copy()
    if "fuel_price_source_variable" not in df.columns:
        df["fuel_price_source_variable"] = pd.Series(pd.NA, index=df.index, dtype="object")
    else:
        df["fuel_price_source_variable"] = df["fuel_price_source_variable"].astype("object")
    if "fuel_price_source_unit" not in df.columns:
        df["fuel_price_source_unit"] = pd.Series(pd.NA, index=df.index, dtype="object")
    else:
        df["fuel_price_source_unit"] = df["fuel_price_source_unit"].astype("object")
    df["fuel_price_carbon_adjusted"] = False
    df["fuel_price_carbon_coefficient"] = 0.0
    if "fuel_price_fallback" not in df.columns:
        df["fuel_price_fallback"] = "none"
    else:
        df["fuel_price_fallback"] = df["fuel_price_fallback"].fillna("none")

    return df


def _baseline_first_fill(df: pd.DataFrame, df_col: str) -> pd.DataFrame:
    """Fill target price gaps from direct C7/C8 rows of the same provider."""
    if df_col == "fuel_price":
        normalized_fuel_key = "_baseline_normalized_fuel"
        df[normalized_fuel_key] = df["fuel_for_price"].map(normalize_fuel)
        keys = [
            "scenario_provider",
            "scenario_geography",
            "scenario_year",
            normalized_fuel_key,
        ]
    else:
        normalized_fuel_key = None
        keys = [
            "scenario_provider",
            "scenario_geography",
            "scenario_year",
            "sector",
            "technology",
        ]
    if any(key not in df.columns for key in keys):
        return df

    donor_mask = df["scenario_type"].eq("baseline") & df[df_col].notna()
    if df_col == "fuel_price":
        donor_mask &= df["fuel_price_fallback"].eq("none")
        donors = (
            df.loc[
                donor_mask,
                keys
                + [
                    df_col,
                    "fuel_price_source_variable",
                    "fuel_price_source_unit",
                ],
            ]
            .groupby(keys, dropna=False, as_index=False)
            .agg(
                baseline_value=(df_col, "median"),
                baseline_source_variable=("fuel_price_source_variable", _combine_unique),
                baseline_source_unit=("fuel_price_source_unit", _combine_unique),
            )
        )
    else:
        donors = (
            df.loc[donor_mask, keys + [df_col]]
            .groupby(keys, dropna=False, as_index=False)[df_col]
            .median()
            .rename(columns={df_col: "baseline_value"})
        )
    if donors.empty:
        if normalized_fuel_key:
            df = df.drop(columns=[normalized_fuel_key])
        return df

    missing = df["scenario_type"].eq("target") & df[df_col].isna()
    candidates = df.loc[missing, keys].copy()
    candidates["_row_index"] = candidates.index
    candidates = candidates.merge(donors, on=keys, how="left").set_index("_row_index")
    filled = candidates["baseline_value"].dropna()
    if filled.empty:
        if normalized_fuel_key:
            df = df.drop(columns=[normalized_fuel_key])
        return df

    df.loc[filled.index, df_col] = filled
    df.loc[filled.index, "gap_filled_columns"] = update_tracking(
        df.loc[filled.index, "gap_filled_columns"], f"{df_col}(baseline)"
    )
    if df_col == "fuel_price":
        df.loc[filled.index, "fuel_price_source_variable"] = candidates.loc[
            filled.index, "baseline_source_variable"
        ]
        df.loc[filled.index, "fuel_price_source_unit"] = candidates.loc[
            filled.index, "baseline_source_unit"
        ]
        df.loc[filled.index, "fuel_price_fallback"] = "baseline"
    if normalized_fuel_key:
        df = df.drop(columns=[normalized_fuel_key])
    print(f"    ✅ Filled {len(filled):,} {df_col} values from provider baselines")
    return df


def _set_lookup_fuel_provenance(
    df: pd.DataFrame, indices: pd.Index, fallback: str
) -> None:
    if len(indices) == 0:
        return
    if fallback == "lookup-cross-fossil":
        variables = "Price|Primary Energy|Coal;Price|Primary Energy|Gas"
        df.loc[indices, "fuel_price_source_variable"] = variables
    else:
        df.loc[indices, "fuel_price_source_variable"] = df.loc[
            indices, "fuel_for_price"
        ].map(canonical_primary_variable)
    df.loc[indices, "fuel_price_source_unit"] = "US$2010/GJ"
    df.loc[indices, "fuel_price_carbon_adjusted"] = False
    df.loc[indices, "fuel_price_carbon_coefficient"] = 0.0
    df.loc[indices, "fuel_price_fallback"] = fallback


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
    else:
        df["gap_filled_columns"] = df["gap_filled_columns"].fillna("")
    df = _seed_fuel_price_provenance(df)

    # Step 3 can carry a fossil fuel_price with no recorded primary-energy
    # source variable/unit (e.g. TIAM-ECN 1.1's Global Coal price). Treat
    # those as missing so the fallback cascade below replaces them with a
    # value that has valid, trackable lineage instead of tripping
    # assert_primary_fuel_price_provenance downstream.
    if "fuel_for_price" in df.columns:
        fossil_mask = df["fuel_for_price"].map(normalize_fuel).isin(FOSSIL_FUELS)
        untracked = (
            fossil_mask
            & df["fuel_price"].notna()
            & df["fuel_price_source_variable"].isna()
        )
        if untracked.any():
            print(
                f"    ⚠️ Clearing {int(untracked.sum())} fossil fuel_price values "
                "with no recorded source lineage for re-fill"
            )
            df.loc[untracked, "fuel_price"] = np.nan
            df.loc[untracked, "fuel_price_fallback"] = "none"

    # Price gaps use direct observations from the same provider's C7/C8
    # scenarios before any cross-provider lookup strategy is considered.
    for price_col in ["fuel_price", "scenario_price"]:
        if price_col in df.columns:
            df = _baseline_first_fill(df, price_col)
    
    filled_counts = {col: 0 for col in gap_fillable_columns}
    
    # Technology fallback mapping
    tech_fallbacks = {
        "CoalCap": "CoalCap - w/o CCS",
        "BiomassCap": "BiomassCap - w/o CCS", 
        "GasCap": "GasCap - w/o CCS",
        "OilCap": "OilCap - w/o CCS",
        # CCS variants taking non-ccs costs in case of missing data
        "CoalCap - w/ CCS": "CoalCap - w/o CCS",
        "BiomassCap - w/ CCS": "BiomassCap - w/o CCS",
        "GasCap - w/ CCS": "GasCap - w/o CCS",
        "OilCap - w/ CCS": "OilCap - w/o CCS"
    }
    
    # Process each column independently for maximum speed
    for df_col in gap_fillable_columns:
        lookup_col = column_mapping[df_col]
        if lookup_col not in lookup_df.columns:
            continue

        if df_col not in df.columns:
            # AR6 source data doesn't report every metric (e.g. capital/O&M
            # cost aren't AR6 variables at all); create the column so it can
            # be populated entirely from the technology lookup table instead
            # of silently staying absent from every downstream output.
            print(f"    ℹ️ {df_col} missing from input; sourcing entirely from lookup table")
            df[df_col] = np.nan
        
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
            if df_col == "fuel_price":
                _set_lookup_fuel_provenance(
                    df, direct_filled.index, "lookup-direct"
                )
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
                        if df_col == "fuel_price":
                            _set_lookup_fuel_provenance(
                                df, fallback_filled.index, "lookup-tech"
                            )
                        df.loc[fallback_filled.index, "gap_filled_columns"] = update_tracking(
                            df.loc[fallback_filled.index, "gap_filled_columns"], f"{df_col}(tech_fallback)"
                        )
                        col_filled += len(fallback_filled)
                        filled_counts[df_col] += len(fallback_filled)
                        print(f"      ✅ {base_tech}->{fallback_tech}: {len(fallback_filled)} values")
        
        # Update missing mask after technology fallbacks
        missing_mask = df[df_col].isna()
        missing_df = df[missing_mask].copy()
        
        # STRATEGY 2b: Cross-fossil fallbacks (Oil -> average of Coal+Gas within same CCS category)
        if len(missing_df) > 0:
            print(f"    ⛽ Strategy 2b: Cross-fossil fallbacks (Oil -> Coal+Gas average)...")
            
            oil_techs_missing = missing_df[missing_df['technology'].str.contains('Oil', na=False)]
            if len(oil_techs_missing) > 0:
                cross_fossil_filled = cross_fossil_fallback_vectorized(oil_techs_missing, lookup_df, df_col, lookup_col)
                if len(cross_fossil_filled) > 0:
                    df.loc[cross_fossil_filled.index, df_col] = cross_fossil_filled
                    if df_col == "fuel_price":
                        _set_lookup_fuel_provenance(
                            df,
                            cross_fossil_filled.index,
                            "lookup-cross-fossil",
                        )
                    df.loc[cross_fossil_filled.index, "gap_filled_columns"] = update_tracking(
                        df.loc[cross_fossil_filled.index, "gap_filled_columns"], f"{df_col}(oil_coal_gas_avg)"
                    )
                    col_filled += len(cross_fossil_filled)
                    filled_counts[df_col] += len(cross_fossil_filled)
                    print(f"      ✅ Oil->Coal+Gas average: {len(cross_fossil_filled)} values")

        # Update missing mask after cross-fossil fallbacks
        missing_mask = df[df_col].isna()
        missing_df = df[missing_mask].copy()
        
        # STRATEGY 3: Global fallback
        if len(missing_df) > 0:
            print(f"    🌍 Strategy 3: Global fallbacks...")
            global_filled = global_fallback_match(missing_df, lookup_df, df_col, lookup_col)
            if len(global_filled) > 0:
                df.loc[global_filled.index, df_col] = global_filled
                if df_col == "fuel_price":
                    _set_lookup_fuel_provenance(
                        df, global_filled.index, "lookup-global"
                    )
                df.loc[global_filled.index, "gap_filled_columns"] = update_tracking(
                    df.loc[global_filled.index, "gap_filled_columns"], f"{df_col}(global)"
                )
                col_filled += len(global_filled)
                filled_counts[df_col] += len(global_filled)
                print(f"      ✅ Global fallback: {len(global_filled)} values")
        
        # Update missing mask after global fallbacks
        missing_mask = df[df_col].isna()
        missing_df = df[missing_mask].copy()
        
        # STRATEGY 4: Hardcoded fallbacks for systematic missing data
        if len(missing_df) > 0 and df_col == "lifetime_years":
            print(f"    🔧 Strategy 4: Hardcoded fallbacks for systematic missing data...")
            
            # OilCap lifetime: calculated mean from GasCap+CoalCap (65,620 data points)
            OILCAP_LIFETIME_FALLBACK = 36.6
            
            oilcap_variants = ["OilCap", "OilCap - w/ CCS", "OilCap - w/o CCS"]
            
            for oil_tech in oilcap_variants:
                oil_missing = missing_df[missing_df['technology'] == oil_tech]
                if len(oil_missing) > 0:
                    df.loc[oil_missing.index, df_col] = OILCAP_LIFETIME_FALLBACK
                    df.loc[oil_missing.index, "gap_filled_columns"] = update_tracking(
                        df.loc[oil_missing.index, "gap_filled_columns"], f"{df_col}(hardcoded_fallback)"
                    )
                    col_filled += len(oil_missing)
                    filled_counts[df_col] += len(oil_missing)
                    print(f"      ✅ {oil_tech} hardcoded fallback: {len(oil_missing)} values ({OILCAP_LIFETIME_FALLBACK} years)")
        
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
        
        # Group by year/stringency for efficiency
        results = []
        for (year, stringency), year_group in missing_df.groupby(['scenario_year', 'stringency']):
            matches = lookup_subset[
                (lookup_subset['year'] == year) &
                (lookup_subset['stringency'] == stringency)
            ]
            
            if len(matches) == 0:
                continue
            
            # Process each row in this year/stringency group
            for idx in year_group.index:
                row = year_group.loc[idx]
                
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
            
            # Group by year/stringency for efficiency
            for (year, stringency), year_group in tech_group.groupby(['scenario_year', 'stringency']):
                matches = tech_lookup[
                    (tech_lookup['year'] == year) &
                    (tech_lookup['stringency'] == stringency)
                ]
                
                if len(matches) == 0:
                    continue
                
                # Process each row in this year/stringency group
                for idx in year_group.index:
                    row = year_group.loc[idx]
                    
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


def cross_fossil_fallback_vectorized(oil_techs_missing: pd.DataFrame, lookup_df: pd.DataFrame,
                                   df_col: str, lookup_col: str) -> pd.Series:
    """Vectorized cross-fossil fallback: Oil -> average of Coal+Gas within same CCS category."""
    
    if len(oil_techs_missing) == 0:
        return pd.Series(dtype=float)
    
    # Technology mapping for Oil -> Coal+Gas fallbacks
    tech_mapping = {
        'OilCap - w/ CCS': ['CoalCap - w/ CCS', 'GasCap - w/ CCS'],
        'OilCap - w/o CCS': ['CoalCap - w/o CCS', 'GasCap - w/o CCS'],
        'OilCap': ['CoalCap', 'GasCap']
    }
    
    results = []
    
    # Process each oil technology type separately for vectorization
    for oil_tech, fallback_techs in tech_mapping.items():
        oil_subset = oil_techs_missing[oil_techs_missing['technology'] == oil_tech]
        if len(oil_subset) == 0:
            continue
        
        # Get all potential matches for Coal and Gas technologies at once
        fallback_lookup = lookup_df[
            (lookup_df['technology'].isin(fallback_techs)) &
            (lookup_df[lookup_col].notna())
        ]
        
        if len(fallback_lookup) == 0:
            continue
        
        # Group oil records by year/stringency for efficient processing
        for (year, stringency), year_group in oil_subset.groupby(['scenario_year', 'stringency']):
            # Get matching Coal+Gas entries for this year/stringency
            year_stringency_matches = fallback_lookup[
                (fallback_lookup['year'] == year) &
                (fallback_lookup['stringency'] == stringency)
            ]
            
            if len(year_stringency_matches) == 0:
                continue
            
            # Process each oil record in this year/stringency group
            for idx in year_group.index:
                row = year_group.loc[idx]
                
                # Handle country matching
                country_list = str(row.get('country_iso2_list', '')).strip()
                if not country_list or country_list == 'nan':
                    target_countries = ['GLOBAL']
                else:
                    target_countries = [c.strip() for c in country_list.split(',') if c.strip()]
                
                # Collect values from Coal and Gas for target countries
                values_to_average = []
                for country in target_countries:
                    country_matches = year_stringency_matches[year_stringency_matches['iso2'] == country]
                    if len(country_matches) > 0:
                        values_to_average.extend(country_matches[lookup_col].tolist())
                
                # If no country match, try GLOBAL
                if not values_to_average:
                    global_matches = year_stringency_matches[year_stringency_matches['iso2'] == 'GLOBAL']
                    if len(global_matches) > 0:
                        values_to_average.extend(global_matches[lookup_col].tolist())
                
                # Calculate average if we have values from Coal/Gas
                if values_to_average:
                    avg_value = np.mean(values_to_average)
                    results.append((idx, avg_value))
    
    if results:
        indices, values = zip(*results)
        return pd.Series(values, index=indices)
    else:
        return pd.Series(dtype=float)


def global_fallback_match(missing_df: pd.DataFrame, lookup_df: pd.DataFrame, 
                         df_col: str, lookup_col: str) -> pd.Series:
    """Global fallback matching using GLOBAL iso2 entries and UNKNOWN stringency fallbacks."""
    
    results = []
    
    # Group by technology for efficient processing
    for tech, tech_group in missing_df.groupby('technology'):
        tech_lookup = lookup_df[
            (lookup_df['technology'] == tech) &
            (lookup_df[lookup_col].notna())
        ]
        
        if len(tech_lookup) == 0:
            continue
        
        # Group by year/stringency for efficiency
        for (year, stringency), year_group in tech_group.groupby(['scenario_year', 'stringency']):
            
            # Strategy 3a: Try GLOBAL geography with same stringency
            global_matches = tech_lookup[
                (tech_lookup['year'] == year) &
                (tech_lookup['stringency'] == stringency) &
                (tech_lookup['iso2'] == 'GLOBAL')
            ]
            
            if len(global_matches) > 0:
                global_val = global_matches[lookup_col].iloc[0]
                # Apply to all rows in this year/stringency group
                for idx in year_group.index:
                    results.append((idx, global_val))
                continue
            
            # Strategy 3b: Try UNKNOWN stringency for each row individually
            for idx in year_group.index:
                row = year_group.loc[idx]
                
                country_list = str(row.get('country_iso2_list', '')).strip()
                if not country_list or country_list == 'nan':
                    target_countries = ['GLOBAL']
                else:
                    target_countries = [c.strip() for c in country_list.split(',') if c.strip()] + ['GLOBAL']
                
                for country in target_countries:
                    unknown_matches = tech_lookup[
                        (tech_lookup['year'] == year) &
                        (tech_lookup['stringency'] == 'UNKNOWN') &
                        (tech_lookup['iso2'] == country)
                    ]
                    
                    if len(unknown_matches) > 0:
                        unknown_val = unknown_matches[lookup_col].iloc[0]
                        results.append((idx, unknown_val))
                        break
    
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
    
    print("📁 Loading input data...")
    df = pd.read_csv("3_final_AR6_target_schema.csv")
    print(f"✅ Loaded {len(df):,} rows")
    
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
    
    # Preserve the provider-wide C7/C8 classification established before
    # gap filling; never replace it with provider-specific hard-coded names.
    if "stringency" in df.columns:
        df["scenario_type"] = classify_scenario_type(df["stringency"])

    assert_primary_fuel_price_provenance(df)
    df, validation_report = validate_fuel_price_quality(df)
    validation_report.to_csv("fuel_price_validation_report.csv", index=False)
    print(
        "✅ Wrote fuel_price_validation_report.csv | "
        f"Groups: {len(validation_report):,}"
    )
    
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
        "stringency", "gap_filled_columns",
        "fuel_price_source_variable", "fuel_price_source_unit",
        "fuel_price_carbon_adjusted", "fuel_price_carbon_coefficient",
        "fuel_price_fallback", "fuel_price_quality_pass"
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
