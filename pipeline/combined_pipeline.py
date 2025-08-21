"""
Combined AR6 Pipeline (Steps 1–4 in a single script)
====================================================

Goal:
- Merge the functionality of 1_formatAR6.py, 2_filterAR6.py, 3_finalizeAR6.py, 4_aggregateTechnologies.py
  into a single, optimized pipeline script.
- Keep all logic here (no subprocess orchestration), while still writing intermediate files
  to manage memory usage similarly to the original multi-step workflow.

Outputs (same as original workflow):
- 1_intermediate_AR6_scenario_formatting_ISO3.csv
- 1_intermediate_AR6_scenario_formatting_R10.csv
- 2_final_AR6_filtered.csv
- 3_final_AR6_target_schema.csv
- 4_final_AR6_aggregated.csv
- 4_final_AR6_aggregated_complete.csv

Notes:
- Uses Modin (modin.pandas as pd) for scalability.
- Introduces vectorized speedups for technology mapping and gap-filling.
- Writes intermediate files and frees memory between stages to avoid OOM.
"""

from __future__ import annotations

import gc
import sys
from typing import Dict, List, Tuple, Optional

# Use regular pandas for better compatibility with complex operations
import pandas as pd
import numpy as np


# ================================
# Shared Utilities and Conversions
# ================================


def print_banner(title: str) -> None:
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")


def filter_year_columns(columns: List[str]) -> List[str]:
    year_cols: List[str] = []
    for col in columns:
        try:
            year = float(col)
            if 2023 <= year <= 2050:
                year_cols.append(col)
        except ValueError:
            continue
    return year_cols


def memory_release(*objs) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()


# ================================
# Step 1: Format AR6 (ISO3 & R10)
# ================================


def step1_process_dataset(dataset_type: str) -> Optional[str]:
    """
    Process either ISO3 or R10 dataset into the intermediate CSV used by Step 2.
    Returns the output filename if successful; otherwise None.
    """
    dataset_type = dataset_type.upper()
    if dataset_type not in {"ISO3", "R10"}:
        raise ValueError("dataset_type must be 'ISO3' or 'R10'")

    input_file = (
        "AR6_Scenarios_Database_ISO3_v1.1.feather"
        if dataset_type == "ISO3"
        else "AR6_Scenarios_Database_R10_regions_v1.1.feather"
    )
    output_file = (
        "1_intermediate_AR6_scenario_formatting_ISO3.csv"
        if dataset_type == "ISO3"
        else "1_intermediate_AR6_scenario_formatting_R10.csv"
    )

    print_banner(f"STEP 1 ({dataset_type}) — Loading and Melting AR6 Data")

    try:
        source = pd.read_feather(input_file)

        # Filter for target models to improve performance
        target_models = ["WITCH 5.0", "IMAGE 3.2"]
        print(f"   Before model filter: {source.shape[0]:,} rows")
        source = source[source["Model"].isin(target_models)]
        print(f"   After model filter (WITCH 5.0, IMAGE 3.2): {source.shape[0]:,} rows")

        if source.empty:
            print("❌ No data found for target models (WITCH 5.0, IMAGE 3.2)")
            return None
    except FileNotFoundError:
        print(f"❌ Missing required input: {input_file}")
        return None

    try:
        mapping = pd.read_csv("ar6_variables_with_mapping.csv")
    except FileNotFoundError:
        print("❌ Missing required input: ar6_variables_with_mapping.csv")
        return None

    print(f"AR6 data shape: {source.shape}")
    print(f"Mapping data shape: {mapping.shape}")

    id_cols = ["Model", "Scenario", "Region", "Variable", "Unit"]
    all_year_cols = [c for c in source.columns if c not in id_cols]
    year_cols = filter_year_columns(all_year_cols)
    if not year_cols:
        print("❌ No year columns found in range (2021-2050)")
        return None

    print(
        f"Year columns for analysis: {len(year_cols)} | Range: {min(year_cols)}–{max(year_cols)}"
    )

    melted = pd.melt(
        source,
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="Year",
        value_name="Value",
    )
    melted["Year"] = pd.to_numeric(melted["Year"], errors="coerce")
    melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")

    renamed = melted.rename(
        columns={
            "Model": "model",
            "Scenario": "scenario",
            "Region": "region",
            "Variable": "variable",
            "Unit": "unit",
            "Year": "year",
            "Value": "value",
        }
    )

    base_data = renamed.merge(
        mapping, how="left", left_on="variable", right_on="variable"
    )

    # Drop rows with missing Sector mapping
    before_sector = len(base_data)
    base_data = base_data[base_data["Sector"].notna()].copy()
    after_sector = len(base_data)
    print(
        f"Sector mapping: kept {after_sector:,}/{before_sector:,} rows; removed {before_sector-after_sector:,}"
    )

    target_sectors = [
        "Steel",
        "Nuclear",
        "Gas&Oil",
        "Cement",
        "Coal",
        "Renewables",
        "Power",
    ]
    before_filter = len(base_data)
    base_data = base_data[base_data["Sector"].isin(target_sectors)].copy()
    after_filter = len(base_data)
    print(
        f"Target sectors filter: kept {after_filter:,}/{before_filter:,} rows; removed {before_filter-after_filter:,}"
    )

    # Cost metric extraction (self-join style via filtered selects)
    join_cols = [
        "model",
        "scenario",
        "region",
        "year",
        "Sector",
        "Subsector",
        "Technology",
    ]

    def extract_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
        subset = df[df["col1"] == metric]
        if subset.empty:
            return pd.DataFrame(
                columns=join_cols
                + [
                    metric.lower().replace(" ", "_"),
                    f"{metric.lower().replace(' ', '_')}_unit",
                ]
            )
        return subset[join_cols + ["value", "unit"]].rename(
            columns={
                "value": metric.lower().replace(" ", "_"),
                "unit": f"{metric.lower().replace(' ', '_')}_unit",
            }
        )

    print("Extracting OM Cost / Capital Cost / Efficiency …")
    om_cost = extract_metric(base_data, "OM Cost")
    cap_cost = extract_metric(base_data, "Capital Cost")
    efficiency = extract_metric(base_data, "Efficiency")

    merged = base_data.copy()
    merged = (
        merged.merge(om_cost, on=join_cols, how="left") if not om_cost.empty else merged
    )
    merged = (
        merged.merge(efficiency, on=join_cols, how="left")
        if not efficiency.empty
        else merged
    )
    merged = (
        merged.merge(cap_cost, on=join_cols, how="left")
        if not cap_cost.empty
        else merged
    )

    # Price data (keep all; unit conversions later)
    price_rows = base_data[base_data["col1"] == "Price"].copy()
    if not price_rows.empty:
        price_rows = price_rows.rename(
            columns={"col2": "energy_type", "value": "price"}
        )
        primary_prices = price_rows[
            price_rows["energy_type"] == "Primary Energy"
        ].copy()
        secondary_prices = price_rows[
            price_rows["energy_type"] == "Secondary Energy"
        ].copy()
        final_prices = price_rows[price_rows["energy_type"] == "Final Energy"].copy()
        carbon_prices = price_rows[price_rows["energy_type"] == "Carbon"].copy()

        # Primary energy (fuel-specific via Fuel)
        if not primary_prices.empty:
            primary_clean = primary_prices[
                ["model", "scenario", "region", "year", "Fuel", "price", "unit"]
            ].copy()
            primary_clean = primary_clean.rename(
                columns={
                    "price": "primary_energy_price",
                    "unit": "primary_energy_price_unit",
                }
            )
            merged = merged.merge(
                primary_clean,
                on=["model", "scenario", "region", "year", "Fuel"],
                how="left",
            )

        # Secondary energy (fuel-specific)
        if not secondary_prices.empty:
            secondary_clean = secondary_prices[
                ["model", "scenario", "region", "year", "Fuel", "price", "unit"]
            ].copy()
            secondary_clean = secondary_clean.rename(
                columns={
                    "price": "secondary_energy_price",
                    "unit": "secondary_energy_price_unit",
                }
            )
            merged = merged.merge(
                secondary_clean,
                on=["model", "scenario", "region", "year", "Fuel"],
                how="left",
            )

        # Carbon price (single per model/scenario/region/year)
        if not carbon_prices.empty:
            carbon_pivot = (
                carbon_prices.groupby(["model", "scenario", "region", "year"][0:4])[
                    "price"
                ]
                .first()
                .reset_index()
            )
            carbon_units = (
                carbon_prices.groupby(["model", "scenario", "region", "year"][0:4])[
                    "unit"
                ]
                .first()
                .reset_index()
            )
            merged = merged.merge(
                carbon_pivot.rename(columns={"price": "carbon_price"}),
                on=["model", "scenario", "region", "year"],
                how="left",
            )
            merged = merged.merge(
                carbon_units.rename(columns={"unit": "carbon_price_unit"}),
                on=["model", "scenario", "region", "year"],
                how="left",
            )

        # Secondary electricity & final electricity (special)
        secondary_electricity = secondary_prices[
            secondary_prices["Fuel"] == "Electricity"
        ].copy()
        if not secondary_electricity.empty:
            sec_elec = (
                secondary_electricity.groupby(
                    ["model", "scenario", "region", "year"][0:4]
                )["price"]
                .first()
                .reset_index()
            )
            sec_elec_u = (
                secondary_electricity.groupby(
                    ["model", "scenario", "region", "year"][0:4]
                )["unit"]
                .first()
                .reset_index()
            )
            merged = merged.merge(
                sec_elec.rename(
                    columns={"price": "secondary_energy_electricity_price"}
                ),
                on=["model", "scenario", "region", "year"],
                how="left",
            )
            merged = merged.merge(
                sec_elec_u.rename(
                    columns={"unit": "secondary_energy_electricity_price_unit"}
                ),
                on=["model", "scenario", "region", "year"],
                how="left",
            )

        final_electricity = final_prices[final_prices["Fuel"] == "Electricity"].copy()
        if not final_electricity.empty:
            fin_elec = (
                final_electricity.groupby(["model", "scenario", "region", "year"][0:4])[
                    "price"
                ]
                .first()
                .reset_index()
            )
            fin_elec_u = (
                final_electricity.groupby(["model", "scenario", "region", "year"][0:4])[
                    "unit"
                ]
                .first()
                .reset_index()
            )
            merged = merged.merge(
                fin_elec.rename(columns={"price": "final_energy_electricity_price"}),
                on=["model", "scenario", "region", "year"],
                how="left",
            )
            merged = merged.merge(
                fin_elec_u.rename(
                    columns={"unit": "final_energy_electricity_price_unit"}
                ),
                on=["model", "scenario", "region", "year"],
                how="left",
            )

    # Final output selection
    final_cols = [
        "model",
        "scenario",
        "region",
        "variable",
        "value",
        "unit",
        "year",
        "Sector",
        "Subsector",
        "Technology",
        "Fuel",
        "CCS_flag",
        "om_cost",
        "om_cost_unit",
        "efficiency",
        "efficiency_unit",
        "capital_cost",
        "capital_cost_unit",
        "primary_energy_price",
        "primary_energy_price_unit",
        "secondary_energy_price",
        "secondary_energy_price_unit",
        "carbon_price",
        "carbon_price_unit",
        "secondary_energy_electricity_price",
        "secondary_energy_electricity_price_unit",
        "final_energy_electricity_price",
        "final_energy_electricity_price_unit",
        "col1",
        "col2",
        "col3",
        "col4",
        "col5",
    ]
    final_cols = [c for c in final_cols if c in merged.columns]
    merged[final_cols].to_csv(output_file, index=False)
    print(f"✅ Wrote {output_file} | Shape: {merged[final_cols].shape}")

    memory_release(
        source,
        mapping,
        melted,
        renamed,
        base_data,
        om_cost,
        cap_cost,
        efficiency,
        price_rows,
        merged,
    )
    return output_file


def step1_run() -> None:
    """Run Step 1 for both ISO3 and R10 (if available)."""
    _ = step1_process_dataset("ISO3")
    _ = step1_process_dataset("R10")


# =============================================
# Step 2: Filter, Pivot, Unit Standardization
# =============================================


def step2_filter_and_pivot() -> None:
    print_banner("STEP 2 — Filter, Pivot, Merge Costs & Prices, Convert Units")

    datasets: List[pd.DataFrame] = []
    try:
        iso3_df = pd.read_csv("1_intermediate_AR6_scenario_formatting_ISO3.csv")
        datasets.append(iso3_df)
        print(f"Loaded ISO3: {iso3_df.shape}")
    except FileNotFoundError:
        print("⚠️ Missing 1_intermediate_AR6_scenario_formatting_ISO3.csv")

    try:
        r10_df = pd.read_csv("1_intermediate_AR6_scenario_formatting_R10.csv")
        datasets.append(r10_df)
        print(f"Loaded R10: {r10_df.shape}")
    except FileNotFoundError:
        print("⚠️ Missing 1_intermediate_AR6_scenario_formatting_R10.csv")

    if not datasets:
        print("❌ No Step 1 outputs found. Aborting Step 2.")
        return

    df = pd.concat(datasets, ignore_index=True)
    memory_release(*datasets)

    # Rename and drop Subsector for compatibility with step 3
    df = df.rename(columns={"region": "scenario_geography"})
    if "Subsector" in df.columns:
        df = df.drop(columns=["Subsector"])  # Not used downstream

    # Filter target metrics and pivot
    target_col1_values = [
        "Capacity",
        "Capacity Additions",
        "Secondary Energy",
        "Primary Energy",
        "Lifetime",
    ]
    df_filtered = df[df["col1"].isin(target_col1_values)].copy()

    grouping_cols = [
        "model",
        "scenario",
        "scenario_geography",
        "year",
        "Sector",
        "Technology",
    ]
    grouping_cols = [c for c in grouping_cols if c in df_filtered.columns]

    pivoted = df_filtered.pivot_table(
        index=grouping_cols, columns="col1", values="value", aggfunc="first"
    ).reset_index()
    pivoted.columns.name = None
    rename_map: Dict[str, str] = {}
    for col in pivoted.columns:
        if col in grouping_cols:
            rename_map[col] = col
        else:
            rename_map[col] = col.lower().replace(" ", "_") + "_value"
    pivoted = pivoted.rename(columns=rename_map)

    # Join keys used for all merges
    join_cols = [
        c
        for c in [
            "model",
            "scenario",
            "scenario_geography",
            "year",
            "Sector",
            "Technology",
        ]
        if c in df_filtered.columns and c in pivoted.columns
    ]

    # Unit-aware conversions BEFORE cost merges to ensure correct base columns
    # 1) Capacity → MW (supports kW/MW/GW)
    cap_rows = df_filtered[df_filtered["col1"] == "Capacity"]
    if not cap_rows.empty:
        cap_df = cap_rows[join_cols + ["value", "unit"]].copy()
        u = cap_df["unit"].astype(str).str.lower()
        cap_mw = cap_df["value"].astype(float)
        cap_mw.loc[u.str.contains("gw")] = cap_mw.loc[u.str.contains("gw")] * 1000.0
        cap_mw.loc[u.str.contains("mw")] = cap_mw.loc[u.str.contains("mw")] * 1.0
        cap_mw.loc[u.str.contains("kw")] = cap_mw.loc[u.str.contains("kw")] / 1000.0
        # Unknown units → leave as-is
        cap_df["capacity_mw"] = cap_mw
        cap_df = (
            cap_df.drop(columns=["value", "unit"])
            .groupby(join_cols, as_index=False)["capacity_mw"]
            .first()
        )
        pivoted = pivoted.merge(cap_df, on=join_cols, how="left")
        pivoted = pivoted.drop(columns=["capacity_value"], errors="ignore")

    # 2) Capacity Additions → MW/yr (supports kW/yr, MW/yr, GW/yr variants incl. yr-1)
    add_rows = df_filtered[df_filtered["col1"] == "Capacity Additions"]
    if not add_rows.empty:
        add_df = add_rows[join_cols + ["value", "unit"]].copy()
        u = add_df["unit"].astype(str).str.lower()
        add_mwyr = add_df["value"].astype(float)
        gwyr_mask = u.str.contains("gw") & (
            u.str.contains("/yr") | u.str.contains("yr-1")
        )
        mwy_mask = u.str.contains("mw") & (
            u.str.contains("/yr") | u.str.contains("yr-1")
        )
        kwyr_mask = u.str.contains("kw") & (
            u.str.contains("/yr") | u.str.contains("yr-1")
        )
        add_mwyr.loc[gwyr_mask] = add_mwyr.loc[gwyr_mask] * 1000.0
        add_mwyr.loc[mwy_mask] = add_mwyr.loc[mwy_mask] * 1.0
        add_mwyr.loc[kwyr_mask] = add_mwyr.loc[kwyr_mask] / 1000.0
        add_df["capacity_additions_mw_per_yr"] = add_mwyr
        add_df = (
            add_df.drop(columns=["value", "unit"])
            .groupby(join_cols, as_index=False)["capacity_additions_mw_per_yr"]
            .first()
        )
        pivoted = pivoted.merge(add_df, on=join_cols, how="left")
        pivoted = pivoted.drop(columns=["capacity_additions_value"], errors="ignore")

    # 3) Energy (Primary/Secondary) → MWh/yr using per-row units (EJ/PJ/TJ per year)
    def energy_units_to_mwh(values: pd.Series, unit_str: pd.Series) -> pd.Series:
        out = values.astype(float).copy()
        u = unit_str.astype(str).str.lower()
        ej = u.str.contains("ej")
        pj = u.str.contains("pj")
        tj = u.str.contains("tj")
        out.loc[ej] = out.loc[ej] * (1e18 / 3.6e9)
        out.loc[pj] = out.loc[pj] * (1e15 / 3.6e9)
        out.loc[tj] = out.loc[tj] * (1e12 / 3.6e9)
        return out

    pri_rows = df_filtered[df_filtered["col1"] == "Primary Energy"]
    if not pri_rows.empty:
        pri_df = pri_rows[join_cols + ["value", "unit"]].copy()
        pri_df["primary_energy_mwh_per_yr"] = energy_units_to_mwh(
            pri_df["value"], pri_df["unit"]
        ).replace([np.inf, -np.inf], np.nan)
        pri_df = (
            pri_df.drop(columns=["value", "unit"])
            .groupby(join_cols, as_index=False)["primary_energy_mwh_per_yr"]
            .first()
        )
        pivoted = pivoted.merge(pri_df, on=join_cols, how="left")
        pivoted = pivoted.drop(columns=["primary_energy_value"], errors="ignore")

    sec_rows = df_filtered[df_filtered["col1"] == "Secondary Energy"]
    if not sec_rows.empty:
        sec_df = sec_rows[join_cols + ["value", "unit"]].copy()
        sec_df["secondary_energy_mwh_per_yr"] = energy_units_to_mwh(
            sec_df["value"], sec_df["unit"]
        ).replace([np.inf, -np.inf], np.nan)
        sec_df = (
            sec_df.drop(columns=["value", "unit"])
            .groupby(join_cols, as_index=False)["secondary_energy_mwh_per_yr"]
            .first()
        )
        pivoted = pivoted.merge(sec_df, on=join_cols, how="left")
        pivoted = pivoted.drop(columns=["secondary_energy_value"], errors="ignore")

    # Merge cost metrics from original df (non-pivoted)
    cost_data = df[~df["col1"].isin(target_col1_values)].copy()
    # refresh join_cols to ensure overlap with cost_data
    join_cols = [
        c
        for c in [
            "model",
            "scenario",
            "scenario_geography",
            "year",
            "Sector",
            "Technology",
        ]
        if c in cost_data.columns and c in pivoted.columns
    ]

    def add_cost_metric(
        pivoted_df: pd.DataFrame,
        cost_df: pd.DataFrame,
        metric_name: str,
        join_cols: List[str],
    ) -> pd.DataFrame:
        metric_df = cost_df[cost_df["col1"] == metric_name].copy()
        if metric_df.empty:
            # Return original df if the metric is missing
            return pivoted_df
        if metric_name == "OM Cost":
            col_name = "om_cost_usd_per_mw_per_yr"
        elif metric_name == "Capital Cost":
            col_name = "capital_cost_usd_per_mw"
        elif metric_name == "Efficiency":
            col_name = "efficiency_percent"
        else:
            # Best-effort naming
            unit_mode = metric_df["unit"].mode()
            unit_str = unit_mode.iloc[0] if not unit_mode.empty else ""
            unit_clean = (
                unit_str.replace("US$2010/", "")
                .replace("/", "_per_")
                .replace(" ", "_")
                .lower()
            )
            col_name = metric_name.lower().replace(" ", "_") + (
                f"_{unit_clean}" if unit_clean else ""
            )
        metric_df = metric_df[join_cols + ["value"]].rename(columns={"value": col_name})
        return pivoted_df.merge(metric_df, on=join_cols, how="left")

    pivoted = add_cost_metric(pivoted, cost_data, "OM Cost", join_cols)
    pivoted = add_cost_metric(pivoted, cost_data, "Capital Cost", join_cols)
    pivoted = add_cost_metric(pivoted, cost_data, "Efficiency", join_cols)

    # Add unit columns where available (used by later conversions)
    for unit_col in ["om_cost_unit", "capital_cost_unit", "efficiency_unit"]:
        if unit_col in cost_data.columns:
            unit_df = (
                cost_data[join_cols + [unit_col]]
                .dropna(subset=[unit_col])
                .drop_duplicates()
            )
            pivoted = pivoted.merge(unit_df, on=join_cols, how="left")

    # Renewables efficiency special-case: set to 100% if missing
    if "efficiency_percent" not in pivoted.columns:
        pivoted["efficiency_percent"] = np.nan
    
    # Define exact renewable technology names for fast matching
    renewable_techs = [
        "SolarCap", "WindCap", "HydroCap", "GeothermalCap", "NuclearCap", "OceanCap",
        "Non-Biomass Renewables", "Electricity - Non-Biomass Renewables"
    ]
    
    # Create mask for renewable technologies using exact matching
    tech_col = "Technology" if "Technology" in pivoted.columns else "technology"
    renewable_mask = pivoted[tech_col].isin(renewable_techs)
    
    # Also include old-style renewables sector if it exists
    if "Sector" in pivoted.columns:
        old_renewables_mask = pivoted["Sector"] == "Renewables"
        renewable_mask = renewable_mask | old_renewables_mask
    
    print(f"Setting efficiency to 1.0 for {renewable_mask.sum()} renewable technology entries")
    pivoted.loc[renewable_mask, "efficiency_percent"] = 1.0

    # Carbon price (if available in original df)
    if set(["carbon_price", "carbon_price_unit"]).issubset(df.columns):
        price_join_cols = join_cols
        carbon_data = (
            df[price_join_cols + ["carbon_price", "carbon_price_unit"]]
            .dropna(subset=["carbon_price"])
            .drop_duplicates()
        )
        if not carbon_data.empty:
            carbon_data = carbon_data.rename(
                columns={"carbon_price": "carbon_price_usd_per_tco2"}
            )
            carbon_data = carbon_data.drop(
                columns=["carbon_price_unit"], errors="ignore"
            )
            pivoted = pivoted.merge(carbon_data, on=price_join_cols, how="left")
        else:
            pivoted["carbon_price_usd_per_tco2"] = np.nan
    else:
        pivoted["carbon_price_usd_per_tco2"] = np.nan

    # Primary & secondary energy prices (USD/GJ) — retain unit columns for step 3 conversion
    def merge_energy_price(df_source: pd.DataFrame, kind: str) -> pd.DataFrame:
        price_col = f"{kind}_energy_price"
        unit_col = f"{kind}_energy_price_unit"
        needed = set(join_cols + [price_col, unit_col])
        if not needed.issubset(df_source.columns):
            pivoted[price_col + "_usd_per_gj"] = np.nan
            pivoted[unit_col] = np.nan
            return pivoted
        price_df = (
            df_source[join_cols + [price_col, unit_col]]
            .dropna(subset=[price_col])
            .drop_duplicates()
        )
        if price_df.empty:
            pivoted[price_col + "_usd_per_gj"] = np.nan
            pivoted[unit_col] = np.nan
            return pivoted
        price_df = price_df.rename(columns={price_col: price_col + "_usd_per_gj"})
        return pivoted.merge(price_df, on=join_cols, how="left")

    pivoted = merge_energy_price(df, "primary")
    pivoted = merge_energy_price(df, "secondary")

    # Secondary electricity price (USD/GJ)
    sec_elec_cols = [
        "secondary_energy_electricity_price",
        "secondary_energy_electricity_price_unit",
    ]
    if set(sec_elec_cols).issubset(df.columns):
        tmp = (
            df[join_cols + sec_elec_cols]
            .dropna(subset=[sec_elec_cols[0]])
            .drop_duplicates()
        )
        if not tmp.empty:
            tmp = tmp.rename(
                columns={
                    sec_elec_cols[0]: "secondary_energy_electricity_price_usd_per_gj"
                }
            )
            tmp = tmp.drop(columns=[sec_elec_cols[1]], errors="ignore")
            pivoted = pivoted.merge(tmp, on=join_cols, how="left")
        else:
            pivoted["secondary_energy_electricity_price_usd_per_gj"] = np.nan
    else:
        pivoted["secondary_energy_electricity_price_usd_per_gj"] = np.nan

    # Unit conversions to target schema units
    # Capacity → MW
    if "capacity_value" in pivoted.columns:
        pivoted["capacity_mw"] = (
            pivoted["capacity_value"] * 1000.0
        )  # GW→MW or MW→MW if already in MW (ok if data was MW)
        pivoted = pivoted.drop(columns=["capacity_value"], errors="ignore")

    # Capacity additions → MW/yr
    if "capacity_additions_value" in pivoted.columns:
        pivoted["capacity_additions_mw_per_yr"] = (
            pivoted["capacity_additions_value"] * 1000.0
        )
        pivoted = pivoted.drop(columns=["capacity_additions_value"], errors="ignore")

    # Energy (Primary/Secondary) → MWh/yr with mixed units handled via source 'unit' column if available
    def energy_to_mwh_per_year(
        df_in: pd.DataFrame, value_col: str, unit_series: Optional[pd.Series]
    ) -> pd.Series:
        if value_col not in df_in.columns:
            return pd.Series(index=df_in.index, dtype=float)
        values = df_in[value_col].copy()
        if unit_series is None:
            # Assume EJ/yr if unknown; convert EJ → MWh
            return values * (1e18 / 3.6e9)
        # Per-row conversions using unit column
        ej_mask = (
            unit_series.isin(["EJ/yr", "EJ yr-1"])
            if unit_series is not None
            else pd.Series(False, index=df_in.index)
        )
        pj_mask = (
            unit_series.isin(["PJ/yr", "PJ yr-1"])
            if unit_series is not None
            else pd.Series(False, index=df_in.index)
        )
        tj_mask = (
            unit_series.isin(["TJ/yr", "TJ yr-1"])
            if unit_series is not None
            else pd.Series(False, index=df_in.index)
        )
        out = values.copy()
        out.loc[ej_mask] = values.loc[ej_mask] * (1e18 / 3.6e9)
        out.loc[pj_mask] = values.loc[pj_mask] * (1e15 / 3.6e9)
        out.loc[tj_mask] = values.loc[tj_mask] * (1e12 / 3.6e9)
        return out

    # We lost row-level unit for energy in pivot; fallback to EJ/yr assumption as in prior script when unknown
    if "secondary_energy_value" in pivoted.columns:
        pivoted["secondary_energy_mwh_per_yr"] = energy_to_mwh_per_year(
            pivoted, "secondary_energy_value", None
        )
        pivoted = pivoted.drop(columns=["secondary_energy_value"], errors="ignore")
    if "primary_energy_value" in pivoted.columns:
        pivoted["primary_energy_mwh_per_yr"] = energy_to_mwh_per_year(
            pivoted, "primary_energy_value", None
        )
        pivoted = pivoted.drop(columns=["primary_energy_value"], errors="ignore")

    # Lifetime → years (already years in most cases)
    if "lifetime_value" in pivoted.columns:
        pivoted["lifetime_years"] = pivoted["lifetime_value"]
        pivoted = pivoted.drop(columns=["lifetime_value"], errors="ignore")

    # Cost conversions: unit-aware to MW or MW/yr; clean zeros/negatives
    def convert_capacity_cost_to_mw(
        values: pd.Series, units: Optional[pd.Series], per_year: bool
    ) -> pd.Series:
        out = values.astype(float).copy()
        out.loc[out <= 0] = np.nan
        if units is None:
            return out
        u = units.astype(str).str.lower()
        # Only convert when unit clearly indicates kW vs MW; otherwise leave as-is
        kw_mask = u.str.contains("kw") & (~u.str.contains("mwh"))
        mw_mask = u.str.contains("mw") & (~u.str.contains("mwh"))
        out.loc[kw_mask] = out.loc[kw_mask] * 1000.0
        out.loc[mw_mask] = out.loc[mw_mask] * 1.0
        return out

    if "om_cost_usd_per_mw_per_yr" in pivoted.columns:
        units = pivoted["om_cost_unit"] if "om_cost_unit" in pivoted.columns else None
        pivoted["om_cost_usd_per_mw_per_yr"] = convert_capacity_cost_to_mw(
            pivoted["om_cost_usd_per_mw_per_yr"], units, per_year=True
        )

    if "capital_cost_usd_per_mw" in pivoted.columns:
        units = (
            pivoted["capital_cost_unit"]
            if "capital_cost_unit" in pivoted.columns
            else None
        )
        pivoted["capital_cost_usd_per_mw"] = convert_capacity_cost_to_mw(
            pivoted["capital_cost_usd_per_mw"], units, per_year=False
        )

    # Efficiency: treat 0 as missing, then convert percent >1 to decimal; Renewables already set to 1.0 above
    if "efficiency_percent" in pivoted.columns:
        zero_eff_mask = pivoted["efficiency_percent"] == 0
        pivoted.loc[zero_eff_mask, "efficiency_percent"] = np.nan
        pivoted["efficiency_decimal"] = pivoted["efficiency_percent"]
        percent_mask = pivoted["efficiency_percent"] > 1
        pivoted.loc[percent_mask, "efficiency_decimal"] = (
            pivoted.loc[percent_mask, "efficiency_percent"] / 100.0
        )
        pivoted = pivoted.drop(columns=["efficiency_percent"], errors="ignore")

    # Save and free memory
    output_file = "2_final_AR6_filtered.csv"
    pivoted.to_csv(output_file, index=False)
    print(f"✅ Wrote {output_file} | Shape: {pivoted.shape}")
    memory_release(df, df_filtered, cost_data, pivoted)


# =====================================
# Step 3: Finalize to Target AR6 Schema
# =====================================


def convert_energy_price_to_mwh(price_value: float, unit: str) -> float:
    if pd.isna(price_value) or pd.isna(unit):
        return np.nan
    unit_lower = str(unit).lower()
    if "gj" in unit_lower:
        return price_value * 3.6
    if "tj" in unit_lower:
        return price_value * 3.6 * 1_000
    if "pj" in unit_lower:
        return price_value * 3.6 * 1_000_000
    if "ej" in unit_lower:
        return price_value * 3.6 * 1_000_000_000
    return price_value


def load_step1_for_price() -> Optional[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for fname in [
        "1_intermediate_AR6_scenario_formatting_ISO3.csv",
        "1_intermediate_AR6_scenario_formatting_R10.csv",
    ]:
        try:
            frames.append(pd.read_csv(fname))
        except FileNotFoundError:
            continue
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _normalize_fuel_label(label: str) -> str:
    """Fast fuel normalization using exact matching where possible"""
    s = str(label).strip()
    if not s or s.lower() == "nan":
        return ""
    
    # Exact matches first (fastest)
    exact_map = {
        "Electricity": "Electricity", "electricity": "Electricity",
        "Gas": "Gas", "gas": "Gas", "Gases": "Gas", "gases": "Gas",
        "Oil": "Oil", "oil": "Oil", "Liquid": "Oil", "liquid": "Oil", "Liquids": "Oil", "liquids": "Oil",
        "Coal": "Coal", "coal": "Coal", "Solid": "Coal", "solid": "Coal", "Solids": "Coal", "solids": "Coal",
        "Biomass": "Biomass", "biomass": "Biomass",
        "Nuclear": "Nuclear", "nuclear": "Nuclear",
        "Hydro": "Hydro", "hydro": "Hydro",
        "Solar": "Solar", "solar": "Solar",
        "Wind": "Wind", "wind": "Wind",
        "Geothermal": "Geothermal", "geothermal": "Geothermal"
    }
    
    if s in exact_map:
        return exact_map[s]
    
    # Fallback to contains for edge cases
    s_lower = s.lower()
    if "electric" in s_lower:
        return "Electricity"
    if "gas" in s_lower:
        return "Gas"
    if "oil" in s_lower or "liquid" in s_lower:
        return "Oil"
    if "coal" in s_lower or "solid" in s_lower:
        return "Coal"
    if "biomass" in s_lower or s_lower.startswith("bio"):
        return "Biomass"
    if "nuclear" in s_lower:
        return "Nuclear"
    if "hydro" in s_lower:
        return "Hydro"
    if "solar" in s_lower:
        return "Solar"
    if "wind" in s_lower:
        return "Wind"
    if "geothermal" in s_lower:
        return "Geothermal"
    return s.title()


def build_price_tables(step1_df: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    if step1_df is None:
        return {
            "elec": pd.DataFrame(),
            "primary": pd.DataFrame(),
            "secondary_by_fuel": pd.DataFrame(),
        }

    prices = step1_df[step1_df["col1"] == "Price"].copy()
    if prices.empty:
        return {
            "elec": pd.DataFrame(),
            "primary": pd.DataFrame(),
            "secondary_by_fuel": pd.DataFrame(),
        }

    # Vectorized price conversion to USD/MWh
    unit_lower = prices["unit"].astype(str).str.lower()
    vals = prices["value"].astype(float)
    converted = vals.copy()
    gj = unit_lower.str.contains("gj")
    tj = unit_lower.str.contains("tj")
    pj = unit_lower.str.contains("pj")
    ej = unit_lower.str.contains("ej")
    converted.loc[gj] = vals.loc[gj] * 3.6
    converted.loc[tj] = vals.loc[tj] * 3.6 * 1_000
    converted.loc[pj] = vals.loc[pj] * 3.6 * 1_000_000
    converted.loc[ej] = vals.loc[ej] * 3.6 * 1_000_000_000
    prices["price_usd_per_mwh"] = converted

    # Normalize fuel labels upfront
    prices["Fuel_norm"] = prices["Fuel"].apply(_normalize_fuel_label)

    # Electricity (Secondary Energy, Electricity fuel)
    elec = prices[
        (prices["col2"] == "Secondary Energy") & (prices["Fuel_norm"] == "Electricity")
    ][["model", "scenario", "region", "year", "price_usd_per_mwh"]].copy()
    elec = elec.groupby(["model", "scenario", "region", "year"], as_index=False)[
        "price_usd_per_mwh"
    ].first()

    # Primary energy (not fuel-specific; take first per group)
    primary = prices[(prices["col2"] == "Primary Energy")][
        ["model", "scenario", "region", "year", "price_usd_per_mwh"]
    ].copy()
    primary = primary.groupby(["model", "scenario", "region", "year"], as_index=False)[
        "price_usd_per_mwh"
    ].first()

    # Secondary energy by fuel (fuel-specific table)
    secondary_by_fuel = prices[(prices["col2"] == "Secondary Energy")][
        ["model", "scenario", "region", "year", "Fuel_norm", "price_usd_per_mwh"]
    ].copy()
    secondary_by_fuel = secondary_by_fuel.groupby(
        ["model", "scenario", "region", "year", "Fuel_norm"], as_index=False
    )["price_usd_per_mwh"].first()

    return {"elec": elec, "primary": primary, "secondary_by_fuel": secondary_by_fuel}


def step3_finalize_target_schema() -> None:
    print_banner("STEP 3 — Finalize Target Schema")
    df = pd.read_csv("2_final_AR6_filtered.csv", low_memory=False)

    target = pd.DataFrame()
    target["scenario_provider"] = df["model"]
    target["scenario"] = df["scenario"]
    target["scenario_geography"] = df["scenario_geography"]
    target["sector"] = df["Sector"]
    target["technology"] = df["Technology"]
    target["scenario_year"] = df["year"]

    # Load metadata for scenario_type (if present)
    target["scenario_type"] = ""
    try:
        meta_df = pd.read_excel(
            "AR6_Scenarios_Database_metadata_indicators_v1.1 2.xlsx",
            sheet_name="meta_Ch3vetted_withclimate",
        )
        if set(["Model", "Scenario", "Category"]).issubset(meta_df.columns):
            meta_df = meta_df[["Model", "Scenario", "Category"]].dropna()
            meta_df["lookup_key"] = (
                meta_df["Model"].astype(str) + "|||" + meta_df["Scenario"].astype(str)
            )
            lookup = meta_df.set_index("lookup_key")["Category"].to_dict()
            key = (
                target["scenario_provider"].astype(str)
                + "|||"
                + target["scenario"].astype(str)
            )
            target["scenario_type"] = key.map(lookup).fillna("")
    except FileNotFoundError:
        print("⚠️ Metadata Excel not found. scenario_type left empty.")

    # Technology type
    def classify_tech(sector: str, technology: str) -> str:
        if sector == "Renewables":
            return "greentech"
        for kw in ["Solar", "Wind", "Hydro", "Nuclear"]:
            if kw in str(technology):
                return "greentech"
        return "carbontech"

    target["technology_type"] = [
        classify_tech(sec, tech)
        for sec, tech in zip(target["sector"], target["technology"])
    ]

    # Price unit & indicator (fixed as USD/MWh as in original script)
    target["price_unit"] = "USD/MWh"
    target["price_indicator"] = np.nan

    # scenario_price / fuel_price using step1 original (vectorized merges)
    step1_df = load_step1_for_price()
    price_tables = build_price_tables(step1_df)

    # Fuel mapping for fuel_price
    fuel_map = {
        "GasCap": "Gas",
        "GasCap_w/ CCS": "Gas",
        "GasCap_w/o CCS": "Gas",
        "CoalCap": "Coal",
        "CoalCap_w/ CCS": "Coal",
        "CoalCap_w/o CCS": "Coal",
        "OilCap": "Oil",
        "OilCap_w/ CCS": "Oil",
        "OilCap_w/o CCS": "Oil",
        "BiomassCap": "Biomass",
        "BiomassCap_w/ CCS": "Biomass",
        "BiomassCap_w/o CCS": "Biomass",
        "NuclearCap": "Nuclear",
        "HydroCap": "Hydro",
        "WindCap": "Wind",
        "SolarCap": "Solar",
        "GeothermalCap": "Geothermal",
        "OceanCap": "Ocean",
    }

    # Build keys for joins
    join_key = ["scenario_provider", "scenario", "scenario_geography", "scenario_year"]
    # Prepare price tables with matching column names
    elec = (
        price_tables["elec"].rename(
            columns={
                "model": "scenario_provider",
                "scenario": "scenario",
                "region": "scenario_geography",
                "year": "scenario_year",
                "price_usd_per_mwh": "scenario_price_electricity",
            }
        )
        if not price_tables["elec"].empty
        else pd.DataFrame(columns=join_key + ["scenario_price_electricity"])
    )

    primary = (
        price_tables["primary"].rename(
            columns={
                "model": "scenario_provider",
                "scenario": "scenario",
                "region": "scenario_geography",
                "year": "scenario_year",
                "price_usd_per_mwh": "scenario_price_primary",
            }
        )
        if not price_tables["primary"].empty
        else pd.DataFrame(columns=join_key + ["scenario_price_primary"])
    )

    sec_by_fuel = (
        price_tables["secondary_by_fuel"].rename(
            columns={
                "model": "scenario_provider",
                "scenario": "scenario",
                "region": "scenario_geography",
                "year": "scenario_year",
                "Fuel_norm": "fuel_for_price_norm",
                "price_usd_per_mwh": "fuel_price",
            }
        )
        if not price_tables["secondary_by_fuel"].empty
        else pd.DataFrame(columns=join_key + ["fuel_for_price", "fuel_price"])
    )

    # scenario_price: Power/Renewables -> electricity; others -> primary
    target = target.merge(elec, on=join_key, how="left")
    target = target.merge(primary, on=join_key, how="left")
    is_power_like = (
        target["sector"].isin(["Power", "Renewables"])
        if "sector" in target.columns
        else pd.Series(False, index=target.index)
    )
    target["scenario_price"] = np.where(
        is_power_like,
        target["scenario_price_electricity"],
        target["scenario_price_primary"],
    )
    target = target.drop(
        columns=["scenario_price_electricity", "scenario_price_primary"],
        errors="ignore",
    )

    # fuel_price: technology → fuel mapping then merge with secondary-by-fuel prices
    target["fuel_for_price"] = target["technology"].map(fuel_map).fillna("Gas")
    target["fuel_for_price_norm"] = target["fuel_for_price"].apply(
        _normalize_fuel_label
    )
    target = target.merge(
        sec_by_fuel, on=join_key + ["fuel_for_price_norm"], how="left"
    )
    target = target.drop(columns=["fuel_for_price_norm"], errors="ignore")
    
    # Set fuel_price to 0 for renewable technologies (they don't consume fuel)
    renewable_techs = [
        "SolarCap", "WindCap", "HydroCap", "GeothermalCap", "NuclearCap", "OceanCap",
        "Non-Biomass Renewables", "Electricity - Non-Biomass Renewables"
    ]
    
    is_renewable_tech = target["technology"].isin(renewable_techs)
    
    # Set fuel price to 0 for renewables (free fuel)
    target.loc[is_renewable_tech, "fuel_price"] = 0.0
    print(f"Set fuel_price=0 for {is_renewable_tech.sum()} renewable technology entries")

    # Pathway logic (Power/Renewables vs Coal/Gas&Oil)
    # Determine pathway_unit and scenario_pathway from df columns (vectorized)
    has_primary = (
        df["primary_energy_mwh_per_yr"].notna()
        if "primary_energy_mwh_per_yr" in df.columns
        else pd.Series(False, index=df.index)
    )
    has_secondary = (
        df["secondary_energy_mwh_per_yr"].notna()
        if "secondary_energy_mwh_per_yr" in df.columns
        else pd.Series(False, index=df.index)
    )
    has_capacity = (
        df["capacity_mw"].notna()
        if "capacity_mw" in df.columns
        else pd.Series(False, index=df.index)
    )

    # pathway_unit default MWh/yr, set to MW only where appropriate
    target["pathway_unit"] = "MWh/yr"
    idx = ~has_primary & ~has_secondary & has_capacity
    if len(target) == len(df):
        target.loc[idx, "pathway_unit"] = "MW"

    # scenario_pathway by sector rules
    sec_series = target["sector"].astype(str)
    scenario_pathway = pd.Series(np.nan, index=target.index, dtype=float)
    if "primary_energy_mwh_per_yr" in df.columns:
        scenario_pathway = np.where(
            sec_series.isin(["Coal", "Gas&Oil"]),
            df["primary_energy_mwh_per_yr"],
            scenario_pathway,
        )
    # For Power/Renewables
    if "secondary_energy_mwh_per_yr" in df.columns:
        use_secondary = sec_series.isin(["Power", "Renewables"]) & has_secondary
        scenario_pathway = np.where(
            use_secondary, df["secondary_energy_mwh_per_yr"], scenario_pathway
        )
    if "capacity_mw" in df.columns:
        use_capacity = (
            sec_series.isin(["Power", "Renewables"]) & ~has_secondary & has_capacity
        )
        scenario_pathway = np.where(use_capacity, df["capacity_mw"], scenario_pathway)
    target["scenario_pathway"] = scenario_pathway

    # Capacity factor (Power/Renewables only): secondary_energy / (capacity * 8760)
    if "secondary_energy_mwh_per_yr" in df.columns and "capacity_mw" in df.columns:
        denom = df["capacity_mw"] * 8760.0
        cf = np.where(
            sec_series.isin(["Power", "Renewables"])
            & df["secondary_energy_mwh_per_yr"].notna()
            & df["capacity_mw"].notna()
            & (df["capacity_mw"] > 0),
            df["secondary_energy_mwh_per_yr"] / denom,
            np.nan,
        )
        target["scenario_capacity_factor"] = cf
    else:
        target["scenario_capacity_factor"] = np.nan
    target["scenario_capacity_factor"] = target["scenario_capacity_factor"].clip(0, 1)

    # Geography → ISO2 list mapping
    def build_iso_mapping() -> Dict[str, str]:
        country_map = {
            "ARG": "AR",
            "AUS": "AU",
            "BRA": "BR",
            "CAN": "CA",
            "CHN": "CN",
            "IDN": "ID",
            "IND": "IN",
            "JPN": "JP",
            "KOR": "KR",
            "MEX": "MX",
            "RUS": "RU",
            "SAU": "SA",
            "TUR": "TR",
            "USA": "US",
            "ZAF": "ZA",
            "AGO": "AO",
            "AUT": "AT",
            "BEL": "BE",
            "BGD": "BD",
            "BGR": "BG",
            "BIH": "BA",
            "BOL": "BO",
            "CHE": "CH",
            "CHL": "CL",
            "COL": "CO",
            "CYP": "CY",
            "CZE": "CZ",
            "DEU": "DE",
            "DNK": "DK",
            "DZA": "DZ",
            "ECU": "EC",
            "EGY": "EG",
            "ESP": "ES",
            "EST": "EE",
            "ETH": "ET",
            "FIN": "FI",
            "FRA": "FR",
            "GBR": "GB",
            "GHA": "GH",
            "GRC": "GR",
            "HRV": "HR",
            "HUN": "HU",
            "IRL": "IE",
            "ISL": "IS",
            "ITA": "IT",
            "KAZ": "KZ",
            "KEN": "KE",
            "LBY": "LY",
            "LTU": "LT",
            "LUX": "LU",
            "LVA": "LV",
            "MAR": "MA",
            "MDG": "MG",
            "MLT": "MT",
            "MOZ": "MZ",
            "NGA": "NG",
            "NLD": "NL",
            "NOR": "NO",
            "NZL": "NZ",
            "PAK": "PK",
            "PER": "PE",
            "POL": "PL",
            "PRT": "PT",
            "ROU": "RO",
            "SRB": "RS",
            "SVK": "SK",
            "SVN": "SI",
            "SWE": "SE",
            "THA": "TH",
            "TUN": "TN",
            "TWN": "TW",
            "UGA": "UG",
            "VEN": "VE",
            "VNM": "VN",
        }
        regional = {
            "EU": [
                "AT",
                "BE",
                "BG",
                "HR",
                "CY",
                "CZ",
                "DK",
                "EE",
                "FI",
                "FR",
                "DE",
                "GR",
                "HU",
                "IE",
                "IT",
                "LV",
                "LT",
                "LU",
                "MT",
                "NL",
                "PL",
                "PT",
                "RO",
                "SK",
                "SI",
                "ES",
                "SE",
            ]
        }
        r10_map: Dict[str, str] = {}
        try:
            r10_lookup = pd.read_csv("r10_region_lookup.csv")
            if set(["R10 Region", "ISO2"]).issubset(r10_lookup.columns):
                for _, r in r10_lookup.iterrows():
                    r10 = r["R10 Region"]
                    iso2_raw = r["ISO2"]
                    if pd.notna(iso2_raw) and str(iso2_raw).strip():
                        parts = [
                            p.strip() for p in str(iso2_raw).split("|") if p.strip()
                        ]
                        r10_map[r10] = ",".join(sorted(parts))
        except FileNotFoundError:
            pass
        r10_map["R10ROWO"] = "ROW"

        full = {**country_map}
        for region, iso2s in regional.items():
            full[region] = ",".join(sorted(iso2s))
        full.update(r10_map)
        return full

    iso_map = build_iso_mapping()
    target["country_iso2_list"] = (
        target["scenario_geography"].map(iso_map).fillna(target["scenario_geography"])
    )  # fallback

    # Bring additional columns from Step 2
    for col in [
        "lifetime_years",
        "efficiency_decimal",
        "capacity_additions_mw_per_yr",
        "om_cost_usd_per_mw_per_yr",
        "capital_cost_usd_per_mw",
        "carbon_price_usd_per_tco2",
    ]:
        if col in df.columns:
            target[col] = df[col]

    # Create global geography aggregations before saving
    print_banner("STEP 3a — Creating Global Geography Aggregations")
    target_with_global = create_global_geography(target)

    # Apply OilCap cost heuristic (CCS-aware) before temporal interpolation
    print_banner("STEP 3b — OilCap Cost Heuristic (CCS-Aware)")
    target_with_global = apply_oilcap_cost_heuristic(target_with_global)

    # Save and cleanup
    out = "3_final_AR6_target_schema.csv"
    cols = [
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
        "fuel_for_price",
        "pathway_unit",
        "scenario_pathway",
        "scenario_capacity_factor",
        "scenario_year",
        "country_iso2_list",
        "lifetime_years",
        "efficiency_decimal",
        "capacity_additions_mw_per_yr",
        "om_cost_usd_per_mw_per_yr",
        "capital_cost_usd_per_mw",
        "carbon_price_usd_per_tco2",
    ]
    cols = [c for c in cols if c in target_with_global.columns]
    target_with_global[cols].to_csv(out, index=False)
    print(f"✅ Wrote {out} | Shape: {target_with_global[cols].shape}")
    memory_release(df, target, target_with_global)


# ======================================
# Global Aggregation Helper
# ======================================


def apply_oilcap_cost_heuristic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply CCS-aware OilCap cost heuristic: 
    - OilCap w/ CCS = average(GasCap w/ CCS, CoalCap w/ CCS)
    - OilCap w/o CCS = average(GasCap w/o CCS, CoalCap w/o CCS)
    """
    print("🛢️ Applying CCS-aware OilCap cost heuristic...")
    
    df_result = df.copy()
    cost_cols = ["capital_cost_usd_per_mw", "om_cost_usd_per_mw_per_yr"]
    available_cost_cols = [col for col in cost_cols if col in df.columns]
    
    if not available_cost_cols:
        print("   No cost columns found, skipping OilCap heuristic")
        return df_result
    
    # Define exact technology names for fast exact matching
    oilcap_techs = ["OilCap", "OilCap_w/ CCS", "OilCap_w/o CCS"]
    gascap_techs = ["GasCap", "GasCap_w/ CCS", "GasCap_w/o CCS"]
    coalcap_techs = ["CoalCap", "CoalCap_w/ CCS", "CoalCap_w/o CCS"]
    
    # Technology to CCS status mapping
    tech_ccs_map = {
        "OilCap": "unknown", "GasCap": "unknown", "CoalCap": "unknown",
        "OilCap_w/ CCS": "with_ccs", "GasCap_w/ CCS": "with_ccs", "CoalCap_w/ CCS": "with_ccs",
        "OilCap_w/o CCS": "without_ccs", "GasCap_w/o CCS": "without_ccs", "CoalCap_w/o CCS": "without_ccs"
    }
    
    # Identify OilCap technologies using exact matching
    oilcap_mask = df["technology"].isin(oilcap_techs)
    if not oilcap_mask.any():
        print("   No OilCap technologies found")
        return df_result
    
    filled_counts = {col: 0 for col in available_cost_cols}
    
    for cost_col in available_cost_cols:
        print(f"   Processing {cost_col}...")
        
        # Find OilCap rows missing this cost
        missing_mask = oilcap_mask & df[cost_col].isna()
        
        if not missing_mask.any():
            print(f"     No missing {cost_col} values for OilCap")
            continue
            
        print(f"     Filling {missing_mask.sum():,} missing {cost_col} values")
        
        # Group by scenario dimensions to calculate averages efficiently
        grouping_cols = [
            "scenario_provider", "scenario", "scenario_geography", 
            "scenario_year", "sector"
        ]
        
        # Map technologies to CCS status using exact matching
        df_result["_ccs_status"] = df_result["technology"].map(tech_ccs_map).fillna("unknown")
        
        # Calculate GasCap and CoalCap averages by CCS status
        for ccs_status in ["with_ccs", "without_ccs", "unknown"]:
            ccs_mask = df_result["_ccs_status"] == ccs_status
            oil_missing = missing_mask & ccs_mask
            
            if not oil_missing.any():
                continue
                
            print(f"       Processing CCS status: {ccs_status} ({oil_missing.sum():,} rows)")
            
            # Find corresponding Gas and Coal technologies with same CCS status using exact matching
            gas_techs_for_ccs = [tech for tech in gascap_techs if tech_ccs_map.get(tech) == ccs_status]
            coal_techs_for_ccs = [tech for tech in coalcap_techs if tech_ccs_map.get(tech) == ccs_status]
            
            gas_mask = df_result["technology"].isin(gas_techs_for_ccs)
            coal_mask = df_result["technology"].isin(coal_techs_for_ccs)
            
            # Calculate averages by grouping dimensions
            gas_avg = (df_result.loc[gas_mask, grouping_cols + [cost_col]]
                      .groupby(grouping_cols, as_index=False)[cost_col]
                      .mean()
                      .rename(columns={cost_col: f"{cost_col}_gas"}))
            
            coal_avg = (df_result.loc[coal_mask, grouping_cols + [cost_col]]
                       .groupby(grouping_cols, as_index=False)[cost_col]
                       .mean()
                       .rename(columns={cost_col: f"{cost_col}_coal"}))
            
            # Merge averages
            oil_data = df_result.loc[oil_missing, grouping_cols].reset_index()
            oil_with_gas = oil_data.merge(gas_avg, on=grouping_cols, how="left")
            oil_with_both = oil_with_gas.merge(coal_avg, on=grouping_cols, how="left")
            
            # Calculate OilCap cost as average of Gas and Coal
            gas_col = f"{cost_col}_gas"
            coal_col = f"{cost_col}_coal"
            
            # Calculate average where both are available
            both_available = oil_with_both[gas_col].notna() & oil_with_both[coal_col].notna()
            oil_with_both.loc[both_available, f"{cost_col}_avg"] = (
                (oil_with_both.loc[both_available, gas_col] + 
                 oil_with_both.loc[both_available, coal_col]) / 2
            )
            
            # Use single value where only one is available
            only_gas = oil_with_both[gas_col].notna() & oil_with_both[coal_col].isna()
            oil_with_both.loc[only_gas, f"{cost_col}_avg"] = oil_with_both.loc[only_gas, gas_col]
            
            only_coal = oil_with_both[gas_col].isna() & oil_with_both[coal_col].notna()
            oil_with_both.loc[only_coal, f"{cost_col}_avg"] = oil_with_both.loc[only_coal, coal_col]
            
            # Apply calculated values back to dataframe
            filled_values = oil_with_both[f"{cost_col}_avg"].notna()
            if filled_values.any():
                original_indices = oil_with_both.loc[filled_values, "index"]
                df_result.loc[original_indices, cost_col] = oil_with_both.loc[filled_values, f"{cost_col}_avg"].values
                filled_counts[cost_col] += filled_values.sum()
        
        # Clean up temporary column
        df_result = df_result.drop(columns=["_ccs_status"])
    
    total_filled = sum(filled_counts.values())
    print(f"   OilCap heuristic completed: {total_filled:,} cost values filled")
    for col, count in filled_counts.items():
        if count > 0:
            print(f"     {col}: {count:,} values")
    
    return df_result


def create_global_geography(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Create 'Global' scenario_geography entries by aggregating across all geographies 
    for each scenario. Uses sums for capacity-like metrics and weighted/simple averages for prices.
    
    Args:
        df_in: DataFrame with scenario data across different geographies
        
    Returns:
        DataFrame with additional rows for 'Global' scenario_geography
    """
    print("🌍 Creating Global scenario_geography aggregations...")
    
    df = df_in.copy()
    
    # Define grouping columns (exclude geography for global aggregation)
    grouping_cols = [
        "scenario_provider", 
        "scenario", 
        "scenario_type",
        "sector", 
        "technology", 
        "scenario_year"
    ]
    grouping_cols = [c for c in grouping_cols if c in df.columns]
    
    # Define aggregation strategies
    sum_cols = [
        "scenario_pathway",  # Sum capacities/energy
        "capacity_additions_mw_per_yr"
    ]
    sum_cols = [c for c in sum_cols if c in df.columns]
    
    # Weighted average by scenario_pathway for these columns
    weighted_avg_cols = [
        "scenario_capacity_factor",
        "efficiency_decimal", 
        "lifetime_years",
        "om_cost_usd_per_mw_per_yr",
        "capital_cost_usd_per_mw"
    ]
    weighted_avg_cols = [c for c in weighted_avg_cols if c in df.columns]
    
    # Simple average for price columns (not capacity-weighted)
    simple_avg_cols = [
        "scenario_price",
        "fuel_price", 
        "carbon_price_usd_per_tco2"
    ]
    simple_avg_cols = [c for c in simple_avg_cols if c in df.columns]
    
    def aggregate_to_global(group: pd.DataFrame) -> pd.Series:
        """Aggregate a group of geographies to global values"""
        result = {}
        
        # Sum the additive quantities
        for col in sum_cols:
            result[col] = group[col].sum()
        
        # Weighted averages (weighted by scenario_pathway if available)
        weight_col = "scenario_pathway" if "scenario_pathway" in group.columns else None
        for col in weighted_avg_cols:
            if weight_col and not group[weight_col].isna().all():
                weights = group[weight_col].fillna(0)
                values = group[col]
                # Only include non-NaN values in weighted average
                valid_mask = values.notna() & (weights > 0)
                if valid_mask.any():
                    result[col] = (values[valid_mask] * weights[valid_mask]).sum() / weights[valid_mask].sum()
                else:
                    result[col] = values.mean()  # Fallback to simple mean
            else:
                result[col] = group[col].mean()
        
        # Simple averages for prices
        for col in simple_avg_cols:
            result[col] = group[col].mean()
        
        # Copy metadata from first row (should be same across geographies for a scenario)
        metadata_cols = [c for c in group.columns 
                        if c not in sum_cols + weighted_avg_cols + simple_avg_cols + grouping_cols
                        and c not in ["scenario_geography", "country_iso2_list"]]
        first_row = group.iloc[0]
        for col in metadata_cols:
            result[col] = first_row[col]
            
        # Set geography to "Global" and empty country list
        result["scenario_geography"] = "Global"
        result["country_iso2_list"] = ""  # Global should have empty country list
        
        return pd.Series(result)
    
    # Group by scenario dimensions (excluding geography) and aggregate
    print(f"   Grouping by: {grouping_cols}")
    print(f"   Processing {len(df)} rows across {df['scenario_geography'].nunique()} geographies")
    
    global_aggregated = df.groupby(grouping_cols, as_index=False).apply(aggregate_to_global, include_groups=False)
    
    # Reset index if needed (groupby.apply can create multi-index)
    if isinstance(global_aggregated.index, pd.MultiIndex):
        global_aggregated = global_aggregated.reset_index(drop=True)
    
    print(f"   Created {len(global_aggregated)} global aggregation rows")
    
    # Combine original data with global aggregations
    combined = pd.concat([df, global_aggregated], ignore_index=True)
    print(f"   Total rows after adding Global geography: {len(combined)}")
    
    return combined


# ======================================
# Step 4: Aggregation and Gap-Filling
# ======================================


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
    # Include key data columns regardless of their current dtype
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
            # Try to convert to numeric if not already
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

    # Group by essential identifier columns only to define time series groups
    # Don't include columns that are likely to have many NaN values or are metadata-like
    essential_id_cols = [
        "scenario_provider",
        "scenario",
        "scenario_geography",
        "sector",
        "technology",
    ]
    grouping_cols = [col for col in essential_id_cols if col in df_in.columns]

    print(f"  DEBUG: All columns in df_in: {list(df_in.columns)}")
    print(f"  DEBUG: Grouping columns filter result: {grouping_cols}")

    print(
        f"Temporal interpolation: processing {len(numeric_cols)} numeric columns across {len(grouping_cols)} grouping dimensions"
    )
    print(f"  DEBUG: Numeric columns: {numeric_cols}")
    print(f"  DEBUG: Grouping columns: {grouping_cols}")
    print(f"  DEBUG: Available years in data: {available_years}")
    print(f"  DEBUG: Expected years: {sorted(expected_years)}")

    # Generate all required years
    all_years = list(range(start_year, end_year + 1))

    interpolated_groups = []
    total_groups = 0
    processed_groups = 0

    print(f"  DEBUG: Starting groupby with {len(df_in)} rows")

    # Check for any NaN values in grouping columns
    for col in grouping_cols:
        nan_count = df_in[col].isna().sum()
        if nan_count > 0:
            print(f"  WARNING: {col} has {nan_count} NaN values")

    df_for_groupby = df_in

    # VECTORIZED APPROACH: Use template merge + groupby interpolation
    print(f"  Using vectorized temporal interpolation...")
    
    # Create complete template with all years for each group
    unique_groups = df_for_groupby[grouping_cols].drop_duplicates()
    
    # Create cartesian product of groups × years
    years_df = pd.DataFrame({'scenario_year': all_years})
    complete_template = unique_groups.assign(key=1).merge(years_df.assign(key=1), on='key').drop('key', axis=1)
    
    print(f"  Created template with {len(complete_template):,} group-year combinations")
    
    # Merge with original data
    merged = complete_template.merge(df_for_groupby, on=grouping_cols + ['scenario_year'], how='left')
    
    # Interpolate within each group using vectorized operations
    print("  Applying vectorized interpolation...")
    
    def interpolate_group(group):
        """Vectorized interpolation for a single group"""
        # Sort by year
        group = group.sort_values('scenario_year')
        
        # Fill categorical columns with first non-null value
        categorical_cols = [
            "scenario_type", "technology_type", "price_unit", "price_indicator",
            "fuel_for_price", "pathway_unit", "country_iso2_list", "stringency"
        ]

        for col in categorical_cols:
            if col in group.columns:
                non_null_values = group[col].dropna()
                if len(non_null_values) > 0:
                    fill_value = non_null_values.iloc[0]
                    group[col] = group[col].fillna(fill_value)

        # Interpolate numeric columns
        for col in numeric_cols:
            if col in group.columns:
                # Linear interpolation
                group[col] = group[col].interpolate(method='linear')
                # Forward and backward fill for extrapolation
                group[col] = group[col].bfill().ffill()
        
        return group
    
    # Apply interpolation to each group (vectorized operations within groups)
    try:
        interpolated = merged.groupby(grouping_cols, group_keys=False).apply(interpolate_group)
        
        # Remove rows where interpolation failed (still NaN for all numeric columns)
        # Only require at least ONE numeric column to be non-null to keep the row
        numeric_check = interpolated[numeric_cols].notna().any(axis=1)
        result = interpolated[numeric_check]
        
        total_groups = merged.groupby(grouping_cols).ngroups
        processed_groups = result.groupby(grouping_cols).ngroups
        
        print(f"  ✅ Vectorized interpolation complete")
        print(f"  📈 Result: {len(result):,} rows (from {len(df_in):,})")
        print(f"  📊 Processed {processed_groups:,} groups successfully")

        return result

    except Exception as e:
        print(f"  ERROR: Vectorized interpolation failed: {e}")
        return df_in


def vectorized_mapping(df_in: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized sector/technology mapping via merge instead of row-wise loops."""
    left = df_in.copy()
    right = mapping_df.copy()

    right = right.rename(
        columns={
            "current_sector": "sector",
            "current_technology": "technology",
        }
    )
    left = left.merge(
        right[
            [
                "sector",
                "technology",
                "target_sector",
                "target_technology",
                "aggregation_group",
            ]
        ],
        on=["sector", "technology"],
        how="left",
    )
    # Fallback: if not mapped, keep original
    left["target_sector"] = left["target_sector"].fillna(left["sector"])
    left["target_technology"] = left["target_technology"].fillna(left["technology"])
    left["aggregation_group"] = left["aggregation_group"].fillna("single")
    return left



def build_iso2_geographic_similarity_matrix(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Build a similarity matrix between geographies based on ISO2 country overlap.
    Returns nested dict: {geo1: {geo2: overlap_score, ...}, ...}
    """
    print("🗺️  Building ISO2-based geographic similarity matrix...")
    
    # Get unique geographies and their ISO2 lists
    geo_iso2_map = {}
    for geo in df['scenario_geography'].unique():
        iso2_str = df[df['scenario_geography'] == geo]['country_iso2_list'].iloc[0]
        if pd.notna(iso2_str) and iso2_str.strip():
            # Parse comma-separated ISO2 codes
            iso2_set = set(code.strip() for code in str(iso2_str).split(',') if code.strip())
            geo_iso2_map[geo] = iso2_set
        else:
            geo_iso2_map[geo] = set()  # Empty for Global or missing
    
    # Calculate overlap scores between all geography pairs
    similarity_matrix = {}
    for geo1 in geo_iso2_map:
        similarity_matrix[geo1] = {}
        for geo2 in geo_iso2_map:
            if geo1 == geo2:
                similarity_matrix[geo1][geo2] = 1.0  # Perfect match
            elif not geo_iso2_map[geo1] or not geo_iso2_map[geo2]:
                similarity_matrix[geo1][geo2] = 0.0  # No overlap if either is empty
            else:
                # Jaccard similarity: intersection / union
                intersection = len(geo_iso2_map[geo1] & geo_iso2_map[geo2])
                union = len(geo_iso2_map[geo1] | geo_iso2_map[geo2])
                similarity_matrix[geo1][geo2] = intersection / union if union > 0 else 0.0
    
    # Print some examples
    print(f"   Built similarity matrix for {len(geo_iso2_map)} geographies")
    print("   Example overlaps:")
    for geo1 in list(geo_iso2_map.keys())[:3]:
        for geo2 in list(geo_iso2_map.keys())[:3]:
            if geo1 != geo2:
                score = similarity_matrix[geo1][geo2]
                if score > 0:
                    print(f"     {geo1} ↔ {geo2}: {score:.2f}")
    
    return similarity_matrix


def create_technology_lookup_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a focused lookup table with best available values for only the technologies we actually use.
    
    Returns DataFrame with columns:
    technology, year, iso2, stringency, capital_cost, om_cost, efficiency, lifetime, capacity_factor
    """
    print("📋 Creating technology lookup table...")
    
    # Define the 15 target technologies we actually use (much faster than all 46)
    target_techs = [
        # Core power technologies
        "SolarCap", "WindCap", "HydroCap", "GeothermalCap", "NuclearCap", 
        "BiomassCap", "CoalCap", "GasCap", "OilCap",
        # CCS variants
        "BiomassCap_w/ CCS", "CoalCap_w/ CCS", "GasCap_w/ CCS", "OilCap_w/ CCS",
        "BiomassCap_w/o CCS", "CoalCap_w/o CCS", "GasCap_w/o CCS", "OilCap_w/o CCS",
        # Other renewables
        "OceanCap", "Non-Biomass Renewables", "Electricity - Non-Biomass Renewables"
    ]
    
    # Only use technologies that exist in our data AND are in our target list
    available_techs = df['technology'].unique()
    unique_techs = [tech for tech in target_techs if tech in available_techs]
    
    unique_years = sorted(df['scenario_year'].unique())
    unique_stringencies = sorted(df['stringency'].unique()) if 'stringency' in df.columns else ['']
    
    # Get only ISO2 codes that are actually used in the data
    all_iso2s = set()
    for iso2_list in df['country_iso2_list'].dropna():
        if str(iso2_list).strip():
            iso2s = [code.strip() for code in str(iso2_list).split(',') if code.strip()]
            all_iso2s.update(iso2s)
    all_iso2s = sorted(all_iso2s)
    
    print(f"   Building lookup for {len(unique_techs)} target techs × {len(unique_years)} years × {len(all_iso2s)} ISO2s × {len(unique_stringencies)} stringencies")
    print(f"   Target technologies: {unique_techs}")
    
    # Columns to populate
    value_cols = ['capital_cost_usd_per_mw', 'om_cost_usd_per_mw_per_yr', 'efficiency_decimal', 'lifetime_years', 'scenario_capacity_factor']
    available_value_cols = [col for col in value_cols if col in df.columns]
    
    # Create all combinations
    lookup_rows = []
    
    for tech in unique_techs:
        for year in unique_years:
            for iso2 in all_iso2s:
                for stringency in unique_stringencies:
                    # Create base row
                    row = {
                        'technology': tech,
                        'year': year,
                        'iso2': iso2,
                        'stringency': stringency
                    }
                    
                    # Fill each value column using hierarchical approach
                    for col in available_value_cols:
                        row[col.replace('_usd_per_mw_per_yr', '').replace('_usd_per_mw', '').replace('_decimal', '').replace('_years', '').replace('scenario_', '')] = None
                    
                    lookup_rows.append(row)
    
    lookup_df = pd.DataFrame(lookup_rows)
    
    # Hierarchical gap-filling for lookup table
    print(f"   Applying hierarchical gap-filling to lookup table...")
    
    for col in available_value_cols:
        col_short = col.replace('_usd_per_mw_per_yr', '').replace('_usd_per_mw', '').replace('_decimal', '').replace('_years', '').replace('scenario_', '')
        print(f"     Processing {col_short}...")
        
        filled_count = 0
        
        # Hierarchy levels for lookup table
        hierarchy = [
            # Most specific: exact matches
            {'tech': True, 'year': True, 'iso2': True, 'stringency': True},
            {'tech': True, 'year': True, 'iso2': True, 'stringency': False},  # Cross stringency
            {'tech': True, 'year': True, 'iso2': False, 'stringency': True},  # Geographic similarity
            {'tech': True, 'year': True, 'iso2': False, 'stringency': False}, # Cross geo+stringency
            {'tech': True, 'year': False, 'iso2': True, 'stringency': True},  # Cross year
            {'tech': True, 'year': False, 'iso2': True, 'stringency': False}, # Cross year+stringency
            {'tech': True, 'year': False, 'iso2': False, 'stringency': True}, # Cross year+geo
            {'tech': True, 'year': False, 'iso2': False, 'stringency': False}, # Tech only
        ]
        
        for level_idx, level in enumerate(hierarchy):
            missing_mask = lookup_df[col_short].isna()
            if not missing_mask.any():
                break
                
            level_fills = 0
            
            for idx in lookup_df.index[missing_mask]:
                target_tech = lookup_df.loc[idx, 'technology']
                target_year = lookup_df.loc[idx, 'year']
                target_iso2 = lookup_df.loc[idx, 'iso2']
                target_stringency = lookup_df.loc[idx, 'stringency']
                
                # Build query for this level
                query_mask = (df['technology'] == target_tech) if level['tech'] else pd.Series(True, index=df.index)
                
                if level['year']:
                    query_mask &= (df['scenario_year'] == target_year)
                    
                if level['stringency'] and 'stringency' in df.columns:
                    query_mask &= (df['stringency'] == target_stringency)
                
                if level['iso2']:
                    # Exact ISO2 match
                    iso2_mask = df['country_iso2_list'].apply(
                        lambda x: target_iso2 in str(x).split(',') if pd.notna(x) else False
                    )
                    query_mask &= iso2_mask
                else:
                    # Geographic similarity - find geographies containing this ISO2
                    iso2_mask = df['country_iso2_list'].apply(
                        lambda x: target_iso2 in str(x).split(',') if pd.notna(x) else False
                    )
                    if iso2_mask.any():
                        query_mask &= iso2_mask
                
                # Get candidates and calculate median
                candidates = df.loc[query_mask & df[col].notna(), col]
                if not candidates.empty:
                    lookup_df.loc[idx, col_short] = candidates.median()
                    level_fills += 1
            
            if level_fills > 0:
                filled_count += level_fills
                print(f"       Level {level_idx + 1}: filled {level_fills:,} values")
        
        print(f"     Total {col_short}: {filled_count:,} values filled")
    
    # Create stringency=NA fallback rows (median across all stringencies)
    print(f"   Creating stringency=NA fallback rows...")
    
    fallback_rows = []
    for tech in unique_techs:
        for year in unique_years:
            for iso2 in all_iso2s:
                # Get median across all stringencies for this tech/year/iso2
                base_mask = (lookup_df['technology'] == tech) & (lookup_df['year'] == year) & (lookup_df['iso2'] == iso2)
                
                row = {
                    'technology': tech,
                    'year': year,
                    'iso2': iso2,
                    'stringency': None  # NA stringency
                }
                
                for col in available_value_cols:
                    col_short = col.replace('_usd_per_mw_per_yr', '').replace('_usd_per_mw', '').replace('_decimal', '').replace('_years', '').replace('scenario_', '')
                    values = lookup_df.loc[base_mask, col_short].dropna()
                    row[col_short] = values.median() if not values.empty else None
                
                fallback_rows.append(row)
    
    fallback_df = pd.DataFrame(fallback_rows)
    lookup_complete = pd.concat([lookup_df, fallback_df], ignore_index=True)
    
    print(f"   Lookup table created: {len(lookup_complete):,} rows ({len(lookup_df):,} specific + {len(fallback_df):,} fallback)")
    
    # Save lookup table as CSV for analysis
    lookup_output_file = "4_technology_lookup_table.csv"
    lookup_complete.to_csv(lookup_output_file, index=False)
    print(f"   💾 Saved lookup table: {lookup_output_file}")
    
    return lookup_complete


def apply_hierarchical_gap_filling(
    df: pd.DataFrame, 
    col: str, 
    hierarchy: List[List[str]], 
    agg_fn_per_col: Optional[Dict[str, str]] = None
) -> int:
    """Apply hierarchical gap-filling for columns not in lookup table (like scenario_capacity_factor)"""
    print(f"     Applying hierarchical gap-filling for {col}...")
    
    agg_fn = agg_fn_per_col.get(col, "median") if agg_fn_per_col else "median"
    filled_count = 0
    
    for level_idx, level in enumerate(hierarchy):
        missing_mask = df[col].isna()
        if not missing_mask.any():
            break
            
        print(f"       Level {level_idx + 1}: {level}")
        
        # For each missing row, find similar rows to fill from
        for idx in df.index[missing_mask]:
            if not df.loc[idx, col] != df.loc[idx, col]:  # Skip if already filled
                continue
                
            # Build query mask for this level
            query_mask = pd.Series(True, index=df.index)
            for group_col in level:
                if group_col in df.columns:
                    target_value = df.loc[idx, group_col]
                    if pd.notna(target_value):
                        query_mask &= (df[group_col] == target_value)
            
            # Get candidates and fill
            candidates = df.loc[query_mask & df[col].notna(), col]
            if not candidates.empty:
                if agg_fn == "mean":
                    fill_value = candidates.mean()
                else:
                    fill_value = candidates.median()
                    
                df.loc[idx, col] = fill_value
                filled_count += 1
                
                # Update tracking
                current_tracking = df.loc[idx, "gap_filled_columns"]
                if pd.isna(current_tracking) or current_tracking == "":
                    df.loc[idx, "gap_filled_columns"] = col
                else:
                    df.loc[idx, "gap_filled_columns"] = str(current_tracking) + "," + col
    
    print(f"     Hierarchical gap-filling completed: {filled_count} values filled")
    return filled_count


def gap_fill_with_lookup_table(
    df_in: pd.DataFrame,
    spec: Dict[str, List[List[str]]],
    global_fallback_hierarchy: List[List[str]],
    agg_fn_per_col: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Gap-fill columns using pre-computed technology lookup table.

    Args:
        df_in: Input DataFrame
        spec: Dict mapping column name -> list of grouping levels (each level is a list of column names)
        global_fallback_hierarchy: Additional hierarchy levels that specifically use Global geography
        agg_fn_per_col: Optional dict mapping column -> aggregation function ('median' or 'mean')

    Returns:
        DataFrame with gap-filled values and tracking of which columns were filled
    """
    df = df_in.copy()
    
    # Try to load pre-computed lookup table first, create if not available
    try:
        lookup_table = pd.read_csv("technology_lookup_table.csv")
        print("🚀 Using pre-computed technology lookup table")
        print(f"   Loaded: {lookup_table.shape[0]:,} rows × {lookup_table.shape[1]} columns")
    except FileNotFoundError:
        print("📋 Pre-computed lookup table not found, creating new one...")
        print("   💡 Tip: Run 'python3 create_technology_lookup.py' once to speed up future runs")
        lookup_table = create_technology_lookup_table(df)

    if "gap_filled_columns" not in df.columns:
        df["gap_filled_columns"] = ""
    else:
        # Ensure it's properly initialized as string
        df["gap_filled_columns"] = df["gap_filled_columns"].fillna("").astype(str)

    print(f"🔧 Gap-filling {len(spec)} columns using lookup table approach...")
    
    # Map of column names in spec to lookup table columns (with units)
    col_mapping = {
        'scenario_capacity_factor': 'capacity_factor',
        'lifetime_years': 'lifetime_years', 
        'efficiency_decimal': 'efficiency_decimal',
        'om_cost_usd_per_mw_per_yr': 'om_cost_usd_per_mw_per_yr',
        'capital_cost_usd_per_mw': 'capital_cost_usd_per_mw',
        # Price columns now available in lookup table with 80% coverage!
        'fuel_price': 'fuel_price_usd_per_mwh',  # Fuel-specific prices (Coal, Gas, Oil, Biomass)
        # scenario_price is handled with sector-aware logic below (not simple mapping)
    }
    
    # Extract ISO2 codes from country_iso2_list for each row
    def extract_iso2_codes(iso2_list_str):
        if pd.isna(iso2_list_str) or not str(iso2_list_str).strip():
            return []
        return [code.strip() for code in str(iso2_list_str).split(',') if code.strip()]
    
    total_filled_by_col = {}
    
    for col in spec.keys():
        if col not in df.columns:
            print(f"   ⚠️ Skipping {col} - column not found")
            continue

        # Check if this column should use lookup table only
        hierarchy = spec[col]
        use_lookup_only = len(hierarchy) == 0
        
        if use_lookup_only:
            # 🎯 SECTOR-AWARE scenario_price handling
            if col == 'scenario_price':
                # scenario_price uses BOTH electricity and fuel prices based on sector
                if 'electricity_price_usd_per_mwh' not in lookup_table.columns or 'fuel_price_usd_per_mwh' not in lookup_table.columns:
                    print(f"   ⚠️ Skipping {col} - price columns not in lookup table")
                    continue
                print(f"   🎯 {col}: SECTOR-AWARE (Power/Renewables=electricity, Others=fuel)")
                lookup_col = None  # Special handling below
            else:
                lookup_col = col_mapping.get(col, col)
                if lookup_col not in lookup_table.columns:
                    print(f"   ⚠️ Skipping {col} - not in lookup table")
                    continue
        else:
            # For hierarchical gap-filling (scenario_capacity_factor, etc.)
            print(f"   🔧 {col}: using hierarchical gap-filling...")
            filled_count = apply_hierarchical_gap_filling(df, col, hierarchy, agg_fn_per_col)
            total_filled_by_col[col] = filled_count
            continue
            
        missing_mask = df[col].isna()
        initial_missing_count = missing_mask.sum()

        if not missing_mask.any():
            print(f"   ✅ {col}: no missing values")
            continue

        print(f"   🔧 {col}: filling {initial_missing_count:,} missing values using lookup table ONLY...")
        
        # VECTORIZED APPROACH: Process missing rows in batches
        missing_data = df[missing_mask].copy()
        if missing_data.empty:
                continue

        print(f"     Processing {len(missing_data):,} missing rows vectorized...")
        
        # Extract ISO2 codes for all missing rows at once
        missing_data['iso2_codes'] = missing_data['country_iso2_list'].apply(extract_iso2_codes)
        
        filled_count = 0
        fill_values = {}
        
        # Process by stringency hierarchy
        stringency_hierarchy = ['specific', 'UNKNOWN', None]
        
        for stringency_level in stringency_hierarchy:
            if filled_count >= len(missing_data):
                break

            remaining_mask = missing_data.index.isin([idx for idx in missing_data.index if idx not in fill_values])
            if not remaining_mask.any():
                break
                
            remaining_data = missing_data[remaining_mask]
            
            # Determine stringency to use
            if stringency_level == 'specific':
                stringency_col = 'stringency' if 'stringency' in remaining_data.columns else None
            elif stringency_level == 'UNKNOWN':
                stringency_col = None
                target_stringency = 'UNKNOWN'
            else:
                stringency_col = None
                target_stringency = None
            
            # 🚀 BLAZING FAST VECTORIZED lookup using pure merge operations (no iterations!)
            if not remaining_data.empty:
                # Prepare stringency values (vectorized)
                if stringency_level == 'specific':
                    remaining_data['lookup_stringency'] = remaining_data['stringency'] if 'stringency' in remaining_data.columns else 'UNKNOWN'
                else:
                    remaining_data['lookup_stringency'] = target_stringency
                
                # VECTORIZED: Extract single-country data without apply()
                remaining_data['iso2_count'] = remaining_data['iso2_codes'].str.len()
                single_country_mask = remaining_data['iso2_count'] == 1
                single_country_data = remaining_data[single_country_mask].copy()
                
                if not single_country_data.empty:
                    # VECTORIZED: Extract first ISO2 without apply()
                    single_country_data['iso2'] = single_country_data['iso2_codes'].str[0]
                    
                    # 🎯 SECTOR-AWARE scenario_price lookup
                    if col == 'scenario_price':
                        # Determine which price column to use based on sector
                        is_power_like = single_country_data['sector'].isin(['Power', 'Renewables'])
                        
                        # Split data by sector type for different lookups
                        power_data = single_country_data[is_power_like].copy()
                        other_data = single_country_data[~is_power_like].copy()
                        
                        merged_lookup_parts = []
                        
                        # Power/Renewables → electricity price
                        if not power_data.empty:
                            power_merged = power_data.merge(
                                lookup_table[['technology', 'year', 'iso2', 'stringency', 'electricity_price_usd_per_mwh']],
                                left_on=['technology', 'scenario_year', 'iso2', 'lookup_stringency'],
                                right_on=['technology', 'year', 'iso2', 'stringency'],
                                how='left'
                            )
                            power_merged[col] = power_merged['electricity_price_usd_per_mwh']
                            merged_lookup_parts.append(power_merged)
                        
                        # Other sectors → fuel price  
                        if not other_data.empty:
                            other_merged = other_data.merge(
                                lookup_table[['technology', 'year', 'iso2', 'stringency', 'fuel_price_usd_per_mwh']],
                                left_on=['technology', 'scenario_year', 'iso2', 'lookup_stringency'],
                                right_on=['technology', 'year', 'iso2', 'stringency'],
                                how='left'
                            )
                            other_merged[col] = other_merged['fuel_price_usd_per_mwh']
                            merged_lookup_parts.append(other_merged)
                        
                        # Combine results
                        if merged_lookup_parts:
                            merged_lookup = pd.concat(merged_lookup_parts, ignore_index=True)
                        else:
                            merged_lookup = pd.DataFrame()
                    else:
                        # BLAZING FAST: Direct vectorized merge with lookup table (normal columns)
                        merged_lookup = single_country_data.merge(
                            lookup_table[['technology', 'year', 'iso2', 'stringency', lookup_col]],
                            left_on=['technology', 'scenario_year', 'iso2', 'lookup_stringency'],
                            right_on=['technology', 'year', 'iso2', 'stringency'],
                            how='left'
                        )
                    
                    # COMPLETELY VECTORIZED: Store successful lookups in one operation
                    if col == 'scenario_price':
                        # For sector-aware scenario_price, check the filled column directly
                        successful_mask = merged_lookup[col].notna()
                        value_col = col
                    else:
                        # For normal columns, check the lookup column
                        if lookup_col and lookup_col in merged_lookup.columns:
                            successful_mask = merged_lookup[lookup_col].notna()
                            value_col = lookup_col
                        else:
                            # Debug information for troubleshooting
                            print(f"   ⚠️ Warning: lookup_col '{lookup_col}' not found in merged_lookup")
                            print(f"      Available columns in merged_lookup: {list(merged_lookup.columns)}")
                            print(f"      Merged lookup shape: {merged_lookup.shape}")
                            if lookup_col in lookup_table.columns:
                                non_null_count = lookup_table[lookup_col].notna().sum()
                                print(f"      '{lookup_col}' exists in lookup_table with {non_null_count} non-null values")
                            successful_mask = pd.Series(False, index=merged_lookup.index)
                            value_col = None
                        
                    if successful_mask.any() and value_col:
                        # Direct dictionary update with vectorized data
                        successful_data = merged_lookup[successful_mask]
                        fill_values.update(dict(zip(single_country_data.index[successful_mask], 
                                                  successful_data[value_col].values)))
                
                # VECTORIZED: Handle multi-country regions  
                multi_country_mask = remaining_data['iso2_count'] > 1
                multi_country_data = remaining_data[multi_country_mask]
                
                for idx, row in multi_country_data.iterrows():
                    if idx in fill_values:
                        continue
                        
                    tech = row['technology']
                    year = row['scenario_year']
                    stringency = row['lookup_stringency']
                    iso2_list = row['iso2_codes']
                    
                    # Get values for all constituent countries  
                    if col == 'scenario_price':
                        # For scenario_price, we need sector-aware lookup
                        sector = row['sector']
                        if sector in ['Power', 'Renewables']:
                            price_col = 'electricity_price_usd_per_mwh'
                        else:
                            price_col = 'fuel_price_usd_per_mwh'
                        
                        country_matches = lookup_table[
                            (lookup_table['technology'] == tech) &
                            (lookup_table['year'] == year) &
                            (lookup_table['iso2'].isin(iso2_list)) &
                            (lookup_table['stringency'] == stringency) &
                            (lookup_table[price_col].notna())
                        ][price_col]
                    else:
                        country_matches = lookup_table[
                            (lookup_table['technology'] == tech) &
                            (lookup_table['year'] == year) &
                            (lookup_table['iso2'].isin(iso2_list)) &
                            (lookup_table['stringency'] == stringency) &
                            (lookup_table[lookup_col].notna())
                        ][lookup_col]
                    
                    if not country_matches.empty:
                        fill_values[idx] = country_matches.median()
                
                # Fallback: geography-agnostic lookup using vectorized merge
                unfilled_indices = [idx for idx in remaining_data.index if idx not in fill_values]
                if unfilled_indices:
                    unfilled_data = remaining_data.loc[unfilled_indices]
                    
                    # Group lookup table by technology, year, stringency and take median across geographies
                    if col == 'scenario_price':
                        # For scenario_price, create separate lookups for each sector type
                        power_unfilled = unfilled_data[unfilled_data['sector'].isin(['Power', 'Renewables'])]
                        other_unfilled = unfilled_data[~unfilled_data['sector'].isin(['Power', 'Renewables'])]
                        
                        # Power/Renewables fallback
                        if not power_unfilled.empty:
                            power_lookup = lookup_table.groupby(['technology', 'year', 'stringency'])['electricity_price_usd_per_mwh'].median().reset_index()
                            power_merged = power_unfilled.merge(
                                power_lookup,
                                left_on=['technology', 'scenario_year', 'lookup_stringency'],
                                right_on=['technology', 'year', 'stringency'],
                                how='left'
                            )
                            power_successful = power_merged['electricity_price_usd_per_mwh'].notna()
                            for idx, value in zip(power_unfilled.index[power_successful], power_merged['electricity_price_usd_per_mwh'][power_successful]):
                                fill_values[idx] = value
                        
                        # Other sectors fallback
                        if not other_unfilled.empty:
                            other_lookup = lookup_table.groupby(['technology', 'year', 'stringency'])['fuel_price_usd_per_mwh'].median().reset_index()
                            other_merged = other_unfilled.merge(
                                other_lookup,
                                left_on=['technology', 'scenario_year', 'lookup_stringency'],
                                right_on=['technology', 'year', 'stringency'],
                                how='left'
                            )
                            other_successful = other_merged['fuel_price_usd_per_mwh'].notna()
                            for idx, value in zip(other_unfilled.index[other_successful], other_merged['fuel_price_usd_per_mwh'][other_successful]):
                                fill_values[idx] = value
                    else:
                        geo_agnostic_lookup = lookup_table.groupby(['technology', 'year', 'stringency'])[lookup_col].median().reset_index()
                        
                        fallback_merged = unfilled_data.merge(
                            geo_agnostic_lookup,
                            left_on=['technology', 'scenario_year', 'lookup_stringency'],
                            right_on=['technology', 'year', 'stringency'],
                            how='left'
                        )
                        
                        # Store fallback values
                        fallback_successful = fallback_merged[lookup_col].notna()
                        for idx, value in zip(unfilled_data.index[fallback_successful], fallback_merged[lookup_col][fallback_successful]):
                            fill_values[idx] = value
        
    # Apply all fill values at once (vectorized)
    if fill_values:
        fill_indices = list(fill_values.keys())
        fill_vals = list(fill_values.values())
        
        df.loc[fill_indices, col] = fill_vals
        filled_count = len(fill_values)
        
        # Update tracking (vectorized)
        for idx in fill_indices:
            current_tracking = df.loc[idx, "gap_filled_columns"]
            if pd.isna(current_tracking) or current_tracking == "":
                df.loc[idx, "gap_filled_columns"] = col
            else:
                df.loc[idx, "gap_filled_columns"] = str(current_tracking) + "," + col
    
    total_filled_by_col[col] = filled_count
    final_missing = df[col].isna().sum()
    print(f"      Filled {filled_count:,} values, {final_missing:,} still missing")
    
    # Summary
    total_filled_rows = (df["gap_filled_columns"] != "").sum()
    print(f"   📊 Gap-filling complete: {total_filled_rows:,} rows received gap-filled values")
    
    if total_filled_by_col:
        print(f"   📊 Columns filled breakdown:")
        for col_name, count in total_filled_by_col.items():
            print(f"     {col_name}: {count:,} rows")

    return df


def step4_aggregate_and_gapfill() -> None:
    print_banner("STEP 4 — Aggregate Technologies and Gap-Fill")
    df = pd.read_csv("3_final_AR6_target_schema.csv")

    mapping_df = None  # Not used when mapping is disabled

    # Temporal interpolation BEFORE aggregation and gap-filling
    # Note: This extends data back to 2023 even though Step 1 filters 2023 < year <= 2050
    # This ensures complete coverage for scenarios missing intermediate years
    print_banner("STEP 4a — Temporal Interpolation")
    
    # Check if we have a pre-computed lookup table that already covers all years
    try:
        lookup_check = pd.read_csv("technology_lookup_table.csv")
        lookup_years = sorted(lookup_check['year'].unique())
        print(f"🚀 Found pre-computed lookup table with years {min(lookup_years)}-{max(lookup_years)}")
        print("   ⚡ Skipping temporal interpolation - lookup table already interpolated!")
        print("   📊 This avoids double interpolation and improves performance")
    except FileNotFoundError:
        print("📋 No pre-computed lookup table found, performing temporal interpolation...")
    df = temporal_interpolation(df, start_year=2023, end_year=2050)

    # Use regular pandas (already converted at top of pipeline)

    # Fix column types before processing to avoid mixed type issues
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

    # Skip expensive copy operation - use df directly for aggregation
    print("⚡ Proceeding with aggregation (skipping expensive copy)...")
    df_map = df  # Use original data without any technology mapping
    # Grouping keys
    grouping_cols = [
        c
        for c in [
            "scenario_provider",
            "scenario",
            "scenario_type",
            "scenario_geography",
            "sector",
            "technology",
            "scenario_year",
        ]
        if c in df_map.columns
    ]

    # Define aggregation spec: sums and weighted averages
    sum_cols = [
        c
        for c in ["scenario_pathway", "capacity_additions_mw_per_yr"]
        if c in df_map.columns
    ]
    avg_cols = [
        c
        for c in [
            "scenario_price",
            "fuel_price",
            "scenario_capacity_factor",
            "lifetime_years",
            "efficiency_decimal",
            "om_cost_usd_per_mw_per_yr",
            "capital_cost_usd_per_mw",
            "carbon_price_usd_per_tco2",
        ]
        if c in df_map.columns
    ]

    # Weighted average helper
    def weighted_avg(group: pd.DataFrame, col: str, weight_col: str) -> float:
        weights = group[weight_col].fillna(1.0)
        vals = group[col]
        denom = weights.sum()
        if denom == 0 or vals.isna().all():
            return vals.mean()
        return (vals * weights).sum() / denom

    # VECTORIZED AGGREGATION: Much faster than .apply() on large datasets
    print(f"⚡ Starting vectorized aggregation on {len(df_map):,} rows...")
    
    # Build aggregation dictionary for pandas agg()
    agg_dict = {}
    
    # Add sums
    for col in sum_cols:
        agg_dict[col] = 'sum'
    
    # Add averages (weighted avg not directly supported, use mean for now)
    for col in avg_cols:
        agg_dict[col] = 'mean'
    
    # Add first() for carry-forward columns
    carry_cols = [
        c for c in df_map.columns 
        if c not in set(sum_cols + avg_cols + grouping_cols)
        and c not in ["target_sector", "target_technology", "aggregation_group"]
    ]
    for col in carry_cols:
        agg_dict[col] = 'first'
    
    print(f"   📊 Aggregating {len(agg_dict)} columns across {df_map.groupby(grouping_cols).ngroups:,} groups...")
    
    # Perform vectorized aggregation (much faster than apply!)
    aggregated = df_map.groupby(grouping_cols, as_index=False).agg(agg_dict)
    
    print(f"✅ Vectorized aggregation complete: {len(aggregated):,} rows")

    # Initialize gap-filled tracking column
    if "gap_filled_columns" not in aggregated.columns:
        aggregated["gap_filled_columns"] = ""
    else:
        # Ensure it's properly initialized as string
        aggregated["gap_filled_columns"] = (
            aggregated["gap_filled_columns"].fillna("").astype(str)
        )

    # Create stringency BEFORE gap-filling so it can be used in hierarchy
    # Map scenario_type to stringency, with UNKNOWN as default
    stringency_map = {
        'C1': 'C1', 'C2': 'C2', 'C3': 'C3', 'C4': 'C4', 
        'C5': 'C5', 'C6': 'C6', 'C7': 'C7', 'C8': 'C8'
    }
    aggregated["stringency"] = aggregated["scenario_type"].map(stringency_map).fillna("UNKNOWN")

    # Clean up zero values that should be treated as missing data
    print("🧹 Cleaning zero values that should be treated as missing...")
    zero_to_na_columns = ["lifetime_years", "efficiency_decimal"]

    for col in zero_to_na_columns:
        if col in aggregated.columns:
            zero_count = (aggregated[col] == 0).sum()
            if zero_count > 0:
                print(f"   {col}: converting {zero_count:,} zero values to NA")
                aggregated[col] = aggregated[col].replace(0, np.nan)
            else:
                print(f"   {col}: no zero values found")

    # Define standard hierarchical gap-filling order for consistent logic
    # This follows a systematic approach from most specific to most general
    # Now includes Global geography fallbacks
    standard_hierarchy = [
        [
            "scenario_year",
            "technology",
            "scenario_geography",
            "scenario_type",
            "stringency",
        ],  # Most specific
        ["scenario_year", "technology", "scenario_type", "stringency"],  # Same policy
        [
            "scenario_year",
            "technology",
            "scenario_geography",
            "scenario_type",
        ],  # Same type, cross-stringency
        [
            "scenario_year",
            "technology",
            "scenario_geography",
            "stringency",
        ],  # Same stringency, cross-type
        ["scenario_year", "technology", "scenario_type"],  # Cross-geo/stringency
        ["scenario_year", "technology", "stringency"],  # Cross-geo/type
        ["scenario_year", "technology", "scenario_geography"],  # Cross-policy
        ["scenario_year", "technology"],  # Same tech only
        [
            "technology",
            "scenario_geography",
            "scenario_type",
            "stringency",
        ],  # Cross-time
        ["technology", "scenario_geography"],  # Cross-time/policy
        ["technology"],  # Most general
    ]

    # Add Global geography-specific fallback levels for better gap-filling
    # These levels specifically use Global geography as a fallback when regional data is missing
    global_fallback_hierarchy = [
        ["scenario_year", "technology", "scenario_type", "stringency"],  # Use Global for same policy
        ["scenario_year", "technology", "scenario_type"],  # Use Global for same type
        ["scenario_year", "technology", "stringency"],  # Use Global for same stringency
        ["scenario_year", "technology"],  # Use Global for same tech/year
        ["technology", "scenario_type", "stringency"],  # Use Global cross-time
        ["technology", "scenario_type"],  # Use Global cross-time/stringency
        ["technology"],  # Use Global most general
    ]

    # Apply gap-filling strategies based on data source
    gap_spec: Dict[str, List[List[str]]] = {}

    # 🚀 ALL COLUMNS NOW USE LOOKUP TABLE! (80%+ coverage, no hierarchical gap-filling needed)
    # Load the pre-computed lookup table
    try:
        lookup_table = pd.read_csv("technology_lookup_table.csv")
        print(f"🚀 Loaded pre-computed lookup table: {lookup_table.shape[0]:,} rows × {lookup_table.shape[1]} columns")
    except FileNotFoundError:
        print("❌ ERROR: technology_lookup_table.csv not found!")
        print("   Please run: python3 create_technology_lookup.py")
        raise
        
    # Check what's actually available in the comprehensive lookup table
    available_lookup_cols = [col for col in lookup_table.columns if col not in ['technology', 'year', 'iso2', 'stringency']]
    print(f"   📊 Available lookup columns: {available_lookup_cols}")
    
    # Use ALL available columns from the comprehensive lookup table
    all_possible_cols = [
        "lifetime_years", "om_cost_usd_per_mw_per_yr", "capital_cost_usd_per_mw", 
        "efficiency_decimal", "fuel_price_usd_per_mwh", "electricity_price_usd_per_mwh"
    ]
    
    lookup_columns = []
    for col in all_possible_cols:
        if col in available_lookup_cols:
            lookup_columns.append(col)
            print(f"   ✅ Including {col} from lookup table")
        else:
            print(f"   ⚠️ {col} not in lookup table, will skip")
    
    print(f"   🚀 Using lookup table for ALL {len(lookup_columns)} columns: {lookup_columns}")
    print(f"   🎉 HIERARCHICAL GAP-FILLING COMPLETELY ELIMINATED!")

    # Map pipeline columns to lookup columns and use lookup table ONLY
    pipeline_to_lookup_mapping = {
        'lifetime_years': 'lifetime_years',
        'om_cost_usd_per_mw_per_yr': 'om_cost_usd_per_mw_per_yr', 
        'capital_cost_usd_per_mw': 'capital_cost_usd_per_mw',
        'efficiency_decimal': 'efficiency_decimal',
        'fuel_price': 'fuel_price_usd_per_mwh',  # Always fuel-specific price
        # scenario_price is SECTOR-AWARE and handled separately below
    }
    
    for pipeline_col, lookup_col in pipeline_to_lookup_mapping.items():
        if pipeline_col in aggregated.columns and lookup_col in available_lookup_cols:
            gap_spec[pipeline_col] = []  # Empty hierarchy = lookup table only
            print(f"   📋 {pipeline_col} → {lookup_col} (lookup table only)")
    
    # 🎯 SECTOR-AWARE scenario_price mapping:
    # Power/Renewables → electricity_price_usd_per_mwh (they sell electricity)
    # Other sectors → fuel_price_usd_per_mwh (they sell the fuel)
    if 'scenario_price' in aggregated.columns:
        gap_spec['scenario_price'] = []  # Use lookup table only
        print(f"   📋 scenario_price → SECTOR-AWARE (Power=electricity, Others=fuel) (lookup table only)")
    
    # Note: scenario_capacity_factor removed - use only if value exists, no gap-filling

    agg_fn = {"fuel_price": "mean"}  # use mean for fuel cascade; median elsewhere
    aggregated = gap_fill_with_lookup_table(aggregated, gap_spec, global_fallback_hierarchy, agg_fn)

    # Scenario type update based on stringency values
    if "scenario_type" in aggregated.columns:
        aggregated["scenario_type"] = np.where(
            aggregated["stringency"].isin(["C1", "C2"]), "target", "baseline"
        )

    # Write main aggregated output with columns ordering similar to original
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
    extra_cols = [c for c in aggregated.columns if c not in base_cols]
    final_cols = [c for c in base_cols + extra_cols if c in aggregated.columns]
    out_file = "4_final_AR6_aggregated.csv"
    aggregated[final_cols].to_csv(out_file, index=False)
    print(f"✅ Wrote {out_file} | Shape: {aggregated[final_cols].shape}")

    # Complete-case filtering
    critical = [
        c
        for c in [
            "scenario_pathway",
            "scenario_price",
            "om_cost_usd_per_mw_per_yr",
            "capital_cost_usd_per_mw",
        ]
        if c in aggregated.columns
    ]
    complete_mask = (
        aggregated[critical].notna().all(axis=1)
        if critical
        else pd.Series(True, index=aggregated.index)
    )
    if "efficiency_decimal" in aggregated.columns:
        # Define exact renewable technology names that should have efficiency data
        renewable_techs = [
            "SolarCap", "WindCap", "HydroCap", "GeothermalCap", "NuclearCap", "OceanCap",
            "Non-Biomass Renewables", "Electricity - Non-Biomass Renewables"
        ]
        
        # Create mask for renewable technologies using exact matching
        is_renewable = aggregated["technology"].isin(renewable_techs)
        
        # Efficiency condition: renewable technologies OR legacy Renewables sector should have efficiency
        eff_cond = ~(
            (aggregated["sector"].isin(["Power", "Renewables"]) | is_renewable)
            & aggregated["efficiency_decimal"].isna()
        )
        complete_mask = complete_mask & eff_cond

    final_complete = aggregated.loc[complete_mask, final_cols].copy()
    complete_file = "4_final_AR6_aggregated_complete.csv"
    final_complete.to_csv(complete_file, index=False)
    print(f"✅ Wrote {complete_file} | Shape: {final_complete.shape}")

    memory_release(df, mapping_df, df_map, aggregated, final_complete)


# =====
# main
# =====


def main() -> None:
    print_banner("AR6 Combined Pipeline — Start")
    # Step 1: create intermediate ISO3/R10 files; each step frees memory before the next
    step1_run()
    # Step 2
    step2_filter_and_pivot()
    # Step 3
    step3_finalize_target_schema()
    # Step 4
    step4_aggregate_and_gapfill()
    print_banner("AR6 Combined Pipeline — Done")


if __name__ == "__main__":
    main()
