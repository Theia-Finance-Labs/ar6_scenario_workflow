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
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import external step modules
from step4_gapfill_simple import step4_gapfill_only
from step5_complete_cases import step5_complete_cases
from step6_scenario_tech_filter import step6_scenario_tech_filter

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
        "data/AR6_Scenarios_Database_ISO3_v1.1.feather"
        if dataset_type == "ISO3"
        else "data/AR6_Scenarios_Database_R10_regions_v1.1.feather"
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
        target_models = ["WITCH 5.0"]  # , "IMAGE 3.2"]
        target_scenarios = ["EN_NPi2020_500", "CO_CurPol"]
        print(f"   Before model filter: {source.shape[0]:,} rows")
        #source = source[
        #    source["Model"].isin(target_models)
        #    & source["Scenario"].isin(target_scenarios)
        #]
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
        "Economy",
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
        "Fuel",
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

    # Define renewable technologies (now in Power sector, not Renewables sector)
    renewable_tech_keywords = [
        "Solar",
        "Wind",
        "Hydro",
        "Geothermal",
        "Nuclear",
        "Non-Biomass Renewables",
        "Electricity - Non-Biomass Renewables",
    ]

    # Create mask for renewable technologies based on technology name
    tech_col = "Technology" if "Technology" in pivoted.columns else "technology"
    renewable_mask = (
        pivoted[tech_col]
        .astype(str)
        .apply(lambda x: any(keyword in x for keyword in renewable_tech_keywords))
    )

    # Also include old-style renewables sector if it exists
    if "Sector" in pivoted.columns:
        old_renewables_mask = pivoted["Sector"] == "Renewables"
        renewable_mask = renewable_mask | old_renewables_mask

    print(
        f"Setting efficiency to 1.0 for {renewable_mask.sum()} renewable technology entries"
    )
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
    s = str(label).strip().lower()
    if not s or s == "nan":
        return ""
    if "storage" in s:
        return "Electricity"  # Storage technologies use electricity
    if "electric" in s:
        return "Electricity"
    if "gas" in s or "gases" in s:
        return "Gas"
    if "oil" in s or "liquid" in s or "liquids" in s:
        return "Oil"
    if "coal" in s or "solid" in s or "solids" in s:
        return "Coal"
    if "biomass" in s or s.startswith("bio"):
        return "Biomass"
    if "nuclear" in s:
        return "Nuclear"
    if "hydro" in s:
        return "Hydro"
    if "solar" in s:
        return "Solar"
    if "wind" in s:
        return "Wind"
    if "geothermal" in s:
        return "Geothermal"
    return s.title()


def build_price_tables(step1_df: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Build structured price tables from raw AR6 price data.
    
    Returns 4 price tables:
    - elec: Electricity prices (secondary energy) → Power sector revenue
    - primary: General primary energy prices → Currently unused
    - primary_by_fuel: Primary energy by fuel type → All sectors' input costs
    - secondary_by_fuel: Secondary energy by fuel type → Non-Power sectors' revenue
    
    Economic interpretation:
    - Primary prices: Raw material costs (coal, gas, oil, biomass)
    - Secondary prices: Processed product prices (electricity, refined fuels)
    """
    if step1_df is None:
        return {
            "elec": pd.DataFrame(),
            "primary": pd.DataFrame(),
            "primary_by_fuel": pd.DataFrame(),
            "secondary_by_fuel": pd.DataFrame(),
        }

    prices = step1_df[step1_df["col1"] == "Price"].copy()
    if prices.empty:
        return {
            "elec": pd.DataFrame(),
            "primary": pd.DataFrame(),
            "primary_by_fuel": pd.DataFrame(),
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

    # Primary energy by fuel (fuel-specific table for fuel_price)
    primary_by_fuel = prices[(prices["col2"] == "Primary Energy")][
        ["model", "scenario", "region", "year", "Fuel_norm", "price_usd_per_mwh"]
    ].copy()
    primary_by_fuel = primary_by_fuel.groupby(
        ["model", "scenario", "region", "year", "Fuel_norm"], as_index=False
    )["price_usd_per_mwh"].first()

    # Secondary energy by fuel (fuel-specific table)
    secondary_by_fuel = prices[(prices["col2"] == "Secondary Energy")][
        ["model", "scenario", "region", "year", "Fuel_norm", "price_usd_per_mwh"]
    ].copy()
    secondary_by_fuel = secondary_by_fuel.groupby(
        ["model", "scenario", "region", "year", "Fuel_norm"], as_index=False
    )["price_usd_per_mwh"].first()

    return {"elec": elec, "primary": primary, "primary_by_fuel": primary_by_fuel, "secondary_by_fuel": secondary_by_fuel}


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
    target["Fuel"] = df["Fuel"]

    # Load metadata for scenario_type and stringency mapping
    target["scenario_type"] = "target"  # Default scenario type
    target["stringency"] = "UNKNOWN"    # Default stringency
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
            target["stringency"] = key.map(lookup).fillna("UNKNOWN")
            print(f"✅ Mapped stringency for {(target['stringency'] != 'UNKNOWN').sum():,} scenarios")
    except FileNotFoundError:
        print("⚠️ Metadata Excel not found. stringency set to UNKNOWN.")

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

    # ========================================================================
    # PRICING LOGIC OVERVIEW
    # ========================================================================
    # This section assigns two key price fields to each technology:
    #
    # 1. scenario_price (REVENUE - what firms sell their output for):
    #    - Power/Renewables: electricity prices (secondary energy)
    #    - Coal/Gas&Oil: processed fuel prices (secondary energy by fuel type)
    #
    # 2. fuel_price (COSTS - what firms pay for input fuels):
    #    - All sectors: raw material prices (primary energy by fuel type)
    #    - Renewables: 0 (no fuel consumption)
    #
    # Economic Logic:
    # - Revenue > Costs creates realistic profit margins
    # - Secondary prices > Primary prices reflects value-added processing
    # ========================================================================

    # Load price data from step1 and create structured price tables
    step1_df = load_step1_for_price()
    price_tables = build_price_tables(step1_df)

    # Build keys for joins across all price merges
    join_key = ["scenario_provider", "scenario", "scenario_geography", "scenario_year"]
    
    # ========================================================================
    # PREPARE PRICE TABLES WITH CONSISTENT COLUMN NAMES
    # ========================================================================
    # Table 1: Electricity prices (secondary energy) for Power sector revenue
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

    # Table 2: Primary energy prices (general) - currently unused but kept for compatibility
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

    # Table 3: Primary energy prices by fuel (for fuel_price - input costs)
    primary_by_fuel = (
        price_tables["primary_by_fuel"].rename(
            columns={
                "model": "scenario_provider",
                "scenario": "scenario",
                "region": "scenario_geography",
                "year": "scenario_year",
                "Fuel_norm": "fuel_for_price_norm",
                "price_usd_per_mwh": "fuel_price",
            }
        )
        if not price_tables["primary_by_fuel"].empty
        else pd.DataFrame(columns=join_key + ["fuel_for_price_norm", "fuel_price"])
    )

    # ========================================================================
    # ASSIGN SCENARIO_PRICE (REVENUE) BY SECTOR
    # ========================================================================
    # Power/Renewables: Get electricity prices (what they sell)
    # Coal/Gas&Oil: Get processed fuel prices (what they sell)
    
    # Step 1: Merge electricity prices for Power sector
    target = target.merge(elec, on=join_key, how="left")
    
    # Step 2: Prepare secondary energy prices by fuel for non-Power sectors
    # Table 4: Secondary energy prices by fuel (for non-Power sector revenue)
    sec_by_fuel_for_scenario = (
        price_tables["secondary_by_fuel"].rename(
            columns={
                "model": "scenario_provider",
                "scenario": "scenario",
                "region": "scenario_geography",
                "year": "scenario_year",
                "Fuel_norm": "fuel_for_scenario_norm",
                "price_usd_per_mwh": "scenario_price_secondary",
            }
        )
        if not price_tables["secondary_by_fuel"].empty
        else pd.DataFrame(columns=join_key + ["fuel_for_scenario_norm", "scenario_price_secondary"])
    )
    
    # Step 3: Merge secondary energy prices by fuel type for Coal/Gas&Oil sectors
    target["fuel_for_scenario_norm"] = target["Fuel"].apply(_normalize_fuel_label)
    target = target.merge(
        sec_by_fuel_for_scenario, 
        on=join_key + ["fuel_for_scenario_norm"], 
        how="left"
    )
    target = target.drop(columns=["fuel_for_scenario_norm"], errors="ignore")
    
    # Step 4: Assign scenario_price based on sector type
    is_power_like = (
        target["sector"].isin(["Power", "Renewables"])
        if "sector" in target.columns
        else pd.Series(False, index=target.index)
    )
    target["scenario_price"] = np.where(
        is_power_like,
        target["scenario_price_electricity"],    # Power: electricity prices
        target["scenario_price_secondary"],      # Others: processed fuel prices
    )
    
    # Clean up temporary columns
    target = target.drop(
        columns=["scenario_price_electricity", "scenario_price_secondary"],
        errors="ignore",
    )

    # ========================================================================
    # ASSIGN FUEL_PRICE (INPUT COSTS) FOR ALL SECTORS
    # ========================================================================
    # All sectors get primary energy prices (raw material costs)
    # Renewables get fuel_price = 0 (no fuel consumption)
    
    # Step 1: Map technology fuel types to primary energy prices
    target["fuel_for_price"] = target["Fuel"]
    target["fuel_for_price_norm"] = target["fuel_for_price"].apply(_normalize_fuel_label)
    
    # Step 2: Merge primary energy prices by fuel type (input costs for all sectors)
    target = target.merge(
        primary_by_fuel, on=join_key + ["fuel_for_price_norm"], how="left"
    )
    target = target.drop(columns=["fuel_for_price_norm"], errors="ignore")

    # Step 3: Override fuel_price to 0 for renewable technologies (no fuel consumption)
    renewable_tech_keywords = [
        "Solar",
        "Wind", 
        "Hydro",
        "Geothermal",
        "Nuclear",
        "Non-Biomass Renewables",
        "Electricity - Non-Biomass Renewables",
    ]

    is_renewable_tech = (
        target["technology"]
        .astype(str)
        .apply(lambda x: any(keyword in x for keyword in renewable_tech_keywords))
    )

    target.loc[is_renewable_tech, "fuel_price"] = 0.0
    print(
        f"Set fuel_price=0 for {is_renewable_tech.sum()} renewable technology entries"
    )
    
    # ========================================================================
    # PRICING ASSIGNMENT COMPLETE
    # ========================================================================
    # Final result:
    # - scenario_price: Revenue from selling output (secondary energy prices)
    # - fuel_price: Cost of input fuels (primary energy prices, 0 for renewables)
    # - Economic margin: scenario_price - fuel_price = processing value added
    # ========================================================================

    # ========================================================================
    # CALCULATE FUEL INTENSITY
    # ========================================================================
    # Fuel intensity = Primary Energy / Secondary Energy
    # Represents conversion efficiency from primary fuel to secondary output
    # Example: GasCap fuel_intensity = Primary Energy|Gas|Electricity / Secondary Energy|Electricity|Gas
    
    print("🔥 Calculating fuel intensity (Primary Energy / Secondary Energy)...")
    
    # Initialize fuel_intensity column
    target["fuel_intensity"] = np.nan
    
    # Calculate fuel intensity where both primary and secondary energy data exist
    if "primary_energy_mwh_per_yr" in df.columns and "secondary_energy_mwh_per_yr" in df.columns:
        # Create mask for valid calculations (both values > 0)
        valid_mask = (
            df["primary_energy_mwh_per_yr"].notna() & 
            df["secondary_energy_mwh_per_yr"].notna() &
            (df["primary_energy_mwh_per_yr"] > 0) & 
            (df["secondary_energy_mwh_per_yr"] > 0)
        )
        
        if len(target) == len(df) and valid_mask.any():
            # Calculate fuel intensity: Primary Energy / Secondary Energy
            target.loc[valid_mask, "fuel_intensity"] = (
                df.loc[valid_mask, "primary_energy_mwh_per_yr"] / 
                df.loc[valid_mask, "secondary_energy_mwh_per_yr"]
            )
            
            calculated_count = valid_mask.sum()
            print(f"   ✅ Calculated fuel_intensity for {calculated_count:,} entries")
            
            # Show statistics
            fuel_intensity_values = target.loc[valid_mask, "fuel_intensity"]
            print(f"   📊 Fuel Intensity Statistics:")
            print(f"      Mean: {fuel_intensity_values.mean():.3f}")
            print(f"      Median: {fuel_intensity_values.median():.3f}")
            print(f"      Min: {fuel_intensity_values.min():.3f}")
            print(f"      Max: {fuel_intensity_values.max():.3f}")
            
            # Set fuel_intensity to 1.0 for renewable technologies (no fuel conversion loss)
            renewable_fuel_intensity_mask = is_renewable_tech & valid_mask
            if renewable_fuel_intensity_mask.any():
                target.loc[renewable_fuel_intensity_mask, "fuel_intensity"] = 1.0
                renewable_count = renewable_fuel_intensity_mask.sum()
                print(f"   🌱 Set fuel_intensity=1.0 for {renewable_count:,} renewable entries (no conversion loss)")
        else:
            print("   ⚠️  No valid primary/secondary energy data for fuel intensity calculation")
    else:
        print("   ⚠️  Primary or secondary energy columns not found")
    
    # ========================================================================
    # FUEL INTENSITY CALCULATION COMPLETE
    # ========================================================================
    # fuel_intensity represents the ratio of primary to secondary energy:
    # - Values > 1: Energy loss during conversion (typical for thermal plants)
    # - Values = 1: No conversion loss (renewables, direct conversion)
    # - Values < 1: Theoretical efficiency gain (rare, may indicate data issues)
    # ========================================================================

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
        use_secondary = sec_series.isin(["Power", "Renewables"])
        scenario_pathway = np.where(
            use_secondary, df["secondary_energy_mwh_per_yr"], scenario_pathway
        )
    if "capacity_mw" in df.columns:
        use_capacity = sec_series.isin(["Power", "Renewables"]) & has_capacity
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

    # CRITICAL: Apply extreme value filtering after unit conversions and before save
    print_banner("STEP 3b — Extreme Value Filtering")
    print("🚫 Applying extreme value filtering to remove unrealistic values...")
    
    # Define bounds for extreme value filtering (user-specified)
    bounds = {
        'efficiency_decimal': (0.2, 1.0),
        'scenario_price': (0, 200),
        'fuel_price': (0, 200),
        'capital_cost_usd_per_mw': (0, 1e7),
        'om_cost_usd_per_mw_per_yr': (0, 5e5)
    }
    
    # Count violations before filtering
    violations_before = 0
    for col, (min_val, max_val) in bounds.items():
        if col in target_with_global.columns:
            violations = ((target_with_global[col] < min_val) | (target_with_global[col] > max_val)).sum()
            violations_before += violations
            if violations > 0:
                print(f"   {col}: {violations:,} values outside bounds ({min_val}-{max_val})")
    
    print(f"   Total violations before filtering: {violations_before:,}")
    
    # Apply filtering by setting out-of-bounds values to NaN
    filtered_count = 0
    for col, (min_val, max_val) in bounds.items():
        if col in target_with_global.columns:
            # Create mask for out-of-bounds values
            out_of_bounds = (target_with_global[col] < min_val) | (target_with_global[col] > max_val)
            count_filtered = out_of_bounds.sum()
            
            if count_filtered > 0:
                target_with_global.loc[out_of_bounds, col] = np.nan
                filtered_count += count_filtered
                print(f"   ✅ Filtered {count_filtered:,} out-of-bounds values for {col}")
    
    if filtered_count > 0:
        print(f"✅ Total extreme values filtered: {filtered_count:,}")
    else:
        print("✅ No extreme values found to filter")

    # Save and cleanup
    out = "3_final_AR6_target_schema.csv"
    cols = [
        "scenario_provider",
        "scenario",
        "scenario_type",
        "stringency",
        "scenario_geography",
        "sector",
        "technology",
        "technology_type",
        "price_unit",
        "price_indicator",
        "scenario_price",
        "fuel_price",
        "fuel_for_price",
        "fuel_intensity",
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
        "scenario_year",
    ]
    grouping_cols = [c for c in grouping_cols if c in df.columns]

    # Define aggregation strategies
    sum_cols = [
        "scenario_pathway",  # Sum capacities/energy
        "capacity_additions_mw_per_yr",
    ]
    sum_cols = [c for c in sum_cols if c in df.columns]

    # Weighted average by scenario_pathway for these columns
    weighted_avg_cols = [
        "scenario_capacity_factor",
        "efficiency_decimal",
        "lifetime_years",
        "om_cost_usd_per_mw_per_yr",
        "capital_cost_usd_per_mw",
    ]
    weighted_avg_cols = [c for c in weighted_avg_cols if c in df.columns]

    # Simple average for price columns (not capacity-weighted)
    simple_avg_cols = ["scenario_price", "fuel_price", "carbon_price_usd_per_tco2"]
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
                    result[col] = (
                        values[valid_mask] * weights[valid_mask]
                    ).sum() / weights[valid_mask].sum()
                else:
                    result[col] = values.mean()  # Fallback to simple mean
            else:
                result[col] = group[col].mean()

        # Simple averages for prices
        for col in simple_avg_cols:
            result[col] = group[col].mean()

        # Copy metadata from first row (should be same across geographies for a scenario)
        metadata_cols = [
            c
            for c in group.columns
            if c not in sum_cols + weighted_avg_cols + simple_avg_cols + grouping_cols
            and c not in ["scenario_geography", "country_iso2_list"]
        ]
        first_row = group.iloc[0]
        for col in metadata_cols:
            result[col] = first_row[col]

        # Set geography to "Global" and empty country list
        result["scenario_geography"] = "Global"
        result["country_iso2_list"] = ""  # Global should have empty country list

        return pd.Series(result)

    # Group by scenario dimensions (excluding geography) and aggregate
    print(f"   Grouping by: {grouping_cols}")
    print(
        f"   Processing {len(df)} rows across {df['scenario_geography'].nunique()} geographies"
    )

    global_aggregated = df.groupby(grouping_cols, as_index=False).apply(
        aggregate_to_global, include_groups=False
    )

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

# Step 4 function is now imported from step4_aggregate_and_gapfill module


# =====
# main
# =====


def main() -> None:
    print_banner("AR6 Combined Pipeline — Start")
    # Step 1: create intermediate ISO3/R10 files; each step frees memory before the next
    step1_run()
    # Step 2
    step2_filter_and_pivot()
    # Step 3 (now includes stringency mapping)
    step3_finalize_target_schema()
    # Step 4 (uses stringency for gap-filling)
    step4_gapfill_only()
    # Step 5 (complete cases filtering)
    step5_complete_cases()
    # Step 6 (scenario technology filtering)
    step6_scenario_tech_filter()
    print_banner("AR6 Combined Pipeline — Done")


if __name__ == "__main__":
    main()
