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

# Import external step 4 module
from step4_aggregate_and_gapfill import step4_aggregate_and_gapfill

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
        target_models = ["WITCH 5.0"]  # , "IMAGE 3.2"]
        target_scenarios = ["EN_NPi2020_500", "CO_CurPol"]
        print(f"   Before model filter: {source.shape[0]:,} rows")
        source = source[
            source["Model"].isin(target_models)
            & source["Scenario"].isin(target_scenarios)
        ]
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

    # Set fuel price to 0 for renewables (free fuel)
    target.loc[is_renewable_tech, "fuel_price"] = 0.0
    print(
        f"Set fuel_price=0 for {is_renewable_tech.sum()} renewable technology entries"
    )

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

    # Add debugging for the groupby operation
    try:
        grouped = df_for_groupby.groupby(grouping_cols, dropna=False)
        print(f"  DEBUG: Created groupby object with {grouped.ngroups} groups")

        for group_key, group_df in grouped:
            total_groups += 1

            if total_groups <= 3:  # Debug first few groups
                print(f"  DEBUG: Processing group {total_groups}: {group_key}")
                print(f"  DEBUG: Group has {len(group_df)} rows")

            if group_df.empty:
                print(f"  DEBUG: Skipping empty group {total_groups}")
                continue

            # Get available years and sort
            available_years = sorted(group_df["scenario_year"].dropna().unique())
            if not available_years:
                print(f"  DEBUG: Skipping group {total_groups} - no valid years")
                continue

            # Check if this group needs interpolation
            group_missing = expected_years - set(available_years)
            if not group_missing:
                # No missing years for this group, keep as-is
                print(
                    f"  DEBUG: Group {total_groups} complete, adding {len(group_df)} rows as-is"
                )
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

            # Merge with existing data - ensure both DataFrames are regular pandas
            merge_cols = ["scenario_year"] + list(group_meta.keys())

            # Continue with regular pandas DataFrames

            try:
                merged = all_years_df.merge(group_df, on=merge_cols, how="left")
            except Exception as e:
                print(f"  ERROR in merge for group {total_groups}: {e}")
                # Skip this group and continue
                continue

            # Fill categorical/string columns with values from the group (should be same for all years)
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
                    # Forward fill categorical values within this group
                    non_null_values = merged[col].dropna()
                    if len(non_null_values) > 0:
                        # Use the first non-null value for all rows in this group
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

    print(f"  DEBUG: Finished processing {total_groups} total groups")

    if not interpolated_groups:
        print("⚠️ No groups processed during temporal interpolation")
        print(
            f"  DEBUG: total_groups={total_groups}, processed_groups={processed_groups}"
        )
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


def build_iso2_geographic_similarity_matrix(
    df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Build a similarity matrix between geographies based on ISO2 country overlap.
    Returns nested dict: {geo1: {geo2: overlap_score, ...}, ...}
    """
    print("🗺️  Building ISO2-based geographic similarity matrix...")

    # Get unique geographies and their ISO2 lists
    geo_iso2_map = {}
    for geo in df["scenario_geography"].unique():
        iso2_str = df[df["scenario_geography"] == geo]["country_iso2_list"].iloc[0]
        if pd.notna(iso2_str) and iso2_str.strip():
            # Parse comma-separated ISO2 codes
            iso2_set = set(
                code.strip() for code in str(iso2_str).split(",") if code.strip()
            )
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
                similarity_matrix[geo1][geo2] = (
                    intersection / union if union > 0 else 0.0
                )

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
    Create a comprehensive lookup table with best available values for each
    technology/year/iso2/stringency combination using hierarchical gap-filling.

    Returns DataFrame with columns:
    technology, year, iso2, stringency, capital_cost, om_cost, efficiency, lifetime, capacity_factor
    """
    print("📋 Creating technology lookup table...")

    # Extract all unique combinations
    unique_techs = df["technology"].unique()
    unique_years = sorted(df["scenario_year"].unique())
    unique_stringencies = (
        sorted(df["stringency"].unique()) if "stringency" in df.columns else [""]
    )

    # Get all ISO2 codes from country_iso2_list
    all_iso2s = set()
    for iso2_list in df["country_iso2_list"].dropna():
        if str(iso2_list).strip():
            iso2s = [code.strip() for code in str(iso2_list).split(",") if code.strip()]
            all_iso2s.update(iso2s)
    all_iso2s = sorted(all_iso2s)

    print(
        f"   Building lookup for {len(unique_techs)} techs × {len(unique_years)} years × {len(all_iso2s)} ISO2s × {len(unique_stringencies)} stringencies"
    )

    # Columns to populate
    value_cols = [
        "capital_cost_usd_per_mw",
        "om_cost_usd_per_mw_per_yr",
        "efficiency_decimal",
        "lifetime_years",
        "scenario_capacity_factor",
    ]
    available_value_cols = [col for col in value_cols if col in df.columns]

    # Create all combinations
    lookup_rows = []

    for tech in unique_techs:
        for year in unique_years:
            for iso2 in all_iso2s:
                for stringency in unique_stringencies:
                    # Create base row
                    row = {
                        "technology": tech,
                        "year": year,
                        "iso2": iso2,
                        "stringency": stringency,
                    }

                    # Fill each value column using hierarchical approach
                    for col in available_value_cols:
                        row[
                            col.replace("_usd_per_mw_per_yr", "")
                            .replace("_usd_per_mw", "")
                            .replace("_decimal", "")
                            .replace("_years", "")
                            .replace("scenario_", "")
                        ] = None

                    lookup_rows.append(row)

    lookup_df = pd.DataFrame(lookup_rows)

    # Hierarchical gap-filling for lookup table
    print(f"   Applying hierarchical gap-filling to lookup table...")

    for col in available_value_cols:
        col_short = (
            col.replace("_usd_per_mw_per_yr", "")
            .replace("_usd_per_mw", "")
            .replace("_decimal", "")
            .replace("_years", "")
            .replace("scenario_", "")
        )
        print(f"     Processing {col_short}...")

        filled_count = 0

        # Hierarchy levels for lookup table
        hierarchy = [
            # Most specific: exact matches
            {"tech": True, "year": True, "iso2": True, "stringency": True},
            {
                "tech": True,
                "year": True,
                "iso2": True,
                "stringency": False,
            },  # Cross stringency
            {
                "tech": True,
                "year": True,
                "iso2": False,
                "stringency": True,
            },  # Geographic similarity
            {
                "tech": True,
                "year": True,
                "iso2": False,
                "stringency": False,
            },  # Cross geo+stringency
            {
                "tech": True,
                "year": False,
                "iso2": True,
                "stringency": True,
            },  # Cross year
            {
                "tech": True,
                "year": False,
                "iso2": True,
                "stringency": False,
            },  # Cross year+stringency
            {
                "tech": True,
                "year": False,
                "iso2": False,
                "stringency": True,
            },  # Cross year+geo
            {
                "tech": True,
                "year": False,
                "iso2": False,
                "stringency": False,
            },  # Tech only
        ]

        for level_idx, level in enumerate(hierarchy):
            missing_mask = lookup_df[col_short].isna()
            if not missing_mask.any():
                break

            level_fills = 0

            for idx in lookup_df.index[missing_mask]:
                target_tech = lookup_df.loc[idx, "technology"]
                target_year = lookup_df.loc[idx, "year"]
                target_iso2 = lookup_df.loc[idx, "iso2"]
                target_stringency = lookup_df.loc[idx, "stringency"]

                # Build query for this level
                query_mask = (
                    (df["technology"] == target_tech)
                    if level["tech"]
                    else pd.Series(True, index=df.index)
                )

                if level["year"]:
                    query_mask &= df["scenario_year"] == target_year

                if level["stringency"] and "stringency" in df.columns:
                    query_mask &= df["stringency"] == target_stringency

                if level["iso2"]:
                    # Exact ISO2 match
                    iso2_mask = df["country_iso2_list"].apply(
                        lambda x: (
                            target_iso2 in str(x).split(",") if pd.notna(x) else False
                        )
                    )
                    query_mask &= iso2_mask
                else:
                    # Geographic similarity - find geographies containing this ISO2
                    iso2_mask = df["country_iso2_list"].apply(
                        lambda x: (
                            target_iso2 in str(x).split(",") if pd.notna(x) else False
                        )
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
                base_mask = (
                    (lookup_df["technology"] == tech)
                    & (lookup_df["year"] == year)
                    & (lookup_df["iso2"] == iso2)
                )

                row = {
                    "technology": tech,
                    "year": year,
                    "iso2": iso2,
                    "stringency": None,  # NA stringency
                }

                for col in available_value_cols:
                    col_short = (
                        col.replace("_usd_per_mw_per_yr", "")
                        .replace("_usd_per_mw", "")
                        .replace("_decimal", "")
                        .replace("_years", "")
                        .replace("scenario_", "")
                    )
                    values = lookup_df.loc[base_mask, col_short].dropna()
                    row[col_short] = values.median() if not values.empty else None

                fallback_rows.append(row)

    fallback_df = pd.DataFrame(fallback_rows)
    lookup_complete = pd.concat([lookup_df, fallback_df], ignore_index=True)

    print(
        f"   Lookup table created: {len(lookup_complete):,} rows ({len(lookup_df):,} specific + {len(fallback_df):,} fallback)"
    )

    # Save lookup table as CSV for analysis
    lookup_output_file = "4_technology_lookup_table.csv"
    lookup_complete.to_csv(lookup_output_file, index=False)
    print(f"   💾 Saved lookup table: {lookup_output_file}")

    return lookup_complete


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

    # Create comprehensive technology lookup table
    lookup_table = create_technology_lookup_table(df)

    if "gap_filled_columns" not in df.columns:
        df["gap_filled_columns"] = ""
    else:
        # Ensure it's properly initialized as string
        df["gap_filled_columns"] = df["gap_filled_columns"].fillna("").astype(str)

    print(f"🔧 Gap-filling {len(spec)} columns using lookup table approach...")

    # Map of column names in spec to lookup table columns
    col_mapping = {
        "scenario_capacity_factor": "capacity_factor",
        "lifetime_years": "lifetime",
        "efficiency_decimal": "efficiency",
        "om_cost_usd_per_mw_per_yr": "om_cost",
        "capital_cost_usd_per_mw": "capital_cost",
    }

    # Extract ISO2 codes from country_iso2_list for each row
    def extract_iso2_codes(iso2_list_str):
        if pd.isna(iso2_list_str) or not str(iso2_list_str).strip():
            return []
        return [code.strip() for code in str(iso2_list_str).split(",") if code.strip()]

    total_filled_by_col = {}

    for col in spec.keys():
        if col not in df.columns:
            print(f"   ⚠️ Skipping {col} - column not found")
            continue

        lookup_col = col_mapping.get(col, col)
        if lookup_col not in lookup_table.columns:
            print(f"   ⚠️ Skipping {col} - not in lookup table")
            continue

        missing_mask = df[col].isna()
        initial_missing_count = missing_mask.sum()

        if not missing_mask.any():
            print(f"   ✅ {col}: no missing values")
            continue

        print(
            f"   🔧 {col}: filling {initial_missing_count:,} missing values using lookup table..."
        )

        filled_count = 0

        # For each missing row, find best match in lookup table
        for idx in df.index[missing_mask]:
            tech = df.loc[idx, "technology"]
            year = df.loc[idx, "scenario_year"]
            stringency = (
                df.loc[idx, "stringency"] if "stringency" in df.columns else None
            )
            iso2_list = extract_iso2_codes(df.loc[idx, "country_iso2_list"])

            best_value = None

            # Lookup hierarchy: try specific stringency first, then fallback to NA stringency
            for try_stringency in [stringency, None]:
                if best_value is not None:
                    break

                # For regional scenarios (multiple ISO2s), calculate median across constituent countries
                if len(iso2_list) > 1:
                    # Regional scenario - get values for all constituent countries
                    regional_values = []
                    for iso2 in iso2_list:
                        lookup_mask = (
                            (lookup_table["technology"] == tech)
                            & (lookup_table["year"] == year)
                            & (lookup_table["iso2"] == iso2)
                            & (lookup_table["stringency"] == try_stringency)
                            & (lookup_table[lookup_col].notna())
                        )

                        matches = lookup_table.loc[lookup_mask, lookup_col]
                        if not matches.empty:
                            regional_values.append(matches.iloc[0])

                    if regional_values:
                        best_value = pd.Series(regional_values).median()
                        break

                else:
                    # Single country scenario - direct lookup
                    for iso2 in iso2_list:
                        lookup_mask = (
                            (lookup_table["technology"] == tech)
                            & (lookup_table["year"] == year)
                            & (lookup_table["iso2"] == iso2)
                            & (lookup_table["stringency"] == try_stringency)
                            & (lookup_table[lookup_col].notna())
                        )

                        matches = lookup_table.loc[lookup_mask, lookup_col]
                        if not matches.empty:
                            best_value = matches.iloc[0]  # Take first match
                            break

                # If no ISO2 match found, try without ISO2 constraint (fallback)
                if best_value is None:
                    lookup_mask = (
                        (lookup_table["technology"] == tech)
                        & (lookup_table["year"] == year)
                        & (lookup_table["stringency"] == try_stringency)
                        & (lookup_table[lookup_col].notna())
                    )

                    matches = lookup_table.loc[lookup_mask, lookup_col]
                    if not matches.empty:
                        best_value = (
                            matches.median()
                        )  # Use median across all geographies

            # Apply the found value
            if best_value is not None:
                df.loc[idx, col] = best_value
                filled_count += 1

                # Update tracking
                current_tracking = df.loc[idx, "gap_filled_columns"]
                if pd.isna(current_tracking) or current_tracking == "":
                    df.loc[idx, "gap_filled_columns"] = col
                else:
                    df.loc[idx, "gap_filled_columns"] = (
                        str(current_tracking) + "," + col
                    )

        total_filled_by_col[col] = filled_count
        final_missing = df[col].isna().sum()
        print(f"      Filled {filled_count:,} values, {final_missing:,} still missing")

    # Summary
    total_filled_rows = (df["gap_filled_columns"] != "").sum()
    print(
        f"   📊 Gap-filling complete: {total_filled_rows:,} rows received gap-filled values"
    )

    if total_filled_by_col:
        print(f"   📊 Columns filled breakdown:")
        for col_name, count in total_filled_by_col.items():
            print(f"     {col_name}: {count:,} rows")

    return df


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
    # Step 3
    step3_finalize_target_schema()
    # Step 4
    step4_aggregate_and_gapfill()
    print_banner("AR6 Combined Pipeline — Done")


if __name__ == "__main__":
    main()
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

    # ===== OILCAP COST HEURISTIC =====
    # Apply OilCap cost heuristic: use average of GasCap and CoalCap costs
    # print("🛢️ Applying OilCap cost heuristic (average of GasCap and CoalCap)...")

    # oilcap_mask = df["technology"].str.contains("OilCap", na=False)
    # oilcap_rows = oilcap_mask.sum()
    # print(f"   Found {oilcap_rows:,} OilCap rows")

    # if oilcap_rows > 0:
    #     # Group by scenario dimensions for cost averaging
    #     cost_grouping_cols = [
    #         "scenario_provider",
    #         "scenario",
    #         "scenario_geography",
    #         "scenario_year",
    #         "sector",
    #     ]
    #     cost_cols = ["capital_cost_usd_per_mw", "om_cost_usd_per_mw_per_yr"]

    #     filled_counts = {"capital_cost": 0, "om_cost": 0}

    #     for cost_col in cost_cols:
    #         if cost_col not in df.columns:
    #             continue

    #         # Find OilCap rows missing this cost
    #         missing_cost_mask = oilcap_mask & df[cost_col].isna()
    #         missing_count = missing_cost_mask.sum()

    #         if missing_count > 0:
    #             print(
    #                 f"   Filling {missing_count:,} missing {cost_col} values for OilCap..."
    #             )

    #             # For each missing OilCap row, find GasCap and CoalCap costs in same scenario/geography/year
    #             for idx in df.index[missing_cost_mask]:
    #                 scenario_data = df.loc[idx, cost_grouping_cols].to_dict()

    #                 # Find GasCap and CoalCap costs for same scenario/geography/year
    #                 base_mask = True
    #                 for col, val in scenario_data.items():
    #                     if col in df.columns:
    #                         base_mask = base_mask & (df[col] == val)

    #                 gas_mask = base_mask & df["technology"].str.contains(
    #                     "GasCap", na=False
    #                 )
    #                 coal_mask = base_mask & df["technology"].str.contains(
    #                     "CoalCap", na=False
    #                 )

    #                 gas_costs = df.loc[gas_mask & df[cost_col].notna(), cost_col]
    #                 coal_costs = df.loc[coal_mask & df[cost_col].notna(), cost_col]

    #                 # Calculate average if both are available
    #                 costs_to_average = []
    #                 if not gas_costs.empty:
    #                     costs_to_average.append(gas_costs.mean())
    #                 if not coal_costs.empty:
    #                     costs_to_average.append(coal_costs.mean())

    #                 if costs_to_average:
    #                     avg_cost = sum(costs_to_average) / len(costs_to_average)
    #                     df.loc[idx, cost_col] = avg_cost
    #                     filled_counts[cost_col.split("_")[0]] += 1

    #     print(
    #         f"   OilCap heuristic filled: {filled_counts['capital']} capital costs, {filled_counts['om']} OM costs"
    #     )

    # ===== TECHNOLOGY MAPPING DISABLED =====
    # Skip technology mapping to keep all technologies separate
    # This prevents artificial aggregation of different technology variants
    print("🚫 Technology mapping DISABLED - keeping all technologies separate")
    df_map = df.copy()  # Use original data without any technology mapping

    # # Vectorized mapping from mapping file (first pass)
    # df_map = vectorized_mapping(df, mapping_df)
    # df_map["sector"] = df_map["target_sector"]
    # df_map["technology"] = df_map["target_technology"]

    # # Enforce canonical final targets
    # FINAL_TARGETS: set[tuple[str, str]] = {
    #     ("Coal", "Coal"),
    #     ("Oil&Gas", "Oil"),
    #     ("Oil&Gas", "Gas"),
    #     ("Power", "SolarCap"),
    #     ("Power", "CoalCap"),
    #     ("Power", "GasCap"),
    #     ("Power", "OilCap"),
    #     ("Power", "BiomassCap"),
    #     ("Power", "WindCap"),
    #     ("Power", "HydroCap"),
    #     ("Power", "NuclearCap"),
    #     ("Power", "GeothermalCap"),
    #     ("Steel", "BF-BOF"),
    #     ("Steel", "DRI-BOF"),
    #     ("Steel", "EAF"),
    # }

    # def canonicalize_to_final_targets(df_in: pd.DataFrame) -> pd.DataFrame:
    #     dfc = df_in.copy()
    #     sec = dfc["sector"].astype(str).str.lower()
    #     tech = dfc["technology"].astype(str).str.lower()

    #     # Start with identity
    #     sec_out = dfc["sector"].astype(str).copy()
    #     tech_out = dfc["technology"].astype(str).copy()

    #     # Normalize sector names first
    #     sec_out = sec_out.mask(sec.isin(["gas&oil", "oil&gas"]), "Oil&Gas")

    #     # Coal sector → (Coal, Coal)
    #     coal_mask = sec.eq("coal")
    #     sec_out = sec_out.mask(coal_mask, "Coal")
    #     tech_out = tech_out.mask(coal_mask, "Coal")

    #     # Oil&Gas sector → tech either Oil or Gas
    #     og_mask = sec.isin(["oil&gas", "gas&oil"]) | sec_out.eq("Oil&Gas")
    #     gas_mask = og_mask & (tech.str.contains("gas"))
    #     oil_mask = og_mask & (tech.str.contains("oil"))
    #     sec_out = sec_out.mask(og_mask, "Oil&Gas")
    #     tech_out = tech_out.mask(gas_mask, "Gas")
    #     tech_out = tech_out.mask(oil_mask, "Oil")

    #     # Power-like sectors (Power, Renewables, Nuclear → Power)
    #     power_like = sec.isin(["power", "renewables", "nuclear"]) | sec_out.isin(["Power", "Renewables", "Nuclear"])
    #     sec_out = sec_out.mask(power_like, "Power")

    #     # Map power technologies to Cap variants
    #     def map_power_tech(name: str) -> str:
    #         n = name.lower()
    #         if any(k in n for k in ["solar", "pv", "csp"]):
    #             return "SolarCap"
    #         if "wind" in n:
    #             return "WindCap"
    #         if "hydro" in n:
    #             return "HydroCap"
    #         if "nuclear" in n:
    #             return "NuclearCap"
    #         if "geothermal" in n:
    #             return "GeothermalCap"
    #         if any(k in n for k in ["biomass", "bio"]):
    #             return "BiomassCap"
    #         if "coal" in n:
    #             return "CoalCap"
    #         if "gas" in n:
    #             return "GasCap"
    #         if "oil" in n:
    #             return "OilCap"
    #         return name

    #     power_idx = power_like[power_like].index
    #     tech_out.loc[power_idx] = tech_out.loc[power_idx].apply(map_power_tech)

    #     # Steel mapping to 3 categories
    #     steel_mask = sec.eq("steel") | sec_out.eq("Steel")
    #     def map_steel_tech(name: str) -> str:
    #         n = name.lower()
    #         if "dri" in n:
    #             return "DRI-BOF"
    #         if "eaf" in n:
    #             return "EAF"
    #         # default steel route
    #         return "BF-BOF"

    #     steel_idx = steel_mask[steel_mask].index
    #     sec_out = sec_out.mask(steel_mask, "Steel")
    #     tech_out.loc[steel_idx] = tech_out.loc[steel_idx].apply(map_steel_tech)

    #     # Apply canonical
    #     dfc["sector"] = sec_out
    #     dfc["technology"] = tech_out

    #     # Filter to final allowed set
    #     pair = list(zip(dfc["sector"], dfc["technology"]))
    #     keep = [p in FINAL_TARGETS for p in pair]
    #     return dfc.loc[keep].copy()

    # before_rows = len(df_map)
    # df_map = canonicalize_to_final_targets(df_map)
    # after_rows = len(df_map)
    # print(f"Canonical targets: kept {after_rows:,}/{before_rows:,} rows")

    print(f"Keeping all original technologies: {len(df_map):,} rows")

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

    # Aggregate with a single groupby apply to minimize Python overhead
    def aggregate_group(group: pd.DataFrame) -> pd.Series:
        out: Dict[str, float] = {}
        # Sums
        for col in sum_cols:
            out[col] = group[col].sum()
        # Weighted averages by scenario_pathway
        for col in avg_cols:
            out[col] = (
                weighted_avg(
                    group,
                    col,
                    "scenario_pathway" if "scenario_pathway" in group.columns else None,
                )
                if "scenario_pathway" in group.columns
                else group[col].mean()
            )
        # Carry-forward non-agg columns (take first)
        carry_cols = [
            c
            for c in group.columns
            if c not in set(sum_cols + avg_cols)
            and c not in ["target_sector", "target_technology", "aggregation_group"]
        ]
        carry_first = group[carry_cols].iloc[0]
        for c in carry_cols:
            out.setdefault(c, carry_first[c])
        return pd.Series(out)

    aggregated = df_map.groupby(grouping_cols, as_index=False).apply(aggregate_group)
    # groupby.apply with as_index=False returns index columns too; ensure flat frame
    if isinstance(aggregated.columns, pd.MultiIndex):
        aggregated.columns = [
            "_".join([str(c) for c in tup if c != ""]) for tup in aggregated.columns
        ]

    # Initialize gap-filled tracking column
    if "gap_filled_columns" not in aggregated.columns:
        aggregated["gap_filled_columns"] = ""
    else:
        # Ensure it's properly initialized as string
        aggregated["gap_filled_columns"] = (
            aggregated["gap_filled_columns"].fillna("").astype(str)
        )

    # Create stringency BEFORE gap-filling so it can be used in hierarchy
    aggregated["stringency"] = aggregated.get("scenario_type", np.nan)

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
        # [
        #     "scenario_year",
        #     "technology",
        #     "scenario_geography",
        #     "scenario_type",
        #     "stringency",
        # ],  # Most specific
        # ["scenario_year", "technology", "scenario_type", "stringency"],  # Same policy
        # [
        #     "scenario_year",
        #     "technology",
        #     "scenario_geography",
        #     "scenario_type",
        # ],  # Same type, cross-stringency
        # [
        #     "scenario_year",
        #     "technology",
        #     "scenario_geography",
        #     "stringency",
        # ],  # Same stringency, cross-type
        # ["scenario_year", "technology", "scenario_type"],  # Cross-geo/stringency
        # ["scenario_year", "technology", "stringency"],  # Cross-geo/type
        # ["scenario_year", "technology", "scenario_geography"],  # Cross-policy
        # ["scenario_year", "technology"],  # Same tech only
        # [
        #     "technology",
        #     "scenario_geography",
        #     "scenario_type",
        #     "stringency",
        # ],  # Cross-time
        # ["technology", "scenario_geography"],  # Cross-time/policy
        ["technology"],  # Most general
    ]

    # Add Global geography-specific fallback levels for better gap-filling
    # These levels specifically use Global geography as a fallback when regional data is missing
    global_fallback_hierarchy = [
        # [
        #     "scenario_year",
        #     "technology",
        #     "scenario_type",
        #     "stringency",
        # ],  # Use Global for same policy
        # ["scenario_year", "technology", "scenario_type"],  # Use Global for same type
        # ["scenario_year", "technology", "stringency"],  # Use Global for same stringency
        # ["scenario_year", "technology"],  # Use Global for same tech/year
        # ["technology", "scenario_type", "stringency"],  # Use Global cross-time
        # ["technology", "scenario_type"],  # Use Global cross-time/stringency
        ["technology"],  # Use Global most general
    ]

    # Apply standard hierarchy to all gap-fillable columns for consistency
    gap_spec: Dict[str, List[List[str]]] = {}

    # Technology-specific columns use the full standard hierarchy
    tech_columns = [
        "scenario_capacity_factor",
        "lifetime_years",
        "efficiency_decimal",
        "om_cost_usd_per_mw_per_yr",
        "capital_cost_usd_per_mw",
    ]

    for col in tech_columns:
        if col in aggregated.columns:
            gap_spec[col] = standard_hierarchy

    # Price columns use a simplified hierarchy focused on temporal and geographic consistency
    price_hierarchy = [
        # ["scenario_year", "technology", "scenario_geography", "scenario_type"],
        # ["scenario_year", "technology", "scenario_geography"],
        # ["scenario_year", "technology", "scenario_type"],
        # ["scenario_year", "technology"],
        # ["technology", "scenario_geography"],
        ["technology"],
    ]

    if "scenario_price" in aggregated.columns:
        gap_spec["scenario_price"] = price_hierarchy

    # Fuel price uses fuel-specific hierarchy with cross-fuel fallbacks
    if "fuel_price" in aggregated.columns:
        gap_spec["fuel_price"] = [
            # [
            #     "scenario_year",
            #     "fuel_for_price",
            #     "scenario_geography",
            #     "scenario_type",
            #     "stringency",
            # ],
            # ["scenario_year", "fuel_for_price", "scenario_geography", "scenario_type"],
            # ["scenario_year", "fuel_for_price", "scenario_geography"],
            # ["scenario_year", "fuel_for_price", "scenario_type"],
            # ["scenario_year", "fuel_for_price"],
            # ["fuel_for_price", "scenario_geography"],
            ["fuel_for_price"],
        ]

    agg_fn = {"fuel_price": "mean"}  # use mean for fuel cascade; median elsewhere
    # aggregated = gap_fill_with_lookup_table(
    #     aggregated, gap_spec, global_fallback_hierarchy, agg_fn
    # )

    # Scenario type update based on stringency values
    if "scenario_type" in aggregated.columns:
        # Create baseline mask combining both WITCH and IMAGE baseline scenarios
        baseline_mask = (
            ((aggregated["scenario_provider"] == "WITCH 5.0") 
             & aggregated["scenario"].isin(["CO_CurPol", "EN_NoPolicy"]))
            | ((aggregated["scenario_provider"] == "IMAGE 3.2")
               & aggregated["scenario"].isin(["SSP1-baseline", "SSP2-baseline"]))
        )
        
        # Apply scenario types based on the combined mask
        aggregated.loc[baseline_mask, "scenario_type"] = "baseline"
        aggregated.loc[~baseline_mask, "scenario_type"] = "target"

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
            aggregated["technology"]
            .astype(str)
            .apply(lambda x: any(keyword in x for keyword in renewable_tech_keywords))
        )

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
