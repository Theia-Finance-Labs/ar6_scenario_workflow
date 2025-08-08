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

try:
    import modin.pandas as pd  # type: ignore
except Exception:
    import pandas as pd  # Fallback if Modin not available
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
            if 2020 < year <= 2050:
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

    print(f"Year columns for analysis: {len(year_cols)} | Range: {min(year_cols)}–{max(year_cols)}")

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

    base_data = renamed.merge(mapping, how="left", left_on="variable", right_on="variable")

    # Drop rows with missing Sector mapping
    before_sector = len(base_data)
    base_data = base_data[base_data["Sector"].notna()].copy()
    after_sector = len(base_data)
    print(
        f"Sector mapping: kept {after_sector:,}/{before_sector:,} rows; removed {before_sector-after_sector:,}"
    )

    target_sectors = ["Steel", "Nuclear", "Gas&Oil", "Cement", "Coal", "Renewables", "Power"]
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
            return pd.DataFrame(columns=join_cols + [metric.lower().replace(" ", "_"), f"{metric.lower().replace(' ', '_')}_unit"])
        return (
            subset[join_cols + ["value", "unit"]]
            .rename(columns={"value": metric.lower().replace(" ", "_"), "unit": f"{metric.lower().replace(' ', '_')}_unit"})
        )

    print("Extracting OM Cost / Capital Cost / Efficiency …")
    om_cost = extract_metric(base_data, "OM Cost")
    cap_cost = extract_metric(base_data, "Capital Cost")
    efficiency = extract_metric(base_data, "Efficiency")

    merged = base_data.copy()
    merged = merged.merge(om_cost, on=join_cols, how="left") if not om_cost.empty else merged
    merged = merged.merge(efficiency, on=join_cols, how="left") if not efficiency.empty else merged
    merged = merged.merge(cap_cost, on=join_cols, how="left") if not cap_cost.empty else merged

    # Price data (keep all; unit conversions later)
    price_rows = base_data[base_data["col1"] == "Price"].copy()
    if not price_rows.empty:
        price_rows = price_rows.rename(columns={"col2": "energy_type", "value": "price"})
        primary_prices = price_rows[price_rows["energy_type"] == "Primary Energy"].copy()
        secondary_prices = price_rows[price_rows["energy_type"] == "Secondary Energy"].copy()
        final_prices = price_rows[price_rows["energy_type"] == "Final Energy"].copy()
        carbon_prices = price_rows[price_rows["energy_type"] == "Carbon"].copy()

        # Primary energy (fuel-specific via Fuel)
        if not primary_prices.empty:
            primary_clean = primary_prices[["model", "scenario", "region", "year", "Fuel", "price", "unit"]].copy()
            primary_clean = primary_clean.rename(
                columns={"price": "primary_energy_price", "unit": "primary_energy_price_unit"}
            )
            merged = merged.merge(
                primary_clean,
                on=["model", "scenario", "region", "year", "Fuel"],
                how="left",
            )

        # Secondary energy (fuel-specific)
        if not secondary_prices.empty:
            secondary_clean = secondary_prices[["model", "scenario", "region", "year", "Fuel", "price", "unit"]].copy()
            secondary_clean = secondary_clean.rename(
                columns={"price": "secondary_energy_price", "unit": "secondary_energy_price_unit"}
            )
            merged = merged.merge(
                secondary_clean,
                on=["model", "scenario", "region", "year", "Fuel"],
                how="left",
            )

        # Carbon price (single per model/scenario/region/year)
        if not carbon_prices.empty:
            carbon_pivot = carbon_prices.groupby(["model", "scenario", "region", "year"][0:4])["price"].first().reset_index()
            carbon_units = carbon_prices.groupby(["model", "scenario", "region", "year"][0:4])["unit"].first().reset_index()
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
        secondary_electricity = secondary_prices[secondary_prices["Fuel"] == "Electricity"].copy()
        if not secondary_electricity.empty:
            sec_elec = secondary_electricity.groupby(["model", "scenario", "region", "year"][0:4])["price"].first().reset_index()
            sec_elec_u = secondary_electricity.groupby(["model", "scenario", "region", "year"][0:4])["unit"].first().reset_index()
            merged = merged.merge(
                sec_elec.rename(columns={"price": "secondary_energy_electricity_price"}),
                on=["model", "scenario", "region", "year"],
                how="left",
            )
            merged = merged.merge(
                sec_elec_u.rename(columns={"unit": "secondary_energy_electricity_price_unit"}),
                on=["model", "scenario", "region", "year"],
                how="left",
            )

        final_electricity = final_prices[final_prices["Fuel"] == "Electricity"].copy()
        if not final_electricity.empty:
            fin_elec = final_electricity.groupby(["model", "scenario", "region", "year"][0:4])["price"].first().reset_index()
            fin_elec_u = final_electricity.groupby(["model", "scenario", "region", "year"][0:4])["unit"].first().reset_index()
            merged = merged.merge(
                fin_elec.rename(columns={"price": "final_energy_electricity_price"}),
                on=["model", "scenario", "region", "year"],
                how="left",
            )
            merged = merged.merge(
                fin_elec_u.rename(columns={"unit": "final_energy_electricity_price_unit"}),
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

    memory_release(source, mapping, melted, renamed, base_data, om_cost, cap_cost, efficiency, price_rows, merged)
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

    pivoted = (
        df_filtered.pivot_table(index=grouping_cols, columns="col1", values="value", aggfunc="first")
        .reset_index()
    )
    pivoted.columns.name = None
    rename_map: Dict[str, str] = {}
    for col in pivoted.columns:
        if col in grouping_cols:
            rename_map[col] = col
        else:
            rename_map[col] = col.lower().replace(" ", "_") + "_value"
    pivoted = pivoted.rename(columns=rename_map)

    # Join keys used for all merges
    join_cols = [c for c in ["model", "scenario", "scenario_geography", "year", "Sector", "Technology"] if c in df_filtered.columns and c in pivoted.columns]

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
        cap_df = cap_df.drop(columns=["value", "unit"])\
                         .groupby(join_cols, as_index=False)["capacity_mw"].first()
        pivoted = pivoted.merge(cap_df, on=join_cols, how="left")
        pivoted = pivoted.drop(columns=["capacity_value"], errors="ignore")

    # 2) Capacity Additions → MW/yr (supports kW/yr, MW/yr, GW/yr variants incl. yr-1)
    add_rows = df_filtered[df_filtered["col1"] == "Capacity Additions"]
    if not add_rows.empty:
        add_df = add_rows[join_cols + ["value", "unit"]].copy()
        u = add_df["unit"].astype(str).str.lower()
        add_mwyr = add_df["value"].astype(float)
        gwyr_mask = u.str.contains("gw") & (u.str.contains("/yr") | u.str.contains("yr-1"))
        mwy_mask = u.str.contains("mw") & (u.str.contains("/yr") | u.str.contains("yr-1"))
        kwyr_mask = u.str.contains("kw") & (u.str.contains("/yr") | u.str.contains("yr-1"))
        add_mwyr.loc[gwyr_mask] = add_mwyr.loc[gwyr_mask] * 1000.0
        add_mwyr.loc[mwy_mask] = add_mwyr.loc[mwy_mask] * 1.0
        add_mwyr.loc[kwyr_mask] = add_mwyr.loc[kwyr_mask] / 1000.0
        add_df["capacity_additions_mw_per_yr"] = add_mwyr
        add_df = add_df.drop(columns=["value", "unit"])\
                       .groupby(join_cols, as_index=False)["capacity_additions_mw_per_yr"].first()
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
        pri_df["primary_energy_mwh_per_yr"] = energy_units_to_mwh(pri_df["value"], pri_df["unit"])\
            .replace([np.inf, -np.inf], np.nan)
        pri_df = pri_df.drop(columns=["value", "unit"])\
                       .groupby(join_cols, as_index=False)["primary_energy_mwh_per_yr"].first()
        pivoted = pivoted.merge(pri_df, on=join_cols, how="left")
        pivoted = pivoted.drop(columns=["primary_energy_value"], errors="ignore")

    sec_rows = df_filtered[df_filtered["col1"] == "Secondary Energy"]
    if not sec_rows.empty:
        sec_df = sec_rows[join_cols + ["value", "unit"]].copy()
        sec_df["secondary_energy_mwh_per_yr"] = energy_units_to_mwh(sec_df["value"], sec_df["unit"])\
            .replace([np.inf, -np.inf], np.nan)
        sec_df = sec_df.drop(columns=["value", "unit"])\
                       .groupby(join_cols, as_index=False)["secondary_energy_mwh_per_yr"].first()
        pivoted = pivoted.merge(sec_df, on=join_cols, how="left")
        pivoted = pivoted.drop(columns=["secondary_energy_value"], errors="ignore")

    # Merge cost metrics from original df (non-pivoted)
    cost_data = df[~df["col1"].isin(target_col1_values)].copy()
    # refresh join_cols to ensure overlap with cost_data
    join_cols = [c for c in ["model", "scenario", "scenario_geography", "year", "Sector", "Technology"] if c in cost_data.columns and c in pivoted.columns]

    def add_cost_metric(pivoted_df: pd.DataFrame, cost_df: pd.DataFrame, metric_name: str, join_cols: List[str]) -> pd.DataFrame:
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
            unit_clean = unit_str.replace("US$2010/", "").replace("/", "_per_").replace(" ", "_").lower()
            col_name = metric_name.lower().replace(" ", "_") + (f"_{unit_clean}" if unit_clean else "")
        metric_df = metric_df[join_cols + ["value"]].rename(columns={"value": col_name})
        return pivoted_df.merge(metric_df, on=join_cols, how="left")

    pivoted = add_cost_metric(pivoted, cost_data, "OM Cost", join_cols)
    pivoted = add_cost_metric(pivoted, cost_data, "Capital Cost", join_cols)
    pivoted = add_cost_metric(pivoted, cost_data, "Efficiency", join_cols)

    # Add unit columns where available (used by later conversions)
    for unit_col in ["om_cost_unit", "capital_cost_unit", "efficiency_unit"]:
        if unit_col in cost_data.columns:
            unit_df = (
                cost_data[join_cols + [unit_col]].dropna(subset=[unit_col]).drop_duplicates()
            )
            pivoted = pivoted.merge(unit_df, on=join_cols, how="left")

    # Renewables efficiency special-case: set to 100% if missing
    if "efficiency_percent" not in pivoted.columns:
        pivoted["efficiency_percent"] = np.nan
    renewables_mask = pivoted["Sector"] == "Renewables"
    pivoted.loc[renewables_mask, "efficiency_percent"] = 1.0

    # Carbon price (if available in original df)
    if set(["carbon_price", "carbon_price_unit"]).issubset(df.columns):
        price_join_cols = join_cols
        carbon_data = df[price_join_cols + ["carbon_price", "carbon_price_unit"]].dropna(subset=["carbon_price"]).drop_duplicates()
        if not carbon_data.empty:
            carbon_data = carbon_data.rename(columns={"carbon_price": "carbon_price_usd_per_tco2"})
            carbon_data = carbon_data.drop(columns=["carbon_price_unit"], errors="ignore")
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
        price_df = df_source[join_cols + [price_col, unit_col]].dropna(subset=[price_col]).drop_duplicates()
        if price_df.empty:
            pivoted[price_col + "_usd_per_gj"] = np.nan
            pivoted[unit_col] = np.nan
            return pivoted
        price_df = price_df.rename(columns={price_col: price_col + "_usd_per_gj"})
        return pivoted.merge(price_df, on=join_cols, how="left")

    pivoted = merge_energy_price(df, "primary")
    pivoted = merge_energy_price(df, "secondary")

    # Secondary electricity price (USD/GJ)
    sec_elec_cols = ["secondary_energy_electricity_price", "secondary_energy_electricity_price_unit"]
    if set(sec_elec_cols).issubset(df.columns):
        tmp = df[join_cols + sec_elec_cols].dropna(subset=[sec_elec_cols[0]]).drop_duplicates()
        if not tmp.empty:
            tmp = tmp.rename(columns={sec_elec_cols[0]: "secondary_energy_electricity_price_usd_per_gj"})
            tmp = tmp.drop(columns=[sec_elec_cols[1]], errors="ignore")
            pivoted = pivoted.merge(tmp, on=join_cols, how="left")
        else:
            pivoted["secondary_energy_electricity_price_usd_per_gj"] = np.nan
    else:
        pivoted["secondary_energy_electricity_price_usd_per_gj"] = np.nan

    # Unit conversions to target schema units
    # Capacity → MW
    if "capacity_value" in pivoted.columns:
        pivoted["capacity_mw"] = pivoted["capacity_value"] * 1000.0  # GW→MW or MW→MW if already in MW (ok if data was MW)
        pivoted = pivoted.drop(columns=["capacity_value"], errors="ignore")

    # Capacity additions → MW/yr
    if "capacity_additions_value" in pivoted.columns:
        pivoted["capacity_additions_mw_per_yr"] = pivoted["capacity_additions_value"] * 1000.0
        pivoted = pivoted.drop(columns=["capacity_additions_value"], errors="ignore")

    # Energy (Primary/Secondary) → MWh/yr with mixed units handled via source 'unit' column if available
    def energy_to_mwh_per_year(df_in: pd.DataFrame, value_col: str, unit_series: Optional[pd.Series]) -> pd.Series:
        if value_col not in df_in.columns:
            return pd.Series(index=df_in.index, dtype=float)
        values = df_in[value_col].copy()
        if unit_series is None:
            # Assume EJ/yr if unknown; convert EJ → MWh
            return values * (1e18 / 3.6e9)
        # Per-row conversions using unit column
        ej_mask = unit_series.isin(["EJ/yr", "EJ yr-1"]) if unit_series is not None else pd.Series(False, index=df_in.index)
        pj_mask = unit_series.isin(["PJ/yr", "PJ yr-1"]) if unit_series is not None else pd.Series(False, index=df_in.index)
        tj_mask = unit_series.isin(["TJ/yr", "TJ yr-1"]) if unit_series is not None else pd.Series(False, index=df_in.index)
        out = values.copy()
        out.loc[ej_mask] = values.loc[ej_mask] * (1e18 / 3.6e9)
        out.loc[pj_mask] = values.loc[pj_mask] * (1e15 / 3.6e9)
        out.loc[tj_mask] = values.loc[tj_mask] * (1e12 / 3.6e9)
        return out

    # We lost row-level unit for energy in pivot; fallback to EJ/yr assumption as in prior script when unknown
    if "secondary_energy_value" in pivoted.columns:
        pivoted["secondary_energy_mwh_per_yr"] = energy_to_mwh_per_year(pivoted, "secondary_energy_value", None)
        pivoted = pivoted.drop(columns=["secondary_energy_value"], errors="ignore")
    if "primary_energy_value" in pivoted.columns:
        pivoted["primary_energy_mwh_per_yr"] = energy_to_mwh_per_year(pivoted, "primary_energy_value", None)
        pivoted = pivoted.drop(columns=["primary_energy_value"], errors="ignore")

    # Lifetime → years (already years in most cases)
    if "lifetime_value" in pivoted.columns:
        pivoted["lifetime_years"] = pivoted["lifetime_value"]
        pivoted = pivoted.drop(columns=["lifetime_value"], errors="ignore")

    # Cost conversions: unit-aware to MW or MW/yr; clean zeros/negatives
    def convert_capacity_cost_to_mw(values: pd.Series, units: Optional[pd.Series], per_year: bool) -> pd.Series:
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
        units = pivoted["capital_cost_unit"] if "capital_cost_unit" in pivoted.columns else None
        pivoted["capital_cost_usd_per_mw"] = convert_capacity_cost_to_mw(
            pivoted["capital_cost_usd_per_mw"], units, per_year=False
        )

    # Efficiency: treat 0 as missing, then convert percent >1 to decimal; Renewables already set to 1.0 above
    if "efficiency_percent" in pivoted.columns:
        zero_eff_mask = pivoted["efficiency_percent"] == 0
        pivoted.loc[zero_eff_mask, "efficiency_percent"] = np.nan
        pivoted["efficiency_decimal"] = pivoted["efficiency_percent"]
        percent_mask = pivoted["efficiency_percent"] > 1
        pivoted.loc[percent_mask, "efficiency_decimal"] = pivoted.loc[percent_mask, "efficiency_percent"] / 100.0
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


def build_price_tables(step1_df: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    if step1_df is None:
        return {"elec": pd.DataFrame(), "primary": pd.DataFrame(), "secondary_by_fuel": pd.DataFrame()}

    prices = step1_df[step1_df["col1"] == "Price"].copy()
    if prices.empty:
        return {"elec": pd.DataFrame(), "primary": pd.DataFrame(), "secondary_by_fuel": pd.DataFrame()}

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

    # Electricity (Secondary Energy, Electricity fuel)
    elec = prices[(prices["col2"] == "Secondary Energy") & (prices["Fuel"] == "Electricity")][[
        "model", "scenario", "region", "year", "price_usd_per_mwh"
    ]].copy()
    elec = elec.groupby(["model", "scenario", "region", "year"], as_index=False)["price_usd_per_mwh"].first()

    # Primary energy (not fuel-specific; take first per group)
    primary = prices[(prices["col2"] == "Primary Energy")][[
        "model", "scenario", "region", "year", "price_usd_per_mwh"
    ]].copy()
    primary = primary.groupby(["model", "scenario", "region", "year"], as_index=False)["price_usd_per_mwh"].first()

    # Secondary energy by fuel (fuel-specific table)
    secondary_by_fuel = prices[(prices["col2"] == "Secondary Energy")][[
        "model", "scenario", "region", "year", "Fuel", "price_usd_per_mwh"
    ]].copy()
    secondary_by_fuel = secondary_by_fuel.groupby(
        ["model", "scenario", "region", "year", "Fuel"], as_index=False
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
        meta_df = pd.read_excel("AR6_Scenarios_Database_metadata_indicators_v1.1 2.xlsx", sheet_name="meta_Ch3vetted_withclimate")
        if set(["Model", "Scenario", "Category"]).issubset(meta_df.columns):
            meta_df = meta_df[["Model", "Scenario", "Category"]].dropna()
            meta_df["lookup_key"] = meta_df["Model"].astype(str) + "|||" + meta_df["Scenario"].astype(str)
            lookup = meta_df.set_index("lookup_key")["Category"].to_dict()
            key = target["scenario_provider"].astype(str) + "|||" + target["scenario"].astype(str)
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
        classify_tech(sec, tech) for sec, tech in zip(target["sector"], target["technology"])
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
    elec = price_tables["elec"].rename(columns={
        "model": "scenario_provider",
        "scenario": "scenario",
        "region": "scenario_geography",
        "year": "scenario_year",
        "price_usd_per_mwh": "scenario_price_electricity"
    }) if not price_tables["elec"].empty else pd.DataFrame(columns=join_key + ["scenario_price_electricity"])

    primary = price_tables["primary"].rename(columns={
        "model": "scenario_provider",
        "scenario": "scenario",
        "region": "scenario_geography",
        "year": "scenario_year",
        "price_usd_per_mwh": "scenario_price_primary"
    }) if not price_tables["primary"].empty else pd.DataFrame(columns=join_key + ["scenario_price_primary"])

    sec_by_fuel = price_tables["secondary_by_fuel"].rename(columns={
        "model": "scenario_provider",
        "scenario": "scenario",
        "region": "scenario_geography",
        "year": "scenario_year",
        "Fuel": "fuel_for_price",
        "price_usd_per_mwh": "fuel_price"
    }) if not price_tables["secondary_by_fuel"].empty else pd.DataFrame(columns=join_key + ["fuel_for_price", "fuel_price"])

    # scenario_price: Power/Renewables -> electricity; others -> primary
    target = target.merge(elec, on=join_key, how="left")
    target = target.merge(primary, on=join_key, how="left")
    is_power_like = target["sector"].isin(["Power", "Renewables"]) if "sector" in target.columns else pd.Series(False, index=target.index)
    target["scenario_price"] = np.where(
        is_power_like,
        target["scenario_price_electricity"],
        target["scenario_price_primary"],
    )
    target = target.drop(columns=["scenario_price_electricity", "scenario_price_primary"], errors="ignore")

    # fuel_price: technology → fuel mapping then merge with secondary-by-fuel prices
    target["fuel_for_price"] = target["technology"].map(fuel_map).fillna("Gas")
    target = target.merge(sec_by_fuel, on=join_key + ["fuel_for_price"], how="left")

    # Pathway logic (Power/Renewables vs Coal/Gas&Oil)
    # Determine pathway_unit and scenario_pathway from df columns (vectorized)
    has_primary = df["primary_energy_mwh_per_yr"].notna() if "primary_energy_mwh_per_yr" in df.columns else pd.Series(False, index=df.index)
    has_secondary = df["secondary_energy_mwh_per_yr"].notna() if "secondary_energy_mwh_per_yr" in df.columns else pd.Series(False, index=df.index)
    has_capacity = df["capacity_mw"].notna() if "capacity_mw" in df.columns else pd.Series(False, index=df.index)

    # pathway_unit default MWh/yr, set to MW only where appropriate
    target["pathway_unit"] = "MWh/yr"
    idx = (~has_primary & ~has_secondary & has_capacity)
    if len(target) == len(df):
        target.loc[idx, "pathway_unit"] = "MW"

    # scenario_pathway by sector rules
    sec_series = target["sector"].astype(str)
    scenario_pathway = pd.Series(np.nan, index=target.index, dtype=float)
    if "primary_energy_mwh_per_yr" in df.columns:
        scenario_pathway = np.where(sec_series.isin(["Coal", "Gas&Oil"]), df["primary_energy_mwh_per_yr"], scenario_pathway)
    # For Power/Renewables
    if "secondary_energy_mwh_per_yr" in df.columns:
        use_secondary = sec_series.isin(["Power", "Renewables"]) & has_secondary
        scenario_pathway = np.where(use_secondary, df["secondary_energy_mwh_per_yr"], scenario_pathway)
    if "capacity_mw" in df.columns:
        use_capacity = sec_series.isin(["Power", "Renewables"]) & ~has_secondary & has_capacity
        scenario_pathway = np.where(use_capacity, df["capacity_mw"], scenario_pathway)
    target["scenario_pathway"] = scenario_pathway

    # Capacity factor (Power/Renewables only): secondary_energy / (capacity * 8760)
    if "secondary_energy_mwh_per_yr" in df.columns and "capacity_mw" in df.columns:
        denom = df["capacity_mw"] * 8760.0
        cf = np.where(
            sec_series.isin(["Power", "Renewables"]) & df["secondary_energy_mwh_per_yr"].notna() & df["capacity_mw"].notna() & (df["capacity_mw"] > 0),
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
                        parts = [p.strip() for p in str(iso2_raw).split("|") if p.strip()]
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
    target["country_iso2_list"] = target["scenario_geography"].map(iso_map).fillna(target["scenario_geography"])  # fallback

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
    cols = [c for c in cols if c in target.columns]
    target[cols].to_csv(out, index=False)
    print(f"✅ Wrote {out} | Shape: {target[cols].shape}")
    memory_release(df, target)


# ======================================
# Step 4: Aggregation and Gap-Filling
# ======================================

def vectorized_mapping(df_in: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized sector/technology mapping via merge instead of row-wise loops."""
    left = df_in.copy()
    right = mapping_df.rename(
        columns={
            "current_sector": "sector",
            "current_technology": "technology",
        }
    )
    left = left.merge(
        right[["sector", "technology", "target_sector", "target_technology", "aggregation_group"]],
        on=["sector", "technology"],
        how="left",
    )
    # Fallback: if not mapped, keep original
    left["target_sector"] = left["target_sector"].fillna(left["sector"])
    left["target_technology"] = left["target_technology"].fillna(left["technology"])
    left["aggregation_group"] = left["aggregation_group"].fillna("single")
    return left


def gap_fill_grouped_median(df_in: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Vectorized gap-fill: grouped medians via transform with hierarchical fallbacks."""
    df = df_in.copy()
    df["gap_filled_columns"] = ""

    # Levels: (technology, scenario_geography) → (technology)
    levels = [
        ["technology", "scenario_geography"],
        ["technology"],
    ]

    for col in cols:
        if col not in df.columns:
            continue
        missing = df[col].isna()
        original_missing_idx = df.index[missing]
        filled_series = df[col].copy()
        filled_any = np.zeros(len(df), dtype=bool)
        for level in levels:
            keys = [c for c in level if c in df.columns]
            if not keys:
                continue
            med = df.groupby(keys)[col].transform("median")
            need = filled_series.isna() & med.notna()
            filled_series = filled_series.where(~need, med)
            filled_any |= need.to_numpy()
        df[col] = filled_series
        # Track filled positions for this column
        just_filled_idx = df.index[filled_any & df.index.isin(original_missing_idx)]
        if len(just_filled_idx) > 0:
            df.loc[just_filled_idx, "gap_filled_columns"] = (
                df.loc[just_filled_idx, "gap_filled_columns"].replace({"": col})
            )
            # Append if already non-empty
            non_empty_mask = df.loc[just_filled_idx, "gap_filled_columns"].ne(col)
            df.loc[just_filled_idx[non_empty_mask], "gap_filled_columns"] = (
                df.loc[just_filled_idx[non_empty_mask], "gap_filled_columns"] + "," + col
            )
    return df


def step4_aggregate_and_gapfill() -> None:
    print_banner("STEP 4 — Aggregate Technologies and Gap-Fill")
    df = pd.read_csv("3_final_AR6_target_schema.csv")
    try:
        mapping_df = pd.read_csv("technology_mapping.csv")
    except FileNotFoundError:
        print("❌ Missing technology_mapping.csv — aborting step 4")
        return

    # Vectorized mapping from mapping file (first pass)
    df_map = vectorized_mapping(df, mapping_df)
    df_map["sector"] = df_map["target_sector"]
    df_map["technology"] = df_map["target_technology"]

    # Enforce canonical final targets
    FINAL_TARGETS: set[tuple[str, str]] = {
        ("Coal", "Coal"),
        ("Oil&Gas", "Oil"),
        ("Oil&Gas", "Gas"),
        ("Power", "SolarCap"),
        ("Power", "CoalCap"),
        ("Power", "GasCap"),
        ("Power", "OilCap"),
        ("Power", "BiomassCap"),
        ("Power", "WindCap"),
        ("Power", "HydroCap"),
        ("Power", "NuclearCap"),
        ("Power", "GeothermalCap"),
        ("Steel", "BF-BOF"),
        ("Steel", "DRI-BOF"),
        ("Steel", "EAF"),
    }

    def canonicalize_to_final_targets(df_in: pd.DataFrame) -> pd.DataFrame:
        dfc = df_in.copy()
        sec = dfc["sector"].astype(str).str.lower()
        tech = dfc["technology"].astype(str).str.lower()

        # Start with identity
        sec_out = dfc["sector"].astype(str).copy()
        tech_out = dfc["technology"].astype(str).copy()

        # Normalize sector names first
        sec_out = sec_out.mask(sec.isin(["gas&oil", "oil&gas"]), "Oil&Gas")

        # Coal sector → (Coal, Coal)
        coal_mask = sec.eq("coal")
        sec_out = sec_out.mask(coal_mask, "Coal")
        tech_out = tech_out.mask(coal_mask, "Coal")

        # Oil&Gas sector → tech either Oil or Gas
        og_mask = sec.isin(["oil&gas", "gas&oil"]) | sec_out.eq("Oil&Gas")
        gas_mask = og_mask & (tech.str.contains("gas"))
        oil_mask = og_mask & (tech.str.contains("oil"))
        sec_out = sec_out.mask(og_mask, "Oil&Gas")
        tech_out = tech_out.mask(gas_mask, "Gas")
        tech_out = tech_out.mask(oil_mask, "Oil")

        # Power-like sectors (Power, Renewables, Nuclear → Power)
        power_like = sec.isin(["power", "renewables", "nuclear"]) | sec_out.isin(["Power", "Renewables", "Nuclear"])
        sec_out = sec_out.mask(power_like, "Power")

        # Map power technologies to Cap variants
        def map_power_tech(name: str) -> str:
            n = name.lower()
            if any(k in n for k in ["solar", "pv", "csp"]):
                return "SolarCap"
            if "wind" in n:
                return "WindCap"
            if "hydro" in n:
                return "HydroCap"
            if "nuclear" in n:
                return "NuclearCap"
            if "geothermal" in n:
                return "GeothermalCap"
            if any(k in n for k in ["biomass", "bio"]):
                return "BiomassCap"
            if "coal" in n:
                return "CoalCap"
            if "gas" in n:
                return "GasCap"
            if "oil" in n:
                return "OilCap"
            return name

        power_idx = power_like[power_like].index
        tech_out.loc[power_idx] = tech_out.loc[power_idx].apply(map_power_tech)

        # Steel mapping to 3 categories
        steel_mask = sec.eq("steel") | sec_out.eq("Steel")
        def map_steel_tech(name: str) -> str:
            n = name.lower()
            if "dri" in n:
                return "DRI-BOF"
            if "eaf" in n:
                return "EAF"
            # default steel route
            return "BF-BOF"

        steel_idx = steel_mask[steel_mask].index
        sec_out = sec_out.mask(steel_mask, "Steel")
        tech_out.loc[steel_idx] = tech_out.loc[steel_idx].apply(map_steel_tech)

        # Apply canonical
        dfc["sector"] = sec_out
        dfc["technology"] = tech_out

        # Filter to final allowed set
        pair = list(zip(dfc["sector"], dfc["technology"]))
        keep = [p in FINAL_TARGETS for p in pair]
        return dfc.loc[keep].copy()

    before_rows = len(df_map)
    df_map = canonicalize_to_final_targets(df_map)
    after_rows = len(df_map)
    print(f"Canonical targets: kept {after_rows:,}/{before_rows:,} rows")

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
    sum_cols = [c for c in ["scenario_pathway", "capacity_additions_mw_per_yr"] if c in df_map.columns]
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
            out[col] = weighted_avg(group, col, "scenario_pathway" if "scenario_pathway" in group.columns else None) if "scenario_pathway" in group.columns else group[col].mean()
        # Carry-forward non-agg columns (take first)
        carry_cols = [c for c in group.columns if c not in set(sum_cols + avg_cols) and c not in ["target_sector", "target_technology", "aggregation_group"]]
        carry_first = group[carry_cols].iloc[0]
        for c in carry_cols:
            out.setdefault(c, carry_first[c])
        return pd.Series(out)

    aggregated = df_map.groupby(grouping_cols, as_index=False).apply(aggregate_group)
    # groupby.apply with as_index=False returns index columns too; ensure flat frame
    if isinstance(aggregated.columns, pd.MultiIndex):
        aggregated.columns = ["_".join([str(c) for c in tup if c != ""]) for tup in aggregated.columns]

    # Gap-fill selected columns using grouped medians
    gap_cols = [
        c
        for c in [
            "scenario_price",
            "scenario_capacity_factor",
            "lifetime_years",
            "efficiency_decimal",
            "om_cost_usd_per_mw_per_yr",
            "capital_cost_usd_per_mw",
        ]
        if c in aggregated.columns
    ]
    aggregated = gap_fill_grouped_median(aggregated, gap_cols)

    # Stringency column and scenario_type update
    aggregated["stringency"] = aggregated.get("scenario_type", np.nan)
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
    ]
    extra_cols = [c for c in aggregated.columns if c not in base_cols]
    final_cols = [c for c in base_cols + extra_cols if c in aggregated.columns]
    out_file = "4_final_AR6_aggregated.csv"
    aggregated[final_cols].to_csv(out_file, index=False)
    print(f"✅ Wrote {out_file} | Shape: {aggregated[final_cols].shape}")

    # Complete-case filtering
    critical = [c for c in ["scenario_pathway", "scenario_price", "om_cost_usd_per_mw_per_yr", "capital_cost_usd_per_mw"] if c in aggregated.columns]
    complete_mask = aggregated[critical].notna().all(axis=1) if critical else pd.Series(True, index=aggregated.index)
    if "efficiency_decimal" in aggregated.columns:
        eff_cond = ~(
            aggregated["sector"].isin(["Power", "Renewables"]) & aggregated["efficiency_decimal"].isna()
        )
        complete_mask = complete_mask & eff_cond

    final_complete = aggregated.loc[complete_mask].copy()
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


