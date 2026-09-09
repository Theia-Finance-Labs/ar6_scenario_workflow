from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from combined_pipeline import (  # noqa: E402
    _iter_csv_model_partitions,
    build_price_tables,
    capacity_pathway_mask,
    out_of_bounds_mask,
)
from fuel_price_validation import (  # noqa: E402
    assert_primary_fuel_price_provenance,
    classify_scenario_type,
    validate_fuel_price_quality,
)
from step4_gapfill_simple import ultra_fast_gap_fill  # noqa: E402


def _price_row(**overrides):
    row = {
        "model": "Provider",
        "scenario": "Target",
        "region": "Region",
        "year": 2040,
        "col1": "Price",
        "col2": "Primary Energy",
        "Fuel": "Gas",
        "value": 70.0,
        "unit": "US$2010/GJ",
        "variable": "Price|Primary Energy|Gas",
    }
    row.update(overrides)
    return row


def _gapfill_row(**overrides):
    row = {
        "scenario_provider": "Provider",
        "scenario": "Target",
        "scenario_type": "target",
        "stringency": "C3",
        "scenario_geography": "Region",
        "scenario_year": 2040,
        "sector": "Power",
        "technology": "GasCap",
        "fuel_for_price": "Gas",
        "country_iso2_list": "GLOBAL",
        "fuel_price": np.nan,
        "scenario_price": 250.0,
        "gap_filled_columns": "",
        "fuel_price_source_variable": np.nan,
        "fuel_price_source_unit": np.nan,
    }
    row.update(overrides)
    return row


def test_primary_price_above_200_keeps_value_and_lineage():
    tables = build_price_tables(pd.DataFrame([_price_row()]))
    gas = tables["primary_by_fuel"].iloc[0]
    assert gas["price_usd_per_mwh"] == pytest.approx(252.0)
    assert gas["fuel_price_source_variable"] == "Price|Primary Energy|Gas"
    assert gas["fuel_price_source_unit"] == "US$2010/GJ"
    assert not out_of_bounds_mask(pd.Series([252.0]), 0.0, None).iloc[0]


def test_csv_streaming_keeps_each_model_in_one_partition(tmp_path):
    source = pd.DataFrame(
        [
            _price_row(model="A", year_value=1.0),
            _price_row(model="B", year_value=2.0),
            _price_row(model="A", year_value=3.0),
        ]
    ).rename(columns={"year_value": "2030"})
    source = source[["model", "scenario", "region", "variable", "unit", "2030"]]
    source.columns = ["Model", "Scenario", "Region", "Variable", "Unit", "2030"]
    path = tmp_path / "ar6.csv"
    source.to_csv(path, index=False)
    partitions = list(
        _iter_csv_model_partitions(
            path, {"Price|Primary Energy|Gas"}, partition_count=2, chunk_size=1
        )
    )
    seen = {}
    for partition_number, partition in enumerate(partitions):
        for model in partition["Model"].unique():
            assert model not in seen
            seen[model] = partition_number
    assert set(seen) == {"A", "B"}


def test_partitioned_step1_uses_one_fixed_csv_schema(tmp_path, monkeypatch):
    source = pd.DataFrame(
        [
            {
                "Model": "Provider",
                "Scenario": "Scenario",
                "Region": "CHN",
                "Variable": "Price|Primary Energy|Gas",
                "Unit": "US$2010/GJ",
                "2030": 6.0,
            }
        ]
    )
    mapping = pd.DataFrame(
        [
            {
                "variable": "Price|Primary Energy|Gas",
                "Sector": "Power",
                "Subsector": "Power",
                "Technology": "GasCap",
                "Fuel": "Gas",
                "CCS_flag": False,
                "col1": "Price",
                "col2": "Primary Energy",
                "col3": np.nan,
                "col4": np.nan,
                "col5": np.nan,
            }
        ]
    )
    source_path = tmp_path / "AR6_Scenarios_Database_ISO3_v1.1.csv"
    source.to_csv(source_path, index=False)
    mapping.to_csv(tmp_path / "ar6_variables_with_mapping.csv", index=False)
    monkeypatch.chdir(tmp_path)

    from combined_pipeline import step1_process_dataset

    output = step1_process_dataset("ISO3", str(source_path))
    written = pd.read_csv(output)
    assert {
        "om_cost",
        "efficiency",
        "capital_cost",
        "primary_energy_price",
        "carbon_price",
    } <= set(written.columns)


def test_c7_c8_are_provider_wide_baselines_and_capacity_rows_use_mw():
    categories = pd.Series(["C1", "C6", "C7", "C8", "UNKNOWN", np.nan])
    assert classify_scenario_type(categories).tolist() == [
        "target",
        "target",
        "baseline",
        "baseline",
        "target",
        "target",
    ]
    mask = capacity_pathway_mask(
        pd.Series(["Power", "Renewables", "Coal"]),
        pd.Series([100.0, 25.0, 50.0]),
    )
    assert mask.tolist() == [True, True, False]


def test_baseline_fill_precedes_lookup_and_preserves_primary_provenance():
    rows = [
        _gapfill_row(fuel_for_price="Gases"),
        _gapfill_row(
            scenario="Baseline",
            scenario_type="baseline",
            stringency="C7",
            fuel_price=24.0,
            fuel_price_source_variable="Price|Primary Energy|Gas",
            fuel_price_source_unit="US$2010/GJ",
        ),
    ]
    lookup = pd.DataFrame(
        [
            {
                "technology": "GasCap",
                "year": 2040,
                "stringency": "C3",
                "iso2": "GLOBAL",
                "fuel_price_usd_per_mwh": 99.0,
                "electricity_price_usd_per_mwh": 88.0,
            }
        ]
    )
    result = ultra_fast_gap_fill(pd.DataFrame(rows), lookup)
    target = result[result["scenario"].eq("Target")].iloc[0]
    assert target["fuel_price"] == pytest.approx(24.0)
    assert target["fuel_price_fallback"] == "baseline"
    assert target["fuel_price_source_variable"] == "Price|Primary Energy|Gas"
    assert target["fuel_price_carbon_adjusted"] is False or not bool(
        target["fuel_price_carbon_adjusted"]
    )
    assert target["scenario_price"] == pytest.approx(250.0)


def test_lookup_fill_records_exact_strategy():
    lookup = pd.DataFrame(
        [
            {
                "technology": "GasCap",
                "year": 2040,
                "stringency": "C3",
                "iso2": "GLOBAL",
                "fuel_price_usd_per_mwh": 31.0,
                "electricity_price_usd_per_mwh": 80.0,
            }
        ]
    )
    result = ultra_fast_gap_fill(pd.DataFrame([_gapfill_row()]), lookup)
    row = result.iloc[0]
    assert row["fuel_price"] == pytest.approx(31.0)
    assert row["fuel_price_fallback"] == "lookup-direct"
    assert row["fuel_price_source_variable"] == "Price|Primary Energy|Gas"
    assert_primary_fuel_price_provenance(result)


def test_direct_fossil_price_without_lineage_fails_fast():
    row = _gapfill_row(
        fuel_price=24.0,
        fuel_price_source_variable=np.nan,
        fuel_price_source_unit=np.nan,
    )
    with pytest.raises(AssertionError, match="provenance"):
        assert_primary_fuel_price_provenance(pd.DataFrame([row]).assign(
            fuel_price_carbon_adjusted=False,
            fuel_price_carbon_coefficient=0.0,
        ))


def _quality_fixture(target_prices=(10.1, 10.2, 10.3)):
    rows = []
    for year, carbon, target_price in zip(
        [2030, 2035, 2040], [10.0, 20.0, 30.0], target_prices
    ):
        common = {
            "scenario_provider": "Provider",
            "scenario_geography": "Region",
            "scenario_year": year,
            "sector": "Power",
            "technology": "GasCap",
            "fuel_for_price": "Gas",
            "fuel_price_source_variable": "Price|Primary Energy|Gas",
            "fuel_price_source_unit": "US$2010/GJ",
            "fuel_price_carbon_adjusted": False,
            "fuel_price_carbon_coefficient": 0.0,
            "fuel_price_fallback": "none",
        }
        rows.append(
            {
                **common,
                "scenario": "Baseline",
                "scenario_type": "baseline",
                "fuel_price": 10.0,
                "carbon_price_usd_per_tco2": 0.0,
            }
        )
        rows.append(
            {
                **common,
                "scenario": "Target",
                "scenario_type": "target",
                "fuel_price": target_price,
                "carbon_price_usd_per_tco2": carbon,
            }
        )
    return pd.DataFrame(rows)


def test_carbon_slope_quality_threshold_and_continuity():
    annotated, report = validate_fuel_price_quality(_quality_fixture())
    group = report.iloc[0]
    assert group["carbon_slope_tco2_per_mwh"] == pytest.approx(0.01)
    assert bool(group["slope_pass"])
    assert bool(group["continuity_pass"])
    assert annotated["fuel_price_quality_pass"].all()


def test_adjacent_five_year_jump_is_flagged_without_dropping_rows():
    source = _quality_fixture(target_prices=(10.1, 30.0, 30.1))
    annotated, report = validate_fuel_price_quality(source)
    assert len(annotated) == len(source)
    assert report["continuity_violation_count"].iloc[0] == 1
    assert not bool(report["fuel_price_quality_pass"].iloc[0])
    assert not annotated["fuel_price_quality_pass"].any()


FULL_OUTPUT = os.environ.get("AR6_SCENARIOS_CSV")


@pytest.mark.skipif(not FULL_OUTPUT, reason="set AR6_SCENARIOS_CSV for full-data checks")
def test_full_output_iiasa_acceptance_and_provenance():
    df = pd.read_csv(FULL_OUTPUT, low_memory=False)
    assert_primary_fuel_price_provenance(df)
    adjusted = (
        df["fuel_price_carbon_adjusted"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )
    assert adjusted.notna().all()
    assert not adjusted.any()

    expected = [
        ("WITCH 5.0", "EN_NPi2020_500", "CHN", "Gas", 2035, 23.8703),
        ("WITCH 5.0", "EN_NPi2020_500", "CHN", "Gas", 2040, 24.0694),
        ("WITCH 5.0", "EN_NPi2020_500", "CHN", "Coal", 2035, 6.6356),
        ("WITCH 5.0", "EN_NPi2020_500", "CHN", "Coal", 2040, 6.7125),
        ("WITCH 5.0", "EN_NPi2020_500", "CHN", "Coal", 2045, 6.7841),
        ("IMAGE 3.0", "CO_2Deg2020", "R10CHINA+", "Gas", 2030, 39.0211),
        ("IMAGE 3.0", "CO_2Deg2020", "R10CHINA+", "Gas", 2040, 30.6327),
        (
            "REMIND-MAgPIE 2.1-4.2",
            "SusDev_SSP2-PkBudg900",
            "R10CHINA+",
            "Gas",
            2030,
            20.6993,
        ),
        (
            "REMIND-MAgPIE 2.1-4.2",
            "SusDev_SSP2-PkBudg900",
            "R10CHINA+",
            "Gas",
            2040,
            24.5538,
        ),
    ]
    normalized_fuel = df["fuel_for_price"].astype(str).str.title()
    for provider, scenario, geography, fuel, year, value in expected:
        matches = df[
            df["scenario_provider"].eq(provider)
            & df["scenario"].eq(scenario)
            & df["scenario_geography"].eq(geography)
            & normalized_fuel.eq(fuel)
            & pd.to_numeric(df["scenario_year"], errors="coerce").eq(year)
        ]["fuel_price"].dropna()
        assert not matches.empty, (provider, scenario, geography, fuel, year)
        assert np.isclose(matches.astype(float), value, atol=0.0005).all()


@pytest.mark.skipif(not FULL_OUTPUT, reason="set AR6_SCENARIOS_CSV for full-data checks")
def test_full_output_slope_and_node_continuity():
    df = pd.read_csv(FULL_OUTPUT, low_memory=False)
    _, report = validate_fuel_price_quality(df)
    testable_slopes = report[report["slope_testable"]]
    assert not testable_slopes.empty
    assert testable_slopes["slope_pass"].all()
    testable_continuity = report[report["continuity_testable"]]
    assert not testable_continuity.empty
    assert testable_continuity["continuity_pass"].all()

    required_groups = {
        ("WITCH 5.0", "Gas", "CHN"),
        ("WITCH 5.0", "Coal", "CHN"),
        ("IMAGE 3.0", "Gas", "R10CHINA+"),
        ("REMIND-MAgPIE 2.1-4.2", "Gas", "R10CHINA+"),
    }
    observed_testable = set(
        report.loc[
            report["slope_testable"] & report["continuity_testable"],
            ["scenario_provider", "fuel", "scenario_geography"],
        ].itertuples(index=False, name=None)
    )
    assert required_groups <= observed_testable
