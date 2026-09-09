"""Fuel-price lineage assertions and provider/fuel/geography quality checks."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


FOSSIL_FUELS = {"Coal", "Gas", "Oil"}
PRIMARY_PREFIX = "Price|Primary Energy|"
ALLOWED_FOSSIL_SOURCE_VARIABLES = {
    f"{PRIMARY_PREFIX}{fuel}" for fuel in FOSSIL_FUELS
}
SLOPE_LIMIT_TCO2_PER_MWH = 0.02
MAX_ADJACENT_NODE_RATIO = 2.0


def classify_scenario_type(stringency: pd.Series) -> pd.Series:
    """Map AR6 C7/C8 to baseline and every other category to target."""
    return pd.Series(
        np.where(stringency.isin(["C7", "C8"]), "baseline", "target"),
        index=stringency.index,
    )


def normalize_fuel(value: object) -> str:
    text = str(value).strip().lower()
    if "coal" in text or "solid" in text:
        return "Coal"
    if "gas" in text or "gases" in text:
        return "Gas"
    if "oil" in text or "liquid" in text:
        return "Oil"
    if "biomass" in text or text.startswith("bio"):
        return "Biomass"
    return str(value).strip().title() if pd.notna(value) else ""


def canonical_primary_variable(fuel: object) -> str | None:
    normalized = normalize_fuel(fuel)
    return f"{PRIMARY_PREFIX}{normalized}" if normalized in FOSSIL_FUELS else None


def _source_components(value: object) -> Iterable[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def _coerce_boolean(series: pd.Series) -> pd.Series:
    """Parse persisted CSV booleans without treating the string 'False' as true."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(True)
    normalized = series.astype("string").str.strip().str.lower()
    parsed = normalized.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
        }
    )
    return parsed.fillna(True).astype(bool)


def assert_primary_fuel_price_provenance(df: pd.DataFrame) -> None:
    """Reject any priced fossil row without exclusively primary-energy lineage."""
    required = {
        "fuel_price",
        "fuel_for_price",
        "fuel_price_source_variable",
        "fuel_price_source_unit",
        "fuel_price_carbon_adjusted",
        "fuel_price_carbon_coefficient",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise AssertionError(f"Missing fuel-price provenance columns: {missing}")

    fuels = df["fuel_for_price"].map(normalize_fuel)
    fossil = fuels.isin(FOSSIL_FUELS) & df["fuel_price"].notna()
    bad_rows = []
    for idx, value in df.loc[fossil, "fuel_price_source_variable"].items():
        components = list(_source_components(value))
        if not components or any(
            component not in ALLOWED_FOSSIL_SOURCE_VARIABLES
            or "Secondary Energy" in component
            or "Final Energy" in component
            for component in components
        ):
            bad_rows.append(idx)

    missing_units = df.loc[fossil, "fuel_price_source_unit"].isna()
    adjusted = _coerce_boolean(df.loc[fossil, "fuel_price_carbon_adjusted"])
    coefficients = pd.to_numeric(
        df.loc[fossil, "fuel_price_carbon_coefficient"], errors="coerce"
    )
    bad_coefficients = coefficients.isna() | ~np.isclose(coefficients, 0.0)

    problems = []
    if bad_rows:
        problems.append(f"{len(bad_rows)} non-primary/missing source variables")
    if missing_units.any():
        problems.append(f"{int(missing_units.sum())} missing source units")
    if adjusted.any():
        problems.append(f"{int(adjusted.sum())} carbon-adjusted rows")
    if bad_coefficients.any():
        problems.append(f"{int(bad_coefficients.sum())} non-zero/missing coefficients")
    if problems:
        raise AssertionError("Invalid fossil fuel-price provenance: " + "; ".join(problems))


def _node_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["fuel"] = work["fuel_for_price"].map(normalize_fuel)
    work = work[work["fuel"].isin(FOSSIL_FUELS) & work["fuel_price"].notna()].copy()
    work["fuel_price"] = pd.to_numeric(work["fuel_price"], errors="coerce")
    work["carbon_price_usd_per_tco2"] = pd.to_numeric(
        work.get("carbon_price_usd_per_tco2"), errors="coerce"
    )
    keys = [
        "scenario_provider",
        "scenario",
        "scenario_type",
        "scenario_geography",
        "scenario_year",
        "fuel",
    ]
    return (
        work.groupby(keys, dropna=False, as_index=False)
        .agg(
            fuel_price=("fuel_price", "first"),
            distinct_prices=("fuel_price", "nunique"),
            carbon_price_usd_per_tco2=("carbon_price_usd_per_tco2", "max"),
        )
    )


def validate_fuel_price_quality(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate rows and return one validation record per provider/fuel/geography."""
    nodes = _node_table(df)
    group_keys = ["scenario_provider", "fuel", "scenario_geography"]
    report_rows = []

    if nodes.empty:
        result = df.copy()
        result["fuel_price_quality_pass"] = True
        return result, pd.DataFrame()

    baseline = (
        nodes[nodes["scenario_type"].eq("baseline")]
        .groupby(group_keys + ["scenario_year"], dropna=False, as_index=False)[
            "fuel_price"
        ]
        .median()
        .rename(columns={"fuel_price": "baseline_fuel_price"})
    )
    target = nodes[nodes["scenario_type"].eq("target")].merge(
        baseline, on=group_keys + ["scenario_year"], how="left"
    )

    all_groups = nodes[group_keys].drop_duplicates()
    for group in all_groups.itertuples(index=False, name=None):
        selector = pd.Series(True, index=nodes.index)
        for key, value in zip(group_keys, group):
            selector &= nodes[key].eq(value)
        group_nodes = nodes[selector]

        target_selector = pd.Series(True, index=target.index)
        for key, value in zip(group_keys, group):
            target_selector &= target[key].eq(value)
        pairs = target[target_selector].dropna(
            subset=[
                "fuel_price",
                "baseline_fuel_price",
                "carbon_price_usd_per_tco2",
            ]
        )
        slope_testable = (
            len(pairs) >= 3
            and pairs["carbon_price_usd_per_tco2"].nunique() >= 2
        )
        slope = np.nan
        if slope_testable:
            x = pairs["carbon_price_usd_per_tco2"].to_numpy(dtype=float)
            y = (pairs["fuel_price"] - pairs["baseline_fuel_price"]).to_numpy(
                dtype=float
            )
            slope = float(np.polyfit(x, y, 1)[0])
        slope_pass = bool(slope_testable and abs(slope) < SLOPE_LIMIT_TCO2_PER_MWH)

        continuity_testable = False
        continuity_pass = True
        continuity_violations = 0
        non_positive_count = int((group_nodes["fuel_price"] <= 0).sum())
        for _, series in group_nodes.groupby(
            ["scenario_provider", "scenario", "scenario_geography", "fuel"],
            dropna=False,
        ):
            series = series.sort_values("scenario_year")
            previous = series.shift(1)
            adjacent = series["scenario_year"].sub(previous["scenario_year"]).eq(5)
            if not adjacent.any():
                continue
            continuity_testable = True
            current_price = series.loc[adjacent, "fuel_price"].astype(float)
            previous_price = previous.loc[adjacent, "fuel_price"].astype(float)
            non_positive = (current_price <= 0) | (previous_price <= 0)
            ratio = np.maximum(current_price.abs(), previous_price.abs()) / np.minimum(
                current_price.abs(), previous_price.abs()
            )
            failures = non_positive | (ratio > MAX_ADJACENT_NODE_RATIO)
            continuity_violations += int(failures.sum())

        conflicts = int((group_nodes["distinct_prices"] > 1).sum())
        continuity_pass = bool(
            continuity_testable
            and continuity_violations == 0
            and conflicts == 0
            and non_positive_count == 0
        )
        quality_pass = bool(slope_pass and continuity_pass)
        reasons = []
        if not slope_testable:
            reasons.append("slope_untestable")
        elif not slope_pass:
            reasons.append("slope_limit")
        if not continuity_testable:
            reasons.append("continuity_untestable")
        elif continuity_violations:
            reasons.append("continuity_ratio")
        if conflicts:
            reasons.append("conflicting_node_prices")
        if non_positive_count:
            reasons.append("non_positive_fossil_price")

        report_rows.append(
            {
                **dict(zip(group_keys, group)),
                "slope_pair_count": len(pairs),
                "slope_testable": slope_testable,
                "carbon_slope_tco2_per_mwh": slope,
                "slope_pass": slope_pass,
                "continuity_testable": continuity_testable,
                "continuity_violation_count": continuity_violations,
                "conflicting_node_count": conflicts,
                "non_positive_node_count": non_positive_count,
                "continuity_pass": continuity_pass,
                "fuel_price_quality_pass": quality_pass,
                "failure_reasons": ";".join(reasons),
            }
        )

    report = pd.DataFrame(report_rows)
    result = df.copy()
    result["_quality_fuel"] = result["fuel_for_price"].map(normalize_fuel)
    result = result.merge(
        report[group_keys + ["fuel_price_quality_pass"]],
        left_on=["scenario_provider", "_quality_fuel", "scenario_geography"],
        right_on=group_keys,
        how="left",
    )
    fossil = result["_quality_fuel"].isin(FOSSIL_FUELS)
    result.loc[~fossil, "fuel_price_quality_pass"] = True
    result.loc[fossil, "fuel_price_quality_pass"] = result.loc[
        fossil, "fuel_price_quality_pass"
    ].fillna(False)
    result["fuel_price_quality_pass"] = result["fuel_price_quality_pass"].astype(bool)
    result = result.drop(columns=["_quality_fuel", "fuel"], errors="ignore")
    return result, report
