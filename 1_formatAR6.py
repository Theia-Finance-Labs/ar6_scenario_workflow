"""
AR6 Climate Scenario Data Processing Pipeline

This script processes AR6 climate scenario data to extract cost metrics and price data.
It replicates the functionality of a dbt SQL pipeline in Python.

Input files:
- AR6_Scenarios_Database_ISO3_v1.1.feather: Main AR6 scenario database (country-level)
- AR6_Scenarios_Database_R10_regions_v1.1.feather: AR6 regional data (R10 regions)
- ar6_variables_with_mapping.csv: Variable mappings for sectors/technologies

Output:
- 1_intermediate_AR6_scenario_formatting_ISO3.csv: Processed ISO3 data
- 1_intermediate_AR6_scenario_formatting_R10.csv: Processed R10 data

The pipeline:
1. Loads and melts AR6 data from wide to long format
2. Joins with variable mappings to categorize technologies
3. Extracts cost metrics (OM Cost, Capital Cost, Efficiency) via self-joins
4. Processes price data (primary/secondary energy, carbon, electricity)
5. Filters for rows with complete cost data
6. Outputs final dataset with 30+ columns
"""

import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION - Set which dataset to process
# =============================================================================

# Change this to process different datasets
DATASET_TYPE = "ISO3"  # Options: "ISO3" or "R10"

if DATASET_TYPE == "ISO3":
    INPUT_FILE = "AR6_Scenarios_Database_ISO3_v1.1.csv"
    OUTPUT_FILE = "1_intermediate_AR6_scenario_formatting_ISO3.csv"
elif DATASET_TYPE == "R10":
    INPUT_FILE = "AR6_Scenarios_Database_R10_regions_v1.1.csv"
    OUTPUT_FILE = "1_intermediate_AR6_scenario_formatting_R10.csv"
else:
    raise ValueError("DATASET_TYPE must be 'ISO3' or 'R10'")

print("=" * 80)
print("AR6 CLIMATE SCENARIO DATA PROCESSING PIPELINE")
print("=" * 80)
print(f"Processing: {DATASET_TYPE} dataset")
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")

## =============================================================================
## DATA LOADING AND PREPARATION
## =============================================================================

print("Loading input files...")
source = pd.read_csv(INPUT_FILE)
source = source.loc[source["Model"] == "WITCH 5.0", :]
excel_mapping = pd.read_csv("ar6_variables_with_mapping.csv")

print(f"AR6 data shape: {source.shape}")
print(f"Mapping data shape: {excel_mapping.shape}")

## =============================================================================
## DATA TRANSFORMATION: WIDE TO LONG FORMAT
## =============================================================================

print("Melting AR6 data from wide to long format...")
id_cols = ["Model", "Scenario", "Region", "Variable", "Unit"]

# Filter year columns to only include years > 2020 BEFORE melting (major memory savings)
all_year_cols = [col for col in source.columns if col not in id_cols]
print(f"Total year columns found: {len(all_year_cols)}")

# Convert column names to numeric and filter for years > 2020
year_cols = []
for col in all_year_cols:
    try:
        year = float(col)
        if year > 2020 and year <= 2050:
            year_cols.append(col)
    except ValueError:
        # Skip non-numeric column names
        continue

print(f"Year columns for analysis (2021-2050): {len(year_cols)} columns")
print(
    f"Year range: {min([float(col) for col in year_cols])}-{max([float(col) for col in year_cols])}"
)

# Melt the data (years are columns, need to convert to rows)
melted = pd.melt(
    source, id_vars=id_cols, value_vars=year_cols, var_name="Year", value_name="Value"
)

# Data cleaning: convert types and filter out missing/invalid values
melted["Year"] = pd.to_numeric(melted["Year"], errors="coerce")
# Removed filtering - keep all data including missing values and extreme values
melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")

print(f"Melted data shape: {melted.shape}")

# Rename columns for consistency (matching dbt naming convention)
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

## =============================================================================
## VARIABLE MAPPING AND BASE DATA CREATION
## =============================================================================

print("Joining with mapping data...")
# Join scenario data with variable mappings to categorize technologies
base_data = (
    renamed.merge(excel_mapping, how="left", left_on="variable", right_on="variable")
    # Removed dropna and drop_duplicates filtering - keep all data
)

print(f"Base data shape after join: {base_data.shape}")

# Filter out rows where Sector is NA (unmapped variables)
print("Filtering out rows with missing Sector mapping...")
initial_rows = len(base_data)
base_data = base_data[base_data["Sector"].notna()].copy()
final_rows = len(base_data)
removed_rows = initial_rows - final_rows

print(f"   Before Sector filtering: {initial_rows:,} rows")
print(f"   After Sector filtering: {final_rows:,} rows")
print(
    f"   Removed {removed_rows:,} rows ({removed_rows/initial_rows*100:.1f}%) with missing Sector mapping"
)

print(f"   Available sectors: {sorted(base_data['Sector'].unique())}")

# Filter for only target sectors of interest
print("Filtering for target sectors only...")
target_sectors = [
    "Steel",
    "Nuclear",
    "Gas&Oil",
    "Cement",
    "Coal",
    "Renewables",
    "Power",
]
initial_rows_sector = len(base_data)
base_data = base_data[base_data["Sector"].isin(target_sectors)].copy()
final_rows_sector = len(base_data)
removed_rows_sector = initial_rows_sector - final_rows_sector

print(f"   Before target sector filtering: {initial_rows_sector:,} rows")
print(f"   After target sector filtering: {final_rows_sector:,} rows")
print(
    f"   Removed {removed_rows_sector:,} rows ({removed_rows_sector/initial_rows_sector*100:.1f}%) from non-target sectors"
)

print(f"   Final sectors: {sorted(base_data['Sector'].unique())}")

## =============================================================================
## COST METRICS EXTRACTION
## =============================================================================

# Define join columns for cost metric self-joins
join_cols = ["model", "scenario", "region", "year", "Sector", "Subsector", "Technology"]


def get_metric(df, metric_name):
    """Extract a specific cost metric (OM Cost, Capital Cost, Efficiency) from base data"""
    metric_data = df[df["col1"] == metric_name]
    if metric_data.empty:
        return pd.DataFrame(
            columns=join_cols
            + [
                f"{metric_name.lower().replace(' ', '_')}",
                f"{metric_name.lower().replace(' ', '_')}_unit",
            ]
        )

    return metric_data[join_cols + ["value", "unit"]].rename(
        columns={
            "value": f"{metric_name.lower().replace(' ', '_')}",
            "unit": f"{metric_name.lower().replace(' ', '_')}_unit",
        }
    )


print("Extracting cost and efficiency metrics...")
# Self-joins to extract different cost metrics
om_cost = get_metric(base_data, "OM Cost")
efficiency = get_metric(base_data, "Efficiency")
capital_cost = get_metric(base_data, "Capital Cost")

print(f"OM Cost rows: {len(om_cost)}")
print(f"Efficiency rows: {len(efficiency)}")
print(f"Capital Cost rows: {len(capital_cost)}")

# Merge all cost metrics back onto base data
print("Merging cost metrics...")
merged = base_data.copy()

if not om_cost.empty:
    merged = merged.merge(om_cost, on=join_cols, how="left")
else:
    merged["om_cost"] = np.nan
    merged["om_cost_unit"] = np.nan

if not efficiency.empty:
    merged = merged.merge(efficiency, on=join_cols, how="left")
else:
    merged["efficiency"] = np.nan
    merged["efficiency_unit"] = np.nan

if not capital_cost.empty:
    merged = merged.merge(capital_cost, on=join_cols, how="left")
else:
    merged["capital_cost"] = np.nan
    merged["capital_cost_unit"] = np.nan

print(f"Merged data shape: {merged.shape}")

## =============================================================================
## COST DATA FILTERING AND EFFICIENCY RULES
## =============================================================================

print("Filtering for rows with cost metrics...")
# Removed cost completeness filtering - keep all rows regardless of cost data availability
filtered = merged.copy()

print(f"Filtered data shape: {filtered.shape}")

## =============================================================================
## PRICE DATA PROCESSING
## =============================================================================
# Extract and process price data to create fuel-specific price columns
# Grouping: model, scenario, region, year (and Fuel for fuel-specific prices)
print("Processing price data...")
price_rows = base_data[base_data["col1"] == "Price"].copy()
print(f"Price rows found: {len(price_rows)}")

if not price_rows.empty:
    # Clean and prepare price data
    price_rows = price_rows[
        ["model", "scenario", "region", "year", "Fuel", "col2", "value", "unit"]
    ].copy()
    # Removed dropna filtering - keep all price data
    price_rows = price_rows.rename(columns={"col2": "energy_type", "value": "price"})

    print(f"Price rows after cleaning: {len(price_rows)}")
    print("Unique energy types:", price_rows["energy_type"].unique())

    # Separate price data by energy type for different processing
    primary_prices = price_rows[price_rows["energy_type"] == "Primary Energy"].copy()
    secondary_prices = price_rows[
        price_rows["energy_type"] == "Secondary Energy"
    ].copy()
    final_prices = price_rows[price_rows["energy_type"] == "Final Energy"].copy()
    carbon_prices = price_rows[price_rows["energy_type"] == "Carbon"].copy()

    # Extract electricity-specific price data (separate columns)
    secondary_electricity = secondary_prices[
        secondary_prices["Fuel"] == "Electricity"
    ].copy()
    final_electricity = final_prices[final_prices["Fuel"] == "Electricity"].copy()

    # Start with filtered cost data
    final = filtered.copy()

    # 1. FUEL-SPECIFIC PRIMARY ENERGY PRICES (Memory-Optimized)
    # Maps the primary energy price for whatever fuel is in the 'Fuel' column
    if not primary_prices.empty:
        print("Processing primary energy prices...")

        # Memory-efficient approach: direct merge instead of pivot
        # Prepare price data with clean column names
        primary_clean = primary_prices[
            ["model", "scenario", "region", "year", "Fuel", "price", "unit"]
        ].copy()
        primary_clean = primary_clean.rename(
            columns={
                "price": "primary_energy_price",
                "unit": "primary_energy_price_unit",
            }
        )

        # Removed extreme value filtering - keep all price data

        # Merge directly on matching fuel
        final = final.merge(
            primary_clean,
            on=["model", "scenario", "region", "year", "Fuel"],
            how="left",
        )

        print(
            f"     ✅ Added primary energy prices: {final['primary_energy_price'].notna().sum():,} values"
        )

    else:
        final["primary_energy_price"] = np.nan
        final["primary_energy_price_unit"] = np.nan

    # 2. FUEL-SPECIFIC SECONDARY ENERGY PRICES (Memory-Optimized)
    # Maps the secondary energy price for whatever fuel is in the 'Fuel' column
    if not secondary_prices.empty:
        print("Processing secondary energy prices...")

        # Memory-efficient approach: direct merge instead of pivot
        secondary_clean = secondary_prices[
            ["model", "scenario", "region", "year", "Fuel", "price", "unit"]
        ].copy()
        secondary_clean = secondary_clean.rename(
            columns={
                "price": "secondary_energy_price",
                "unit": "secondary_energy_price_unit",
            }
        )

        # Removed extreme value filtering - keep all price data

        # Merge directly on matching fuel
        final = final.merge(
            secondary_clean,
            on=["model", "scenario", "region", "year", "Fuel"],
            how="left",
        )

        print(
            f"     ✅ Added secondary energy prices: {final['secondary_energy_price'].notna().sum():,} values"
        )

    else:
        final["secondary_energy_price"] = np.nan
        final["secondary_energy_price_unit"] = np.nan

    # 3. CARBON PRICE (single column, not fuel-specific)
    # All technologies face the same carbon price in a given model/scenario/region/year
    if not carbon_prices.empty:
        print("Processing carbon prices...")

        carbon_pivot = (
            carbon_prices.groupby(["model", "scenario", "region", "year"])["price"]
            .first()
            .reset_index()
        )
        carbon_units_pivot = (
            carbon_prices.groupby(["model", "scenario", "region", "year"])["unit"]
            .first()
            .reset_index()
        )

        final = final.merge(
            carbon_pivot.rename(columns={"price": "carbon_price"}),
            on=["model", "scenario", "region", "year"],
            how="left",
        )
        final = final.merge(
            carbon_units_pivot.rename(columns={"unit": "carbon_price_unit"}),
            on=["model", "scenario", "region", "year"],
            how="left",
        )
    else:
        final["carbon_price"] = np.nan
        final["carbon_price_unit"] = np.nan

    # 4. ELECTRICITY-SPECIFIC PRICES (separate columns for secondary and final energy)
    # These are separate from fuel-specific prices above because electricity is special
    if not secondary_electricity.empty:
        print("Processing secondary electricity prices...")

        sec_elec_pivot = (
            secondary_electricity.groupby(["model", "scenario", "region", "year"])[
                "price"
            ]
            .first()
            .reset_index()
        )
        sec_elec_units_pivot = (
            secondary_electricity.groupby(["model", "scenario", "region", "year"])[
                "unit"
            ]
            .first()
            .reset_index()
        )

        final = final.merge(
            sec_elec_pivot.rename(
                columns={"price": "secondary_energy_electricity_price"}
            ),
            on=["model", "scenario", "region", "year"],
            how="left",
        )
        final = final.merge(
            sec_elec_units_pivot.rename(
                columns={"unit": "secondary_energy_electricity_price_unit"}
            ),
            on=["model", "scenario", "region", "year"],
            how="left",
        )
    else:
        final["secondary_energy_electricity_price"] = np.nan
        final["secondary_energy_electricity_price_unit"] = np.nan

    if not final_electricity.empty:
        print("Processing final electricity prices...")

        final_elec_pivot = (
            final_electricity.groupby(["model", "scenario", "region", "year"])["price"]
            .first()
            .reset_index()
        )
        final_elec_units_pivot = (
            final_electricity.groupby(["model", "scenario", "region", "year"])["unit"]
            .first()
            .reset_index()
        )

        final = final.merge(
            final_elec_pivot.rename(
                columns={"price": "final_energy_electricity_price"}
            ),
            on=["model", "scenario", "region", "year"],
            how="left",
        )
        final = final.merge(
            final_elec_units_pivot.rename(
                columns={"unit": "final_energy_electricity_price_unit"}
            ),
            on=["model", "scenario", "region", "year"],
            how="left",
        )
    else:
        final["final_energy_electricity_price"] = np.nan
        final["final_energy_electricity_price_unit"] = np.nan

    print(f"Final data shape after price merge: {final.shape}")

    # Report price data coverage
    price_cols = [
        "primary_energy_price",
        "secondary_energy_price",
        "carbon_price",
        "secondary_energy_electricity_price",
        "final_energy_electricity_price",
    ]
    for col in price_cols:
        if col in final.columns:
            non_na_count = final[col].notna().sum()
            print(f"{col}: {non_na_count} non-NA values")

else:
    print("No price data found, proceeding without price columns")
    final = filtered.copy()

## =============================================================================
## PRICE DATA FILTERING (CRITICAL FOR COMPLETE DATASETS)
## =============================================================================
print("\n🎯 PRICE DATA FILTERING REMOVED...")
print("   Keeping all rows regardless of price data completeness")

initial_rows = len(final)
print(f"   Total rows: {initial_rows:,} rows")

# Removed all price filtering - keep all data regardless of price completeness
final_rows = len(final)

print(f"   No rows removed - all data retained")

# Report remaining data by sector
print("   Data by sector (no filtering applied):")
for sector in sorted(final["Sector"].unique()):
    sector_count = len(final[final["Sector"] == sector])
    sector_models = final[final["Sector"] == sector]["model"].nunique()
    print(f"     📊 {sector}: {sector_count:,} rows from {sector_models} models")

## =============================================================================
## FINAL OUTPUT PREPARATION
## =============================================================================

print("Preparing final output...")

# Define column order for final output
final_cols = [
    # Core scenario data
    "model",
    "scenario",
    "region",
    "variable",
    "value",
    "unit",
    "year",
    # Technology categorization
    "Sector",
    "Subsector",
    "Technology",
    "Fuel",
    "CCS_flag",
    # Cost metrics
    "om_cost",
    "om_cost_unit",
    "efficiency",
    "efficiency_unit",
    "capital_cost",
    "capital_cost_unit",
    # Price data (if available)
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
    # Additional mapping columns
    "col1",
    "col2",
    "col3",
    "col4",
    "col5",
]

# Select only columns that exist in the dataframe
final_cols = [col for col in final_cols if col in final.columns]
final = final[final_cols]

print(f"Final output shape: {final.shape}")
print(f"Final columns: {len(final_cols)}")

# Output to CSV
final.to_csv(OUTPUT_FILE, index=False)
print(f"Output written to {OUTPUT_FILE}")

## =============================================================================
## SUMMARY AND VALIDATION
## =============================================================================

print("\n" + "=" * 60)
print("PROCESSING SUMMARY")
print("=" * 60)

print(f"Dataset: {DATASET_TYPE}")
print(f"Final dataset: {len(final):,} rows × {len(final.columns)} columns")
print(f"Unique models: {final['model'].nunique()}")
print(f"Unique scenarios: {final['scenario'].nunique()}")
print(f"Unique regions: {final['region'].nunique()}")
print(f"Year range: {final['year'].min()}-{final['year'].max()}")

print("\n" + "=" * 60)
print("PROCESSING COMPLETE!")
print("=" * 60)
