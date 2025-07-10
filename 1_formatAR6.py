"""
AR6 Climate Scenario Data Processing Pipeline

This script processes AR6 climate scenario data to extract cost metrics and price data.
It replicates the functionality of a dbt SQL pipeline in Python.

Input files:
- AR6_Scenarios_Database_ISO3_v1.1.csv: Main AR6 scenario database
- ar6_variables_with_mapping.csv: Variable mappings for sectors/technologies

Output:
- AR6_scenario_with_costs_and_prices.csv: Processed data with cost metrics and price data

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

## =============================================================================
## DATA LOADING AND PREPARATION
## =============================================================================

print("Loading input files...")
source = pd.read_feather("AR6_Scenarios_Database_ISO3_v1.1.feather")
excel_mapping = pd.read_csv("ar6_variables_with_mapping.csv")

print(f"AR6 data shape: {source.shape}")
print(f"Mapping data shape: {excel_mapping.shape}")

# Uncomment the lines below to test with a single model-scenario combination
# print("Filtering to Model-Scenario combination with most cost data for testing...")
# best_model = "POLES GECO2019"
# best_scenario = "NDCMCS"
# print(f"Using Model: {best_model}, Scenario: {best_scenario}")
# source = source[(source['Model'] == best_model) & (source['Scenario'] == best_scenario)]
# print(f"Filtered AR6 data shape: {source.shape}")

## =============================================================================
## DATA TRANSFORMATION: WIDE TO LONG FORMAT
## =============================================================================

print("Melting AR6 data from wide to long format...")
id_cols = ['Model', 'Scenario', 'Region', 'Variable', 'Unit']
year_cols = [col for col in source.columns if col not in id_cols]

# Melt the data (years are columns, need to convert to rows)
melted = pd.melt(source, 
                 id_vars=id_cols, 
                 value_vars=year_cols, 
                 var_name='Year', 
                 value_name='Value')

# Data cleaning: convert types and filter out missing/invalid values
melted['Year'] = pd.to_numeric(melted['Year'], errors='coerce')
melted = melted.dropna(subset=['Year', 'Value'])
melted = melted[melted['Value'] != '']

# Clean extreme values that are likely data errors (e.g., -2.08e+32)
melted['Value'] = pd.to_numeric(melted['Value'], errors='coerce')
melted = melted[(melted['Value'].abs() <= 1e10) | melted['Value'].isna()]

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
        "Value": "value"
    }
)

## =============================================================================
## VARIABLE MAPPING AND BASE DATA CREATION
## =============================================================================

print("Joining with mapping data...")
# Join scenario data with variable mappings to categorize technologies
base_data = (
    renamed
    .merge(excel_mapping, how="left", left_on="variable", right_on="variable")
    .dropna(subset=['value'])
    .drop_duplicates()
)

print(f"Base data shape after join: {base_data.shape}")

## =============================================================================
## COST METRICS EXTRACTION
## =============================================================================

# Define join columns for cost metric self-joins
join_cols = [
    "model", "scenario", "region", "year", "Sector", "Subsector", "Technology"
]

def get_metric(df, metric_name):
    """Extract a specific cost metric (OM Cost, Capital Cost, Efficiency) from base data"""
    metric_data = df[df["col1"] == metric_name]
    if metric_data.empty:
        return pd.DataFrame(columns=join_cols + [f"{metric_name.lower().replace(' ', '_')}", f"{metric_name.lower().replace(' ', '_')}_unit"])
    
    return (
        metric_data[join_cols + ["value", "unit"]]
        .rename(columns={"value": f"{metric_name.lower().replace(' ', '_')}", "unit": f"{metric_name.lower().replace(' ', '_')}_unit"})
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
# Keep only rows where all three cost metrics have values (or efficiency is renewable)
filtered = merged[
    (merged["om_cost"].notna()) & (merged["om_cost"] != "") &
    (merged["capital_cost"].notna()) & (merged["capital_cost"] != "") &
    (
        (merged["efficiency"].notna() & (merged["efficiency"] != "")) |
        (merged["Sector"] == "Renewables")
    )
]

print(f"Filtered data shape: {filtered.shape}")

# Business rule: Set efficiency for Renewables to 100%
filtered = filtered.copy()
filtered.loc[filtered["Sector"] == "Renewables", "efficiency"] = "100"
filtered.loc[filtered["Sector"] == "Renewables", "efficiency_unit"] = "%"

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
    price_rows = price_rows[["model", "scenario", "region", "year", "Fuel", "col2", "value", "unit"]].copy()
    price_rows = price_rows.dropna(subset=["col2", "value"])
    price_rows = price_rows.rename(columns={"col2": "energy_type", "value": "price"})
    
    print(f"Price rows after cleaning: {len(price_rows)}")
    print("Unique energy types:", price_rows["energy_type"].unique())
    
    # Separate price data by energy type for different processing
    primary_prices = price_rows[price_rows["energy_type"] == "Primary Energy"].copy()
    secondary_prices = price_rows[price_rows["energy_type"] == "Secondary Energy"].copy()
    final_prices = price_rows[price_rows["energy_type"] == "Final Energy"].copy()
    carbon_prices = price_rows[price_rows["energy_type"] == "Carbon"].copy()
    
    # Extract electricity-specific price data (separate columns)
    secondary_electricity = secondary_prices[secondary_prices["Fuel"] == "Electricity"].copy()
    final_electricity = final_prices[final_prices["Fuel"] == "Electricity"].copy()
    
    # Start with filtered cost data
    final = filtered.copy()
    
    # 1. FUEL-SPECIFIC PRIMARY ENERGY PRICES
    # Maps the primary energy price for whatever fuel is in the 'Fuel' column
    if not primary_prices.empty:
        print("Processing primary energy prices...")
        
        # Pivot prices and units by fuel type
        primary_pivot = primary_prices.pivot_table(
            index=["model", "scenario", "region", "year"],
            columns="Fuel", values="price", aggfunc="first"
        ).reset_index()
        
        primary_units_pivot = primary_prices.pivot_table(
            index=["model", "scenario", "region", "year"], 
            columns="Fuel", values="unit", aggfunc="first"
        ).reset_index()
        
        # Efficient vectorized function to map fuel-specific prices
        def get_fuel_price_and_unit(df, price_pivot, units_pivot, price_col_name, unit_col_name):
            # Merge price data
            merged = df.merge(price_pivot, on=["model", "scenario", "region", "year"], how="left")
            merged = merged.merge(units_pivot, on=["model", "scenario", "region", "year"], how="left", suffixes=('', '_unit'))
            
            # Map fuel-specific prices using vectorized operations
            price_values = []
            unit_values = []
            
            for _, row in merged.iterrows():
                fuel = row["Fuel"]
                if pd.isna(fuel) or fuel == "" or fuel not in price_pivot.columns:
                    price_values.append(np.nan)
                    unit_values.append(np.nan)
                else:
                    price_val = row[fuel] if fuel in merged.columns else np.nan
                    unit_val = row[f"{fuel}_unit"] if f"{fuel}_unit" in merged.columns else np.nan
                    
                    # Filter extreme values (data errors)
                    if pd.notna(price_val) and abs(price_val) <= 1e10:
                        price_values.append(price_val)
                        unit_values.append(unit_val)
                    else:
                        price_values.append(np.nan)
                        unit_values.append(unit_val)
            
            return price_values, unit_values
        
        price_vals, unit_vals = get_fuel_price_and_unit(
            final, primary_pivot, primary_units_pivot, 
            "primary_energy_price", "primary_energy_price_unit"
        )
        final["primary_energy_price"] = price_vals
        final["primary_energy_price_unit"] = unit_vals
    else:
        final["primary_energy_price"] = np.nan
        final["primary_energy_price_unit"] = np.nan
    
    # 2. FUEL-SPECIFIC SECONDARY ENERGY PRICES  
    # Maps the secondary energy price for whatever fuel is in the 'Fuel' column
    if not secondary_prices.empty:
        print("Processing secondary energy prices...")
        
        secondary_pivot = secondary_prices.pivot_table(
            index=["model", "scenario", "region", "year"],
            columns="Fuel", values="price", aggfunc="first"
        ).reset_index()
        
        secondary_units_pivot = secondary_prices.pivot_table(
            index=["model", "scenario", "region", "year"],
            columns="Fuel", values="unit", aggfunc="first"
        ).reset_index()
        
        price_vals, unit_vals = get_fuel_price_and_unit(
            final, secondary_pivot, secondary_units_pivot,
            "secondary_energy_price", "secondary_energy_price_unit"
        )
        final["secondary_energy_price"] = price_vals
        final["secondary_energy_price_unit"] = unit_vals
    else:
        final["secondary_energy_price"] = np.nan
        final["secondary_energy_price_unit"] = np.nan
    
    # 3. CARBON PRICE (single column, not fuel-specific)
    # All technologies face the same carbon price in a given model/scenario/region/year
    if not carbon_prices.empty:
        print("Processing carbon prices...")
        
        carbon_pivot = carbon_prices.groupby(["model", "scenario", "region", "year"])["price"].first().reset_index()
        carbon_units_pivot = carbon_prices.groupby(["model", "scenario", "region", "year"])["unit"].first().reset_index()
        
        final = final.merge(
            carbon_pivot.rename(columns={"price": "carbon_price"}),
            on=["model", "scenario", "region", "year"], how="left"
        )
        final = final.merge(
            carbon_units_pivot.rename(columns={"unit": "carbon_price_unit"}),
            on=["model", "scenario", "region", "year"], how="left"
        )
    else:
        final["carbon_price"] = np.nan
        final["carbon_price_unit"] = np.nan
    
    # 4. ELECTRICITY-SPECIFIC PRICES (separate columns for secondary and final energy)
    # These are separate from fuel-specific prices above because electricity is special
    if not secondary_electricity.empty:
        print("Processing secondary electricity prices...")
        
        sec_elec_pivot = secondary_electricity.groupby(["model", "scenario", "region", "year"])["price"].first().reset_index()
        sec_elec_units_pivot = secondary_electricity.groupby(["model", "scenario", "region", "year"])["unit"].first().reset_index()
        
        final = final.merge(
            sec_elec_pivot.rename(columns={"price": "secondary_energy_electricity_price"}),
            on=["model", "scenario", "region", "year"], how="left"
        )
        final = final.merge(
            sec_elec_units_pivot.rename(columns={"unit": "secondary_energy_electricity_price_unit"}),
            on=["model", "scenario", "region", "year"], how="left"
        )
    else:
        final["secondary_energy_electricity_price"] = np.nan
        final["secondary_energy_electricity_price_unit"] = np.nan
    
    if not final_electricity.empty:
        print("Processing final electricity prices...")
        
        final_elec_pivot = final_electricity.groupby(["model", "scenario", "region", "year"])["price"].first().reset_index()
        final_elec_units_pivot = final_electricity.groupby(["model", "scenario", "region", "year"])["unit"].first().reset_index()
        
        final = final.merge(
            final_elec_pivot.rename(columns={"price": "final_energy_electricity_price"}),
            on=["model", "scenario", "region", "year"], how="left"
        )
        final = final.merge(
            final_elec_units_pivot.rename(columns={"unit": "final_energy_electricity_price_unit"}),
            on=["model", "scenario", "region", "year"], how="left"
        )
    else:
        final["final_energy_electricity_price"] = np.nan
        final["final_energy_electricity_price_unit"] = np.nan
    
    print(f"Final data shape after price merge: {final.shape}")
    
    # Report price data coverage
    price_cols = ["primary_energy_price", "secondary_energy_price", "carbon_price", 
                  "secondary_energy_electricity_price", "final_energy_electricity_price"]
    for col in price_cols:
        if col in final.columns:
            non_na_count = final[col].notna().sum()
            print(f"{col}: {non_na_count} non-NA values")
        
else:
    print("No price data found, proceeding without price columns")
    final = filtered.copy()

## =============================================================================
## FINAL OUTPUT PREPARATION
## =============================================================================

print("Preparing final output...")

# Define column order for final output
final_cols = [
    # Core scenario data
    "model", "scenario", "region", "variable", "value", "unit", "year",
    # Technology categorization
    "Sector", "Subsector", "Technology", "Fuel", "CCS_flag",
    # Cost metrics
    "om_cost", "om_cost_unit", "efficiency", "efficiency_unit",
    "capital_cost", "capital_cost_unit",
    # Price data (if available)
    "primary_energy_price", "primary_energy_price_unit",
    "secondary_energy_price", "secondary_energy_price_unit", 
    "carbon_price", "carbon_price_unit",
    "secondary_energy_electricity_price", "secondary_energy_electricity_price_unit",
    "final_energy_electricity_price", "final_energy_electricity_price_unit",
    # Additional mapping columns
    "col1", "col2", "col3", "col4", "col5"
]

# Select only columns that exist in the dataframe
final_cols = [col for col in final_cols if col in final.columns]
final = final[final_cols]

print(f"Final output shape: {final.shape}")
print(f"Final columns: {len(final_cols)}")

# Output to CSV
output_file = "1_intermediate_AR6_scenario_formatting.csv"
final.to_csv(output_file,index=False)
print(f"Output written to {output_file}")

## =============================================================================
## SUMMARY AND VALIDATION
## =============================================================================

print("\n" + "="*60)
print("PROCESSING SUMMARY")
print("="*60)

print(f"Final dataset: {len(final):,} rows × {len(final.columns)} columns")
print(f"Unique models: {final['model'].nunique()}")
print(f"Unique scenarios: {final['scenario'].nunique()}")
print(f"Unique regions: {final['region'].nunique()}")
print(f"Year range: {final['year'].min()}-{final['year'].max()}")

# Show sample of final data
print("\nSample of final data:")
print(final[["model", "scenario", "region", "Sector", "Technology", "year", "om_cost", "capital_cost"]].head())

# Show price data sample if available
price_cols = [col for col in final.columns if 'price' in col and not col.endswith('_unit')]
if price_cols:
    sample_with_prices = final.dropna(subset=price_cols, how='all').head()
    if not sample_with_prices.empty:
        print(f"\nSample rows with price data:")
        display_cols = ["model", "scenario", "Fuel", "year"] + price_cols[:3]  # Show first 3 price columns
        print(sample_with_prices[display_cols])
    else:
        print("\nNo rows found with price data")

print("\n" + "="*60)
print("PROCESSING COMPLETE!")
print("="*60)
