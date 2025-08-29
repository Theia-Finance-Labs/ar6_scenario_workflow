"""
AR6 Technology Aggregation and Gap-Filling Pipeline
===================================================

This script handles complex technology mapping, aggregation, and comprehensive gap-filling.

TECHNOLOGY MAPPING DECISION FRAMEWORK:
=====================================

1. DIRECT MAPPINGS (1:1):
   - Steel technologies: BF-OHF, DRI-EAF, DRI-BOF, BF-EAF, EAF, BOF
   - Power with CCS: BiomassCap_w/ CCS, CoalCap_w/ CCS, GasCap_w/ CCS, OilCap_w/ CCS
   - Power without CCS: BiomassCap_w/o CCS, CoalCap_w/o CCS, GasCap_w/o CCS, OilCap_w/o CCS
   - Individual renewables: GeothermalCap, HydroCap, NuclearCap, RenewablesCap

2. AGGREGATION MAPPINGS (Sum pathways, average costs):
   - Coal sector: All coal technologies → single "Coal" target
   - Oil&Gas sector: All gas technologies → single "Gas" target
   - Oil&Gas sector: All oil technologies → single "Oil" target
   - Power sector: All biomass power → "BiomassCap" (except CCS variants)
   - Power sector: All coal power → "CoalCap" (except CCS variants)
   - Power sector: All gas power → "GasCap" (except CCS variants)
   - Power sector: All oil power → "OilCap" (except CCS variants)
   - Solar: PV + CSP → "SolarCap"
   - Wind: Onshore + Offshore → "WindCap"

3. SECTOR TRANSFORMATIONS:
   - Renewables → Power: Geothermal, Hydro, Nuclear, Non-Biomass Renewables, Ocean
   - Hydrogen → Power: All hydrogen technologies mapped to appropriate power targets
   - Gas&Oil → Oil&Gas: Sector name change

AGGREGATION RULES:
=================
- Pathway values (scenario_pathway, capacity_additions): SUM
- Cost/price values (scenario_price, om_cost, capital_cost): WEIGHTED AVERAGE by pathway
- Technical metrics (lifetime_years, efficiency_decimal): WEIGHTED AVERAGE by pathway
- Capacity factors: WEIGHTED AVERAGE by pathway

Key Features:
1. Maps current sector-technology combinations to target combinations
2. Aggregates similar technologies with appropriate mathematical operations
3. Comprehensive gap-filling for key metrics using grouped medians
4. Sums pathway values (capacity, energy production)
5. Averages cost and price values (weighted by capacity)
6. Outputs final target schema with gap-filling tracking

Input:
- 3_final_AR6_target_schema.csv: Output from step 3
- technology_mapping.csv: Technology mapping definitions

Output:
- 4_final_AR6_aggregated.csv: Final aggregated dataset with gap-filled data

Author: Energy Data Processing Pipeline
Version: 4.2
Last Updated: 2024
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("AR6 TECHNOLOGY AGGREGATION AND FINAL SCHEMA TRANSFORMATION")
print("=" * 80)

# =============================================================================
# STEP 1: DATA LOADING
# =============================================================================
print("\n🔄 STEP 1: Loading data and mapping files...")

# Load the target schema data from step 3
try:
    df = pd.read_csv("3_final_AR6_target_schema.csv")
    print(
        f"   ✅ Loaded target schema data: {df.shape[0]:,} rows × {df.shape[1]} columns"
    )
except FileNotFoundError:
    print("   ❌ File not found: 3_final_AR6_target_schema.csv")
    print("   Please run step 3 first: python3 3_finalizeAR6.py")
    exit(1)

# Load technology mapping
try:
    mapping_df = pd.read_csv("technology_mapping.csv")
    print(f"   ✅ Loaded technology mapping: {mapping_df.shape[0]:,} rows")
except FileNotFoundError:
    print("   ❌ File not found: technology_mapping.csv")
    print("   Please ensure the technology mapping file exists")
    exit(1)

# Show current sector-technology distribution
print(f"\n📊 Current sector-technology distribution:")
current_combinations = (
    df.groupby(["sector", "technology"]).size().reset_index(name="count")
)
for _, row in current_combinations.iterrows():
    print(f"   {row['sector']} - {row['technology']}: {row['count']:,} rows")

# =============================================================================
# STEP 2: APPLY TECHNOLOGY MAPPING
# =============================================================================
print(f"\n🔄 STEP 2: Applying technology mapping...")

# Create mapping dictionary
mapping_dict = {}
for _, row in mapping_df.iterrows():
    key = (row["current_sector"], row["current_technology"])
    value = {
        "target_sector": row["target_sector"],
        "target_technology": row["target_technology"],
        "aggregation_group": row["aggregation_group"],
    }
    mapping_dict[key] = value

print(f"   📋 Created mapping for {len(mapping_dict)} technology combinations")

# Apply mapping to create new sector and technology columns
df["current_sector_tech"] = df["sector"] + "|" + df["technology"]
df["mapped_sector"] = df["sector"]  # Default to current
df["mapped_technology"] = df["technology"]  # Default to current
df["aggregation_group"] = "single"  # Default aggregation group

# Apply mappings
mapped_count = 0
for idx, row in df.iterrows():
    key = (row["sector"], row["technology"])
    if key in mapping_dict:
        mapping = mapping_dict[key]
        df.at[idx, "mapped_sector"] = mapping["target_sector"]
        df.at[idx, "mapped_technology"] = mapping["target_technology"]
        df.at[idx, "aggregation_group"] = mapping["aggregation_group"]
        mapped_count += 1

print(f"   ✅ Applied mapping to {mapped_count:,} rows")

# =============================================================================
# STEP 3: IDENTIFY AGGREGATION GROUPS
# =============================================================================
print(f"\n🔄 STEP 3: Analyzing aggregation requirements...")

# Check which aggregation groups exist
agg_groups = df["aggregation_group"].value_counts()
print(f"   📊 Aggregation groups found:")
for group, count in agg_groups.items():
    print(f"     {group}: {count:,} rows")

# Identify groups that need aggregation (not 'single')
aggregation_groups = df[df["aggregation_group"] != "single"][
    "aggregation_group"
].unique()
print(f"   🔧 Groups requiring aggregation: {list(aggregation_groups)}")

# =============================================================================
# STEP 4: PERFORM AGGREGATION
# =============================================================================
print(f"\n🔄 STEP 4: Performing technology aggregation...")


def aggregate_technology_group(group_df):
    """
    Aggregate technologies within a group based on decision framework:
    - direct_mapping: No aggregation needed (1:1 mapping)
    - sum_pathways_average_costs: Sum pathways, average costs/technical metrics
    """

    if len(group_df) == 1:
        # No aggregation needed
        return group_df.iloc[0]

    # Get the decision framework from the first row
    decision_framework = group_df.iloc[0]["aggregation_group"]
    mapped_technology = group_df.iloc[0]["mapped_technology"]

    print(
        f"     Aggregating {len(group_df)} rows for {mapped_technology} (framework: {decision_framework})"
    )

    # Create aggregated row
    agg_row = group_df.iloc[0].copy()  # Start with first row as template

    # Define columns by aggregation type
    sum_cols = ["scenario_pathway", "capacity_additions_mw_per_yr"]

    avg_cols = [
        "scenario_price",
        "scenario_capacity_factor",
        "lifetime_years",
        "efficiency_decimal",
        "om_cost_usd_per_mw_per_yr",
        "capital_cost_usd_per_mw",
        "carbon_price_usd_per_tco2",
    ]

    # Sum pathway-related values (additive metrics)
    for col in sum_cols:
        if col in group_df.columns:
            agg_row[col] = group_df[col].sum()

    # Calculate weighted averages for costs and prices (use pathway as weight where available)
    weights = group_df["scenario_pathway"].fillna(1)  # Use 1 if pathway is missing
    total_weight = weights.sum()

    if total_weight > 0:
        for col in avg_cols:
            if col in group_df.columns:
                weighted_values = group_df[col] * weights
                agg_row[col] = weighted_values.sum() / total_weight
    else:
        # Fallback to simple average if no weights available
        for col in avg_cols:
            if col in group_df.columns:
                agg_row[col] = group_df[col].mean()

    return agg_row


# Group by all identifying columns plus mapped sector/technology
grouping_cols = [
    "scenario_provider",
    "scenario",
    "scenario_type",
    "scenario_geography",
    "mapped_sector",
    "mapped_technology",
    "scenario_year",
]

# Check which columns exist in the dataframe
grouping_cols = [col for col in grouping_cols if col in df.columns]
print(f"   📋 Grouping by: {grouping_cols}")

# Perform aggregation
aggregated_rows = []
grouped = df.groupby(grouping_cols)

for group_key, group_df in grouped:
    agg_row = aggregate_technology_group(group_df)
    aggregated_rows.append(agg_row)

# Create aggregated dataframe
aggregated_df = pd.DataFrame(aggregated_rows)

# Update sector and technology columns with mapped values
aggregated_df["sector"] = aggregated_df["mapped_sector"]
aggregated_df["technology"] = aggregated_df["mapped_technology"]

# Drop temporary columns
cols_to_drop = [
    "current_sector_tech",
    "mapped_sector",
    "mapped_technology",
    "aggregation_group",
]
cols_to_drop = [col for col in cols_to_drop if col in aggregated_df.columns]
aggregated_df = aggregated_df.drop(columns=cols_to_drop)

print(f"   ✅ Aggregation completed: {aggregated_df.shape[0]:,} rows")

# =============================================================================
# STEP 5: COMPREHENSIVE GAP-FILLING
# =============================================================================
print(f"\n🔧 STEP 5: Comprehensive gap-filling for key metrics...")

# Columns to gap-fill
gap_fill_columns = [
    "scenario_price",
    "scenario_capacity_factor",
    "lifetime_years",
    "efficiency_decimal",
    "om_cost_usd_per_mw_per_yr",
    "capital_cost_usd_per_mw",
]

# Filter to only include columns that exist in the dataframe
gap_fill_columns = [col for col in gap_fill_columns if col in aggregated_df.columns]
print(f"   📋 Gap-filling columns: {gap_fill_columns}")

# Initialize tracking column
aggregated_df["gap_filled_columns"] = ""


def gap_fill_with_grouped_medians(df, columns_to_fill):
    """
    Gap-fill specified columns using grouped medians with hierarchical fallback.
    Groups by: technology, scenario_geography (removed scenario_type for better coverage)
    """

    # Create a copy to avoid modifying original
    df_filled = df.copy()

    # Grouping hierarchy for gap-filling (removed scenario_type for better coverage)
    grouping_levels = [
        ["technology", "scenario_geography"],  # Most specific
        ["technology"],  # Least specific
    ]

    # Track which columns were filled for each row
    filled_tracking = df_filled["gap_filled_columns"].copy()

    for col in columns_to_fill:
        print(f"   🔧 Gap-filling {col}...")
        initial_missing = df_filled[col].isna().sum()

        # Try each grouping level
        for level_idx, group_cols in enumerate(grouping_levels):
            # Filter to only include columns that exist
            existing_group_cols = [
                col for col in group_cols if col in df_filled.columns
            ]
            if not existing_group_cols:
                continue

            # Calculate medians for this grouping level
            medians = df_filled.groupby(existing_group_cols)[col].median()

            # Fill missing values using these medians
            for idx, row in df_filled.iterrows():
                if pd.isna(row[col]):
                    # Create key for median lookup
                    key_parts = []
                    for group_col in existing_group_cols:
                        key_parts.append(row[group_col])

                    if len(key_parts) == 1:
                        lookup_key = key_parts[0]
                    else:
                        lookup_key = tuple(key_parts)

                    # Look up median value
                    if lookup_key in medians and pd.notna(medians[lookup_key]):
                        df_filled.at[idx, col] = medians[lookup_key]

                        # Track that this column was filled
                        current_tracking = filled_tracking.at[idx]
                        if current_tracking:
                            filled_tracking.at[idx] = current_tracking + f",{col}"
                        else:
                            filled_tracking.at[idx] = col

        final_missing = df_filled[col].isna().sum()
        filled_count = initial_missing - final_missing
        print(
            f"     ✅ {col}: {filled_count:,} values filled, {final_missing:,} still missing"
        )

    # Update the tracking column
    df_filled["gap_filled_columns"] = filled_tracking

    return df_filled


# Perform gap-filling
aggregated_df = gap_fill_with_grouped_medians(aggregated_df, gap_fill_columns)

# Report gap-filling results
print(f"\n   📊 Gap-filling summary:")
for col in gap_fill_columns:
    if col in aggregated_df.columns:
        initial_missing = df[col].isna().sum() if col in df.columns else 0
        final_missing = aggregated_df[col].isna().sum()
        filled_count = initial_missing - final_missing
        print(f"     {col}: {filled_count:,} values filled")

# Count rows with gap-filling
rows_with_gap_filling = aggregated_df["gap_filled_columns"].notna().sum()
print(f"   📈 Rows with gap-filling: {rows_with_gap_filling:,}")

# =============================================================================
# STEP 6: FINAL SCHEMA VALIDATION
# =============================================================================
print(f"\n🔄 STEP 6: Final schema validation...")

# Show final sector-technology distribution
print(f"   📊 Final sector-technology distribution:")
final_combinations = (
    aggregated_df.groupby(["sector", "technology"]).size().reset_index(name="count")
)
for _, row in final_combinations.iterrows():
    print(f"     {row['sector']} - {row['technology']}: {row['count']:,} rows")

# Check for any unmapped technologies
# Create a set of all valid target sector-technology combinations
valid_combinations = set(
    zip(mapping_df["target_sector"], mapping_df["target_technology"])
)

# Create boolean mask using vectorized operations
sector_tech_combinations = list(
    zip(aggregated_df["sector"], aggregated_df["technology"])
)
unmapped_mask = ~pd.Series(sector_tech_combinations).isin(valid_combinations)

# Filter to get unmapped rows
unmapped = aggregated_df[unmapped_mask]

if len(unmapped) > 0:
    print(f"   ⚠️  Found {len(unmapped)} rows with unmapped technologies:")
    unmapped_techs = (
        unmapped.groupby(["sector", "technology"]).size().reset_index(name="count")
    )
    for _, row in unmapped_techs.iterrows():
        print(f"     {row['sector']} - {row['technology']}: {row['count']:,} rows")

# =============================================================================
# STEP 7: OUTPUT GENERATION
# =============================================================================
print(f"\n💾 STEP 7: Saving final aggregated dataset...")

# Ensure proper column ordering
target_cols = [
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
    "pathway_unit",
    "scenario_pathway",
    "scenario_capacity_factor",
    "scenario_year",
    "country_iso2_list",
]

# Add additional columns that exist
additional_cols = [col for col in aggregated_df.columns if col not in target_cols]
final_cols = target_cols + additional_cols

# Select columns that actually exist
final_cols = [col for col in final_cols if col in aggregated_df.columns]
final_output = aggregated_df[final_cols]

# Save output
output_file = "4_final_AR6_aggregated.csv"
final_output.to_csv(output_file, index=False)
print(f"   💾 Saved to: {output_file}")
print(f"   📊 Final shape: {final_output.shape}")

# =============================================================================
# STEP 8: COMPLETENESS CHECK FOR GAP-FILLED COLUMNS
# =============================================================================
print(f"\n📊 STEP 8: Completeness check for gap-filled columns...")

print(f"   📈 Completeness after gap-filling:")
for col in gap_fill_columns:
    if col in aggregated_df.columns:
        completeness = aggregated_df[col].notna().sum() / len(aggregated_df) * 100
        missing_count = aggregated_df[col].isna().sum()
        print(f"     {col}: {completeness:.1f}% complete ({missing_count:,} missing)")

# Check if we should drop rows with missing values
total_missing_by_row = aggregated_df[gap_fill_columns].isna().sum(axis=1)
rows_with_any_missing = (total_missing_by_row > 0).sum()
print(
    f"\n   📊 Rows with any missing values in gap-filled columns: {rows_with_any_missing:,}"
)

if rows_with_any_missing > 0:
    print(
        f"   💡 Recommendation: Consider dropping {rows_with_any_missing:,} rows with missing values"
    )
    print(
        f"   💡 This would leave {(len(aggregated_df) - rows_with_any_missing):,} complete rows"
    )
else:
    print(f"   ✅ All rows have complete data for gap-filled columns!")

# =============================================================================
# STEP 9: SUMMARY REPORT
# =============================================================================
print(f"\n📋 STEP 9: Aggregation and gap-filling summary...")

print(f"   📈 Data transformation:")
print(f"     Input rows: {df.shape[0]:,}")
print(f"     After aggregation: {len(aggregated_rows):,}")
print(f"     After gap-filling: {len(aggregated_df):,}")
print(
    f"     Reduction: {df.shape[0] - len(aggregated_df):,} rows ({(df.shape[0] - len(aggregated_df))/df.shape[0]*100:.1f}%)"
)

print(f"\n   🏭 Final sector summary:")
sector_summary = aggregated_df["sector"].value_counts()
for sector, count in sector_summary.items():
    print(f"     {sector}: {count:,} rows")

print(f"\n   ⚙️  Technology summary by sector:")
for sector in sorted(aggregated_df["sector"].unique()):
    sector_data = aggregated_df[aggregated_df["sector"] == sector]
    tech_counts = sector_data["technology"].value_counts()
    print(f"     {sector}:")
    for tech, count in tech_counts.items():
        print(f"       • {tech}: {count:,} rows")

print(f"\n🎉 TECHNOLOGY AGGREGATION AND GAP-FILLING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("Final dataset ready for analysis with:")
print("• Aggregated technologies: Similar technologies combined intelligently")
print("• Comprehensive gap-filling: Key metrics filled using grouped medians")
print("• Gap-filling tracking: Column 'gap_filled_columns' shows what was filled")
print("• Summed pathways: Capacity and energy values properly aggregated")
print("• Averaged costs/prices: Weighted by capacity for accuracy")
print("• Target schema compliance: Ready for downstream analysis")
print("=" * 80)

# =============================================================================
# STEP 10: CREATE STRINGENCY COLUMN AND UPDATE SCENARIO_TYPE
# =============================================================================
print(f"\n🔄 STEP 10: Creating stringency column and updating scenario_type...")

# Copy scenario_type to stringency column
aggregated_df["stringency"] = aggregated_df["scenario_type"]


# Update scenario_type based on stringency rules
def update_scenario_type(row):
    """Update scenario_type based on stringency rules"""
    stringency = row["stringency"]

    if stringency in ["C1", "C2"]:
        return "target"
    else:
        return "baseline"


aggregated_df["scenario_type"] = aggregated_df.apply(update_scenario_type, axis=1)

# Report the transformation
print(f"   📊 Stringency column created from scenario_type")
print(f"   📊 Scenario_type updated:")
print(
    f"     • Target scenarios (C1, C2): {(aggregated_df['scenario_type'] == 'target').sum():,} rows"
)
print(
    f"     • Baseline scenarios (others): {(aggregated_df['scenario_type'] == 'baseline').sum():,} rows"
)

# Show stringency distribution
print(f"   📈 Stringency distribution:")
stringency_counts = aggregated_df["stringency"].value_counts()
for stringency, count in stringency_counts.items():
    print(f"     • {stringency}: {count:,} rows")

# =============================================================================
# STEP 11: FINAL OUTPUT GENERATION
# =============================================================================
print(f"\n💾 STEP 11: Saving final aggregated dataset...")

# Ensure proper column ordering
target_cols = [
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
    "pathway_unit",
    "scenario_pathway",
    "scenario_capacity_factor",
    "scenario_year",
    "country_iso2_list",
    "stringency",
]

# Add additional columns that exist
additional_cols = [col for col in aggregated_df.columns if col not in target_cols]
final_cols = target_cols + additional_cols

# Select columns that actually exist
final_cols = [col for col in final_cols if col in aggregated_df.columns]
final_output = aggregated_df[final_cols]

# Save output
output_file = "4_final_AR6_aggregated.csv"
final_output.to_csv(output_file, index=False)
print(f"   💾 Saved to: {output_file}")
print(f"   📊 Final shape: {final_output.shape}")

# =============================================================================
# STEP 12: COMPLETE CASE FILTERING
# =============================================================================
print(f"\n🎯 STEP 12: Filtering for complete cases...")

# Define critical columns that must be complete
critical_columns = [
    "scenario_pathway",
    "scenario_price",
    "om_cost_usd_per_mw_per_yr",
    "capital_cost_usd_per_mw",
]

# Add efficiency_decimal for Power and Renewables sectors only
efficiency_condition = (final_output["sector"].isin(["Power", "Renewables"])) & (
    final_output["efficiency_decimal"].isna()
)

# Filter to only include columns that exist
existing_critical_columns = [
    col for col in critical_columns if col in final_output.columns
]
print(
    f"   📋 Critical columns for complete case filtering: {existing_critical_columns}"
)

# Create complete case mask
complete_case_mask = final_output[existing_critical_columns].notna().all(axis=1)

# Apply efficiency condition for Power/Renewables
if "efficiency_decimal" in final_output.columns:
    efficiency_mask = ~efficiency_condition
    complete_case_mask = complete_case_mask & efficiency_mask
    print(f"   ⚡ Applied efficiency requirement for Power/Renewables sectors")

# Apply the filter
initial_rows = len(final_output)
final_complete = final_output[complete_case_mask].copy()
final_rows = len(final_complete)
dropped_rows = initial_rows - final_rows

print(f"   📊 Complete case filtering results:")
print(f"     • Initial rows: {initial_rows:,}")
print(f"     • Complete rows: {final_rows:,}")
print(f"     • Dropped rows: {dropped_rows:,} ({dropped_rows/initial_rows*100:.1f}%)")

# Analyze what's causing the drops
print(f"\n   🔍 Analysis of dropped rows by column:")
for col in existing_critical_columns:
    col_missing = final_output[col].isna().sum()
    print(f"     • {col}: {col_missing:,} missing values")

if "efficiency_decimal" in final_output.columns:
    efficiency_missing = efficiency_condition.sum()
    print(
        f"     • efficiency_decimal (Power/Renewables only): {efficiency_missing:,} missing values"
    )

# Analyze by sector
print(f"\n   🏭 Complete case analysis by sector:")
for sector in sorted(final_output["sector"].unique()):
    sector_data = final_output[final_output["sector"] == sector]
    sector_complete = final_complete[final_complete["sector"] == sector]
    sector_total = len(sector_data)
    sector_complete_count = len(sector_complete)
    sector_completeness = sector_complete_count / sector_total * 100
    print(
        f"     • {sector}: {sector_completeness:.1f}% ({sector_complete_count:,}/{sector_total:,})"
    )

# =============================================================================
# STEP 13: FINAL OUTPUT GENERATION
# =============================================================================
print(f"\n💾 STEP 13: Saving final complete dataset...")

# Save the complete dataset
output_file_complete = "4_final_AR6_aggregated_complete.csv"
final_complete.to_csv(output_file_complete, index=False)
print(f"   💾 Saved complete dataset to: {output_file_complete}")
print(f"   📊 Final complete shape: {final_complete.shape}")

# =============================================================================
# STEP 14: SUMMARY REPORT
# =============================================================================
print(f"\n📋 STEP 14: Final summary...")

print(f"   📈 Data transformation:")
print(f"     Input rows: {df.shape[0]:,}")
print(f"     After aggregation: {len(aggregated_rows):,}")
print(f"     After gap-filling: {len(aggregated_df):,}")
print(f"     After complete case filtering: {len(final_complete):,}")
print(
    f"     Total reduction: {df.shape[0] - len(final_complete):,} rows ({(df.shape[0] - len(final_complete))/df.shape[0]*100:.1f}%)"
)

print(f"\n   🏭 Final sector summary (complete cases only):")
sector_summary = final_complete["sector"].value_counts()
for sector, count in sector_summary.items():
    print(f"     {sector}: {count:,} rows")

print(f"\n   ⚙️  Technology summary by sector (complete cases only):")
for sector in sorted(final_complete["sector"].unique()):
    sector_data = final_complete[final_complete["sector"] == sector]
    tech_counts = sector_data["technology"].value_counts()
    print(f"     {sector}:")
    for tech, count in tech_counts.items():
        print(f"       • {tech}: {count:,} rows")

print(f"\n   🎯 Scenario type summary (complete cases only):")
scenario_summary = final_complete["scenario_type"].value_counts()
for scenario_type, count in scenario_summary.items():
    print(f"     {scenario_type}: {count:,} rows")

print(
    f"\n🎉 TECHNOLOGY AGGREGATION, GAP-FILLING, AND COMPLETE CASE FILTERING COMPLETED SUCCESSFULLY!"
)
print("=" * 80)
print("Final complete dataset ready for analysis with:")
print("• Complete cases only: All critical columns have values")
print("• Aggregated technologies: Similar technologies combined intelligently")
print("• Comprehensive gap-filling: Key metrics filled using grouped medians")
print("• Gap-filling tracking: Column 'gap_filled_columns' shows what was filled")
print("• Stringency classification: C1/C2 → target, others → baseline")
print("• Summed pathways: Capacity and energy values properly aggregated")
print("• Averaged costs/prices: Weighted by capacity for accuracy")
print("• Target schema compliance: Ready for downstream analysis")
print("• No missing values: All required columns are complete")
print("=" * 80)
