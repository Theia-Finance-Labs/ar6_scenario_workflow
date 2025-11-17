#!/usr/bin/env python3
"""
Step 5: Filter to Complete Cases Only

This script filters the gap-filled AR6 dataset to include only scenarios with complete data
for all essential technical and economic parameters. Gap-filled values are considered complete.

Input: 4_final_AR6_gapfilled_complete.csv
Output: 5_final_AR6_complete_cases.csv

Author: Claude Code
Date: 2025-09-15
"""

import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('step5_complete_cases.log'),
        logging.StreamHandler()
    ]
)

def identify_essential_columns():
    """
    Define the essential columns that must be non-null for a complete case.
    
    Returns:
        list: Column names that are essential for complete cases
    """
    essential_cols = [
        # Core scenario identifiers
        'scenario_provider',
        'scenario', 
        'scenario_type',
        'scenario_geography',
        'sector',
        'technology',
        'technology_type',
        'scenario_year',
        
        # Essential technical parameters
        'efficiency_decimal',
        'lifetime_years',
        'scenario_capacity_factor',
        
        # Essential economic parameters  
        'capital_cost_usd_per_mw',
        'om_cost_usd_per_mw_per_yr',
        'scenario_price',
        
        # Core pathway data
        'scenario_pathway',
        
        # Fuel price (for fossil fuel technologies)
        'fuel_price'
    ]
    
    return essential_cols

def check_completeness(df, essential_cols):
    """
    Check completeness of the dataset for essential columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        essential_cols (list): List of essential column names
        
    Returns:
        pd.Series: Boolean mask indicating complete cases
    """
    logging.info("Checking completeness for essential columns...")
    
    # Create mask for complete cases
    complete_mask = pd.Series(True, index=df.index)
    
    for col in essential_cols:
        if col in df.columns:
            # Check for null values
            col_complete = df[col].notna()
            
            # For string columns, also check for empty strings
            if df[col].dtype == 'object':
                col_complete = col_complete & (df[col] != '')
            
            complete_mask = complete_mask & col_complete
            
            missing_count = (~col_complete).sum()
            logging.info(f"  {col}: {missing_count:,} missing values")
        else:
            logging.warning(f"  {col}: Column not found in dataset")
            complete_mask = pd.Series(False, index=df.index)
    
    return complete_mask

def handle_fuel_price_exceptions(df, complete_mask):
    """
    Handle special cases for fuel_price completeness.
    Green technologies (solar, wind, hydro, nuclear, geothermal) don't need fuel prices.
    
    Args:
        df (pd.DataFrame): Input dataframe
        complete_mask (pd.Series): Current completeness mask
        
    Returns:
        pd.Series: Updated completeness mask
    """
    logging.info("Handling fuel price exceptions for green technologies...")
    
    # Define green technologies that don't require fuel prices
    green_techs = [
        'SolarCap - PV', 'SolarCap - CSP',
        'WindCap - Onshore', 'WindCap - Offshore', 
        'HydroCap', 'GeothermalCap', 'NuclearCap'
    ]
    
    # For green technologies, missing fuel_price is acceptable
    green_tech_mask = df['technology'].isin(green_techs)
    fuel_price_missing = df['fuel_price'].isna()
    
    # Allow missing fuel prices for green technologies
    green_exception = green_tech_mask & fuel_price_missing
    
    # Update complete mask to include green technologies with missing fuel prices
    complete_mask = complete_mask | green_exception
    
    logging.info(f"  Allowed {green_exception.sum():,} green technology cases with missing fuel prices")
    
    return complete_mask

def add_ebitda_check(df):
    """
    Add EBITDA_check column to verify economic consistency.
    
    EBITDA_check should be TRUE when:
    om_cost_usd_per_mw_per_yr <= scenario_capacity_factor * hours_per_year * scenario_price
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with EBITDA_check column added
    """
    logging.info("Adding EBITDA_check column...")
    
    # Constants
    hours_per_year = 8760  # 24 hours * 365 days
    
    # Calculate expected annual revenue per MW
    # Revenue = capacity_factor * hours_per_year * price_per_MWh
    df['annual_revenue_per_mw'] = (
        df['scenario_capacity_factor'] * hours_per_year * df['scenario_price']
    )
    
    # EBITDA check: O&M costs should not exceed annual revenue
    # TRUE when O&M <= Revenue, FALSE when O&M > Revenue
    df['EBITDA_check'] = df['om_cost_usd_per_mw_per_yr'] <= df['annual_revenue_per_mw']
    
    # Handle cases where capacity factor or price might be zero/null
    df['EBITDA_check'] = df['EBITDA_check'].fillna(False)
    
    false_count = (~df['EBITDA_check']).sum()
    false_pct = false_count / len(df) * 100
    
    logging.info(f"  EBITDA_check FALSE cases: {false_count:,} ({false_pct:.2f}%)")
    
    return df

def analyze_ebitda_failures(df):
    """
    Analyze scenarios where EBITDA_check is FALSE.
    
    Args:
        df (pd.DataFrame): Dataframe with EBITDA_check column
    """
    logging.info("\n=== EBITDA CHECK ANALYSIS ===")
    
    false_cases = df[~df['EBITDA_check']]
    
    if len(false_cases) == 0:
        logging.info("✅ All cases pass EBITDA check!")
        return
    
    logging.info(f"Total FALSE cases: {len(false_cases):,}")
    
    # Breakdown by sector
    logging.info("\n--- FALSE Cases by Sector ---")
    sector_breakdown = false_cases.groupby('sector').size().sort_values(ascending=False)
    for sector, count in sector_breakdown.items():
        total_sector = len(df[df['sector'] == sector])
        pct = count / total_sector * 100
        logging.info(f"  {sector}: {count:,}/{total_sector:,} ({pct:.1f}%)")
    
    # Breakdown by technology
    logging.info("\n--- FALSE Cases by Technology (Top 10) ---")
    tech_breakdown = false_cases.groupby('technology').size().sort_values(ascending=False)
    for tech, count in tech_breakdown.head(10).items():
        total_tech = len(df[df['technology'] == tech])
        pct = count / total_tech * 100
        logging.info(f"  {tech}: {count:,}/{total_tech:,} ({pct:.1f}%)")
    
    # Breakdown by scenario
    logging.info("\n--- FALSE Cases by Scenario (Top 10) ---")
    scenario_breakdown = false_cases.groupby('scenario').size().sort_values(ascending=False)
    for scenario, count in scenario_breakdown.head(10).items():
        total_scenario = len(df[df['scenario'] == scenario])
        pct = count / total_scenario * 100
        logging.info(f"  {scenario}: {count:,}/{total_scenario:,} ({pct:.1f}%)")
    
    # Statistical analysis of the problematic cases
    logging.info("\n--- Statistical Analysis of FALSE Cases ---")
    logging.info(f"O&M Cost Statistics:")
    logging.info(f"  Mean: ${false_cases['om_cost_usd_per_mw_per_yr'].mean():,.0f}")
    logging.info(f"  Median: ${false_cases['om_cost_usd_per_mw_per_yr'].median():,.0f}")
    logging.info(f"  Max: ${false_cases['om_cost_usd_per_mw_per_yr'].max():,.0f}")
    
    logging.info(f"Annual Revenue Statistics:")
    logging.info(f"  Mean: ${false_cases['annual_revenue_per_mw'].mean():,.0f}")
    logging.info(f"  Median: ${false_cases['annual_revenue_per_mw'].median():,.0f}")
    logging.info(f"  Min: ${false_cases['annual_revenue_per_mw'].min():,.0f}")
    
    # Check for common patterns
    logging.info("\n--- Common Patterns in FALSE Cases ---")
    
    # Low capacity factors
    low_cf = false_cases[false_cases['scenario_capacity_factor'] < 0.1]
    logging.info(f"  Cases with capacity factor < 10%: {len(low_cf):,}")
    
    # Low prices
    low_price = false_cases[false_cases['scenario_price'] < 50]
    logging.info(f"  Cases with price < $50/MWh: {len(low_price):,}")
    
    # High O&M costs
    high_om = false_cases[false_cases['om_cost_usd_per_mw_per_yr'] > 100000]
    logging.info(f"  Cases with O&M > $100k/MW/yr: {len(high_om):,}")

def filter_complete_cases(input_file, output_file):
    """
    Main function to filter dataset to complete cases only and add EBITDA check.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    """
    logging.info(f"Starting complete cases filtering...")
    logging.info(f"Input file: {input_file}")
    logging.info(f"Output file: {output_file}")
    
    # Read input data
    logging.info("Reading input data...")
    df = pd.read_csv(input_file, low_memory=False)
    logging.info(f"Input dataset shape: {df.shape}")
    
    # Get essential columns
    essential_cols = identify_essential_columns()
    logging.info(f"Essential columns defined: {len(essential_cols)}")
    
    # Check original completeness
    logging.info("\n=== COMPLETENESS ANALYSIS ===")
    complete_mask = check_completeness(df, essential_cols)
    
    # Handle fuel price exceptions
    complete_mask = handle_fuel_price_exceptions(df, complete_mask)
    
    # Filter to complete cases
    complete_df = df[complete_mask].copy()
    
    # Add EBITDA check
    complete_df = add_ebitda_check(complete_df)
    
    # Analyze EBITDA failures
    analyze_ebitda_failures(complete_df)
    
    # Log results
    logging.info(f"\n=== FILTERING RESULTS ===")
    logging.info(f"Original rows: {len(df):,}")
    logging.info(f"Complete cases: {len(complete_df):,}")
    logging.info(f"Rows removed: {len(df) - len(complete_df):,}")
    logging.info(f"Retention rate: {len(complete_df)/len(df)*100:.2f}%")
    
    # Analyze by sector
    logging.info(f"\n=== RETENTION BY SECTOR ===")
    for sector in sorted(df['sector'].unique()):
        sector_total = len(df[df['sector'] == sector])
        sector_complete = len(complete_df[complete_df['sector'] == sector])
        retention = sector_complete/sector_total*100 if sector_total > 0 else 0
        logging.info(f"  {sector}: {sector_complete:,}/{sector_total:,} ({retention:.1f}%)")
    
    # Analyze by technology
    logging.info(f"\n=== TECHNOLOGIES WITH LOW RETENTION (<90%) ===")
    for tech in sorted(df['technology'].unique()):
        tech_total = len(df[df['technology'] == tech])
        tech_complete = len(complete_df[complete_df['technology'] == tech])
        retention = tech_complete/tech_total*100 if tech_total > 0 else 0
        if retention < 90:
            logging.info(f"  {tech}: {tech_complete:,}/{tech_total:,} ({retention:.1f}%)")
    
    # Save output
    logging.info(f"\nSaving complete cases to: {output_file}")
    complete_df.to_csv(output_file, index=False)
    logging.info("Complete cases filtering completed successfully!")
    
    return complete_df

def step5_complete_cases():
    """
    Wrapper function for importable use from combined pipeline.
    Calls filter_complete_cases with default file paths.
    """
    input_file = "4_final_AR6_gapfilled_complete.csv"
    output_file = "5_final_AR6_complete_cases.csv"
    
    if not Path(input_file).exists():
        logging.error(f"Input file not found: {input_file}")
        return
    
    return filter_complete_cases(input_file, output_file)

def main():
    """Main execution function."""
    # File paths
    input_file = "4_final_AR6_gapfilled_complete.csv"
    output_file = "5_final_AR6_complete_cases.csv"
    
    # Check input file exists
    if not Path(input_file).exists():
        logging.error(f"Input file not found: {input_file}")
        return
    
    # Run filtering
    try:
        result_df = filter_complete_cases(input_file, output_file)
        
        # Final summary
        print(f"\n{'='*50}")
        print(f"COMPLETE CASES FILTERING SUMMARY")
        print(f"{'='*50}")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Complete cases: {len(result_df):,}")
        print(f"Log file: step5_complete_cases.log")
        
    except Exception as e:
        logging.error(f"Error during filtering: {str(e)}")
        raise

if __name__ == "__main__":
    main()