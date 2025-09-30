#!/usr/bin/env python3
"""
Carbon Price Analysis Script
Analyzes carbon price coverage across the AR6 data pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_carbon_price_coverage():
    """Analyze carbon price coverage across pipeline stages"""
    
    # Define the files to analyze in pipeline order
    pipeline_files = [
        "2_final_AR6_filtered.csv",
        "3_final_AR6_target_schema.csv", 
        "4_final_AR6_gapfilled.csv",
        "5_final_AR6_complete_cases.csv",
        "6_final_AR6_viable_scenarios.csv"
    ]
    
    print("Carbon Price Coverage Analysis")
    print("=" * 50)
    
    # Analyze each stage
    for i, filename in enumerate(pipeline_files):
        filepath = Path(filename)
        if not filepath.exists():
            print(f"⚠️  {filename} not found")
            continue
            
        print(f"\n{i+1}. Analyzing {filename}")
        try:
            # Read the file
            df = pd.read_csv(filepath)
            
            # Check if carbon_price column exists
            if 'carbon_price_usd_per_tco2' not in df.columns:
                print(f"   ❌ No carbon_price_usd_per_tco2 column found")
                print(f"   📋 Available columns: {', '.join(df.columns[:10])}...")
                continue
                
            # Analyze carbon price coverage
            total_rows = len(df)
            non_null_prices = df['carbon_price_usd_per_tco2'].notna().sum()
            null_prices = df['carbon_price_usd_per_tco2'].isna().sum()
            
            print(f"   📊 Total rows: {total_rows:,}")
            print(f"   ✅ Rows with carbon price: {non_null_prices:,} ({non_null_prices/total_rows*100:.1f}%)")
            print(f"   ❌ Rows without carbon price: {null_prices:,} ({null_prices/total_rows*100:.1f}%)")
            
            # Get unique scenarios with and without carbon price data
            if 'scenario' in df.columns:
                scenarios_with_price = df[df['carbon_price_usd_per_tco2'].notna()]['scenario'].nunique()
                total_scenarios = df['scenario'].nunique()
                scenarios_without_price = total_scenarios - scenarios_with_price
                
                print(f"   🎯 Scenarios with carbon price: {scenarios_with_price}/{total_scenarios}")
                print(f"   ⚠️  Scenarios without carbon price: {scenarios_without_price}/{total_scenarios}")
                
        except Exception as e:
            print(f"   ❌ Error reading {filename}: {str(e)}")
    
    # Generate detailed scenario analysis for final file
    print(f"\n" + "="*50)
    print("DETAILED SCENARIO ANALYSIS - Final File")
    print("="*50)
    
    try:
        df_final = pd.read_csv("6_final_AR6_viable_scenarios.csv")
        
        if 'carbon_price_usd_per_tco2' in df_final.columns and 'scenario' in df_final.columns:
            # Create scenario summary
            scenario_summary = df_final.groupby(['scenario', 'scenario_year']).agg({
                'carbon_price_usd_per_tco2': ['count', lambda x: x.notna().sum(), 'first']
            }).round(2)
            
            scenario_summary.columns = ['total_records', 'records_with_carbon_price', 'sample_carbon_price']
            scenario_summary['has_carbon_price'] = scenario_summary['records_with_carbon_price'] > 0
            scenario_summary = scenario_summary.reset_index()
            
            # Create summary by scenario (aggregating across years)
            scenario_yearly_summary = scenario_summary.groupby('scenario').agg({
                'has_carbon_price': 'any',
                'records_with_carbon_price': 'sum',
                'total_records': 'sum'
            }).reset_index()
            
            scenario_yearly_summary['carbon_price_coverage_pct'] = (
                scenario_yearly_summary['records_with_carbon_price'] / 
                scenario_yearly_summary['total_records'] * 100
            ).round(1)
            
            # Flag scenarios
            scenario_yearly_summary['carbon_price_flag'] = scenario_yearly_summary['has_carbon_price'].map({
                True: '✅ HAS_CARBON_PRICE', 
                False: '❌ NO_CARBON_PRICE'
            })
            
            print(f"\nSCENARIO SUMMARY TABLE")
            print("-" * 80)
            print(scenario_yearly_summary[['scenario', 'carbon_price_flag', 'carbon_price_coverage_pct', 'records_with_carbon_price', 'total_records']].to_string(index=False))
            
            # Summary statistics
            scenarios_with_carbon = scenario_yearly_summary['has_carbon_price'].sum()
            total_scenarios = len(scenario_yearly_summary)
            
            print(f"\n📈 SUMMARY STATISTICS:")
            print(f"   Total scenarios: {total_scenarios}")
            print(f"   Scenarios with carbon price (at least 1 year): {scenarios_with_carbon}")
            print(f"   Scenarios without carbon price: {total_scenarios - scenarios_with_carbon}")
            print(f"   Percentage with carbon price: {scenarios_with_carbon/total_scenarios*100:.1f}%")
            
            # Save the summary table
            scenario_yearly_summary.to_csv("carbon_price_scenario_summary.csv", index=False)
            print(f"\n💾 Saved detailed summary to: carbon_price_scenario_summary.csv")
            
        else:
            print("❌ Required columns not found in final file")
            
    except Exception as e:
        print(f"❌ Error analyzing final file: {str(e)}")

if __name__ == "__main__":
    analyze_carbon_price_coverage()