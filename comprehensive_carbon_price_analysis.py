#!/usr/bin/env python3
"""
Comprehensive Carbon Price Analysis
Analyzes carbon price data from raw AR6 through final processed data
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_raw_carbon_price():
    """Analyze carbon price in raw AR6 database"""
    
    print("STEP 1: RAW AR6 DATABASE ANALYSIS")
    print("=" * 50)
    
    try:
        # Load raw AR6 data
        df_raw = pd.read_feather("data/AR6_Scenarios_Database_ISO3_v1.1.feather")
        
        # Filter for carbon price data
        carbon_price_data = df_raw[df_raw['Variable'] == 'Price|Carbon'].copy()
        
        print(f"📊 Total carbon price records in raw data: {len(carbon_price_data):,}")
        
        if len(carbon_price_data) == 0:
            print("❌ No carbon price data found in raw database")
            return None
            
        # Melt the data to get year-value pairs
        year_cols = [col for col in carbon_price_data.columns if col.isdigit()]
        melted = carbon_price_data.melt(
            id_vars=['Model', 'Scenario', 'Region', 'Variable', 'Unit'],
            value_vars=year_cols,
            var_name='Year',
            value_name='CarbonPrice'
        )
        
        # Remove null values
        melted = melted[melted['CarbonPrice'].notna()]
        
        print(f"📈 Non-null carbon price data points: {len(melted):,}")
        print(f"🎯 Unique scenarios with carbon price: {melted['Scenario'].nunique()}")
        print(f"🌍 Unique regions with carbon price: {melted['Region'].nunique()}")
        print(f"📅 Year range: {melted['Year'].min()} - {melted['Year'].max()}")
        print(f"💰 Carbon price range: ${melted['CarbonPrice'].min():.2f} - ${melted['CarbonPrice'].max():.2f}")
        
        # Sample scenarios with carbon price
        sample_scenarios = melted['Scenario'].unique()[:10]
        print(f"\n📋 Sample scenarios with carbon price data:")
        for scenario in sample_scenarios:
            scenario_data = melted[melted['Scenario'] == scenario]
            years_with_data = sorted(scenario_data['Year'].unique())
            print(f"   - {scenario}: Years {years_with_data[:5]}{'...' if len(years_with_data) > 5 else ''}")
        
        return melted
        
    except Exception as e:
        print(f"❌ Error analyzing raw carbon price: {str(e)}")
        return None

def trace_carbon_price_through_pipeline(raw_carbon_data):
    """Trace what happens to carbon price data through the processing pipeline"""
    
    print(f"\n\nSTEP 2: TRACING CARBON PRICE THROUGH PIPELINE")
    print("=" * 50)
    
    if raw_carbon_data is None:
        print("❌ No raw carbon price data to trace")
        return
    
    # Get scenarios from raw data that have carbon price
    raw_scenarios_with_carbon = set(raw_carbon_data['Scenario'].unique())
    print(f"🎯 Raw scenarios with carbon price: {len(raw_scenarios_with_carbon)}")
    
    # Check each pipeline stage
    pipeline_files = [
        "2_final_AR6_filtered.csv",
        "3_final_AR6_target_schema.csv", 
        "4_final_AR6_gapfilled.csv",
        "5_final_AR6_complete_cases.csv",
        "6_final_AR6_viable_scenarios.csv"
    ]
    
    remaining_scenarios = raw_scenarios_with_carbon.copy()
    
    for i, filename in enumerate(pipeline_files):
        filepath = Path(filename)
        if not filepath.exists():
            print(f"⚠️  {filename} not found")
            continue
            
        print(f"\n{i+1}. Checking {filename}")
        try:
            # Read just the scenario column to check which scenarios remain
            df_chunk = pd.read_csv(filepath, usecols=['scenario'], nrows=100000)
            unique_scenarios_in_file = set(df_chunk['scenario'].unique())
            
            # Check overlap with carbon price scenarios
            scenarios_with_carbon_remaining = remaining_scenarios.intersection(unique_scenarios_in_file)
            scenarios_lost = remaining_scenarios - scenarios_with_carbon_remaining
            
            print(f"   📊 Total scenarios in file: {len(unique_scenarios_in_file)}")
            print(f"   ✅ Carbon price scenarios remaining: {len(scenarios_with_carbon_remaining)}")
            print(f"   ❌ Carbon price scenarios lost: {len(scenarios_lost)}")
            
            if len(scenarios_lost) > 0 and len(scenarios_lost) <= 5:
                print(f"   🚨 Lost scenarios: {list(scenarios_lost)}")
            elif len(scenarios_lost) > 5:
                print(f"   🚨 Sample lost scenarios: {list(scenarios_lost)[:5]}...")
            
            remaining_scenarios = scenarios_with_carbon_remaining
            
        except Exception as e:
            print(f"   ❌ Error checking {filename}: {str(e)}")
    
    return remaining_scenarios

def generate_final_summary_table():
    """Generate final summary table with carbon price flags"""
    
    print(f"\n\nSTEP 3: GENERATING FINAL SUMMARY TABLE")
    print("=" * 50)
    
    try:
        # Load the final data
        df_final = pd.read_csv("6_final_AR6_viable_scenarios.csv")
        
        # Get unique scenarios
        scenarios_in_final = df_final['scenario'].unique()
        print(f"📊 Total scenarios in final data: {len(scenarios_in_final)}")
        
        # Load raw carbon price data to check which scenarios originally had carbon price
        df_raw = pd.read_feather("data/AR6_Scenarios_Database_ISO3_v1.1.feather")
        carbon_price_data = df_raw[df_raw['Variable'] == 'Price|Carbon']
        
        # Get scenarios that have carbon price in raw data
        scenarios_with_carbon_price = set(carbon_price_data['Scenario'].unique())
        print(f"🎯 Scenarios with carbon price in raw data: {len(scenarios_with_carbon_price)}")
        
        # Create summary table
        summary_data = []
        for scenario in scenarios_in_final:
            has_carbon_price_in_raw = scenario in scenarios_with_carbon_price
            
            # Count records for this scenario in final data
            scenario_records = len(df_final[df_final['scenario'] == scenario])
            
            # Check years available
            scenario_data = df_final[df_final['scenario'] == scenario]
            years_available = sorted(scenario_data['scenario_year'].unique())
            year_range = f"{min(years_available)}-{max(years_available)}" if years_available else "None"
            
            summary_data.append({
                'scenario': scenario,
                'has_carbon_price_in_raw_data': has_carbon_price_in_raw,
                'carbon_price_flag': '✅ HAD_CARBON_PRICE' if has_carbon_price_in_raw else '❌ NO_CARBON_PRICE',
                'records_in_final_data': scenario_records,
                'year_range_in_final_data': year_range,
                'years_available': len(years_available)
            })
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by carbon price availability and scenario name
        summary_df = summary_df.sort_values(['has_carbon_price_in_raw_data', 'scenario'], ascending=[False, True])
        
        # Display summary
        print(f"\nSCENARIO CARBON PRICE SUMMARY")
        print("-" * 80)
        display_df = summary_df[['scenario', 'carbon_price_flag', 'records_in_final_data', 'year_range_in_final_data']]
        print(display_df.to_string(index=False))
        
        # Summary statistics
        scenarios_with_carbon = summary_df['has_carbon_price_in_raw_data'].sum()
        total_scenarios = len(summary_df)
        
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   Total scenarios in final data: {total_scenarios}")
        print(f"   Scenarios that had carbon price in raw data: {scenarios_with_carbon}")
        print(f"   Scenarios that never had carbon price: {total_scenarios - scenarios_with_carbon}")
        print(f"   Percentage that had carbon price: {scenarios_with_carbon/total_scenarios*100:.1f}%")
        
        # Save the summary
        summary_df.to_csv("carbon_price_final_summary.csv", index=False)
        print(f"\n💾 Saved detailed summary to: carbon_price_final_summary.csv")
        
        return summary_df
        
    except Exception as e:
        print(f"❌ Error generating summary table: {str(e)}")
        return None

def main():
    """Main analysis function"""
    
    print("COMPREHENSIVE CARBON PRICE ANALYSIS")
    print("=" * 60)
    print("This analysis will:")
    print("1. Check carbon price data in raw AR6 database")
    print("2. Trace what happens to carbon price scenarios through processing")
    print("3. Generate final summary table with carbon price flags")
    print("=" * 60)
    
    # Step 1: Analyze raw carbon price data
    raw_carbon_data = analyze_raw_carbon_price()
    
    # Step 2: Trace through pipeline
    remaining_scenarios = trace_carbon_price_through_pipeline(raw_carbon_data)
    
    # Step 3: Generate final summary
    summary_df = generate_final_summary_table()
    
    print(f"\n\n" + "="*60)
    print("FINAL CONCLUSION")
    print("="*60)
    
    if raw_carbon_data is not None and len(raw_carbon_data) > 0:
        print("✅ Carbon price data EXISTS in the raw AR6 database")
        print(f"   - {raw_carbon_data['Scenario'].nunique()} scenarios have carbon price data")
        print(f"   - {len(raw_carbon_data):,} data points across years")
        
        if remaining_scenarios is not None:
            print(f"   - {len(remaining_scenarios)} scenarios with carbon price made it to final data")
            
            if len(remaining_scenarios) == 0:
                print("\n🚨 ISSUE IDENTIFIED:")
                print("   All scenarios with carbon price data were filtered out during processing!")
                print("   This explains why your final data has no carbon price information.")
                print("\n🔧 RECOMMENDATIONS:")
                print("   1. Review filtering criteria in processing steps")
                print("   2. Check if carbon price scenarios match your target:")
                print("      - Geographic regions")  
                print("      - Technologies")
                print("      - Scenario types")
                print("   3. Consider relaxing filters to include carbon price scenarios")
            else:
                print(f"\n⚠️  PARTIAL ISSUE:")
                print(f"   Only {len(remaining_scenarios)} out of {raw_carbon_data['Scenario'].nunique()} carbon price scenarios remain")
                print("   Some carbon price data was lost during processing")
    else:
        print("❌ No carbon price data found in raw AR6 database")
        print("   The gaps in your final data are due to missing source data")

if __name__ == "__main__":
    main()