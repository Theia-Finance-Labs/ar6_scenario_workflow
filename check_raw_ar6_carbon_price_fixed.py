#!/usr/bin/env python3
"""
Check raw AR6 database for carbon price data - Fixed version
"""

import pandas as pd
import numpy as np

def check_raw_ar6_carbon_price():
    """Check if carbon price data exists in raw AR6 database"""
    
    print("Checking raw AR6 database for carbon price data")
    print("=" * 60)
    
    # Check the AR6 database
    ar6_file = "data/AR6_Scenarios_Database_ISO3_v1.1.feather"
    
    try:
        print(f"\n📁 Loading {ar6_file}...")
        df = pd.read_feather(ar6_file)
        
        print(f"📊 Total rows: {len(df):,}")
        print(f"📋 Total columns: {len(df.columns)}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Look for carbon price related columns
        carbon_price_cols = [col for col in df.columns if 'carbon' in col.lower() and 'price' in col.lower()]
        print(f"\n🔍 Carbon price columns found: {carbon_price_cols}")
        
        # Look for any price columns
        price_cols = [col for col in df.columns if 'price' in col.lower()]
        print(f"💰 All price-related columns: {price_cols}")
        
        # Look for variables that might be carbon price
        variable_col = None
        value_col = None
        for col in df.columns:
            if col.lower() in ['variable', 'variables']:
                variable_col = col
            if col.lower() in ['value', 'values']:
                value_col = col
        
        print(f"📊 Variable column: {variable_col}")
        print(f"📊 Value column: {value_col}")
        
        if variable_col is not None:
            variables = df[variable_col].unique()
            carbon_price_variables = [var for var in variables if 'carbon' in str(var).lower() and 'price' in str(var).lower()]
            print(f"\n📈 Carbon price variables in data: {len(carbon_price_variables)}")
            if carbon_price_variables:
                print("   - " + "\n   - ".join(carbon_price_variables))  # Show all
                
                # Analyze carbon price data coverage if we have a value column
                if value_col is not None:
                    for var in carbon_price_variables:
                        var_data = df[df[variable_col] == var]
                        if len(var_data) > 0:
                            non_null_values = var_data[value_col].notna().sum()
                            total_values = len(var_data)
                            
                            # Get scenario info
                            scenario_col = None
                            for col in df.columns:
                                if 'scenario' in col.lower():
                                    scenario_col = col
                                    break
                            
                            if scenario_col:
                                scenarios_with_data = var_data[var_data[value_col].notna()][scenario_col].nunique()
                                total_scenarios = var_data[scenario_col].nunique()
                                scenario_info = f"Scenarios: {scenarios_with_data}/{total_scenarios}"
                            else:
                                scenario_info = "Scenario info not available"
                            
                            print(f"\n   📊 {var}:")
                            print(f"      Values: {non_null_values:,}/{total_values:,} ({non_null_values/total_values*100:.1f}%)")
                            print(f"      {scenario_info}")
                            
                            # Sample values
                            sample_values = var_data[var_data[value_col].notna()][value_col].head(5).tolist()
                            print(f"      Sample values: {sample_values}")
                            
                            # Unique scenarios with this variable
                            if scenario_col:
                                unique_scenarios = var_data[var_data[value_col].notna()][scenario_col].unique()[:10]
                                print(f"      Sample scenarios: {list(unique_scenarios)}")
            else:
                print("   ❌ No carbon price variables found")
        
        # Check other potential carbon price columns
        potential_carbon_cols = [col for col in df.columns if 'carbon' in col.lower()]
        print(f"\n🔍 All carbon-related columns: {potential_carbon_cols}")
        
        return df, carbon_price_variables if variable_col is not None else []
        
    except Exception as e:
        print(f"❌ Error loading {ar6_file}: {str(e)}")
        return None, []

if __name__ == "__main__":
    # Check raw AR6 database
    df, carbon_price_vars = check_raw_ar6_carbon_price()
    
    # Summary
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if df is not None:
        print(f"✅ Raw AR6 database loaded successfully")
        print(f"📊 Total records: {len(df):,}")
        print(f"🔍 Carbon price variables: {len(carbon_price_vars)}")
        
        if len(carbon_price_vars) == 0:
            print("\n🚨 CONCLUSION: No carbon price data found in raw AR6 database")
            print("   The gaps in your final data are because the original AR6 database")
            print("   does not contain carbon price information.")
        else:
            print(f"\n✅ CONCLUSION: Carbon price data exists in raw AR6 database")
            print("   The gaps in your final data may be due to filtering or processing steps")
            print("   that removed scenarios with carbon price data, or the data may not")
            print("   overlap with your target scenarios/regions/technologies.")
    else:
        print("❌ Could not load raw AR6 database")