#!/usr/bin/env python3
"""
Check raw AR6 database for carbon price data
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
        
        # Look for carbon price related columns
        carbon_price_cols = [col for col in df.columns if 'carbon' in col.lower() and 'price' in col.lower()]
        print(f"\n🔍 Carbon price columns found: {carbon_price_cols}")
        
        # Look for any price columns
        price_cols = [col for col in df.columns if 'price' in col.lower()]
        print(f"💰 All price-related columns: {price_cols}")
        
        # Look for variables that might be carbon price
        if 'Variable' in df.columns:
            variables = df['Variable'].unique()
            carbon_price_variables = [var for var in variables if 'carbon' in str(var).lower() and 'price' in str(var).lower()]
            print(f"\n📈 Carbon price variables in data: {len(carbon_price_variables)}")
            if carbon_price_variables:
                print("   - " + "\n   - ".join(carbon_price_variables[:10]))  # Show first 10
                
                # Analyze carbon price data coverage
                for var in carbon_price_variables[:3]:  # Check first 3 variables
                    var_data = df[df['Variable'] == var]
                    non_null_values = var_data['Value'].notna().sum()
                    total_values = len(var_data)
                    scenarios_with_data = var_data[var_data['Value'].notna()]['Scenario'].nunique()
                    total_scenarios = var_data['Scenario'].nunique()
                    
                    print(f"\n   📊 {var}:")
                    print(f"      Values: {non_null_values:,}/{total_values:,} ({non_null_values/total_values*100:.1f}%)")
                    print(f"      Scenarios: {scenarios_with_data}/{total_scenarios}")
                    
                    # Sample values
                    sample_values = var_data[var_data['Value'].notna()]['Value'].head(5).tolist()
                    print(f"      Sample values: {sample_values}")
            else:
                print("   ❌ No carbon price variables found")
        
        # Check other potential carbon price columns
        potential_carbon_cols = [col for col in df.columns if 'carbon' in col.lower()]
        print(f"\n🔍 All carbon-related columns: {potential_carbon_cols}")
        
        return df, carbon_price_variables if 'Variable' in df.columns else []
        
    except Exception as e:
        print(f"❌ Error loading {ar6_file}: {str(e)}")
        return None, []

def check_ar6_variables_mapping():
    """Check the AR6 variables mapping file"""
    
    print(f"\n" + "="*60)
    print("Checking AR6 variables mapping")
    print("="*60)
    
    try:
        df_vars = pd.read_csv("ar6_variables_with_mapping.csv")
        print(f"📊 Variables mapping file has {len(df_vars)} rows")
        
        # Look for carbon price variables
        carbon_vars = df_vars[df_vars['variable'].str.contains('carbon.*price', case=False, na=False)]
        print(f"🔍 Carbon price variables in mapping: {len(carbon_vars)}")
        
        if len(carbon_vars) > 0:
            print("\nCarbon price variables found:")
            for _, row in carbon_vars.iterrows():
                print(f"   - {row['variable']}")
                if 'mapped_variable' in row:
                    print(f"     → {row['mapped_variable']}")
        
        return carbon_vars
        
    except Exception as e:
        print(f"❌ Error reading variables mapping: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Check raw AR6 database
    df, carbon_price_vars = check_raw_ar6_carbon_price()
    
    # Check variables mapping
    carbon_mapping = check_ar6_variables_mapping()
    
    # Summary
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if df is not None:
        print(f"✅ Raw AR6 database loaded successfully")
        print(f"📊 Total records: {len(df):,}")
        print(f"🔍 Carbon price variables: {len(carbon_price_vars)}")
    else:
        print("❌ Could not load raw AR6 database")
    
    print(f"📋 Carbon price variables in mapping: {len(carbon_mapping)}")
    
    if len(carbon_price_vars) == 0 and len(carbon_mapping) == 0:
        print("\n🚨 CONCLUSION: No carbon price data found in raw AR6 database")
        print("   The gaps in your final data are because the original AR6 database")
        print("   does not contain carbon price information for most scenarios.")
    elif len(carbon_price_vars) > 0:
        print(f"\n✅ CONCLUSION: Carbon price data exists in raw AR6 database")
        print("   The gaps in your final data may be due to filtering or processing steps")
        print("   that removed scenarios with carbon price data.")