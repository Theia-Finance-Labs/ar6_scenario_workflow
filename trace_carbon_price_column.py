#!/usr/bin/env python3
"""
Trace carbon price column through intermediate CSVs
"""

import pandas as pd
import numpy as np
from pathlib import Path

def check_carbon_price_column(filepath, stage_name):
    """Check carbon price column in a specific file"""
    
    print(f"\n{stage_name}: {filepath}")
    print("-" * 60)
    
    try:
        # Check if file exists
        if not Path(filepath).exists():
            print("❌ File not found")
            return
        
        # Read just the header first
        df_header = pd.read_csv(filepath, nrows=0)
        columns = list(df_header.columns)
        
        # Check if carbon price column exists
        carbon_price_col = None
        for col in columns:
            if 'carbon_price' in col.lower():
                carbon_price_col = col
                break
        
        if carbon_price_col is None:
            print("❌ No carbon_price column found")
            print(f"📋 Available columns ({len(columns)}): {', '.join(columns[:10])}{'...' if len(columns) > 10 else ''}")
            return
        
        print(f"✅ Carbon price column found: '{carbon_price_col}'")
        
        # Read a sample of data to check if it's populated
        sample_size = min(10000, 50000)  # Read reasonable sample
        df_sample = pd.read_csv(filepath, nrows=sample_size)
        
        if carbon_price_col not in df_sample.columns:
            print("❌ Column disappeared when reading data")
            return
        
        # Check population status
        total_rows_in_sample = len(df_sample)
        non_null_count = df_sample[carbon_price_col].notna().sum()
        null_count = df_sample[carbon_price_col].isna().sum()
        
        print(f"📊 Sample size: {total_rows_in_sample:,} rows")
        print(f"✅ Non-null values: {non_null_count:,} ({non_null_count/total_rows_in_sample*100:.1f}%)")
        print(f"❌ Null values: {null_count:,} ({null_count/total_rows_in_sample*100:.1f}%)")
        
        # Show sample values if any exist
        if non_null_count > 0:
            sample_values = df_sample[df_sample[carbon_price_col].notna()][carbon_price_col].head(5).tolist()
            print(f"💰 Sample values: {sample_values}")
        else:
            print("💰 Sample values: None (all null)")
        
        # Check unique values
        unique_vals = df_sample[carbon_price_col].unique()
        print(f"🔍 Unique values count: {len(unique_vals)}")
        if len(unique_vals) <= 10:
            print(f"🔍 All unique values: {list(unique_vals)}")
        else:
            print(f"🔍 Sample unique values: {list(unique_vals[:10])}")
            
        return {
            'file': filepath,
            'has_carbon_price_col': True,
            'carbon_price_col_name': carbon_price_col,
            'sample_size': total_rows_in_sample,
            'non_null_count': non_null_count,
            'null_count': null_count,
            'unique_values': len(unique_vals)
        }
        
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")
        return {
            'file': filepath,
            'has_carbon_price_col': False,
            'error': str(e)
        }

def trace_carbon_price_pipeline():
    """Trace carbon price column through entire pipeline"""
    
    print("TRACING CARBON PRICE COLUMN THROUGH PIPELINE")
    print("=" * 80)
    
    # Define pipeline stages
    pipeline_stages = [
        ("1. Raw AR6 (ISO3)", "data/AR6_Scenarios_Database_ISO3_v1.1.feather"),
        ("2. Intermediate ISO3", "1_intermediate_AR6_scenario_formatting_ISO3.csv"),
        ("3. Filtered", "2_final_AR6_filtered.csv"),
        ("4. Target Schema", "3_final_AR6_target_schema.csv"),
        ("5. Gap Filled", "4_final_AR6_gapfilled.csv"),
        ("6. Complete Cases", "5_final_AR6_complete_cases.csv"),
        ("7. Final Viable", "6_final_AR6_viable_scenarios.csv")
    ]
    
    results = []
    
    for stage_name, filepath in pipeline_stages:
        if filepath.endswith('.feather'):
            # Handle feather file differently
            print(f"\n{stage_name}: {filepath}")
            print("-" * 60)
            try:
                df = pd.read_feather(filepath)
                print(f"📊 Feather file loaded: {len(df):,} rows, {len(df.columns)} columns")
                
                # Check for carbon price variable
                if 'Variable' in df.columns:
                    carbon_price_data = df[df['Variable'] == 'Price|Carbon']
                    print(f"✅ Price|Carbon variable found: {len(carbon_price_data):,} records")
                    print(f"🎯 Scenarios with carbon price: {carbon_price_data['Scenario'].nunique()}")
                else:
                    print("❌ No Variable column found in feather file")
                    
                results.append({
                    'stage': stage_name,
                    'file': filepath,
                    'has_carbon_price_data': True if 'Variable' in df.columns else False,
                    'format': 'feather'
                })
            except Exception as e:
                print(f"❌ Error reading feather file: {str(e)}")
                results.append({
                    'stage': stage_name,
                    'file': filepath,
                    'error': str(e),
                    'format': 'feather'
                })
        else:
            # Handle CSV files
            result = check_carbon_price_column(filepath, stage_name)
            if result:
                result['stage'] = stage_name
                results.append(result)
    
    # Summary
    print(f"\n\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    
    for i, result in enumerate(results):
        if result.get('format') == 'feather':
            print(f"{i+1}. {result['stage']}: Raw data with Price|Carbon variable")
        elif result.get('has_carbon_price_col'):
            status = "✅ POPULATED" if result.get('non_null_count', 0) > 0 else "❌ EMPTY"
            print(f"{i+1}. {result['stage']}: {status} - {result.get('non_null_count', 0):,}/{result.get('sample_size', 0):,} values")
        else:
            print(f"{i+1}. {result['stage']}: ❌ NO COLUMN")
    
    return results

if __name__ == "__main__":
    results = trace_carbon_price_pipeline()