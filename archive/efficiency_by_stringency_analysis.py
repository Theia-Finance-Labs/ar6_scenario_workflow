#!/usr/bin/env python3
"""
Efficiency by Stringency Analysis
=================================

Quick analysis of efficiency values by stringency category to understand
if the low efficiency issues are related to specific climate ambition levels.
"""

import pandas as pd
import numpy as np

def analyze_efficiency_by_stringency():
    """Analyze efficiency patterns by stringency category"""
    print("📊 Efficiency by Stringency Analysis")
    print("=" * 50)
    
    # Load data
    print("📂 Loading AR6 data...")
    df = pd.read_csv('4_final_AR6_gapfilled_complete.csv', low_memory=False)
    print(f"   Loaded {len(df):,} rows")
    
    # Focus on carbontech (non-renewable) with efficiency data
    carbontech_df = df[
        (df['technology_type'] == 'carbontech') & 
        (df['efficiency_decimal'].notna())
    ].copy()
    
    print(f"\n📈 Carbontech with efficiency data: {len(carbontech_df):,} rows")
    
    # Overall efficiency statistics by stringency
    print(f"\n🎯 Efficiency Statistics by Stringency Category:")
    efficiency_by_stringency = carbontech_df.groupby('stringency')['efficiency_decimal'].agg([
        'count', 'mean', 'median', 'min', 'max', 'std'
    ]).round(4)
    
    # Add percentage columns
    efficiency_by_stringency['pct_very_low'] = carbontech_df.groupby('stringency')['efficiency_decimal'].apply(
        lambda x: ((x < 0.1).sum() / len(x) * 100)
    ).round(2)
    
    efficiency_by_stringency['pct_negative'] = carbontech_df.groupby('stringency')['efficiency_decimal'].apply(
        lambda x: ((x < 0).sum() / len(x) * 100)
    ).round(2)
    
    print(efficiency_by_stringency.to_string())
    
    # Technology-specific analysis for problematic technologies
    print(f"\n🔧 Low Efficiency Technologies by Stringency:")
    problematic_techs = ['BiomassCap', 'BiomassCap - w/o CCS', 'BiomassCap - w/ CCS', 'CoalCap - w/ CCS']
    
    for tech in problematic_techs:
        tech_df = carbontech_df[carbontech_df['technology'] == tech]
        if len(tech_df) == 0:
            continue
            
        print(f"\n   {tech}:")
        tech_by_stringency = tech_df.groupby('stringency')['efficiency_decimal'].agg([
            'count', 'mean', 'min', 'max'
        ]).round(4)
        
        if len(tech_by_stringency) > 0:
            print(f"     {tech_by_stringency.to_string()}")
        
        # Show most problematic cases
        very_low = tech_df[tech_df['efficiency_decimal'] < 0.1]
        if len(very_low) > 0:
            stringency_counts = very_low['stringency'].value_counts()
            print(f"     Very low efficiency (<0.1) by stringency: {dict(stringency_counts)}")
    
    # Gap-filling analysis by stringency
    print(f"\n🔍 Gap-filling Patterns by Stringency:")
    gapfilled_efficiency = df[df['gap_filled_columns'].str.contains('efficiency_decimal', na=False)]
    if len(gapfilled_efficiency) > 0:
        gapfill_by_stringency = gapfilled_efficiency['stringency'].value_counts()
        total_by_stringency = df['stringency'].value_counts()
        
        print("   Stringency | Gap-filled Efficiency | Total Scenarios | Gap-fill Rate")
        print("   " + "-" * 65)
        for stringency in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'UNKNOWN']:
            if stringency in gapfill_by_stringency.index and stringency in total_by_stringency.index:
                gapfilled = gapfill_by_stringency[stringency]
                total = total_by_stringency[stringency]
                rate = gapfilled / total * 100
                print(f"   {stringency:>10} | {gapfilled:>19,} | {total:>15,} | {rate:>11.1f}%")
    
    print(f"\n✅ Analysis Complete!")
    print(f"\nKey Insights:")
    print(f"- Efficiency issues across all stringency categories")
    print(f"- More ambitious scenarios (C1-C3) vs less ambitious (C6-C8)")
    print(f"- Gap-filling patterns by climate ambition level")

if __name__ == "__main__":
    analyze_efficiency_by_stringency()