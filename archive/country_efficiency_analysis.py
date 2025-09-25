#!/usr/bin/env python3
"""
Country-Specific Efficiency Analysis by Stringency
===================================================

Analyzes efficiency patterns across China, Brazil, USA, Germany, and South Africa
to understand how technology efficiency varies by country and climate ambition level.
"""

import pandas as pd
import numpy as np

# Target countries
TARGET_COUNTRIES = {
    'CN': 'China',
    'BR': 'Brazil', 
    'US': 'USA',
    'DE': 'Germany',
    'ZA': 'South Africa'
}

def analyze_country_efficiency_patterns():
    """Analyze efficiency patterns by country and stringency"""
    print("🌍 Country-Specific Efficiency Analysis by Stringency")
    print("=" * 60)
    
    # Load data
    print("📂 Loading AR6 data...")
    df = pd.read_csv('4_final_AR6_gapfilled_complete.csv', low_memory=False)
    print(f"   Loaded {len(df):,} rows")
    
    # Focus on carbontech with efficiency data
    carbontech_df = df[
        (df['technology_type'] == 'carbontech') & 
        (df['efficiency_decimal'].notna())
    ].copy()
    
    print(f"📈 Carbontech with efficiency data: {len(carbontech_df):,} rows")
    
    # Filter to target countries and analyze each
    print(f"\n🎯 Efficiency Analysis by Country and Stringency:")
    
    country_results = {}
    
    for iso_code, country_name in TARGET_COUNTRIES.items():
        # Filter for this country
        country_mask = carbontech_df['country_iso2_list'].str.contains(iso_code, na=False)
        country_df = carbontech_df[country_mask]
        
        if len(country_df) == 0:
            print(f"\n   {country_name}: No data available")
            continue
            
        print(f"\n   {country_name} ({iso_code}): {len(country_df):,} rows")
        
        # Overall efficiency stats by stringency
        efficiency_by_stringency = country_df.groupby('stringency')['efficiency_decimal'].agg([
            'count', 'mean', 'median', 'min', 'max'
        ]).round(4)
        
        # Add problematic value percentages
        for stringency in efficiency_by_stringency.index:
            stringency_data = country_df[country_df['stringency'] == stringency]['efficiency_decimal']
            very_low_pct = (stringency_data < 0.1).sum() / len(stringency_data) * 100
            negative_pct = (stringency_data < 0).sum() / len(stringency_data) * 100
            efficiency_by_stringency.loc[stringency, 'very_low_pct'] = round(very_low_pct, 2)
            efficiency_by_stringency.loc[stringency, 'negative_pct'] = round(negative_pct, 2)
        
        print(f"     Overall efficiency by stringency:")
        print(f"     {efficiency_by_stringency.to_string()}")
        
        # Store for comparison
        country_results[country_name] = efficiency_by_stringency
        
        # Technology-specific analysis for problematic technologies
        problematic_techs = ['BiomassCap - w/o CCS', 'CoalCap - w/ CCS']
        
        for tech in problematic_techs:
            tech_df = country_df[country_df['technology'] == tech]
            if len(tech_df) > 50:  # Only analyze if sufficient data
                print(f"\\n     {tech}:")
                tech_by_stringency = tech_df.groupby('stringency')['efficiency_decimal'].agg([
                    'count', 'mean', 'min', 'max'
                ]).round(4)
                
                print(f"       {tech_by_stringency.to_string()}")
                
                # Show worst cases
                very_low = tech_df[tech_df['efficiency_decimal'] < 0.1]
                if len(very_low) > 0:
                    worst_stringencies = very_low['stringency'].value_counts()
                    print(f"       Very low efficiency (<0.1): {dict(worst_stringencies)}")
    
    # Cross-country comparison
    print(f"\n🔄 Cross-Country Efficiency Comparison:")
    
    if len(country_results) >= 2:
        print(f"\\n   Mean Efficiency by Country and Stringency:")
        
        # Create comparison table
        comparison_data = []
        for country, results in country_results.items():
            for stringency in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']:
                if stringency in results.index:
                    comparison_data.append({
                        'country': country,
                        'stringency': stringency,
                        'mean_efficiency': results.loc[stringency, 'mean'],
                        'count': results.loc[stringency, 'count'],
                        'very_low_pct': results.loc[stringency, 'very_low_pct']
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Pivot for easier viewing
            pivot_mean = comparison_df.pivot(index='country', columns='stringency', values='mean_efficiency')
            pivot_count = comparison_df.pivot(index='country', columns='stringency', values='count')
            
            print(f"\\n   Mean Efficiency (by country × stringency):")
            print(f"   {pivot_mean.round(4).to_string()}")
            
            print(f"\\n   Data Point Counts (by country × stringency):")
            print(f"   {pivot_count.to_string()}")
            
            # Find patterns
            print(f"\\n   Key Patterns:")
            
            # Best/worst performing countries by stringency
            for stringency in ['C1', 'C3', 'C5', 'C8']:
                if stringency in pivot_mean.columns:
                    stringency_means = pivot_mean[stringency].dropna()
                    if len(stringency_means) > 0:
                        best_country = stringency_means.idxmax()
                        worst_country = stringency_means.idxmin()
                        best_eff = stringency_means.max()
                        worst_eff = stringency_means.min()
                        
                        print(f"     {stringency}: Best={best_country}({best_eff:.3f}), Worst={worst_country}({worst_eff:.3f})")
            
            # Countries with most problematic efficiency values
            very_low_by_country = comparison_df.groupby('country')['very_low_pct'].mean().sort_values(ascending=False)
            print(f"\\n   Countries with highest % very low efficiency (<0.1):")
            for country, pct in very_low_by_country.head(3).items():
                print(f"     {country}: {pct:.2f}%")
    
    print(f"\n✅ Country Analysis Complete!")
    print(f"\nKey Insights:")
    print(f"- Efficiency patterns vary significantly by country and stringency")
    print(f"- More ambitious scenarios (C1-C3) may have different efficiency assumptions")
    print(f"- Country-specific technology deployment and efficiency assumptions")
    print(f"- See generated violin plots for detailed distributions")

if __name__ == "__main__":
    analyze_country_efficiency_patterns()