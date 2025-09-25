#!/usr/bin/env python3
"""
Create descriptive statistics summary for AR6 scenarios.
Analyzes 5_final_AR6_complete_cases.csv to generate a summary of:
- Provider/scenario names
- Technologies included
- Amount of gapfilling
"""

import pandas as pd
import numpy as np

def create_scenario_summary():
    # Load the complete cases data
    print("Loading data...")
    df = pd.read_csv('5_final_AR6_complete_cases.csv', low_memory=False)
    
    print(f"Loaded {len(df):,} records")
    
    # Create summary statistics by scenario provider and scenario
    print("Creating summary statistics...")
    
    summary_list = []
    
    # Group by scenario_provider and scenario
    grouped = df.groupby(['scenario_provider', 'scenario'])
    
    for (provider, scenario), group in grouped:
        # Count technologies
        technologies = group['technology'].unique()
        tech_count = len(technologies)
        tech_list = ', '.join(sorted(technologies))
        
        # Count records with and without gap filling
        gap_filled_records = group['gap_filled_columns'].notna().sum()
        total_records = len(group)
        gap_fill_percentage = (gap_filled_records / total_records) * 100
        
        # Analyze types of gap filling
        gap_filled_data = group[group['gap_filled_columns'].notna()]
        unique_gap_patterns = gap_filled_data['gap_filled_columns'].nunique() if len(gap_filled_data) > 0 else 0
        
        # Most common gap filling pattern
        most_common_gap_pattern = ""
        if len(gap_filled_data) > 0:
            pattern_counts = gap_filled_data['gap_filled_columns'].value_counts()
            if len(pattern_counts) > 0:
                most_common_gap_pattern = pattern_counts.index[0]
        
        # Count unique countries
        unique_countries = 0
        if 'country_iso2_list' in group.columns:
            # Count unique non-null country entries
            country_data = group[group['country_iso2_list'].notna()]
            if len(country_data) > 0:
                all_countries = set()
                for country_list in country_data['country_iso2_list']:
                    if pd.notna(country_list):
                        countries = str(country_list).split(',')
                        all_countries.update([c.strip() for c in countries if c.strip()])
                unique_countries = len(all_countries)
        
        # Year range
        year_min = group['scenario_year'].min()
        year_max = group['scenario_year'].max()
        
        # EBITDA flag statistics - check if positive at least once in time series
        # Group by technology to check each technology's time series
        tech_with_positive_ebitda = 0
        total_techs_in_scenario = len(group['technology'].unique())
        
        for tech in group['technology'].unique():
            tech_data = group[group['technology'] == tech]
            if (tech_data['EBITDA_check'] == True).any():
                tech_with_positive_ebitda += 1
        
        ebitda_positive_tech_percentage = (tech_with_positive_ebitda / total_techs_in_scenario) * 100 if total_techs_in_scenario > 0 else 0
        
        # Analyze specific gapfilled columns
        def categorize_gapfill_columns(gap_string):
            if pd.isna(gap_string):
                return {'costs': False, 'efficiency_lifetime': False, 'fuel_price': False, 'scenario_price': False}
            
            gap_str = str(gap_string).lower()
            return {
                'costs': any(col in gap_str for col in ['capital_cost', 'om_cost']),
                'efficiency_lifetime': any(col in gap_str for col in ['efficiency_decimal', 'lifetime_years']),
                'fuel_price': 'fuel_price' in gap_str,
                'scenario_price': 'scenario_price' in gap_str
            }
        
        # Count gapfilling by category
        costs_gapfilled = 0
        efficiency_lifetime_gapfilled = 0
        fuel_price_gapfilled = 0
        scenario_price_gapfilled = 0
        
        for gap_string in group['gap_filled_columns']:
            categories = categorize_gapfill_columns(gap_string)
            if categories['costs']:
                costs_gapfilled += 1
            if categories['efficiency_lifetime']:
                efficiency_lifetime_gapfilled += 1
            if categories['fuel_price']:
                fuel_price_gapfilled += 1
            if categories['scenario_price']:
                scenario_price_gapfilled += 1
        
        # Calculate percentages
        costs_gapfill_pct = (costs_gapfilled / total_records) * 100
        efficiency_lifetime_gapfill_pct = (efficiency_lifetime_gapfilled / total_records) * 100
        fuel_price_gapfill_pct = (fuel_price_gapfilled / total_records) * 100
        scenario_price_gapfill_pct = (scenario_price_gapfilled / total_records) * 100
        
        summary_list.append({
            'scenario_provider': provider,
            'scenario': scenario,
            'total_records': total_records,
            'technology_count': tech_count,
            'technologies_included': tech_list,
            'records_with_gapfilling': gap_filled_records,
            'records_without_gapfilling': total_records - gap_filled_records,
            'gapfill_percentage': round(gap_fill_percentage, 2),
            'costs_gapfilled_count': costs_gapfilled,
            'costs_gapfilled_percentage': round(costs_gapfill_pct, 2),
            'efficiency_lifetime_gapfilled_count': efficiency_lifetime_gapfilled,
            'efficiency_lifetime_gapfilled_percentage': round(efficiency_lifetime_gapfill_pct, 2),
            'fuel_price_gapfilled_count': fuel_price_gapfilled,
            'fuel_price_gapfilled_percentage': round(fuel_price_gapfill_pct, 2),
            'scenario_price_gapfilled_count': scenario_price_gapfilled,
            'scenario_price_gapfilled_percentage': round(scenario_price_gapfill_pct, 2),
            'unique_gapfill_patterns': unique_gap_patterns,
            'most_common_gapfill_pattern': most_common_gap_pattern,
            'technologies_with_positive_ebitda': tech_with_positive_ebitda,
            'total_technologies_in_scenario': total_techs_in_scenario,
            'technologies_with_positive_ebitda_percentage': round(ebitda_positive_tech_percentage, 2),
            'unique_countries': unique_countries,
            'year_range_start': year_min,
            'year_range_end': year_max,
            'scenario_type': group['scenario_type'].iloc[0] if 'scenario_type' in group.columns else '',
            'scenario_geography': group['scenario_geography'].iloc[0] if 'scenario_geography' in group.columns else ''
        })
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_list)
    
    # Sort by scenario provider and scenario
    summary_df = summary_df.sort_values(['scenario_provider', 'scenario'])
    
    # Save to CSV
    output_file = 'scenario_summary_statistics.csv'
    summary_df.to_csv(output_file, index=False)
    
    print(f"\nSummary statistics saved to {output_file}")
    print(f"Generated summary for {len(summary_df)} unique scenario combinations")
    print(f"Total scenario providers: {summary_df['scenario_provider'].nunique()}")
    print(f"Total unique scenarios: {len(summary_df)}")
    
    # Display some key statistics
    print("\n=== KEY STATISTICS ===")
    print(f"Average technologies per scenario: {summary_df['technology_count'].mean():.1f}")
    print(f"Average gapfill percentage: {summary_df['gapfill_percentage'].mean():.1f}%")
    print(f"Scenarios with >50% gapfilling: {(summary_df['gapfill_percentage'] > 50).sum()}")
    print(f"Scenarios with no gapfilling: {(summary_df['gapfill_percentage'] == 0).sum()}")
    print(f"Average % of technologies with positive EBITDA: {summary_df['technologies_with_positive_ebitda_percentage'].mean():.1f}%")
    print(f"Scenarios where >50% of techs have positive EBITDA: {(summary_df['technologies_with_positive_ebitda_percentage'] > 50).sum()}")
    print(f"Scenarios where 0% of techs have positive EBITDA: {(summary_df['technologies_with_positive_ebitda_percentage'] == 0).sum()}")
    print(f"Average costs gapfilled: {summary_df['costs_gapfilled_percentage'].mean():.1f}%")
    print(f"Average efficiency/lifetime gapfilled: {summary_df['efficiency_lifetime_gapfilled_percentage'].mean():.1f}%")
    print(f"Average fuel price gapfilled: {summary_df['fuel_price_gapfilled_percentage'].mean():.1f}%")
    print(f"Average scenario price gapfilled: {summary_df['scenario_price_gapfilled_percentage'].mean():.1f}%")
    
    # Top scenario providers by number of scenarios
    print("\n=== TOP SCENARIO PROVIDERS ===")
    provider_counts = summary_df['scenario_provider'].value_counts().head(10)
    print(provider_counts)
    
    # Technologies summary
    print("\n=== TECHNOLOGY COVERAGE ===")
    tech_counts = summary_df['technology_count'].value_counts().sort_index()
    print(tech_counts)
    
    return summary_df

if __name__ == "__main__":
    summary_df = create_scenario_summary()