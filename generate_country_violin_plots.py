#!/usr/bin/env python3
"""
Country-Specific AR6 Violin Plot Generator
==========================================

This script generates violin plots for specific countries (China, Brazil, USA, 
Germany, South Africa) showing efficiency and other metrics by stringency category.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
METRICS = [
    'efficiency_decimal',
    'lifetime_years', 
    'om_cost_usd_per_mw_per_yr',
    'capital_cost_usd_per_mw',
    'scenario_price',
    'fuel_price'
]

METRIC_LABELS = {
    'efficiency_decimal': 'Efficiency\\n(decimal)',
    'lifetime_years': 'Lifetime\\n(years)',
    'om_cost_usd_per_mw_per_yr': 'O&M Cost\\n(USD/MW/yr)',
    'capital_cost_usd_per_mw': 'Capital Cost\\n(USD/MW)',
    'scenario_price': 'Electricity Price\\n(USD/MWh)',
    'fuel_price': 'Fuel Price\\n(USD/MWh)'
}

# Target countries with their ISO codes and names
TARGET_COUNTRIES = {
    'CN': 'China',
    'BR': 'Brazil', 
    'US': 'USA',
    'DE': 'Germany',
    'ZA': 'South Africa'
}

# Stringency order for consistent plotting
STRINGENCY_ORDER = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'UNKNOWN']

def load_and_filter_data():
    """Load data and filter for target countries"""
    print("📂 Loading AR6 data with stringency...")
    
    # Try to load the complete file first (with extreme value filtering applied)
    for filename in ['4_final_AR6_gapfilled_complete.csv', '../4_final_AR6_gapfilled_complete.csv']:
        try:
            df = pd.read_csv(filename, low_memory=False)
            print(f"   ✅ Loaded {len(df):,} rows from {filename}")
            print(f"   📝 Note: Data includes extreme value filtering (efficiency: 0.2-1.0, prices: 0-200 USD/MWh, costs: reasonable bounds)")
            break
        except FileNotFoundError:
            continue
    else:
        print("   ❌ No AR6 files found!")
        return None, {}
    
    # Check if stringency column exists
    if 'stringency' not in df.columns:
        print("   ⚠️  No stringency column found!")
        return None, {}
    
    # Filter data to countries of interest
    print("🌍 Filtering data by target countries...")
    country_data = {}
    
    for iso_code, country_name in TARGET_COUNTRIES.items():
        # Filter rows where country_iso2_list contains the target country code
        country_mask = df['country_iso2_list'].str.contains(iso_code, na=False)
        country_df = df[country_mask].copy()
        
        if len(country_df) > 0:
            print(f"   {country_name} ({iso_code}): {len(country_df):,} rows")
            
            # Show stringency distribution for this country
            stringency_counts = country_df['stringency'].value_counts()
            print(f"     Stringency distribution: {dict(stringency_counts)}")
            
            # Filter out gap-filled values
            print(f"     Filtering out gap-filled values...")
            df_original = country_df.copy()
            gap_filled_count = 0
            
            for idx, row in country_df.iterrows():
                gap_filled_str = row.get('gap_filled_columns')
                if pd.isna(gap_filled_str) or gap_filled_str == '':
                    continue
                    
                # Parse the gap-filled columns
                gap_filled_cols = []
                for col_str in str(gap_filled_str).split(','):
                    base_col = col_str.split('(')[0].strip()
                    if base_col in METRICS:
                        gap_filled_cols.append(base_col)
                
                # Set gap-filled values to NaN
                for col in gap_filled_cols:
                    if col in df_original.columns:
                        df_original.at[idx, col] = np.nan
                        gap_filled_count += 1
            
            print(f"     Set {gap_filled_count:,} gap-filled values to NaN")
            country_data[country_name] = df_original
        else:
            print(f"   {country_name} ({iso_code}): No data found")
    
    return df, country_data

def get_technologies_with_data(country_data, min_data_points=20):
    """Get technologies that have sufficient data across countries"""
    print("📊 Finding technologies with sufficient data...")
    
    tech_data_counts = {}
    
    for country_name, country_df in country_data.items():
        for tech in country_df['technology'].unique():
            if pd.isna(tech):
                continue
                
            tech_df = country_df[country_df['technology'] == tech]
            
            # Count non-null values across all metrics
            data_count = 0
            for metric in METRICS:
                if metric in tech_df.columns:
                    data_count += tech_df[metric].notna().sum()
            
            if tech not in tech_data_counts:
                tech_data_counts[tech] = {}
            tech_data_counts[tech][country_name] = data_count
    
    # Filter to technologies with data in at least 3 countries
    good_techs = {}
    for tech, country_counts in tech_data_counts.items():
        countries_with_data = sum(1 for count in country_counts.values() if count >= min_data_points)
        if countries_with_data >= 3:
            total_data = sum(country_counts.values())
            good_techs[tech] = total_data
    
    # Sort by total data points
    sorted_techs = sorted(good_techs.items(), key=lambda x: x[1], reverse=True)
    selected_techs = [tech for tech, count in sorted_techs[:15]]  # Top 15
    
    print(f"   Selected {len(selected_techs)} technologies:")
    for tech, total_count in sorted_techs[:15]:
        print(f"     {tech}: {total_count:,} total data points")
    
    return selected_techs

def create_country_violin_plot(country_data, technology, output_dir):
    """Create violin plots for a technology across countries"""
    print(f"📈 Creating country violin plots for {technology}...")
    
    # Check which countries have data for this technology
    countries_with_data = []
    for country_name, country_df in country_data.items():
        tech_df = country_df[country_df['technology'] == technology]
        if len(tech_df) > 10:  # Minimum threshold
            countries_with_data.append(country_name)
    
    if len(countries_with_data) == 0:
        print(f"   No countries with sufficient data for {technology}")
        return
    
    print(f"   Countries with data: {', '.join(countries_with_data)}")
    
    # Create figure - one row per country
    n_countries = len(countries_with_data)
    fig, axes = plt.subplots(n_countries, len(METRICS), figsize=(18, 4*n_countries))
    if n_countries == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Technology: {technology} by Country and Stringency', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Colors for stringency categories
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for country_idx, country_name in enumerate(countries_with_data):
        country_df = country_data[country_name]
        tech_df = country_df[country_df['technology'] == technology].copy()
        
        for metric_idx, metric in enumerate(METRICS):
            ax = axes[country_idx, metric_idx]
            
            # Filter to non-null metric values
            metric_values = tech_df[tech_df[metric].notna()]
            
            if len(metric_values) == 0:
                ax.text(0.5, 0.5, f'No Data\\n{country_name}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, alpha=0.5)
                ax.set_title(METRIC_LABELS[metric], fontweight='bold', fontsize=10)
                continue
            
            # Create data for violin plot by stringency
            stringency_data = []
            stringency_labels = []
            
            for stringency in STRINGENCY_ORDER:
                stringency_values = metric_values[metric_values['stringency'] == stringency][metric].dropna()
                if len(stringency_values) >= 3:  # Minimum for violin plot
                    stringency_data.append(stringency_values.values)
                    stringency_labels.append(stringency)
            
            if len(stringency_data) == 0:
                ax.text(0.5, 0.5, f'Insufficient Data\\n{country_name}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, alpha=0.5)
                ax.set_title(METRIC_LABELS[metric], fontweight='bold', fontsize=10)
                continue
            
            # Create violin plot
            try:
                parts = ax.violinplot(stringency_data, positions=range(len(stringency_labels)), 
                                    showmeans=True, showextrema=True, showmedians=True)
                
                # Style violin plots with stringency colors
                for j, pc in enumerate(parts['bodies']):
                    stringency_idx = STRINGENCY_ORDER.index(stringency_labels[j])
                    color = colors[stringency_idx % len(colors)]
                    pc.set_facecolor(color)
                    pc.set_alpha(0.6)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(0.5)
                
                # Style lines
                if 'cmeans' in parts:
                    parts['cmeans'].set_color('red')
                    parts['cmeans'].set_linewidth(1.5)
                if 'cmedians' in parts:
                    parts['cmedians'].set_color('orange')
                    parts['cmedians'].set_linewidth(1.5)
                
                # Add scatter overlay (sample for readability)
                for j, stringency in enumerate(stringency_labels):
                    stringency_values = metric_values[metric_values['stringency'] == stringency][metric].dropna()
                    if len(stringency_values) > 0:
                        # Sample points if too many
                        if len(stringency_values) > 50:
                            sample_values = stringency_values.sample(50)
                        else:
                            sample_values = stringency_values
                        
                        x_jitter = np.random.normal(j, 0.05, len(sample_values))
                        ax.scatter(x_jitter, sample_values, alpha=0.4, s=6, color='darkblue', zorder=3)
                
                # Customize axis
                ax.set_xticks(range(len(stringency_labels)))
                ax.set_xticklabels(stringency_labels, rotation=45, fontsize=8)
                ax.set_title(METRIC_LABELS[metric], fontweight='bold', fontsize=10, pad=5)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add country label on leftmost plot
                if metric_idx == 0:
                    ax.set_ylabel(f'{country_name}\\nValue', fontweight='bold')
                
                # Format y-axis based on metric type
                if metric in ['om_cost_usd_per_mw_per_yr', 'capital_cost_usd_per_mw']:
                    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
                elif metric in ['scenario_price', 'fuel_price']:
                    ax.set_ylim(top=500)  # Cap price plots at 500 USD/MWh for better readability
                
            except Exception as e:
                print(f"   Warning: Could not create violin plot for {metric} in {country_name}: {e}")
                ax.text(0.5, 0.5, f'Plot Error\\n{country_name}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, alpha=0.5)
                ax.set_title(METRIC_LABELS[metric], fontweight='bold', fontsize=10)
    
    # Legend removed for cleaner plots
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3, left=0.08)
    
    # Save plot
    filename = f"{technology.replace('/', '_').replace(' ', '_')}_country_violin.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Saved: {filename}")

def main():
    """Main function"""
    print("🌍 AR6 Country-Specific Violin Plot Generator")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("plots_country_stringency")
    output_dir.mkdir(exist_ok=True)
    
    # Load and filter data by country
    df, country_data = load_and_filter_data()
    
    if not country_data:
        print("❌ No country data available")
        return
    
    # Find technologies with sufficient data
    selected_techs = get_technologies_with_data(country_data)
    
    if not selected_techs:
        print("❌ No technologies with sufficient data across countries")
        return
    
    print(f"\\n📊 Generating country violin plots for {len(selected_techs)} technologies...")
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Generate plots
    for i, technology in enumerate(selected_techs, 1):
        print(f"\\n[{i}/{len(selected_techs)}] Processing {technology}...")
        try:
            create_country_violin_plot(country_data, technology, output_dir)
        except Exception as e:
            print(f"   ❌ Error creating plot for {technology}: {e}")
            continue
    
    print(f"\\n✅ Complete! Generated plots in {output_dir.absolute()}")
    print("\\nPlot Features:")
    print("- Each row = different country (China, Brazil, USA, Germany, South Africa)")
    print("- Each column = different metric (efficiency, lifetime, costs, prices)")
    print("- X-axis: Stringency categories (C1=most ambitious → C8=least ambitious)")
    print("- Colors: Different colors for each stringency category")
    print("- Shows country-specific efficiency patterns by climate ambition level")

if __name__ == "__main__":
    main()