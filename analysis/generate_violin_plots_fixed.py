
def extract_stringency_from_scenario(scenario, temp_mapping):
    """Extract stringency from scenario name using temperature/CO2 target patterns"""
    if pd.isna(scenario):
        return 'UNKNOWN'
    
    scenario_str = str(scenario)
    
    # First, try exact C1-C8 pattern (for scenarios that already have it)
    for stringency in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']:
        if stringency in scenario_str:
            return stringency
    
    # Then try temperature/CO2 target mapping
    for temp_target in sorted(temp_mapping.keys(), key=len, reverse=True):
        if temp_target in scenario_str:
            return temp_mapping[temp_target]
    
    # Special cases
    if 'NPi' in scenario_str:
        return 'C8'
    elif 'INDC' in scenario_str and any(x in scenario_str for x in ['2030', '2050']):
        return 'C5'
    
    return 'UNKNOWN'

# Temperature/CO2 target to stringency mapping
TEMP_MAPPING = {'300': 'C1', '400': 'C1', '500': 'C2', '600': 'C2', '700': 'C3', '800': 'C3', '900': 'C4', '1000': 'C4', '1100': 'C5', '1200': 'C5', '1300': 'C6', '1400': 'C6', '1500': 'C7', '1600': 'C7', '1700': 'C8', '1800': 'C8', '1900': 'C8', '2000': 'C8'}
\n#!/usr/bin/env python3
"""
AR6 Non-Gap-Filled Data Violin Plot Generator
==============================================

This script analyzes the original (non-gap-filled) values from 4_final_AR6_gapfilled.csv
and creates publication-ready violin plots showing the distribution of key metrics
by technology and stringency level.

For each technology, creates a grid plot with:
- Y-axis: Stringency levels (C1-C8, UNKNOWN)
- X-axis: 6 metrics (efficiency, lifetime_years, om_cost, capital_cost, scenario_price, fuel_price)
- Violin plots with scatter overlay showing all countries

Output: PNG files in analysis/plots/ directory
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
    'efficiency_decimal': 'Efficiency\n(decimal)',
    'lifetime_years': 'Lifetime\n(years)',
    'om_cost_usd_per_mw_per_yr': 'O&M Cost\n(USD/MW/yr)',
    'capital_cost_usd_per_mw': 'Capital Cost\n(USD/MW)',
    'scenario_price': 'Electricity Price\n(USD/MWh)',
    'fuel_price': 'Fuel Price\n(USD/MWh)'
}

# Stringency order for consistent plotting
STRINGENCY_ORDER = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'UNKNOWN']

def load_and_filter_data():
    """
    Load the gap-filled data and create a version with only original (non-gap-filled) values
    by setting gap-filled values to NaN.
    """
    print("📂 Loading 4_final_AR6_gapfilled.csv...")
    df = pd.read_csv('../4_final_AR6_gapfilled.csv', low_memory=False)
    print(f"   Loaded {len(df):,} rows")
    
    # Create a copy for filtering
    df_original = df.copy()
    
    # Parse gap_filled_columns and set those columns to NaN
    print("🔍 Filtering out gap-filled values...")
    gap_filled_count = 0
    
    for idx, row in df.iterrows():
        if idx % 100000 == 0:
            print(f"   Processed {idx:,} rows...")
            
        gap_filled_str = row.get('gap_filled_columns')
        if pd.isna(gap_filled_str) or gap_filled_str == '':
            continue
            
        # Parse the gap-filled columns (comma-separated, may have suffixes like (global))
        gap_filled_cols = []
        for col_str in str(gap_filled_str).split(','):
            # Extract base column name (remove suffixes like (global), (tech_fallback))
            base_col = col_str.split('(')[0].strip()
            if base_col in METRICS:
                gap_filled_cols.append(base_col)
        
        # Set gap-filled values to NaN
        for col in gap_filled_cols:
            if col in df_original.columns:
                df_original.at[idx, col] = np.nan
                gap_filled_count += 1
    
    print(f"   Set {gap_filled_count:,} gap-filled values to NaN")
    
    # Add stringency column using improved mapping
    df_original['stringency'] = df_original['scenario'].apply(
        lambda x: extract_stringency_from_scenario(x, TEMP_MAPPING)
    )
    
    # Filter to technologies with sufficient data for plotting
    print("📊 Filtering technologies with sufficient data...")
    tech_data_counts = {}
    
    for tech in df_original['technology'].unique():
        if pd.isna(tech):
            continue
        tech_df = df_original[df_original['technology'] == tech]
        
        # Count non-null values across all metrics
        data_count = 0
        for metric in METRICS:
            if metric in tech_df.columns:
                data_count += tech_df[metric].notna().sum()
        
        if data_count > 50:  # Minimum threshold for meaningful plots
            tech_data_counts[tech] = data_count
    
    # Sort technologies by data availability
    sorted_techs = sorted(tech_data_counts.items(), key=lambda x: x[1], reverse=True)
    selected_techs = [tech for tech, count in sorted_techs[:20]]  # Top 20 technologies
    
    print(f"   Selected {len(selected_techs)} technologies with sufficient data:")
    for tech, count in sorted_techs[:20]:
        print(f"     {tech}: {count:,} data points")
    
    return df_original, selected_techs

def create_violin_plot(df, technology, output_dir):
    """
    Create a violin plot for a specific technology showing all metrics by stringency.
    """
    print(f"📈 Creating violin plot for {technology}...")
    
    # Filter data for this technology
    tech_df = df[df['technology'] == technology].copy()
    
    if len(tech_df) == 0:
        print(f"   No data found for {technology}")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Technology: {technology}', fontsize=16, fontweight='bold', y=0.98)
    
    axes = axes.flatten()
    
    for i, metric in enumerate(METRICS):
        ax = axes[i]
        
        # Prepare data for this metric
        plot_data = []
        metric_values = tech_df[tech_df[metric].notna()]
        
        if len(metric_values) == 0:
            ax.text(0.5, 0.5, 'No Data\nAvailable', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, alpha=0.5)
            ax.set_title(METRIC_LABELS[metric], fontweight='bold')
            continue
        
        # Create data for violin plot
        stringency_data = []
        stringency_labels = []
        
        for stringency in STRINGENCY_ORDER:
            stringency_values = metric_values[metric_values['stringency'] == stringency][metric].dropna()
            if len(stringency_values) > 0:
                stringency_data.append(stringency_values.values)
                stringency_labels.append(stringency)
        
        if len(stringency_data) == 0:
            ax.text(0.5, 0.5, 'No Data\nAvailable', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, alpha=0.5)
            ax.set_title(METRIC_LABELS[metric], fontweight='bold')
            continue
        
        # Create violin plot
        try:
            parts = ax.violinplot(stringency_data, positions=range(len(stringency_labels)), 
                                showmeans=True, showextrema=True, showmedians=True)
            
            # Style violin plots
            for pc in parts['bodies']:
                pc.set_facecolor('#1f77b4')
                pc.set_alpha(0.6)
                pc.set_edgecolor('black')
                pc.set_linewidth(0.5)
            
            # Style mean, median, extrema
            if 'cmeans' in parts:
                parts['cmeans'].set_color('red')
                parts['cmeans'].set_linewidth(2)
            if 'cmedians' in parts:
                parts['cmedians'].set_color('orange')
                parts['cmedians'].set_linewidth(2)
            if 'cbars' in parts:
                parts['cbars'].set_color('black')
                parts['cbars'].set_linewidth(1)
            if 'cmins' in parts and 'cmaxes' in parts:
                parts['cmins'].set_color('black')
                parts['cmaxes'].set_color('black')
                parts['cmins'].set_linewidth(1)
                parts['cmaxes'].set_linewidth(1)
            
            # Add scatter overlay
            for j, stringency in enumerate(stringency_labels):
                stringency_values = metric_values[metric_values['stringency'] == stringency][metric].dropna()
                if len(stringency_values) > 0:
                    # Add jitter to x-coordinates for better visibility
                    x_jitter = np.random.normal(j, 0.05, len(stringency_values))
                    ax.scatter(x_jitter, stringency_values, alpha=0.3, s=8, color='darkblue', zorder=3)
            
            # Customize axis
            ax.set_xticks(range(len(stringency_labels)))
            ax.set_xticklabels(stringency_labels, rotation=45)
            ax.set_ylabel('Value')
            ax.set_title(METRIC_LABELS[metric], fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Format y-axis based on metric type
            if metric in ['om_cost_usd_per_mw_per_yr', 'capital_cost_usd_per_mw']:
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            elif metric in ['scenario_price', 'fuel_price']:
                ax.set_ylabel('USD/MWh')
            elif metric == 'lifetime_years':
                ax.set_ylabel('Years')
            elif metric == 'efficiency_decimal':
                ax.set_ylabel('Efficiency (0-1)')
            
        except Exception as e:
            print(f"   Warning: Could not create violin plot for {metric}: {e}")
            ax.text(0.5, 0.5, f'Plot Error:\n{str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, alpha=0.5)
            ax.set_title(METRIC_LABELS[metric], fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)
    
    # Save plot
    filename = f"{technology.replace('/', '_').replace(' ', '_')}_violin_plot.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   Saved: {filename}")

def main():
    """Main analysis workflow"""
    print("🎻 AR6 Violin Plot Generator")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    
    # Load and filter data
    df_original, selected_techs = load_and_filter_data()
    
    print(f"\n📊 Generating violin plots for {len(selected_techs)} technologies...")
    
    # Generate plots for each technology
    for i, technology in enumerate(selected_techs, 1):
        print(f"\n[{i}/{len(selected_techs)}] Processing {technology}...")
        try:
            create_violin_plot(df_original, technology, output_dir)
        except Exception as e:
            print(f"   Error creating plot for {technology}: {e}")
            continue
    
    print(f"\n✅ Complete! Generated plots in {output_dir.absolute()}")
    print("\nPlot Legend:")
    print("- Violin shapes: Distribution density")
    print("- Red line: Mean value")
    print("- Orange line: Median value") 
    print("- Black dots: Individual data points (with jitter)")
    print("- Grid: Stringency (C1-C8) vs Metrics")

if __name__ == "__main__":
    main()