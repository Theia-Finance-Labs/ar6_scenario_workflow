#!/usr/bin/env python3
"""
AR6 Violin Plot Generator with Proper Stringency 
================================================

This script generates violin plots using the official AR6 stringency mapping
that has been added to the processed files.
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

# Stringency order for consistent plotting
STRINGENCY_ORDER = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'UNKNOWN']

def load_and_filter_data():
    """Load data with proper stringency and filter out gap-filled values"""
    print("📂 Loading AR6 data with stringency...")
    
    # Try to load the complete file first
    for filename in ['../4_final_AR6_gapfilled_complete.csv', '../4_final_AR6_gapfilled.csv']:
        try:
            df = pd.read_csv(filename, low_memory=False)
            print(f"   ✅ Loaded {len(df):,} rows from {filename}")
            break
        except FileNotFoundError:
            print(f"   ⚠️  {filename} not found, trying next...")
            continue
    else:
        print("   ❌ No AR6 files found!")
        return None, []
    
    # Check if stringency column exists
    if 'stringency' not in df.columns:
        print("   ⚠️  No stringency column found!")
        print("   Please run add_stringency_to_existing_files.py first")
        return None, []
    
    # Show stringency distribution
    print("   📊 Stringency distribution in loaded data:")
    stringency_counts = df['stringency'].value_counts()
    for stringency in STRINGENCY_ORDER:
        count = stringency_counts.get(stringency, 0)
        if count > 0:
            print(f"     {stringency}: {count:,}")
    
    # Create a copy for filtering
    df_original = df.copy()
    
    # Parse gap_filled_columns and set those columns to NaN
    print("🔍 Filtering out gap-filled values...")
    gap_filled_count = 0
    
    for idx, row in df.iterrows():
        if idx % 500000 == 0:
            print(f"   Processed {idx:,} rows...")
            
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
    
    print(f"   Set {gap_filled_count:,} gap-filled values to NaN")
    
    # Filter to technologies with sufficient data
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
        
        if data_count > 50:  # Minimum threshold
            tech_data_counts[tech] = data_count
    
    # Sort technologies by data availability
    sorted_techs = sorted(tech_data_counts.items(), key=lambda x: x[1], reverse=True)
    selected_techs = [tech for tech, count in sorted_techs[:20]]  # Top 20
    
    print(f"   Selected {len(selected_techs)} technologies with sufficient data:")
    for tech, count in sorted_techs[:10]:  # Show top 10
        print(f"     {tech}: {count:,} data points")
    
    return df_original, selected_techs

def create_violin_plot(df, technology, output_dir):
    """Create violin plot for a technology by stringency"""
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
        metric_values = tech_df[tech_df[metric].notna()]
        
        if len(metric_values) == 0:
            ax.text(0.5, 0.5, 'No Data\\nAvailable', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, alpha=0.5)
            ax.set_title(METRIC_LABELS[metric], fontweight='bold')
            continue
        
        # Create data for violin plot by stringency
        stringency_data = []
        stringency_labels = []
        
        for stringency in STRINGENCY_ORDER:
            stringency_values = metric_values[metric_values['stringency'] == stringency][metric].dropna()
            if len(stringency_values) > 0:
                stringency_data.append(stringency_values.values)
                stringency_labels.append(stringency)
        
        if len(stringency_data) == 0:
            ax.text(0.5, 0.5, 'No Data\\nAvailable', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, alpha=0.5)
            ax.set_title(METRIC_LABELS[metric], fontweight='bold')
            continue
        
        # Create violin plot
        try:
            parts = ax.violinplot(stringency_data, positions=range(len(stringency_labels)), 
                                showmeans=True, showextrema=True, showmedians=True)
            
            # Style violin plots with stringency colors
            colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            for j, pc in enumerate(parts['bodies']):
                color = colors[j % len(colors)]
                pc.set_facecolor(color)
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
            
            # Add scatter overlay
            for j, stringency in enumerate(stringency_labels):
                stringency_values = metric_values[metric_values['stringency'] == stringency][metric].dropna()
                if len(stringency_values) > 0:
                    # Add jitter to x-coordinates
                    x_jitter = np.random.normal(j, 0.05, len(stringency_values))
                    ax.scatter(x_jitter, stringency_values, alpha=0.3, s=8, color='darkblue', zorder=3)
            
            # Customize axis
            ax.set_xticks(range(len(stringency_labels)))
            ax.set_xticklabels(stringency_labels, rotation=45)
            ax.set_ylabel('Value')
            ax.set_title(METRIC_LABELS[metric], fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Format y-axis
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
            ax.text(0.5, 0.5, f'Plot Error', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, alpha=0.5)
            ax.set_title(METRIC_LABELS[metric], fontweight='bold')
    
    # Add legend for stringency
    legend_elements = []
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Only add legend for stringencies that have data
    tech_stringencies = tech_df['stringency'].value_counts()
    for i, stringency in enumerate(STRINGENCY_ORDER):
        if stringency in tech_stringencies:
            color = colors[i % len(colors)]
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.6, label=stringency))
    
    if legend_elements:
        fig.legend(handles=legend_elements, title='Stringency Category', 
                  bbox_to_anchor=(0.02, 0.98), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.3)
    
    # Save plot
    filename = f"{technology.replace('/', '_').replace(' ', '_')}_stringency_violin.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Saved: {filename}")

def main():
    """Main function"""
    print("🎻 AR6 Violin Plot Generator (With Proper Stringency)")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("plots_with_stringency")
    output_dir.mkdir(exist_ok=True)
    
    # Load and filter data
    df_original, selected_techs = load_and_filter_data()
    
    if df_original is None:
        return
    
    print(f"\\n📊 Generating violin plots for {len(selected_techs)} technologies...")
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Generate plots
    for i, technology in enumerate(selected_techs, 1):
        print(f"\\n[{i}/{len(selected_techs)}] Processing {technology}...")
        try:
            create_violin_plot(df_original, technology, output_dir)
        except Exception as e:
            print(f"   ❌ Error creating plot for {technology}: {e}")
            continue
    
    print(f"\\n✅ Complete! Generated plots in {output_dir.absolute()}")
    print("\\nPlot Features:")
    print("- Violin shapes: Distribution density by stringency (C1-C8)")
    print("- Colors: Different colors for each stringency category")
    print("- Red line: Mean value")
    print("- Orange line: Median value")
    print("- Dots: Individual data points (original, non-gap-filled)")
    print("- X-axis: Stringency categories (C1=most ambitious, C8=least ambitious)")

if __name__ == "__main__":
    main()