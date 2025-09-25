# Analysis Directory

This directory contains visualization and analysis scripts for the processed AR6 climate scenario data.

## Current Analysis Script

### `generate_violin_plots.py`
**Purpose**: Creates violin plots showing the distribution of key economic metrics across technologies and climate policy stringency levels.

**Data Source**: Uses `6_final_AR6_viable_scenarios.csv` (economically viable scenarios only)

**Analysis Features**:
- **Technology-by-Technology Analysis**: Creates separate plots for each energy technology
- **Stringency Breakdown**: Shows how economic parameters vary by climate policy stringency (C1-C8)
- **Key Metrics Analyzed**:
  - `efficiency_decimal` - Technology conversion efficiency 
  - `lifetime_years` - Expected technology lifespan
  - `om_cost_usd_per_mw_per_yr` - Operations & maintenance costs
  - `capital_cost_usd_per_mw` - Capital investment costs
  - `scenario_price` - Electricity/energy prices
  - `fuel_price` - Fuel costs (0 for renewables)

**Output**: 
- PNG files in `analysis/plots/` directory
- One plot per technology showing distribution across all metrics and stringency levels
- Publication-ready violin plots with scatter overlay

## Usage

```bash
cd analysis/
python generate_violin_plots.py
```

**Prerequisites**: 
- Complete pipeline must be run first to generate `6_final_AR6_viable_scenarios.csv`
- Script will automatically create `plots/` directory for outputs

## Understanding the Plots

**Violin Shape**: Shows the distribution density of values
**Scatter Points**: Individual data points (countries/regions) 
**Y-Axis**: Climate policy stringency levels (C1=most ambitious, C8=least ambitious)
**X-Axis**: 6 key economic metrics for investment analysis

**Interpretation**:
- **Wide violin**: High variability in that metric across regions/scenarios
- **Narrow violin**: Consistent values across regions/scenarios  
- **C1-C3 stringency**: Ambitious climate policies (Paris Agreement targets)
- **C4-C8 stringency**: Less ambitious policies (higher temperature outcomes)

## Economic Viability Focus

These plots show only **economically viable scenarios** - technologies that passed:
1. **Complete data requirements**: All essential parameters present (gap-filled OK)
2. **EBITDA viability**: Positive cash flow in at least one year of operation
3. **Realistic bounds**: Extreme values filtered out

This focuses the analysis on **realistic investment opportunities** rather than all theoretical scenarios.

---

**Note**: Old analysis scripts and plots using outdated data have been moved to `archive/` directory.