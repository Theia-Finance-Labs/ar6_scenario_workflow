# AR6 Climate Scenario Economic Viability Pipeline

A comprehensive pipeline that transforms raw IPCC AR6 climate scenario data into economically viable investment scenarios. The pipeline processes 2+ million records across 1,900+ scenarios from 111+ climate models, applying intelligent gap-filling, economic viability analysis, and comprehensive filtering to identify realistic clean energy investment opportunities.

## What This Pipeline Does

This pipeline takes raw AR6 climate scenario data and answers the key question: **"Which clean energy technologies are economically viable investments in different climate scenarios?"**

### The Process in Simple Terms:

1. **Data Formatting & Integration** - Converts complex climate model outputs into usable energy technology datasets
2. **Intelligent Gap-Filling** - Uses technology lookup tables to fill missing cost, efficiency, and price data
3. **Economic Viability Analysis** - Calculates EBITDA to determine which technologies are profitable investments
4. **Scenario Filtering** - Keeps only scenarios with complete data and economically viable technologies
5. **Comprehensive Statistics** - Generates detailed summaries of gap-filling and viability across all scenarios

### Key Outputs:
- **Economic Viability**: 55.7% of technologies show positive EBITDA (profitable)
- **Data Completeness**: 88.4% of missing data successfully gap-filled
- **Investment Focus**: Final dataset contains only economically viable scenarios (270MB from 3.2GB original)

## Pipeline Architecture

```
Raw AR6 Climate Data (3+ GB)
         ↓
[COMBINED PIPELINE] - Steps 1-4 orchestrated in one script
    ├─ Step 1: Format & melt wide data to long format
    ├─ Step 2: Filter, pivot, and standardize units  
    ├─ Step 3: Apply target schema and price mapping
    └─ Step 4: Intelligent gap-filling using technology lookup
         ↓
Complete Gap-Filled Dataset (730 MB)
         ↓
[Step 5] Complete Cases Filter → Only scenarios with all essential data
         ↓  
Complete Scenarios Dataset (756 MB)
         ↓
[Step 6] Economic Viability Filter → Only profitable technologies  
         ↓
Viable Investment Scenarios (270 MB)
         ↓
[Analysis] Comprehensive Statistics → Gap-filling & viability metrics
```

## The Combined Pipeline Explained

The **`combined_pipeline.py`** is the heart of this system - a single orchestrator script that runs Steps 1-4 in sequence with memory optimization between steps. It combines data processing AND gap-filling into one optimized workflow. The subsequent filtering and analysis steps (5-6) remain separate because they serve different analytical purposes and may be run independently. Here's what the combined pipeline orchestrates:

### Step 1: Data Formatting & Cost Integration
- **Input**: Raw AR6 feather files with wide-format data (columns for each year)
- **Process**: 
  - Melts wide format (columns = years) to long format (year as row)
  - Integrates cost metrics (OM cost, capital cost, efficiency) from variable mappings
  - Processes energy price data (primary, secondary, carbon pricing)
  - Filters to target energy sectors and technologies
- **Output**: `1_intermediate_AR6_scenario_formatting_ISO3.csv` (3.2GB)

### Step 2: Filtering, Pivoting & Unit Standardization  
- **Input**: Long-format melted data
- **Process**:
  - **Smart Unit Conversions**: GW→MW (*1000), EJ/yr→MWh/yr (*2.78×10¹¹), etc.
  - **Data Pivoting**: Converts metrics (Capacity, Energy, etc.) from rows to columns
  - **Cost Integration**: Merges OM costs, capital costs, efficiency data
  - **Renewable Efficiency**: Auto-sets renewables to 100% efficiency (no fuel conversion losses)
- **Output**: `2_final_AR6_filtered.csv` (761MB)

### Step 3: Target Schema & Price Mapping
- **Input**: Filtered and pivoted data  
- **Process**:
  - **Geography Mapping**: Country codes to ISO2 format, regional aggregations
  - **Price Logic**: Power/Renewables use electricity prices, fossil fuels use primary energy prices
  - **Pathway Calculation**: Determines energy (MWh/yr) vs capacity (MW) reporting
  - **Global Aggregation**: Creates worldwide totals by summing across regions
  - **Extreme Value Filtering**: Removes unrealistic values (efficiency >100%, negative costs)
- **Output**: `3_final_AR6_target_schema.csv` (1.0GB)

### Step 4: Intelligent Gap-Filling  
- **Input**: Target schema with missing data
- **Process**:
  - **Technology Lookup**: Uses `technology_lookup_table.csv` for missing parameters
  - **Multi-Strategy Filling**: 
    1. Direct technology match
    2. Fallback to technology families (e.g., "CoalCap" → "CoalCap - w/o CCS")  
    3. Global averages by technology type
    4. Stringency-based filling (climate policy scenarios)
  - **Transparent Tracking**: Records exactly which columns were gap-filled for each row
- **Output**: `4_final_AR6_gapfilled_complete.csv` (730MB)

**Why Steps 1-4 Are Combined:**
These steps form a complete data processing pipeline that must run in sequence. The orchestrator optimizes memory usage between steps and eliminates redundant file I/O operations.

### Memory Optimization Features:
- **Progressive Memory Release**: Deletes intermediate DataFrames after each step
- **Vectorized Operations**: Uses pandas vectorization instead of slow row-by-row operations
- **Chunked Processing**: Handles 2M+ row datasets efficiently

## Individual Pipeline Steps (Run After Combined Pipeline)

### Step 5: Complete Cases Filter (`step5_complete_cases.py`)
- **Purpose**: Keep only scenarios with all essential parameters present
- **Logic**: Gap-filled data counts as "complete" - focuses on investment-ready scenarios
- **Essential Parameters**: efficiency, lifetime, costs, capacity, energy data
- **Output**: `5_final_AR6_complete_cases.csv` (756MB)

### Step 6: Economic Viability Filter (`step6_scenario_tech_filter.py`)  
- **Purpose**: Filter to economically profitable technologies using EBITDA analysis
- **EBITDA Calculation**: `(Revenue - OM Costs) / Capacity` where Revenue = Energy × Price
- **Viability Logic**: Technology is viable if EBITDA > 0 in at least one year of time series
- **Result**: Only scenarios with profitable investment opportunities
- **Output**: `6_final_AR6_viable_scenarios.csv` (270MB)

## Analysis & Statistics

### Scenario Summary (`create_scenario_summary.py`)
Generates comprehensive statistics showing:

**Gap-Filling Analysis by Category:**
- **Costs** (Capital + OM): 75.6% average gap-filled
- **Efficiency/Lifetime**: 81.9% average gap-filled  
- **Fuel Prices**: 61.3% average gap-filled
- **Scenario Prices**: 58.5% average gap-filled

**Economic Viability Metrics:**
- **55.7%** of technologies show positive EBITDA per scenario (profitable)
- **1,229 scenarios** have >50% of technologies profitable
- **655 scenarios** have 0% profitable technologies (not investment-worthy)

**Coverage Statistics:**  
- **1,923 unique scenarios** across **111 scenario providers**
- **15.2 technologies per scenario** on average (range: 2-20)
- **Technology mix**: Power sector dominates, followed by renewables

### Visualization Tools
- `generate_country_violin_plots.py` - Country-level investment analysis
- `analysis/generate_violin_plots*.py` - Various economic distribution plots

## Key Supporting Files

### Lookup & Reference Data
```
ar6_variables_with_mapping.csv - Maps AR6 variables to technology categories
technology_lookup_table.csv - Technology parameters for gap-filling  
r10_region_lookup.csv - Regional geography mappings
stringency_mapping_reference.csv - Climate policy stringency classifications
```

### Processing Controls
```
create_technology_lookup.py - Generates/updates gap-filling lookup tables
```

## Quick Start

### Run Complete Pipeline
```bash
# Full pipeline (takes ~30-45 minutes for full dataset)
python pipeline/combined_pipeline.py    # Steps 1-4: Data processing, formatting & gap-filling
python pipeline/step5_complete_cases.py # Step 5: Complete cases only  
python pipeline/step6_scenario_tech_filter.py # Step 6: Viable investments only

# Generate comprehensive statistics
python create_scenario_summary.py       # Analysis summary
```

### Key Outputs to Check
```bash
# Final viable scenarios (your main result)
ls -lh 6_final_AR6_viable_scenarios.csv   # ~270MB economically viable scenarios

# Comprehensive statistics  
ls -lh scenario_summary_statistics.csv    # Detailed gap-filling & viability analysis
```

## Data Quality & Results

### Pipeline Effectiveness
```
Starting data: 3.2GB raw climate scenarios (millions of records)
After gap-filling: 730MB complete scenarios (88.4% data recovery)
After viability filter: 270MB profitable scenarios (realistic investments)
Processing time: ~45 minutes for full dataset
```

### Gap-Filling Success Rates
```
Overall success: 88.4% of missing data successfully filled
Efficiency/Lifetime: 81.9% (highest priority parameters)
Costs (Capital/OM): 75.6% (critical for viability) 
Prices (Fuel/Scenario): ~60% (market-dependent data)
```

### Economic Viability Results  
```
Profitable technologies: 55.7% show positive EBITDA
Investment-ready scenarios: 1,229 out of 1,923 (63.9%)
Technology coverage: 2-20 technologies per scenario
Geographic coverage: Global with country-level detail
```

## Understanding the Results

**What constitutes a "viable scenario"?**
- All essential technical parameters present (gap-filled counts as present)
- At least some technologies show positive EBITDA (profitable operations) 
- Realistic parameter values (extreme outliers filtered out)

**What does the gap-filling tell us?**
- Which climate models provide complete vs incomplete data
- Where the pipeline adds the most value (efficiency/lifetime data most often missing)
- How much we can trust each scenario (less gap-filling = more reliable)

**How to use the final dataset:**
- `6_final_AR6_viable_scenarios.csv` = Investment-ready clean energy scenarios
- `scenario_summary_statistics.csv` = Meta-analysis of data quality and economic potential
- Filter by geography, technology, or stringency level for specific investment analysis

---

**Pipeline Version**: 6.0 (Complete Economic Viability Analysis)  
**Dataset Coverage**: 111+ climate models, 1,900+ scenarios, 20+ energy technologies  
**Final Output**: 270MB economically viable clean energy investment scenarios

*For technical details about gap-filling methodology or data quality, see `scenario_summary_statistics.csv`*