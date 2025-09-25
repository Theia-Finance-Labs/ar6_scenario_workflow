# AR6 Climate Scenario Data Processing Pipeline

A comprehensive, production-ready pipeline for processing IPCC AR6 climate scenario data into analysis-ready datasets with economic viability analysis, gap-filling, and scenario statistics.

## Overview

This pipeline transforms raw AR6 climate scenario data through a 6-step process that includes data formatting, filtering, gap-filling, viability analysis, and comprehensive scenario summarization. The pipeline handles complex energy data structures, cost metrics, and economic viability calculations while maintaining data integrity through comprehensive validation.

## Current Active Pipeline (Steps 1-6)

### Core Pipeline Files (`pipeline/`)

**🚀 `combined_pipeline.py`** - Main unified pipeline (Steps 1-4)
- Combines formatting, filtering, unit conversion, and gap-filling
- Memory-optimized for large datasets (2M+ records)  
- Includes extreme value filtering and global geography aggregation
- **Status**: Active production pipeline

**⚡ `step4_gapfill_simple.py`** - Ultra-fast vectorized gap-filling
- Technology lookup-based gap-filling with fallback strategies
- Tracks which columns were gap-filled for transparency
- **Status**: Active, imported by combined_pipeline.py

**📊 `step5_complete_cases.py`** - Complete cases filtering  
- Filters to scenarios with all essential parameters (gap-filled = complete)
- Identifies viable investment scenarios
- **Status**: Active (Sep 15, 2024)

**🎯 `step6_scenario_tech_filter.py`** - Technology viability filtering
- EBITDA-based viability analysis 
- Filters to economically viable scenarios by technology
- **Status**: Active (Sep 15, 2024)

### Current Outputs (Generated Dataset Files)

**Pipeline Stage Outputs:**
```
1_intermediate_AR6_scenario_formatting_ISO3.csv (3.2GB) - Step 1 formatted data
2_final_AR6_filtered.csv (761MB) - Step 2 filtered & pivoted  
3_final_AR6_target_schema.csv (1.0GB) - Step 3 target schema
4_final_AR6_gapfilled.csv (1.3GB) - Step 4 gap-filled data
4_final_AR6_gapfilled_complete.csv (730MB) - Step 4 complete cases
5_final_AR6_complete_cases.csv (756MB) - Step 5 complete scenarios
6_final_AR6_viable_scenarios.csv (270MB) - Step 6 viable scenarios
```

**Analysis Outputs:**
- `scenario_summary_statistics.csv` (180KB) - Comprehensive scenario statistics with gap-filling analysis and EBITDA viability metrics

## Analysis & Visualization Files

### Main Directory Analysis Scripts

**📈 `create_scenario_summary.py`** - Scenario statistics generator
- Creates comprehensive scenario summaries with provider/scenario names
- Analyzes technologies included per scenario
- **Categorized gap-filling analysis**: Costs, Efficiency/Lifetime, Fuel Price, Scenario Price percentages
- **EBITDA viability analysis**: Technologies with positive EBITDA at least once in time series
- **Status**: Active (Sep 25, 2024)

**🔧 `create_technology_lookup.py`** - Technology lookup table creator
- Generates technology parameter lookup tables for gap-filling
- **Status**: Active

**📊 `generate_country_violin_plots.py`** - Country-level visualization
- Creates violin plots for country-specific analysis
- **Status**: Active

### Analysis Directory (`analysis/`)

**Stringency Analysis:**
- `create_proper_stringency_mapping.py` - AR6 stringency classification
- `fix_stringency_mapping.py` - Stringency data corrections
- `ar6_proper_stringency_mapping.csv` - Proper stringency mappings
- `scenario_stringency_mapping.csv` - Scenario stringency lookup

**Visualization Scripts:**
- `generate_violin_plots*.py` (5 variants) - Different violin plot analyses
- `analyze_non_gapfilled_efficiency.py` - Efficiency gap analysis

## Supporting Data Files

### Lookup Tables
```
ar6_variables_with_mapping.csv - Variable to technology mapping
r10_region_lookup.csv - Regional geography mappings  
technology_lookup_table.csv - Technology parameters for gap-filling
stringency_mapping_reference.csv - Stringency classification reference
```

### Intermediate Files (`intermediates/`)
- Backup copies of key intermediate processing stages

## Archive Directory - Stale/Deprecated Code ⚠️

The `archive/` directory contains **deprecated code** that is no longer used in the current pipeline:

### 🚫 **DEPRECATED** - Original Multi-Step Scripts
```
1_formatAR6.py - Replaced by combined_pipeline.py steps 1-2
2_filterAR6.py - Replaced by combined_pipeline.py steps 1-2  
3_finalizeAR6.py - Replaced by combined_pipeline.py step 3
4_aggregateTechnologies.py - Replaced by step4_gapfill_simple.py
```

### 🚫 **DEPRECATED** - Old Analysis Scripts
```
stringency_mapper.py - Replaced by analysis/create_proper_stringency_mapping.py
country_efficiency_analysis.py - Replaced by generate_country_violin_plots.py
gap_analysis.py - Analysis superseded by scenario summary
cost_diagnosis.py - Debug script, no longer needed
```

### 🚫 **STALE** Data Files in Archive
- Old CSV outputs from deprecated pipeline stages
- Outdated technology mappings
- Debug/test files

**⚠️ IMPORTANT**: Do not use files from `archive/` directory - they are kept for reference only.

## Pipeline Architecture & Data Flow

```
Raw AR6 Data (feather files)
    ↓
[Step 1-4] combined_pipeline.py → 4_final_AR6_gapfilled_complete.csv
    ↓
[Step 5] step5_complete_cases.py → 5_final_AR6_complete_cases.csv  
    ↓
[Step 6] step6_scenario_tech_filter.py → 6_final_AR6_viable_scenarios.csv
    ↓
[Analysis] create_scenario_summary.py → scenario_summary_statistics.csv
```

## Key Features

### 🛡️ **Advanced Gap-Filling System**
- **Categorized tracking**: Costs (75.6% avg), Efficiency/Lifetime (81.9% avg), Fuel Price (61.3% avg), Scenario Price (58.5% avg)
- Technology lookup with intelligent fallbacks
- Transparent gap-fill column tracking
- Extreme value filtering with configurable bounds

### 📊 **Economic Viability Analysis** 
- EBITDA-based technology viability assessment
- Time-series analysis (positive EBITDA at least once = viable)
- Average 55.7% of technologies show viability per scenario
- 1,229 scenarios have >50% viable technologies

### 🌍 **Global Aggregation**
- Automatic creation of "Global" geography entries
- Capacity-weighted averages for technical parameters
- Simple averages for price data

### 🔍 **Comprehensive Statistics**
The scenario summary provides:
- 1,923 unique scenario combinations across 111 providers  
- Technology coverage analysis (2-20 technologies per scenario)
- Gap-filling breakdown by parameter category
- EBITDA viability by technology within scenarios
- Geographic coverage and year ranges

## Usage

### Run Complete Pipeline
```bash
# Run full pipeline (Steps 1-6)
python pipeline/combined_pipeline.py  # Steps 1-4
python pipeline/step5_complete_cases.py  # Step 5
python pipeline/step6_scenario_tech_filter.py  # Step 6

# Generate scenario summary
python create_scenario_summary.py
```

### Individual Steps
```bash
# Just gap-filling and later steps
python pipeline/step4_gapfill_simple.py
python pipeline/step5_complete_cases.py  
python pipeline/step6_scenario_tech_filter.py
```

## Data Quality Metrics

### Latest Pipeline Run Results
```
Total scenarios processed: 1,923 unique combinations
Average gap-filling rate: 88.4% of records required gap-filling
Technology viability: 55.7% of technologies show positive EBITDA
Complete cases: 756MB final dataset with all essential parameters
Viable scenarios: 270MB economically feasible scenarios
```

### Gap-Filling Breakdown by Category
```
Efficiency/Lifetime parameters: 81.9% gap-filled (most common)
Cost parameters (Cap/OM): 75.6% gap-filled  
Fuel prices: 61.3% gap-filled
Scenario prices: 58.5% gap-filled
```

### EBITDA Viability Analysis
```
Scenarios with >50% viable technologies: 1,229 (63.9%)
Scenarios with 0% viable technologies: 655 (34.1%) 
Average positive EBITDA rate: 55.7% of technologies per scenario
```

## Dependencies

```python
pandas>=1.5.0
numpy>=1.20.0  
pyarrow>=10.0.0  # For feather format support
matplotlib>=3.5.0  # For visualization scripts
seaborn>=0.11.0  # For violin plots
```

## File Status Summary

### ✅ **ACTIVE FILES** (Current Pipeline)
- `pipeline/combined_pipeline.py` - Main pipeline
- `pipeline/step4_gapfill_simple.py` - Gap-filling engine  
- `pipeline/step5_complete_cases.py` - Complete cases filter
- `pipeline/step6_scenario_tech_filter.py` - Viability filter
- `create_scenario_summary.py` - Statistics generator
- All CSV outputs (1_* through 6_*, scenario_summary_statistics.csv)

### ⚠️ **DEPRECATED/STALE FILES** (Do Not Use)
- Everything in `archive/` directory
- Old analysis scripts replaced by current versions
- Debug and test files with obsolete logic

### 📊 **REFERENCE FILES** (Supporting Data)
- `ar6_variables_with_mapping.csv` - Variable mappings
- `technology_lookup_table.csv` - Gap-filling lookup
- `r10_region_lookup.csv` - Geographic mappings
- `stringency_mapping_reference.csv` - Climate stringency data

---

**Pipeline Version**: 6.0 (Complete Economic Viability Analysis)  
**Last Updated**: September 2024  
**Current Dataset**: 6_final_AR6_viable_scenarios.csv (270MB, economically viable scenarios)

For questions about specific components or to understand the gap-filling methodology, see the comprehensive scenario summary statistics in `scenario_summary_statistics.csv`.