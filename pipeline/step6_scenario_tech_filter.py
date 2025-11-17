#!/usr/bin/env python3
"""
Step 6: Filter Scenarios by Technology EBITDA Compliance

This script filters scenarios to only include those where ALL Power sector technologies
have at least one year where the EBITDA_check condition is TRUE. This ensures that
every technology in the selected scenarios has at least some economically viable data.

Input: 5_final_AR6_complete_cases.csv
Output: 6_final_AR6_viable_scenarios.csv

Author: Claude Code
Date: 2025-09-15
"""

import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('step6_scenario_tech_filter.log'),
        logging.StreamHandler()
    ]
)

def get_power_technologies(df):
    """
    Get all unique Power sector technologies in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        list: List of Power sector technology names
    """
    power_techs = df[df['sector'] == 'Power']['technology'].unique()
    return sorted(power_techs.tolist())

def check_scenario_tech_compliance(df, scenario, power_technologies):
    """
    Check if a scenario has at least one passing year for all Power technologies.
    
    Args:
        df (pd.DataFrame): Full dataframe
        scenario (str): Scenario name to check
        power_technologies (list): List of all Power technologies
        
    Returns:
        dict: Compliance status with details
    """
    scenario_data = df[df['scenario'] == scenario]
    scenario_power = scenario_data[scenario_data['sector'] == 'Power']
    
    # Get technologies present in this scenario
    scenario_techs = set(scenario_power['technology'].unique())
    
    # Check each technology for at least one passing year
    tech_compliance = {}
    
    for tech in power_technologies:
        tech_data = scenario_power[scenario_power['technology'] == tech]
        
        if len(tech_data) == 0:
            # Technology not present in scenario
            tech_compliance[tech] = {
                'present': False,
                'has_passing_year': False,
                'total_rows': 0,
                'passing_rows': 0
            }
        else:
            # Technology is present, check if any year passes
            passing_rows = tech_data[tech_data['EBITDA_check'] == True]
            has_passing = len(passing_rows) > 0
            
            tech_compliance[tech] = {
                'present': True,
                'has_passing_year': has_passing,
                'total_rows': len(tech_data),
                'passing_rows': len(passing_rows)
            }
    
    # Check if ALL present technologies have at least one passing year
    present_techs = [tech for tech, data in tech_compliance.items() if data['present']]
    compliant_techs = [tech for tech, data in tech_compliance.items() 
                      if data['present'] and data['has_passing_year']]
    
    all_techs_compliant = len(present_techs) == len(compliant_techs) and len(present_techs) > 0
    
    return {
        'scenario': scenario,
        'all_techs_compliant': all_techs_compliant,
        'present_techs': present_techs,
        'compliant_techs': compliant_techs,
        'tech_compliance': tech_compliance,
        'present_tech_count': len(present_techs),
        'compliant_tech_count': len(compliant_techs)
    }

def flag_viable_scenarios(input_file, output_file):
    """
    Main function to flag scenarios by technology EBITDA compliance instead of filtering.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    """
    logging.info(f"Starting scenario technology flagging...")
    logging.info(f"Input file: {input_file}")
    logging.info(f"Output file: {output_file}")
    
    # Read input data
    logging.info("Reading input data...")
    df = pd.read_csv(input_file, low_memory=False)
    logging.info(f"Input dataset shape: {df.shape}")
    
    # Get all Power sector technologies
    power_technologies = get_power_technologies(df)
    logging.info(f"Power sector technologies: {len(power_technologies)}")
    for tech in power_technologies:
        logging.info(f"  - {tech}")
    
    # Get all unique scenarios
    scenarios = df['scenario'].unique()
    logging.info(f"Total scenarios to evaluate: {len(scenarios):,}")
    
    # Check each scenario's compliance
    logging.info("\n=== EVALUATING SCENARIO COMPLIANCE ===")
    viable_scenarios = []
    scenario_compliance_map = {}
    scenario_analysis = []
    
    for i, scenario in enumerate(scenarios):
        if (i + 1) % 100 == 0:
            logging.info(f"Processed {i + 1}/{len(scenarios)} scenarios...")
        
        compliance_result = check_scenario_tech_compliance(df, scenario, power_technologies)
        scenario_analysis.append(compliance_result)
        
        # Store compliance status for this scenario
        scenario_compliance_map[scenario] = compliance_result['all_techs_compliant']
        
        if compliance_result['all_techs_compliant']:
            viable_scenarios.append(scenario)
    
    logging.info(f"\n=== FLAGGING RESULTS ===")
    logging.info(f"Scenarios evaluated: {len(scenarios):,}")
    logging.info(f"Viable scenarios found: {len(viable_scenarios):,}")
    logging.info(f"Non-viable scenarios: {len(scenarios) - len(viable_scenarios):,}")
    logging.info(f"Non-viable rate: {(len(scenarios) - len(viable_scenarios))/len(scenarios)*100:.1f}%")
    
    # Add scenario_viable flag to all records instead of filtering
    df['scenario_viable'] = df['scenario'].map(scenario_compliance_map)
    
    logging.info(f"Original dataset rows: {len(df):,}")
    logging.info(f"Flagged viable rows: {df['scenario_viable'].sum():,}")
    logging.info(f"Flagged non-viable rows: {(~df['scenario_viable']).sum():,}")
    logging.info(f"Viable data percentage: {df['scenario_viable'].sum()/len(df)*100:.1f}%")
    
    # Analyze non-viable scenarios
    logging.info(f"\n=== NON-VIABLE SCENARIO ANALYSIS ===")
    non_viable_analysis = [result for result in scenario_analysis if not result['all_techs_compliant']]
    
    if len(non_viable_analysis) > 0:
        # Count how many scenarios fail due to each technology
        tech_failure_counts = {}
        for result in non_viable_analysis:
            for tech in power_technologies:
                if tech in result['tech_compliance']:
                    tech_data = result['tech_compliance'][tech]
                    if tech_data['present'] and not tech_data['has_passing_year']:
                        tech_failure_counts[tech] = tech_failure_counts.get(tech, 0) + 1
        
        logging.info("Technologies causing scenario non-viability (Top 10):")
        sorted_failures = sorted(tech_failure_counts.items(), key=lambda x: x[1], reverse=True)
        for tech, count in sorted_failures[:10]:
            pct = count / len(non_viable_analysis) * 100
            logging.info(f"  {tech}: {count:,} scenarios ({pct:.1f}%)")
    
    # Analyze viable scenarios
    logging.info(f"\n=== VIABLE SCENARIO ANALYSIS ===")
    viable_analysis = [result for result in scenario_analysis if result['all_techs_compliant']]
    
    if len(viable_analysis) > 0:
        logging.info("Sample viable scenarios:")
        for result in viable_analysis[:5]:
            logging.info(f"  {result['scenario']}: {result['compliant_tech_count']}/{result['present_tech_count']} technologies")
        
        # Technology coverage in viable scenarios
        tech_coverage = {}
        for tech in power_technologies:
            present_count = sum(1 for result in viable_analysis 
                              if tech in result['tech_compliance'] and result['tech_compliance'][tech]['present'])
            tech_coverage[tech] = present_count
        
        logging.info("\nTechnology coverage in viable scenarios:")
        for tech, count in sorted(tech_coverage.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(viable_scenarios) * 100
            logging.info(f"  {tech}: {count:,}/{len(viable_scenarios):,} scenarios ({pct:.1f}%)")
    
    # Save ALL data with viability flag instead of filtering
    logging.info(f"\nSaving all scenarios with viability flag to: {output_file}")
    df.to_csv(output_file, index=False)
    
    # Final summary
    logging.info(f"\n=== FINAL SUMMARY ===")
    logging.info(f"Total scenarios: {len(scenarios):,}")
    logging.info(f"Viable scenarios: {len(viable_scenarios):,}")
    logging.info(f"Non-viable scenarios: {len(scenarios) - len(viable_scenarios):,}")
    logging.info(f"Total rows in output: {len(df):,}")
    logging.info(f"Rows flagged as viable: {df['scenario_viable'].sum():,}")
    logging.info(f"Power technologies evaluated: {len(power_technologies):,}")
    logging.info("Scenario flagging completed successfully!")
    
    return df, viable_scenarios

def step6_scenario_tech_filter():
    """
    Wrapper function for importable use from combined pipeline.
    Calls flag_viable_scenarios with default file paths.
    """
    input_file = "5_final_AR6_complete_cases.csv"
    output_file = "6_final_AR6_viable_scenarios.csv"
    
    if not Path(input_file).exists():
        logging.error(f"Input file not found: {input_file}")
        return
    
    result_df, viable_scenarios = flag_viable_scenarios(input_file, output_file)
    return result_df

def main():
    """Main execution function."""
    # File paths
    input_file = "5_final_AR6_complete_cases.csv"
    output_file = "6_final_AR6_viable_scenarios.csv"
    
    # Check input file exists
    if not Path(input_file).exists():
        logging.error(f"Input file not found: {input_file}")
        return
    
    # Run flagging
    try:
        result_df, viable_scenarios = flag_viable_scenarios(input_file, output_file)
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"SCENARIO TECHNOLOGY FLAGGING SUMMARY")
        print(f"{'='*60}")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Total scenarios: {result_df['scenario'].nunique():,}")
        print(f"Viable scenarios: {len(viable_scenarios):,}")
        print(f"Output dataset rows: {len(result_df):,}")
        print(f"Viable data rows: {result_df['scenario_viable'].sum():,}")
        print(f"Log file: step6_scenario_tech_filter.log")
        
    except Exception as e:
        logging.error(f"Error during flagging: {str(e)}")
        raise

if __name__ == "__main__":
    main()