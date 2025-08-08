"""
Check if technology mapping is being applied correctly
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("CHECKING TECHNOLOGY MAPPING APPLICATION")
print("=" * 80)

# Load the data from step 3
df = pd.read_csv("3_final_AR6_target_schema.csv")
print(f"✅ Loaded step 3 data: {df.shape[0]:,} rows")

# Load technology mapping
mapping_df = pd.read_csv("technology_mapping.csv")
print(f"✅ Loaded technology mapping: {mapping_df.shape[0]:,} rows")

# Check current sector-technology combinations
print(f"\n📊 Current sector-technology combinations in step 3:")
current_combinations = df.groupby(['sector', 'technology']).size().reset_index(name='count')
for _, row in current_combinations.iterrows():
    print(f"   {row['sector']} - {row['technology']}: {row['count']:,} rows")

# Check what should be mapped
print(f"\n📋 Technology mapping entries:")
for _, row in mapping_df.iterrows():
    print(f"   {row['current_sector']} - {row['current_technology']} → {row['target_sector']} - {row['target_technology']} ({row['aggregation_group']})")

# Check if any current combinations are missing from mapping
print(f"\n🔍 Checking for unmapped combinations:")
mapped_combinations = set()
for _, row in mapping_df.iterrows():
    mapped_combinations.add((row['current_sector'], row['current_technology']))

unmapped = []
for _, row in current_combinations.iterrows():
    if (row['sector'], row['technology']) not in mapped_combinations:
        unmapped.append((row['sector'], row['technology']))

if unmapped:
    print(f"   ⚠️  Unmapped combinations found:")
    for sector, tech in unmapped:
        count = current_combinations[(current_combinations['sector'] == sector) & (current_combinations['technology'] == tech)]['count'].iloc[0]
        print(f"     • {sector} - {tech}: {count:,} rows")
else:
    print(f"   ✅ All combinations are mapped")

# Check specific technologies of interest
print(f"\n🎯 Checking specific technologies:")
techs_to_check = [
    ('Gas&Oil', 'Gas - Gases'),
    ('Gas&Oil', 'Gas - Liquids'), 
    ('Gas&Oil', 'Gas - Solids'),
    ('Gas&Oil', 'Oil - Gases'),
    ('Gas&Oil', 'Oil - Liquids'),
    ('Gas&Oil', 'Oil - Solids'),
    ('Power', 'Wind - Onshore'),
    ('Power', 'Wind - Offshore'),
    ('Renewables', 'Wind - Onshore'),
    ('Renewables', 'Wind - Offshore')
]

for sector, tech in techs_to_check:
    if (sector, tech) in mapped_combinations:
        mapping = mapping_df[(mapping_df['current_sector'] == sector) & (mapping_df['current_technology'] == tech)].iloc[0]
        print(f"   ✅ {sector} - {tech} → {mapping['target_sector']} - {mapping['target_technology']} ({mapping['aggregation_group']})")
    else:
        count = len(df[(df['sector'] == sector) & (df['technology'] == tech)])
        print(f"   ❌ {sector} - {tech}: {count:,} rows (NOT MAPPED)")

print(f"\n" + "=" * 80)
print("MAPPING CHECK COMPLETE")
print("=" * 80) 