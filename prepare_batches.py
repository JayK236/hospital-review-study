import pandas as pd
import os

# ── Load and combine the three Outscraper files ───────────────────────────────
files = [
    'Outscraper-20260518015430m12_records_1-239.xlsx',
    'Outscraper-20260518015439m89_records_240-478.xlsx',
    'Outscraper-20260518015441m6c_records_479-716.xlsx',
]

print("Loading files...")
dfs = []
for f in files:
    if os.path.exists(f):
        df = pd.read_excel(f)
        dfs.append(df)
        print(f"  {f}: {len(df)} rows")
    else:
        print(f"  MISSING: {f}")

combined = pd.concat(dfs, ignore_index=True)
combined = combined[combined['review_text'].notna()].copy()
print(f"\nCombined usable reviews: {len(combined):,}")

# ── Define city batches ───────────────────────────────────────────────────────
batches = {
    'sf_new': [
        'San Francisco VA Medical Center',
        "UCSF Benioff Children's Hospital - San Francisco",
        'Kindred Hospital San Francisco Bay Area',
        'Kaiser Permanente San Francisco Mission Bay Medical Offices',
        'Kaiser Permanente 2238 Geary Medical Offices',
        'Kaiser Permanente South San Francisco Medical Center',
        'Mills-Peninsula Medical Center - Burlingame Campus',
    ],
    'sacramento': [
        'UC Davis Medical Center',
        'UC Davis Medical Center Emergency Department',
        'Kaiser Permanente Sacramento Medical Center',
        'Sutter Medical Center, Sacramento',
        'Sutter Medical Center Emergency Department',
        'Mercy San Juan Medical Center',
        'Dignity Health - Mercy General Hospital',
        'Dignity Health - Methodist Hospital of Sacramento',
        'Dignity Health - Mercy Hospital of Folsom',
        'Dignity Health - Woodland Memorial Hospital',
        'Sacramento VA Medical Center - VA Northern California Health Care System',
        'Vibra Hospital of Sacramento',
        'Sierra Vista Hospital',
        'Sutter Center for Psychiatry',
        'Kaiser Permanente South Sacramento Medical Center',
        'Kaiser Permanente Downtown Commons Medical Offices',
    ],
    'oakland_eastbay': [
        'Highland Hospital',
        'Highland Hospital Emergency Department',
        'Kaiser Permanente Oakland Medical Center',
        'Kaiser Permanente Oakland Broadway Medical Offices',
        'Kaiser Permanente Oakland Fabiola Medical Offices',
        'Eden Medical Center',
        'Alta Bates Summit Medical Center | Alta Bates Campus',
        'Alta Bates Summit Medical Center | Summit Campus',
        "UCSF Benioff Children's Hospital - Oakland",
        'San Leandro Hospital',
        'Alameda Hospital',
        'Kaiser Permanente San Leandro Medical Center',
        'AHMC Seton Medical Center',
        'St. Rose Hospital',
    ],
    'san_jose_southbay': [
        'Kaiser Permanente San Jose Medical Center',
        'Kaiser Permanente Santa Clara Medical Center',
        'Good Samaritan Hospital',
        "O'Connor Hospital",
        "O'Connor Hospital Emergency Room",
        'Regional Medical Center Emergency Room',
        'St. Louise Regional Hospital',
        'San Jose Behavioral Health',
    ],
    'community': [
        'Action Urgent Care',
        'One Community Health - Midtown Health Center',
        'UCSF Health-GoHealth Urgent Care',
    ],
}

# ── Save each batch as a separate Excel file ──────────────────────────────────
print("\nSaving city batches:")
print("="*55)
for batch_name, hospital_list in batches.items():
    batch_df = combined[combined['name'].isin(hospital_list)].copy()
    batch_df = batch_df.reset_index(drop=True)
    if len(batch_df) == 0:
        print(f"  {batch_name}: NO MATCHING HOSPITALS FOUND")
        continue
    out_file = f'batch_{batch_name}.xlsx'
    batch_df.to_excel(out_file, index=False)
    cost = len(batch_df) * 0.002
    print(f"  {batch_name}: {len(batch_df):,} reviews → {out_file} (~${cost:.0f} API cost)")
    # Show hospital breakdown
    for name, count in batch_df['name'].value_counts().items():
        print(f"    {name[:50]:<50} {count:>5}")

print("\nDone! Run hospital_analysis.py on each batch file separately.")
print("Suggested order: sf_new → sacramento → oakland_eastbay → san_jose_southbay → community")
print("Top up API credits to $50+ before starting.")
