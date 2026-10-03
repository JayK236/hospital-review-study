"""
Merge the hospital_address_lookup.csv (built by geocode_hospitals.py) onto
every features CSV in the folder. Run this AFTER geocode_hospitals.py
completes successfully.
"""

import pandas as pd
import glob

# ── 1. Load the address lookup table ──────────────────────────────────────────
lookup = pd.read_csv('hospital_address_lookup.csv')
print(f"Loaded lookup table: {len(lookup)} hospitals")
print(f"  With valid zip codes: {lookup['zip_code'].notna().sum()}")

lookup_dict = lookup.set_index('hospital_name')[['full_address', 'zip_code', 'latitude', 'longitude']].to_dict('index')

# ── 2. Merge onto every features CSV ──────────────────────────────────────────
features_files = glob.glob('*_features*.csv')
print(f"\nFound {len(features_files)} features files to update:")

for f in features_files:
    df = pd.read_csv(f)

    # Determine which column holds the hospital name in this file
    name_col = None
    if 'hospital_name' in df.columns:
        name_col = 'hospital_name'
    elif 'hospital' in df.columns:
        name_col = 'hospital'

    if name_col is None:
        print(f"  {f}: SKIPPED — no hospital name column found")
        continue

    def lookup_field(name, field):
        entry = lookup_dict.get(name)
        return entry[field] if entry else None

    df['full_address'] = df[name_col].apply(lambda n: lookup_field(n, 'full_address'))
    df['zip_code'] = df[name_col].apply(lambda n: lookup_field(n, 'zip_code'))
    df['latitude'] = df[name_col].apply(lambda n: lookup_field(n, 'latitude'))
    df['longitude'] = df[name_col].apply(lambda n: lookup_field(n, 'longitude'))

    matched = df['zip_code'].notna().sum()
    df.to_csv(f, index=False)
    print(f"  {f}: updated — {matched}/{len(df)} rows matched to a zip code")

print("\nDone! Every features CSV now has full_address, zip_code, latitude, longitude columns.")
print("Next step: use zip_code to join with US Census ACS demographic data for race analysis.")
