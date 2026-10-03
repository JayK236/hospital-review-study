"""
Geocode all unique hospitals in the dataset using OpenStreetMap's free
Nominatim service (no API key required). This replaces manual address
lookups with a scalable, accurate solution.

Since every review at a given hospital shares the same location, we only
need to geocode ~50 unique hospital names ONCE, then merge the result
back onto every row in every features CSV.

Install dependency first:
    pip3 install geopy
"""

import pandas as pd
import glob
import time
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ── 1. Collect every unique hospital name across all batch xlsx files ────────
batch_files = glob.glob('batch_*.xlsx')
print(f"Found batch files: {batch_files}")

all_hospitals = set()
for f in batch_files:
    df = pd.read_excel(f)
    all_hospitals.update(df['name'].dropna().unique())

# Also pull from any features CSVs with a hospital_name column, in case
# some hospitals only exist there
for f in glob.glob('*_features*.csv'):
    try:
        df = pd.read_csv(f)
        if 'hospital_name' in df.columns:
            all_hospitals.update(df['hospital_name'].dropna().unique())
    except Exception:
        pass

all_hospitals = sorted(all_hospitals)
print(f"\nTotal unique hospitals to geocode: {len(all_hospitals)}")

# ── 2. Geocode each unique hospital name ──────────────────────────────────────
geolocator = Nominatim(user_agent="hospital_review_study_jay")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)  # Nominatim rate limit

results = []
for i, hospital in enumerate(all_hospitals):
    query = f"{hospital}, California, USA"
    try:
        location = geocode(query)
        if location:
            address = location.address
            lat = location.latitude
            lon = location.longitude
            # Try to pull zip code from the address string (last 5-digit token)
            zip_code = None
            for token in address.replace(',', ' ').split():
                if token.isdigit() and len(token) == 5:
                    zip_code = token
            results.append({
                'hospital_name': hospital,
                'full_address': address,
                'latitude': lat,
                'longitude': lon,
                'zip_code': zip_code,
            })
            print(f"[{i+1}/{len(all_hospitals)}] OK: {hospital[:50]:<50} -> {zip_code}")
        else:
            results.append({
                'hospital_name': hospital,
                'full_address': None,
                'latitude': None,
                'longitude': None,
                'zip_code': None,
            })
            print(f"[{i+1}/{len(all_hospitals)}] NOT FOUND: {hospital}")
    except Exception as e:
        results.append({
            'hospital_name': hospital,
            'full_address': None,
            'latitude': None,
            'longitude': None,
            'zip_code': None,
        })
        print(f"[{i+1}/{len(all_hospitals)}] ERROR: {hospital} - {e}")

# ── 3. Save the lookup table ───────────────────────────────────────────────────
lookup_df = pd.DataFrame(results)
lookup_df.to_csv('hospital_address_lookup.csv', index=False)
print(f"\nSaved lookup table: hospital_address_lookup.csv")
print(f"Successfully geocoded: {lookup_df['zip_code'].notna().sum()} / {len(lookup_df)}")

# ── 4. Show any hospitals that failed — these need manual lookup ─────────────
missing = lookup_df[lookup_df['zip_code'].isna()]
if len(missing) > 0:
    print(f"\n{len(missing)} hospitals need manual address lookup:")
    print(missing['hospital_name'].to_string(index=False))
    print("\nFor these, search the hospital name on Google Maps and add the")
    print("zip code manually to hospital_address_lookup.csv before merging.")

print("\nNext step: run merge_addresses.py to attach this lookup to your features CSVs.")
