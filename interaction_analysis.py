import pandas as pd

#load batch hospitals 

batch2 = pd.read_csv('batch2_features.csv')
batch2 = batch2.rename(columns={'hospital_name': 'hospital'})

# Load all three hospitals
zuck = pd.read_csv('extracted_features.csv')
zuck['hospital'] = 'Zuckerberg'

kaiser = pd.read_csv('kaiser_features.csv')
kaiser['hospital'] = 'Kaiser'

ucsf = pd.read_csv('ucsf_features.csv')
ucsf['hospital'] = 'UCSF'

df = pd.concat([zuck, kaiser, ucsf, batch2], ignore_index=True)

# Clean up: keep only clear gender and meaningful role categories
df_clean = df[
    df['gender'].isin(['male', 'female']) &
    df['role'].isin(['patient', 'caretaker'])
].copy()

print("=== SAMPLE SIZES: ROLE x GENDER x HOSPITAL ===")
print(df_clean.groupby(['hospital', 'role', 'gender']).size().to_string())

print("\n=== AVERAGE RATING: ROLE x GENDER (ALL HOSPITALS) ===")
cross = df_clean.groupby(['role', 'gender'])['review_rating'].agg(['mean', 'count']).round(2)
cross.columns = ['avg_rating', 'count']
print(cross.to_string())

print("\n=== AVERAGE RATING: ROLE x GENDER x HOSPITAL ===")
cross_hosp = df_clean.groupby(['hospital', 'role', 'gender'])['review_rating'].agg(['mean', 'count']).round(2)
cross_hosp.columns = ['avg_rating', 'count']
print(cross_hosp.to_string())

print("\n=== GENDER GAP WITHIN EACH ROLE (ALL HOSPITALS) ===")
for role in ['patient', 'caretaker']:
    role_df = df_clean[df_clean['role'] == role]
    male_avg = role_df[role_df['gender'] == 'male']['review_rating'].mean()
    female_avg = role_df[role_df['gender'] == 'female']['review_rating'].mean()
    gap = female_avg - male_avg
    print(f"{role.capitalize()}s: Male={male_avg:.2f}, Female={female_avg:.2f}, Gap={gap:+.2f}")

print("\n=== GENDER GAP WITHIN EACH ROLE BY HOSPITAL ===")
for hospital in ['Zuckerberg', 'UCSF', 'Kaiser']:
    print(f"\n{hospital}:")
    hosp_df = df_clean[df_clean['hospital'] == hospital]
    for role in ['patient', 'caretaker']:
        role_df = hosp_df[hosp_df['role'] == role]
        male = role_df[role_df['gender'] == 'male']['review_rating']
        female = role_df[role_df['gender'] == 'female']['review_rating']
        if len(male) > 0 and len(female) > 0:
            gap = female.mean() - male.mean()
            print(f"  {role.capitalize()}s: Male={male.mean():.2f} (n={len(male)}), Female={female.mean():.2f} (n={len(female)}), Gap={gap:+.2f}")
        else:
            print(f"  {role.capitalize()}s: insufficient data")
