import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
import glob
warnings.filterwarnings('ignore')

zuck   = pd.read_csv('extracted_features.csv');  zuck['hospital'] = 'Zuckerberg'
kaiser = pd.read_csv('kaiser_features.csv');      kaiser['hospital'] = 'Kaiser'
batch2 = pd.read_csv('batch2_features.csv')
batch2 = batch2.rename(columns={'hospital_name': 'hospital'})

ucsf_files = sorted(glob.glob('ucsf_features*.csv'))
if ucsf_files:
    ucsf = pd.read_csv(ucsf_files[-1]); ucsf['hospital'] = 'UCSF'
    print(f"Loaded UCSF from {ucsf_files[-1]}")
else:
    ucsf = None

def add_length(df, path):
    orig = pd.read_excel(path)
    orig = orig.dropna(subset=['review_text']).reset_index(drop=True)
    df = df.reset_index(drop=True)
    n = min(len(df), len(orig))
    df = df.iloc[:n].copy()
    df['review_length'] = orig['review_text'].str.len().values[:n]
    return df

kaiser = add_length(kaiser, 'Outscraper-20260402191145s52.xlsx')
batch2 = add_length(batch2, 'Outscraper-20260425025255s7e.xlsx')
if ucsf is not None:
    ucsf = add_length(ucsf, 'Outscraper-20260405020712s9f.xlsx')
zuck['review_length'] = np.nan

dfs = [zuck, kaiser, batch2]
if ucsf is not None:
    dfs.append(ucsf)
df = pd.concat(dfs, ignore_index=True)

df['gender_clean'] = df['gender'].apply(lambda x: x if str(x).lower() in ['male','female'] else None)

def clean_role(r):
    r = str(r).lower()
    if r == 'patient': return 'patient'
    if r == 'caretaker': return 'caretaker'
    if r == 'patient/caretaker': return 'patient_caretaker'
    return None
df['role_clean'] = df['role'].apply(clean_role)

def hosp_type(h):
    h = str(h)
    if 'Kaiser' in h: return 'private_integrated'
    if 'CPMC' in h: return 'private_nonprofit'
    if 'UCSF' in h: return 'public_academic'
    if 'Zuckerberg' in h or 'ZSFG' in h: return 'public_safety_net'
    if 'Family' in h: return 'community_health'
    return 'other'
df['hospital_type'] = df['hospital'].apply(hosp_type)

df_model = df.dropna(subset=['review_rating','gender_clean','role_clean','review_length']).copy()
print(f"\nModel dataset: {len(df_model)} reviews")
print(df_model['hospital'].value_counts().to_string())

print("\n" + "="*65)
print("PER-FACILITY: rating ~ gender + role + review_length")
print("="*65)
for hosp in sorted(df_model['hospital'].value_counts()[df_model['hospital'].value_counts()>=20].index):
    sub = df_model[df_model['hospital']==hosp].copy()
    if sub['gender_clean'].nunique()<2 or sub['role_clean'].nunique()<2: continue
    try:
        m = smf.ols('review_rating ~ C(gender_clean, Treatment("female")) + C(role_clean, Treatment("caretaker")) + review_length', data=sub).fit()
        print(f"\n── {hosp[:50]} (n={len(sub)}, R²={m.rsquared:.3f}) ──")
        for name,coef,pval in zip(m.params.index,m.params.values,m.pvalues.values):
            if name=='Intercept': continue
            stars='***' if pval<0.001 else '**' if pval<0.01 else '*' if pval<0.05 else ''
            label=name.replace('C(gender_clean, Treatment("female"))[T.male]','gender: male vs female').replace('C(role_clean, Treatment("caretaker"))[T.patient]','role: patient vs caretaker').replace('review_length','review length')
            print(f"  {label:<45} coef={coef:+.3f}  p={pval:.3f} {stars}")
    except Exception as e:
        print(f"\n── {hosp}: skipped ({e})")

print("\n" + "="*65)
print("POOLED: rating ~ gender + role + review_length + hospital_type")
print("="*65)
try:
    pm = smf.ols('review_rating ~ C(gender_clean, Treatment("female")) + C(role_clean, Treatment("caretaker")) + review_length + C(hospital_type, Treatment("public_safety_net"))', data=df_model).fit()
    print(f"\nR²={pm.rsquared:.3f}  N={len(df_model)}")
    for name,coef,pval in zip(pm.params.index,pm.params.values,pm.pvalues.values):
        stars='***' if pval<0.001 else '**' if pval<0.01 else '*' if pval<0.05 else ''
        label=name.replace('C(gender_clean, Treatment("female"))[T.male]','gender: male (vs female)').replace('C(role_clean, Treatment("caretaker"))[T.patient]','role: patient (vs caretaker)').replace('C(hospital_type, Treatment("public_safety_net"))[T.','hospital: ').replace(']',' (vs public safety-net)').replace('review_length','review length')
        print(f"  {label:<55} {coef:>+7.3f}  p={pval:.3f} {stars}")
except Exception as e:
    print(f"Pooled failed: {e}")

print("\n" + "="*65)
print("INTERACTION: rating ~ gender * role + review_length + hospital_type")
print("Tests if gender effect differs between patients and caretakers.")
print("="*65)
try:
    im = smf.ols('review_rating ~ C(gender_clean, Treatment("female")) * C(role_clean, Treatment("caretaker")) + review_length + C(hospital_type, Treatment("public_safety_net"))', data=df_model).fit()
    print(f"\nR²={im.rsquared:.3f}  N={len(df_model)}")
    for name,coef,pval in zip(im.params.index,im.params.values,im.pvalues.values):
        stars='***' if pval<0.001 else '**' if pval<0.01 else '*' if pval<0.05 else ''
        label=name.replace('C(gender_clean, Treatment("female"))[T.male]','gender: male').replace('C(role_clean, Treatment("caretaker"))[T.patient]','role: patient').replace('C(hospital_type, Treatment("public_safety_net"))[T.','hospital: ').replace(']',' (vs public safety-net)').replace('review_length','review length').replace(':','  x  ')
        print(f"  {label:<60} {coef:>+7.3f}  p={pval:.3f} {stars}")
except Exception as e:
    print(f"Interaction model failed: {e}")

print("\n* p<0.05  ** p<0.01  *** p<0.001")
