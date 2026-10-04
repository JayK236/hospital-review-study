"""
Research Synthesis Agent — Hospital Review Study
Run this after each new batch is processed to get an updated findings report
and refresh the GitHub Pages dashboard data.

Usage:
    python synthesis_agent.py

Output:
    synthesis_report_latest.txt  — current findings narrative
    synthesis_history.jsonl      — append-only log of all runs
    data.json                    — dashboard data file (commit + push to update site)
"""

import anthropic
import pandas as pd
import numpy as np
import glob
import os
import json
import datetime
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load all processed features CSVs ───────────────────────────────────────

def load_all_features():
    files = glob.glob('*_features*.csv')
    if not files:
        raise FileNotFoundError("No *_features*.csv files found. Run hospital_analysis.py first.")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Infer market from filename
        if 'sacramento' in f.lower():
            df['market'] = 'Sacramento'
        elif 'sf' in f.lower() or 'san_francisco' in f.lower() or 'batch' in f.lower():
            df['market'] = 'San Francisco'
        elif 'oakland' in f.lower():
            df['market'] = 'Oakland'
        elif 'sanjose' in f.lower() or 'san_jose' in f.lower():
            df['market'] = 'San Jose'
        else:
            df['market'] = 'Unknown'
        df['source_file'] = f
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined):,} reviews from {len(files)} files")
    return combined

# ── 2. Compute summary statistics ─────────────────────────────────────────────

def compute_stats(df):
    stats = {}

    # Basic counts
    stats['total_reviews'] = len(df)
    stats['markets_processed'] = sorted(df['market'].unique().tolist())
    stats['files_processed'] = sorted(df['source_file'].unique().tolist())

    # Determine which column holds hospital name
    name_col = 'hospital_name' if 'hospital_name' in df.columns else \
               'hospital' if 'hospital' in df.columns else \
               'name' if 'name' in df.columns else None
    if name_col:
        stats['unique_hospitals'] = int(df[name_col].nunique())
        stats['hospital_list'] = sorted(df[name_col].dropna().unique().tolist())

    # Role distribution
    role_col = 'role' if 'role' in df.columns else 'reviewer_type'
    rating_col = 'review_rating' if 'review_rating' in df.columns else 'rating'

    if role_col in df.columns:
        role_counts = df[role_col].value_counts()
        stats['role_distribution'] = {k: int(v) for k, v in role_counts.items()}

        # Average rating by role
        if rating_col in df.columns:
            stats['avg_rating_by_role'] = df.groupby(role_col)[rating_col].mean().round(3).to_dict()
            stats['overall_avg_rating'] = round(float(df[rating_col].mean()), 3)

    # Gender distribution
    gender_col = 'gender' if 'gender' in df.columns else 'Inferred_Gender'
    if gender_col in df.columns:
        stats['gender_distribution'] = {k: int(v) for k, v in df[gender_col].value_counts().items()}

        # Gender gap by role (flat keys for JSON)
        if role_col in df.columns and rating_col in df.columns:
            gender_role = df.groupby([gender_col, role_col])[rating_col].mean().round(3)
            flat = {}
            for (gender, role), val in gender_role.items():
                key = f"{str(gender).lower()}_{str(role).lower().replace('/', '_')}"
                flat[key] = round(float(val), 3)
            stats['avg_rating_by_gender_role'] = flat

    # Review length
    text_col = 'review_text' if 'review_text' in df.columns else None
    if text_col and rating_col in df.columns:
        df = df.copy()
        df['review_length'] = df[text_col].fillna('').apply(len)
        stats['avg_review_length'] = round(float(df['review_length'].mean()), 1)
        stats['review_length_rating_corr'] = round(float(df['review_length'].corr(df[rating_col])), 4)

        # Length buckets for chart
        bins = [0, 100, 300, 600, 1000, 99999]
        labels = ['0–100', '101–300', '301–600', '601–1000', '1000+']
        df['length_bucket'] = pd.cut(df['review_length'], bins=bins, labels=labels, right=True)
        bucket_stats = df.groupby('length_bucket', observed=True)[rating_col].agg(['mean', 'std']).reset_index()
        stats['review_length_buckets'] = [
            {'label': str(row['length_bucket']), 'mean': round(float(row['mean']), 3), 'sd': round(float(row['std']), 3)}
            for _, row in bucket_stats.iterrows()
            if not pd.isna(row['mean'])
        ]

    # Per-market breakdown
    stats['reviews_per_market'] = {k: int(v) for k, v in df['market'].value_counts().items()}

    return stats

# ── 3. Hospital type ratings ───────────────────────────────────────────────────

def get_hospital_type(name):
    name = str(name).lower()
    if 'kaiser' in name: return 'Kaiser'
    elif 'ucsf' in name or 'uc davis' in name or 'uc san' in name: return 'Public Academic'
    elif 'zuckerberg' in name or 'general' in name or 'county' in name: return 'Safety Net'
    elif 'va ' in name or 'veteran' in name: return 'VA'
    elif 'sutter' in name or 'cpmc' in name or 'dignity' in name or 'st.' in name: return 'Private Nonprofit'
    elif 'kindred' in name or 'select' in name: return 'Long Term Acute'
    elif 'psychiatric' in name or 'behavioral' in name or 'sierra vista' in name: return 'Psychiatric'
    else: return 'Other'

def compute_hospital_type_ratings(df):
    name_col = 'hospital_name' if 'hospital_name' in df.columns else \
               'hospital' if 'hospital' in df.columns else None
    rating_col = 'review_rating' if 'review_rating' in df.columns else 'rating'
    if not name_col or rating_col not in df.columns:
        return {}
    df = df.copy()
    df['hospital_type'] = df[name_col].apply(get_hospital_type)
    type_ratings = df.groupby('hospital_type')[rating_col].mean().round(3)
    return {k: round(float(v), 3) for k, v in type_ratings.items()}

# ── 4. Run OLS regression and extract key coefficients ────────────────────────

def run_regression(df):
    results = {}

    rating_col = 'review_rating' if 'review_rating' in df.columns else 'rating'
    role_col = 'role' if 'role' in df.columns else 'reviewer_type'
    gender_col = 'gender' if 'gender' in df.columns else 'Inferred_Gender'

    # Prepare binary variables
    reg_df = df.copy()
    reg_df['is_patient'] = (reg_df[role_col].str.lower() == 'patient').astype(int)
    reg_df['is_male'] = (reg_df[gender_col].str.lower() == 'male').astype(int)

    if 'review_text' in reg_df.columns:
        reg_df['review_length'] = reg_df['review_text'].fillna('').apply(len)

    name_col = 'hospital_name' if 'hospital_name' in reg_df.columns else \
               'hospital' if 'hospital' in reg_df.columns else None
    if name_col:
        reg_df['hospital_type'] = reg_df[name_col].apply(get_hospital_type)

    try:
        if 'review_length' in reg_df.columns and name_col:
            formula = f'{rating_col} ~ is_patient * is_male + review_length + C(hospital_type)'
        elif 'review_length' in reg_df.columns:
            formula = f'{rating_col} ~ is_patient * is_male + review_length'
        else:
            formula = f'{rating_col} ~ is_patient * is_male'

        model = smf.ols(formula, data=reg_df).fit()

        results['r_squared'] = round(float(model.rsquared), 4)
        results['n_obs'] = int(model.nobs)

        coef = model.params
        pval = model.pvalues

        for var in ['is_patient', 'is_male', 'is_patient:is_male', 'review_length']:
            if var in coef:
                results[f'coef_{var}'] = round(float(coef[var]), 6)
                results[f'pval_{var}'] = round(float(pval[var]), 6)

        if name_col:
            results['hospital_type_coefs'] = {k: round(float(v), 4) for k, v in coef.items() if 'hospital_type' in k}

        results['regression_status'] = 'success'

    except Exception as e:
        results['regression_status'] = f'failed: {e}'

    return results

# ── 5. Load previous findings for comparison ──────────────────────────────────

def load_previous_findings():
    if not os.path.exists('synthesis_history.jsonl'):
        return None
    with open('synthesis_history.jsonl', 'r') as f:
        lines = f.readlines()
    if not lines:
        return None
    return json.loads(lines[-1])

# ── 6. Call the synthesis agent ───────────────────────────────────────────────

def run_synthesis_agent(stats, regression, previous_run=None):
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    prev_summary = "None — this is the first synthesis run." if not previous_run else \
                   f"Previous run ({previous_run.get('timestamp', 'unknown')}): " \
                   f"{previous_run.get('total_reviews', '?')} reviews, " \
                   f"markets: {previous_run.get('markets_processed', '?')}"

    prompt = f"""You are a health services researcher analyzing hospital Google reviews across California.

=== CURRENT DATASET ===
{json.dumps(stats, indent=2)}

=== REGRESSION RESULTS (pooled OLS) ===
{json.dumps(regression, indent=2)}

=== PREVIOUS RUN ===
{prev_summary}

=== YOUR TASKS ===

1. FINDINGS UPDATE (2–3 paragraphs, research language):
   - Report the total sample size, markets covered, and hospital count
   - Report key regression findings: hospital type R², gender×role interaction (coefficient + p-value), review length effect
   - Note any patient vs caretaker rating gap
   - Be precise — use the exact numbers from the data above

2. CHANGES FROM PREVIOUS RUN (if applicable):
   - Did total n increase? By how much?
   - Did any key coefficient shift meaningfully (>0.05)?
   - Were new markets added?

3. MISSING DATA / NEXT STEPS:
   - Which markets are not yet processed?
   - What would change the results most if added?

4. ABSTRACT SENTENCE:
   - One sentence suitable for updating the abstract, with current n and key finding

Write clearly. Do not hedge with "may" or "might" when reporting regression results — state what the model shows."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

# ── 7. Save results + write data.json for dashboard ──────────────────────────

def save_results(stats, regression, narrative, hospital_type_ratings):
    timestamp = datetime.datetime.now().isoformat()

    # ── 7a. Narrative text report ─────────────────────────────────────────────
    with open('synthesis_report_latest.txt', 'w') as f:
        f.write(f"SYNTHESIS REPORT — {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        f.write(narrative)
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("RAW STATS:\n")
        f.write(json.dumps(stats, indent=2))
        f.write("\n\nREGRESSION:\n")
        f.write(json.dumps(regression, indent=2))

    # ── 7b. History log ───────────────────────────────────────────────────────
    record = {
        'timestamp': timestamp,
        'total_reviews': stats.get('total_reviews'),
        'markets_processed': stats.get('markets_processed'),
        'unique_hospitals': stats.get('unique_hospitals'),
        'r_squared': regression.get('r_squared'),
        'coef_is_patient': regression.get('coef_is_patient'),
        'coef_interaction': regression.get('coef_is_patient:is_male'),
        'narrative_preview': narrative[:300]
    }
    with open('synthesis_history.jsonl', 'a') as f:
        f.write(json.dumps(record) + '\n')

    # ── 7c. data.json for GitHub Pages dashboard ──────────────────────────────
    dashboard_data = {
        'last_updated': timestamp,
        'total_reviews': stats.get('total_reviews'),
        'unique_hospitals': stats.get('unique_hospitals'),
        'markets_processed': stats.get('markets_processed', []),
        'reviews_per_market': stats.get('reviews_per_market', {}),
        'overall_avg_rating': stats.get('overall_avg_rating'),
        'avg_rating_by_role': stats.get('avg_rating_by_role', {}),
        'avg_rating_by_gender_role': stats.get('avg_rating_by_gender_role', {}),
        'review_length_buckets': stats.get('review_length_buckets', []),
        'hospital_type_ratings': hospital_type_ratings,
        'r_squared': regression.get('r_squared'),
        'regression': {
            'coef_is_patient': regression.get('coef_is_patient'),
            'pval_is_patient': regression.get('pval_is_patient'),
            'coef_is_male': regression.get('coef_is_male'),
            'pval_is_male': regression.get('pval_is_male'),
            'coef_is_patient:is_male': regression.get('coef_is_patient:is_male'),
            'pval_is_patient:is_male': regression.get('pval_is_patient:is_male'),
            'coef_review_length': regression.get('coef_review_length'),
            'pval_review_length': regression.get('pval_review_length'),
        },
        'narrative': narrative,
    }
    with open('data.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"\nSaved to synthesis_report_latest.txt")
    print(f"Run logged to synthesis_history.jsonl")
    print(f"Dashboard data written to data.json  ← commit + push this to update the site")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Research Synthesis Agent ===\n")

    df = load_all_features()

    print("Computing statistics...")
    stats = compute_stats(df)

    print("Computing hospital type ratings...")
    hospital_type_ratings = compute_hospital_type_ratings(df)

    print("Running regression...")
    regression = run_regression(df)
    print(f"  R² = {regression.get('r_squared', 'N/A')}, n = {regression.get('n_obs', 'N/A')}")

    previous = load_previous_findings()
    if previous:
        print(f"Comparing against previous run: {previous.get('timestamp', '?')}")

    print("Calling synthesis agent...")
    narrative = run_synthesis_agent(stats, regression, previous)

    save_results(stats, regression, narrative, hospital_type_ratings)

    print("\n" + "=" * 60)
    print(narrative)
    print("\n" + "=" * 60)
    print("\nTo update the dashboard:")
    print("  git add data.json && git commit -m 'Update dashboard data' && git push")
