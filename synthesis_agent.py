"""
Research Synthesis Agent
Reads all processed features CSVs + regression outputs, then uses Claude to
write an updated findings section and flag any changes from the previous run.

after each new batch is processed:
    python synthesis_agent.py
"""

import anthropic
import pandas as pd
import numpy as np
import glob
import os
import json
from datetime import datetime
import statsmodels.formula.api as smf

# ── 1. Load all features CSVs ─────────────────────────────────────────────────

def load_all_features():
    files = glob.glob('*_features*.csv')
    if not files:
        raise FileNotFoundError("No *_features*.csv files found. Run the extraction pipeline first.")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df['source_file'] = f
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} rows from {len(files)} files: {files}")
    return combined

# ── 2. Compute summary statistics ─────────────────────────────────────────────

def compute_summary_stats(df):
    stats = {}

    stats['total_reviews'] = len(df)
    stats['source_files'] = df['source_file'].unique().tolist()

    # Determine name column
    name_col = 'hospital_name' if 'hospital_name' in df.columns else ('hospital' if 'hospital' in df.columns else None)

    if name_col:
        stats['unique_hospitals'] = int(df[name_col].nunique())
        stats['hospitals'] = df[name_col].value_counts().head(10).to_dict()

    # Role distribution
    if 'role' in df.columns:
        role_counts = df['role'].value_counts().to_dict()
        stats['role_distribution'] = role_counts
        for role in ['patient', 'caretaker']:
            mask = df['role'].str.lower() == role
            if mask.sum() > 0:
                stats[f'avg_rating_{role}'] = round(df.loc[mask, 'review_rating'].mean(), 3)

    # Gender distribution
    gender_col = None
    for col in ['gender', 'Inferred_Gender']:
        if col in df.columns:
            gender_col = col
            break
    if gender_col:
        stats['gender_distribution'] = df[gender_col].value_counts().to_dict()

    # Review length
    if 'review_text' in df.columns:
        df['review_length'] = df['review_text'].str.len()
        stats['avg_review_length'] = round(df['review_length'].mean(), 1)

    # Overall avg rating
    if 'review_rating' in df.columns:
        stats['overall_avg_rating'] = round(df['review_rating'].mean(), 3)
        stats['rating_distribution'] = df['review_rating'].value_counts().sort_index().to_dict()

    return stats

# ── 3. Run regression and extract key coefficients ───────────────────────────

def run_regression(df):
    results = {}

    # Normalize column names
    gender_col = 'gender' if 'gender' in df.columns else ('Inferred_Gender' if 'Inferred_Gender' in df.columns else None)
    role_col = 'role' if 'role' in df.columns else ('reviewer_type' if 'reviewer_type' in df.columns else None)
    name_col = 'hospital_name' if 'hospital_name' in df.columns else ('hospital' if 'hospital' in df.columns else None)

    if not all([gender_col, role_col, 'review_rating' in df.columns]):
        results['error'] = "Missing columns for regression"
        return results

    reg_df = df.copy()
    if 'review_text' in reg_df.columns:
        reg_df['review_length'] = reg_df['review_text'].str.len().fillna(0)
    else:
        reg_df['review_length'] = 0

    # Standardize gender and role
    reg_df['gender_binary'] = reg_df[gender_col].str.lower().map({'male': 1, 'female': 0})
    reg_df['is_patient'] = reg_df[role_col].str.lower().str.contains('patient', na=False).astype(int)
    reg_df['is_caretaker'] = reg_df[role_col].str.lower().str.contains('caretaker', na=False).astype(int)

    reg_df = reg_df.dropna(subset=['review_rating', 'gender_binary'])

    if len(reg_df) < 50:
        results['error'] = f"Too few rows for regression after filtering: {len(reg_df)}"
        return results

    try:
        # Pooled model with interaction
        model = smf.ols(
            'review_rating ~ gender_binary + is_patient + gender_binary:is_patient + review_length',
            data=reg_df
        ).fit()

        results['n'] = int(len(reg_df))
        results['r_squared'] = round(model.rsquared, 4)
        results['gender_coef'] = round(model.params.get('gender_binary', np.nan), 4)
        results['gender_pval'] = round(model.pvalues.get('gender_binary', np.nan), 4)
        results['patient_coef'] = round(model.params.get('is_patient', np.nan), 4)
        results['patient_pval'] = round(model.pvalues.get('is_patient', np.nan), 4)
        results['interaction_coef'] = round(model.params.get('gender_binary:is_patient', np.nan), 4)
        results['interaction_pval'] = round(model.pvalues.get('gender_binary:is_patient', np.nan), 4)
        results['review_length_coef'] = round(model.params.get('review_length', np.nan), 6)
        results['review_length_pval'] = round(model.pvalues.get('review_length', np.nan), 4)

        # Gender gap by role
        male_patients = reg_df[(reg_df['gender_binary'] == 1) & (reg_df['is_patient'] == 1)]['review_rating']
        female_patients = reg_df[(reg_df['gender_binary'] == 0) & (reg_df['is_patient'] == 1)]['review_rating']
        male_caretakers = reg_df[(reg_df['gender_binary'] == 1) & (reg_df['is_caretaker'] == 1)]['review_rating']
        female_caretakers = reg_df[(reg_df['gender_binary'] == 0) & (reg_df['is_caretaker'] == 1)]['review_rating']

        if len(male_patients) > 0 and len(female_patients) > 0:
            results['gender_gap_patients'] = round(male_patients.mean() - female_patients.mean(), 3)
        if len(male_caretakers) > 0 and len(female_caretakers) > 0:
            results['gender_gap_caretakers'] = round(male_caretakers.mean() - female_caretakers.mean(), 3)

        # Hospital type if available
        if name_col:
            reg_df['is_kaiser'] = reg_df[name_col].str.lower().str.contains('kaiser', na=False).astype(int)
            reg_df['is_ucsf'] = reg_df[name_col].str.lower().str.contains('ucsf', na=False).astype(int)
            reg_df['is_zuckerberg'] = reg_df[name_col].str.lower().str.contains('zuckerberg|general', na=False).astype(int)

            hospital_model = smf.ols(
                'review_rating ~ gender_binary + is_patient + gender_binary:is_patient + review_length + is_kaiser + is_ucsf + is_zuckerberg',
                data=reg_df
            ).fit()

            results['kaiser_coef'] = round(hospital_model.params.get('is_kaiser', np.nan), 4)
            results['kaiser_pval'] = round(hospital_model.pvalues.get('is_kaiser', np.nan), 4)
            results['r_squared_with_hospital'] = round(hospital_model.rsquared, 4)

    except Exception as e:
        results['regression_error'] = str(e)

    return results

# ── 4. Load previous synthesis for comparison ─────────────────────────────────

def load_previous_synthesis():
    files = sorted(glob.glob('synthesis_report_*.json'))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)

# ── 5. Run the synthesis agent ────────────────────────────────────────────────

def run_synthesis_agent(stats, regression, previous=None):
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    previous_str = json.dumps(previous, indent=2) if previous else "None — this is the first run."

    prompt = f"""You are a health services research assistant. You have just processed a new batch of hospital Google reviews and need to update the research findings.

=== CURRENT DATASET STATISTICS ===
{json.dumps(stats, indent=2)}

=== REGRESSION RESULTS ===
{json.dumps(regression, indent=2)}

=== PREVIOUS SYNTHESIS (for comparison) ===
{previous_str}

=== Tasks To Do ===

1. FINDINGS UPDATE (2-3 paragraphs, research language):
   Write an updated findings section covering:
   - Total dataset size and markets covered
   - Gender × role interaction (is it significant? what direction?)
   - Review length as a predictor
   - Hospital type effects if available
   - Any change from previous run worth flagging

2. CHANGES FROM PREVIOUS RUN:
   List any result that shifted meaningfully (e.g. p-value crossed 0.05, coefficient changed by >0.1, n increased substantially). If first run, say so.

3. ABSTRACT SENTENCE:
   One sentence suitable for dropping into a conference abstract. Be precise with numbers.

4. MISSING DATA:
   What markets or batches are still needed to complete the analysis?

Use exact numbers from the statistics. Flag if any result looks anomalous.
"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

# ── 6. Save outputs ───────────────────────────────────────────────────────────

def save_outputs(stats, regression, synthesis_text):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save structured data as JSON (for next run's comparison)
    snapshot = {
        'timestamp': timestamp,
        'stats': stats,
        'regression': regression,
    }
    json_path = f'synthesis_report_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(snapshot, f, indent=2)

    # Save human-readable text
    txt_path = f'synthesis_report_{timestamp}.txt'
    with open(txt_path, 'w') as f:
        f.write(f"RESEARCH SYNTHESIS REPORT\nGenerated: {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        f.write(synthesis_text)
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("\nREGRESSION COEFFICIENTS:\n")
        for k, v in regression.items():
            f.write(f"  {k}: {v}\n")

    # Always overwrite the "latest" file for easy access
    with open('synthesis_report_latest.txt', 'w') as f:
        f.write(f"RESEARCH SYNTHESIS REPORT\nGenerated: {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        f.write(synthesis_text)

    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved: synthesis_report_latest.txt")
    return txt_path

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Research Synthesis Agent ===\n")

    df = load_all_features()
    print(f"\nComputing statistics...")
    stats = compute_summary_stats(df)

    print("Running regression...")
    regression = run_regression(df)

    print("Loading previous synthesis for comparison...")
    previous = load_previous_synthesis()

    print("Running synthesis agent...\n")
    synthesis = run_synthesis_agent(stats, regression, previous)

    print("\n" + "=" * 60)
    print(synthesis)
    print("=" * 60)

    save_outputs(stats, regression, synthesis)
    print("\nDone.")
