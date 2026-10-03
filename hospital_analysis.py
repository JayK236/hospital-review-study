import pandas as pd
import anthropic
import json
import time
import os

df = pd.read_excel('Outscraper-20260425025255s7e.xlsx')
df_reviews = df[df['review_text'].notna()].copy()
df_reviews = df_reviews[['author_title', 'review_text', 'review_rating']].reset_index(drop=True)
print('Loaded ' + str(len(df_reviews)) + ' reviews')

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def extract_features(author_name, review_text):
    prompt = "Analyze this hospital review and return ONLY valid JSON.\nReviewer name: " + str(author_name) + "\nReview: " + str(review_text) + "\nReturn JSON with fields: role, gender, race, disease, role_evidence. For role use: patient, caretaker, patient/caretaker, patient/caretaker/unclear, or unclear. For gender: male, female, or unclear. For race: infer from name or unclear. For disease: condition mentioned or none_mentioned."
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = message.content[0].text
    raw = raw.replace('```json', '').replace('```', '').strip()
    print("RAW: " + raw)
    return json.loads(raw)

results = []
for i, row in df_reviews.iterrows():
    try:
        features = extract_features(row['author_title'], row['review_text'])
        features['author_title'] = row['author_title']
        features['review_rating'] = row['review_rating']
        results.append(features)
    except Exception as e:
        print("Error on row " + str(i) + ": " + str(e))
        results.append({'author_title': row['author_title'], 'review_rating': row['review_rating'], 'role': 'error', 'gender': 'error', 'race': 'error', 'disease': 'error', 'role_evidence': str(e)})
    if (i + 1) % 10 == 0:
        print("Processed " + str(i + 1) + " of " + str(len(df_reviews)))
    time.sleep(0.3)

base_name = "batch2_features"
version = 1

while os.path.exists(f"{base_name}_v{version}.csv"):
    version += 1

filename = f"{base_name}_v{version}.csv"

df_results = pd.DataFrame(results)
df_results.to_csv('batch2_features.csv', index=False)

print(f"Done! Saved to {filename}")
print(df_results['role'].value_counts())
