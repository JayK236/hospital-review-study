import pandas as pd

df = pd.read_csv('batch2_features.csv')

print("=== AVERAGE RATING BY ROLE ===")
print(df.groupby('role')['review_rating'].mean().round(2).sort_values())

print("\n=== COUNT BY ROLE ===")
print(df['role'].value_counts())

print("\n=== TOP DISEASES MENTIONED ===")
diseases = df[df['disease'] != 'none_mentioned']['disease']
print(diseases.value_counts().head(15))

print("\n=== PATIENT REVIEWS - AVERAGE RATING ===")
patients = df[df['role'] == 'patient']
caretakers = df[df['role'] == 'caretaker']
print("Patients avg rating: " + str(round(patients['review_rating'].mean(), 2)))
print("Caretakers avg rating: " + str(round(caretakers['review_rating'].mean(), 2)))

print("\n=== RATING DISTRIBUTION - PATIENTS ===")
print(patients['review_rating'].value_counts().sort_index())

print("\n=== RATING DISTRIBUTION - CARETAKERS ===")
print(caretakers['review_rating'].value_counts().sort_index())
print("\n=== GENDER DISTRIBUTION ===")
print(df['gender'].value_counts())

print("\n=== AVERAGE RATING BY GENDER ===")
print(df.groupby('gender')['review_rating'].mean().round(2).sort_values())

print("\n=== RACE DISTRIBUTION ===")
print(df['race'].value_counts())

print("\n=== AVERAGE RATING BY RACE ===")
print(df.groupby('race')['review_rating'].mean().round(2).sort_values())
