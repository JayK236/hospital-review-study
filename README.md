# Hospital Patient Experience Study

### Modeling Demographic Predictors of Online Hospital Ratings Across California Markets

Independent research project analyzing 63,500+ Google Reviews across 30+ hospital facilities in San Francisco, Sacramento, Oakland, and San Jose, California.

---

## Research Question

Do reviewer characteristics — role (patient vs. caretaker), gender, and review length — systematically predict hospital star ratings independent of care quality? And do these effects replicate across geographically distinct markets?

---

## Key Findings

- **Hospital type** is the strongest predictor of ratings (pooled R² = 0.23–0.31). Kaiser Permanente facilities score 1.7–2.3 stars above public safety-net hospitals after controlling for reviewer characteristics.
- **Gender × role interaction** replicates independently across SF and Sacramento (SF: p=0.009; Sacramento: p<0.001). Male caretakers are the most negative reviewer group; the male deficit largely disappears among patients.
- **Review length** negatively predicts ratings across all models and facilities (coef ≈ −0.001/character, p<0.001), consistent with longer reviews reflecting higher emotional arousal.
- **Reviewer role** independently predicts ratings: patients rate facilities 0.25–1.96 stars higher than caretakers depending on facility type.

---

## Dataset

| Market | Facilities | Reviews | Status |
|--------|-----------|---------|--------|
| San Francisco | 15 | ~18,000 | ✅ Processed |
| Sacramento | 15 | ~18,400 | ✅ Processed |
| Oakland / East Bay | TBD | ~14,800 | ⏳ Pending |
| San Jose / South Bay | TBD | ~8,500 | ⏳ Pending |
| **Total** | **30+** | **63,500+** | |

Reviews collected via [Outscraper](https://outscraper.com/) from Google Maps. Raw data files excluded from this repo (privacy + size).

---

## Pipeline

```
batch_*.xlsx  (raw Outscraper output)
      │
      ▼
hospital_analysis.py   — LLM feature extraction via Claude API
      │                   extracts: reviewer role, gender, race, disease
      ▼
*_features*.csv
      │
      ├── analysis.py          — descriptive stats
      ├── interaction_analysis.py  — gender gap by role, per hospital
      └── linear_model.py      — OLS regression (per-facility + pooled)

geocode_hospitals.py   — geocodes ~50 unique hospitals via Nominatim (free)
merge_addresses.py     — joins address/zip/lat/lon onto all features CSVs
```

---

## Setup

```bash
pip install anthropic pandas statsmodels geopy openpyxl
```

Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Or create a `.env` file (never commit this):
```
ANTHROPIC_API_KEY=your-key-here
```

---

## Repo Structure

```
hospital-review-study/
├── hospital_analysis.py      # LLM extraction pipeline
├── analysis.py               # Descriptive statistics
├── interaction_analysis.py   # Gender × role interaction analysis
├── linear_model.py           # OLS regression models
├── geocode_hospitals.py      # Geocode hospitals via Nominatim
├── merge_addresses.py        # Merge address lookup onto features CSVs
├── .gitignore
└── README.md
```

---

## Methods Summary

Reviewer role (patient, caretaker, unclear), gender (inferred from name and review text), mentioned medical condition, and review length were extracted from unstructured review text using a large language model (Claude, Anthropic). Ordinary least squares regression models were fit per facility and pooled across facilities, with hospital type as a covariate. An interaction term (gender × role) tested whether gender effects differed between patients and caretakers.

---

## Pending / Next Steps

- [ ] Process Oakland and San Jose batches
- [ ] Run `geocode_hospitals.py` → `merge_addresses.py` for all markets
- [ ] Join US Census ACS tract-level demographics (race, income) via FIPS tract
- [ ] Submit abstract to AcademyHealth Annual Research Meeting

---

## Author

Jayanth Karuturi — jayanthkaruturi@gmail.com

*Independent research, not affiliated with any hospital system or payer.*
