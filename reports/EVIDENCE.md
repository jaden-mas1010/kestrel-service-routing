# Routing validation evidence

- **Data:** 10,822 historical labelled requests; 2,178 new requests.
- **Holdout:** chronologically latest 2,165 rows (20%). Earlier 8,657 used to train the evaluated model.
- **Holdout first-assignment accuracy:** 93.67% (2,028 correct / 137 wrong).
- **Macro-F1:** 0.938; an additional check against weak minority-team performance.
- **Low-confidence review flag (<75% model probability):** 11.2% of holdout. These scores are not probability-calibrated.
- **Model:** Word unigram/bigram TF-IDF + Logistic Regression, retrained on all historical labels for prediction.
- **Caution:** Metrics measure historical *initial team labels*, NOT correctness of final resolution. Repeated messages and operational drift can bias the estimate. Route by policy and track actual transfers in a shadow pilot.

## Recall by team
- Billing: 336 cases, recall 97.6%
- Filters & Consumables: 289 cases, recall 86.9%
- Installs & Demo: 235 cases, recall 97.4%
- Product Advice: 225 cases, recall 95.6%
- Repairs: 624 cases, recall 92.3%
- Returns & Replacement: 254 cases, recall 93.7%
- Warranty Claims: 202 cases, recall 94.6%

## Known failure modes
Ambiguous “please call back” messages, payment mentioned as background rather than a Billing issue, purifier breakdown versus replacement filter, damaged delivery versus technical fault, and renamed teams. Review these cases manually rather than trusting a high model score.
