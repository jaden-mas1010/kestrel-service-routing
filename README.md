# RouteDesk | Kestrel Home Service-Request Router

**A working, local-first service-request routing application.** It takes a single Kestrel service request as JSON, recommends one of the seven legitimate team labels, presents human-readable evidence, and explicitly flags low-confidence requests. It does not need an LLM, a paid API key, a GPU or an external database.

## Important: assessment data not bundled in this starter

The actual eight assessment files have **not yet been supplied to this repository**. The resulting official accuracy, predicted labels and business decision **cannot be stated until those files have been checked and the router trained on them**. The optional synthetic demo is for smoke-testing software only and **must never be submitted** as `predictions.csv`.

## Run on a clean machine (Python 3.10+)

```bash
git clone <YOUR-REPO-URL>
cd kestrel-route
python -m venv .venv
# Mac/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn kestrel.app:app --host 127.0.0.1 --port 8000
```

Go to **http://127.0.0.1:8000**. The console and `/api/health` start **without any API key**, even if the model has not yet been trained. If no model is found, the UI clearly says "Setup required" instead of fabricating a routing decision. The project owner should deliver a **trained** model file for full offline reviewer functionality after checking that it does not expose sensitive customer data.

### Verify the app without the actual pack (fake test only)

```bash
python examples/create_synthetic_demo.py /tmp/kestrel_fake_data
python -m kestrel.inspect --data /tmp/kestrel_fake_data
python -m kestrel.train --data /tmp/kestrel_fake_data --out .
uvicorn kestrel.app:app --host 127.0.0.1 --port 8000
```

**DEMO ONLY**: These fictional records and fictional teams verify that the engineering pipeline, CSV export and screen work. Delete `predictions.csv`, `models/router.joblib` and generated report files after that test; they are **not** the assessment submission. On Windows use a folder such as `demo_data` instead of `/tmp/kestrel_fake_data`. Never confuse demo metrics with the real evaluation.

## Train on the actual Kestrel pack

1. Copy the provided original `.csv` source files to `data/` *locally*. Leave customer data out of the public GitHub repo unless publication is explicitly permitted. Keep the original email thread, data dictionary, team descriptions and ops-policy document available for decision-making.
2. Run `python -m kestrel.inspect --data data`. Read the printed filenames, exact team labels, missingness, class distribution and proposed text/category fields.
3. Read `ops-policy.pdf`, team definitions, `email-thread.txt` and the data dictionary **before** finalising rules or features. In `config.json` explicitly set `train_file`, `test_file`, `id_column`, `target_column`, `date_column`, `text_columns`, `category_columns` if inference is wrong or ambiguous. Never use post-resolution columns as model input.
4. Add only policy rules that are *actually supported by* the operations policy, in `policy_rules.json`; set `verified=true` and `enabled=true` only after testing the override on held-out examples. As shipped, no operational rule is asserted.
5. Train and export:

   ```bash
   python -m kestrel.train --data data --out .
   python -m kestrel.verify --data data --submission predictions.csv
   python -m pytest -q
   uvicorn kestrel.app:app --host 127.0.0.1 --port 8000
   ```

6. Verify `predictions.csv` has the **exact headers and row order** from `sample_submission.csv`, with one of the seven original team names per ID, no missing labels and no duplicated IDs. **Commit `predictions.csv` to the repository root** as the brief requires.
7. Validate `reports/EVIDENCE.md` and `RITU_MEMO.md` with the real figures. Review audit mistakes individually in `reports/audit_cases.csv` and look for costly operational exceptions. Update the memo with findings from the actual policy and email thread.
8. Save and submit your three-minute screen recording (no slides) and complete **every question** in the official submission form. Use honest metrics, precise hours spent and real repo/Drive URLs.

### Project outputs

| File | Description |
| --- | --- |
| `predictions.csv` | Final hidden-label prediction file; exact sample schema. **Commit this.** |
| `models/router.joblib` | Trained local router; share only if security review permits. |
| `reports/metrics.json` | Full audit metrics, model comparison and 95% Wilson accuracy interval. |
| `reports/EVIDENCE.md` | Held-out sample, mistakes, per-team recall, conditions under which it fails. |
| `reports/audit_cases.csv` | Individual held-out decisions for manual error review; **keep private**. |
| `RITU_MEMO.md` | Automatically drafted business memo with measured accuracy and licence-spend ceiling; read/edit before submission. |
| `policy_rules.json` | Explicit and disabled-by-default operations-rule configuration. |
| `config.json` | Verifiable mapping of dataset columns and trust boundary. |

## JSON API

**Start**: `uvicorn kestrel.app:app --host 127.0.0.1 --port 8000`

- `GET /api/health` — service readiness and API key status.
- `GET /api/status` — actual routing-team options, expected inputs and measured holdout metrics.
- `POST /api/route` — single ticket, one team, score, readable feature evidence, top-three alternatives and `REVIEW` / `AUTO_ROUTE` / `POLICY` status.
- `GET /api/examples` — limited local unlabeled requests for loading into the UI.
- `GET /docs` — interactive OpenAPI documentation.

Send the **actual CSV field names**. Example on a dataset using `issue_text`, `product_line` and `channel` (adjust based on your data dictionary):

```bash
curl -X POST http://127.0.0.1:8000/api/route \
  -H 'Content-Type: application/json' \
  -d '{"record":{"issue_text":"Purifier leaking near filter housing","product_line":"water purifier","channel":"online"}}'
```

The output is not an LLM-generated claim about what the customer should do. It exposes the router's predicted team and the features contributing to its statistical decision. `confidence` is a **model-estimated, not independently calibrated probability**; it should not be interpreted as a warranty or service-level guarantee.

## Modelling and evaluation

- **No feature leakage:** only fields present when a customer creates a new request and in both labelled/unlabelled data are considered. Resolver notes, routing decisions, final handling outcomes, contact identifiers, and after-the-fact status fields are explicitly prohibited.
- **Three-way validation:** prefer chronological build / development / untouched audit (70/15/15). If a date is missing or new labels would make a historical split impossible, the fallback split is disclosed in `metrics.json` and the evidence report.
- **Compare candidates:** word TF-IDF logistic regression; word+categorical logistic regression; and a robust word+character n-gram+categorical variant. Character features can help Hinglish, spelling mistakes and product model variants. Model selected on development accuracy, with macro-F1 tie-break, **before** audit.
- **Audit:** exact first-team match **accuracy** (the task's primary scoring quantity), per-team recall, macro-F1, confusion matrix, error-rate count, and model-only confidence-based review coverage. Low-confidence cases still receive a label for challenge scoring; in live operations they should be reviewed.
- **Train final:** after auditing the selected design, fit the selected architecture on all labelled data and route all unmatched new requests in sample-submission order.
- **Business framing:** replacing a ₹3.2 lakh annual routing contract has at most ₹3.2 lakh gross licence savings; net savings are unknown until people, hosting, quality and maintenance are costed. Do not recommend immediate cancellation on historical accuracy alone.

**Limitations:** Training labels may reproduce historic misroutes. A team that resolved a request may differ from its first routing team. Model confidence is not calibrated for all inputs. New service policies, unusual warranty exceptions, missing description text, shifting product mixes and multilingual requests require human review. The auto-generated memo cannot substitute for verifying operational policy.

## Security / handover

- Python training and inference operate locally. No external chat-model endpoints and no paid API keys.
- The console binds to loopback (`127.0.0.1`) in the documented command; do not expose it publicly without authentication, rate limits, request logging/redaction and a security review.
- Customer data, resolution cases and the trained vocabulary may include personal or business-sensitive text. Audit before making any artefact public. **Do not commit real raw tickets by default**.
- Loading `joblib` files is unsafe if the file is untrusted; only load the training artefact you generated yourself.
- To update rules, edit `policy_rules.json`, re-run training and restart the app. Policy overrides can change outcomes and require their own held-out evaluation before live rollout.
- Run the regression checks with `pip install -r requirements-dev.txt && python -m pytest -q`.

## Submission decisions

Write all deviations from the client's request and your reasons in `docs/DECISIONS.md`. Use `docs/SCREEN_RECORDING_GUIDE.md` to plan a candid under-three-minute walk-through. Both should describe real work, not invented attempts or results.
