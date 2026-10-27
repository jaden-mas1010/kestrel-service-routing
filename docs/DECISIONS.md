# Project decisions — complete as you inspect the real pack

## 1. What is the client's requested label?
- Requested: match original routed team on historical records at least 90%, using one of seven team labels.
- Open question: does the resolution log reveal a mismatch between original routed and best resolving team? Count the mismatch and review policy before proposing any changed target.
- Decision pending: [fill after inspecting original files].

## 2. What is allowed as a prediction-time feature?
- Use only data present when a new customer request arrives.
- Exclude resolution log, resolver notes, routed team, approved refunds, closure status, downstream fields, and personally identifying columns.
- Decision pending: [confirm exact fields with `data_dictionary`].

## 3. Policy exceptions and confidence handling
- Default: no rules are enabled unless the policy is confirmed and manually tested.
- Model always emits an allowed team label for offline scoring; dashboard flags uncertain predictions for review in a real pilot.
- Decision pending: [enter each real policy override and its test].

## 4. Honest discarded approaches
- Baseline and alternative feature models are compared on a development split; keep only the architecture that scores best on dev.
- After measuring, record which candidates were actually rejected and why. **Do not claim experiments you did not run.**

## 5. Deployment caveats
- This is an operational prototype, not a public multi-user application. Loopback only; no log retention, production authentication or queue integration.
- A one-week shadow trial is required before retiring any current paid bot.
